#!/usr/bin/env python3
"""One command submits policy and reward resources as a Slurm heterogeneous job."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import uuid


def topology(settings):
    recipes = {'train.yml': (1, 8), 'train_4machine.yml': (4, 32)}
    if settings['recipe'] not in recipes:
        raise ValueError('recipe must be train.yml or train_4machine.yml')
    return recipes[settings['recipe']]


def validate(settings):
    topology(settings)
    files = [settings['metadata'], settings['omnigen_transformer_bin'], settings['conda_sh'],
             str(Path(settings['text_encoder']) / 'config.json'),
             str(Path(settings['vae_model']) / 'vae/config.json'),
             str(Path(settings['reward_pretrained']) / 'config.json'),
             str(Path(settings['reward_project']) / 'src/models/custom_qwen3_vl_reranker.py'),
             str(Path(settings['train_env']) / 'bin/python'),
             str(Path(settings['reward_env']) / 'bin/python')]
    for value in files:
        path = Path(value)
        if not path.is_file() or not os.access(path, os.R_OK):
            raise FileNotFoundError(f'Missing/unreadable; edit settings.json: {path}')
    for key in ('data_root', 'reward_checkpoint'):
        if not Path(settings[key]).exists():
            raise FileNotFoundError(f'Missing {key}: {settings[key]}')
    for role in ('policy', 'reward'):
        resource = settings['slurm'][role]
        count = int(resource['gres'].rsplit(':', 1)[-1])
        if count != (8 if role == 'policy' else 1):
            raise ValueError(f'{role} must request {8 if role == "policy" else 1} GPU(s) per node')
        if int(resource['cpus_per_task']) < 1:
            raise ValueError('cpus_per_task must be positive')
    if int(settings['reward_ready_timeout']) < 1:
        raise ValueError('reward_ready_timeout must be positive')


def directive(key, value):
    # SBATCH values are options, not shell code. Reject whitespace/newline injection.
    value = str(value)
    if key in ('output', 'error'):
        if '\n' in value or '\r' in value:
            raise ValueError('Slurm log paths must not contain newlines')
        return f'#SBATCH --{key}=' + shlex.quote(value)
    if not re.fullmatch(r'[A-Za-z0-9_.:/,+@=-]+', value):
        raise ValueError(f'Invalid Slurm {key}: {value!r}')
    return f'#SBATCH --{key}={value}'


def render_job(submission, settings, profile):
    nodes, _ = topology(settings)
    lines = ['#!/bin/bash', '#SBATCH --job-name=omnigen2_editscore_qwen3']
    for index, (role, count) in enumerate((('policy', nodes), ('reward', 1))):
        if index:
            lines.append('#SBATCH hetjob')
        resource = settings['slurm'][role]
        options = {'account': settings['slurm']['account'], 'partition': resource['partition'],
                   'nodes': count, 'ntasks-per-node': 1, 'cpus-per-task': resource['cpus_per_task'],
                   'mem': resource['mem'], 'gres': resource['gres'], 'time': settings['slurm']['time']}
        if resource.get('qos'):
            options['qos'] = resource['qos']
        if index == 0:
            options.update(output=str(submission / 'logs/job_%j.out'), error=str(submission / 'logs/job_%j.err'))
        lines.extend(directive(key, value) for key, value in options.items())
    lines += ['', 'set -euo pipefail', 'export PYTHONUNBUFFERED=1',
              'exec python3 ' + shlex.quote(str(profile / 'auto_run.py')) + ' ' + shlex.quote(str(submission)), '']
    return '\n'.join(lines)


def freeze_submission(profile, settings, destination):
    """Freeze settings and scripts at submit time, preserving the helper directory layout."""
    destination.mkdir(parents=True, exist_ok=False)
    (destination / 'logs').mkdir()
    frozen_bundle = destination / 'snapshot/h200_qwen3_rl'
    frozen_profile = frozen_bundle / 'editscore_config'
    frozen_profile.mkdir(parents=True)
    for name in ('prepare_profile.py', 'auto_run.py', 'launch_policy.sh', 'start_reward.sh',
                 'train.yml', 'train_4machine.yml'):
        shutil.copy2(profile / name, frozen_profile / name)
    shutil.copytree(profile / 'upstream_reference', frozen_profile / 'upstream_reference')
    for name in ('prepare_run.py', 'reward_server.py', 'reward_runtime.py'):
        shutil.copy2(profile.parent / name, frozen_bundle / name)
    (frozen_profile / 'settings.json').write_text(json.dumps(settings, indent=2) + '\n', encoding='utf-8')
    # Snapshot upstream trainer too. Model modules remain in the original repository.
    shutil.copy2(profile.parent.parent / 'train.py', destination / 'upstream_train.py')
    (destination / 'repo_rl_path.txt').write_text(str(profile.parent.parent.resolve()))
    job = destination / 'run_train.slurm'
    job.write_text(render_job(destination, settings, frozen_profile), encoding='utf-8', newline='\n')
    return job


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only', action='store_true', help='Write the frozen job without submitting')
    args = parser.parse_args()
    profile = Path(__file__).resolve().parent
    settings = json.loads((profile / 'settings.json').read_text(encoding='utf-8'))
    validate(settings)
    if not args.prepare_only and not shutil.which('sbatch'):
        raise RuntimeError('sbatch not found; run this command on the Slurm login node')
    destination = profile / 'outputs' / ('submission_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:8])
    job = freeze_submission(profile, settings, destination)
    print(f'Frozen submission: {destination}', flush=True)
    print(f'Policy topology: {topology(settings)} (nodes, GPUs); reward: 1 additional GPU', flush=True)
    if args.prepare_only:
        print(f'Prepared; submit with: sbatch {job}', flush=True)
        return
    result = subprocess.run(['sbatch', '--parsable', '--chdir', str(destination), str(job)],
                            text=True, capture_output=True, check=True)
    (destination / 'job_id.txt').write_text(result.stdout.strip() + '\n')
    print(f'Submitted job: {result.stdout.strip()}; logs: {destination / "logs"}', flush=True)


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        if isinstance(exc, subprocess.CalledProcessError) and exc.stderr:
            print(exc.stderr, file=sys.stderr)
        sys.exit(1)
