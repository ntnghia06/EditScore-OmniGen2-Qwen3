#!/usr/bin/env python3
"""Run inside a two-component Slurm allocation; manage reward and policy lifetimes."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request


def hosts(group):
    value = os.environ.get(f'SLURM_JOB_NODELIST_HET_GROUP_{group}')
    if not value:
        raise RuntimeError('Expected a Slurm heterogeneous allocation; submit with submit_train.py')
    return subprocess.check_output(['scontrol', 'show', 'hostnames', value], text=True).split()


def step(group, nodes, cpus, gres):
    return ['srun', f'--het-group={group}', '--exact', '--kill-on-bad-exit=1',
            f'--nodes={nodes}', f'--ntasks={nodes}', '--ntasks-per-node=1',
            f'--cpus-per-task={cpus}', f'--gres={gres}']


def stop(process):
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def wait_ready(url, run_id, process, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f'Reward process exited with code {process.returncode}; inspect logs/reward.log')
        try:
            with urllib.request.urlopen(url + '/ping', timeout=3) as response:
                result = json.load(response)
            if result.get('ready') and result.get('run_id') == run_id:
                return
        except (OSError, ValueError):
            pass
        print('Waiting for this job\'s Qwen3 reward server...', flush=True)
        time.sleep(5)
    raise TimeoutError('Reward did not become ready within reward_ready_timeout; inspect logs/reward.log')


def main():
    submission = Path(sys.argv[1]).resolve()
    profile = Path(__file__).resolve().parent
    settings = json.loads((profile / 'settings.json').read_text())
    policy_hosts, reward_hosts = hosts(0), hosts(1)
    expected_nodes = 1 if settings['recipe'] == 'train.yml' else 4
    if len(policy_hosts) != expected_nodes or len(reward_hosts) != 1:
        raise RuntimeError('Allocated node counts do not match selected recipe')
    run = submission / 'run'
    train_python = str(Path(settings['train_env']) / 'bin/python')
    reward_python = str(Path(settings['reward_env']) / 'bin/python')
    p_resource, r_resource = settings['slurm']['policy'], settings['slurm']['reward']
    reward_step = step(1, 1, r_resource['cpus_per_task'], r_resource['gres'])
    # Discover a free port on the actual reward host, never on the login node.
    port_code = 'import socket; s=socket.socket(); s.bind(("0.0.0.0",0)); print(s.getsockname()[1]); s.close()'
    port = int(subprocess.check_output(reward_step + [reward_python, '-c', port_code], text=True).strip())
    settings.update(reward_host=reward_hosts[0], reward_bind_host='0.0.0.0', port=port)
    (profile / 'settings.json').write_text(json.dumps(settings, indent=2) + '\n')
    os.environ.update(CONDA_SH=settings['conda_sh'], TRAIN_ENV=settings['train_env'],
                      REWARD_ENV=settings['reward_env'], PYTHONUNBUFFERED='1',
                      TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                      WANDB_MODE=settings.get('wandb_mode', 'offline'), OMP_NUM_THREADS='1')
    rl = (submission / 'repo_rl_path.txt').read_text().strip()
    # Prepare using the frozen upstream trainer, but import model code from the repository.
    subprocess.run([train_python, '-B', str(profile / 'prepare_profile.py'), '--run-dir', str(run),
                    '--config', settings['recipe'], '--repo-rl', rl,
                    '--trainer-source', str(submission / 'upstream_train.py')], check=True)
    reward_process = policy_process = None
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'Received signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        with (submission / 'logs/reward.log').open('w') as reward_log, \
             (submission / 'logs/policy.log').open('w') as policy_log:
            reward_process = subprocess.Popen(reward_step + ['bash', str(profile / 'start_reward.sh'), str(run)],
                                               stdout=reward_log, stderr=subprocess.STDOUT)
            url = f'http://{reward_hosts[0]}:{port}'
            wait_ready(url, str(run), reward_process, int(settings['reward_ready_timeout']))
            print(f'Reward ready: {url}; starting {expected_nodes * 8} policy GPUs', flush=True)
            policy_step = step(0, expected_nodes, p_resource['cpus_per_task'], p_resource['gres'])
            # Expand SLURM_PROCID on the remote task, one Accelerate launcher per policy node.
            remote = 'exec bash "$1" "$2" "${SLURM_PROCID:?}" "$3"'
            policy_process = subprocess.Popen(policy_step + ['bash', '-c', remote, 'policy',
                                               str(profile / 'launch_policy.sh'), str(run), policy_hosts[0]],
                                               stdout=policy_log, stderr=subprocess.STDOUT)
            while policy_process.poll() is None:
                if reward_process.poll() is not None:
                    raise RuntimeError('Reward server exited during training; stopping policy')
                time.sleep(5)
            if policy_process.returncode:
                raise RuntimeError(f'Policy exited with code {policy_process.returncode}; inspect logs/policy.log')
            print(f'Training complete; outputs are listed in {run / "launch.json"}', flush=True)
    finally:
        stop(policy_process)
        stop(reward_process)


if __name__ == '__main__':
    main()
