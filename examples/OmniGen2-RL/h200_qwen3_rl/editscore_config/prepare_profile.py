#!/usr/bin/env python3
"""Freeze the upstream distributed recipe with existing Qwen3/data settings."""
import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil


def load_helpers(bundle):
    spec = importlib.util.spec_from_file_location('_editscore_prepare_helpers', bundle / 'prepare_run.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_recipe(config, reference):
    """Hyperparameters must match the selected upstream file exactly."""
    for key in ('train', 'transport', 'val', 'logger', 'seed', 'data', 'model'):
        actual, expected = config[key], reference[key]
        if key == 'data':
            actual, expected = dict(actual), dict(expected)
            actual.pop('data_path')
            expected.pop('data_path')
        if actual != expected:
            raise ValueError(f'{key} differs from upstream recipe; this profile requires original settings')


def main():
    import yaml
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', choices=('train.yml', 'train_4machine.yml'), default='train.yml')
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--write-probe-mib', type=int, default=512)
    args = parser.parse_args()
    profile = Path(__file__).resolve().parent
    bundle, rl = profile.parent, profile.parent.parent
    helper = load_helpers(bundle)
    config = yaml.safe_load((profile / args.config).read_text(encoding='utf-8'))
    original = ('omnigen2_edit_rl_single_machine_editscore7b.yml' if args.config == 'train.yml'
                else 'omnigen2_edit_rl_4machine_editscore7b_avg4.yml')
    reference = yaml.safe_load((profile / 'upstream_reference/options' / original).read_text())
    check_recipe(config, reference)
    overrides = json.loads((profile / 'settings.json').read_text(encoding='utf-8'))
    base = Path(overrides.pop('base_settings'))
    base = base if base.is_absolute() else profile / base
    settings = json.loads(base.read_text(encoding='utf-8'))
    settings.update(overrides)
    for path in (Path(settings['reward_project']) / 'src/models/custom_qwen3_vl_reranker.py',
                 Path(settings['reward_pretrained']) / 'config.json', Path(settings['reward_checkpoint'])):
        if not path.exists() or not os.access(path, os.R_OK):
            raise FileNotFoundError(f'Missing/unreadable reward input: {path}')
    if not 1 <= int(settings['port']) <= 65535:
        raise ValueError('Reward port must be in 1..65535')
    run = args.run_dir.resolve()
    run.mkdir(parents=True, exist_ok=False)
    helper.probe_write_capacity(run, args.write_probe_mib)
    roots = settings['hf_cache_roots']
    transformer = Path(settings['omnigen_transformer_bin']) if settings['omnigen_transformer_bin'] else rl / 'pretrained_models/OmniGen2/transformer/pytorch_model.bin'
    if not transformer.is_file():
        for root in roots:
            candidate = Path(root) / 'OmniGen2/transformer/pytorch_model.bin'
            if candidate.is_file():
                transformer = candidate
                break
    if not transformer.is_file():
        raise FileNotFoundError('Set omnigen_transformer_bin in inherited settings.json')
    config['model']['pretrained_model_path'] = str(transformer.resolve())
    config['model']['pretrained_text_encoder_model_name_or_path'] = str(helper.resolve_model(settings['text_encoder'], 'Qwen/Qwen2.5-VL-3B-Instruct', 'config.json', roots))
    config['model']['pretrained_vae_model_name_or_path'] = str(helper.resolve_model(settings['vae_model'], 'black-forest-labs/FLUX.1-dev', 'vae/config.json', roots))
    rows = helper.prepare_rows(Path(settings['metadata']), Path(settings['data_root']))
    with (run / 'training_pairs.jsonl').open('w', encoding='utf-8', newline='\n') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    (run / 'data.yml').write_text(yaml.safe_dump({'data': [{'path': str(run / 'training_pairs.jsonl'), 'ratio': 1}]}))
    config['data']['data_path'] = str(run / 'data.yml')
    config['reward_server_config'] = str(run / 'reward_server.yml')
    reward_config = {'server': {'hosts': [settings['reward_host']], 'proxy_port': settings['port']}}
    (run / 'reward_server.yml').write_text(yaml.safe_dump(reward_config))
    (run / 'train.yml').write_text(yaml.safe_dump(config, sort_keys=False))
    (run / 'settings.json').write_text(json.dumps(settings, indent=2), encoding='utf-8')
    # Use upstream distributed/FSDP training, never the single-GPU trainer.
    trainer = (rl / 'train.py').read_text(encoding='utf-8')
    # Stop at the requested optimizer limit even before the dataloader epoch ends.
    needle = '        for step, batch in enumerate(train_dataloader):\n'
    if trainer.count(needle) != 1:
        raise ValueError('Upstream training loop changed; review the max-step guard')
    trainer = trainer.replace(needle, needle + "            if global_step >= args.train.max_train_steps:\n                break\n")
    ast.parse(trainer)
    (run / 'train_distributed.py').write_text(trainer, encoding='utf-8', newline='\n')
    reward = (bundle / 'reward_server.py').read_text(encoding='utf-8')
    needle = "HTTPServer(('127.0.0.1', port), Handler)"
    if reward.count(needle) != 1:
        raise ValueError('Reward server binding changed; review distributed binding')
    reward = reward.replace(needle, "HTTPServer((worker.settings.get('reward_bind_host', '127.0.0.1'), port), Handler)")
    (run / 'reward_server.py').write_text(reward, encoding='utf-8', newline='\n')
    shutil.copy2(bundle / 'reward_runtime.py', run / 'reward_runtime.py')
    (run / 'repo_rl_path.txt').write_text(str(rl))
    world_size = config['train']['global_batch_size'] // config['train']['batch_size']
    metadata = {'policy_processes': world_size, 'machines': world_size // 8,
                'dataset_pairs': len(rows), 'reward_host': settings['reward_host'],
                'reward_port': settings['port'], 'upstream_train_sha256': hashlib.sha256((rl / 'train.py').read_bytes()).hexdigest(),
                'output_dir': str(run / 'experiments' / config['name'])}
    (run / 'launch.json').write_text(json.dumps(metadata, indent=2))
    print(json.dumps({'run': str(run), **metadata}, indent=2), flush=True)


if __name__ == '__main__':
    main()
