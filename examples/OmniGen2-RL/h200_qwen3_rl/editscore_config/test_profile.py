"""CPU checks for original recipe preservation and executable run preparation."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import unittest
from unittest.mock import patch

from PIL import Image
import yaml
import prepare_profile

PROFILE = Path(__file__).resolve().parent
BUNDLE = PROFILE.parent
RL = BUNDLE.parent


class ProfileTests(unittest.TestCase):
    def test_recipes_preserve_original_hyperparameters(self):
        for recipe, original in (
            ('train.yml', 'omnigen2_edit_rl_single_machine_editscore7b.yml'),
            ('train_4machine.yml', 'omnigen2_edit_rl_4machine_editscore7b_avg4.yml'),
        ):
            actual = yaml.safe_load((PROFILE / recipe).read_text())
            expected = yaml.safe_load((RL / 'options' / original).read_text())
            prepare_profile.check_recipe(actual, expected)
            # Also cover top-level values beyond the required train/model sections.
            for key in expected:
                if key not in ('name', 'reward_server_config', 'data'):
                    self.assertEqual(actual[key], expected[key], key)

    def test_reference_manifest(self):
        root = PROFILE / 'upstream_reference'
        for row in json.loads((root / 'manifest.json').read_text())['files']:
            self.assertEqual(hashlib.sha256((root / row['path']).read_bytes()).hexdigest(), row['sha256'])

    def test_prepare_real_source_and_two_condition_data(self):
        root = PROFILE / '_verification'
        root.mkdir(exist_ok=False)
        try:
            fake_rl = root / 'rl'
            bundle = fake_rl / 'h200_qwen3_rl'
            profile = bundle / 'editscore_config'
            profile.mkdir(parents=True)
            for name in ('prepare_run.py', 'reward_server.py', 'reward_runtime.py'):
                shutil.copy2(BUNDLE / name, bundle / name)
            shutil.copy2(RL / 'train.py', fake_rl / 'train.py')
            shutil.copy2(PROFILE / 'train.yml', profile / 'train.yml')
            (profile / 'upstream_reference/options').mkdir(parents=True)
            shutil.copy2(RL / 'options/omnigen2_edit_rl_single_machine_editscore7b.yml', profile / 'upstream_reference/options')
            project = root / 'reward_project'
            (project / 'src/models').mkdir(parents=True)
            (project / 'src/models/custom_qwen3_vl_reranker.py').write_text('# fixture')
            pretrained = root / 'reward_pretrained'
            pretrained.mkdir()
            (pretrained / 'config.json').write_text('{}')
            checkpoint = root / 'checkpoint.pt'
            checkpoint.write_bytes(b'fixture')
            transformer = root / 'transformer.bin'
            transformer.write_bytes(b'fixture')
            for folder, config in (('text', 'config.json'), ('vae', 'vae/config.json')):
                path = root / folder / config
                path.parent.mkdir(parents=True)
                path.write_text('{}')
            Image.new('RGB', (32, 48)).save(root / 'source.png')
            metadata = root / 'metadata.jsonl'
            metadata.write_text('\n'.join(json.dumps({'key': str(i), 'instruction': f'edit {i}', 'source_img': 'source.png'}) for i in range(2)))
            settings = dict(metadata=str(metadata.resolve()), data_root=str(root.resolve()),
                            reward_project=str(project.resolve()), reward_pretrained=str(pretrained.resolve()),
                            reward_checkpoint=str(checkpoint.resolve()), omnigen_transformer_bin=str(transformer.resolve()),
                            text_encoder=str((root / 'text').resolve()), vae_model=str((root / 'vae').resolve()), hf_cache_roots=[])
            (bundle / 'settings.json').write_text(json.dumps(settings))
            overrides = {'base_settings': '../settings.json', 'reward_host': 'reward-node',
                         'reward_bind_host': '0.0.0.0', 'port': 18888}
            (profile / 'settings.json').write_text(json.dumps(overrides))
            run = root / 'run'
            with patch.object(prepare_profile, '__file__', str(profile / 'prepare_profile.py')), \
                 patch.object(sys, 'argv', ['prepare_profile.py', '--run-dir', str(run), '--write-probe-mib', '0']):
                prepare_profile.main()
            topology = json.loads((run / 'launch.json').read_text())
            self.assertEqual((topology['policy_processes'], topology['machines']), (8, 1))
            config = yaml.safe_load((run / 'train.yml').read_text())
            self.assertEqual(config['train']['batch_size'], 18)
            self.assertEqual(config['train']['max_train_steps'], 1000)
            self.assertEqual(yaml.safe_load((run / 'reward_server.yml').read_text())['server']['hosts'], ['reward-node'])
            rows = [json.loads(line) for line in (run / 'training_pairs.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows), 2)
            self.assertEqual(json.loads(rows[0]['meta_data'])['sample_key'], '0')
            self.assertEqual(rows[0]['input_images'], [str((root / 'source.png').resolve())])
            trainer = (run / 'train_distributed.py').read_text()
            ast.parse(trainer)
            self.assertIn('torch.distributed.scatter_object_list', trainer)
            self.assertIn('if global_step >= args.train.max_train_steps:', trainer)
            server = (run / 'reward_server.py').read_text()
            ast.parse(server)
            self.assertIn("worker.settings.get('reward_bind_host'", server)
            self.assertEqual(json.loads((run / 'settings.json').read_text())['reward_checkpoint'], str(checkpoint.resolve()))
        finally:
            shutil.rmtree(root)


if __name__ == '__main__':
    unittest.main()
