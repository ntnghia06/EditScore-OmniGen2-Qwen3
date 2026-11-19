"""Exercise real Accelerate CPU save/load and AdamW state; mock only PEFT APIs."""
import copy
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from accelerate import Accelerator
from compact_checkpoint import register_checkpoint_hooks, save_checkpoint, COMPLETE, MARKER
from test_omnigen2_gedit_fireward import ToyTransformer, Config, fake_peft
from generate_omnigen2_gedit_fireward import resolve_checkpoint, apply_checkpoint


class AdapterConfig(Config):
    def save_pretrained(self, directory):
        (Path(directory) / 'adapter_config.json').write_text(json.dumps(self.__dict__))


class Scheduler:
    def __init__(self):
        self.count = 7
    def state_dict(self):
        return {'count': self.count}
    def load_state_dict(self, state):
        self.count = state['count']


class CompactTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parent / '_verification' / self._testMethodName
        self.root.mkdir(parents=True, exist_ok=True)
    def tearDown(self):
        shutil.rmtree(self.root)
    def model(self):
        model = ToyTransformer()
        model.requires_grad_(False)
        config = AdapterConfig(r=1, lora_alpha=2, target_modules=['to_q'], bias='none', use_rslora=False, use_dora=False)
        model.add_adapter(config)
        model.peft_config = {'default': config}
        return model

    def test_real_accelerate_roundtrip_optimizer_scheduler_and_rng(self):
        accelerator = Accelerator(cpu=True)
        model = self.model()
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01)
        model, optimizer = accelerator.prepare(model, optimizer)
        reference = torch.ones(1, 2)
        loss = model(torch.zeros(1, 2), ref_image_hidden_states=reference).square().mean()
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
        scheduler = Scheduler()
        original = copy.deepcopy(model.state_dict())
        optimizer_original = copy.deepcopy(optimizer.state_dict())
        destination = self.root / 'checkpoint-20'
        with patch.dict(sys.modules, {'peft': fake_peft()}):
            register_checkpoint_hooks(accelerator, model, base_info={'transformer': '/frozen/base.bin'}, scheduler=scheduler)
            save_checkpoint(accelerator, model, destination)
            expected_random = torch.rand(5)
            self.assertFalse((destination / 'model.safetensors').exists())
            self.assertFalse((destination / 'pytorch_model.bin').exists())
            self.assertTrue((destination / COMPLETE).is_file())
            self.assertTrue((destination / MARKER).is_file())
            self.assertTrue((destination / 'optimizer.bin').is_file())
            self.assertTrue((destination / 'random_states_0.pkl').is_file())
            self.assertEqual(resolve_checkpoint(destination)['kind'], 'peft')
            inferred = apply_checkpoint(ToyTransformer(), resolve_checkpoint(destination))
            for key, value in original.items():
                self.assertTrue(torch.equal(inferred.state_dict()[key], value))
            with torch.no_grad():
                for parameter in model.parameters():
                    if parameter.requires_grad:
                        parameter.add_(50)
            optimizer.state.clear()
            scheduler.count = 100
            torch.manual_seed(999)
            accelerator.load_state(str(destination))
            self.assertTrue(torch.equal(torch.rand(5), expected_random))
        self.assertEqual(scheduler.count, 7)
        for key, value in original.items():
            self.assertTrue(torch.equal(model.state_dict()[key], value))
        restored_optimizer = optimizer.state_dict()
        self.assertEqual(restored_optimizer['param_groups'], optimizer_original['param_groups'])
        for key, values in optimizer_original['state'].items():
            for field, value in values.items():
                self.assertTrue(torch.equal(restored_optimizer['state'][key][field], value))
        # Hooks must not remove Accelerate's model registration during load.
        self.assertEqual(len(accelerator._models), 1)
        with patch.dict(sys.modules, {'peft': fake_peft()}):
            save_checkpoint(accelerator, model, self.root / 'checkpoint-40')

    def test_failed_write_never_publishes_checkpoint(self):
        destination = self.root / 'checkpoint-20'
        def fail(path):
            Path(path).mkdir()
            (Path(path) / 'optimizer.bin').write_bytes(b'partial')
            raise OSError(122, 'Disk quota exceeded')
        accelerator = SimpleNamespace(save_state=fail, _omnigen_checkpoint_format='lora')
        with self.assertRaises(OSError):
            save_checkpoint(accelerator, None, destination)
        self.assertFalse(destination.exists())
        self.assertTrue((self.root / '.checkpoint-20.incomplete').exists())

    def test_backbone_must_be_frozen(self):
        accelerator = Accelerator(cpu=True)
        model = self.model()
        model.backbone.requires_grad_(True)
        with self.assertRaisesRegex(ValueError, 'LoRA-only'):
            register_checkpoint_hooks(accelerator, model)

    def test_legacy_full_checkpoint_still_loads(self):
        accelerator = Accelerator(cpu=True)
        model = self.model()
        model = accelerator.prepare(model)
        original = copy.deepcopy(model.state_dict())
        legacy = self.root / 'legacy'
        accelerator.save_state(str(legacy))
        with torch.no_grad():
            for p in model.parameters():
                p.add_(10)
        with patch.dict(sys.modules, {'peft': fake_peft()}):
            register_checkpoint_hooks(accelerator, model)
            accelerator.load_state(str(legacy))
        for key, value in original.items():
            self.assertTrue(torch.equal(model.state_dict()[key], value))

    def test_actual_timm_scheduler_state_is_weights_only_compatible(self):
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from omnigen2.optim.scheduler.step_lr import StepLRScheduler
        parameter = torch.nn.Parameter(torch.zeros(1))
        optimizer = torch.optim.AdamW([parameter], lr=5e-5)
        scheduler = StepLRScheduler(optimizer, decay_t=1, decay_rate=1,
                                   warmup_t=20, warmup_lr_init=1e-7, t_in_epochs=False)
        scheduler.step(12)
        original = scheduler.state_dict()
        file = self.root / 'scheduler.pt'
        torch.save(original, file)
        state = torch.load(file, map_location='cpu', weights_only=True)
        scheduler.warmup_t = 100
        scheduler.load_state_dict(state)
        self.assertEqual(scheduler.state_dict(), original)


if __name__ == '__main__':
    unittest.main(verbosity=2)
