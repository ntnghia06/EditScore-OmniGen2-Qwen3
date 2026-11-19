"""CPU checks for submission snapshots, allocation layout and process cleanup."""
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import auto_run
import submit_train

PROFILE = Path(__file__).resolve().parent


class AutoTests(unittest.TestCase):
    def setUp(self):
        self.root = PROFILE / '_auto_verification'
        self.root.mkdir(exist_ok=False)
        self.settings = json.loads((PROFILE / 'settings.json').read_text())
    def tearDown(self):
        shutil.rmtree(self.root)

    def test_both_allocation_layouts(self):
        for recipe, nodes in (('train.yml', 1), ('train_4machine.yml', 4)):
            self.settings['recipe'] = recipe
            text = submit_train.render_job(Path('/tmp/submission'), self.settings, Path('/tmp/profile'))
            self.assertEqual(text.count('#SBATCH hetjob'), 1)
            policy, reward = text.split('#SBATCH hetjob')
            self.assertIn(f'#SBATCH --nodes={nodes}', policy)
            self.assertIn('#SBATCH --gres=gpu:8', policy)
            self.assertIn('#SBATCH --nodes=1', reward)
            self.assertIn('#SBATCH --gres=gpu:1', reward)
            self.assertIn('auto_run.py', text)
        with self.assertRaises(ValueError):
            submit_train.directive('qos', 'normal\n#SBATCH --exclusive')

    def test_snapshot_preserves_settings_and_trainer(self):
        destination = self.root / 'submission'
        job = submit_train.freeze_submission(PROFILE, self.settings, destination)
        frozen = destination / 'snapshot/h200_qwen3_rl/editscore_config'
        self.assertTrue(job.is_file())
        self.assertEqual(json.loads((frozen / 'settings.json').read_text()), self.settings)
        self.assertEqual((destination / 'upstream_train.py').read_bytes(), (PROFILE.parent.parent / 'train.py').read_bytes())
        self.assertTrue((frozen.parent / 'prepare_run.py').is_file())
        self.assertTrue((frozen / 'upstream_reference/manifest.json').is_file())
        self.settings['recipe'] = 'train_4machine.yml'
        self.assertEqual(json.loads((frozen / 'settings.json').read_text())['recipe'], 'train.yml')

    def test_success_stops_reward_after_policy_finishes(self):
        submission = self.root / 'submission'
        submit_train.freeze_submission(PROFILE, self.settings, submission)
        frozen = submission / 'snapshot/h200_qwen3_rl/editscore_config'
        reward, policy = Mock(), Mock()
        reward.poll.return_value = None
        policy.poll.side_effect = [None, 0, 0]
        policy.returncode = 0
        try:
            with patch.object(auto_run, '__file__', str(frozen / 'auto_run.py')), \
                 patch.object(auto_run.sys, 'argv', ['auto_run.py', str(submission)]), \
                 patch.object(auto_run, 'hosts', side_effect=[['policy-0'], ['reward-0']]), \
                 patch.object(auto_run.subprocess, 'check_output', return_value='18888\n'), \
                 patch.object(auto_run.subprocess, 'run') as prepare, \
                 patch.object(auto_run.subprocess, 'Popen', side_effect=[reward, policy]) as popen, \
                 patch.object(auto_run, 'wait_ready') as ready, \
                 patch.object(auto_run.time, 'sleep'), \
                 patch.object(auto_run.signal, 'signal'), \
                 patch.dict(auto_run.os.environ, {}, clear=False):
                auto_run.main()
                self.assertIn('--trainer-source', prepare.call_args.args[0])
                self.assertEqual(ready.call_args.args[0], 'http://reward-0:18888')
                self.assertEqual(len(popen.call_args_list), 2)
                self.assertIn('${SLURM_PROCID:?}', popen.call_args_list[1].args[0][-5])
            reward.terminate.assert_called_once()
            policy.terminate.assert_not_called()
        finally:
            pass

    def test_readiness_rejects_other_jobs_server(self):
        response = Mock()
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        process = SimpleNamespace(poll=lambda: None)
        with patch.object(auto_run.time, 'monotonic', side_effect=[0, 0, 2]), \
             patch.object(auto_run.time, 'sleep'), \
             patch.object(auto_run.urllib.request, 'urlopen', return_value=response), \
             patch.object(auto_run.json, 'load', return_value={'ready': True, 'run_id': 'other-job'}):
            with self.assertRaises(TimeoutError):
                auto_run.wait_ready('http://reward:1', 'this-job', process, 1)

    def test_policy_failure_still_stops_reward(self):
        submission = self.root / 'submission'
        submit_train.freeze_submission(PROFILE, self.settings, submission)
        frozen = submission / 'snapshot/h200_qwen3_rl/editscore_config'
        reward, policy = Mock(), Mock()
        reward.poll.return_value = None
        policy.poll.return_value = 1
        policy.returncode = 1
        with patch.object(auto_run, '__file__', str(frozen / 'auto_run.py')), \
             patch.object(auto_run.sys, 'argv', ['auto_run.py', str(submission)]), \
             patch.object(auto_run, 'hosts', side_effect=[['policy-0'], ['reward-0']]), \
             patch.object(auto_run.subprocess, 'check_output', return_value='18888\n'), \
             patch.object(auto_run.subprocess, 'run'), \
             patch.object(auto_run.subprocess, 'Popen', side_effect=[reward, policy]), \
             patch.object(auto_run, 'wait_ready'), \
             patch.object(auto_run.signal, 'signal'), \
             patch.dict(auto_run.os.environ, {}, clear=False):
            with self.assertRaisesRegex(RuntimeError, 'Policy exited'):
                auto_run.main()
        reward.terminate.assert_called_once()


if __name__ == '__main__':
    unittest.main()
