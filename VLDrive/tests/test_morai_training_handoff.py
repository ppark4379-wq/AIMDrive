"""Handoff lifecycle tests use mocked PIDs; never signal a training process."""
import json
import signal
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch

from tools import morai_training_handoff as handoff


class HandoffCleanupTest(unittest.TestCase):
    def test_exit_race_during_child_termination_still_continues_original(self):
        child = MagicMock(pid=98765)
        child.poll.return_value = None
        child.wait.return_value = 0
        report = {}
        with patch.object(handoff.os, 'killpg', side_effect=ProcessLookupError), \
             patch.object(handoff, 'signal_original') as original:
            handoff.rollback(child, 43210, 'token', True, report)
        original.assert_called_once_with(43210, 'token', signal.SIGCONT)
        self.assertTrue(report['old_training_resumed'])
        child.wait.assert_called_once_with(timeout=15)

    def test_failed_child_cleanup_cannot_skip_original_continuation(self):
        report = {}
        with patch.object(handoff, 'stop_replacement', side_effect=RuntimeError('cleanup failed')), \
             patch.object(handoff, 'signal_original') as original:
            with self.assertRaisesRegex(RuntimeError, 'cleanup failed'):
                handoff.rollback(object(), 43210, 'token', True, report)
        original.assert_called_once_with(43210, 'token', signal.SIGCONT)
        self.assertTrue(report['old_training_resumed'])

    def test_force_kill_exit_race_is_tolerated(self):
        child = MagicMock(pid=98765)
        child.poll.return_value = None
        child.wait.side_effect = [subprocess.TimeoutExpired('replacement', 15), 0]
        with patch.object(handoff.os, 'killpg', side_effect=[None, ProcessLookupError]) as kill:
            handoff.stop_replacement(child)
        self.assertEqual([call.args[1] for call in kill.call_args_list], [signal.SIGTERM, signal.SIGKILL])
        self.assertEqual(child.wait.call_count, 2)

    def test_recycled_original_pid_is_never_signaled(self):
        with patch.object(handoff, 'process_identity', return_value=([], 'new-token')), \
             patch.object(handoff.os, 'kill') as kill:
            with self.assertRaisesRegex(RuntimeError, 'PID identity changed'):
                handoff.signal_original(43210, 'old-token', signal.SIGCONT)
        kill.assert_not_called()


class HandoffSnapshotTest(unittest.TestCase):
    def test_invalid_optimizer_snapshot_is_not_published(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, destination = Path(tmp) / 'source.pt', Path(tmp) / 'handoff.pt'
            torch.save({'state_dict': {'weight': torch.ones(1)}, 'steps': 7, 'epoch': 1,
                        'optimizer': {'state': {}}}, source)
            with self.assertRaisesRegex(ValueError, 'usable weights/optimizer'):
                handoff.snapshot_checkpoint(source, destination)
            self.assertFalse(destination.exists())
            self.assertTrue(source.exists())

    def test_valid_snapshot_is_a_verified_independent_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, destination = Path(tmp) / 'source.pt', Path(tmp) / 'handoff.pt'
            payload = {'state_dict': {'weight': torch.tensor([3.])}, 'steps': 7, 'epoch': 1,
                       'optimizer': {'state': {0: {'step': torch.tensor(7.)}}}}
            torch.save(payload, source)
            result = handoff.snapshot_checkpoint(source, destination)
            self.assertEqual(result['sha256'], handoff.sha256(source))
            self.assertEqual(result['sha256'], handoff.sha256(destination))
            source.write_bytes(b'later checkpoint')
            self.assertEqual(torch.load(destination, weights_only=True)['steps'], 7)
            self.assertEqual(result['steps'], 7)
            self.assertFalse(destination.with_suffix('.pt.partial').exists())


class HandoffCoordinatorTest(unittest.TestCase):
    def run_coordinator(self, failure=None):
        events = []
        result = {}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old, control_root, config = root / 'old', root / 'control', root / 'config.json'
            old.mkdir()
            prior = {'batch_size': 1, 'initial_checkpoint_sha256': 'anchor-hash',
                     'trainable_scope': 'full', 'velocity_source': 'ego_topic',
                     'args': {'frame_sampling': 'all', 'checkpoint_selection': 'scenario_guarded',
                              'checkpoint_interval': 1000, 'epochs': 3, 'learning_rate': 1e-5,
                              'detection_loss_policy': 'lmdrive_lav_v1', 'early_stop_patience': 1,
                              'threads': 4, 'seed': 42, 'log_interval': 50}}
            (old / 'config.json').write_text(json.dumps(prior))
            config.write_text(json.dumps({'checkpoint_sha256': 'anchor-hash'}))
            source = old / 'checkpoint-progress.pth.tar'
            source.write_bytes(b'previous durable checkpoint')
            child = MagicMock(pid=98765)
            child_dead = [False]
            child.poll.side_effect = lambda: 1 if child_dead[0] else None
            handlers = {signal.SIGINT: signal.default_int_handler, signal.SIGTERM: signal.SIG_DFL}
            originals = handlers.copy()
            handler_calls = []

            def replace_handler(signum, callback):
                previous = handlers[signum]
                handlers[signum] = callback
                handler_calls.append((signum, callback))
                return previous

            def launch(command, **kwargs):
                result['new'] = Path(command[command.index('--output') + 1])
                events.append('launch')
                return child

            def source_identity(pid):
                # Simulate one completed atomic periodic save after baseline.
                temporary = source.with_suffix('.fresh')
                temporary.write_bytes(b'new durable checkpoint')
                temporary.replace(source)
                events.append('new_checkpoint')
                return [], 'old-token'

            def old_signal(pid, token, signum):
                self.assertEqual((pid, token), (43210, 'old-token'))
                events.append(('original_signal', signum))
                if failure == 'during_stop' and signum == signal.SIGSTOP:
                    handlers[signal.SIGTERM](signal.SIGTERM, None)

            def snapshot(src, dest):
                self.assertEqual(src, source)
                events.append('snapshot')
                return {'steps': 1000, 'epoch': 1, 'sha256': 'snapshot-hash'}

            def wait(path, process, timeout, phase):
                events.append(phase)
                if phase == 'prepare_corrected_baseline':
                    return {'baseline_completed': True, 'anchor_sha256': 'anchor-hash'}
                if phase == 'restore_model_and_optimizer':
                    if failure == 'sigterm':
                        handlers[signal.SIGTERM](signal.SIGTERM, None)
                    return {'weights_identical': True,
                            'optimizer_state_identical': failure != 'bad_restore',
                            'checkpoint_sha256': 'snapshot-hash', 'restored_global_step': 1000}
                if phase == 'first_resumed_update':
                    if failure == 'dead_after_save':
                        child_dead[0] = True
                    return {'from_step': 1000, 'step': 1001,
                            'checkpoint': str(result['new'] / 'checkpoint-progress.pth.tar'),
                            'checkpoint_sha256': 'saved-hash'}
                self.fail('Unexpected phase: ' + phase)

            def check_saved(path):
                events.append('verified_saved_hash')
                return 'wrong-hash' if failure == 'bad_saved_hash' else 'saved-hash'

            argv = ['handoff', '--old-run', str(old), '--config', str(config),
                    '--control-root', str(control_root)]
            with patch.object(handoff.sys, 'argv', argv), \
                 patch.object(handoff, 'find_trainer', return_value=(43210, 'old-token')), \
                 patch.object(handoff, 'process_identity', side_effect=source_identity), \
                 patch.object(handoff, 'signal_original', side_effect=old_signal), \
                 patch.object(handoff.signal, 'signal', side_effect=replace_handler), \
                 patch.object(handoff.subprocess, 'Popen', side_effect=launch), \
                 patch.object(handoff, 'snapshot_checkpoint', side_effect=snapshot), \
                 patch.object(handoff, 'wait_for', side_effect=wait), \
                 patch.object(handoff, 'sha256', side_effect=check_saved), \
                 patch.object(handoff, 'stop_replacement', side_effect=lambda c: events.append('stop_replacement')), \
                 patch('builtins.print'):
                if failure is None:
                    handoff.main()
                else:
                    with self.assertRaises((ValueError, RuntimeError)):
                        handoff.main()
            reports = list(control_root.glob('handoff_*/handoff.json'))
            self.assertEqual(len(reports), 1)
            result['report'] = json.loads(reports[0].read_text())
            result['events'] = events
            result['handlers'] = handler_calls
            self.assertEqual(handlers, originals)
        return result

    def test_original_terminated_only_after_restore_and_first_saved_update(self):
        result = self.run_coordinator()
        events = result['events']
        term = events.index(('original_signal', signal.SIGTERM))
        for proof in ('prepare_corrected_baseline', 'snapshot', 'restore_model_and_optimizer',
                      'first_resumed_update', 'verified_saved_hash'):
            self.assertLess(events.index(proof), term)
        self.assertEqual(events[-1], ('original_signal', signal.SIGCONT))
        self.assertNotIn('stop_replacement', events)
        self.assertEqual(result['report']['status'], 'resumed')

    def test_invalid_restore_or_saved_proof_rolls_back_without_old_termination(self):
        for failure in ('bad_restore', 'bad_saved_hash', 'dead_after_save'):
            with self.subTest(failure=failure):
                result = self.run_coordinator(failure)
                self.assertIn('stop_replacement', result['events'])
                self.assertIn(('original_signal', signal.SIGCONT), result['events'])
                self.assertNotIn(('original_signal', signal.SIGTERM), result['events'])
                self.assertEqual(result['report']['status'], 'rolled_back')

    def test_sigterm_and_stop_window_interruptions_route_through_cleanup(self):
        for failure in ('sigterm', 'during_stop'):
            with self.subTest(failure=failure):
                result = self.run_coordinator(failure)
                self.assertIn((signal.SIGTERM, handoff.interrupt_handoff), result['handlers'])
                self.assertIn((signal.SIGTERM, signal.SIG_IGN), result['handlers'])
                self.assertIn(('original_signal', signal.SIGCONT), result['events'])
                self.assertNotIn(('original_signal', signal.SIGTERM), result['events'])
                self.assertTrue(result['report']['old_training_resumed'])
                self.assertEqual(result['report']['status'], 'rolled_back')


if __name__ == '__main__':
    unittest.main()
