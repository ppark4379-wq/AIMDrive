"""Bounded overnight runs: lifecycle, failure persistence and no fake PASS."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from MoraiDrive.expert.tools.run_ve_hz_check import campaign, findings, wait_background_start


class CampaignTests(unittest.TestCase):
    def run_campaign(self, root, failure=None, interrupted=False):
        sessions = []
        clock = [100.]
        def factory(*args, **kwargs):
            self.assertFalse(kwargs['observe_gt'])
            session = Mock(error=None, rejected=0)
            session.sender.poll.return_value = None
            session.performance_line.return_value = 'test Hz'
            session.snapshot.return_value = dict(ve_fresh=True)
            if failure:
                session.await_ready.side_effect = failure
            sessions.append(session)
            return session
        def report(folder):
            path = folder/'timing_report.json'
            path.write_text(json.dumps(dict(accepted_outputs=10,jpeg_warning_occurrences=2,
                input_selection=dict(lidar=dict(reused_selected_sample=0)))))
            return path
        def sleep(seconds):
            if interrupted:
                raise KeyboardInterrupt()
            clock[0] += seconds
        with patch('MoraiDrive.expert.tools.run_ve_hz_check.time.monotonic',side_effect=lambda:clock[0]), \
             patch('MoraiDrive.expert.tools.run_ve_hz_check.time.sleep',side_effect=sleep), \
             patch('MoraiDrive.expert.tools.run_ve_hz_check.process_rss_mib',return_value=123.):
            code = campaign(root,'E3',4.,2,factory,Mock(),lambda:False,report)
        return code, sessions, json.loads((root/'campaign_report.json').read_text())

    def test_success_is_measuring_complete_not_a_semantic_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            code,sessions,report = self.run_campaign(Path(tmp))
        self.assertEqual(code,0)
        self.assertEqual(report['status'],'COMPLETE')
        self.assertEqual(report['completed_segments'],2)
        self.assertEqual(report['segments_with_issues'],2)
        self.assertEqual(report['control_commands_sent'],0)
        self.assertFalse(report['performance_passed_automatically'])
        for session in sessions:
            session.close.assert_called_once()

    def test_failed_start_stops_repetition_preserves_report_and_closes_child(self):
        with tempfile.TemporaryDirectory() as tmp:
            code,sessions,report = self.run_campaign(Path(tmp),RuntimeError('camera missing'))
        self.assertEqual(code,1)
        self.assertEqual(len(sessions),1)
        sessions[0].close.assert_called_once()
        self.assertEqual(report['status'],'FAILED')
        self.assertEqual(report['reason'],'camera missing')
        self.assertEqual(report['completed_segments'],0)

    def test_interruption_closes_model_and_keeps_partial_measurement(self):
        with tempfile.TemporaryDirectory() as tmp:
            code,sessions,report = self.run_campaign(Path(tmp),interrupted=True)
        self.assertEqual(code,130)
        sessions[0].close.assert_called_once()
        self.assertEqual(report['status'],'INTERRUPTED')
        self.assertFalse(report['segments'][0]['duration_completed'])

    def test_correct_reassembly_still_reports_remaining_fragment_losses(self):
        report = dict(accepted_outputs=100,jpeg_warning_occurrences=0,
            lidar_gate_counter_differs_from_selected_sample=0,
            sensors=dict(rgb_front=dict(final_counters=dict(counters=dict(
                frames_started=120,end_markers=110,discarded_torn_images=10)))),
            input_selection=dict(lidar=dict(reused_selected_sample=0)))
        self.assertEqual(findings(report),dict(rgb_front=dict(discarded_torn_images=10)))

    def test_background_success_requires_fresh_input_and_failures_surface(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            child=Mock()
            child.poll.return_value=None
            states=[dict(status='RUNNING',phase='model_and_sensors_starting'),
                    dict(status='RUNNING',phase='measuring',input_fresh=True,performance='9Hz')]
            (root/'status.json').write_text(json.dumps(states[0]))
            with patch('MoraiDrive.expert.tools.run_ve_hz_check.time.sleep',
                       side_effect=lambda _: (root/'status.json').write_text(json.dumps(states[1]))) as sleep:
                self.assertEqual(wait_background_start(root,child),0)
                sleep.assert_called_once()
            (root/'status.json').write_text(json.dumps(dict(status='FAILED',reason='missing lidar')))
            self.assertEqual(wait_background_start(root,child),1)


if __name__ == '__main__':
    unittest.main()
