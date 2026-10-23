"""GT filming must reject pre-existing contacts before starting UDP control."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from ..tools.run_gt_udp_full_route import GtUdpFullLoop, UdpController


class GtUdpStartupTests(unittest.TestCase):
    def test_clean_start_records_raw_state(self):
        with tempfile.TemporaryDirectory() as directory:
            state = dict(x=1., y=2., collision_objects=[])
            context = SimpleNamespace(gt_output=Path(directory),
                                      get_ego_motion_state=lambda: state)
            GtUdpFullLoop.check_startup_contact(context)
            record = json.loads((context.gt_output / 'startup_contact.json').read_text())
            self.assertEqual(record['raw_state'], state)
            self.assertFalse(record['control_backend_started'])

    def test_self_and_foreign_contacts_are_not_silently_filtered(self):
        for contacts in (['Ego'], ['vehicle_2'], ['Ego', 'vehicle_2']):
            with self.subTest(contacts=contacts), tempfile.TemporaryDirectory() as directory:
                context = SimpleNamespace(gt_output=Path(directory),
                    get_ego_motion_state=lambda: dict(collision_objects=contacts))
                with self.assertRaisesRegex(RuntimeError, '출발 전부터'):
                    GtUdpFullLoop.check_startup_contact(context)
                self.assertEqual(json.loads((context.gt_output / 'startup_contact.json').read_text())
                                 ['collision_objects'], sorted(contacts))

    def test_contact_failure_does_not_launch_backend(self):
        context = SimpleNamespace(is_running=lambda: False,
            before_start=Mock(side_effect=RuntimeError('contact')))
        # No process, log, or output directory access may occur after rejection.
        with self.assertRaisesRegex(RuntimeError, 'contact'):
            UdpController.start_process(context)


if __name__ == '__main__':
    unittest.main()
