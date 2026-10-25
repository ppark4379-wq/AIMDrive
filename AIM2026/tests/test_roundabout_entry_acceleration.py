"""Entry acceleration applies to admitted following, not post-merge cruising."""
from types import SimpleNamespace as NS
import unittest

from AIM2026.Expert.src.control.path_controller import PathController
from AIM2026.Expert.src.control.pid import Pid
from AIM2026.Expert.src.config.parameters import lattice as p


class RoundaboutEntryAccelerationTests(unittest.TestCase):
    def result(self, **changes):
        details = dict(phase='follow', front_gap=8., entering=True,
                       committed=False, hold_gap=-1., speed_limits={'path_speed':8.})
        details.update(changes)
        return NS(mode='roundabout_entry',acc_target_velocity=4.,
                  debug_details={'roundabout_entry':details})

    def gain(self, result, target=4., speed=3.):
        return PathController._roundabout_entry_acceleration_gain(target,NS(velocity=speed),result)

    def test_follow_entry_uses_gain_without_changing_safe_target(self):
        result = self.result()
        gain = self.gain(result)
        self.assertEqual(gain,p.ROUNDABOUT_ENTRY_ACCEL_P_GAIN)
        base = Pid(.1,0.,0.,.03).get_output(4.,3.,True)
        faster = Pid(.1,0.,0.,.03).get_output(4.,3.,True, gain)
        self.assertGreater(faster,base)
        self.assertLessEqual(faster,1.)
        self.assertEqual(result.acc_target_velocity,4.)

    def test_hold_point_allowed_but_earlier_approach_not_boosted(self):
        self.assertIsNotNone(self.gain(self.result(entering=False,hold_gap=.2)))
        self.assertIsNone(self.gain(self.result(entering=False,hold_gap=4.)))

    def test_committed_yield_collision_and_other_limits_do_not_boost(self):
        for details in ({'committed':True}, {'phase':'yield'}, {'phase':'emergency'},
                        {'phase':'collision_slowdown'}, {'phase':'invalid_path'}):
            with self.subTest(details=details):
                self.assertIsNone(self.gain(self.result(**details)))
        self.assertIsNone(self.gain(self.result(),target=3.5))
        self.assertIsNone(self.gain(self.result(),target=0.))
        self.assertIsNone(self.gain(self.result(),speed=5.))
        result=self.result();result.mode='highway_lane'
        self.assertIsNone(self.gain(result))

    def test_gain_never_strengthens_braking_or_persists_in_pid(self):
        pid = Pid(.1,0.,0.,.03)
        self.assertAlmostEqual(pid.get_output(2.,3.,True,.8),-.1)
        pid.get_output(4.,3.,True,.8)
        self.assertAlmostEqual(pid.get_output(4.,3.,True),.1)


if __name__ == '__main__':
    unittest.main()
