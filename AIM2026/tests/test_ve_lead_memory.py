import copy
import json
import unittest
from unittest.mock import Mock

from AIM2026.Inference.vision.ve_lead_memory import VeLeadMemory
from AIM2026.UDP.Detection import Box


LANES = {'ego': [(0, 0), (200, 0)], 'adjacent': [(0, 3.5), (200, 3.5)]}


def box(**changes):
    return dict(center_world=[20., 0.], heading_deg=0., planning_heading_deg=0.,
                width_m=2.5, length_m=10., speed_mps=15., track_id=1,
                motion_state='moving', motion_direction_observed=True,
                planning_speed_source='position_history', planning_speed_mps=5.,
                observed_speed_mps=5., **changes)


def packet(t, boxes, **changes):
    return dict(generation='a', sensor_received_ns=round(t*1e9),
                object_received_ns=round(t*1e9), boxes=boxes,
                ego_pose=dict(x=0., y=0., yaw_grid_rad=0.), **changes)


class LeadMemoryTests(unittest.TestCase):
    def test_gap_uses_observed_speed_then_expires_from_real_observation(self):
        memory = VeLeadMemory(LANES)
        original = box()
        self.assertIs(memory.apply(packet(10, [original]))['boxes'][0], original)
        for t in (10.2, 10.6, 10.99):
            held = memory.apply(packet(t, []))['boxes'][0]
            self.assertAlmostEqual(held['center_world'][0], 20+5*(t-10))
            self.assertTrue(held['detection_held'])
            self.assertEqual(held['detection_last_observed_ns'], 10000000000)
        self.assertEqual(memory.apply(packet(11, []))['boxes'], [])
        self.assertEqual(original['center_world'], [20, 0])

    def test_adjacent_rear_small_and_lidar_boxes_do_not_seed(self):
        for changes in [dict(center_world=[20., 3.5]), dict(center_world=[-3., 0.]),
                        dict(length_m=4.), dict(detection_origin='lidar_supplement')]:
            memory = VeLeadMemory(LANES)
            b = box(); b.update(changes)
            memory.apply(packet(1, [b]))
            self.assertEqual(memory.apply(packet(1.2, []))['boxes'], [])

    def test_unknown_speed_is_not_extrapolated_with_ve_speed(self):
        memory = VeLeadMemory(LANES)
        b = box(); b.update(planning_speed_source='ve_provisional', motion_state='uncertain')
        memory.apply(packet(1, [b]))
        held = memory.apply(packet(1.5, []))['boxes'][0]
        self.assertEqual(held['center_world'], b['center_world'])
        self.assertIsNone(held['planning_speed_mps'])

    def test_new_id_reappearance_or_supplement_replaces_hold_without_duplicate(self):
        for origin in ('ve', 'lidar_supplement'):
            memory = VeLeadMemory(LANES)
            memory.apply(packet(1, [box()]))
            real = box(); real.update(track_id=2, center_world=[21., 0.], detection_origin=origin)
            result = memory.apply(packet(1.2, [real]))
            self.assertEqual(result['boxes'], [real])
            self.assertEqual(result['ve_lead_held_count'], 0)
            if origin == 'lidar_supplement':
                self.assertEqual(memory.apply(packet(1.3, []))['boxes'], [])

    def test_lane_change_epoch_and_clock_reset_remove_hold(self):
        for case in ('lane', 'epoch', 'clock'):
            memory = VeLeadMemory(LANES); memory.apply(packet(1, [box()]))
            p = packet(1.2, [])
            if case == 'lane': p['ego_pose']['y'] = 3.5
            if case == 'epoch': p['generation'] = 'b'
            if case == 'clock': p['object_received_ns'] = 500000000
            self.assertEqual(memory.apply(p)['boxes'], [])

    def test_wire_origin_survives_decode(self):
        b = Box(20., 0., 0., 2.5, 10., 5., detection_origin='lidar_supplement')
        self.assertEqual(Box.from_wire(b.to_wire()).detection_origin, 'lidar_supplement')

    def test_inputs_does_not_feed_held_output_to_tracker(self):
        from AIM2026.Expert.runtime import Inputs
        from unittest.mock import patch
        tracker = Mock()
        tracker.update.side_effect = lambda row: copy.deepcopy(row)
        bridge = Inputs('a', 'digest', tracker, dedup=False, ve_lead_memory=VeLeadMemory(LANES))
        with patch('AIM2026.Expert.runtime.decode'):
            for seq, t, boxes in [(1, 1., [box()]), (2, 1.2, []), (3, 1.4, []), (4, 2., [])]:
                p = packet(t, boxes); p.update(sequence=seq, checkpoint_sha256='digest')
                bridge.accept(json.dumps(p))
                self.assertEqual(len(bridge.packet['boxes']), 0 if seq == 4 else 1)
        self.assertEqual([len(c.args[0]['boxes']) for c in tracker.update.call_args_list], [1, 0, 0, 0])


if __name__ == '__main__':
    unittest.main()
