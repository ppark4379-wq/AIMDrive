import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

from AIM2026.Expert.tools.ve_gt_comparison import (
    analyze, gt_motion_check, in_roi, match_centers, nearest, residual_jitter,
    roi, score_frame, serialize_gt, summarize, timeline)


def vector(x=0., y=0., z=0.):
    return NS(x=x, y=y, z=z)


def message():
    obj = NS(unique_id=7, type=1, name='car', position=vector(10., 0.),
             velocity=vector(10.8, 14.4), size=vector(4.4, 1.8, 1.5), heading=0.)
    return NS(header=NS(stamp=NS(to_nsec=lambda: 1_000_000_000), frame_id='ObjectInfo'),
              npc_list=[obj], obstacle_list=[], pedestrian_list=[], num_of_npcs=1,
              num_of_obstacle=0, num_of_pedestrian=0)


def box(x=10., y=0.):
    return dict(track_id='npc:7', kind='npc', name='car', center_world=[x,y],
                heading_deg=0., length_m=4.4, width_m=1.8, speed_mps=5.)


def prediction(x=10., y=0.):
    return dict(box(x,y), track_id=1, planning_speed_mps=5., motion_state='moving')


class ComparisonTests(unittest.TestCase):
    def test_live_native_size_axes_and_planar_kmh_conversion(self):
        row = serialize_gt(message(), 123)
        b = row['boxes'][0]
        self.assertEqual((b['length_m'], b['width_m'], b['speed_mps']), (4.4, 1.8, 5.))
        self.assertEqual(b['velocity_raw_kmh'], [10.8, 14.4, 0.])
        self.assertEqual(b['raw_size_xyz_m'], [4.4, 1.8, 1.5])

    def test_count_invalid_and_duplicate_gt_are_flagged(self):
        msg = message()
        msg.npc_list *= 2
        self.assertEqual(len(serialize_gt(msg, 1)['errors']), 2)
        msg = message()
        msg.npc_list[0].size.x = float('nan')
        row = serialize_gt(msg, 1)
        self.assertEqual(row['boxes'], [])
        self.assertTrue(row['errors'])

    def test_exact_identical_bbox_and_two_speed_outputs(self):
        p = prediction()
        p.update(speed_mps=4., planning_speed_mps=6., heading_deg=180.)
        result = score_frame([box()], [p], dict(x=0.,y=0.,yaw_deg=0.))
        m = result['matched'][0]
        self.assertAlmostEqual(m['iou'], 1.)
        self.assertEqual(m['heading_error_deg_mod180'], 0.)
        self.assertEqual(m['raw_speed_error_mps'], 1.)
        self.assertEqual(m['raw_speed_bias_mps'], -1.)
        self.assertEqual(m['planning_speed_error_mps'], 1.)

    def test_miss_and_false_prediction_are_not_hidden_by_regression(self):
        score = score_frame([box(),box(20.)], [prediction(),prediction(28.)], dict(yaw_deg=0.))
        result = summarize([dict(score=score)])
        self.assertEqual((result['missed_gt'], result['unmatched_ve']), (1,1))
        self.assertEqual(result['center_recall_2m'], .5)
        self.assertEqual(result['metrics']['center_error_m']['count'], 1)

    def test_outside_gate_cannot_steal_valid_assignment(self):
        self.assertEqual(len(match_centers([box(0.),box(3.)],
                                          [prediction(1.),prediction(5.)])),2)

    def test_gt_motion_checks_kmh_against_displacement(self):
        rows = []
        for i in range(20):
            m = message()
            m.npc_list[0].position.x = .4*i
            m.npc_list[0].velocity = vector(14.4)
            r = serialize_gt(m,1_000_000_000+i*100_000_000)
            r['source_stamp_ns'] = r['received_ns']
            rows.append(r)
        v = gt_motion_check(rows)['npc:7']
        self.assertAlmostEqual(v['displacement_to_raw_ratio']['mean'],1/3.6)
        self.assertLess(v['kmh_conversion_error_mps']['max'],1e-10)

    def test_residual_jitter_subtracts_real_motion(self):
        frames = []
        for i in range(5):
            g,p = box(10+i),prediction(10.3+i)
            frames.append(dict(target_ns=i*100_000_000,gt=[g],ve=[p],
                score=score_frame([g],[p],dict(yaw_deg=0.))))
        result = residual_jitter(frames)
        self.assertEqual(result['center_residual_step_m']['count'],4)
        self.assertLess(result['center_residual_step_m']['max'],1e-10)
        self.assertEqual(result['ve_id_switches'],0)

    def test_roi_includes_partial_and_uses_ego_rotation(self):
        area = roi(dict(x=0.,y=0.,yaw_deg=90.))
        self.assertTrue(in_roi(box(0.,30.), area))
        self.assertFalse(in_roi(box(0.,33.), area))

    def test_pairing_rejects_gap_and_stale_repeated_header(self):
        rows = [serialize_gt(message(), 1_000_000_000 + i*60_000_000) for i in range(4)]
        times = timeline(rows)
        self.assertIs(nearest(rows,times,1_010_000_000),rows[0])
        self.assertIsNone(nearest(rows,times,1_120_000_000))
        self.assertIsNone(nearest(rows,times,1_500_000_000))

    def test_comparison_filters_to_actual_expert_callbacks_and_saves_artifacts(self):
        from data.src.loaders.morai_map_geometry import validate_map_basis
        from inference.realtime_ve_objects import MAP_BASIS
        g = serialize_gt(message(),1_000_000_000)
        p = prediction()
        p['heading_deg'] = validate_map_basis(MAP_BASIS).convergence_degrees([10.,0.,0.])
        packet = dict(sequence=1,checkpoint_sha256='hash',boxes=[p],
            ego_pose=dict(x=0.,y=0.,yaw_raw_deg=0.),
            audit=dict(selected_samples=dict(lidar=dict(received_ns=1_000_000_000))),
            delivery_audit=dict(ros_stamp_ns=123))
        other = copy.deepcopy(packet)
        other['delivery_audit']['ros_stamp_ns'] = 456
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            for name, rows in [('gt_objects',[g]),('adapter',[packet,other]),
                ('expert_input',[dict(ros_stamp_ns=123,received_ns=1_030_000_000)])]:
                (output/(name+'.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in rows))
            result = analyze(output)
            self.assertEqual(result['expert_packets'],1)
            self.assertAlmostEqual(result['summaries']['input_receipt']['metrics']['iou']['mean'],1.)
            self.assertEqual(result['summaries']['expert_receipt']['matched'],1)
            self.assertFalse(result['gt_used_for_driving'])
            self.assertTrue((output/'gt_comparison_review.html').is_file())


if __name__ == '__main__':
    unittest.main()
