"""Wire-contract checks for class-unknown perception; no ROS/model needed."""
import importlib.util
import json
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

MODULE_PATH = Path(__file__).resolve().parents[1] / 'inference/realtime_ve_objects.py'
spec = importlib.util.spec_from_file_location('live_ve_objects', MODULE_PATH)
live = importlib.util.module_from_spec(spec)
spec.loader.exec_module(live)


def obj():
    return NS(position=NS(), size=NS(), velocity=NS())


def message():
    return NS(header=NS(), npc_list=[], obstacle_list=[], pedestrian_list=[])


class LiveObjectsTests(unittest.TestCase):
    def setUp(self):
        self.meta = {'ready': True, 'stamp_ns': 22000000033,
                     'sensor_received_monotonic_ns': 1000000000}
        self.config = {'checkpoint': live.CHECKPOINTS['E3'], 'map_basis': live.MAP_BASIS}
        self.box = dict(center_world=[-59., -130.], width_m=.35, length_m=1.1,
                        heading_deg=82., speed_mps=2.5)

    def test_wire_units_order_and_unknown_identity(self):
        result = live.make_object_message([self.box], self.meta, 17, message, obj,
                                         lambda sec, nsec: (sec, nsec))
        self.assertEqual(result.header.stamp, (22, 33))
        self.assertEqual(result.header.seq, 17)
        self.assertEqual(result.header.frame_id, 'map')
        self.assertEqual((result.num_of_npcs, result.num_of_obstacle, result.num_of_pedestrian), (1, 0, 0))
        item = result.npc_list[0]
        self.assertEqual((item.size.x, item.size.y, item.size.z), (1.1, .35, 0.))
        self.assertEqual((item.type, item.unique_id, item.name), (-1, 1, 've_unknown'))
        self.assertEqual((item.velocity.x, item.velocity.y), (9., 0.))
        self.assertEqual(result.obstacle_list, [])

    def test_invalid_footprint_does_not_create_valid_empty_message(self):
        for update in ({'width_m': 0.}, {'heading_deg': float('nan')},
                       {'center_world': [float('inf'), 1.]}):
            with self.subTest(update=update), self.assertRaises(ValueError):
                live.make_object_message([dict(self.box, **update)], self.meta,
                    1, message, obj, lambda *args: args)

    def test_valid_empty_has_matching_stamp_and_zero_counts(self):
        result = live.make_object_message([], self.meta, 4, message, obj, lambda *a: a)
        self.assertEqual((result.header.seq, result.header.stamp), (4, (22, 33)))
        self.assertEqual((result.num_of_npcs, result.num_of_obstacle, result.num_of_pedestrian), (0, 0, 0))

    def test_unknown_speed_is_not_world_motion_or_static_class(self):
        for speed in (None, float('nan'), -1.):
            result = live.make_object_message([dict(self.box, speed_mps=speed)],
                self.meta, 1, message, obj, lambda *a: a)
            self.assertEqual(result.npc_list[0].velocity.x, 0.)
            self.assertEqual(result.npc_list[0].type, -1)

    def test_frame_local_ids_do_not_leak_gt_identity(self):
        altered = dict(self.box, track_id='gt_actor_123', kind='pedestrian', is_static=True)
        result = live.make_object_message([altered, self.box], self.meta, 1, message, obj, lambda *a: a)
        self.assertEqual([item.unique_id for item in result.npc_list], [1, 2])
        self.assertEqual([item.type for item in result.npc_list], [-1, -1])
        self.assertEqual([item.name for item in result.npc_list], ['ve_unknown', 've_unknown'])

    def test_source_stale_or_future_after_inference_is_invalid(self):
        self.assertEqual(live.can_publish(self.meta, 1350000000, .35), (True, None))
        self.assertFalse(live.can_publish(self.meta, 1350000001, .35)[0])
        self.assertFalse(live.can_publish(self.meta, 999999999, .35)[0])
        self.assertFalse(live.can_publish(dict(self.meta, stamp_ns=0), 1100000000, .35)[0])

    def test_failure_health_never_certifies_empty_objects(self):
        report = live.make_health(self.config, 'run-id', 20, self.meta,
                                 ready=False, invalid_reason='lidar_stale', now_ns=1300000000)
        self.assertFalse(report['ready'])
        self.assertEqual(report['invalid_reason'], 'lidar_stale')
        self.assertEqual(report['sequence'], 20)
        self.assertEqual(report['stamp_ns'], self.meta['stamp_ns'])
        self.assertEqual(report['schema'], 'morai_ve_perception_health_v1')
        self.assertFalse(report['persistent_tracks_available'])
        json.dumps(report, allow_nan=False)

    def test_target_topic_control_or_gt_is_rejected(self):
        doc = dict(schema='morai_live_ve_objects_v1', checkpoints=live.CHECKPOINTS,
                   map_basis=live.MAP_BASIS, object_topic='/Object_topic',
                   health_topic='/ve/perception_health')
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'config.json'
            path.write_text(json.dumps(doc))
            with self.assertRaisesRegex(ValueError, 'separate VE'):
                live.load_config(path)

    def test_lidar_frame_id_is_exact_and_explicit_or_known_default(self):
        doc = dict(schema='morai_live_ve_objects_v1', checkpoints=live.CHECKPOINTS,
                   map_basis=live.MAP_BASIS, object_topic='/ve/Object_topic',
                   health_topic='/ve/perception_health')
        with tempfile.TemporaryDirectory() as temp, patch.object(Path, 'is_file', return_value=True):
            path = Path(temp) / 'config.json'
            path.write_text(json.dumps(doc))
            self.assertEqual(live.load_config(path)['expected_lidar_frame_id'], 'Lidar3D-5')
            path.write_text(json.dumps(dict(doc, expected_lidar_frame_id='Lidar3D-6')))
            self.assertEqual(live.load_config(path)['expected_lidar_frame_id'], 'Lidar3D-6')
            for invalid in ('', ' Lidar3D-5', None):
                path.write_text(json.dumps(dict(doc, expected_lidar_frame_id=invalid)))
                with self.assertRaisesRegex(ValueError, 'exact ROS'):
                    live.load_config(path)

    def test_metadata_check_does_not_load_ros_model_or_sensor(self):
        config = dict(self.config, checkpoint_label='E3', object_topic='/ve/Object_topic',
                      health_topic='/ve/perception_health')
        with patch.object(live, 'load_config', return_value=config), \
             patch.object(Path, 'stat', return_value=NS(st_size=123)), \
             patch.object(live, 'run') as run, patch('builtins.print'):
            self.assertEqual(live.main(['--config', '/unused', '--check']), 0)
        run.assert_not_called()

    def test_actual_ros_serialization_sequence_matches_health_and_gate(self):
        repo = MODULE_PATH.parents[2]
        internal = repo / 'MoraiDrive/data_collection/internal_sensor_record'
        sys.path.insert(0, str(internal))
        from runtime_paths import CATKIN_ROOT, ROS_ROOT
        sys.path.extend([str(CATKIN_ROOT / 'devel/lib/python3/dist-packages'),
                         str(ROS_ROOT / 'lib/python3/dist-packages')])
        try:
            from rospy.msg import serialize_message
            from genpy import Time
            from morai_msgs.msg import ObjectStatusList, ObjectStatus
        except ImportError:
            self.skipTest('ROS-generated messages unavailable; pure wire-contract tests still apply')
        sys.path.insert(0, str(repo / 'MoraiDrive/expert/src'))
        from expert.src.planning.ve_perception_planning import VeInputGate
        from expert.src.obstacle.object_info import ObjectInfo
        m = live.make_object_message([self.box], self.meta, 99,
                                    ObjectStatusList, ObjectStatus, Time)
        stream = io.BytesIO()
        publisher = NS(publish=lambda message: serialize_message(stream, 1, message))
        actual_sequence = live.publish_object_message(publisher, m)
        self.assertEqual(actual_sequence, 1)
        wire = ObjectStatusList().deserialize(stream.getvalue()[4:])
        self.assertEqual(wire.header.seq, actual_sequence)
        health = live.make_health(self.config, 'serialization-test', actual_sequence,
                                 self.meta, ready=True, object_count=1, now_ns=1100000000)
        item = wire.npc_list[0]
        converted = ObjectInfo(item.position.x, item.position.y, 0., -1,
            name=item.name, is_static=False, size_x=item.size.x, size_y=item.size.y,
            heading_deg=item.heading, unique_id=None)
        gate = VeInputGate(dict(expected_checkpoint_sha256=self.config['checkpoint']['sha256'],
                                expected_map_basis=self.config['map_basis'], max_age_sec=.35))
        gate.accept_objects([converted], wire.header.seq, self.meta['stamp_ns'], 1.1, wire.header.frame_id)
        gate.accept_health(health, 1.1)
        values, valid, _ = gate.select(1.15, self.meta['stamp_ns'] + 50000000, 1.15)
        self.assertTrue(valid['ready'])
        self.assertTrue(valid['input_validated'])
        self.assertEqual(len(values), 1)


if __name__ == '__main__':
    unittest.main()
