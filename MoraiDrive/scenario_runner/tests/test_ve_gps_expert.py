"""Direct GPS+VE integration keeps the original planner's object contract."""
from pathlib import Path
import json
import struct
import math
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'MoraiDrive/scenario_runner/legacy_expert'))
from run_ve_gps_expert import planner_inputs, configure_udp_steering, place_route_start, encode_planner_control
from gt_udp_protocol import pack_control


class GpsExpertInputTests(unittest.TestCase):
    def test_planner_radians_are_converted_to_normalized_wire_and_pedals_preserved(self):
        for degrees, expected in ((0.,0.), (10.,.25), (-20.,-.5), (40.,1.), (80.,1.)):
            control = SimpleNamespace(accel=.3,brake=0.,front_steer=math.radians(degrees),rear_steer=0.)
            wire = encode_planner_control(control,math.radians(40.))
            self.assertEqual(len(wire),55)
            self.assertAlmostEqual(struct.unpack_from('<f',wire,49)[0],expected)
            self.assertAlmostEqual(struct.unpack_from('<f',wire,41)[0],.3)
            self.assertEqual(struct.unpack_from('<f',wire,45)[0],0.)
        with self.assertRaises(ValueError):
            encode_planner_control(control,0.)
    def test_start_placement_sends_only_bounded_setup_writes(self):
        sys.path.insert(0, str(ROOT/'MoraiDrive/data_collection/third_party/grpc_inha_univ/src/proto'))
        from morai.common.object_info_pb2 import ObjectInfo
        from morai.common.object_identifier_pb2 import ObjectIdentifier
        from morai.common.enum_pb2 import STATUS_CODE_SUCCESS
        stub = SimpleNamespace(**{name: Mock(return_value=SimpleNamespace(status=STATUS_CODE_SUCCESS))
            for name in ('SetVehicleEgoCruise','SetVelocity','SetTransform')})
        actor = ObjectInfo(id=ObjectIdentifier(value='Ego'),object_type=1)
        place_route_start(stub,actor,(-131.,-428.,28.,61.))
        self.assertFalse(stub.SetVehicleEgoCruise.call_args.args[0].cruise_on)
        self.assertEqual(stub.SetVelocity.call_count,2)
        for call in stub.SetVelocity.call_args_list:
            self.assertEqual(call.args[0].velocity,0.)
            self.assertEqual(call.kwargs,dict(timeout=1.))
        request = stub.SetTransform.call_args.args[0]
        self.assertEqual((request.transform.location.x,request.transform.location.y,
                          request.transform.location.z,request.transform.rotation.z),(-131.,-428.,28.,61.))
        stub.SetTransform.reset_mock()
        stub.SetVelocity.return_value = SimpleNamespace(status=-1,description='denied')
        with self.assertRaisesRegex(RuntimeError,'velocity_zero'):
            place_route_start(stub,actor,(-131.,-428.,28.,61.))
        stub.SetTransform.assert_not_called()

    def test_left_and_right_path_errors_produce_corrective_udp_steering(self):
        from AIM2026.Expert.src.control.path_controller import PathController
        from AIM2026.Expert.src.vehicle_state import VehicleState
        from AIM2026.Expert.src.localization.point import Point
        config = json.loads((ROOT/'AIM2026/Expert/src/config/config.json').read_text())
        for profile, physical_left_sign in (('normal', -1.), ('competition', 1.)):
            configure_udp_steering(config, profile)
            for y, direction in ((1., -1), (-1., 1)):
                with self.subTest(profile=profile, y=y):
                    controller = PathController(config)
                    result = SimpleNamespace(mode='normal', active=False, reason='', debug_details={}, highway_state='')
                    control = controller.control_path([Point(x, 0.) for x in range(30)],
                        VehicleState(0.,y,0.,3.), 3., result, sync_mode=False, control_dt=1./30)
                    # PurePursuit -> per-build transport -> physical direction.
                    wire = encode_planner_control(control,controller.max_steering)
                    self.assertGreater(direction*physical_left_sign*struct.unpack_from('<f',wire,49)[0], 0)
        configure_udp_steering(config)
        self.assertEqual(config['common']['realtime_steering_sign'], -1.)
        with self.assertRaises(KeyError):
            configure_udp_steering(config, 'unverified')

    def test_grid_heading_speed_and_local_motion_tracks_are_preserved(self):
        packet = dict(ego_position_source='GPS', ego_status_source='UDP_status_speed_IMU_heading',
            ego_pose=dict(x=-60., y=-130., yaw_raw_deg=90., yaw_grid_rad=1.58, speed_mps=10.),
            boxes=[dict(center_world=[-58.,-110.], planning_speed_mps=0., motion_state='stationary',
                length_m=3., width_m=2., planning_heading_deg=91., track_id=42),
                dict(center_world=[-60.,-105.], planning_speed_mps=4., motion_state='moving',
                length_m=4., width_m=2., planning_heading_deg=89., track_id=43)])
        vehicle, objects = planner_inputs(packet)
        self.assertEqual((vehicle.position.x,vehicle.position.y,vehicle.yaw,vehicle.velocity),
                         (-60.,-130.,1.58,10.))
        self.assertEqual([o.velocity for o in objects], [0.,4.])
        self.assertEqual([o.is_static for o in objects], [True,False])
        self.assertEqual([o.unique_id for o in objects], [42,43])
        self.assertEqual([o.type for o in objects], [-1,-1])
        latest_pose = dict(ego_position_source='GPS', ego_status_source='UDP_status_speed_IMU_heading',
            ego_pose=dict(x=-61., y=-120., yaw_raw_deg=80., yaw_grid_rad=1.4, speed_mps=12.))
        vehicle, objects = planner_inputs(packet, latest_pose)
        self.assertEqual((vehicle.position.x,vehicle.position.y,vehicle.yaw,vehicle.velocity),
                         (-61.,-120.,1.4,12.))
        # Changing control time must never re-transform the old world boxes.
        self.assertEqual((objects[0].position.x,objects[0].position.y),(-58.,-110.))
        self.assertEqual(packet['ego_pose']['x'],-60.)
        latest_pose['ego_position_source'] = '/Ego_topic'
        with self.assertRaises(ValueError):
            planner_inputs(packet, latest_pose)
        packet['ego_position_source'] = '/Ego_topic'
        with self.assertRaises(ValueError):
            planner_inputs(packet)


if __name__ == '__main__':
    unittest.main()
