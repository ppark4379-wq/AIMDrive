from expert.src.config.paths import JAMMING_SECTION_CONFIG
"""Jamming-only road/body gates must survive workers and recovery exits."""
import copy
import csv
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import shapely
from shapely.geometry import Polygon

from expert.src.localization.point import Point
from expert.src.vehicle_state import VehicleState
from expert.src.obstacle.object_info import ObjectInfo
from expert.src.planning.jamming.road_boundary import RoadBoundary, OWN_LINK, OPPOSING_LINK
from expert.src.planning.jamming.recorded_planner import JammingPlanner
from expert.src.planning.jamming.section import ROOT
from expert.src.planning.jamming.dynamic_recovery import recovery_result
from expert.src.planning.jamming.dynamic_planner import DynamicObstaclePlanner
from expert.src.planning.lattice.structures import LatticePlanningResult


def links():
    def link(name, first, last, x, start_node, end_node, tunnel=False):
        return dict(idx=name, points=[[x, first, 0.], [x, last, 0.]],
                    width_start=3.5, width_end=3.5, road_type='2' if tunnel else '1',
                    from_node_idx=start_node, to_node_idx=end_node)
    return [link('own_before', -180., 0., 0., 'oa', 'ob'),
            link(OWN_LINK, 0., 100., 0., 'ob', 'oc', True),
            link('own_after', 100., 200., 0., 'oc', 'od'),
            link('opposite_before', 200., 100., -3.5, 'pa', 'pb'),
            link(OPPOSING_LINK, 100., 0., -3.5, 'pb', 'pc', True),
            link('opposite_after', 0., -180., -3.5, 'pc', 'pd')]


class RoadBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.boundary = RoadBoundary.from_links(links())

    def test_body_dimensions_rear_and_front_are_not_interchangeable(self):
        body = self.boundary.body_policy
        self.assertEqual(body, dict(rear_m=.85, front_m=3.85, width_m=2.17))
        corners = self.boundary.corners_world([[0., 20.]], [math.pi/2], margin=0.)[0]
        self.assertAlmostEqual(float(corners[:, 1].min()), 19.15)
        self.assertAlmostEqual(float(corners[:, 1].max()), 23.85)
        smaller = RoadBoundary.from_links(links(), body_policy=dict(rear_m=.1, front_m=2., width_m=1.))
        self.assertEqual(smaller.policy_sha256, self.boundary.policy_sha256)

    def test_measured_wider_body_matches_model_metadata_and_public_copy_is_safe(self):
        boundary = RoadBoundary.from_links(links(), body_policy={'width_m': 2.25193})
        self.assertEqual(boundary.policy['planner_model']['width_m'], 2.25193)
        self.assertEqual(boundary.body_policy['width_m'], 2.25193)
        self.assertTrue(self.boundary.mask_world([[.5, 20.]], [math.pi/2])[0])
        self.assertFalse(boundary.mask_world([[.5, 20.]], [math.pi/2])[0])
        source_hash, policy_hash = boundary.source_sha256, boundary.policy_sha256
        exported_body = boundary.body_policy
        exported_body['width_m'] = .1
        exported_policy = boundary.policy
        exported_policy['body']['width_m'] = .1
        self.assertFalse(boundary.mask_world([[.5, 20.]], [math.pi/2])[0])
        self.assertEqual(boundary.body_policy['width_m'], 2.25193)
        self.assertEqual((boundary.source_sha256, boundary.policy_sha256), (source_hash, policy_hash))
        with self.assertRaises(AttributeError):
            boundary.body_policy = {'width_m': .1}

    def test_portal_and_source_ends_check_all_corners(self):
        self.assertTrue(self.boundary.mask_world([[0., -.1]], [math.pi/2])[0])
        self.assertFalse(self.boundary.mask_world([[0., -119.9]], [math.pi/2])[0])
        self.assertFalse(self.boundary.mask_world([[0., 168.]], [math.pi/2])[0])
        self.assertFalse(self.boundary.mask_world([[1., 20.]], [math.pi/2])[0])

    def test_planning_room_does_not_change_runtime_or_baseline_margins(self):
        points = np.array([[.4, -20.], [.4, 20.], [.4, 120.]])
        heading = np.full(3, math.pi/2)
        np.testing.assert_array_equal(self.boundary.mask_world(points, heading), [True, True, True])
        np.testing.assert_array_equal(self.boundary.planning_mask_world(points, heading), [True, False, True])
        self.assertEqual(self.boundary.policy['planning_road_margin_m'], .35)
        self.assertEqual(self.boundary.policy['runtime_road_margin_m'], .15)
        stronger = RoadBoundary.from_links(links(), planning_margin_m=.45)
        self.assertEqual(stronger.source_sha256, self.boundary.source_sha256)
        self.assertNotEqual(stronger.policy_sha256, self.boundary.policy_sha256)
        with self.assertRaisesRegex(ValueError, 'fixed at .15'):
            RoadBoundary.from_links(links(), runtime_margin_m=.14)

    def test_observed_sedan_edge_hugging_plan_is_rejected_and_runtime_stop_is_preserved(self):
        root = Path(__file__).resolve().parents[3]
        boundary = RoadBoundary.from_map_dir(root/'map_data/R_KR_PG_KATRI_2025', body_policy={
            'rear_m': .8604676685031636, 'front_m': 3.914249370522757, 'width_m': 2.366195770601937})
        planned = [[-.49214333241283015, -508.28192038560235]]
        heading = [math.radians(-140.97320788721453)]
        self.assertTrue(boundary.mask_world(planned, heading)[0])
        self.assertFalse(boundary.planning_mask_world(planned, heading)[0])
        self.assertFalse(boundary.mask_world([[-.4902738332748413, -508.20849609375]],
                                            [math.radians(-138.20196533203125)])[0])

    def test_entire_obb_rejects_hole_even_when_all_corners_are_inside(self):
        boundary = copy.copy(self.boundary)
        hole = Polygon([[-.1, 21.], [.1, 21.], [.1, 21.5], [-.1, 21.5]])
        boundary._polygon = boundary._polygon.difference(hole)
        shapely.prepare(boundary._polygon)
        corners = boundary.corners_world([[0., 20.]], [math.pi/2])[0]
        self.assertTrue(all(shapely.covers(boundary._polygon, shapely.Point(*point)) for point in corners))
        self.assertFalse(boundary.mask_world([[0., 20.]], [math.pi/2])[0])

    def test_physical_sat_detects_obstacle_behind_actor_pivot(self):
        behind = ObjectInfo(0., 18.5, 0., 2, size_x=.3, size_y=.3, heading_deg=90.)
        self.assertFalse(self.boundary.collision_mask_world([[0., 20.]], [math.pi/2], [behind])[0])

    def test_moving_prediction_includes_actual_rear_bumper(self):
        behind = ObjectInfo(0., 19., .1, 1, size_x=.3, size_y=.3, heading_deg=90.)
        ego = VehicleState(0., 20., math.pi/2, 0.)
        path = [Point(0., y) for y in np.arange(20., 30., .2)]
        self.assertIsNone(DynamicObstaclePlanner.conflict(path, ego, [behind], 0., horizon=0.))
        collision = DynamicObstaclePlanner.conflict(path, ego, [behind], 0., horizon=0.,
                                                    road_boundary=self.boundary)
        self.assertEqual(collision['time'], 0.)

    def test_missing_connected_geometry_fails_closed_and_hash_changes(self):
        with self.assertRaisesRegex(ValueError, 'Insufficient connected'):
            RoadBoundary.from_links(links()[:-1])
        changed = links()
        changed[0]['width_start'] = changed[0]['width_end'] = 3.4
        self.assertNotEqual(self.boundary.source_sha256, RoadBoundary.from_links(changed).source_sha256)

    def test_repository_duplicate_node_does_not_invent_zero_heading(self):
        root = Path(__file__).resolve().parents[3]
        boundary = RoadBoundary.from_map_dir(root/'map_data/R_KR_PG_KATRI_2025')
        points = np.loadtxt(root/'map_data/R_KR_PG_KATRI_2025/2026_molit_comp_global_path.txt')[:, :2]
        index = 3768
        next_index = index+1
        while np.linalg.norm(points[next_index]-points[index])<1e-7:
            next_index += 1
        heading = math.atan2(*(points[next_index]-points[index])[::-1])
        self.assertTrue(boundary.mask_world(points[index:index+1], [heading])[0])


class JammingBoundaryGateTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        settings = json.loads(JAMMING_SECTION_CONFIG.read_text())
        settings.update(start_enu=[0., 0., 0.], end_enu=[0., 100., 0.])
        (self.root/'section.json').write_text(json.dumps(settings))
        directory = self.root/'recordings'
        directory.mkdir()
        names = []
        for name, x in [('straight', 0.), ('left', -3.5), ('outside', 4.)]:
            filename = name+'.csv'
            with (directory/filename).open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=['x', 'y', 'yaw_rad'])
                writer.writeheader()
                for y in np.arange(0., 100.01, .2):
                    writer.writerow(dict(x=x, y=y, yaw_rad=math.pi/2))
            names.append(filename)
        (self.root/'routes.json').write_text(json.dumps(dict(routes=names)))
        self.boundary = RoadBoundary.from_links(links())
        self.planner = JammingPlanner(self.root, road_boundary=self.boundary, background=False)
        self.addCleanup(self.planner.close)
        self.route = [Point(0., y) for y in np.arange(-45., 121., .2)]

    def test_actual_search_uses_boundary_and_complete_selected_path_rechecks(self):
        ego = VehicleState(0., 0., math.pi/2, 0.)
        obstacle = ObjectInfo(0., 35., 0., 2, is_static=True, size_x=1., size_y=1., heading_deg=90.)
        result = self.planner.run(ego, self.route, [dict(object_info=obstacle)], global_path=self.route)
        self.assertGreater(result.acc_target_velocity, 0., result.debug_details)
        self.assertNotIn('outside.csv', self.planner.local.selected_names)
        checks = self.planner.validate_world_path(result.path, [obstacle], ego=ego, global_path=self.route)
        self.assertTrue(checks['passed'], checks)

    def test_shallow_snapshot_keeps_boundary_and_whole_rear_collision(self):
        worker = copy.copy(self.planner.local)
        self.assertIs(type(worker), type(self.planner.local))
        self.assertIs(worker.road_boundary, self.boundary)
        behind = ObjectInfo(0., 18.5, 0., 2, size_x=.3, size_y=.3, heading_deg=90.)
        self.assertFalse(worker.collision_free([[0., 20.]], [math.pi/2], [behind], .12))

    def test_real_pose_clearance_uses_runtime_while_search_edges_use_planning(self):
        owner = self.planner.local
        ego = VehicleState(.4, 20., math.pi/2, 0.)
        owner._setup(self.route, ego)
        self.assertTrue(owner._envelope_clear([[.4, 20.]], [math.pi/2]))
        self.assertFalse(owner._envelope_mask(np.array([[.4, 20.]]), np.array([math.pi/2]))[0])
        curve = lambda y, derivative=0: np.full_like(np.asarray(y, dtype=float), .4 if derivative==0 else 0.)
        self.assertIsNone(owner._edge(curve, 20., 25., []))
        self.assertFalse(owner._planning_road_context)
        self.assertTrue(owner._envelope_clear([[.4, 20.]], [math.pi/2]))

    def test_grid_direct_mask_and_search_exception_restore_runtime_context(self):
        owner = self.planner.local
        ego = VehicleState(.4, 20., math.pi/2, 0.)
        owner._setup(self.route, ego)
        def interrupted_search(snapshot, state, objects):
            self.assertTrue(snapshot._planning_road_context)
            self.assertFalse(snapshot._envelope_mask(np.array([[.4, 20.]]), np.array([math.pi/2]))[0])
            self.assertFalse(snapshot._envelope_clear([[.4, 20.]], [math.pi/2]))
            raise RuntimeError('interrupted fallback search')
        with patch.object(type(owner).__mro__[1], '_search', new=interrupted_search):
            with self.assertRaisesRegex(RuntimeError, 'interrupted fallback'):
                owner._search(ego, [])
        self.assertFalse(owner._planning_road_context)
        self.assertTrue(owner._envelope_clear([[.4, 20.]], [math.pi/2]))

    def test_waiting_handoff_and_final_result_cannot_bypass_road(self):
        ego = VehicleState(0., 20., math.pi/2, 0.)
        outside = [Point(1., y) for y in np.arange(20., 45., .2)]
        nominal = LatticePlanningResult(outside, mode='static_obstacle', acc_target_velocity=5.)
        guarded = self.planner.local.guard_result(nominal, ego, [])
        self.assertEqual(guarded.acc_target_velocity, 0.)
        self.assertEqual(guarded.debug_details['static_avoidance']['error'], 'road_boundary')
        waiting = self.planner.local._waiting_result(ego, outside, [], 'planning')
        self.assertEqual(waiting.acc_target_velocity, 0.)
        with patch.object(type(self.planner.local).__mro__[1], '_early_handoff_ready', return_value=True):
            self.planner.local.goal_y = 50.
            self.assertFalse(self.planner.local._early_handoff_ready(ego, outside, []))

    def test_special_recovery_does_not_approve_a_road_violation(self):
        owner = self.planner.local
        owner.path = [Point(1., y) for y in np.arange(20., 50., .2)]
        owner.curves = [lambda y: np.zeros_like(y)]
        owner.global_curve = lambda y, derivative=0: np.zeros_like(y)
        owner.envelope_y = np.array([-100., 200.])
        owner.envelope_low = np.array([-6., -6.])
        owner.envelope_high = np.array([2., 2.])
        ego = VehicleState(1., 20., math.pi/2, 0.)
        self.assertIsNone(recovery_result(owner, ego, [], -.03))

    def test_observed_owned_union_overrides_transport_and_motion_fails_stop(self):
        row = dict(raw_actor_id='7', position_xyz=[0., 35., 0.], rotation_xyz_deg=[0., 0., 90.],
                   raw_bbox_normalized=dict(center_xyz=[0., 35., 0.], width_m=1.85, length_m=4.97, heading_deg=90.),
                   center_xyz=[0., 36.3, 0.], heading_deg=90., length_m=5., width_m=1.9)
        self.planner.set_object_geometries([row])
        raw = ObjectInfo(0., 35., 0., 1, unique_id=7, size_x=1.85, size_y=4.97, heading_deg=90.)
        normalized = self.planner._obstacles([dict(object_info=raw)])[0]
        self.assertEqual((normalized.size_x, normalized.size_y), (5., 1.9))
        self.assertAlmostEqual(normalized.position.y, 36.3)
        self.assertEqual((raw.size_x, raw.size_y, raw.position.y), (1.85, 4.97, 35.))
        raw.velocity = .1
        result = self.planner.run(VehicleState(0., 0., math.pi/2, 0.), self.route,
                                  [dict(object_info=raw)], global_path=self.route)
        self.assertEqual(result.acc_target_velocity, 0.)
        self.assertEqual(result.debug_details['static_avoidance']['error'], 'owned_actor_geometry_changed')

    def test_native_ros_and_grpc_dimension_orders_normalize_to_same_owned_union(self):
        row = dict(raw_actor_id='7', position_xyz=[0., 35., 0.], rotation_xyz_deg=[0., 0., 90.],
                   raw_bbox_normalized=dict(center_xyz=[0., 35., 0.], width_m=1.85, length_m=4.97, heading_deg=90.),
                   center_xyz=[0., 35.65, 0.], heading_deg=90., length_m=6.32, width_m=1.91)
        self.planner.set_object_geometries([row])
        results = []
        for dimensions in ((4.97, 1.85), (1.85, 4.97)):
            with self.subTest(dimensions=dimensions):
                raw = ObjectInfo(0., 35., 0., 1, unique_id=7,
                    size_x=dimensions[0], size_y=dimensions[1], heading_deg=90.)
                normalized = self.planner._obstacles([dict(object_info=raw)])[0]
                self.assertIsNone(self.planner._scene_error)
                self.assertEqual((raw.size_x, raw.size_y), dimensions)
                results.append((normalized.position.x, normalized.position.y,
                                normalized.size_x, normalized.size_y, normalized.heading_deg))
        self.assertEqual(results, [(0., 35.65, 6.32, 1.91, 90.)]*2)

    def test_native_size_rejects_negative_scaled_and_union_disguised_inputs(self):
        row = dict(raw_actor_id='7', position_xyz=[0., 35., 0.], rotation_xyz_deg=[0., 0., 90.],
                   raw_bbox_normalized=dict(center_xyz=[0., 35., 0.], width_m=1.85, length_m=4.97, heading_deg=90.),
                   center_xyz=[0., 35.65, 0.], heading_deg=90., length_m=6.32, width_m=1.91)
        self.planner.set_object_geometries([row])
        for dimensions in ((-1.85, 4.97), (1.85*1.2, 4.97*1.2), (6.32, 1.91)):
            with self.subTest(dimensions=dimensions):
                self.planner._scene_error = None
                raw = ObjectInfo(0., 35., 0., 1, unique_id=7,
                    size_x=dimensions[0], size_y=dimensions[1], heading_deg=90.)
                self.planner._obstacles([dict(object_info=raw)])
                self.assertEqual(self.planner._scene_error, 'owned_actor_geometry_changed')
        result = self.planner.run(VehicleState(0., 0., math.pi/2, 0.), self.route,
                                  [dict(object_info=raw)], global_path=self.route)
        self.assertEqual(result.acc_target_velocity, 0.)

    def prius_row(self):
        return dict(raw_actor_id='7', actor_kind='vehicle', model='2019_Toyota_Prius',
            position_xyz=[0., 35., 0.], rotation_xyz_deg=[0., 0., 90.],
            raw_bbox_normalized=dict(center_xyz=[0., 36.327, 0.], width_m=2.115309953689575,
                length_m=4.5748419761657715, heading_deg=90.),
            center_xyz=[0., 36.327, 0.], heading_deg=90., length_m=4.575044, width_m=2.130007,
            vehicle_spec_contract=dict(source='bounded_GetVehicleSpec', raw_actor_id='7',
                model='2019_Toyota_Prius', size_lwh_m=[4.575, 1.76, 1.47]))

    def test_verified_prius_ros_nominal_size_keeps_wider_collider_union(self):
        row = self.prius_row()
        self.planner.set_object_geometries([row])
        # ROS nominal width differs by 35.5cm from the actual gRPC collider.
        for name, dimensions in [('2019_Toyota_Prius', (4.575, 1.76)),
                                 ('7', (2.115309953689575, 4.5748419761657715)),
                                 ('7', (4.5748419761657715, 2.115309953689575))]:
            with self.subTest(dimensions=dimensions):
                raw = ObjectInfo(0., 35., 0., 1, name=name, unique_id=7,
                                 size_x=dimensions[0], size_y=dimensions[1], heading_deg=90.)
                value = self.planner._scene_object(raw)
                self.assertIsNone(self.planner._scene_error)
                self.assertEqual((value.size_x, value.size_y), (4.575044, 2.130007))
                self.assertGreater(value.size_y, 1.76)
                self.assertAlmostEqual(value.position.y, 36.327)
                self.assertEqual((raw.size_x, raw.size_y), dimensions)
        nominal = ObjectInfo(0., 36.327, 0., 1, size_x=4.575, size_y=1.76, heading_deg=90.)
        # A vehicle at 2.75m lateral separation clears the narrower ROS box,
        # but must be blocked by the measured 2.13m collider/surface union.
        self.assertTrue(self.planner.local.collision_free([[2.75, 36.327]], [math.pi/2], [nominal], .12))
        self.assertFalse(self.planner.local.collision_free([[2.75, 36.327]], [math.pi/2], [value], .12))

    def test_spec_requires_exact_actor_model_source_kind_and_containment(self):
        changes = [('source', 'guessed'), ('raw_actor_id', '8'), ('model', 'another_vehicle'),
                   ('size_lwh_m', [4.575, float('nan'), 1.47]),
                   ('size_lwh_m', [4.575, -1.76, 1.47]),
                   ('size_lwh_m', [4.575, 3., 1.47]), ('size_lwh_m', [True, 1.76, 1.47])]
        for key, value in changes:
            with self.subTest(key=key, value=value):
                row = self.prius_row()
                row['vehicle_spec_contract'][key] = value
                with self.assertRaises(ValueError):
                    self.planner.set_object_geometries([row])
        row = self.prius_row()
        row['actor_kind'] = 'obstacle'
        with self.assertRaises(ValueError):
            self.planner.set_object_geometries([row])

    def test_spec_does_not_accept_unknown_axes_scale_or_input_rebinding(self):
        self.planner.set_object_geometries([self.prius_row()])
        for name, kind, dimensions in [('2019_Toyota_Prius', 1, (1.76, 4.575)),
                                      ('2019_Toyota_Prius', 1, (4.575*1.2, 1.76*1.2)),
                                      ('another_vehicle', 1, (4.575, 1.76)),
                                      ('2019_Toyota_Prius', 2, (4.575, 1.76))]:
            with self.subTest(name=name, kind=kind, dimensions=dimensions):
                self.planner._scene_error = None
                raw = ObjectInfo(0., 35., 0., kind, name=name, unique_id=7,
                                 size_x=dimensions[0], size_y=dimensions[1], heading_deg=90.)
                self.assertIs(self.planner._scene_object(raw), raw)
                self.assertEqual(self.planner._scene_error, 'owned_actor_geometry_changed')

    def scooter_row(self):
        return dict(raw_actor_id='7', actor_kind='obstacle', model='Scooter1',
            position_xyz=[0., 35., 0.], rotation_xyz_deg=[0., 0., 90.],
            raw_bbox_normalized=dict(center_xyz=[0., 35., 0.], width_m=1.20839,
                length_m=3.441415, heading_deg=90.),
            center_xyz=[0., 35., 0.], heading_deg=90., length_m=3.441416, width_m=1.20839,
            static_surface_contract=dict(source='sealed_lidar_static_catalog', raw_actor_id='7',
                model='Scooter1', size_lw_m=[2.26, .81], geometry_source=dict(
                    type='sealed_lidar_verified_static_footprint', catalog_sha256='a'*64)))

    def test_verified_scooter_ros_surface_size_keeps_larger_physical_union(self):
        self.planner.set_object_geometries([self.scooter_row()])
        for name, dimensions in [('Scooter1', (2.26, .81)), ('7', (1.20839, 3.441415)),
                                 ('7', (3.441415, 1.20839))]:
            with self.subTest(dimensions=dimensions):
                raw = ObjectInfo(0., 35., 0., 2, name=name, unique_id=7,
                                 size_x=dimensions[0], size_y=dimensions[1], heading_deg=90.)
                value = self.planner._scene_object(raw)
                self.assertIsNone(self.planner._scene_error)
                self.assertEqual((value.size_x, value.size_y), (3.441416, 1.20839))
                self.assertEqual((raw.size_x, raw.size_y), dimensions)
        nominal = ObjectInfo(0., 35., 0., 2, size_x=2.26, size_y=.81, heading_deg=90.)
        self.assertTrue(self.planner.local.collision_free([[2.3, 35.]], [math.pi/2], [nominal], .12))
        self.assertFalse(self.planner.local.collision_free([[2.3, 35.]], [math.pi/2], [value], .12))

    def test_static_surface_contract_rejects_unbound_source_identity_or_dimensions(self):
        for key, value in [('source', 'guessed'), ('raw_actor_id', '8'), ('model', 'another_scooter'),
                           ('size_lw_m', [2.26, float('nan')]), ('size_lw_m', [2.26, -1.]),
                           ('size_lw_m', [5., .81]), ('size_lw_m', [True, .81])]:
            with self.subTest(key=key, value=value):
                row = self.scooter_row()
                row['static_surface_contract'][key] = value
                with self.assertRaises(ValueError):
                    self.planner.set_object_geometries([row])
        for source in [dict(type='guessed', catalog_sha256='a'*64),
                       dict(type='sealed_lidar_verified_static_footprint', catalog_sha256='')]:
            row = self.scooter_row()
            row['static_surface_contract']['geometry_source'] = source
            with self.assertRaises(ValueError):
                self.planner.set_object_geometries([row])
        row = self.scooter_row()
        row['actor_kind'] = 'vehicle'
        with self.assertRaises(ValueError):
            self.planner.set_object_geometries([row])

    def test_static_surface_does_not_accept_unknown_axes_scale_or_model_rebinding(self):
        self.planner.set_object_geometries([self.scooter_row()])
        for name, kind, dimensions in [('Scooter1', 2, (.81, 2.26)), ('Scooter1', 2, (4., 2.)),
                                      ('another_scooter', 2, (2.26, .81)), ('Scooter1', 1, (2.26, .81))]:
            with self.subTest(name=name, kind=kind, dimensions=dimensions):
                self.planner._scene_error = None
                raw = ObjectInfo(0., 35., 0., kind, name=name, unique_id=7,
                                 size_x=dimensions[0], size_y=dimensions[1], heading_deg=90.)
                self.assertIs(self.planner._scene_object(raw), raw)
                self.assertEqual(self.planner._scene_error, 'owned_actor_geometry_changed')

    def native_static_row(self, model='TrafficBarrel', physical=(.7224, .6529), metadata=(.81, .81, .95)):
        row = dict(raw_actor_id='7', actor_kind='obstacle', model=model,
            position_xyz=[0., 35., 0.], rotation_xyz_deg=[0., 0., 90.],
            raw_bbox_normalized=dict(center_xyz=[0., 35., 0.], width_m=physical[1],
                length_m=physical[0], heading_deg=90.),
            center_xyz=[0., 35., 0.], heading_deg=90., length_m=physical[0], width_m=physical[1])
        frames = []
        for index in range(3):
            frames.append(dict(receipt_monotonic=10.05+.05*index, source_sequence=10+index,
                source_stamp_ns=1000000000+index*50000000, publisher_node='/native_object_bridge',
                row=dict(raw_actor_id='7', actor_kind='obstacle', name=model,
                    position_xyz=[0., 35., 0.], heading_deg=90., size_xyz=list(metadata), velocity_raw=0.)))
        row['static_surface_contract'] = dict(source='verified_native_ROS_static_metadata',
            raw_actor_id='7', model=model, size_lw_m=list(metadata[:2]), size_lwh_m=list(metadata),
            binding_monotonic=10., validated_monotonic=10.2, native_frames=frames,
            geometry_source=dict(type='sealed_lidar_verified_static_footprint', catalog_sha256='a'*64))
        return row

    def test_native_traffic_metadata_does_not_redefine_smaller_physical_union(self):
        row = self.native_static_row()
        self.planner.set_object_geometries([row])
        value = self.planner._scene_object(ObjectInfo(0., 35., 0., 2, name='TrafficBarrel',
            unique_id=7, size_x=.81, size_y=.81, heading_deg=90.))
        self.assertIsNone(self.planner._scene_error)
        self.assertEqual((value.size_x, value.size_y), (.7224, .6529))
        # The captured metadata signature is immutable even if the caller
        # modifies its copy after registration.
        row['static_surface_contract']['size_lw_m'][0] = 10.
        self.assertEqual(self.planner._object_geometries['7']['static_surface_contract']['size_lw_m'], [.81, .81])

    def test_native_electric_scooter_axes_do_not_reorient_the_physical_box(self):
        row = self.native_static_row('ElectricScooter1', physical=(.64, 1.21), metadata=(1.21, .56, 1.1))
        self.planner.set_object_geometries([row])
        value = self.planner._scene_object(ObjectInfo(0., 35., 0., 2, name='ElectricScooter1',
            unique_id=7, size_x=1.21, size_y=.56, heading_deg=90.))
        self.assertIsNone(self.planner._scene_error)
        self.assertEqual((value.size_x, value.size_y, value.heading_deg), (.64, 1.21, 90.))

    @staticmethod
    def reviewed_static3d_source(model='Bike3'):
        return dict(type='reviewed_unit3d_static_surface_and_api_collider_union',
                    validation_scope='static_unit_visible_surface_multiview_3d',
                    model_key='obstacle|' + model + '|1,1,1',
                    registry_sha256='a'*64, profiles_sha256='b'*64,
                    confirmation_sha256='c'*64, camera_review_sha256='d'*64,
                    runtime_catalog_path='/tmp/reviewed-runtime-catalog.json',
                    runtime_catalog_sha256='e'*64, measurement_sha256='f'*64)

    def test_reviewed_static3d_native_signature_keeps_larger_physical_union(self):
        row = self.native_static_row('Bike3', physical=(2., 1.2), metadata=(1.725, .58, 1.66))
        row['static_surface_contract']['geometry_source'] = self.reviewed_static3d_source()
        self.planner.set_object_geometries([row])
        value = self.planner._scene_object(ObjectInfo(0., 35., 0., 2, name='Bike3',
            unique_id=7, size_x=1.725, size_y=.58, heading_deg=90.))
        self.assertIsNone(self.planner._scene_error)
        self.assertEqual((value.size_x, value.size_y), (2., 1.2))

    def test_reviewed_static3d_requires_distinct_scope_identity_and_evidence(self):
        mutations = [('type', 'measured_bbox_only'), ('validation_scope', 'static_guess'),
                     ('model_key', 'obstacle|Bike2|1,1,1'), ('model_key', 'obstacle|Bike3|2,2,2'),
                     ('registry_sha256', ''), ('profiles_sha256', 'x'*64),
                     ('confirmation_sha256', ''), ('camera_review_sha256', ''),
                     ('runtime_catalog_sha256', ''), ('measurement_sha256', '')]
        for key, value in mutations:
            with self.subTest(key=key):
                row = self.native_static_row('Bike3', physical=(2., 1.2), metadata=(1.725, .58, 1.66))
                source = self.reviewed_static3d_source()
                source[key] = value
                row['static_surface_contract']['geometry_source'] = source
                with self.assertRaises(ValueError):
                    self.planner.set_object_geometries([row])

    def test_reviewed_static3d_still_requires_three_verified_native_frames(self):
        row = self.native_static_row('Bike3', physical=(2., 1.2), metadata=(1.725, .58, 1.66))
        row['static_surface_contract']['geometry_source'] = self.reviewed_static3d_source()
        row['static_surface_contract']['native_frames'][1]['row']['name'] = 'Bike2'
        with self.assertRaises(ValueError):
            self.planner.set_object_geometries([row])

    def test_native_static_requires_three_fresh_same_publisher_ordered_frames(self):
        mutations = [lambda c: c.update(native_frames=c['native_frames'][:2]),
                     lambda c: c.update(binding_monotonic=10.1),
                     lambda c: c.update(validated_monotonic=11.),
                     lambda c: c['native_frames'][1].update(publisher_node='/unrelated'),
                     lambda c: c['native_frames'][1].update(publisher_node=''),
                     lambda c: c['native_frames'][1].update(source_sequence=10),
                     lambda c: c['native_frames'][1].update(source_stamp_ns=1),
                     lambda c: c['native_frames'][1].update(receipt_monotonic=10.05)]
        for index, change in enumerate(mutations):
            with self.subTest(index=index):
                row = self.native_static_row()
                change(row['static_surface_contract'])
                with self.assertRaises(ValueError):
                    self.planner.set_object_geometries([row])

    def test_native_static_rejects_rebinding_pose_motion_and_size_drift(self):
        changes = [dict(raw_actor_id='8'), dict(actor_kind='vehicle'), dict(name='another'),
                   dict(position_xyz=[.2, 35., 0.]), dict(heading_deg=93.),
                   dict(velocity_raw=.06), dict(size_xyz=[.83, .81, .95]),
                   dict(size_xyz=[.81, float('nan'), .95])]
        for change in changes:
            with self.subTest(change=change):
                row = self.native_static_row()
                row['static_surface_contract']['native_frames'][1]['row'].update(change)
                with self.assertRaises(ValueError):
                    self.planner.set_object_geometries([row])

    def test_native_static_zero_stamps_are_allowed_without_weakening_nonzero_order(self):
        row = self.native_static_row()
        row['static_surface_contract']['native_frames'][1]['source_stamp_ns'] = 0
        self.planner.set_object_geometries([row])
        self.assertEqual(self.planner._object_geometries['7']['width_m'], .6529)
        other = self.native_static_row()
        other['static_surface_contract']['native_frames'][1]['source_stamp_ns'] = 0
        other['static_surface_contract']['native_frames'][2]['source_stamp_ns'] = 999999999
        with self.assertRaises(ValueError):
            self.planner.set_object_geometries([other])

    def test_scene_geometry_cannot_change_after_planning_and_missing_owned_actor_stops(self):
        row = dict(raw_actor_id='7', position_xyz=[0., 35., 0.], rotation_xyz_deg=[0., 0., 90.],
                   raw_bbox_normalized=dict(center_xyz=[0., 35., 0.], heading_deg=90., length_m=1., width_m=1.),
                   center_xyz=[0., 35., 0.], heading_deg=90., length_m=1., width_m=1.)
        self.planner.set_object_geometries([row])
        result = self.planner.run(VehicleState(0., 0., math.pi/2, 0.), self.route, [], global_path=self.route)
        # An absent scene is an explicit stop even when normal planning has no
        # obstacles to activate on.
        self.assertIsNotNone(result)
        self.assertEqual(result.acc_target_velocity, 0.)
        with self.assertRaisesRegex(ValueError, 'before planning'):
            self.planner.set_road_boundary(self.boundary)


if __name__ == '__main__':
    unittest.main()
