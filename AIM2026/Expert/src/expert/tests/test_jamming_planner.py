from expert.src.config.paths import JAMMING_SECTION_CONFIG
"""ENU recordings must drive the same footprint-checked planner after rotation."""
import csv
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from expert.src.localization.point import Point
from expert.src.vehicle_state import VehicleState
from expert.src.obstacle.object_info import ObjectInfo
from expert.src.planning.jamming.section import JammingSection, ROOT, next_route_name, register_route
from expert.src.planning.jamming.recorded_planner import JammingPlanner
from expert.src.planning.jamming.prepare_routes import prepare_library, PREPARED_DIRECTORY
from expert.src.planning.lattice.lattice_planner import LatticePlanner
from expert.src.planning.section_planning import LatticePlanningProcess
from expert.src.planning.static_obstacle.recorded_planner import RecordedStaticPlanner

SPEC = importlib.util.spec_from_file_location("jamming_recorder", Path(__file__).resolve().parents[3] / "record_static_route.py")
recorder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(recorder)


class JammingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        (self.root / "section.json").write_bytes(JAMMING_SECTION_CONFIG.read_bytes())
        (self.root / "routes.json").write_text('{"routes": []}', encoding="utf-8")
        self.section = JammingSection(self.root)
        self.section.output_dir.mkdir()

    def record(self, name="jamming_01", offset=0., end=None, start_side=0., start_forward=0., end_side=0.):
        section = self.section
        finish = section.length if end is None else end
        y = np.linspace(start_forward, finish, int((finish-start_forward)/.2)+1)
        smooth = lambda v: (lambda u: u*u*(3.-2.*u))(np.clip(v, 0., 1.))
        x = offset*smooth(y/20.)*smooth((section.length-y)/20.)
        x += start_side*(1.-smooth((y-start_forward)/50.))
        x += end_side*smooth((y-(finish-50.))/50.)
        slope = np.gradient(x, y)
        path = section.output_dir / (name+".csv")
        with path.open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=recorder.FIELDS)
            writer.writeheader()
            for i, (px, py, derivative) in enumerate(zip(x, y, slope)):
                wx, wy = section.to_world((px, py))
                yaw = math.atan2(1., derivative)-section.rotation
                writer.writerow(dict(x=wx, y=wy, z=30., yaw_rad=yaw,
                    heading_deg=math.degrees(yaw), s_m=py, speed_mps=2., source_stamp_s=i*.1))
        return path

    def planner(self, populated=True, background=False):
        if populated:
            for name, offset in (("straight", 0.), ("right", 4.)):
                register_route(self.record(name, offset), self.section)
        planner = JammingPlanner(self.root, background=background)
        self.addCleanup(planner.close)
        return planner

    def ego(self, x=0., y=0., yaw=math.pi/2):
        return VehicleState(*self.section.to_world((x, y)), yaw-self.section.rotation, 0.)

    def route(self):
        return [Point(*self.section.to_world((0., y))) for y in
                np.arange(-24., self.section.length+20., .2)]

    def obstacle(self, x=0., y=35., size=1.):
        return ObjectInfo(*self.section.to_world((x, y)), 0., 2, is_static=True,
            size_x=size, size_y=size, heading_deg=math.degrees(self.section.angle))

    def test_rotation_preserves_endpoints_distances_and_heading(self):
        section = self.section
        np.testing.assert_allclose(section.to_local(section.start), (0., 0.), atol=1e-10)
        np.testing.assert_allclose(section.to_local(section.end), (0., section.length), atol=1e-10)
        self.assertAlmostEqual(section.yaw_to_local(section.angle), math.pi/2)
        for point in ((-4., 0.), (3., 43.), (-2., 130.)):
            np.testing.assert_allclose(section.to_local(section.to_world(point)), point, atol=1e-10)
        self.assertAlmostEqual(math.dist(section.to_world((1., 2.)), section.to_world((4., 6.))), 5.)

    def test_requested_check_command_records_and_check_only_does_not(self):
        with patch.object(recorder, "JammingSection", return_value=self.section):
            args = recorder.parse_arguments(["--check"])
            self.assertFalse(args.check_only)
            self.assertEqual(args.output_dir, self.section.output_dir)
            self.assertEqual((args.start_x, args.start_y), (31.42, -486.50))
            self.assertEqual(args.name, "jamming_01")
            check = recorder.parse_arguments(["--check-only"])
            self.assertTrue(check.check_only)
            self.assertIsNone(check.name)
        static = recorder.parse_arguments(["--section", "static", "--check"])
        self.assertTrue(static.check_only)
        self.assertEqual((static.start_x, static.start_y), (-59.65, -159.06))

    def test_complete_recording_registers_without_overwriting(self):
        path = self.record()
        before = path.read_bytes()
        args = SimpleNamespace(jamming_section=self.section)
        self.assertTrue(recorder.finish_recording(path, args))
        self.assertEqual(json.loads((self.root/"routes.json").read_text())["routes"], [path.name])
        self.assertEqual(next_route_name(self.section.output_dir), "jamming_02")
        self.assertTrue(json.loads(path.with_suffix(".status.json").read_text())["ready_for_planner"])
        self.assertEqual(path.read_bytes(), before)
        register_route(path, self.section)
        self.assertEqual(len(json.loads((self.root/"routes.json").read_text())["routes"]), 1)

    def test_partial_or_failed_recording_is_never_registered(self):
        args = SimpleNamespace(jamming_section=self.section)
        path = self.record(end=30.)
        self.assertFalse(recorder.finish_recording(path, args))
        complete = self.record("jamming_02")
        self.assertFalse(recorder.finish_recording(complete, args, "Telemetry gap"))
        self.assertEqual(json.loads((self.root/"routes.json").read_text())["routes"], [])

    def test_lateral_starts_pass_recorder_registration_and_planner_loading(self):
        with patch.object(recorder, "JammingSection", return_value=self.section):
            args = recorder.parse_arguments(["--check"])
        for name, side in (("left_start", -8.), ("right_start", 8.)):
            x, y = self.section.to_world((side, 4.))
            self.assertGreater(math.hypot(x-self.section.start[0], y-self.section.start[1]), 5.)
            recorder.validate_recording_start(dict(x=x, y=y), args)
            path = self.record(name, start_side=side, start_forward=4.)
            self.assertTrue(recorder.finish_recording(path, args))
        planner = self.planner(populated=False)
        self.assertIsNone(planner.local.error)
        self.assertEqual(len(planner.local.records), 2)

    def test_start_line_rejects_longitudinal_and_lateral_outliers_consistently(self):
        with patch.object(recorder, "JammingSection", return_value=self.section):
            args = recorder.parse_arguments([])
        for index, (side, forward) in enumerate(((0., 5.1), (0., -20.1), (12.1, 0.), (-12.1, 0.))):
            with self.subTest(side=side, forward=forward):
                x, y = self.section.to_world((side, forward))
                with self.assertRaisesRegex(ValueError, 'Outside Jamming start line'):
                    recorder.validate_recording_start(dict(x=x, y=y), args)
                path = self.record('outside_%d' % index, start_side=side, start_forward=forward)
                self.assertFalse(recorder.finish_recording(path, args))
        self.assertEqual(json.loads((self.root/"routes.json").read_text())["routes"], [])

    def test_reported_start_positions_and_approach_can_record_and_register(self):
        with patch.object(recorder, "JammingSection", return_value=self.section):
            args = recorder.parse_arguments(['--check'])
        starts = [(39.501, -486.821), (39.342, -486.929),
                  self.section.to_world((-4.11, -6.65)),
                  self.section.to_world((11.9, -19.9))]
        for i, (x, y) in enumerate(starts):
            recorder.validate_recording_start(dict(x=x, y=y), args)
            side, forward = self.section.to_local((x, y))
            path = self.record('approach_%d' % i, start_side=side, start_forward=forward)
            self.assertTrue(recorder.finish_recording(path, args))
        planner = self.planner(populated=False)
        self.assertIsNone(planner.local.error)
        self.assertEqual(len(planner.local.records), len(starts))

    def test_static_recorder_keeps_circular_start_check(self):
        args = recorder.parse_arguments(['--section', 'static'])
        recorder.validate_recording_start(dict(x=args.start_x+3., y=args.start_y+3.), args)
        with self.assertRaisesRegex(ValueError, 'Outside start radius'):
            recorder.validate_recording_start(dict(x=args.start_x+4., y=args.start_y+4.), args)

    def test_lateral_endings_register_but_incomplete_end_progress_does_not(self):
        args = SimpleNamespace(jamming_section=self.section)
        for i, side in enumerate((-7., 7.)):
            path = self.record('lateral_end_%d' % i, end=self.section.length-3.5, end_side=side)
            self.assertTrue(recorder.finish_recording(path, args))
        for i, (progress, side) in enumerate(((self.section.length-5.1, 0.),
                                              (self.section.length+5.1, 0.),
                                              (self.section.length, 12.1))):
            path = self.record('invalid_end_%d' % i, end=progress, end_side=side)
            self.assertFalse(recorder.finish_recording(path, args))
        planner = self.planner(populated=False)
        self.assertIsNone(planner.local.error)
        self.assertEqual(len(planner.local.records), 2)

    def test_jamming_start_overrides_cannot_bypass_registration_gate(self):
        with patch.object(recorder, "JammingSection", return_value=self.section):
            with self.assertRaises(SystemExit):
                recorder.parse_arguments(['--start-radius', '20'])

    def test_common_line_keeps_all_routes_and_discards_only_the_tail(self):
        shortest = self.record('short', end=self.section.length-4.8, end_side=-2.)
        longest = self.record('long', end=self.section.length+1., end_side=7.)
        reverse = self.record('tail_reverse', end=self.section.length-1.)
        with reverse.open(encoding='utf-8', newline='') as stream:
            last = list(csv.DictReader(stream))[-1]
        x, y = self.section.to_world((0.,self.section.length-1.1))
        last.update(x=x,y=y,s_m=float(last['s_m'])+.1)
        with reverse.open('a',encoding='utf-8',newline='') as stream:
            csv.DictWriter(stream,fieldnames=recorder.FIELDS).writerow(last)
        paths = [shortest,longest,reverse]
        originals = {p.name:p.read_bytes() for p in paths}
        requested = list(self.section.to_world((0.,self.section.length-4.)))
        manifest = prepare_library(self.section,[p.name for p in paths],requested)
        self.assertAlmostEqual(manifest['common_end_forward_m'],self.section.length-4.8)
        self.assertAlmostEqual(manifest['actual_end_before_requested_m'],.8)
        for path in paths:
            self.assertEqual(path.read_bytes(),originals[path.name])
            with (self.root/PREPARED_DIRECTORY/path.name).open(encoding='utf-8',newline='') as stream:
                rows = list(csv.DictReader(stream))
            local = self.section.validate_recording([tuple(float(r[k]) for k in ('x','y','yaw_rad')) for r in rows])
            self.assertAlmostEqual(local[-1][1],manifest['common_end_forward_m'])
        planner = self.planner(populated=False)
        self.assertIsNone(planner.local.error)
        self.assertEqual(len(planner.local.records),3)
        for _,data,_ in planner.local.records:
            self.assertAlmostEqual(data[-1,1],self.section.length-self.section.return_blend_m)
        new_path = self.record('new')
        self.assertTrue(recorder.finish_recording(new_path,SimpleNamespace(jamming_section=self.section)))
        updated = json.loads((self.root/'routes.json').read_text(encoding='utf-8'))
        self.assertEqual(len(updated['routes']),4)
        self.assertEqual(updated['requested_end_enu'],requested)
        self.assertEqual(json.loads(new_path.with_suffix('.status.json').read_text())['prepared_path'],
                         PREPARED_DIRECTORY+'/'+new_path.name)

    def test_common_line_rejects_reverse_inside_retained_segment(self):
        path = self.record('reverse_middle')
        with path.open(encoding='utf-8',newline='') as stream:
            rows = list(csv.DictReader(stream))
        x,y = self.section.to_world((0.,39.9))
        rows[201].update(x=x,y=y)
        with path.open('w',encoding='utf-8',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=recorder.FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        before=(self.root/'routes.json').read_bytes()
        with self.assertRaisesRegex(ValueError,'reverses'):
            prepare_library(self.section,[path.name])
        self.assertEqual((self.root/'routes.json').read_bytes(),before)
        self.assertFalse((self.root/PREPARED_DIRECTORY).exists())

    def test_existing_library_does_not_leak_into_jamming(self):
        planner = self.planner(populated=False)
        self.assertEqual(planner.local.records, [])
        self.assertIn("No complete Jamming routes", planner.local.error)
        route = self.route()
        result = planner.run(self.ego(), route[:80], [{'object_info': self.obstacle()}], global_path=route)
        self.assertEqual(result.acc_target_velocity, 0.)
        self.assertEqual(result.reason, "static_recorded_library_error")

    def test_inactive_outside_section_wrong_direction_and_no_obstacles(self):
        planner = self.planner(populated=False)
        route = self.route()
        for ego in (self.ego(y=-100.), self.ego(x=30.), self.ego(y=200.), self.ego(yaw=-math.pi/2)):
            self.assertIsNone(planner.run(ego, route, [{'object_info': self.obstacle()}], global_path=route))
        self.assertIsNone(planner.run(self.ego(), route, [], global_path=route))

    def test_rotated_obb_check_matches_world_check_and_keeps_inputs(self):
        planner = self.planner()
        obstacle = self.obstacle(x=2.)
        original = (obstacle.position.copy(), obstacle.heading_deg)
        objects = planner._obstacles([{'object_info': obstacle}])
        local_xy = np.array([[0., 0.], [0., 35.], [2., 35.], [4., 70.]])
        local_yaw = np.full(4, math.pi/2)
        world_xy = np.array([self.section.to_world(p) for p in local_xy])
        world_yaw = local_yaw-self.section.rotation
        original_planner = RecordedStaticPlanner(LatticePlanner())
        np.testing.assert_array_equal(planner.local._collision_mask(local_xy, local_yaw, objects),
            original_planner._collision_mask(world_xy, world_yaw, [obstacle]))
        np.testing.assert_array_equal(original[0], obstacle.position)
        self.assertEqual(original[1], obstacle.heading_deg)

    def test_avoidance_returns_enu_path_and_rechecks_new_obstacle(self):
        planner = self.planner()
        route, ego, obstacle = self.route(), self.ego(), self.obstacle()
        result = planner.run(ego, route[:80], [{'object_info': obstacle}], global_path=route)
        self.assertTrue(result.active, result.debug_details)
        self.assertTrue(result.debug_details['static_avoidance']['path_validated'], result.debug_details)
        self.assertGreater(result.acc_target_velocity, 0.)
        self.assertEqual(result.mode, "static_obstacle")
        self.assertEqual(result.debug_details['jamming']['path_frame'], "ENU")
        np.testing.assert_allclose([self.section.to_local(p) for p in result.path], planner.local.path, atol=1e-9)
        self.assertGreater(max(p.x for p in planner.local.path), 2.)
        self.assertEqual(len(result.candidates), 1)
        self.assertIs(result.candidates[0].points, result.path)
        np.testing.assert_allclose(result.candidates[0].points, result.path)
        original = np.array(result.path)
        blocking = ObjectInfo(*result.path[150], 0., 2, is_static=True, size_x=3., size_y=3.)
        stopped = planner.run(ego, route[:80], [{'object_info': obstacle}, {'object_info': blocking}], global_path=route)
        self.assertEqual(stopped.acc_target_velocity, 0.)
        np.testing.assert_allclose(stopped.path, original)
        # Restoring the clear scene allows the normal global-route handoff.
        end_ego = self.ego(y=self.section.length+.2)
        end_reference = [p for p in route if self.section.to_local(p)[1] > self.section.length-.2]
        end = planner.run(end_ego, end_reference, [{'object_info': obstacle}], global_path=route)
        self.assertEqual(end.reason, "static_recorded_rejoined")
        self.assertFalse(end.active)
        np.testing.assert_allclose(end.path, end_reference, atol=1e-9)

    def test_process_routes_this_section_to_jamming(self):
        planner = self.planner(populated=False)
        process = LatticePlanningProcess(highway_zone_points=())
        process.recorded_jamming.close()
        process.recorded_jamming = planner
        self.addCleanup(process.recorded_static.executor.shutdown, True)
        with patch.object(process.static_avoidance, 'plan', side_effect=AssertionError("Old zone used")):
            result = process.run(self.ego(), self.route()[:80], [{'object_info': self.obstacle()}], global_path=self.route())
        self.assertEqual(result.debug_details['jamming']['section'], 'Jamming')
        self.assertEqual(result.acc_target_velocity, 0.)

    def test_background_preplan_and_handoff_use_repository_global_route(self):
        planner = self.planner(background=True)
        route_file = Path(__file__).resolve().parents[3] / "map_data/R_KR_PG_KATRI_2025/2026_molit_comp_global_path.txt"
        route = [Point(*row[:2]) for row in np.loadtxt(route_file)]
        objects = [{'object_info': self.obstacle()}]
        approach = self.ego(y=-30.)
        nearest = min(range(len(route)), key=lambda i: approach.position.distance(route[i]))
        reference = route[nearest:nearest+80]
        planner.run(approach, reference, objects, global_path=route)
        self.assertIsNotNone(planner.local.pending)
        worker = planner.local.pending.result(timeout=20.)
        self.assertTrue(worker.path, "No complete return path on the repository map")
        self.assertTrue(planner.local.preplanned)
        approaching = planner.run(approach, reference, objects, global_path=route)
        self.assertEqual(approaching.reason, 'static_recorded_approach')
        np.testing.assert_allclose(approaching.path, reference, atol=1e-9)

        entry = self.ego(x=worker.path[0].x, y=worker.path[0].y+.01,
                         yaw=worker.planner._path_heading(worker.path, 0))
        nearest = min(range(len(route)), key=lambda i: entry.position.distance(route[i]))
        result = planner.run(entry, route[nearest:nearest+80], objects, global_path=route)
        self.assertTrue(result.active, result.debug_details)
        self.assertTrue(result.debug_details['static_avoidance']['path_validated'])
        self.assertGreater(result.acc_target_velocity, 0., result.debug_details)
        self.assertIsNone(planner.local.pending)
        self.assertGreaterEqual(planner.local.path[-1].y,
                                self.section.length+self.section.global_tail_m-.1)

    def test_gt_warms_before_activation_and_keeps_plan_until_entry(self):
        planner = self.planner(background=True)
        route, objects = self.route(), [{'object_info': self.obstacle()}]
        # GT is visible well before the old 20 m activation gate, even while
        # Ego is on a laterally displaced approach to this section.
        self.assertIsNone(planner.run(self.ego(x=30., y=-150.), route[:80], objects, global_path=route))
        future = planner.local.pending
        self.assertIsNotNone(future)
        self.assertTrue(future.result(timeout=20.).path)
        for y in (-80., -40., -30., -20.1):
            reference = [p for p in route if self.section.to_local(p)[1] >= y][:80]
            result = planner.run(self.ego(y=y), reference, objects, global_path=route)
            self.assertEqual(result.reason, 'static_recorded_approach')
            self.assertFalse(result.active)
            self.assertIs(planner.local.pending, future)
            np.testing.assert_allclose(result.path, reference, atol=1e-9)
            self.assertEqual(len(result.candidates), 1)
            self.assertTrue(result.candidates[0].selected_preview)
        result = planner.run(self.ego(y=-19.99), route[:80], objects, global_path=route)
        self.assertTrue(result.debug_details['static_avoidance']['path_validated'], result.debug_details)
        self.assertIsNone(planner.local.pending)

    def test_captured_jamming_stop_generates_complete_path_in_background(self):
        import time
        snapshot = json.loads((Path(__file__).parent/'jamming_scene_20261006.json').read_text())
        planner = JammingPlanner(background=True)
        self.addCleanup(planner.close)
        route_file = Path(__file__).resolve().parents[3] / 'map_data/R_KR_PG_KATRI_2025/2026_molit_comp_global_path.txt'
        route = [Point(*row[:2]) for row in np.loadtxt(route_file)]
        ego = VehicleState(*snapshot['ego'])
        objects = [{'object_info': ObjectInfo(**{k:v for k,v in row.items() if k!='group'})}
                   for row in snapshot['objects']]
        nearest = min(range(len(route)), key=lambda i: ego.position.distance(route[i]))
        reference = route[nearest:nearest+50]
        self.assertEqual(len(planner.local.records), 9)
        result = planner.run(ego, reference, objects, global_path=route)
        deadline = time.monotonic()+20.
        # Keep the normal per-frame work running beside the worker.
        while not result.active and time.monotonic() < deadline:
            time.sleep(.01)
            result = planner.run(ego, reference, objects, global_path=route)
        self.assertTrue(result.debug_details['static_avoidance']['path_validated'], result.debug_details)
        self.assertEqual(planner.local.search_details['method'], 'recorded_connections')
        self.assertGreater(result.acc_target_velocity, 0.)
        xy = np.asarray(result.path)
        yaw = np.array([planner.local.planner._path_heading(result.path, i) for i in range(len(xy))])
        self.assertTrue(planner.local.collision_free(xy, yaw, [o['object_info'] for o in objects], .12))
        self.assertGreaterEqual(planner.local.path[-1].y, planner.section.length+9.9)

    def test_static_barrel_at_entry_does_not_deadlock_in_approach_acc(self):
        import time
        from expert.src.autonomous_driving import AutonomousDriving
        snapshot = json.loads((Path(__file__).parent/'jamming_entry_scene_20261006.json').read_text())
        route_file = Path(__file__).resolve().parents[3] / 'map_data/R_KR_PG_KATRI_2025/2026_molit_comp_global_path.txt'
        route = [Point(*row[:2]) for row in np.loadtxt(route_file)]
        objects = [{'object_info': ObjectInfo(**{k:v for k,v in row.items() if k!='group'})}
                   for row in snapshot['objects']]
        for pose in (snapshot['ego'], snapshot['approach_ego']):
            with self.subTest(pose=pose):
                planner = JammingPlanner(background=True)
                try:
                    ego = VehicleState(*pose)
                    self.assertLess(planner._ego(ego).position.y, 0.)
                    nearest = min(range(len(route)), key=lambda i: ego.position.distance(route[i]))
                    reference = route[nearest:nearest+50]
                    result = planner.run(ego, reference, objects, global_path=route)
                    deadline = time.monotonic()+20.
                    while not result.active and time.monotonic()<deadline:
                        time.sleep(.01)
                        result = planner.run(ego, reference, objects, global_path=route)
                    self.assertEqual(result.reason, 'static_recorded_follow', result.debug_details)
                    self.assertTrue(result.debug_details['static_avoidance']['path_validated'])
                    self.assertGreater(result.acc_target_velocity, .2)
                    self.assertLess(result.path[0].distance(ego.position), .01)
                    # Existing ACC must relinquish these static objects only
                    # after the approach + avoidance path is validated.
                    controller = AutonomousDriving.__new__(AutonomousDriving)
                    controller.lattice_planning_process = SimpleNamespace(
                        roundabout_entry_should_ignore_legacy_acc=lambda *_: False)
                    self.assertEqual(controller._legacy_acc_entries(ego,result,objects), [])
                    moving = {'object_info': ObjectInfo(ego.position.x,ego.position.y,2.,1)}
                    self.assertEqual(controller._legacy_acc_entries(ego,result,objects+[moving]), [moving])
                    blocked = ObjectInfo(*result.path[20],0.,2,is_static=True,size_x=3.,size_y=3.)
                    result = planner.run(ego, reference, objects+[{'object_info':blocked}], global_path=route)
                    self.assertFalse(result.debug_details['static_avoidance']['path_validated'])
                    self.assertEqual(result.acc_target_velocity,0.)
                    self.assertEqual(controller._legacy_acc_entries(ego,result,objects),objects)
                finally:
                    planner.close()


if __name__ == "__main__":
    unittest.main()
