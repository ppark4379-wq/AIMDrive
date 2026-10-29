"""Preserve supplement tracking while separating NPC spawn configuration."""
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from AIM2026.Inference.vision.lane_tracking import LaneTracker
ROOT = Path(__file__).resolve().parents[2]

class SupplementIndependenceTests(unittest.TestCase):
    def test_lane_geometry_and_tracking_match_committed_baseline(self):
        import ast
        import copy
        import types
        import yaml
        def baseline(path):
            return subprocess.check_output(['git', 'show', '5c9d9cc0:'+path], cwd=ROOT).decode()
        path = 'AIM2026/Inference/vision/lane_tracking.py'
        old = baseline(path)
        module = types.ModuleType('baseline_lane_tracking')
        sys.modules[module.__name__] = module
        exec(compile(old, path, 'exec'), module.__dict__)
        config = yaml.safe_load(baseline('AIM2026/utils/scenario_runner/config/full_loop.yaml'))['scenarios']['full_loop']
        links = {row['idx']: row for row in json.loads((ROOT/'AIM2026/Expert/map_data/R_KR_PG_KATRI_2025/link_set.json').read_text())}
        road = SimpleNamespace(link_set=links, get_link_points=lambda key: links[key]['points'])
        current = LaneTracker.from_map(road.link_set)
        # The factory is the only changed tracker method.
        def methods(source):
            cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == 'LaneTracker')
            return {n.name: ast.dump(n) for n in cls.body if isinstance(n, ast.FunctionDef) and n.name not in ('from_context', 'from_map')}
        self.assertEqual(methods(old), methods((ROOT/path).read_text()))
        for policy in ('dense', 'competition'):
            scene = copy.deepcopy(config)
            scene['highway_npc']['managed_traffic']['policy'] = policy
            scene['highway_npc']['managed_traffic']['competition'] = yaml.safe_load(baseline('AIM2026/utils/scenario_runner/config/highway_competition.yaml'))['highway_npc']['managed_traffic']['competition']
            previous = module.LaneTracker.from_context(SimpleNamespace(cfg=scene, map_loader=road))
            for name in ('middle', 'left'):
                self.assertEqual(vars(current.lanes[name]), vars(previous.lanes[name]))
        memory = 'AIM2026/Inference/vision/ve_lead_memory.py'
        self.assertEqual(ast.dump(ast.parse(baseline(memory))), ast.dump(ast.parse((ROOT/memory).read_text())))

    def test_missing_vehicle_continues_straight_at_fork(self):
        from AIM2026.Inference.vision.lane_tracking import Lane
        links = {r['idx']: r for r in json.loads((ROOT/'AIM2026/Expert/map_data/R_KR_PG_KATRI_2025/link_set.json').read_text())}
        definitions = json.loads((ROOT/'AIM2026/Expert/src/config/highway_lanes.json').read_text())
        for ids in definitions.values():
            for a, b in zip(ids, ids[1:]):
                self.assertEqual(links[a]['to_node_idx'], links[b]['from_node_idx'])
                self.assertEqual(links[a]['points'][-1], links[b]['points'][0])
        tracker = LaneTracker.from_map(links)
        lane = tracker.lanes['merge_adjacent']
        fork_s = lane.project(links['A2256W000420']['points'][-1][:2])[1]
        for i in range(11):
            center, heading = lane.at(fork_s-12+i)
            box = dict(track_id=7, center_world=center, heading_deg=heading,
                       planning_heading_deg=heading, length_m=7., width_m=2.5)
            tracker.apply(dict(sensor_received_ns=i*100_000_000, boxes=[box]))
        held = tracker.apply(dict(sensor_received_ns=2_000_000_000, boxes=[]))['boxes']
        self.assertEqual(len(held), 1)
        self.assertTrue(held[0]['lane_track_confirmed'])
        self.assertTrue(held[0]['detection_held'])
        center = held[0]['center_world']
        self.assertLess(Lane(links['A2256W000408']['points']).project(center)[0], .1)
        self.assertGreater(Lane(links['A2256W000179']['points']).project(center)[0], .2)

    def test_factory_does_not_need_npc_configuration(self):
        from AIM2026.Inference.vision.road_lanes import map_lanes
        links = {row['idx']: row for row in json.loads((ROOT/'AIM2026/Expert/map_data/R_KR_PG_KATRI_2025/link_set.json').read_text())}
        self.assertEqual(set(map_lanes(links)), {'merge_adjacent', 'middle', 'left'})
        self.assertEqual(len(LaneTracker.from_map(links).lanes), 3)

if __name__ == '__main__':
    unittest.main()
