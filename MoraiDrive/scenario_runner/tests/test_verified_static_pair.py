import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import yaml

from MoraiDrive.scenario_runner.scenarios.full_loop_parts.static_obstacle import (
    verified_pair_presets, load_static_obstacle_preset,
)
from MoraiDrive.scenario_runner.scenarios.full_loop_parts.highway import _merge_static_vehicle_pose


class VerifiedPairTests(unittest.TestCase):
    def test_rejects_failed_unverified_and_wrong_count_layouts(self):
        source = yaml.safe_load(verified_pair_presets()[0].read_text())
        with tempfile.TemporaryDirectory() as directory:
            for index, mutate in enumerate([
                lambda d: None,
                lambda d: d['verification'].update(collision_free=False),
                lambda d: d['verification']['lane_compliance'].update(passed=False),
                lambda d: d['generation'].update(candidate_only_until_expert_passes=True),
                lambda d: d['obstacles'].pop(),
                lambda d: d.update(blockage_release={'enabled': True}),
            ]):
                document = copy.deepcopy(source)
                mutate(document)
                (Path(directory) / ('%d.yaml' % index)).write_text(yaml.safe_dump(document))
            self.assertEqual([p.name for p in verified_pair_presets(directory)], ['0.yaml'])

    def test_all_real_pairs_preserve_original_geometry(self):
        pool = verified_pair_presets()
        self.assertGreater(len(pool), 0)
        for path in pool:
            source = yaml.safe_load(path.read_text())
            points = [[o['position'][k] for k in ('x', 'y', 'z')] for o in source['obstacles']]
            loaded = load_static_obstacle_preset(path, source['map_name'], points)
            self.assertEqual(len(loaded['placements']), 2)
            for actual, original in zip(loaded['placements'], source['obstacles']):
                for field in ('model', 'position', 'yaw_deg', 'scale'):
                    self.assertEqual(actual[field], original[field])

    def test_disabled_merge_car_needs_no_spawn_or_route_lookup(self):
        context = SimpleNamespace(cfg={'highway_merge_static_vehicle': {'enabled': False}})
        self.assertIsNone(_merge_static_vehicle_pose(context))


if __name__ == '__main__':
    unittest.main()
