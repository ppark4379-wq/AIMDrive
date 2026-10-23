"""Extraction regression: frozen improved DR is a test oracle, never a runtime backend."""
import ast
from collections import Counter, deque
import copy
import gzip
import hashlib
import json
from pathlib import Path
import sys
import threading
import types
import unittest
from unittest.mock import patch

from gps_jamming.localization import gps_aligned_hold, gps_blackout_hold, map_geometry
from localization_support import BASIS


FIXTURES = Path(__file__).parent / 'fixtures' / 'dr_baseline'
ORACLE_NAME = '_gps_jamming_test_only_dr_baseline'
IMPORT_MAP = {
    'inference.sensors.gps_blackout_hold': ORACLE_NAME + '.gps_blackout_hold',
    'data.src.loaders.morai_map_geometry': ORACLE_NAME + '.map_geometry',
}


class RemapImports(ast.NodeTransformer):
    def visit_ImportFrom(self, node):
        if node.module in IMPORT_MAP:
            node.module = IMPORT_MAP[node.module]
        return node


def baseline_sources():
    manifest = json.loads((FIXTURES / 'manifest.json').read_text())
    sources = {}
    for name, evidence in manifest['sources'].items():
        raw = (FIXTURES / (name + '.txt')).read_bytes()
        if hashlib.sha256(raw).hexdigest() != evidence['sha256']:
            raise AssertionError('Frozen oracle changed: ' + name)
        sources[name] = raw.decode()
    return sources


def load_oracle():
    """Load immutable sources under private test names; remap imports only."""
    sources = baseline_sources()
    package = types.ModuleType(ORACLE_NAME)
    package.__path__ = []
    sys.modules[ORACLE_NAME] = package
    modules = {}
    for stem in ('map_geometry', 'gps_blackout_hold', 'gps_aligned_hold'):
        name = ORACLE_NAME + '.' + stem
        module = types.ModuleType(name)
        module.__package__ = ORACLE_NAME
        module.__file__ = str(FIXTURES / (stem + '.py.txt'))
        tree = RemapImports().visit(ast.parse(sources[stem + '.py']))
        sys.modules[name] = module
        exec(compile(tree, module.__file__, 'exec'), module.__dict__)
        modules[stem] = module
    return modules


class ReplayHistory:
    def __init__(self):
        self.history = deque(maxlen=200)
        self.history_lock = threading.Lock()


class ReplayGps:
    def __init__(self, clock):
        self.clock = clock
        self.fix = None
        self.identity = {}
        self.port = 10506
        self.packets = self.samples = 0

    def latest_snapshot(self):
        if self.fix is None:
            return None, float('inf'), {}
        return dict(self.fix), (self.clock[0] - self.identity['received_ns']) / 1e9, dict(self.identity)

    def diagnostics(self):
        return dict(packets=self.packets, samples=self.samples)

    def close(self):
        pass


class ExtractionEquivalenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.oracle = load_oracle()

    def test_core_source_ast_differs_only_in_import_paths(self):
        """Protect all equations, defaults, state transitions and metadata fields."""
        originals = baseline_sources()
        for module in (gps_blackout_hold, gps_aligned_hold):
            name = Path(module.__file__).name
            original = ast.parse(originals[name])
            for node in ast.walk(original):
                if isinstance(node, ast.ImportFrom):
                    if node.module == 'inference.sensors.gps_blackout_hold':
                        node.module, node.level = 'gps_blackout_hold', 1
                    elif node.module == 'data.src.loaders.morai_map_geometry':
                        node.module, node.level = 'map_geometry', 1
            extracted = ast.parse(Path(module.__file__).read_text())
            self.assertEqual(ast.dump(original), ast.dump(extracted), name)

    def test_map_projection_methods_preserve_original_ast_and_results(self):
        original = ast.parse(baseline_sources()['map_geometry.py'])
        extracted = ast.parse(Path(map_geometry.__file__).read_text())
        def methods(tree):
            result = {}
            for node in tree.body:
                if isinstance(node, ast.ClassDef) and node.name == 'MapGeometry':
                    result.update((method.name, method) for method in node.body
                                  if isinstance(method, ast.FunctionDef)
                                  and method.name in ('__init__', 'convergence_degrees'))
                elif isinstance(node, ast.FunctionDef) and node.name == 'validate_map_basis':
                    result[node.name] = node
            return result
        left, right = methods(original), methods(extracted)
        self.assertEqual(set(left), set(right))
        for name in left:
            self.assertEqual(ast.dump(left[name]), ast.dump(right[name]), name)
        actual = map_geometry.validate_map_basis(BASIS)
        expected = self.oracle['map_geometry'].validate_map_basis(BASIS)
        for position in ((0., 0., 0.), (29.589355843, -487.650272536, 0.),
                         (-74.860744596, -544.578032671, 0.), (10000., -10000., 10.)):
            self.assertEqual(actual.convergence_degrees(position), expected.convergence_degrees(position))
            east, north = (position[i] + BASIS['origin_xyz'][i] for i in (0, 1))
            self.assertEqual(actual.inverse.transform(east, north), expected.inverse.transform(east, north))
        for changes in ({'schema': 'implicit'}, {'orientation_frame': 'true_enu'},
                        {'origin_xyz': [0., 0.]}, {'crs': 'EPSG:4326'}):
            with self.assertRaises(ValueError):
                map_geometry.validate_map_basis(dict(BASIS, **changes))

    def test_recorded_drive_every_output_matches_frozen_original(self):
        """Same events/polls: exact output, transition, comparison and counter equality.

        This compares the two implementations. Original live poll nanoseconds
        are not recoverable from float control timestamps in the old recording.
        """
        fixture = json.loads(gzip.decompress((FIXTURES / 'real_drive_allowed_events.json.gz').read_bytes()))
        events = sorted(enumerate(fixture['events']), key=lambda item: (item[1][1], item[0]))
        for method in ('legacy', 'aligned'):
            with self.subTest(method=method):
                clock = [0]
                holders, sources = [], []
                for blackout, aligned in ((self.oracle['gps_blackout_hold'], self.oracle['gps_aligned_hold']),
                                          (gps_blackout_hold, gps_aligned_hold)):
                    gps = ReplayGps(clock)
                    status = types.SimpleNamespace(speed=ReplayHistory(), imu=ReplayHistory())
                    kwargs = dict(simulate_zone=blackout.SectionZone(**fixture['zone']))
                    if method == 'aligned':
                        kwargs['anchor_window_s'] = fixture['anchor_window_s']
                        hold = aligned.AlignedGpsBlackoutHold(gps, status, fixture['basis'], **kwargs)
                    else:
                        hold = blackout.GpsBlackoutHold(gps, status, fixture['basis'], **kwargs)
                    holders.append(hold)
                    sources.append((gps, status))
                counts, modes, comparisons = Counter(), Counter(), set()
                cursor = 0
                saw_recovery = False
                last_mode = None
                with patch('time.monotonic_ns', side_effect=lambda: clock[0]):
                    for tick in fixture['polls_ns']:
                        clock[0] = tick
                        while cursor < len(events) and events[cursor][1][1] <= tick:
                            _, (sensor, receipt, source_stamp, value) = events[cursor]
                            cursor += 1
                            counts[sensor] += 1
                            identity = dict(received_ns=receipt, source_stamp_ns=source_stamp,
                                            sample_id=counts[sensor])
                            for gps, status in sources:
                                if sensor == 'gps':
                                    gps.fix, gps.identity = copy.deepcopy(value), dict(identity)
                                    gps.packets = gps.samples = counts[sensor]
                                else:
                                    stream = status.speed if sensor == 'status' else status.imu
                                    stream.history.append((copy.deepcopy(value), dict(identity)))
                        expected, actual = (hold.latest_snapshot() for hold in holders)
                        self.assertEqual(actual, expected, (method, tick))
                        self.assertEqual(holders[1].core.summary(), holders[0].core.summary())
                        self.assertEqual(holders[1].core.xy, holders[0].core.xy)
                        self.assertEqual(holders[1].diagnostics(), holders[0].diagnostics())
                        mode = actual[2]['position_mode']
                        modes[mode] += 1
                        saw_recovery |= last_mode == 'dead_reckoning' and mode == 'gps'
                        last_mode = mode
                        comparisons.update(row['gps_received_ns'] for row in actual[2].get('simulated_comparisons', []))
                self.assertGreater(modes['gps'], 50)
                self.assertGreater(modes['dead_reckoning'], 100)
                self.assertTrue(saw_recovery)
                self.assertGreater(len(comparisons), 200)


if __name__ == '__main__':
    unittest.main()
