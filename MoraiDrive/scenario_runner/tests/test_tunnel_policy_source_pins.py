"""Approval identity follows immutable execution dependencies, never run history."""

import json
from pathlib import Path
import tempfile
import unittest

from ..scenarios.full_loop_parts.tunnel_static_parts import evidence, execution_policy as policy


class TunnelPolicySourcePinsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.expert = self.root/"expert"
        original = self.expert/"src/expert/src/config/config.json"
        original.parent.mkdir(parents=True)
        original.write_text(json.dumps({"planning":{"velocity_profile":{"max_velocity":37.}}}))
        self.controller = self.expert/"src/expert/src/control/path_controller.py"
        self.controller.parent.mkdir(parents=True)
        self.controller.write_text("# reviewed controller revision\n")
        self.map = self.root/"map"
        self.map.mkdir()
        (self.map/"link_set.json").write_text("[]")
        (self.map/"node_set.json").write_text("[]")
        self.library = self.root/"library.json"
        self.library.write_text(json.dumps({"schema":"isolated-policy-source-test","candidates":[]}))
        self.global_cfg = {"paths":{"expert_root":str(self.expert),"mgeo_root":str(self.map),
                                    "runtime_root":str(self.root/"runtime")}}

    def describe(self):
        return policy.describe(self.global_cfg,self.library)

    def test_policy_includes_every_shared_runtime_snapshot_dependency(self):
        helper = policy.PACKAGE_ROOT/"tools/tunnel_static_batch_safety.sh"
        expected = evidence.source_snapshot({},helper,helper)["runtime_files"]
        actual = self.describe()["source_files"]
        for path,sha in expected.items():
            with self.subTest(path=path):
                self.assertEqual(actual.get(path),sha)

    def test_transport_scenario_configuration_and_actual_map_are_pinned(self):
        actual = self.describe()["source_files"]
        paths = [policy.PACKAGE_ROOT/relative for relative in (
            "tools/__init__.py", "tools/run_tunnel_static_trial.sh", "tools/tunnel_static_batch_safety.sh",
            "config/global.yaml", "config/full_loop.yaml", "config/tunnel_static.yaml",
            "scenarios/full_loop_parts/tunnel_static_parts/connection.py", "adapters/morai_grpc.py",
            "core/base_scenario.py", "utils/map_loader.py", "utils/link_graph.py", "utils/transform_utils.py",
            "scenarios/full_loop_parts/actor_lifecycle.py", "scenarios/full_loop_parts/vehicle_common.py",
            "scenarios/full_loop_parts/static_obstacle.py",
        )]
        paths += [self.map/"link_set.json", self.map/"node_set.json", self.controller,
                  policy.PACKAGE_ROOT.parent/"data_collection/tools/internal_sensor_record_env.sh"]
        for path in paths:
            with self.subTest(path=path):
                self.assertEqual(actual.get(str(path.resolve())),policy.digest(path))

    def test_controller_or_map_change_invalidates_policy_identity(self):
        before = policy.identity(self.describe())
        self.controller.write_text("# changed controller revision\n")
        after_controller = policy.identity(self.describe())
        self.assertNotEqual(before,after_controller)
        (self.map/"link_set.json").write_text('[{"different":"road"}]')
        self.assertNotEqual(after_controller,policy.identity(self.describe()))

    def test_mutable_weather_approval_and_log_files_do_not_change_policy_identity(self):
        before = policy.identity(self.describe())
        runtime = self.root/"runtime"
        runtime.mkdir()
        for name in ("full_loop_tunnel_static_profile.json", "current_expert_approvals.json", "trial.log"):
            (runtime/name).write_text("mutable run state\n")
        self.assertEqual(before,policy.identity(self.describe()))


if __name__ == "__main__":
    unittest.main()
