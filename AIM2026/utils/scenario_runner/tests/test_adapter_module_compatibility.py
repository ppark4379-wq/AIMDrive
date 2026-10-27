"""Legacy adapter imports must retain one implementation and shared globals."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from scenario_runner.project_paths import RUNNER_ROOT


MODULES = {
    "adapters.ve_objects": "adapters.ve.objects",
    "adapters.ve_signal": "adapters.ve.signal",
    "legacy_expert.ve_udp_session": "adapters.ve.session",
    "legacy_expert.gt_udp_protocol": "adapters.udp.protocol",
    "legacy_expert.run_gt_udp_expert": "adapters.udp.gt_driver",
    "legacy_expert.run_ve_gps_expert": "adapters.udp.gps_ve_driver",
    "legacy_expert.ros.input_timing": "adapters.ros.input_timing",
    "legacy_expert.ros.synchronized_inputs": "adapters.ros.synchronized_inputs",
    "legacy_expert.ros.grpc_sync_inputs": "adapters.ros.grpc_sync_inputs",
}


class AdapterCompatibilityTests(unittest.TestCase):
    def assert_import_order(self, legacy_first):
        # A fresh process catches duplicate class/global state in either order.
        script = '''import importlib, json, sys
from pathlib import Path
from scenario_runner.project_paths import RUNNER_ROOT
pairs = json.loads(sys.argv[1])
legacy_first = sys.argv[2] == 'True'
sys.path.insert(0, str(RUNNER_ROOT / 'legacy_expert'))
for old, new in pairs.items():
    order = [old, new] if legacy_first else [new, old]
    modules = [importlib.import_module('scenario_runner.' + name) for name in order]
    canonical = importlib.import_module('scenario_runner.' + new)
    assert modules[0] is modules[1] is canonical, (old, new)
    assert Path(canonical.__file__).resolve() == RUNNER_ROOT.joinpath(*new.split('.')).with_suffix('.py')
    canonical._compatibility_probe = object()
    for prefix in ('scenario_runner.', 'MoraiDrive.scenario_runner.', 'AIM2026.utils.scenario_runner.'):
        alias = importlib.import_module(prefix + old)
        assert alias is canonical and alias._compatibility_probe is canonical._compatibility_probe
    if old.startswith('legacy_expert.') and not old.startswith('legacy_expert.ros.'):
        assert importlib.import_module(old.rsplit('.', 1)[1]) is canonical
    del canonical._compatibility_probe
'''
        with tempfile.TemporaryDirectory(prefix="adapter-import-identity-") as working:
            result = subprocess.run([sys.executable, "-B", "-c", script,
                json.dumps(MODULES), str(legacy_first)], cwd=working,
                env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_legacy_first(self):
        self.assert_import_order(True)

    def test_canonical_first(self):
        self.assert_import_order(False)

    def test_legacy_package_import_from_runner_directory(self):
        script = '''import importlib, json
import legacy_expert.run_ve_gps_expert
for old, new in json.loads(__import__('sys').argv[1]).items():
    if old.startswith('legacy_expert.'):
        assert importlib.import_module(old) is importlib.import_module('scenario_runner.' + new)
'''
        result = subprocess.run([sys.executable, "-B", "-c", script, json.dumps(MODULES)],
            cwd=str(RUNNER_ROOT), env=dict(os.environ, PYTHONPATH="", PYTHONNOUSERSITE="1"),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)


class RosAdapterCompatibilityTests(unittest.TestCase):
    helpers = ("input_timing", "synchronized_inputs", "grpc_sync_inputs")

    def run_isolated(self, arguments):
        import importlib.util

        # Shared Expert geometry needs NumPy. Expose that installed dependency
        # explicitly while skipping site initialization and inherited paths, so
        # another runner or ROS overlay cannot satisfy imports by accident.
        numpy_root = Path(importlib.util.find_spec("numpy").origin).resolve().parent.parent
        roots = (RUNNER_ROOT.parents[2], RUNNER_ROOT.parent, RUNNER_ROOT,
                 RUNNER_ROOT / "legacy_expert", numpy_root)
        environment = dict(os.environ, PYTHONPATH=os.pathsep.join(map(str, roots)),
                           PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1")
        with tempfile.TemporaryDirectory(prefix="ros-adapter-compatibility-") as working:
            return subprocess.run([sys.executable, "-B", "-S", *arguments],
                                  cwd=working, env=environment,
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                  universal_newlines=True, timeout=30)

    def test_parent_imports_do_not_load_ros_or_helpers(self):
        script = '''import importlib, importlib.abc, sys
from scenario_runner.project_paths import activate_project_paths
activate_project_paths()
class RejectEagerDependencies(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        roots = ('rospy', 'tf', 'numpy', 'morai_msgs', 'geometry_msgs',
                 'nav_msgs', 'std_msgs', 'AIM2026.Expert')
        if (any(fullname == root or fullname.startswith(root + '.') for root in roots)
                or fullname.startswith('scenario_runner.adapters.ros.')):
            raise AssertionError('ROS package eagerly imported ' + fullname)
sys.meta_path.insert(0, RejectEagerDependencies())
package = importlib.import_module(sys.argv[1])
assert hasattr(package, '__path__')
assert not any(name.startswith('scenario_runner.adapters.ros.') for name in sys.modules)
'''
        for package in ("scenario_runner.legacy_expert.ros",
                        "MoraiDrive.scenario_runner.legacy_expert.ros",
                        "AIM2026.utils.scenario_runner.legacy_expert.ros",
                        "legacy_expert.ros", "ros"):
            with self.subTest(package=package):
                result = self.run_isolated(["-c", script, package])
                self.assertEqual(result.returncode, 0, result.stderr)

    def assert_bare_ros_import_order(self, legacy_first):
        script = '''import importlib, json, sys
from scenario_runner.project_paths import activate_project_paths
activate_project_paths()
for helper in json.loads(sys.argv[1]):
    old = 'ros.' + helper
    new = 'scenario_runner.adapters.ros.' + helper
    order = (old, new) if sys.argv[2] == 'True' else (new, old)
    first, second = map(importlib.import_module, order)
    assert first is second, (old, new)
    marker = object()
    first._ros_compatibility_probe = marker
    for parent in ('ros', 'legacy_expert.ros', 'scenario_runner.legacy_expert.ros',
                   'MoraiDrive.scenario_runner.legacy_expert.ros',
                   'AIM2026.utils.scenario_runner.legacy_expert.ros'):
        alias = importlib.import_module(parent + '.' + helper)
        assert alias is first and alias._ros_compatibility_probe is marker
        imported = __import__(parent, fromlist=[helper])
        assert getattr(imported, helper) is first
    del first._ros_compatibility_probe
'''
        result = self.run_isolated(["-c", script, json.dumps(self.helpers), str(legacy_first)])
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_bare_ros_first_shares_canonical_module_state(self):
        self.assert_bare_ros_import_order(True)

    def test_canonical_first_shares_bare_ros_module_state(self):
        self.assert_bare_ros_import_order(False)

    def test_helpers_support_module_execution_without_output(self):
        # They expose importable helpers, not a driving CLI. Their historical
        # successful module execution imports definitions and exits quietly.
        for parent in ("scenario_runner.legacy_expert.ros",
                       "AIM2026.utils.scenario_runner.legacy_expert.ros",
                       "legacy_expert.ros", "ros", "scenario_runner.adapters.ros"):
            for helper in self.helpers:
                with self.subTest(module=parent + "." + helper):
                    result = self.run_isolated(["-m", parent + "." + helper])
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(result.stdout, "")
                    self.assertEqual(result.stderr, "")

    def test_historical_prefix_module_execution_after_project_bootstrap(self):
        # The historical MoraiDrive namespace is installed by the AIM project
        # bootstrap. Use the same runpy path as -m after that explicit bootstrap.
        script = '''import runpy, sys
from scenario_runner.project_paths import activate_project_paths
activate_project_paths()
runpy.run_module(sys.argv[1], run_name='__main__', alter_sys=True)
'''
        for helper in self.helpers:
            with self.subTest(helper=helper):
                module = "MoraiDrive.scenario_runner.legacy_expert.ros." + helper
                result = self.run_isolated(["-c", script, module])
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout, "")
                self.assertEqual(result.stderr, "")


if __name__ == "__main__":
    unittest.main()
