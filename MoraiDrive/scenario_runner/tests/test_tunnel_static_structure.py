"""Relocated tunnel runtime keeps command compatibility and one implementation."""

import ast
import importlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

from .. import tools
from ..scenarios.full_loop_parts.tunnel_static_parts import evidence


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT.parent


class TunnelStaticStructureTests(unittest.TestCase):
    def command(self, *arguments):
        environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        return subprocess.run([sys.executable, *arguments], cwd=str(PROJECT_ROOT), env=environment,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=15.)

    def test_legacy_imports_share_the_canonical_module_and_preserve_its_spec(self):
        for old,target in tools._TUNNEL_LEGACY_MODULES.items():
            with self.subTest(old=old):
                canonical = importlib.import_module(target)
                legacy = importlib.import_module("scenario_runner.tools."+old)
                self.assertIs(legacy,canonical)
                self.assertEqual(canonical.__spec__.name,target)
                self.assertEqual(canonical.__package__,target.rsplit(".",1)[0])

    def test_legacy_patching_reaches_the_same_runtime_function(self):
        legacy = importlib.import_module("scenario_runner.tools.replay_tunnel_static_library")
        canonical = importlib.import_module(tools._TUNNEL_LEGACY_MODULES["replay_tunnel_static_library"])
        with patch.object(legacy,"digest",return_value="patched"):
            self.assertEqual(canonical.digest("unused"),"patched")

    def test_normal_runtime_imports_do_not_load_development_tool_modules(self):
        code = ("import json,sys; "
                "from scenario_runner.scenarios.full_loop_parts.tunnel_static_parts import replay,execution,trial,safety; "
                "print(json.dumps(sorted(name for name in sys.modules "
                "if name.startswith('scenario_runner.tools.tunnel_static.'))))")
        result = self.command("-c",code)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(json.loads(result.stdout.splitlines()[-1]),[])

    def test_every_legacy_command_and_grouped_entrypoint_still_support_help(self):
        for old in tools._TUNNEL_LEGACY_MODULES:
            with self.subTest(old=old):
                result = self.command("-m","scenario_runner.tools."+old,"--help")
                self.assertEqual(result.returncode,0,result.stderr)
                self.assertNotIn("Traceback",result.stderr)
        for arguments in (("--help",),("trial","--help"),("verify-models","--help")):
            with self.subTest(grouped=arguments):
                result = self.command("-m","scenario_runner.tools.tunnel_static",*arguments)
                self.assertEqual(result.returncode,0,result.stderr)

    def test_flat_tools_keep_only_two_tunnel_static_runtime_launch_wrappers(self):
        files = {path.name for path in (PACKAGE_ROOT/"tools").glob("*tunnel_static*") if path.is_file()}
        self.assertEqual(files,{"run_tunnel_static_trial.sh","tunnel_static_batch_safety.sh"})
        for path in (PACKAGE_ROOT/"tools/tunnel_static").glob("*.sh"):
            result = subprocess.run(["bash","-n",str(path)],capture_output=True,text=True,timeout=5.)
            self.assertEqual(result.returncode,0,result.stderr)

    def test_fresh_source_pins_cover_canonical_code_and_legacy_dispatch(self):
        helper = PACKAGE_ROOT/"tools/tunnel_static_batch_safety.sh"
        snapshot = evidence.source_snapshot({},helper,helper)
        files = snapshot["runtime_files"]
        for relative in ("tools/__init__.py","scenarios/full_loop_parts/tunnel_static_parts/replay.py",
                         "scenarios/full_loop_parts/tunnel_static_parts/execution.py",
                         "scenarios/full_loop_parts/tunnel_static_parts/trial.py",
                         "scenarios/full_loop_parts/tunnel_static_parts/safety.py"):
            path = PACKAGE_ROOT/relative
            self.assertEqual(files[str(path.resolve())],evidence.digest(path))
        self.assertTrue(all(Path(path).is_file() for path in files))

    def test_deferred_development_relative_imports_resolve_after_relocation(self):
        for path in (PACKAGE_ROOT/"tools/tunnel_static").glob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node,ast.ImportFrom) and node.level and node.module:
                    target = importlib.util.resolve_name("."*node.level+node.module,
                                                         "scenario_runner.tools.tunnel_static")
                    with self.subTest(file=path.name,line=node.lineno,target=target):
                        self.assertIsNotNone(importlib.util.find_spec(target))


if __name__ == "__main__":
    unittest.main()
