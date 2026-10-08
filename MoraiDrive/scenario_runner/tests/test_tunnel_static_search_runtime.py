"""Source freezing and real subprocess ownership for the search coordinator."""

import ast
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

from ..scenarios.full_loop_parts.tunnel_static_parts import runtime


class DurableRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="tunnel-search-runtime-")
        self.root = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)

    def test_durable_json_refuses_corruption_and_preserves_previous_on_invalid_write(self):
        path = self.root / "nested/state.json"
        runtime.atomic_json(path, {"cursor": 4})
        with self.assertRaises(ValueError):
            runtime.atomic_json(path, {"cursor": float("nan")})
        self.assertEqual(runtime.read_json(path), {"cursor": 4})
        self.assertEqual(list(path.parent.glob("*.tmp")), [])
        path.write_text("{interrupted", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "durable JSON"):
            runtime.read_json(path, {})
        self.assertEqual(runtime.read_json(self.root / "absent", {"default": True}), {"default": True})

    def test_source_pins_catch_program_evidence_and_production_changes(self):
        program, evidence = self.root / "program.py", self.root / "measurement.json"
        program.write_text("original program\n")
        evidence.write_text('{"measured": 2}')
        with patch("scenario_runner.scenarios.full_loop_parts.tunnel_static_parts.evidence.source_snapshot",
                   return_value={"fixed": "production"}) as production:
            pins = runtime.source_snapshot([program], [evidence])
            runtime.require_sources(pins)
            evidence.write_text('{"measured": 3}')
            with self.assertRaisesRegex(RuntimeError, "evidence changed"):
                runtime.require_sources(pins)
            evidence.write_text('{"measured": 2}')
            program.write_text("changed program\n")
            with self.assertRaisesRegex(RuntimeError, "evidence changed"):
                runtime.require_sources(pins)
            program.write_text("original program\n")
            production.return_value = {"fixed": "changed"}
            with self.assertRaisesRegex(RuntimeError, "sources/evidence changed"):
                runtime.require_sources(pins)
        self.assertIn(str(evidence), pins["files"])
        self.assertIn(str(program), pins["files"])

    def test_new_run_pins_console_relay_without_refreshing_old_evidence(self):
        relay_source = self.root / "relay.py"
        relay_source.write_text("original log relay\n")
        with patch("scenario_runner.scenarios.full_loop_parts.tunnel_static_parts.evidence.source_snapshot",
                   return_value={"fixed": "production"}), \
                patch.object(runtime.subprocess_log, "__file__", str(relay_source)):
            pins = runtime.source_snapshot()
            self.assertIn(str(relay_source), pins["files"])
            runtime.require_sources(pins)
            relay_source.write_text("changed log relay\n")
            with self.assertRaisesRegex(RuntimeError, "sources/evidence changed"):
                runtime.require_sources(pins)

    def test_shared_lock_rejects_nested_roots_and_releases_after_exception(self):
        lock = self.root / ".batch_execution.lock"
        with runtime.ExecutionLock(lock):
            with self.assertRaisesRegex(RuntimeError, "shared execution lock"):
                with runtime.ExecutionLock(lock):
                    self.fail("A second live coordinator acquired the same lock")
        with self.assertRaisesRegex(ValueError, "fixture"):
            with runtime.ExecutionLock(lock):
                raise ValueError("fixture")
        with runtime.ExecutionLock(lock):
            pass
        self.assertEqual(runtime.GLOBAL_EXECUTION_LOCK,
                         runtime.PACKAGE_ROOT / ".runtime/tunnel_static_trials/.batch_execution.lock")

    def fake_process(self, *, pid=123, ticks="456", root=None, argv=None, pgid=None):
        proc = self.root / "proc"
        node = proc / str(pid)
        node.mkdir(parents=True, exist_ok=True)
        argv = argv or ["/usr/bin/python3", "-u", "-m", "scenario_runner.tools.run_tunnel_static_trial"]
        (node / "cmdline").write_bytes(b"\0".join(os.fsencode(value) for value in argv) + b"\0")
        (node / "environ").write_bytes(os.fsencode(runtime.ROOT_ENV + "=" + str(root or self.root)) + b"\0")
        (node / "stat").write_text(str(pid) + " (python fixture) " +
                                   " ".join(["S", "1", str(pgid or pid)] + ["0"] * 16 + [ticks]))
        return proc

    def test_every_signal_requires_birth_root_entrypoint_and_own_process_group(self):
        module = "scenario_runner.tools.run_tunnel_static_trial"
        command = ["bash", "run_tunnel_static_trial.sh"]
        proc = self.fake_process()
        identity = runtime.process_identity(123, self.root, module, launch_command=command, proc_root=proc)
        kill = Mock()
        self.assertTrue(runtime.signal_owned_group(identity, signal.SIGINT, proc_root=proc, killpg=kill))
        kill.assert_called_once_with(123, signal.SIGINT)
        for change in ({"ticks": "457"}, {"root": self.root / "foreign"},
                       {"argv": ["python3", "-m", "foreign_worker"]}, {"pgid": 999}):
            self.fake_process(**change)
            kill.reset_mock()
            with self.assertRaises(RuntimeError):
                runtime.signal_owned_group(identity, signal.SIGKILL, proc_root=proc, killpg=kill)
            kill.assert_not_called()

    def test_wrapper_exec_to_expected_module_preserves_same_birth(self):
        command = ["bash", "/tmp/run_tunnel_static_trial.sh", "--fixture", "one"]
        proc = self.fake_process(argv=command)
        identity = runtime.process_identity(123, self.root, "scenario_runner.tools.run_tunnel_static_trial",
                                            launch_command=command, proc_root=proc)
        self.fake_process()
        kill = Mock()
        self.assertTrue(runtime.signal_owned_group(identity, signal.SIGINT, proc_root=proc, killpg=kill))
        kill.assert_called_once()

    def test_runtime_has_no_sdk_ros_or_model_api_calls(self):
        tree = ast.parse(Path(runtime.__file__).read_text())
        imports = [node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
        imports += [entry.name for node in ast.walk(tree) if isinstance(node, ast.Import) for entry in node.names]
        self.assertFalse(any(any(word in name.lower() for word in ("grpc", "rospy", "openai", "requests")) for name in imports))
        self.assertFalse(any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                             and node.func.id in ("eval", "exec") for node in ast.walk(tree)))


@unittest.skipUnless(Path("/proc/self/stat").exists(), "Linux process birth tokens are required")
class RealSubprocessTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="tunnel-search-subprocess-")
        self.root = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)
        self.processes = []
        self.addCleanup(self.cleanup_processes)

    def cleanup_processes(self):
        for process in self.processes:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=5.)

    def worker(self, name, source):
        path = self.root / (name + ".py")
        path.write_text(source, encoding="utf-8")
        return [sys.executable, "-u", "-m", name]

    def test_timeout_kills_owned_group_and_preserves_unrelated_process(self):
        foreign = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(20)"], start_new_session=True)
        self.processes.append(foreign)
        command = self.worker("owned_timeout_fixture", "import os, signal, subprocess, sys, time\n"
                              "signal.signal(signal.SIGINT, signal.SIG_IGN)\n"
                              "child = subprocess.Popen([sys.executable, '-c', 'import signal,time; signal.signal(signal.SIGINT,signal.SIG_IGN); time.sleep(20)'])\n"
                              "open('child.pid', 'w').write(str(child.pid))\n"
                              "print(os.environ['AIMDRIVE_TUNNEL_BATCH_ROOT'], flush=True)\n"
                              "time.sleep(20)\n")
        starts = []
        result = runtime.bounded_run(command, self.root / "worker.log", .8, campaign_root=self.root,
                                     stage="offline_fixture", grace=.2, cwd=self.root, on_start=starts.append)
        self.assertTrue(result["timed_out"])
        self.assertTrue(result["forced_kill"])
        self.assertEqual(result["returncode"], -signal.SIGKILL)
        self.assertEqual(len(starts), 1)
        self.assertIsNotNone(starts[0]["process_identity"])
        self.assertEqual(starts[0]["process_identity"], result["process_identity"])
        self.assertIsNone(foreign.poll())
        child = int((self.root / "child.pid").read_text())
        deadline = time.monotonic() + 3.
        while time.monotonic() < deadline and runtime._raw_identity(child) is not None:
            time.sleep(.05)
        self.assertIsNone(runtime._raw_identity(child), "Owned group descendant survived shutdown")
        self.assertFalse((self.root / "__pycache__").exists())

    def test_interrupt_callback_and_checkpoint_are_recorded(self):
        command = self.worker("owned_interrupt_fixture", "import time\ntime.sleep(20)\n")
        checkpoint = self.root / "inflight.json"
        result = runtime.bounded_run(command, self.root / "worker.log", 10., campaign_root=self.root,
                                     stage="trial_fixture", stop_requested=lambda: checkpoint.exists(),
                                     cwd=self.root, grace=.3,
                                     on_start=lambda fields: runtime.atomic_json(checkpoint, fields))
        self.assertTrue(result["interrupted"])
        self.assertFalse(result["timed_out"])
        self.assertEqual(runtime.read_json(checkpoint)["process_identity"], result["process_identity"])
        self.assertIsNone(runtime._raw_identity(result["process_pid"]))

    def test_wrapper_exec_is_proven_with_real_module_handoff(self):
        self.worker("owned_exec_fixture", "import signal,time\nsignal.signal(signal.SIGINT, signal.SIG_IGN)\ntime.sleep(20)\n")
        wrapper = self.root / "launcher.sh"
        wrapper.write_text("#!/usr/bin/env bash\nexec " + sys.executable + " -u -m owned_exec_fixture\n")
        result = runtime.bounded_run(["bash", str(wrapper)], self.root / "wrapper.log", .5,
                                     campaign_root=self.root, stage="wrapper_fixture", grace=.2,
                                     expected_module="owned_exec_fixture", cwd=self.root)
        self.assertTrue(result["timed_out"])
        self.assertTrue(result["forced_kill"])
        self.assertIsNone(runtime._raw_identity(result["process_pid"]))


if __name__ == "__main__":
    unittest.main()
