"""Real-time terminal visibility without changing durable subprocess output."""

import io
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest

from ..utils.subprocess_log import FileConsoleRelay, show_helper_console_line


class ObservedConsole(io.StringIO):
    def __init__(self, first_line=None):
        super().__init__()
        self.first_line = first_line
        self.visible = threading.Event()

    def write(self, text):
        result = super().write(text)
        if self.first_line is None or self.first_line in self.getvalue():
            self.visible.set()
        return result


class FileConsoleRelayTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="scenario-console-relay-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_appended_lines_visible_before_file_is_finished(self):
        path = self.root / "trial.log"
        old = b"earlier invocation\n"
        path.write_bytes(old)
        console = ObservedConsole("[PLAN] running\n")
        relay = FileConsoleRelay(path, start_offset=len(old), console=console).start()
        try:
            with path.open("ab", buffering=0) as stream:
                stream.write(b"[PLAN] running\n")
                self.assertTrue(console.visible.wait(1.), "First live line was held until the file finished")
                stream.write("[CTRL] 한국어\n".encode("utf-8"))
            self.assertTrue(relay.stop())
        finally:
            relay.stop()
        self.assertEqual(console.getvalue(), "[PLAN] running\n[CTRL] 한국어\n")
        self.assertEqual(path.read_bytes(), old + console.getvalue().encode("utf-8"))

    def test_filter_only_hides_setup_successes_and_keeps_errors_and_expert_lines(self):
        path = self.root / "trial.log"
        text = ("[KCityRunner] loading yaml: /tmp/config.yaml\n"
                "[gRPC] connected to 127.0.0.1:7789\n"
                "[KCityRunner] loading yaml: ERROR missing file\n"
                "[gRPC] connected to endpoint; warning: stale state\n"
                "[PLAN] selected\n[STATIC] active\n[CTRL] steer=1\nACC ON_Vehicle\n"
                "Traceback (most recent call last):\nRuntimeError: spawn failed\n")
        path.write_text(text, encoding="utf-8")
        console = io.StringIO()
        relay = FileConsoleRelay(path, console=console, line_filter=show_helper_console_line).start()
        self.assertTrue(relay.stop())
        self.assertEqual(path.read_text(), text)
        self.assertEqual(console.getvalue(), "".join(text.splitlines(keepends=True)[2:]))
        self.assertIsNone(relay.error)

    def test_partial_utf8_and_final_non_newline_fragment_are_preserved(self):
        path = self.root / "trial.log"
        path.touch()
        console = ObservedConsole("ready\n")
        relay = FileConsoleRelay(path, console=console).start()
        with path.open("ab", buffering=0) as stream:
            stream.write(b"ready\n" + "한".encode("utf-8")[:1])
            self.assertTrue(console.visible.wait(1.))
            stream.write("한".encode("utf-8")[1:] + b" no newline")
        self.assertTrue(relay.stop())
        self.assertEqual(console.getvalue(), "ready\n한 no newline")

    def test_final_write_between_eof_read_and_stop_check_is_drained(self):
        path = self.root / "trial.log"
        path.write_bytes(b"first\n")
        console = io.StringIO()
        relay = FileConsoleRelay(path, console=console)

        class FinishAtEOF:
            def __init__(self):
                self.source = path.open("rb")
                self.finished = False

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                self.source.close()

            def seek(self, offset):
                return self.source.seek(offset)

            def read(self, size):
                chunk = self.source.read(size)
                if not chunk and not self.finished:
                    self.finished = True
                    with path.open("ab", buffering=0) as stream:
                        stream.write(b"final stderr\n")
                    relay._stop.set()
                return chunk

        class ReadFixture:
            def open(self, _mode):
                return FinishAtEOF()

        relay.path = ReadFixture()
        relay.start()
        self.assertTrue(relay.stop())
        self.assertEqual(console.getvalue(), "first\nfinal stderr\n")
        self.assertEqual(path.read_bytes(), console.getvalue().encode("utf-8"))

    def test_broken_console_does_not_change_log_or_raise_into_cleanup(self):
        class BrokenConsole:
            def write(self, _text):
                raise BrokenPipeError("terminal closed")

        path = self.root / "trial.log"
        path.write_text("[CTRL] keep durable\n")
        relay = FileConsoleRelay(path, console=BrokenConsole()).start()
        self.assertTrue(relay.stop())
        self.assertEqual(relay.error, "terminal closed")
        self.assertEqual(path.read_text(), "[CTRL] keep durable\n")

    def test_blocked_console_has_bounded_stop_and_does_not_block_file_writer(self):
        entered, release = threading.Event(), threading.Event()

        class BlockedConsole:
            def write(self, _text):
                entered.set()
                release.wait(3.)

            def flush(self):
                pass

        path = self.root / "trial.log"
        path.write_text("first\n")
        relay = FileConsoleRelay(path, console=BlockedConsole()).start()
        try:
            self.assertTrue(entered.wait(1.))
            path.write_text("first\nmore durable output\n")
            started = time.monotonic()
            self.assertFalse(relay.stop(.05))
            self.assertLess(time.monotonic() - started, .3)
        finally:
            release.set()
            self.assertTrue(relay.stop())
        self.assertEqual(path.read_text(), "first\nmore durable output\n")


@unittest.skipUnless(Path("/proc/self/stat").exists(), "Linux process birth tokens are required")
class SubprocessConsoleIntegrationTests(unittest.TestCase):
    def setUp(self):
        from ..scenarios.full_loop_parts.tunnel_static_parts import runtime
        self.runtime = runtime
        self.temporary = tempfile.TemporaryDirectory(prefix="scenario-console-subprocess-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def worker(self, name, source):
        (self.root / (name + ".py")).write_text(source, encoding="utf-8")
        return [sys.executable, "-u", "-m", name]

    def test_stdout_stderr_are_visible_while_child_running_and_file_matches(self):
        command = self.worker("console_early_fixture", "import sys,time\n"
                              "print('[PLAN] early', flush=True)\n"
                              "print('RuntimeError: retained stderr', file=sys.stderr, flush=True)\n"
                              "time.sleep(1.5)\nprint('[CTRL] final', flush=True)\n")
        console = ObservedConsole("[PLAN] early\n")
        results, errors = [], []
        def run():
            try:
                results.append(self.runtime.bounded_run(command, self.root / "trial.log", 5.,
                    campaign_root=self.root, stage="console_fixture", cwd=self.root,
                    console_output=True, console_stream=console))
            except BaseException as error:
                errors.append(error)
        monitor = threading.Thread(target=run)
        monitor.start()
        try:
            self.assertTrue(console.visible.wait(1.))
            self.assertTrue(monitor.is_alive(), "Output appeared only after the child exited")
        finally:
            monitor.join(6.)
        self.assertFalse(monitor.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(results[0]["returncode"], 0)
        self.assertEqual(console.getvalue(), (self.root / "trial.log").read_text())
        self.assertIn("RuntimeError: retained stderr", console.getvalue())

    def test_interrupt_keeps_cleanup_and_final_child_error_visible(self):
        command = self.worker("console_interrupt_fixture", "import signal,sys,time\n"
                              "def stop(_signum,_frame):\n"
                              "    print('RuntimeError: interrupted safely', file=sys.stderr, flush=True)\n"
                              "    sys.exit(0)\n"
                              "signal.signal(signal.SIGINT, stop)\n"
                              "print('[STATIC] ready', flush=True)\ntime.sleep(20)\n")
        console = ObservedConsole("[STATIC] ready\n")
        result = self.runtime.bounded_run(command, self.root / "trial.log", 5.,
            campaign_root=self.root, stage="console_interrupt", cwd=self.root, grace=.5,
            stop_requested=console.visible.is_set, console_output=True, console_stream=console)
        self.assertTrue(result["interrupted"])
        self.assertFalse(result["forced_kill"])
        self.assertIsNone(self.runtime._raw_identity(result["process_pid"]))
        self.assertEqual(console.getvalue(), (self.root / "trial.log").read_text())
        self.assertIn("interrupted safely", console.getvalue())

    def test_default_background_run_does_not_start_console_relay(self):
        from unittest.mock import patch
        command = self.worker("console_quiet_fixture", "print('[PLAN] background')\n")
        with patch.object(self.runtime, "FileConsoleRelay") as relay:
            result = self.runtime.bounded_run(command, self.root / "trial.log", 5.,
                campaign_root=self.root, stage="quiet_fixture", cwd=self.root)
        relay.assert_not_called()
        self.assertEqual(result["returncode"], 0)
        self.assertEqual((self.root / "trial.log").read_text(), "[PLAN] background\n")


if __name__ == "__main__":
    unittest.main()
