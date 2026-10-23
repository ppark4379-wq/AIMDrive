"""Verify Windows script transport without running PowerShell or native UI."""
import base64
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock


INTERNAL = Path(__file__).resolve().parents[1] / "internal_sensor_record"
sys.path.insert(0, str(INTERNAL))
import nas_storage


class PowerShellTransportTests(unittest.TestCase):
    def capture_call(self, source, *, timeout=30, stdout=b"", stderr=b"", returncode=0):
        completed = subprocess.CompletedProcess([], returncode, stdout, stderr)
        with mock.patch.object(nas_storage.subprocess, "run", return_value=completed) as run:
            result = nas_storage.powershell(source, timeout=timeout)
        return result, run.call_args

    def assert_source_transport(self, source, call):
        command = call.args[0]
        self.assertEqual(command[:4], [
            str(nas_storage.POWERSHELL_EXE), "-NoProfile", "-NonInteractive", "-EncodedCommand",
        ])
        self.assertLess(len(subprocess.list2cmdline(command)) + 1, 32767)
        loader = base64.b64decode(command[4], validate=True).decode("utf-16le")
        self.assertIn("[Console]::In.ReadToEnd()", loader)
        self.assertIn("[Convert]::FromBase64String", loader)
        self.assertIn("[Text.Encoding]::Unicode.GetString", loader)
        self.assertTrue(loader.startswith("& ([ScriptBlock]::Create("))
        payload = call.kwargs["input"]
        self.assertIsInstance(payload, bytes)
        payload.decode("ascii")
        self.assertEqual(base64.b64decode(payload, validate=True).decode("utf-16le"), source)
        self.assertTrue(call.kwargs["capture_output"])
        self.assertNotIn("text", call.kwargs)
        self.assertNotIn("encoding", call.kwargs)

    def test_current_bounded_sensor_record_exceeds_old_limit_but_uses_short_launcher(self):
        source = "& {\n" + (INTERNAL / "tools/bounded_sensor_record.ps1").read_text() + "\n}"
        old_payload = base64.b64encode(source.encode("utf-16le")).decode("ascii")
        self.assertGreater(len(old_payload), 32767)
        result, call = self.capture_call(source, timeout=3010)
        self.assertEqual(result, "")
        self.assertEqual(call.kwargs["timeout"], 3010)
        self.assert_source_transport(source, call)

    def test_large_unicode_and_powershell_syntax_roundtrip_with_fixed_command(self):
        source = (
            "$text=@'\r\n한글 🚦 ' \" $() ; ` { }\r\n'@\r\n"
            "Write-Output $text\r\n# " + "센서🚗'\"$()`{};" * 15000
        )
        _, large_call = self.capture_call(source)
        _, short_call = self.capture_call("Write-Output 'short'")
        self.assertGreater(len(large_call.kwargs["input"]), 32767)
        self.assertEqual(large_call.args[0], short_call.args[0])
        self.assert_source_transport(source, large_call)

    def test_short_script_preserves_utf8_bom_json_output(self):
        expected = {"status": "PASS", "message": "센서 연결 정상"}
        stdout = b"\xef\xbb\xbf" + json.dumps(expected, ensure_ascii=False).encode("utf-8") + b"\r\n"
        result, call = self.capture_call("@{status='PASS'}|ConvertTo-Json", stdout=stdout)
        self.assertEqual(json.loads(result), expected)
        self.assertEqual(call.kwargs["timeout"], 30)
        self.assert_source_transport("@{status='PASS'}|ConvertTo-Json", call)

    def test_empty_source_and_stdout_are_supported(self):
        result, call = self.capture_call("")
        self.assertEqual(result, "")
        self.assert_source_transport("", call)

    def test_nonzero_exit_preserves_windows_error_message(self):
        completed = subprocess.CompletedProcess([], 1, b"partial output", b"failure\xff\n")
        with mock.patch.object(nas_storage.subprocess, "run", return_value=completed):
            with self.assertRaises(RuntimeError) as raised:
                nas_storage.powershell("throw 'failure'")
        self.assertEqual(str(raised.exception), "Windows operation failed: failure\ufffd\n")

    def test_oserror_preserves_interop_recovery_message_and_cause(self):
        error = OSError("interop unavailable")
        with mock.patch.object(nas_storage.subprocess, "run", side_effect=error):
            with self.assertRaises(RuntimeError) as raised:
                nas_storage.powershell("Write-Output 'test'")
        self.assertEqual(str(raised.exception), (
            "WSL cannot execute powershell.exe. Close WSL/VS Code, run "
            "`wsl --shutdown` in Windows PowerShell or Command Prompt, then "
            "reopen WSL."
        ))
        self.assertIs(raised.exception.__cause__, error)

    def test_timeout_is_forwarded_and_propagated(self):
        error = subprocess.TimeoutExpired("powershell.exe", 12, output=b"partial")
        with mock.patch.object(nas_storage.subprocess, "run", side_effect=error) as run:
            with self.assertRaises(subprocess.TimeoutExpired) as raised:
                nas_storage.powershell("Start-Sleep -Seconds 30", timeout=12)
        self.assertIs(raised.exception, error)
        self.assertEqual(run.call_args.kwargs["timeout"], 12)


if __name__ == "__main__":
    unittest.main()
