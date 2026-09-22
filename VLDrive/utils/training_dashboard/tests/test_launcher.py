"""Startup isolation, browser handoff, and WSL-to-Windows launch checks; no training."""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
import urllib.request

from utils.training_dashboard import launcher

ROOT = Path(__file__).resolve().parents[3]
WSL = sys.platform == 'linux' and 'microsoft' in os.uname().release.lower()


class LauncherTests(unittest.TestCase):
    def test_disabled_does_not_spawn(self):
        with patch.object(launcher.subprocess, 'Popen') as spawn:
            self.assertIsNone(launcher.launch_dashboard(ROOT, {'enabled': False}))
        spawn.assert_not_called()

    def test_spawn_error_does_not_escape_into_training(self):
        with patch.object(launcher, '_command', side_effect=OSError('UI unavailable')):
            self.assertIsNone(launcher.launch_dashboard(ROOT))

    def test_isolated_environment_and_nonblocking_monitor(self):
        with tempfile.TemporaryDirectory(dir=ROOT / '.runtime') as temp, \
             patch.object(launcher, '_command', return_value=['ui-python']), \
             patch.object(launcher.subprocess, 'Popen') as spawn, \
             patch.object(launcher.threading, 'Thread') as thread, \
             patch.dict(os.environ, {'PYTHONPATH': 'training-packages', 'PYTHONHOME': 'training-home'}):
            self.assertIs(launcher.launch_dashboard(temp), spawn.return_value)
        self.assertNotIn('PYTHONPATH', spawn.call_args.kwargs['env'])
        self.assertNotIn('PYTHONHOME', spawn.call_args.kwargs['env'])
        self.assertTrue(thread.call_args.kwargs['daemon'])
        thread.return_value.start.assert_called_once()

    def test_missing_server_dependency_reports_failure(self):
        with tempfile.TemporaryDirectory(dir=ROOT / '.runtime') as temp, \
             patch.object(launcher.subprocess, 'Popen') as spawn:
            spawn.return_value.pid = 123
            spawn.return_value.poll.return_value = 1
            state = launcher.start_server(temp, port=0, open_browser=False)
            self.assertEqual(state['state'], 'failed')
            self.assertIn('dependencies', state['error'])
            self.assertEqual(json.loads((Path(temp) / 'dashboard/status.json').read_text()), state)

    @unittest.skipUnless(os.name == 'nt', 'Windows UI interpreter integration')
    def test_real_server_port_fallback_and_browser_handoff(self):
        with tempfile.TemporaryDirectory(dir=ROOT / '.runtime') as temp, socket.socket() as occupied:
            occupied.bind(('127.0.0.1', 0))
            occupied.listen()
            preferred = occupied.getsockname()[1]
            state = None
            try:
                with patch.object(launcher.webbrowser, 'open', return_value=True) as browser:
                    state = launcher.start_server(temp, port=preferred)
                self.assertEqual(state['state'], 'ready', state)
                self.assertNotEqual(state['url'], f'http://127.0.0.1:{preferred}')
                browser.assert_called_once_with(state['url'], new=2)
                with urllib.request.urlopen(state['url'] + '/_stcore/health', timeout=5) as response:
                    self.assertEqual(response.status, 200)
            finally:
                if state and state.get('pid'):
                    subprocess.run(['taskkill.exe', '/PID', str(state['pid']), '/T', '/F'],
                                   capture_output=True, check=True, creationflags=subprocess.CREATE_NO_WINDOW)
                    time.sleep(.5)

    @unittest.skipUnless(WSL, 'WSL interop integration')
    def test_wsl_starts_independent_windows_server(self):
        if not (ROOT / '.runtime/dashboard_venv/Scripts/python.exe').is_file():
            self.skipTest('No Windows UI environment')
        with tempfile.TemporaryDirectory(dir=ROOT / '.runtime') as temp:
            status = Path(temp) / 'dashboard/status.json'
            state = {}
            try:
                process = launcher.launch_dashboard(temp, {'open_browser': False, 'port': 0})
                self.assertIsNotNone(process)
                process.wait(timeout=75)
                self.assertTrue(status.exists(), (Path(temp) / 'dashboard/bootstrap.log').read_text())
                state = json.loads(status.read_text())
                self.assertEqual(state['state'], 'ready', state)
                self.assertFalse(state['browser_requested'])
                # Bootstrap has exited; the Windows server remains independently alive.
                python = str(ROOT / '.runtime/dashboard_venv/Scripts/python.exe')
                probe = subprocess.run(['/init', python, launcher._windows_path(Path(python)), '-c',
                    'import sys, urllib.request; print(urllib.request.urlopen(sys.argv[1], timeout=5).status)',
                    state['url'] + '/_stcore/health'], capture_output=True, text=True, timeout=15)
                self.assertEqual(probe.returncode, 0, probe.stderr)
                self.assertIn('200', probe.stdout)
            finally:
                if not state and status.exists():
                    state = json.loads(status.read_text())
                if state.get('pid'):
                    subprocess.run(['/init', '/mnt/c/Windows/System32/taskkill.exe', 'taskkill.exe',
                                    '/PID', str(state['pid']), '/T', '/F'],
                                   capture_output=True, check=True, timeout=15)
                    time.sleep(.5)


if __name__ == '__main__':
    unittest.main()
