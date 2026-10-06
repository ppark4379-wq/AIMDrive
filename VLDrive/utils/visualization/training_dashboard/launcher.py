"""Fail-open, asynchronous Dashboard startup (stdlib only in the trainer)."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser


def _windows_path(path):
    return subprocess.check_output(['wslpath', '-w', str(path)], text=True, timeout=5).strip()


def _command(root, run_dir, options):
    helper = Path(__file__).resolve()
    windows_python = root / '.runtime/dashboard_venv/Scripts/python.exe'
    native_python = root / '.runtime/dashboard_venv/bin/python'
    wsl = sys.platform == 'linux' and 'microsoft' in os.uname().release.lower()
    if wsl and windows_python.is_file():
        # Explicit interop also works when the WSLInterop binfmt entry is absent.
        # /init expects both the executable path and its Windows argv[0].
        command = ['/init', str(windows_python), _windows_path(windows_python),
                   _windows_path(helper), '--run-dir', _windows_path(run_dir)]
    else:
        python = windows_python if os.name == 'nt' else native_python
        command = [str(python) if python.is_file() else sys.executable, str(helper), '--run-dir', str(run_dir)]
    command += ['--port', str(options.get('port', 8501)),
                '--refresh', str(options.get('refresh_seconds', 2))]
    if options.get('app') == 'situation':
        command += ['--app', 'situation']
    if options.get('open_browser', True):
        command.append('--open-browser')
    return command


def _announce(process, status_path):
    try:
        process.wait(timeout=90)
        state = json.loads(status_path.read_text(encoding='utf-8'))
        if state['state'] == 'ready':
            print('[Dashboard] ' + state['url'], flush=True)
            if state.get('browser_error'):
                print('[Dashboard] Browser did not open: ' + state['browser_error'], flush=True)
        else:
            print('[Dashboard] Unavailable; training continues: ' + state.get('error', 'startup failed'), flush=True)
    except Exception as exc:
        print(f'[Dashboard] Could not confirm startup; training continues: {exc}', flush=True)


def launch_dashboard(run_dir, options=None):
    """Return immediately; neither UI failures nor UI lifetime control training."""
    try:
        options = options or {}
        if not options.get('enabled', True):
            return None
        run_dir = Path(run_dir).resolve()
        root = Path(__file__).resolve().parents[3]
        command = _command(root, run_dir, options)
        env = os.environ.copy()
        # Training-only Python packages must not leak into the isolated UI interpreter.
        for key in ('PYTHONPATH', 'PYTHONHOME', 'VIRTUAL_ENV'):
            env.pop(key, None)
        kwargs = {'creationflags': subprocess.CREATE_NO_WINDOW} if os.name == 'nt' else {'start_new_session': True}
        status_path = run_dir / 'dashboard/status.json'
        status_path.parent.mkdir(parents=True, exist_ok=True)
        with (status_path.parent / 'bootstrap.log').open('ab') as log:
            process = subprocess.Popen(command, cwd=root, env=env, stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=log, **kwargs)
        print(f'[Dashboard] Starting automatically; status: {status_path}', flush=True)
        threading.Thread(target=_announce, args=(process, status_path), daemon=True).start()
        return process
    except Exception as exc:
        print(f'[Dashboard] Could not launch; training continues: {exc}', flush=True)
        return None


def _available_port(preferred):
    with socket.socket() as sock:
        try:
            sock.bind(('127.0.0.1', preferred))
        except OSError:
            sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def _save_status(folder, state):
    pending = folder / 'status.pending.json'
    pending.write_text(json.dumps(state, indent=2) + '\n', encoding='utf-8')
    pending.replace(folder / 'status.json')


def start_server(run_dir, *, port=8501, refresh=2, open_browser=True, app='vldrive'):
    """Run in a disposable helper process on the same OS as the UI Python."""
    folder = Path(run_dir).resolve() / 'dashboard'
    folder.mkdir(parents=True, exist_ok=True)
    server = None
    try:
        if not 0 <= port <= 65535 or not 1 <= refresh <= 5:
            raise ValueError('dashboard port must be 0..65535 and refresh_seconds 1..5')
        if app not in ('vldrive', 'situation'):
            raise ValueError('Unknown dashboard app')
        port = _available_port(port)
        url = f'http://127.0.0.1:{port}'
        app_file = 'situation_app.py' if app == 'situation' else 'app.py'
        command = [sys.executable, '-m', 'streamlit', 'run', str(Path(__file__).with_name(app_file)),
                   '--server.headless=true', '--server.address=127.0.0.1', f'--server.port={port}',
                   '--server.fileWatcherType=none', '--browser.gatherUsageStats=false', '--',
                   '--run-dir', str(Path(run_dir).resolve()), '--refresh', str(refresh)]
        kwargs = ({'creationflags': subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP}
                  if os.name == 'nt' else {'start_new_session': True})
        with (folder / 'server.log').open('ab') as log:
            server = subprocess.Popen(command, cwd=Path(__file__).resolve().parents[3],
                                      stdin=subprocess.DEVNULL, stdout=log, stderr=log, **kwargs)
        state = dict(state='starting', pid=server.pid, url=url, browser_requested=open_browser)
        _save_status(folder, state)
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            if server.poll() is not None:
                raise RuntimeError('Streamlit exited; see dashboard/server.log (check UI dependencies)')
            try:
                with opener.open(url + '/_stcore/health', timeout=1) as response:
                    if response.status == 200:
                        break
            except OSError:
                pass
            time.sleep(.2)
        else:
            raise RuntimeError('Streamlit startup timed out; see dashboard/server.log')
        state['state'] = 'ready'
        if open_browser:
            try:
                if not webbrowser.open(url, new=2):
                    state['browser_error'] = 'No browser available; open the URL manually.'
            except Exception as exc:
                state['browser_error'] = str(exc)
        _save_status(folder, state)
        # Keep/reap the handle in embedded tests; this daemon never owns server lifetime.
        threading.Thread(target=server.wait, daemon=True).start()
        return state
    except Exception as exc:
        if server is not None and server.poll() is None:
            if os.name == 'nt':
                # Windows venv Python can have a child interpreter holding the socket.
                subprocess.run(['taskkill.exe', '/PID', str(server.pid), '/T', '/F'],
                               capture_output=True, timeout=10,
                               creationflags=subprocess.CREATE_NO_WINDOW)
            else:
                server.terminate()
            try:
                server.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server.kill()
        state = dict(state='failed', error=str(exc))
        _save_status(folder, state)
        return state


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--port', type=int, default=8501)
    parser.add_argument('--refresh', type=int, default=2)
    parser.add_argument('--open-browser', action='store_true')
    parser.add_argument('--app', choices=('vldrive', 'situation'), default='vldrive')
    args = parser.parse_args()
    start_server(args.run_dir, port=args.port, refresh=args.refresh, open_browser=args.open_browser, app=args.app)
