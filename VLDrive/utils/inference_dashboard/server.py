"""Bounded in-memory HTTP display. No model execution or simulator connection."""
from collections import OrderedDict
import copy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
from urllib.parse import parse_qs, urlsplit
import webbrowser


class Dashboard:
    def __init__(self, options, *, model_label='', clock=time.monotonic):
        self.options = options
        self.clock = clock
        self.lock = threading.Lock()
        self.state = dict(status='starting', reason='', history_count=0, history_resets=0,
                          result=None, navigation=None, streams=[], model=model_label,
                          receive_only=True, refresh_hz=options['refresh_hz'])
        self.images = OrderedDict()
        self.plan = dict(points=[], sources={})
        self.server = None
        self.thread = None
        self.url = None

    def update(self, *, clear_prediction=False, **fields):
        with self.lock:
            self.state.update(copy.deepcopy(fields))
            if clear_prediction:
                self.state['result'] = None
                self.images.clear()

    def set_route(self, episode):
        with self.lock:
            points = episode['route']['points']
            # The display can simplify the polyline; the navigation engine cannot.
            stride = max(1, (len(points)+1999)//2000)
            self.plan = dict(points=points[::stride]+[points[-1]], sources=episode['sources'])

    def show_prediction(self, result, jpeg):
        from utils.vision_encoder_visualization import forward_right
        forward, right = forward_right(result['waypoints_raw'])
        value = dict(result, plot_waypoints_xy=[list(map(float, p)) for p in zip(right, forward)])
        key = str(result['frame'])
        with self.lock:
            self.images[key] = bytes(jpeg)
            while len(self.images) > 3:
                self.images.popitem(last=False)
            value['camera_key'] = key
            self.state.update(status='live_waypoints', reason='', result=value)

    def snapshot(self):
        with self.lock:
            state = copy.deepcopy(self.state)
        result = state['result']
        age = None if result is None else max(0., self.clock()-result['anchor_received_at'])
        state['prediction_age_s'] = age
        state['prediction_valid'] = (result is not None and age <= self.options['max_prediction_age_s']
                                     and state['status'] == 'live_waypoints')
        if not state['prediction_valid']:
            state['result'] = None
        return state

    def image(self, key):
        if not self.snapshot()['prediction_valid']:
            return None
        with self.lock:
            return self.images.get(key)

    def start(self):
        if not self.options['enabled']:
            return None
        dashboard = self
        folder = Path(__file__).parent
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                url = urlsplit(self.path)
                if url.path == '/api/state':
                    body = json.dumps(dashboard.snapshot(), allow_nan=False).encode()
                    mime = 'application/json'
                elif url.path == '/api/route':
                    with dashboard.lock:
                        body = json.dumps(dashboard.plan, allow_nan=False).encode()
                    mime = 'application/json'
                elif url.path == '/api/frame.jpg':
                    body = dashboard.image(parse_qs(url.query).get('frame', [''])[0])
                    mime = 'image/jpeg'
                    if body is None:
                        self.send_error(404, 'No fresh matching anchor camera')
                        return
                elif url.path in ('/', '/app.js', '/style.css'):
                    name = 'index.html' if url.path == '/' else url.path[1:]
                    body = (folder/name).read_bytes()
                    mime = {'index.html':'text/html; charset=utf-8','app.js':'text/javascript',
                            'style.css':'text/css'}[name]
                else:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header('Content-Type', mime)
                self.send_header('Content-Length', str(len(body)))
                self.send_header('Cache-Control', 'no-store')
                self.send_header('X-Content-Type-Options', 'nosniff')
                self.end_headers()
                try:
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError):
                    pass
        # Only the browser interface binds TCP loopback. No external control API.
        self.server = ThreadingHTTPServer(('127.0.0.1', self.options['port']), Handler)
        self.server.daemon_threads = True
        self.url = 'http://127.0.0.1:' + str(self.server.server_port)
        self.thread = threading.Thread(target=self.server.serve_forever, name='vldrive-display', daemon=True)
        self.thread.start()
        print('Live visualization: ' + self.url, flush=True)
        if self.options['open_browser']:
            threading.Thread(target=self._open_browser, daemon=True).start()
        return self.url

    def _open_browser(self):
        try:
            root = Path(__file__).resolve().parents[2]
            windows_python = root/'.runtime/dashboard_venv/Scripts/pythonw.exe'
            if sys.platform == 'linux' and windows_python.is_file() and Path('/init').is_file():
                win_path = subprocess.check_output(['wslpath','-w',str(windows_python)],text=True,timeout=5).strip()
                subprocess.Popen(['/init',str(windows_python),win_path,'-c',
                    'import webbrowser; webbrowser.open(' + repr(self.url) + ')'],
                    stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            else:
                webbrowser.open(self.url, new=2)
        except Exception as error:
            print('Open visualization URL manually: ' + self.url + ' (' + str(error) + ')', flush=True)

    def close(self):
        self.update(status='stopped', clear_prediction=True)
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
            self.thread.join(timeout=2)
