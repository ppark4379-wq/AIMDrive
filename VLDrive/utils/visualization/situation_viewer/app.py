"""Local cached-token Situation inference viewer. No sensor/model files are written."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
from urllib.parse import parse_qs, urlparse

import torch

from VLDrive.data.src.loaders.situation_dataset import ROOT, load_config
from VLDrive.training.situation.predict import CachedSituationReview, local_path

PAGE = Path(__file__).with_name('viewer.html')


class Viewer:
    def __init__(self, config, device):
        self.config, self.device = local_path(config), device
        self.review = None
        self.lock = threading.Lock()
        self.state = dict(busy=False, task=None, done=0, total=0, error=None, generation=0)

    def options(self):
        cfg = load_config(self.config)
        checkpoints = sorted((ROOT / cfg['output_root']).glob('*/best_train.pt'),
                             key=lambda p: p.stat().st_mtime, reverse=True)
        return dict(config=str(self.config), cache=str((ROOT / cfg['cache_output']).resolve()),
                    checkpoints=[str(p) for p in checkpoints], device=self.device)

    def status(self):
        with self.lock:
            return dict(self.state, loaded=self.review is not None)

    def get_review(self):
        with self.lock:
            if self.review is None:
                raise ValueError('설정과 Situation checkpoint를 먼저 불러오세요.')
            return self.review

    def start(self, task, body):
        with self.lock:
            if self.state['busy']:
                raise ValueError('현재 작업이 끝난 후 다시 실행하세요.')
            self.state.update(busy=True, task=task, done=0, total=0, error=None)
        def work():
            try:
                if task == 'load':
                    candidate = CachedSituationReview(body['config'], body['checkpoint'],
                                                      body.get('cache'), self.device)
                    with self.lock:
                        self.review = candidate
                        self.state['generation'] += 1
                else:
                    review = self.get_review()
                    episode = int(body['episode'])
                    total = len(review._episode(episode)['rows'])
                    with self.lock:
                        self.state['total'] = total
                    for index in range(total):
                        review.predict(episode, index)
                        with self.lock:
                            self.state['done'] = index + 1
            except Exception as error:
                with self.lock:
                    self.state['error'] = str(error)
            finally:
                with self.lock:
                    self.state['busy'] = False
        threading.Thread(target=work, daemon=True).start()


def make_handler(viewer):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def allowed(self):
            host = self.headers.get('Host', '')
            valid = {f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}'}
            origin = self.headers.get('Origin')
            return host in valid and (not origin or urlparse(origin).netloc == host)

        def send(self, data, mime='application/json; charset=utf-8', status=200):
            if not isinstance(data, bytes):
                data = json.dumps(data, ensure_ascii=False, allow_nan=False).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type', mime)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            if not self.allowed():
                self.send(dict(error='Local same-origin requests only'), status=403)
                return
            try:
                url = urlparse(self.path)
                query = parse_qs(url.query)
                if url.path == '/':
                    self.send(PAGE.read_bytes(), 'text/html; charset=utf-8')
                elif url.path == '/api/options':
                    self.send(viewer.options())
                elif url.path == '/api/status':
                    self.send(viewer.status())
                elif url.path == '/api/session':
                    self.send(viewer.get_review().summary())
                elif url.path in ('/api/episode', '/api/predict', '/api/image'):
                    review = viewer.get_review()
                    episode = int(query['episode'][0])
                    if url.path == '/api/episode':
                        page = dict(review.episode_page(episode))
                        with review.lock:
                            page['predictions'] = {i: v for (e, i), v in review.predictions.items() if e == episode}
                        self.send(page)
                    else:
                        index = int(query['index'][0])
                        if url.path == '/api/image':
                            self.send(review.image(episode, index, query['view'][0]), 'image/jpeg')
                        else:
                            if viewer.status()['busy']:
                                with review.lock:
                                    result = review.predictions.get((episode, index), dict(pending=True))
                            else:
                                result = review.predict(episode, index)
                            self.send(result)
                else:
                    self.send(dict(error='Not found'), status=404)
            except Exception as error:
                self.send(dict(error=str(error)), status=400)

        def do_POST(self):
            if not self.allowed() or self.headers.get('Content-Type', '').split(';')[0] != 'application/json':
                self.send(dict(error='Local JSON requests only'), status=403)
                return
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 32768:
                    raise ValueError('Invalid request size')
                body = json.loads(self.rfile.read(length))
                task = {'/api/load': 'load', '/api/infer_episode': 'infer_episode'}.get(self.path)
                if task is None:
                    self.send(dict(error='Not found'), status=404)
                    return
                viewer.start(task, body)
                self.send(viewer.status(), status=202)
            except Exception as error:
                self.send(dict(error=str(error)), status=400)
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(ROOT / 'training/situation/config.json'))
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--port', type=int, default=8520)
    args = parser.parse_args()
    torch.set_num_threads(2)
    viewer = Viewer(args.config, args.device)
    server = ThreadingHTTPServer(('127.0.0.1', args.port), make_handler(viewer))
    print(f'Situation inference viewer: http://127.0.0.1:{server.server_port}', flush=True)
    print(f'Device: {args.device}; frozen VE cache only; raw data/checkpoints are read-only.', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
