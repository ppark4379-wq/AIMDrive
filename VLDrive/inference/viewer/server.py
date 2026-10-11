"""Read saved replay outputs on localhost; viewing never allocates a GPU."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import threading
from urllib.parse import parse_qs, urlparse

from PIL import Image, ImageOps

from ..dataset import ROOT, child, read_json


class ReplayView:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.lock = threading.RLock()
        self._episodes = {}

    def summary(self):
        return read_json(self.root / 'run.json')

    def episode(self, index):
        episodes = self.summary()['episodes']
        if type(index) is not int or not 0 <= index < len(episodes):
            raise ValueError('Unknown selected drive')
        path = self.root / 'episodes' / f'{index:03d}.json'
        if not path.exists():
            return dict(pending=True, **episodes[index])
        with self.lock:
            signature = path.stat().st_mtime_ns
            cached = self._episodes.get(index)
            if cached is None or cached[0] != signature:
                cached = (signature, read_json(path))
                self._episodes[index] = cached
            return cached[1]

    def predictions(self, index):
        self.episode(index)
        path = self.root / 'predictions' / f'{index:03d}.jsonl'
        if not path.exists():
            return []
        result = []
        with path.open(encoding='utf-8') as stream:
            for line in stream:
                # The last record may still be being written by the worker.
                if not line.endswith('\n'):
                    break
                result.append(json.loads(line))
        return result

    def image(self, index, position, role):
        episode = self.episode(index)
        if role not in ('camera_1', 'camera_2', 'camera_3', 'camera_4'):
            raise ValueError('Unknown camera')
        if not 0 <= position < len(episode.get('frames', [])):
            raise ValueError('Unknown frame')
        name = episode['frames'][position].get('images', {}).get(role)
        if not name:
            raise FileNotFoundError('Camera not available')
        path = child(ROOT / 'data/raw', name)
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image).convert('RGB')
            image.thumbnail((1280, 720))
            stream = io.BytesIO()
            image.save(stream, format='JPEG', quality=85)
            return stream.getvalue()


def serve(root, port=8503):
    view = ReplayView(root)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            try:
                parsed = urlparse(self.path)
                query = parse_qs(parsed.query)
                integer = lambda key: int(query[key][0])
                if parsed.path == '/':
                    content = Path(__file__).with_name('index.html').read_bytes()
                    content_type = 'text/html; charset=utf-8'
                elif parsed.path == '/api/run':
                    content = view.summary()
                    content_type = 'application/json; charset=utf-8'
                elif parsed.path == '/api/episode':
                    content = view.episode(integer('episode'))
                    content_type = 'application/json; charset=utf-8'
                elif parsed.path == '/api/predictions':
                    content = view.predictions(integer('episode'))
                    content_type = 'application/json; charset=utf-8'
                elif parsed.path == '/image':
                    content = view.image(integer('episode'), integer('frame'), query['role'][0])
                    content_type = 'image/jpeg'
                else:
                    self.send_error(404)
                    return
                if not isinstance(content, bytes):
                    content = json.dumps(content, ensure_ascii=False, allow_nan=False).encode('utf-8')
                self.send_response(200)
                self.send_header('Content-Type', content_type)
                self.send_header('Content-Length', str(len(content)))
                self.send_header('Cache-Control', 'no-store')
                self.send_header('X-Content-Type-Options', 'nosniff')
                self.end_headers()
                self.wfile.write(content)
            except (KeyError, ValueError, IndexError):
                self.send_error(400, 'Invalid replay request')
            except FileNotFoundError:
                self.send_error(404, 'Replay artifact not yet available')

    server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    print(f'Offline replay viewer: http://localhost:{server.server_port}', flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--port', type=int, default=8503)
    args = parser.parse_args()
    serve(args.run, args.port)


if __name__ == '__main__':
    main()
