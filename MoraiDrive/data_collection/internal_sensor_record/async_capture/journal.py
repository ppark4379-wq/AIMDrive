"""Bounded single-writer raw journal. No ROS, decoding, NAS, or simulator calls.

Each record is length-delimited JSON metadata and the exact received payload.
Closed chunks and a final manifest are hashed. A failed/partial run is never a
completed run. Capture completeness covers callbacks admitted here, not packets
that never reached the process. Source timestamps are never rewritten.
"""
from collections import Counter, deque
import hashlib
import json
import os
from pathlib import Path
import struct
import threading
import time
import uuid

SCHEMA = 'morai_async_raw_v1'
MAGIC = b'MORAIR01'
PREFIX = struct.Struct('<IQ')
MAX_PAYLOAD = 128 * 1024**2


def json_bytes(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(',', ':')).encode('utf-8')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value, *, dir_fd=None):
    fd=os.open(os.fspath(path),os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644,dir_fd=dir_fd)
    with os.fdopen(fd,'wb') as f:
        f.write(json_bytes(value) + b'\n')
        f.flush()
        os.fsync(f.fileno())


class RawWriter:
    def __init__(self, root, session, *, queue_bytes=256 * 1024**2,
                 queue_records=4096, chunk_bytes=128 * 1024**2,
                 max_bytes=20 * 1024**3, guard=lambda: None):
        self.root = Path(root)
        self.guard = guard
        self.guard()
        if min(queue_bytes, queue_records, chunk_bytes, max_bytes) <= 0:
            raise ValueError('Storage limits must be positive')
        self.root.mkdir(parents=True, exist_ok=False)
        (self.root / 'chunks').mkdir()
        # Pinned directory handles cannot fall through into the WSL mountpoint
        # directory if a drive is unmounted between health checks.
        self.root_fd=os.open(str(self.root),os.O_RDONLY|os.O_DIRECTORY)
        self.chunks_fd=os.open('chunks',os.O_RDONLY|os.O_DIRECTORY,dir_fd=self.root_fd)
        self.session = dict(session, schema=SCHEMA, run_id=uuid.uuid4().hex,
            storage='raw_serialized_messages', training_visible=False,
            clock_policy='source/receive_wall/receive_monotonic are distinct',
            physical_simultaneity_verified=False)
        write_json('session.json', self.session, dir_fd=self.root_fd)
        self.queue_bytes_limit, self.queue_records_limit = queue_bytes, queue_records
        self.chunk_limit, self.max_bytes = chunk_bytes, max_bytes
        self.cv, self.queue = threading.Condition(), deque()
        self.queued_bytes = self.peak_queue_bytes = self.peak_queue_records = 0
        self.received, self.written, self.rejected = Counter(), Counter(), Counter()
        self.total_bytes = self.record_count = self.admitted = 0
        self.max_queue_wait_ns = 0
        self.chunks, self.errors = [], []
        self.closing = False
        self.file = None
        self.started = time.monotonic_ns()
        self.thread = threading.Thread(target=self._run, name='raw-disk-writer', daemon=True)
        self.thread.start()

    def submit(self, meta, payload):
        if not isinstance(payload, bytes):
            payload = bytes(payload)
        size = len(payload) + len(json_bytes(meta)) + 512
        with self.cv:
            role = meta['role']
            self.received[role] += 1
            if (self.errors or self.closing or len(payload) > MAX_PAYLOAD or
                    self.queued_bytes + size > self.queue_bytes_limit or
                    len(self.queue) >= self.queue_records_limit):
                self.rejected[role] += 1
                if not self.errors:
                    self.errors.append('raw_queue_limit_or_closed')
                return False
            row = dict(meta, record_id=self.admitted,
                       enqueue_monotonic_ns=time.monotonic_ns())
            self.queue.append((row, payload, size))
            self.admitted += 1
            self.queued_bytes += size
            self.peak_queue_bytes = max(self.peak_queue_bytes, self.queued_bytes)
            self.peak_queue_records = max(self.peak_queue_records, len(self.queue))
            self.cv.notify()
            return True

    def _open_chunk(self):
        self.guard()
        name = 'chunks/%06d.raw' % len(self.chunks)
        fd=os.open(Path(name).name,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644,dir_fd=self.chunks_fd)
        self.file = os.fdopen(fd,'wb',buffering=1024**2)
        self.chunk = dict(path=name, bytes=0, records=0, first_record=self.record_count)
        self.hash = hashlib.sha256()
        self._bytes(MAGIC)

    def _bytes(self, value):
        self.file.write(value)
        self.hash.update(value)
        self.chunk['bytes'] += len(value)
        self.total_bytes += len(value)

    def _close_chunk(self):
        if self.file is None:
            return
        self.file.flush()
        os.fsync(self.file.fileno())
        self.file.close()
        self.file = None
        os.fsync(self.chunks_fd)
        self.chunks.append(dict(self.chunk, sha256=self.hash.hexdigest()))

    def _run(self):
        last_flush = time.monotonic()
        try:
            while True:
                with self.cv:
                    self.cv.wait_for(lambda: self.queue or self.closing)
                    if not self.queue:
                        break
                    row, payload, charged = self.queue.popleft()
                wait = time.monotonic_ns() - row['enqueue_monotonic_ns']
                row['writer_queue_wait_ns'] = wait
                metadata = json_bytes(row)
                size = PREFIX.size + len(metadata) + len(payload)
                if self.total_bytes + size + len(MAGIC) > self.max_bytes:
                    raise OSError('raw_run_byte_limit')
                if self.file and self.chunk['bytes'] + size > self.chunk_limit:
                    self._close_chunk()
                if self.file is None:
                    self._open_chunk()
                self._bytes(PREFIX.pack(len(metadata), len(payload)))
                self._bytes(metadata)
                self._bytes(payload)
                self.chunk['records'] += 1
                with self.cv:
                    self.queued_bytes -= charged  # includes the in-flight buffer
                    self.written[row['role']] += 1
                    self.record_count += 1
                    self.max_queue_wait_ns = max(self.max_queue_wait_ns, wait)
                if time.monotonic() - last_flush >= 1:
                    self.file.flush()
                    os.fsync(self.file.fileno())
                    last_flush = time.monotonic()
            self._close_chunk()
        except BaseException as exc:
            with self.cv:
                self.errors.append(type(exc).__name__ + ': ' + str(exc))
            if self.file:
                self.file.close()
                self.file = None

    def status(self):
        with self.cv:
            return dict(received=dict(self.received), written=dict(self.written),
                rejected=dict(self.rejected), records=self.record_count, bytes=self.total_bytes,
                queued_bytes=self.queued_bytes, peak_queue_bytes=self.peak_queue_bytes,
                peak_queue_records=self.peak_queue_records, max_queue_wait_ns=self.max_queue_wait_ns,
                errors=list(self.errors), writer_alive=self.thread.is_alive())

    def close(self, success=True, reason='duration_complete', extra=None):
        with self.cv:
            self.closing = True
            self.cv.notify_all()
        self.thread.join(timeout=30)
        if self.thread.is_alive():
            raise TimeoutError('Raw writer did not drain; leave run unsealed')
        stats = self.status()
        ok = bool(success and not stats['errors'] and self.record_count and
                  self.received == self.written and not self.rejected)
        manifest = dict(schema=SCHEMA, run_id=self.session['run_id'],
            status='completed' if ok else 'failed', reason=reason,
            session_sha256=hashlib.sha256(json_bytes(self.session)+b'\n').hexdigest(), chunks=self.chunks,
            statistics=stats, extra=extra or {}, wall_duration_ns=time.monotonic_ns()-self.started,
            training_visible=False, loss_accounting_scope='callbacks admitted in this process')
        self.guard()
        write_json('raw_manifest.json', manifest, dir_fd=self.root_fd)
        write_json('RAW_COMPLETE.json', dict(schema=SCHEMA, status=manifest['status'],
            manifest_sha256=hashlib.sha256(json_bytes(manifest)+b'\n').hexdigest(), training_visible=False),dir_fd=self.root_fd)
        os.fsync(self.root_fd)
        os.close(self.chunks_fd)
        os.close(self.root_fd)
        return manifest


def verify_run(root, *, require_success=True, payloads=True):
    root = Path(root)
    seal = json.loads((root / 'RAW_COMPLETE.json').read_text())
    manifest = json.loads((root / 'raw_manifest.json').read_text())
    session = json.loads((root / 'session.json').read_text())
    if seal.get('schema') != SCHEMA or manifest.get('schema') != SCHEMA or session.get('schema') != SCHEMA:
        raise ValueError('Unknown raw schema')
    if require_success and (seal.get('status') != 'completed' or manifest.get('status') != 'completed'):
        raise ValueError('Failed raw run cannot be built')
    if seal['manifest_sha256'] != digest(root / 'raw_manifest.json') or manifest['session_sha256'] != digest(root / 'session.json'):
        raise ValueError('Raw metadata hash mismatch')
    if manifest['run_id'] != session['run_id']:
        raise ValueError('Mixed raw session')
    total = 0
    for i, chunk in enumerate(manifest['chunks']):
        if chunk['path'] != 'chunks/%06d.raw' % i or chunk['first_record'] != total:
            raise ValueError('Raw chunk sequence mismatch')
        path = root / chunk['path']
        if path.is_symlink() or path.stat().st_size != chunk['bytes']:
            raise ValueError('Raw chunk missing/size mismatch')
        if payloads and digest(path) != chunk['sha256']:
            raise ValueError('Raw chunk hash mismatch')
        total += chunk['records']
    if set(p.name for p in (root/'chunks').iterdir()) != {Path(c['path']).name for c in manifest['chunks']}:
        raise ValueError('Unsealed/orphan chunk')
    stats = manifest['statistics']
    if total != stats['records'] or (require_success and (stats['received'] != stats['written'] or stats['errors'] or stats['rejected'])):
        raise ValueError('Incomplete raw accounting')
    return manifest, session


def records(root, *, verified=None):
    manifest, session = verified or verify_run(root)
    expected = 0
    for chunk in manifest['chunks']:
        n = 0
        with (Path(root) / chunk['path']).open('rb') as f:
            if f.read(len(MAGIC)) != MAGIC:
                raise ValueError('Raw chunk magic differs')
            while True:
                prefix = f.read(PREFIX.size)
                if not prefix:
                    break
                if len(prefix) != PREFIX.size:
                    raise ValueError('Truncated record prefix')
                nm, np = PREFIX.unpack(prefix)
                if not 0 < nm <= 1024**2 or np > MAX_PAYLOAD:
                    raise ValueError('Invalid raw record length')
                mb, payload = f.read(nm), f.read(np)
                if len(mb) != nm or len(payload) != np:
                    raise ValueError('Truncated raw record')
                meta = json.loads(mb)
                if meta['record_id'] != expected:
                    raise ValueError('Record sequence mismatch')
                yield meta, payload
                expected += 1
                n += 1
        if n != chunk['records']:
            raise ValueError('Chunk record count mismatch')
