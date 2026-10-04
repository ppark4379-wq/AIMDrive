"""Read hashed native files without resolving every parent on every frame.

On POSIX, a small process-local cache pins already resolved native roots by
file descriptor. Every component below that root is opened with O_NOFOLLOW.
Legitimate in-root symlinks take a slower canonical-containment fallback; the
resolved path is then opened with the same no-follow traversal. Callers still
verify the returned bytes against the manifest size and SHA on EVERY read.
"""
import atexit
from collections import OrderedDict
import errno
import os
from pathlib import Path
import threading


_SUPPORTED = (os.name == 'posix' and hasattr(os, 'O_NOFOLLOW')
              and os.open in os.supports_dir_fd)
_ROOT_LIMIT = 128
_ROOTS = OrderedDict()
_LOCK = threading.Lock()
_PID = os.getpid()


def clear_native_root_cache():
    """Release only reader-owned handles (also used by tests/benchmarks)."""
    global _PID, _LOCK
    if os.getpid() != _PID:
        _LOCK = threading.Lock()
        _PID = os.getpid()
    with _LOCK:
        while _ROOTS:
            _, (_, fd) = _ROOTS.popitem(last=False)
            os.close(fd)


atexit.register(clear_native_root_cache)


def _borrow_root(root):
    global _PID
    if os.getpid() != _PID:
        clear_native_root_cache()
    key = os.path.abspath(os.fspath(root))
    with _LOCK:
        if key not in _ROOTS:
            canonical = Path(root).resolve(strict=True)
            fd = os.open(canonical, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            _ROOTS[key] = (canonical, fd)
            if len(_ROOTS) > _ROOT_LIMIT:
                _, (_, expired) = _ROOTS.popitem(last=False)
                os.close(expired)
        canonical, fd = _ROOTS[key]
        _ROOTS.move_to_end(key)
        # A concurrent cache eviction cannot close this reader's own handle.
        return canonical, os.dup(fd)


def _open_beneath(root_fd, parts):
    parent = os.dup(root_fd)
    try:
        for part in parts[:-1]:
            next_fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                              dir_fd=parent)
            os.close(parent)
            parent = next_fd
        return os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent)
    finally:
        os.close(parent)


def read_native_bytes(root, name):
    """Preserve in-root symlink semantics while rejecting relative escapes."""
    rel = Path(name)
    if rel.is_absolute() or '..' in rel.parts or not rel.parts:
        raise ValueError('Unsafe asset path: ' + str(name))
    if not _SUPPORTED:
        canonical = Path(root).resolve()
        path = (canonical / rel).resolve()
        if canonical not in path.parents:
            raise ValueError('Unsafe asset path: ' + str(name))
        return path.read_bytes()
    canonical, root_fd = _borrow_root(root)
    try:
        try:
            fd = _open_beneath(root_fd, rel.parts)
        except OSError as error:
            if error.errno not in (errno.ELOOP, errno.ENOTDIR):
                raise
            # Resolve only when a symlink actually occurs; never cache this
            # relative result, so retargeting cannot bypass containment.
            resolved = (canonical / rel).resolve(strict=True)
            if canonical not in resolved.parents:
                raise ValueError('Unsafe asset path: ' + str(name)) from error
            fd = _open_beneath(root_fd, resolved.relative_to(canonical).parts)
        with os.fdopen(fd, 'rb') as source:
            return source.read()
    finally:
        os.close(root_fd)
