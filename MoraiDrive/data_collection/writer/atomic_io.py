"""Atomic file publication; incomplete writes never replace the destination."""
from contextlib import contextmanager
import json
import os
import tempfile
import numpy as np


@contextmanager
def atomic_path(path):
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".pending-", suffix=os.path.splitext(path)[1], dir=directory)
    os.close(fd)
    try:
        yield temporary
        with open(temporary, "rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json(path, value):
    with atomic_path(path) as temporary:
        with open(temporary, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)


def save_numpy(path, value, **kwargs):
    with atomic_path(path) as temporary:
        with open(temporary, "wb") as stream:
            np.save(stream, value, **kwargs)
