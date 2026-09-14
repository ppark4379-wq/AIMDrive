"""Process-wide advisory locks shared by local capture/training entry points."""
from contextlib import contextmanager
import fcntl
from pathlib import Path


@contextmanager
def process_lock(kind):
    if kind not in ('capture', 'training', 'campaign'):
        raise ValueError('Unknown resource')
    # Keep locks in the configured campaign namespace so different checkouts
    # cannot acquire the same simulator/capture resource concurrently.
    from ssd_storage import SSD_ROOT
    path = SSD_ROOT / 'sensor_record_auto_v1' / '.locks' / (
        'collector_' + kind + '.lock'
    )
    # A clean checkout has neither ``sensor_record_auto_v1`` nor ``.locks``.
    # Direct entry points must therefore be as safe as the public shell
    # wrappers and create the complete ignored runtime hierarchy.
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another local ' + kind + ' process owns ' + str(path))
        yield
