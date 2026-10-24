"""1 ms Windows timer resolution for the control and perception loops.

The Windows default is 15.6 ms, so ``time.sleep(0.033)`` returns after 31 or
47 ms and a 30 Hz loop runs at ~26 Hz with alternating intervals; a 2 ms poll
sleeps 15.6 ms. Since Windows 10 2004 the request is per process, so each
process that paces a loop must ask. No effect on other platforms.
"""
import contextlib
import sys


@contextlib.contextmanager
def high_resolution_timer(period_ms=1):
    winmm = None
    if sys.platform == 'win32':
        try:
            import ctypes
            winmm = ctypes.WinDLL('winmm')
            if winmm.timeBeginPeriod(period_ms) != 0:  # TIMERR_NOERROR
                winmm = None
        except OSError:
            winmm = None
    try:
        yield winmm is not None
    finally:
        if winmm is not None:
            winmm.timeEndPeriod(period_ms)
