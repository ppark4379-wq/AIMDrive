"""Deadline waits immune to backward realtime adjustments on Noetic Python 3.8.

This Python build uses sem_timedwait for Event.wait(timeout), so that API can
oversleep when CLOCK_REALTIME steps backward. time.sleep on Linux Python 3.8
uses a relative select timeout and recomputes interruptions with monotonic time.
"""
import time


def wait_until(stop, deadline_ns, clock=time.monotonic_ns, sleep=time.sleep):
    while not stop.is_set():
        remaining = (deadline_ns - clock()) / 1e9
        if remaining <= 0:
            return False
        # Bound stop response without relying on Event.wait's realtime timeout.
        sleep(min(remaining, .02))
    return True
