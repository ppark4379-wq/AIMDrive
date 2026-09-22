"""Geometry and MGeo helpers used by the K-City scenarios."""

import builtins


def safe_print(*args, **kwargs):
    """A closed console must never prevent actor/process cleanup.

    Only console I/O is best-effort; durable journal/file writes retain their
    own error handling and must not be mistaken for successful recording.
    """
    try:
        builtins.print(*args, **kwargs)
    except (OSError, ValueError):
        pass
