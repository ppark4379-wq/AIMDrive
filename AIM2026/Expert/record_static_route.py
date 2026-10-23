"""Compatibility entry point; keep the recorder's original module-global API."""
from pathlib import Path

_implementation = Path(__file__).resolve().parent / "tools/paths/record_route.py"
# Execute in this module so existing callers that override JammingSection or
# register_route affect the functions' globals exactly as before relocation.
__file__ = str(_implementation)
exec(compile(_implementation.read_bytes(), __file__, "exec"), globals())
