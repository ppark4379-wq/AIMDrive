"""Best-effort failure snapshots; called only when lane projection raises."""
import hashlib
import json
import os
from pathlib import Path
import reprlib
import sys
import tempfile
import traceback
from datetime import datetime, timezone

import numpy as np


def describe(value):
    result = {"type": type(value).__module__ + "." + type(value).__qualname__,
              "object_id": hex(id(value))}
    try:
        if isinstance(value, np.ndarray):
            result.update(shape=list(value.shape), dtype=str(value.dtype), size=int(value.size))
            # Bound diagnostic output even if an input unexpectedly becomes huge.
            result["values_repr"] = reprlib.repr(np.asarray(value).reshape(-1)[:32].tolist())
            result["truncated"] = value.size > 32
        else:
            result["value_repr"] = reprlib.repr(value)
    except Exception as error:
        result["inspection_error"] = type(error).__name__
    return result


def write_project_failure(local_values, source_file):
    """Persist the active exception without casting/mutating planner inputs."""
    error_type, error, tb = sys.exc_info()
    snapshot = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(), "python_executable": sys.executable,
        "python_version": sys.version, "numpy_version": np.__version__,
        "numpy_file": np.__file__, "source_file": str(Path(source_file).resolve()),
        "exception_type": error_type.__name__ if error_type else None,
        "exception": reprlib.repr(error),
        "traceback": "".join(traceback.format_exception(error_type, error, tb)),
        "locals": {},
        "note": "Locals are captured at failure; an unassigned variable may retain a prior iteration value.",
    }
    for name in ("index", "position", "start", "end", "dx", "dy", "length_sq",
                 "ratio", "x", "y", "distance", "segment_length", "best"):
        if name in local_values:
            value = local_values[name]
            snapshot["locals"][name] = describe(value)
            if name in ("position", "start", "end"):
                for axis in ("x", "y"):
                    try:
                        snapshot["locals"][name + "." + axis] = describe(getattr(value, axis))
                    except Exception as exc:
                        snapshot["locals"][name + "." + axis] = {"inspection_error": reprlib.repr(exc)}
        else:
            snapshot["locals"][name] = {"not_assigned": True}
    link = local_values.get("lane_link")
    snapshot["lane_link"] = {"type": describe(link)["type"]}
    for attr in ("id", "points", "cumulative_s"):
        try:
            value = getattr(link, attr)
            snapshot["lane_link"][attr] = describe(value) if attr == "id" else {"length": len(value)}
        except Exception as exc:
            snapshot["lane_link"][attr] = {"inspection_error": reprlib.repr(exc)}
    try:
        snapshot["cumulative_s_at_index"] = describe(link.cumulative_s[local_values["index"]])
    except Exception:
        pass
    try:
        snapshot["source_sha256"] = hashlib.sha256(Path(source_file).read_bytes()).hexdigest()
    except OSError:
        pass
    try:
        snapshot["traceback_frames"] = []
        current = tb
        while current:
            frame = current.tb_frame
            if frame.f_code.co_name in ("project", "__init__", "_projection_xy"):
                values = frame.f_locals
                captured = {
                    "function": frame.f_code.co_name,
                    "file": frame.f_code.co_filename,
                    "line": current.tb_lineno,
                    "lasti": current.tb_lasti,
                    "locals": {name: describe(values[name]) for name in
                               ("self", "x", "y", "s", "distance", "yaw", "best", "lane_link",
                                "point", "coordinates", "coordinate", "value", "values")
                               if name in values},
                    "identity": {},
                }
                if "x" in values:
                    for other in ("self", "best"):
                        if other in values:
                            captured["identity"]["x_is_" + other] = values["x"] is values[other]
                    for other in ("x", "best"):
                        if other in local_values:
                            captured["identity"]["x_is_project_" + other] = values["x"] is local_values[other]
                snapshot["traceback_frames"].append(captured)

            if current.tb_frame.f_code.co_name == "project":
                snapshot["executing_code_file"] = current.tb_frame.f_code.co_filename
                snapshot["executing_line"] = current.tb_lineno
                snapshot["bytecode_sha256"] = hashlib.sha256(current.tb_frame.f_code.co_code).hexdigest()
            current = current.tb_next
    finally:
        # Do not retain exception frames after serializing their descriptions.
        del tb, current
        if "frame" in locals():
            del frame, values
    folder = Path(source_file).resolve().parents[5] / ".runtime" / "diagnostics"
    try:
        folder.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", prefix="lane-project-{}-".format(os.getpid()),
                                         suffix=".json", dir=str(folder), delete=False) as stream:
            json.dump(snapshot, stream, indent=2, ensure_ascii=True)
            destination = stream.name
        print("[ExpertDiagnostic] lane projection failure saved: " + destination, file=sys.stderr, flush=True)
    except Exception as exc:
        print("[ExpertDiagnostic] snapshot write failed: " + reprlib.repr(exc), file=sys.stderr, flush=True)
        print(json.dumps(snapshot, ensure_ascii=True), file=sys.stderr, flush=True)
