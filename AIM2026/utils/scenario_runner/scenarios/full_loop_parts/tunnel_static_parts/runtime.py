"""Durable files and bounded, identity-checked subprocesses for tunnel searches.

This module never imports or contacts the simulator, ROS, or a model API. The
caller holds the shared execution lock only during its live trial stage.
"""

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time

from ....utils import subprocess_log
from ....utils.subprocess_log import FileConsoleRelay, show_helper_console_line
from scenario_runner.project_paths import evidence_path


PACKAGE_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = PACKAGE_ROOT.parent
GLOBAL_EXECUTION_LOCK = PACKAGE_ROOT / ".runtime/tunnel_static_trials/.batch_execution.lock"
ROOT_ENV = "AIMDRIVE_TUNNEL_BATCH_ROOT"


def atomic_json(path, document):
    """Replace one durable JSON checkpoint, including flushing its directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
        directory = os.open(str(path.parent), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        try:
            os.unlink(name)
        except FileNotFoundError:
            pass


def read_json(path, default=None):
    """An absent checkpoint has a default; an existing corrupt one fails closed."""
    try:
        return json.loads(evidence_path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return copy.deepcopy(default)
    except (OSError, ValueError) as error:
        raise RuntimeError("Cannot read durable JSON checkpoint: " + str(path)) from error


def digest(path):
    with evidence_path(path).open("rb") as stream:
        result = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
        return result.hexdigest()


def fingerprint(document):
    return hashlib.sha256(json.dumps(document, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def source_snapshot(additional_files=(), evidence_paths=()):
    """Pin production handoff files and every supplied campaign/evidence file.

    Production's existing snapshot includes planner source and recorded routes,
    SDK handoff, trial/startup, model loaders, configuration and safety helpers.
    The caller supplies new search modules, acceptance, runtime catalog and all
    referenced preview/measurement files through the two explicit collections.
    """
    from .evidence import source_snapshot as production_snapshot
    safety = PACKAGE_ROOT / "tools/tunnel_static_batch_safety.sh"
    paths = [Path(__file__), Path(subprocess_log.__file__),
             *map(Path, additional_files), *map(Path, evidence_paths)]
    return {"schema": "aimdrive_tunnel_search_source_pins_v1",
            "production": production_snapshot({}, safety, safety),
            "files": {str(path.resolve()): digest(path) for path in sorted(set(paths), key=str)}}


def require_sources(snapshot):
    """Verify frozen pins without replacing or refreshing any expected digest."""
    if not isinstance(snapshot, dict) or snapshot.get("schema") != "aimdrive_tunnel_search_source_pins_v1":
        raise RuntimeError("Unsupported search source pins")
    if not isinstance(snapshot.get("files"), dict) or not isinstance(snapshot.get("production"), dict):
        raise RuntimeError("Incomplete search source pins")
    from .evidence import source_snapshot as production_snapshot
    safety = PACKAGE_ROOT / "tools/tunnel_static_batch_safety.sh"
    try:
        current = production_snapshot({}, safety, safety)
        changed = [path for path, expected in snapshot["files"].items() if digest(path) != expected]
    except OSError as error:
        raise RuntimeError("A frozen source or evidence file is missing/unreadable") from error
    if current != snapshot["production"] or changed:
        raise RuntimeError("Frozen search sources/evidence changed; create a new reviewed campaign" +
                           (": " + ", ".join(changed[:5]) if changed else ""))


class ExecutionLock:
    """Nonblocking advisory lock shared by all nested live campaign roots."""

    def __init__(self, path=GLOBAL_EXECUTION_LOCK):
        self.path = Path(path)
        self.stream = None

    def __enter__(self):
        import fcntl  # POSIX-only; imported here so the Windows Expert launcher can import this module
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("a+")
        try:
            fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            self.stream.close()
            self.stream = None
            raise RuntimeError("Another tunnel campaign owns the shared execution lock") from error
        return self

    def __exit__(self, _kind, _value, _traceback):
        if self.stream is not None:
            import fcntl
            try:
                fcntl.flock(self.stream.fileno(), fcntl.LOCK_UN)
            finally:
                self.stream.close()
                self.stream = None


def _raw_identity(pid, proc_root=Path("/proc")):
    node = Path(proc_root) / str(int(pid))
    try:
        fields = (node / "stat").read_text().rsplit(")", 1)[1].split()
        arguments = [os.fsdecode(item) for item in (node / "cmdline").read_bytes().split(b"\0") if item]
        environment = dict(item.split(b"=", 1) for item in (node / "environ").read_bytes().split(b"\0") if b"=" in item)
        if fields[0] == "Z":
            return None
        return {"pid": int(pid), "pgid": int(fields[2]), "start_ticks": str(fields[19]),
                "command": arguments, "campaign_root": os.fsdecode(environment.get(os.fsencode(ROOT_ENV), b""))}
    except FileNotFoundError:
        return None
    except (OSError, ValueError, IndexError) as error:
        raise RuntimeError("Cannot prove campaign process identity: " + str(pid)) from error


def _same_launch(actual, expected):
    return bool(actual and expected and Path(actual[0]).name == Path(expected[0]).name
                and actual[1:] == expected[1:])


def _module_command(arguments, expected_module):
    if not expected_module or not arguments or not Path(arguments[0]).name.startswith("python"):
        return False
    return any(arguments[index:index + 2] == ["-m", expected_module] for index in range(len(arguments) - 1))


def _infer_module(command):
    for index, value in enumerate(command[:-1]):
        if value == "-m":
            return command[index + 1]
    wrappers = {"run_tunnel_static_trial.sh": "scenario_runner.tools.run_tunnel_static_trial",
                "tunnel_static_batch_safety.sh": "scenario_runner.tools.tunnel_static_batch_safety",
                "run_tunnel_static_search.sh": "scenario_runner.tools.run_tunnel_static_search"}
    for argument in command:
        if Path(argument).name in wrappers:
            return wrappers[Path(argument).name]
    return None


def process_identity(pid, campaign_root, expected_module=None, *, launch_command=(), proc_root=Path("/proc")):
    """Read a leader's birth token and prove ROOT plus an exact allowed entrypoint."""
    identity = _raw_identity(pid, proc_root)
    if identity is None:
        return None
    root = str(Path(campaign_root).resolve())
    if (identity["campaign_root"] != root or identity["pgid"] != int(pid)
            or not (_same_launch(identity["command"], list(launch_command))
                    or _module_command(identity["command"], expected_module))):
        raise RuntimeError("Campaign child has an unproved root, process group or entrypoint")
    return {"pid": int(pid), "pgid": int(pid), "start_ticks": identity["start_ticks"],
            "campaign_root": root, "expected_module": expected_module,
            "launch_command": list(launch_command)}


def signal_owned_group(identity, signum, *, proc_root=Path("/proc"), killpg=None):
    """Re-read PID birth, ROOT and entrypoint immediately before every signal."""
    if not isinstance(identity, dict) or not identity.get("launch_command"):
        raise RuntimeError("Cannot signal a process without a frozen launch identity")
    current = process_identity(identity["pid"], identity["campaign_root"], identity.get("expected_module"),
                               launch_command=identity["launch_command"], proc_root=proc_root)
    if current is None:
        return False
    if current != identity:
        raise RuntimeError("Campaign process birth identity changed; refuse to signal its replacement")
    try:
        (killpg or os.killpg)(identity["pgid"], signum)
    except ProcessLookupError:
        return False
    return True


def stop_inflight(identity, *, grace=30., proc_root=Path("/proc")):
    """Safely stop the durable identity of an interrupted owned process group."""
    if not math.isfinite(grace) or not 0 < grace <= 60:
        raise ValueError("Shutdown grace must be positive and at most 60 seconds")
    if identity is None:
        return {"stopped": False, "already_absent": True, "forced_kill": False}
    for signum, budget in ((signal.SIGINT, grace), (signal.SIGKILL, 5.)):
        signaled = signal_owned_group(identity, signum, proc_root=proc_root)
        if not signaled:
            return {"stopped": True, "already_absent": True, "forced_kill": signum == signal.SIGKILL}
        deadline = time.monotonic() + budget
        while time.monotonic() < deadline:
            current = process_identity(identity["pid"], identity["campaign_root"], identity.get("expected_module"),
                                       launch_command=identity["launch_command"], proc_root=proc_root)
            if current is None:
                return {"stopped": True, "already_absent": False, "forced_kill": signum == signal.SIGKILL}
            if current != identity:
                raise RuntimeError("Interrupted process identity changed during bounded shutdown")
            time.sleep(.1)
    raise RuntimeError("Owned campaign process remains after bounded shutdown")


def bounded_run(command, log_path, timeout, *, campaign_root, stage, stop_requested=None,
                on_start=None, expected_module=None, grace=30., cwd=None, on_heartbeat=None,
                console_output=False, console_stream=None):
    """Run one private process group, recording its birth before monitoring it.

    The caller writes its stage checkpoint before calling and uses on_start to
    durably add process_identity. Bash wrappers may exec their known Python
    module; the same PID/group/birth and ROOT must persist through that handoff.
    Interactive runs optionally mirror the log; the child still writes directly
    to the file, so terminal output cannot block control or process cleanup.
    """
    if (not math.isfinite(timeout) or timeout <= 0 or not math.isfinite(grace) or not 0 < grace <= 60
            or not isinstance(stage, str) or not stage or not command):
        raise ValueError("Bounded process needs a stage, command and finite positive deadlines")
    command = [os.fspath(value) for value in command]
    expected_module = expected_module or _infer_module(command)
    root = str(Path(campaign_root).resolve())
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, **{ROOT_ENV: root, "PYTHONDONTWRITEBYTECODE": "1",
                                     "PYTHONUNBUFFERED": "1", "PYTHONFAULTHANDLER": "1"})
    started = time.monotonic()
    timed_out = interrupted = forced_kill = False
    interrupted_at = None
    identity = None
    relay = None
    relay_drained = True
    with log_path.open("ab") as stream:
        if console_output:
            relay = FileConsoleRelay(log_path, start_offset=stream.tell(), console=console_stream,
                                     line_filter=show_helper_console_line).start()
        try:
            process = subprocess.Popen(command, cwd=str(cwd or PROJECT_ROOT), env=environment,
                                       stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        except BaseException:
            if relay is not None:
                relay.stop()
            raise
        try:
            identity = process_identity(process.pid, root, expected_module, launch_command=command)
            fields = {"stage": stage, "process_pid": process.pid, "process_pgid": process.pid,
                      "process_start_ticks": identity["start_ticks"] if identity else None,
                      "process_identity": identity, "command": command, "started_unix": time.time()}
            if on_start is not None:
                on_start(fields)
            last_heartbeat = time.monotonic()
            while process.poll() is None:
                now = time.monotonic()
                should_stop = bool(stop_requested and stop_requested())
                if interrupted_at is None and (now - started >= timeout or should_stop):
                    timed_out = now - started >= timeout
                    interrupted = should_stop and not timed_out
                    if identity is None:
                        raise RuntimeError("Cannot interrupt a child whose launch identity was not captured")
                    signal_owned_group(identity, signal.SIGINT)
                    interrupted_at = now
                if interrupted_at is not None and now - interrupted_at >= grace:
                    signal_owned_group(identity, signal.SIGKILL)
                    forced_kill = True
                    break
                if on_heartbeat is not None and now - last_heartbeat >= 1.:
                    on_heartbeat(fields)
                    last_heartbeat = now
                time.sleep(.1)
            process.wait(timeout=5.)
        finally:
            try:
                if process.poll() is None:
                    if identity is None:
                        identity = process_identity(process.pid, root, expected_module, launch_command=command)
                    if identity is not None:
                        stop_inflight(identity, grace=grace)
                    process.wait(timeout=5.)
            finally:
                if relay is not None:
                    relay_drained = relay.stop()
    result = {"returncode": process.returncode, "timed_out": timed_out, "interrupted": interrupted,
            "forced_kill": forced_kill, "elapsed_sec": time.monotonic() - started,
            "log_path": str(log_path), "process_pid": process.pid, "process_pgid": process.pid,
            "process_start_ticks": identity["start_ticks"] if identity else None,
            "process_identity": identity}
    if relay is not None:
        result.update(console_relay_error=relay.error, console_relay_drained=relay_drained)
    return result
