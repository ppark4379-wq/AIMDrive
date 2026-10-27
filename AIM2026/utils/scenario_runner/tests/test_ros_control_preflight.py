"""Offline lifecycle checks for the scenario-only installed ROS bridge."""
from pathlib import Path
import signal
import subprocess
import sys

import pytest

from scenario_runner.tools import ros_preflight as preflight


class Clock:
    def __init__(self, stopped, stop_at=1000):
        self.now = 0
        self.stopped = stopped
        self.stop_at = stop_at

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds
        if self.now >= self.stop_at:
            self.stopped.append(True)


def lifecycle(monkeypatch, *, master_at=0, control=lambda t: True,
              bridge_dead_at=None, reused=False, listener=lambda t: True,
              stop_at=1000, timeout=30):
    stopped, trace = [], []
    clock = Clock(stopped, stop_at)
    monkeypatch.setattr(preflight, "time", clock)

    class Master:
        def getParamNames(self):
            trace.append((clock.now, "master"))
            if clock.now < master_at:
                raise OSError("not ready")
            return []

        def getSystemState(self):
            trace.append((clock.now, "control"))
            connected = control(clock.now)
            if isinstance(connected, Exception):
                raise connected
            return [], [("/ctrl_cmd", ["/morai"]) ] if connected else [], []

    class Process:
        def poll(self):
            return 2 if bridge_dead_at is not None and clock.now >= bridge_dead_at else None

    monkeypatch.setattr(preflight, "listener_alive", lambda port: listener(clock.now))
    try:
        preflight.keep_control_ready(Master(), lambda *args: trace.append((clock.now, "host_time")),
                                     None if reused else Process(), 9090, timeout, stopped)
        result = "stopped"
    except (RuntimeError, TimeoutError) as exc:
        result = str(exc)
    return result, trace, clock.now


def test_master_precedes_host_time_and_control(monkeypatch):
    result, trace, _ = lifecycle(monkeypatch, master_at=3, control=lambda t: t >= 6, stop_at=8)
    assert result == "stopped"
    assert trace[:4] == [(0, "master"), (1, "master"), (2, "master"), (3, "master")]
    assert trace[4:6] == [(3, "host_time"), (3, "control")]


def test_master_timeout_is_40_seconds(monkeypatch):
    result, _, elapsed = lifecycle(monkeypatch, master_at=41)
    assert "40 seconds" in result
    assert elapsed == 40


def test_control_timeout_starts_after_master_ready(monkeypatch):
    result, _, elapsed = lifecycle(monkeypatch, master_at=7, control=lambda _: False)
    assert "No MORAI /ctrl_cmd subscriber" in result
    assert elapsed == 37


def test_control_loss_fails_on_fifth_consecutive_check(monkeypatch):
    result, _, elapsed = lifecycle(monkeypatch, control=lambda t: t < 2)
    assert "5 consecutive" in result
    assert elapsed == 6


def test_control_reconnection_resets_missing_counter(monkeypatch):
    result, _, elapsed = lifecycle(monkeypatch, control=lambda t: t in (0, 5), stop_at=10)
    assert result == "stopped"
    assert elapsed == 10


def test_master_query_failure_counts_as_control_loss(monkeypatch):
    result, _, elapsed = lifecycle(monkeypatch, control=lambda t: True if t == 0 else OSError())
    assert "5 consecutive" in result
    assert elapsed == 5


@pytest.mark.parametrize("master_at,control,death,expected", [
    (20, lambda _: False, 2, "before the ROS master"),
    (0, lambda _: False, 2, "waiting for /ctrl_cmd"),
    (0, lambda _: True, 2, "during scenario validation"),
])
def test_owned_bridge_exit_by_phase(monkeypatch, master_at, control, death, expected):
    result, _, elapsed = lifecycle(monkeypatch, master_at=master_at, control=control, bridge_dead_at=death)
    assert expected in result
    assert elapsed == death


def test_reused_bridge_listener_is_monitored(monkeypatch):
    result, _, elapsed = lifecycle(monkeypatch, reused=True, listener=lambda t: t < 3)
    assert "existing ROS bridge listener exited" in result
    assert elapsed == 3


def test_keepalive_is_every_30_seconds(monkeypatch, capsys):
    result, _, _ = lifecycle(monkeypatch, stop_at=61)
    assert result == "stopped"
    assert capsys.readouterr().out.count("[KEEPALIVE]") == 2


def test_default_timeout_and_env_override(monkeypatch):
    monkeypatch.delenv("MORAI_CONNECT_TIMEOUT_SECONDS", raising=False)
    assert preflight.build_parser().parse_args([]).timeout == 600
    monkeypatch.setenv("MORAI_CONNECT_TIMEOUT_SECONDS", "900")
    assert preflight.build_parser().parse_args([]).timeout == 900
    assert preflight.build_parser().parse_args(["--timeout", "30"]).timeout == 30


@pytest.mark.parametrize("timeout", ["29", "0", "-1", "30.5", "bad"])
def test_invalid_timeout_rejected_before_any_network(monkeypatch, timeout):
    monkeypatch.setattr(preflight, "listener_alive", lambda *_: pytest.fail("network called"))
    with pytest.raises(SystemExit) as exc:
        preflight.main(["--timeout", timeout])
    assert exc.value.code == 2


def test_check_does_not_open_sockets_or_processes(monkeypatch, capsys):
    monkeypatch.setattr(preflight.importlib.util, "find_spec", lambda _: object())
    monkeypatch.setattr(preflight, "listener_alive", lambda *_: pytest.fail("network called"))
    monkeypatch.setattr(preflight.subprocess, "Popen", lambda *_a, **_kw: pytest.fail("process called"))
    assert preflight.main(["--check"]) == 0
    assert '"network_started": false' in capsys.readouterr().out


@pytest.mark.parametrize("timeout", [False, True])
def test_only_owned_bridge_gets_term_then_35_second_grace(monkeypatch, timeout):
    trace = []
    monkeypatch.setattr(preflight.os, "killpg", lambda pid, sig: trace.append((pid, sig)))

    class Process:
        pid = 123
        def poll(self):
            return None
        def wait(self, timeout=None):
            trace.append(("wait", timeout))
            if timeout == 35 and force_timeout:
                raise subprocess.TimeoutExpired("roslaunch", timeout)

    force_timeout = timeout
    preflight.stop_owned_bridge(Process())
    expected = [(123, signal.SIGTERM), ("wait", 35)]
    if timeout:
        expected += [(123, signal.SIGKILL), ("wait", None)]
    assert trace == expected
    preflight.stop_owned_bridge(None)
    assert trace == expected


def test_baseline_scenario_policy_constants_are_preserved():
    # A read-only B consumer oracle: these assertions pin the policies being
    # translated, while the tests above execute the new implementation.
    source = (Path(__file__).parent / "fixtures/external_consumers/01_internal_sensor_record_preflight.sh.txt").read_text()
    assert 'CONNECT_TIMEOUT="${MORAI_CONNECT_TIMEOUT_SECONDS:-600}"' in source
    assert 'CONNECT_TIMEOUT < 30' in source
    assert 'deadline=$((SECONDS + 40))' in source
    assert source.index('rosparam set /use_sim_time false') < source.index('deadline=$((SECONDS + CONNECT_TIMEOUT))', source.index("PYCTRL")-1500)
    assert 'ctrl_cmd_missing_count >= 5' in source
    assert 'keepalive_report_at=$((SECONDS + 30))' in source
    assert 'kill -TERM "$BRIDGE_PID"' in source
    assert 'local deadline=$((SECONDS + 35))' in source


@pytest.mark.parametrize("reused,fail", [(False, False), (True, False), (False, True)])
def test_main_owns_only_a_bridge_it_started(monkeypatch, reused, fail):
    from types import SimpleNamespace
    trace = []
    process = object()
    monkeypatch.setattr(preflight.importlib.util, "find_spec", lambda _: object())
    monkeypatch.setattr(preflight, "listener_alive", lambda _: reused)
    monkeypatch.setattr(preflight, "build_capture_environment", lambda: {"NATIVE": "1"})
    monkeypatch.setattr(preflight.subprocess, "Popen", lambda command, **kwargs: trace.append((command, kwargs)) or process)
    monkeypatch.setitem(sys.modules, "rosgraph", SimpleNamespace(Master=lambda _: "master"))
    monkeypatch.setitem(sys.modules, "rospy", SimpleNamespace(set_param=lambda *_: None))
    monkeypatch.setattr(preflight.signal, "signal", lambda *_: "previous")
    def run(master, setter, owned, port, timeout, stopped):
        assert (master, owned, port, timeout) == ("master", None if reused else process, 9090, 600)
        if fail:
            raise RuntimeError("test lifecycle failure")
    monkeypatch.setattr(preflight, "keep_control_ready", run)
    monkeypatch.setattr(preflight, "stop_owned_bridge", lambda owned: trace.append(("stop", owned)))
    if fail:
        with pytest.raises(RuntimeError, match="test lifecycle failure"):
            preflight.main(["--scenario-only"])
    else:
        assert preflight.main(["--scenario-only"]) == 0
    assert trace[-1] == ("stop", None if reused else process)
    if not reused:
        assert trace[0] == (["roslaunch", "rosbridge_server", "rosbridge_websocket.launch",
                             "port:=9090", "address:=0.0.0.0"],
                            {"env": {"NATIVE": "1"}, "start_new_session": True})


def baseline_shell_lifecycle(tmp_path, *, master_at=0, control="true", death=10000,
                             reused=False, listener_death=10000, stop_at=1000, timeout=30):
    """Execute B's actual scenario branch with an advancing clock and fake ROS.

    Only process/socket boundary commands and the wall clock are replaced. The
    B master/control/keepalive branches, intervals and failure exits execute.
    """
    import re
    source = (Path(__file__).parent / "fixtures/external_consumers/01_internal_sensor_record_preflight.sh.txt").read_text()
    branch = source[source.index('echo "[4/5]'):source.index('echo "[5/5] 실제')]
    branch = re.sub(r'/usr/bin/python3 "\$INTERNAL/tools/start_instrumented_bridge.py" \\\n.*?BRIDGE_PID=\$!', 'BRIDGE_PID=4242', branch, flags=re.S)
    branch = re.sub(r"/usr/bin/python3 - <<'PYCTRL'\n.*?\nPYCTRL", 'test_control', branch, flags=re.S)
    branch = branch.replace("SECONDS", "FAKE_SECONDS")
    trace_file = tmp_path / "baseline_trace.txt"
    prelude = '''set -Eeuo pipefail
FAKE_SECONDS=0
STARTED_BRIDGE=0
BRIDGE_PID=''
MASTER_URI=fake
BRIDGE_LOG=/dev/null
RUN_MODE=scenario
MORAI_ENDPOINT_IP=192.0.2.1
MORAI_PROJECT_ROOT=fake
DEAD=0
trace() { printf '%s|%s\\n' "$FAKE_SECONDS" "$1" >> "$TRACE_FILE"; }
trap 'trace ended' EXIT
sleep() {
    FAKE_SECONDS=$((FAKE_SECONDS + 1))
    if (( FAKE_SECONDS >= STOP_AT )); then trace stopped; exit 0; fi
}
rosparam() {
    if [[ "$1" == list ]]; then
        trace master
        (( FAKE_SECONDS >= MASTER_AT ))
    else trace host_time; fi
}
test_control() { trace control; CONTROL_COMMAND; }
ss() { if (( REUSED == 1 && FAKE_SECONDS < LISTENER_DEATH )); then echo listener; fi; }
kill() { if [[ "$1" == -0 ]]; then (( DEAD == 0 && FAKE_SECONDS < DEATH )); else DEAD=1; fi; }
wait() { :; }
tail() { :; }
'''.replace('CONTROL_COMMAND', control)
    env = dict(TRACE_FILE=str(trace_file), STOP_AT=str(stop_at), MASTER_AT=str(master_at),
               CONNECT_TIMEOUT=str(timeout), DEATH=str(death), REUSED=str(int(reused)),
               LISTENER_DEATH=str(listener_death))
    completed = subprocess.run(["bash", "-c", prelude + branch], env=env, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5)
    assert completed.returncode in (0, 2), completed.stderr
    rows = [(int(t), name) for t, name in (line.split("|", 1) for line in trace_file.read_text().splitlines())]
    trace = [row for row in rows if row[1] not in ("stopped", "ended")]
    elapsed = rows[-1][0]
    return completed.returncode, trace, elapsed


@pytest.mark.parametrize("case", [
    dict(name="owned_stop", stop_at=8),
    dict(name="reused_stop", reused=True, stop_at=8),
    dict(name="master_delay", master_at=3, stop_at=8),
    dict(name="master_40_boundary", master_at=40, stop_at=43),
    dict(name="master_timeout", master_at=41),
    dict(name="control_delay", control_py=lambda t: t >= 4, control_sh='(( FAKE_SECONDS >= 4 ))', stop_at=8),
    dict(name="control_timeout", master_at=7, control_py=lambda _: False, control_sh='false'),
    dict(name="control_at_deadline", control_py=lambda t: t >= 30, control_sh='(( FAKE_SECONDS >= 30 ))', stop_at=33),
    dict(name="control_loss", control_py=lambda t: t < 2, control_sh='(( FAKE_SECONDS < 2 ))'),
    dict(name="control_reset", control_py=lambda t: t in (0, 5), control_sh='(( FAKE_SECONDS == 0 || FAKE_SECONDS == 5 ))', stop_at=10),
    dict(name="master_bridge_exit", master_at=20, death=2),
    dict(name="control_bridge_exit", control_py=lambda _: False, control_sh='false', death=2),
    dict(name="keepalive_bridge_exit", death=2),
    dict(name="reused_listener_exit", reused=True, listener_death=3),
    dict(name="keepalive_60", stop_at=61),
], ids=lambda case: case["name"])
def test_b_scenario_branch_matches_native_lifecycle(monkeypatch, tmp_path, case):
    values = {key: value for key, value in case.items() if key != "name"}
    control_py = values.pop("control_py", lambda _: True)
    control_sh = values.pop("control_sh", "true")
    code, old_trace, old_time = baseline_shell_lifecycle(tmp_path, control=control_sh, **values)
    result, new_trace, new_time = lifecycle(monkeypatch,
        master_at=values.get("master_at", 0), control=control_py,
        bridge_dead_at=values.get("death"), reused=values.get("reused", False),
        listener=lambda t: t < values.get("listener_death", 10000),
        stop_at=values.get("stop_at", 1000), timeout=values.get("timeout", 30))
    assert (code == 0) == (result == "stopped")
    assert new_time == old_time
    assert new_trace == old_trace
