#!/usr/bin/env bash
# Keep this entrypoint LF-only; Windows Git checkouts are supported.
set -Eeuo pipefail

# Run N independent full laps.  Each lap resets to the official start, selects
# a deterministic fresh weather/time/static/dynamic profile, drives with the
# current team scenario/expert, and records an isolated local Sensor Record session.

TOOLS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INTEGRATION="$(cd "$TOOLS/.." && pwd)"
PROJECT_ROOT="${MORAI_PROJECT_ROOT:-$(cd "$INTEGRATION/../.." && pwd)}"
PUBLIC_TOOLS="$(cd "$INTEGRATION/../tools" && pwd)"
source "$PUBLIC_TOOLS/internal_sensor_record_env.sh"
echo "[PREPARE] PC별 실행 환경과 의존성을 자동 점검합니다."
morai_prepare_internal_sensor_record_environment runtime
morai_load_internal_sensor_record_environment
morai_require_capture_python
morai_require_analysis_python
PROJECT_ROOT="$MORAI_PROJECT_ROOT"
CATKIN_ROOT="$MORAI_CATKIN_WS"
ROS_ROOT="$MORAI_ROS_ROOT"
CATKIN_SETUP="$CATKIN_ROOT/devel/setup.bash"
CATKIN_PYTHON="$CATKIN_ROOT/devel/lib/python3/dist-packages"
ASYNC_DEPS="$MORAI_ASYNC_DEPS"
SSD_ROOT="$(MORAI_PROJECT_ROOT="$PROJECT_ROOT" \
    PYTHONPATH="$INTEGRATION${PYTHONPATH:+:$PYTHONPATH}" \
    /usr/bin/python3 -c 'from ssd_storage import SSD_ROOT; print(SSD_ROOT)')"
BASE="$SSD_ROOT/sensor_record_auto_v1"
LAPS="${1:-}"
CAMPAIGN_ID="${2:-scenario_campaign_$(date +%Y%m%d_%H%M%S)}"
BASE_SEED="${MORAI_BASE_SEED:-26091200}"
LAP_WALL_SECONDS="${MORAI_LAP_WALL_SECONDS:-3000}"
ANALYSIS_PYTHON="$MORAI_ANALYSIS_PYTHON"
CURRENT_LAP=""
CURRENT_RUN=""
COLLECTOR_PID=""

usage() {
    echo "사용법: $0 <바퀴 수> [campaign_id]"
    echo "예:   $0 3"
    echo "선택 환경변수: MORAI_STORAGE_ROOT, MORAI_BASE_SEED, MORAI_LAP_WALL_SECONDS, MORAI_SIMULATOR_PID"
}

if [[ -z "$LAPS" ]] || [[ ! "$LAPS" =~ ^[0-9]+$ ]] || (( LAPS < 1 || LAPS > 100 )); then
    usage >&2
    exit 2
fi
if [[ ! "$CAMPAIGN_ID" =~ ^[A-Za-z0-9_]+$ ]] || (( ${#CAMPAIGN_ID} > 52 )); then
    echo "[FAIL] campaign_id는 영문/숫자/밑줄만 사용하며 52자 이하여야 합니다." >&2
    exit 2
fi
if [[ ! "$BASE_SEED" =~ ^[0-9]+$ ]] || (( BASE_SEED + LAPS >= 2147483648 )); then
    echo "[FAIL] MORAI_BASE_SEED 범위가 잘못됐습니다." >&2
    exit 2
fi
if [[ ! "$LAP_WALL_SECONDS" =~ ^[0-9]+$ ]] || (( LAP_WALL_SECONDS < 600 || LAP_WALL_SECONDS > 3570 )); then
    echo "[FAIL] MORAI_LAP_WALL_SECONDS는 600..3570초여야 합니다." >&2
    exit 2
fi
export MORAI_ANALYSIS_PYTHON="$ANALYSIS_PYTHON"
export MORAI_ASYNC_DEPS="$ASYNC_DEPS"
export MORAI_STORAGE_ROOT="$SSD_ROOT"

/usr/bin/python3 "$INTEGRATION/tools/preflight_local_storage.py" --reserve-gib 70

integration_log="$MORAI_PROJECT_ROOT/.runtime/data_collection/scenario_contract_check.log"
mkdir -p "$(dirname "$integration_log")"
if ! /usr/bin/python3 "$INTEGRATION/tools/run_async_scenario_review.py" \
        --check --raw-camera-transport bridge_dropped --max-speed-kmh 60 \
        >"$integration_log" 2>&1; then
    echo "[FAIL] 최신 Scenario Runner·Expert·수집기 인터페이스가 호환되지 않습니다." >&2
    tail -n 40 "$integration_log" >&2 || true
    exit 2
fi
echo "[PASS] 최신 Scenario Runner·Expert·수집기 정적 연결 검사"

clock_report="$MORAI_PROJECT_ROOT/.runtime/data_collection/clock_preflight.json"
/usr/bin/python3 "$INTEGRATION/tools/preflight_local_clock.py" \
    --wait-seconds "${MORAI_CLOCK_STABILIZATION_WAIT_SECONDS:-90}" \
    --output "$clock_report"

if ! rosparam list >/dev/null 2>&1; then
    echo "[FAIL] ROS master가 없습니다." >&2
    echo "       다른 터미널에서 ./data_collection/tools/01_internal_sensor_record_preflight.sh 를 실행하세요." >&2
    exit 2
fi
if [[ "$(rosparam get /use_sim_time 2>/dev/null || true)" != "true" ]]; then
    echo "[FAIL] /use_sim_time=true가 아닙니다. 01번 실행 파일부터 다시 시작하세요." >&2
    exit 2
fi
readiness="$MORAI_PROJECT_ROOT/.runtime/data_collection/collection_readiness.json"
readiness_error="$MORAI_PROJECT_ROOT/.runtime/data_collection/collection_readiness.stderr"
set +e
/usr/bin/python3 "$INTEGRATION/tools/probe_sensor_record_readiness.py" \
    --seconds 4 --raw-camera-transport bridge_dropped \
    >"$readiness" 2>"$readiness_error"
readiness_code=$?
set -e
if [[ ! -s "$readiness" ]]; then
    echo "[FAIL] 센서 상태 검사기가 보고서를 만들지 못했습니다." >&2
    cat "$readiness_error" >&2 || true
    exit 2
fi
/usr/bin/python3 "$INTEGRATION/tools/render_sensor_record_readiness.py" \
    --bridge-ip "${MORAI_BRIDGE_IP:-}" "$readiness"
if (( readiness_code != 0 )); then
    echo "[FAIL] 센서 preflight가 실패했습니다. 01번 실행 파일의 연결 검사를 먼저 통과하세요." >&2
    exit 2
fi

SIMULATOR_PID="${MORAI_SIMULATOR_PID:-}"
if [[ -z "$SIMULATOR_PID" ]]; then
    POWERSHELL="${MORAI_POWERSHELL_EXE:-$(MORAI_PROJECT_ROOT="$PROJECT_ROOT" PYTHONPATH="$INTEGRATION" /usr/bin/python3 -c 'from runtime_paths import POWERSHELL_EXE; print(POWERSHELL_EXE)')}"
    if [[ ! -x "$POWERSHELL" ]]; then
        echo "[FAIL] Simulator PID 자동 검색이 불가능합니다. MORAI_SIMULATOR_PID를 지정하세요." >&2
        exit 2
    fi
    if ! SIMULATOR_PID="$($POWERSHELL -NoProfile -NonInteractive -Command \
            "\$p=Get-Process Simulator -ErrorAction SilentlyContinue; if(@(\$p).Count -ne 1){exit 3}; [Console]::Write(\$p.Id)" | tr -d '\r')"; then
        echo "[FAIL] Simulator.exe가 없거나 둘 이상입니다. MORAI 하나만 실행하세요." >&2
        exit 2
    fi
fi
if [[ ! "$SIMULATOR_PID" =~ ^[0-9]+$ ]]; then
    echo "[FAIL] 실행 중인 Simulator.exe 하나를 찾지 못했습니다: '$SIMULATOR_PID'" >&2
    exit 2
fi

# A Ctrl-C from an older launcher may have left its scenario child holding the
# advisory capture lock.  Recover only when both the exact executable path and
# a terminal interrupted/failed campaign ledger prove that it is stale.
/usr/bin/python3 - "$INTEGRATION" "$BASE" "$SIMULATOR_PID" <<'PY'
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

integration = Path(sys.argv[1]).resolve()
base = Path(sys.argv[2]).resolve()
simulator_pid = int(sys.argv[3])
lock_path = base / ".locks" / "collector_capture.lock"
campaign_lock_path = base / ".locks" / "collector_campaign.lock"
runner_path = (integration / "tools" / "run_async_scenario_review.py").resolve()
collector_path = (integration / "tools" / "collect_async_raw.py").resolve()

def lock_is_free(path):
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as stream:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return False
            return True
    except OSError as error:
        print("[FAIL] capture lock 저장소를 사용할 수 없습니다: %s: %s" %
              (path, error), file=sys.stderr)
        raise SystemExit(2)

def locks_are_free():
    return lock_is_free(lock_path) and lock_is_free(campaign_lock_path)

if locks_are_free():
    raise SystemExit(0)

terminal = {}
for ledger in (base / "campaigns").glob("*/runs.tsv"):
    try:
        with ledger.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream, delimiter="\t"):
                terminal[row["run_name"]] = row["status"]
    except (OSError, KeyError):
        continue

lock_identities = {}
for path in (lock_path, campaign_lock_path):
    stat = path.stat()
    lock_identities[(stat.st_dev, stat.st_ino)] = path.name

candidates = []
for proc in Path("/proc").glob("[0-9]*"):
    try:
        args = (proc / "cmdline").read_bytes().split(b"\0")
        args = [value.decode(errors="surrogateescape") for value in args if value]
    except (OSError, PermissionError):
        continue
    held_locks = []
    try:
        for descriptor in (proc / "fd").iterdir():
            try:
                stat = descriptor.stat()
            except (OSError, PermissionError):
                continue
            name = lock_identities.get((stat.st_dev, stat.st_ino))
            if name:
                held_locks.append(name)
    except (OSError, PermissionError):
        continue
    if not held_locks:
        continue
    script_arg = next((value for value in args[1:] if value.endswith(".py")), None)
    if script_arg is None:
        candidates.append({"pid": int(proc.name), "kind": "unknown",
                           "held_locks": sorted(held_locks), "args": args[:8],
                           "safe_code_identity": False})
        continue
    try:
        executable = Path(script_arg).resolve()
    except OSError:
        executable = Path(script_arg)
    reference = {runner_path.name: runner_path,
                 collector_path.name: collector_path}.get(executable.name)
    same_code = False
    if reference is not None:
        try:
            same_code = (hashlib.sha256(executable.read_bytes()).digest()
                         == hashlib.sha256(reference.read_bytes()).digest())
        except OSError:
            pass
    if reference is None or "--output" not in args:
        candidates.append({"pid": int(proc.name), "kind": "unknown",
                           "held_locks": sorted(held_locks), "args": args[:8],
                           "script": str(executable),
                           "safe_code_identity": same_code})
        continue
    try:
        output = Path(args[args.index("--output") + 1]).resolve()
    except (IndexError, OSError):
        candidates.append({"pid": int(proc.name), "kind": "unknown",
                           "held_locks": sorted(held_locks), "args": args[:8],
                           "script": str(executable),
                           "safe_code_identity": same_code})
        continue
    if executable.name == runner_path.name:
        run = output
        kind = "scenario_runner"
    else:
        # collect_async_raw output is <base>/<run>/episode_000/raw.
        if output.name != "raw" or output.parent.name != "episode_000":
            continue
        run = output.parent.parent
        kind = "raw_recorder"
    if run.parent != base:
        candidates.append({"pid": int(proc.name), "kind": kind,
                           "held_locks": sorted(held_locks), "args": args[:8],
                           "script": str(executable), "run": run.name,
                           "output": str(output), "status": "different_storage_root",
                           "safe_code_identity": same_code})
        continue
    status = terminal.get(run.name)
    candidates.append({"pid": int(proc.name), "kind": kind,
                       "run": run.name, "output": str(output), "status": status})
    candidates[-1].update(held_locks=sorted(held_locks),
                          script=str(executable), safe_code_identity=same_code)

stale = [row for row in candidates if row.get("safe_code_identity") and
         (row.get("status") == "interrupted"
          or str(row.get("status")).startswith("failed_"))]
active = [row for row in candidates if row not in stale]
if active or not stale or len({row["run"] for row in stale}) != 1:
    print("[FAIL] 다른 수집 프로세스가 capture lock을 사용 중입니다.", file=sys.stderr)
    for row in candidates:
        print("       PID=%s kind=%s run=%s status=%s script=%s" % (
            row.get("pid"), row.get("kind"), row.get("run", "?"),
            row.get("status", "?"), row.get("script", "?")), file=sys.stderr)
    print("       실행 중인 02번 터미널이 있으면 그 실행을 먼저 정상 종료하세요.",
          file=sys.stderr)
    print(json.dumps({"error": "capture_lock_owner_not_safely_recoverable",
                      "candidates": candidates}, ensure_ascii=False), file=sys.stderr)
    raise SystemExit(2)

run_name = stale[0]["run"]
recovery_name = "stale_capture_recovery_" + str(time.time_ns())
stop = subprocess.run(
    [sys.executable, str(integration / "tools" / "recover_sensor_record_stop.py"),
     "--pid", str(simulator_pid), "--name", recovery_name],
    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=45,
)
if stop.returncode:
    print(stop.stderr, file=sys.stderr)
    raise SystemExit(2)

for row in stale:
    try:
        os.kill(row["pid"], signal.SIGINT)
    except ProcessLookupError:
        pass
deadline = time.monotonic() + 90
while time.monotonic() < deadline and not locks_are_free():
    time.sleep(0.2)
if not locks_are_free():
    print(json.dumps({"error": "stale_capture_did_not_release_locks",
                      "candidates": stale}),
          file=sys.stderr)
    raise SystemExit(2)

report = {"schema": "morai_stale_capture_recovery_v1", "run": run_name,
          "processes": stale,
          "sensor_record_stop": json.loads(stop.stdout),
          "signal": "SIGINT", "lock_released": True,
          "recovered_wall_ns": time.time_ns(), "training_visible": False}
report_path = base / (recovery_name + ".json")
report_path.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True) + "\n")
print("[RECOVERED] 이전 중단 run의 capture/campaign lock을 정상 해제했습니다: " + run_name)
PY

requested_campaign="$CAMPAIGN_ID"
if [[ -e "$BASE/campaigns/$CAMPAIGN_ID" ]]; then
    prefix="${CAMPAIGN_ID:0:27}"
    CAMPAIGN_ID="${prefix}_retry_$(date +%Y%m%d_%H%M%S)"
    counter=1
    while [[ -e "$BASE/campaigns/$CAMPAIGN_ID" ]]; do
        CAMPAIGN_ID="${prefix}_retry_$(date +%Y%m%d_%H%M%S)_$counter"
        counter=$((counter + 1))
    done
    echo "[RECOVERED] 기존 campaign '$requested_campaign'은 보존하고 새 이름을 사용합니다: $CAMPAIGN_ID"
fi

CAMPAIGN_DIR="$BASE/campaigns/$CAMPAIGN_ID"
PROFILE_STATE="$CAMPAIGN_DIR/profile_state.json"
RUNS_FILE="$CAMPAIGN_DIR/runs.tsv"
mkdir -p "$CAMPAIGN_DIR"
printf 'lap\tseed\trun_name\tstatus\tweather\ttime_hour\tstatic_obstacle\tdynamic_obstacle\tstarted_utc\tfinished_utc\n' > "$RUNS_FILE"
/usr/bin/python3 - "$CAMPAIGN_DIR/campaign_plan.json" "$CAMPAIGN_ID" "$LAPS" "$BASE_SEED" "$LAP_WALL_SECONDS" "$SIMULATOR_PID" <<'PY'
import json, pathlib, sys, time
path, campaign_id, laps, seed, seconds, pid = sys.argv[1:]
payload = {
    "schema": "morai_sensor_record_campaign_v1",
    "campaign_id": campaign_id,
    "requested_laps": int(laps),
    "base_seed": int(seed),
    "lap_wall_limit_seconds": int(seconds),
    "simulator_pid": int(pid),
    "mode": "Simulation Time / internal sensor sync 10Hz / External Sync OFF",
    "weather_candidates": ["SUNNY", "FOGGY"],
    "time_candidates": [11, 13, 15],
    "static_and_dynamic_obstacles": "fresh deterministic random selection per lap",
    "highway_native_vehicle_cap": "team scenario configuration",
    "competition_reference_spawnpoint_cap_per_point": 50,
    "mission_distribution_changes": [
        "scenario route-light red-stop release after 10 host-monotonic seconds; exact-ID command/readback events are disclosed"
    ],
    "expert_max_speed_kmh": 60,
    "raw_camera_transport": "bridge_dropped",
    "native_camera_storage": "configured local Windows Sensor Record storage",
    "offline_review": "deferred until validate_sensor_record_campaign.py",
    "created_wall_ns": time.time_ns(),
    "training_visible": False,
    "physical_simultaneity_verified": False,
}
pathlib.Path(path).write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
PY

on_signal() {
    trap - INT TERM
    if [[ -n "$CURRENT_LAP" ]]; then
        coordination="$BASE/$CURRENT_RUN/episode_000/sensor_record_coordination"
        if [[ -n "$COLLECTOR_PID" ]] && kill -0 "$COLLECTOR_PID" 2>/dev/null; then
            if [[ -f "$coordination/recording_armed.json" ]]; then
                echo
                echo "[STOP] Sensor Record OFF를 먼저 요청하고 확인합니다..."
                if [[ ! -e "$coordination/stop.request" ]]; then
                    tmp="$coordination/stop.request.$$"
                    printf '{"reason":"campaign_interrupted","request_wall_ns":%s,"training_visible":false}\n' \
                        "$(date +%s%N)" > "$tmp"
                    mv "$tmp" "$coordination/stop.request"
                fi
                stop_deadline=$((SECONDS + 25))
                while kill -0 "$COLLECTOR_PID" 2>/dev/null \
                        && [[ ! -f "$coordination/recording_stopped.json" ]] \
                        && (( SECONDS < stop_deadline )); do
                    sleep 0.2
                done
            fi
            if kill -0 "$COLLECTOR_PID" 2>/dev/null; then
                kill -INT "$COLLECTOR_PID" 2>/dev/null || true
                child_deadline=$((SECONDS + 45))
                while kill -0 "$COLLECTOR_PID" 2>/dev/null && (( SECONDS < child_deadline )); do
                    sleep 0.2
                done
            fi
            wait "$COLLECTOR_PID" 2>/dev/null || true
        fi
        printf '%s\t%s\t%s\tinterrupted\t-\t-\t-\t-\t-\t%s\n' \
            "$CURRENT_LAP" "$((BASE_SEED + CURRENT_LAP - 1))" "$CURRENT_RUN" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$RUNS_FILE"
    fi
    exit 130
}
trap on_signal INT TERM

echo "[START] campaign=$CAMPAIGN_ID, laps=$LAPS, local raw only, storage=$SSD_ROOT, Simulator PID=$SIMULATOR_PID"
echo "        매 바퀴 시작점 reset, Sunny/Foggy, 11/13/15시, 정적·동적 장애물 재선택"
echo "        주행 중에는 30초마다 진행 상태를 출력하며 상세 로그는 run별 supervisor.log에 저장합니다."

for ((lap=1; lap<=LAPS; lap++)); do
    CURRENT_LAP="$lap"
    printf -v lap_tag '%03d' "$lap"
    CURRENT_RUN="${CAMPAIGN_ID}_lap_${lap_tag}"
    seed=$((BASE_SEED + lap - 1))
    started="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf '%s\t%s\t%s\tstarted\t-\t-\t-\t-\t%s\t-\n' \
        "$lap" "$seed" "$CURRENT_RUN" "$started" >> "$RUNS_FILE"
    echo
    echo "[LAP $lap/$LAPS] seed=$seed run=$CURRENT_RUN"
    /usr/bin/python3 "$INTEGRATION/tools/run_sensor_record_scenario.py" \
            --name "$CURRENT_RUN" --pid "$SIMULATOR_PID" \
            --seconds "$LAP_WALL_SECONDS" --full-lap \
            --seed "$seed" --profile-state "$PROFILE_STATE" \
            --raw-camera-transport bridge_dropped --defer-offline-review &
    COLLECTOR_PID=$!
    if wait "$COLLECTOR_PID"; then
        COLLECTOR_PID=""
        finished="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        profile_copy="$BASE/$CURRENT_RUN/episode_000/campaign_profile_state.json"
        if [[ -e "$profile_copy" ]]; then
            echo "[FAIL] 회차별 profile 증거 경로가 이미 있습니다: $profile_copy" >&2
            exit 2
        fi
        cp "$PROFILE_STATE" "$profile_copy"
        read -r weather hour static_model dynamic_model < <(
            /usr/bin/python3 - "$profile_copy" <<'PY'
import json, sys
x=json.load(open(sys.argv[1]))
print(x["weather"], x["time_hour"],
      x["static_obstacle_model"], x["dynamic_obstacle_model"])
PY
        )
        printf '%s\t%s\t%s\tcaptured\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$lap" "$seed" "$CURRENT_RUN" "$weather" "$hour" "$static_model" "$dynamic_model" "$started" "$finished" >> "$RUNS_FILE"
        echo "[CAPTURED] lap=$lap weather=$weather time=${hour}:00 static=$static_model dynamic=$dynamic_model"
        echo "           전체 byte/GT 검증은 3바퀴 수집 후 03번에서 실행합니다."
    else
        code=$?
        COLLECTOR_PID=""
        finished="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        printf '%s\t%s\t%s\tfailed_%s\t-\t-\t-\t-\t%s\t%s\n' \
            "$lap" "$seed" "$CURRENT_RUN" "$code" "$started" "$finished" >> "$RUNS_FILE"
        echo "[FAIL] lap $lap가 실패했습니다. 원본은 보존했고 다음 reset은 진행하지 않습니다." >&2
        /usr/bin/python3 - "$BASE/$CURRENT_RUN" <<'PY' >&2 || true
import json
from pathlib import Path
import sys

run = Path(sys.argv[1])
episode = run / "episode_000"
outcome_path = episode / "outcome.json"
events_path = episode / "events.jsonl"
native_path = episode / "native_audit" / "report.json"
raw_manifest_path = episode / "raw" / "raw_manifest.json"
if not outcome_path.is_file():
    raise SystemExit(0)
outcome = json.loads(outcome_path.read_text())
progress = []
if events_path.is_file():
    for line in events_path.read_text().splitlines():
        row = json.loads(line)
        if row.get("event") == "progress":
            progress.append(row)
native = json.loads(native_path.read_text()) if native_path.is_file() else {}
raw_manifest = (json.loads(raw_manifest_path.read_text())
                if raw_manifest_path.is_file() else {})
last = progress[-1] if progress else {}
route_total = None
episode_path = episode / "episode.json"
if episode_path.is_file():
    route_total = json.loads(episode_path.read_text()).get("route_length_m")
print(
    "[DIAG] scenario_status=%s reason=%s route=%.1f/%s m sim_elapsed=%.1fs "
    "complete_native_bundles=%s native_files_valid=%s" % (
        outcome.get("status"), outcome.get("reason"),
        float(last.get("route_s", 0.0)),
        ("%.1f" % float(route_total)) if route_total is not None else "?",
        float(last.get("elapsed_seconds", 0.0)),
        native.get("complete_bundles", "?"), native.get("files_valid", "?"),
    )
)
if raw_manifest.get("status") == "failed":
    print("[DIAG] raw_status=failed raw_reason=%s" % raw_manifest.get("reason", "?"))
if outcome.get("reason") == "bounded_window_elapsed":
    print(
        "[DIAG] 센서 저장 실패가 아니라 제한 시간 안에 도착점에 도달하지 "
        "못한 주행 완료 실패입니다. 이 run은 부분 데이터로 보존됩니다."
    )
PY
        echo "       검증: ./data_collection/tools/03_validate_internal_sensor_record.sh $CAMPAIGN_ID" >&2
        exit "$code"
    fi
done

CURRENT_LAP=""
/usr/bin/python3 - "$CAMPAIGN_DIR/COLLECTION_COMPLETE.json" "$CAMPAIGN_ID" "$LAPS" <<'PY'
import json, pathlib, sys, time
path, campaign_id, laps = sys.argv[1:]
pathlib.Path(path).write_text(json.dumps({
    "schema": "morai_sensor_record_campaign_completion_v1",
    "campaign_id": campaign_id,
    "captured_laps": int(laps),
    "offline_review_complete": False,
    "completed_wall_ns": time.time_ns(),
    "training_visible": False,
    "physical_simultaneity_verified": False,
}, sort_keys=True) + "\n")
PY

echo
echo "[DONE] $LAPS바퀴 원본 수집이 끝났습니다. 원본은 아직 학습에 공개되지 않습니다."
echo "검증 실행:"
echo "  cd $MORAI_PROJECT_ROOT"
echo "  ./data_collection/tools/03_validate_internal_sensor_record.sh $CAMPAIGN_ID"
