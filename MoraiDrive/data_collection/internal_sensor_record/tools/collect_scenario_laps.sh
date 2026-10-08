#!/usr/bin/env bash
set -Eeuo pipefail

# Run N independent passes of one selected main Scenario Runner mode. Each
# pass resets to that mode's START, applies its deterministic profile, and
# records an isolated local Sensor Record session.

SCENARIO_RUNTIME="${MORAI_SCENARIO_RUNTIME:-current}"
if [[ "$SCENARIO_RUNTIME" != "current" ]]; then
    echo "[FAIL] MORAI_SCENARIO_RUNTIME supports only current; proven mode has been removed." >&2
    exit 2
fi

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
    PYTHONPATH="$INTEGRATION" \
    /usr/bin/python3 -c 'from ssd_storage import SSD_ROOT; print(SSD_ROOT)')"
LEGACY_BASE="$SSD_ROOT/sensor_record_auto_v1"
DATASET_ROOT="$SSD_ROOT/dataset"
usage() {
    echo "사용법: $0 <회차 수> [campaign_id] [--scenario <name>] [--static-preset <name|path>] [--static-plan <json>]"
    echo "예:   $0 3"
    echo "      $0 100 --scenario full_loop_static"
    echo "      $0 30 --scenario full_loop_static --static-preset intersection_pair_01"
    echo "      $0 100 --scenario full_loop_static --static-plan scenario_runner/config/static_scenario_presets/plans/static_balanced_100.json"
    echo "      $0 3 campaign_static --scenario full_loop_static"
    echo "campaign_id를 생략하면 <시나리오_폴더>_lap<회차>_<생성일시>로 자동 생성합니다."
    echo "시나리오: full_loop, full_loop_normal, full_loop_static, full_loop_dynamic,"
    echo "          full_loop_roundabout, full_loop_roundabout_nonuniform,"
    echo "          full_loop_roundabout_speed_random, full_loop_roundabout_brake,"
    echo "          full_loop_highway, full_loop_tunnel_dynamic, full_loop_tunnel_vehicle"
    echo "터널: 보행자는 랜덤 4구간 조합, 차량은 회차별 랜덤 미션·모델로 실행합니다."
    echo "선택 환경변수: MORAI_STORAGE_ROOT, MORAI_BASE_SEED, MORAI_LAP_WALL_SECONDS, MORAI_SIMULATOR_PID"
    echo "                MORAI_SCENARIO_RUNTIME=current (기본 current)"
}

LAPS="${1:-}"
if (( $# > 0 )); then
    shift
fi
CAMPAIGN_ID=""
SCENARIO="full_loop"
STATIC_PRESET=""
STATIC_PLAN=""
while (( $# > 0 )); do
    case "$1" in
        --scenario)
            if (( $# < 2 )); then
                echo "[FAIL] --scenario 뒤에 시나리오 이름이 필요합니다." >&2
                usage >&2
                exit 2
            fi
            SCENARIO="$2"
            shift 2
            ;;
        --scenario=*)
            SCENARIO="${1#--scenario=}"
            shift
            ;;
        --static-preset)
            if (( $# < 2 )); then
                echo "[FAIL] --static-preset 뒤에 프리셋 이름 또는 YAML 경로가 필요합니다." >&2
                usage >&2
                exit 2
            fi
            STATIC_PRESET="$2"
            shift 2
            ;;
        --static-preset=*)
            STATIC_PRESET="${1#--static-preset=}"
            shift
            ;;
        --static-plan)
            if (( $# < 2 )); then
                echo "[FAIL] --static-plan 뒤에 collection plan JSON 경로가 필요합니다." >&2
                usage >&2
                exit 2
            fi
            STATIC_PLAN="$2"
            shift 2
            ;;
        --static-plan=*)
            STATIC_PLAN="${1#--static-plan=}"
            shift
            ;;
        --*)
            echo "[FAIL] 알 수 없는 옵션입니다: $1" >&2
            usage >&2
            exit 2
            ;;
        *)
            if [[ -n "$CAMPAIGN_ID" ]]; then
                echo "[FAIL] campaign_id는 하나만 지정할 수 있습니다." >&2
                usage >&2
                exit 2
            fi
            CAMPAIGN_ID="$1"
            shift
            ;;
    esac
done
case "$SCENARIO" in
    full_loop|full_loop_normal|full_loop_static|full_loop_dynamic|full_loop_roundabout|full_loop_roundabout_nonuniform|full_loop_roundabout_speed_random|full_loop_roundabout_brake|full_loop_highway|full_loop_tunnel_dynamic|full_loop_tunnel_vehicle)
        ;;
    *)
        echo "[FAIL] 지원하지 않는 시나리오입니다: $SCENARIO" >&2
        usage >&2
        exit 2
        ;;
esac
if [[ -n "$STATIC_PRESET" && "$SCENARIO" != "full_loop_static" ]]; then
    echo "[FAIL] --static-preset은 --scenario full_loop_static에서만 사용할 수 있습니다." >&2
    exit 2
fi
if [[ -n "$STATIC_PLAN" && "$SCENARIO" != "full_loop_static" ]]; then
    echo "[FAIL] --static-plan은 --scenario full_loop_static에서만 사용할 수 있습니다." >&2
    exit 2
fi
if [[ -n "$STATIC_PRESET" && -n "$STATIC_PLAN" ]]; then
    echo "[FAIL] --static-preset과 --static-plan은 함께 사용할 수 없습니다." >&2
    exit 2
fi
case "$SCENARIO" in
    full_loop) SCENARIO_FOLDER="full_route" ;;
    full_loop_normal) SCENARIO_FOLDER="basic_driving" ;;
    full_loop_static) SCENARIO_FOLDER="static_obstacle" ;;
    full_loop_dynamic) SCENARIO_FOLDER="dynamic_obstacle" ;;
    full_loop_roundabout) SCENARIO_FOLDER="roundabout" ;;
    full_loop_roundabout_nonuniform) SCENARIO_FOLDER="roundabout_nonuniform" ;;
    full_loop_roundabout_speed_random) SCENARIO_FOLDER="roundabout_speed_random" ;;
    full_loop_roundabout_brake) SCENARIO_FOLDER="roundabout_brake" ;;
    full_loop_highway) SCENARIO_FOLDER="highway" ;;
    full_loop_tunnel_dynamic) SCENARIO_FOLDER="tunnel_dynamic" ;;
    full_loop_tunnel_vehicle) SCENARIO_FOLDER="tunnel_vehicle" ;;
esac
BASE_SEED="${MORAI_BASE_SEED:-26091200}"
LAP_WALL_SECONDS="${MORAI_LAP_WALL_SECONDS:-3000}"
ANALYSIS_PYTHON="$MORAI_ANALYSIS_PYTHON"
CURRENT_LAP=""
CURRENT_RUN=""
COLLECTOR_PID=""

if [[ -z "$LAPS" ]] || [[ ! "$LAPS" =~ ^[0-9]+$ ]] || (( LAPS < 1 || LAPS > 100 )); then
    usage >&2
    exit 2
fi
if [[ -z "$CAMPAIGN_ID" ]]; then
    CAMPAIGN_ID="${SCENARIO_FOLDER}_lap${LAPS}_$(date +%Y%m%d_%H%M%S)"
    echo "[NAME] campaign_id를 자동 생성했습니다: $CAMPAIGN_ID"
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
export MORAI_SCENARIO_RUNTIME

/usr/bin/python3 "$INTEGRATION/tools/preflight_local_storage.py" --reserve-gib 70

integration_log="$MORAI_PROJECT_ROOT/.runtime/data_collection/scenario_contract_check.log"
mkdir -p "$(dirname "$integration_log")"
STATIC_PLAN_TOOL="$PROJECT_ROOT/scenario_runner/tools/static_collection_plan.py"
STATIC_CHECK_PRESET="$STATIC_PRESET"
if [[ -n "$STATIC_PLAN" ]]; then
    if [[ ! -f "$STATIC_PLAN_TOOL" ]]; then
        echo "[FAIL] static collection plan 검사기가 없습니다: $STATIC_PLAN_TOOL" >&2
        exit 2
    fi
    if ! /usr/bin/python3 "$STATIC_PLAN_TOOL" check \
            --plan "$STATIC_PLAN" --laps "$LAPS" --require-native-scale \
            >"$integration_log" 2>&1; then
        echo "[FAIL] 정적 장애물 collection plan이 실행 가능한 상태가 아닙니다." >&2
        cat "$integration_log" >&2 || true
        exit 2
    fi
    STATIC_CHECK_PRESET="$(/usr/bin/python3 "$STATIC_PLAN_TOOL" resolve \
        --plan "$STATIC_PLAN" --lap 1 --field preset)"
fi
scenario_check=(
    /usr/bin/python3 "$INTEGRATION/tools/run_proven_three_lap_scenario.py"
    --check --scenario "$SCENARIO" --nearby-actors
    --expert-runtime "$SCENARIO_RUNTIME"
    --time-mode simulation_free_run --collision-policy record
    --raw-camera-transport bridge_dropped --max-speed-kmh 60
)
if [[ -n "$STATIC_CHECK_PRESET" ]]; then
    scenario_check+=(--static-preset "$STATIC_CHECK_PRESET")
fi
runtime_label="현재 MoraiDrive/expert runtime"
if ! "${scenario_check[@]}" >"$integration_log" 2>&1; then
    echo "[FAIL] 선택한 Scenario Runner·Expert·수집기 인터페이스가 호환되지 않습니다." >&2
    echo "       runtime=$SCENARIO_RUNTIME ($runtime_label)" >&2
    tail -n 40 "$integration_log" >&2 || true
    exit 2
fi
echo "[PASS] Scenario Runner·Expert·수집기 정적 연결 검사"
echo "       runtime=$SCENARIO_RUNTIME ($runtime_label)"

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
/usr/bin/python3 "$INTEGRATION/tools/recover_stale_capture.py" \
    --base "$LEGACY_BASE" --simulator-pid "$SIMULATOR_PID"

requested_campaign="$CAMPAIGN_ID"
STAGING_PARENT="$DATASET_ROOT/.collecting/$SCENARIO_FOLDER"
FINAL_PARENT="$DATASET_ROOT/$SCENARIO_FOLDER"
if [[ -e "$STAGING_PARENT/$CAMPAIGN_ID" || -e "$FINAL_PARENT/$CAMPAIGN_ID" ]]; then
    prefix="${CAMPAIGN_ID:0:27}"
    CAMPAIGN_ID="${prefix}_retry_$(date +%Y%m%d_%H%M%S)"
    counter=1
    while [[ -e "$STAGING_PARENT/$CAMPAIGN_ID" || -e "$FINAL_PARENT/$CAMPAIGN_ID" ]]; do
        CAMPAIGN_ID="${prefix}_retry_$(date +%Y%m%d_%H%M%S)_$counter"
        counter=$((counter + 1))
    done
    echo "[RECOVERED] 기존 campaign '$requested_campaign'은 보존하고 새 이름을 사용합니다: $CAMPAIGN_ID"
fi

CAMPAIGN_ROOT="$STAGING_PARENT/$CAMPAIGN_ID"
CAMPAIGN_DIR="$CAMPAIGN_ROOT/_campaign"
if [[ "$SCENARIO" == "full_loop" ]]; then
    PROFILE_STATE="$CAMPAIGN_DIR/profile_state.json"
else
    PROFILE_STATE="$CAMPAIGN_DIR/${SCENARIO}_profile.json"
fi
RUNS_FILE="$CAMPAIGN_DIR/runs.tsv"
mkdir -p "$CAMPAIGN_DIR"
STATIC_PLAN_ACTIVE=""
if [[ -n "$STATIC_PLAN" ]]; then
    STATIC_PLAN_ACTIVE="$CAMPAIGN_DIR/static_collection_plan.json"
    cp -- "$(readlink -f -- "$STATIC_PLAN")" "$STATIC_PLAN_ACTIVE"
    /usr/bin/python3 "$STATIC_PLAN_TOOL" check \
        --plan "$STATIC_PLAN_ACTIVE" --laps "$LAPS" >/dev/null
fi
printf 'lap\tseed\trun_name\tstatus\tweather\ttime_hour\tstatic_obstacle\tdynamic_obstacle\tstarted_utc\tfinished_utc\n' > "$RUNS_FILE"
/usr/bin/python3 - "$CAMPAIGN_DIR/campaign_plan.json" "$CAMPAIGN_ID" "$LAPS" "$BASE_SEED" "$LAP_WALL_SECONDS" "$SIMULATOR_PID" "$SCENARIO" "$SCENARIO_RUNTIME" "$SCENARIO_FOLDER" "$STATIC_PRESET" "$STATIC_PLAN" "$STATIC_PLAN_ACTIVE" <<'PY'
import hashlib, json, pathlib, sys, time
path, campaign_id, laps, seed, seconds, pid, scenario_name, scenario_runtime, scenario_folder, static_preset, static_plan, static_plan_active = sys.argv[1:]
payload = {
    "schema": "morai_sensor_record_campaign_v1",
    "campaign_id": campaign_id,
    "scenario_name": scenario_name,
    "scenario_runtime": scenario_runtime,
    "dataset_scenario_folder": scenario_folder,
    "scenario_runtime_policy": "current team checkout; requires live validation",
    "static_obstacle_preset_request": static_preset or None,
    "static_collection_plan_request": static_plan or None,
    "static_collection_plan_evidence": (
        pathlib.Path(static_plan_active).name if static_plan_active else None
    ),
    "static_collection_plan_sha256": (
        hashlib.sha256(pathlib.Path(static_plan_active).read_bytes()).hexdigest()
        if static_plan_active else None
    ),
    "tunnel_selection_policy": (
        "random_behavior_and_model" if scenario_name == "full_loop_tunnel_vehicle"
        else "random_section_mix_and_model" if scenario_name == "full_loop_tunnel_dynamic"
        else None
    ),
    "requested_laps": int(laps),
    "base_seed": int(seed),
    "lap_wall_limit_seconds": int(seconds),
    "simulator_pid": int(pid),
    "mode": "Simulation Time / internal sensor sync 10Hz / External Sync OFF",
    "weather_candidates": ["SUNNY", "FOGGY"],
    "time_candidates": [11, 13, 15],
    "environment_schedule": (
        "sealed per-run weather/time from static collection plan"
        if static_plan_active else
        "deterministic fresh selection with adjacent-repeat prevention"
    ),
    "static_and_dynamic_obstacles": "main scenario mode selection; deterministic fresh choice for enabled zones per lap",
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
    trap - INT TERM HUP
    if [[ -n "$CURRENT_LAP" ]]; then
        coordination="$CAMPAIGN_ROOT/$CURRENT_RUN/episode_000/sensor_record_coordination"
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
trap on_signal INT TERM HUP

echo "[START] campaign=$CAMPAIGN_ID, scenario=$SCENARIO, laps=$LAPS, runtime=$SCENARIO_RUNTIME, local raw only, storage=$SSD_ROOT, Simulator PID=$SIMULATOR_PID"
echo "        매 회차 선택 시나리오 START reset, Sunny/Foggy, 11/13/15시, 활성 장애물 재선택"
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
    LAP_STATIC_PRESET="$STATIC_PRESET"
    LAP_SCENARIO_WEATHER=""
    LAP_SCENARIO_HOUR=""
    if [[ -n "$STATIC_PLAN_ACTIVE" ]]; then
        LAP_STATIC_PRESET="$(/usr/bin/python3 "$STATIC_PLAN_TOOL" resolve \
            --plan "$STATIC_PLAN_ACTIVE" --lap "$lap" --field preset)"
        LAP_SCENARIO_WEATHER="$(/usr/bin/python3 "$STATIC_PLAN_TOOL" resolve \
            --plan "$STATIC_PLAN_ACTIVE" --lap "$lap" --field weather)"
        LAP_SCENARIO_HOUR="$(/usr/bin/python3 "$STATIC_PLAN_TOOL" resolve \
            --plan "$STATIC_PLAN_ACTIVE" --lap "$lap" --field time_hour)"
        lap_plan_json="$(/usr/bin/python3 "$STATIC_PLAN_TOOL" resolve \
            --plan "$STATIC_PLAN_ACTIVE" --lap "$lap" --field json)"
        echo "[STATIC PLAN] $lap_plan_json"
    fi
    collector_command=(/usr/bin/python3 "$INTEGRATION/tools/run_sensor_record_scenario.py" \
            --name "$CURRENT_RUN" --pid "$SIMULATOR_PID" \
            --run-root "$CAMPAIGN_ROOT/$CURRENT_RUN" \
            --session-evidence-root "$CAMPAIGN_DIR/runtime/$CURRENT_RUN" \
            --scenario "$SCENARIO" --scenario-runtime "$SCENARIO_RUNTIME" \
            --seconds "$LAP_WALL_SECONDS" --full-lap \
            --seed "$seed" --profile-state "$PROFILE_STATE" \
            --raw-camera-transport bridge_dropped --defer-offline-review)
    if [[ -n "$LAP_STATIC_PRESET" ]]; then
        collector_command+=(--static-preset "$LAP_STATIC_PRESET")
    fi
    if [[ -n "$LAP_SCENARIO_WEATHER" ]]; then
        collector_command+=(
            --scenario-weather "$LAP_SCENARIO_WEATHER"
            --scenario-hour "$LAP_SCENARIO_HOUR"
        )
    fi
    "${collector_command[@]}" &
    COLLECTOR_PID=$!
    if wait "$COLLECTOR_PID"; then
        COLLECTOR_PID=""
        finished="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        profile_copy="$CAMPAIGN_ROOT/$CURRENT_RUN/episode_000/campaign_profile_state.json"
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
        if [[ -n "$LAP_SCENARIO_WEATHER" ]] \
                && { [[ "$weather" != "$LAP_SCENARIO_WEATHER" ]] \
                    || [[ "$hour" != "$LAP_SCENARIO_HOUR" ]]; }; then
            printf '%s\t%s\t%s\tfailed_environment_mismatch\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                "$lap" "$seed" "$CURRENT_RUN" "$weather" "$hour" \
                "$static_model" "$dynamic_model" "$started" "$finished" >> "$RUNS_FILE"
            echo "[FATAL] plan 환경과 실제 MORAI 환경이 다릅니다: expected=${LAP_SCENARIO_WEATHER}/${LAP_SCENARIO_HOUR}, actual=${weather}/${hour}" >&2
            echo "        원본과 증거는 보존하며 다음 reset은 진행하지 않습니다." >&2
            exit 2
        fi
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
        echo "[FAIL] lap $lap가 실패했습니다. 진단 증거를 보존하고 실패 payload를 정리합니다." >&2
        /usr/bin/python3 - "$CAMPAIGN_ROOT/$CURRENT_RUN" <<'PY' >&2 || true
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
        if ! /usr/bin/python3 "$INTEGRATION/tools/cleanup_failed_sensor_record_attempt.py" \
                --run-name "$CURRENT_RUN" --failure-status "failed_$code" \
                --run-root "$CAMPAIGN_ROOT/$CURRENT_RUN" \
                --session-evidence-root "$CAMPAIGN_DIR/runtime/$CURRENT_RUN"; then
            echo "[FATAL] lap $lap의 Sensor Record OFF 또는 payload 소유권을 안전하게 확인하지 못했습니다." >&2
            echo "        안전을 위해 다음 reset을 진행하지 않습니다." >&2
            exit 2
        fi
        echo "[CONTINUE] lap $lap는 실패 회차로 기록됐으며 다음 회차를 시도합니다." >&2
        continue
    fi
done

CURRENT_LAP=""
/usr/bin/python3 - "$CAMPAIGN_DIR/COLLECTION_COMPLETE.json" "$CAMPAIGN_ID" "$LAPS" "$SCENARIO" "$SCENARIO_RUNTIME" "$RUNS_FILE" <<'PY'
import csv
import json
import os
import pathlib
import sys
import time

path, campaign_id, attempts, scenario_name, scenario_runtime, runs_path = sys.argv[1:]
attempts = int(attempts)
latest = {}
with open(runs_path, newline="", encoding="utf-8") as source:
    for row in csv.DictReader(source, delimiter="\t"):
        latest[int(row["lap"])] = row
missing = [lap for lap in range(1, attempts + 1) if lap not in latest]
nonterminal = [
    lap for lap in range(1, attempts + 1)
    if lap in latest
    and latest[lap].get("status") not in ("captured", "completed")
    and not latest[lap].get("status", "").startswith("failed_")
]
if missing or nonterminal:
    raise RuntimeError(
        "cannot seal incomplete attempt ledger: missing=%r nonterminal=%r"
        % (missing, nonterminal)
    )
captured = [
    lap for lap in range(1, attempts + 1)
    if latest[lap].get("status") in ("captured", "completed")
]
failed = [
    lap for lap in range(1, attempts + 1)
    if latest[lap].get("status", "").startswith("failed_")
]
payload = {
    "schema": "morai_sensor_record_campaign_completion_v1",
    "campaign_id": campaign_id,
    "scenario_name": scenario_name,
    "scenario_runtime": scenario_runtime,
    "status": "attempts_completed",
    "requested_attempts": attempts,
    "attempted_laps": attempts,
    "captured_laps": len(captured),
    "failed_laps": len(failed),
    "failed_lap_numbers": failed,
    "offline_review_complete": False,
    "completed_wall_ns": time.time_ns(),
    "training_visible": False,
    "physical_simultaneity_verified": False,
}
destination = pathlib.Path(path)
temporary = destination.with_name(destination.name + ".tmp.%d" % os.getpid())
temporary.write_text(json.dumps(payload, sort_keys=True) + "\n")
os.replace(str(temporary), str(destination))
print("[SUMMARY] attempts=%d captured=%d failed=%d failed_laps=%s" % (
    attempts, len(captured), len(failed),
    ",".join(str(value) for value in failed) if failed else "-",
))
PY

echo
echo "[DONE] scenario=$SCENARIO, 요청한 $LAPS회 시도가 끝났습니다. 원본은 아직 학습에 공개되지 않습니다."
echo "검증 실행:"
echo "  cd $MORAI_PROJECT_ROOT"
echo "  ./data_collection/tools/03_validate_internal_sensor_record.sh $CAMPAIGN_ID"
