"""좌회전 신호 수집: 실제 정지 확인 후 직진 초록과 좌회전 화살표를 대조한다.

경로/Expert 원본은 읽기만 한다. C052 한 신호의 상태만 바꾸며 원상복구한다.
신호 명령 ACK를 정지·출발 증거로 사용하지 않고 매 tick의 실제 Ego를 기록한다.
"""
import copy
import csv
import hashlib
import importlib.util
import json
import math
import os
import time
from pathlib import Path
from scenario_runner.project_paths import morai_path

from . import route, tunnel_baseline
from .modes import get_mission_mode
from ...utils.transform_utils import polyline_length


MODE = "full_loop_traffic_light"
SCHEMA = "morai_traffic_light_mission_v1"
CONTRACT_SCHEMA = "morai_traffic_light_collection_contract_v1"
TARGET_LIGHT = "C1256W000052"
RIGHT_LIGHT = "C1256W000123"
START_INDEX = 1031
STOP_INDEX = 1245
EXIT_INDEX = 1297
RIGHT_EXIT_INDEX = 1185
RED_HOLD_SECONDS = 3.0
STRAIGHT_HOLD_SECONDS = 4.0
CHECK_PERIOD_SECONDS = 0.25
READBACK_TIMEOUT_SECONDS = 3.0
READBACK_STABLE_SECONDS = 0.2
READBACK_MINIMUM_READS = 3
READBACK_PERIOD_SECONDS = 0.05
MAXIMUM_SIGNAL_GAP_SECONDS = 0.75
_NO_ACTUAL = object()
PROJECT = Path(__file__).resolve().parents[3]


def enabled(context):
    return getattr(getattr(context, "mission_mode", None), "name", None) == MODE


def default_contract():
    sources = (
        "scenario_runner/scenarios/segment_modes/traffic_light.py",
        "scenario_runner/scenarios/segment_modes/scenario.py",
        "scenario_runner/scenarios/segment_modes/modes.py",
        "scenario_runner/scenarios/segment_modes/route.py",
        "scenario_runner/scenarios/segment_modes/events.py",
        "scenario_runner/scenarios/segment_modes/expert.py",
        "scenario_runner/scenarios/segment_modes/tunnel_baseline.py",
        "scenario_runner/scenarios/segment_modes/tunnel_static_startup.py",
        "scenario_runner/adapters/expert_controller.py",
        "scenario_runner/adapters/morai_grpc.py",
        "data_collection/third_party/grpc_inha_univ/src/api/simulation_world.py",
        "data_collection/third_party/grpc_inha_univ/morai_standard_proto/morai/infrastructure/traffic_light.proto",
        "../AIM2026/Expert/src/planning/traffic/stop_controller.py",
        "../AIM2026/Expert/src/planning/traffic/route_targets.py",
        "../AIM2026/Expert/src/config/config.json",
        "../AIM2026/Expert/src/config/config.py",
        "scenario_runner/data/R_KR_PR_K-city_2025/link_set.json",
        "scenario_runner/data/R_KR_PR_K-city_2025/node_set.json",
        "scenario_runner/data/R_KR_PR_K-city_2025/2026_molit_comp_global_path.txt",
        "scenario_runner/config/full_loop_links.yaml",
    )
    return dict(
        schema=CONTRACT_SCHEMA, scenario_name=MODE, target_light_id=TARGET_LIGHT,
        command_colors=[1, 16, 33], set_sibling=False, impulse=False,
        readback_timeout_wall_seconds=READBACK_TIMEOUT_SECONDS,
        readback_stable_wall_seconds=READBACK_STABLE_SECONDS,
        readback_minimum_reads=READBACK_MINIMUM_READS,
        maximum_signal_gap_seconds=MAXIMUM_SIGNAL_GAP_SECONDS,
        red_hold_seconds=RED_HOLD_SECONDS, straight_green_hold_seconds=STRAIGHT_HOLD_SECONDS,
        stopped_speed_mps=0.3, stop_window_m=[5.0, 30.0],
        start_global_index=START_INDEX, end_global_index=1348,
        left_stop_global_index=STOP_INDEX, left_exit_global_index=EXIT_INDEX,
        right_exit_global_index=RIGHT_EXIT_INDEX,
        color_head_target_policy="pure_red_green_v1", separate_route_permission_gt=True,
        source_sha256={name: hashlib.sha256(morai_path(name).read_bytes()).hexdigest()
                       for name in sources},
    )


def configure(config):
    config.update(mission_mode=MODE, drive_control_mode="aimdrive_expert",
                  clean_road_mode=False, auto_repeat=False, max_laps=1,
                  timeout_sec=180.0, minimum_completion_progress_ratio=0.95,
                  expert_max_velocity_kmh=30.0, expert_max_cross_track_error_m=1.5,
                  off_route_timeout_sec=3.0, off_route_grace_sec=2.0)
    config["progress_watchdog"] = dict(enabled=True, timeout_sec=45.0,
                                        startup_grace_sec=30.0, minimum_progress_m=2.0)
    config.setdefault("traffic_light_wait_override", {})["enabled"] = False
    # The collector seals extra source files. Preserve that exact contract.
    config.setdefault("_traffic_light_collection", default_contract())


def _persist(context):
    identity = str(getattr(context, "run_instance_id", "unstarted"))
    if Path(identity).name != identity or identity in ("", ".", ".."):
        raise RuntimeError("신호 수집 실행 ID가 올바르지 않습니다")
    runtime = context.global_cfg.get("paths", {}).get("runtime_root") or PROJECT / "scenario_runner/.runtime"
    path = Path(runtime) / "traffic_light" / identity / "mission_report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    context.traffic_light_report["report_path"] = str(path)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(context.traffic_light_report, ensure_ascii=False,
                                    indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def prepare(context):
    if not enabled(context):
        return
    expected = default_contract()
    contract = context.cfg.get("_traffic_light_collection", expected)
    if any(contract.get(key) != value for key, value in expected.items() if key != "source_sha256"):
        raise RuntimeError("좌회전 신호 수집 계약이 기본 행동 계약과 다릅니다")
    pins = contract.get("source_sha256", {})
    if any(pins.get(name) != sha for name, sha in expected["source_sha256"].items()):
        raise RuntimeError("좌회전 신호 수집 소스가 준비 당시 버전과 다릅니다")
    window, mode = context.mission_route, get_mission_mode(MODE)
    if window is None or (window.first_index, window.last_index) != (START_INDEX, 1348):
        raise RuntimeError("좌회전 신호 수집 경로의 시작/종료가 다릅니다")
    route.validate_window(window.full_points, mode)
    stop = context.map_loader.node_set["A1256W000742"]
    if not stop.get("on_stop_line") or stop.get("traffic_light_id") != TARGET_LIGHT:
        raise RuntimeError("좌회전 신호의 실제 정지선 ID가 다릅니다")
    if math.dist(window.full_points[STOP_INDEX][:2], stop["point"][:2]) > 1e-3:
        raise RuntimeError("좌회전 정지선이 원본 경로와 정합하지 않습니다")
    following = context.map_loader.get_link("A2256W000083")
    if str(following.get("related_signal", "")).lower() != "left":
        raise RuntimeError("정지선 이후 경로가 좌회전 링크가 아닙니다")
    stop_s = polyline_length(window.full_points[START_INDEX:STOP_INDEX + 1])
    exit_s = polyline_length(window.full_points[START_INDEX:EXIT_INDEX + 1])
    if not 0.0 < stop_s < exit_s < context.route_length_m:
        raise RuntimeError("신호 정지선/좌회전 종료 위치가 수집 경로 밖입니다")
    context.traffic_light_report = dict(
        schema=SCHEMA, scenario=MODE, status="prepared", phase="red_approach",
        contract=copy.deepcopy(contract), failure_reason="", commands=[], observations=[],
        restore=None, original=None, red_stop_verified=False,
        straight_green_stop_verified=False, left_arrow_departure_verified=False,
        left_turn_completed=False, empty_road_verified=False, maximum_blocking_actors=0,
        start_pose_verified=False, startup_observation=None, right_turn_completed=False,
        last_blocking_actors=[], actor_inventory_checks=0, collision_count=0,
        expert_runtime_verified=False, clock_domain=str(getattr(context.clock, "domain", "unknown")),
        physical_simultaneity_verified=False, physical_semantic_validation_completed=False,
        route=dict(start_global_index=START_INDEX, end_global_index=1348,
                   left_stop_global_index=STOP_INDEX, left_exit_global_index=EXIT_INDEX,
                   right_exit_global_index=RIGHT_EXIT_INDEX,
                   right_exit_route_s_m=polyline_length(window.full_points[START_INDEX:RIGHT_EXIT_INDEX + 1]),
                   left_stop_node_id="A1256W000742", left_stop_signal_id=TARGET_LIGHT,
                   left_stop_route_s_m=stop_s, left_exit_route_s_m=exit_s,
                   length_m=context.route_length_m, after_left_exit_m=context.route_length_m - exit_s),
    )
    context.traffic_light_last_check = None
    context.traffic_light_hold_since = None
    context.traffic_light_straight_start_s = None
    context.traffic_light_pending_command = None
    context.traffic_light_signal_missing_since = None
    _persist(context)


def verify_runtime(context):
    """실제로 export한 경로를 실제 Expert 바인더로 검사한 후에만 프로세스를 시작한다."""
    if not enabled(context):
        return
    controller = context.expert_controller
    config = json.loads(Path(controller.runtime_config_path).read_text(encoding="utf-8"))
    velocity = config["planning"]["velocity_profile"]
    if float(velocity["max_velocity"]) != 30.0 or velocity.get("speed_limit_zones"):
        raise RuntimeError("신호 구간 Expert의 실제 30km/h 상한/경로 speed zone 설정이 다릅니다")
    policy = config["planning"]["traffic_light_stop"]
    if (not policy.get("enabled", True) or RIGHT_LIGHT not in policy.get("ignored_signal_ids", [])
            or TARGET_LIGHT not in policy.get("left_arrow_only_signal_ids", [])):
        raise RuntimeError("Expert의 우회전 ignore/좌회전 화살표 전용 정책이 누락됐습니다")
    mapping = config["map"]
    ordered = context.route_definition["route_links"]
    if (len(ordered) != 39 or mapping.get("route_signal_route_links") != ordered
            or mapping.get("is_closed_path") is not False):
        raise RuntimeError("Expert가 공식 순서 경로/열린 구간 경로를 사용하지 않습니다")
    with open(controller.runtime_path_csv, encoding="utf-8", newline="") as stream:
        actual = [(float(row["x"]), float(row["y"])) for row in csv.DictReader(stream)]
    expected = route.build_expert_route_points(context)
    if len(actual) != len(expected) or any(math.dist(a, b[:2]) > 1e-6 for a, b in zip(actual, expected)):
        raise RuntimeError("Expert의 실제 경로 파일이 수집 경로+35m lookahead와 다릅니다")
    source = Path(context.global_cfg["paths"]["expert_root"]) / "src/planning/traffic/route_targets.py"
    spec = importlib.util.spec_from_file_location("_traffic_mission_actual_route_binding", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    doc = module.build_route_signal_targets(actual, mapping["data_dir"], ordered, False)
    targets = {row["signal_id"]: row for row in doc["targets"]}
    if (doc["route_slice_start_index"] != START_INDEX or set(targets) != {RIGHT_LIGHT, TARGET_LIGHT}
            or targets[TARGET_LIGHT]["path_index"] != STOP_INDEX - START_INDEX
            or targets[TARGET_LIGHT]["signal_movement"] != "left"
            or targets[TARGET_LIGHT]["node_id"] != "A1256W000742"
            or targets[RIGHT_LIGHT]["path_index"] != 1123 - START_INDEX):
        raise RuntimeError("Expert의 신호 ID/좌회전 정지선 바인딩이 실제 경로와 다릅니다")
    context.traffic_light_report.update(expert_runtime_verified=True, expert_signal_targets=doc,
                                       expert_runtime_config_sha256=hashlib.sha256(
                                           Path(controller.runtime_config_path).read_bytes()).hexdigest())
    _persist(context)


def record_start(context):
    """Sensor Record 준비 대기가 끝난 뒤, Expert launch 바로 전에 출발점을 읽는다."""
    if not enabled(context):
        return
    report = context.traffic_light_report
    if not report.get("expert_runtime_verified") or report["observations"]:
        raise RuntimeError("Expert 경로 검증 전 또는 주행 중에는 START 관측을 갱신할 수 없습니다")
    getter = getattr(context, "get_ego_motion_state", context.grpc.get_ego_motion_state)
    ego = getter()
    x, y, speed = float(ego["x"]), float(ego["y"]), float(ego["speed"]) / 3.6
    now = float(context.clock.now())
    distance = math.dist((x, y), context.route_points[0][:2])
    if not all(math.isfinite(v) for v in (x, y, speed, now)) or not 0.0 <= speed <= 0.3 or distance > 1.0:
        raise RuntimeError("Expert 시작 전에 실제 Ego가 START에서 정지하지 않았습니다")
    _, progress = tunnel_baseline._projection(context.route_points, x, y)
    context.traffic_light_report.update(start_pose_verified=True,
        startup_observation=dict(phase="startup", scenario_time=now, route_s=progress,
                                 ego_xy=[x, y], speed_mps=speed, distance_to_start_m=distance))
    _persist(context)


def _inventory(context):
    states, ego_id = tunnel_baseline._inventory(context)
    blockers = []
    for state in states:
        identity = str(state.actor_info.id.value)
        if identity == ego_id:
            continue
        box = getattr(state, "bounding_box", None)
        if box is None:
            raise RuntimeError("도로 점유 검사에 bbox가 없습니다: " + identity)
        center = tunnel_baseline._xyz(box.location, "신호 수집 actor 중심")
        extent = tunnel_baseline._xyz(box.extent, "신호 수집 actor 크기")
        if min(extent) <= 0.0:
            raise RuntimeError("도로 점유 검사 bbox 크기가 올바르지 않습니다: " + identity)
        distance, _ = tunnel_baseline._projection(context.route_points, *center[:2])
        if distance <= 2.5 + math.hypot(*extent[:2]) + 0.5:
            blockers.append(dict(actor_id=identity, center_xyz=center, route_distance_m=distance))
    report = context.traffic_light_report
    report["actor_inventory_checks"] += 1
    report["last_blocking_actors"] = blockers
    report["maximum_blocking_actors"] = max(report["maximum_blocking_actors"], len(blockers))
    report["empty_road_verified"] = not blockers and report["maximum_blocking_actors"] == 0
    if blockers:
        raise RuntimeError("신호 단독 수집 경로에 다른 actor가 있습니다: " + repr(blockers))


def _read_signal(context, *, allow_unavailable=False):
    state = context.grpc.get_traffic_light_state_by_id(TARGET_LIGHT)
    if state is None and allow_unavailable:
        return None
    if (not isinstance(state, dict) or state.get("id") != TARGET_LIGHT
            or type(state.get("color")) is not int or state["color"] < 0):
        raise RuntimeError("정확한 좌회전 신호 상태를 읽지 못했습니다: " + repr(state))
    return copy.deepcopy(state)


def _issue_signal_change(context, color, *, restoring=False):
    """ACK와 적용 확인을 분리한다. 명령은 한 번만 보내고 이후 조회만 한다."""
    wall = time.monotonic()
    record = dict(id=TARGET_LIGHT, target_color=int(color), accepted=False, verified=False,
                  before=None, after=None, immediate_after=None, error=None,
                  impulse=False, set_sibling=False, scenario_time=float(context.clock.now()),
                  command_wall_monotonic=wall, readback_observations=[], readback_stable_seconds=0.,
                  readback_consecutive=0, readback_since_wall=None, status="pending",
                  allowed_colors=[32, 33, 48] if color == 33 and not restoring else [int(color)])
    try:
        result = context.grpc.set_traffic_light_color_verified(
            TARGET_LIGHT, int(color), impulse=False, set_sibling=False)
        record["setter_result"] = copy.deepcopy(result)
        if isinstance(result, dict):
            record.update(accepted=result.get("accepted") is True,
                          before=copy.deepcopy(result.get("before")),
                          immediate_after=copy.deepcopy(result.get("after")),
                          error=result.get("error"))
            if result.get("id") != TARGET_LIGHT:
                record.update(accepted=False, error="신호 명령 응답 ID가 다릅니다")
    except Exception as error:
        record["error"] = str(error) or type(error).__name__
    return record


def _poll_signal_change(context, record, *, actual=_NO_ACTUAL):
    """다음 tick의 실제 상태를 기다리며 연속 3회·0.2초 안정 상태를 요구한다."""
    if record["verified"]:
        return True
    if (not record["accepted"] or not isinstance(record.get("before"), dict)
            or record["before"].get("id") != TARGET_LIGHT):
        record.update(status="failed", error=record["error"] or "신호 변경 명령이 수락되지 않았습니다")
        raise RuntimeError(record["error"])
    wall = time.monotonic()
    read_error = None
    if actual is _NO_ACTUAL:
        try:
            actual = _read_signal(context)
        except RuntimeError as error:
            read_error = str(error)
    valid = (isinstance(actual, dict) and actual.get("id") == TARGET_LIGHT
             and type(actual.get("color")) is int and actual["color"] in record["allowed_colors"])
    row = dict(wall_monotonic=wall, id=actual.get("id") if isinstance(actual, dict) else None,
               color=actual.get("color") if isinstance(actual, dict) else None)
    if read_error:
        row["read_error"] = read_error
    record["readback_observations"].append(row)
    record["after"] = copy.deepcopy(actual)
    if valid:
        if record["readback_since_wall"] is None:
            record["readback_since_wall"] = wall
        record["readback_consecutive"] += 1
        record["readback_stable_seconds"] = wall - record["readback_since_wall"]
    else:
        record.update(readback_since_wall=None, readback_consecutive=0, readback_stable_seconds=0.)
    within_deadline = wall - record["command_wall_monotonic"] <= READBACK_TIMEOUT_SECONDS
    if (within_deadline and record["readback_consecutive"] >= READBACK_MINIMUM_READS
            and record["readback_stable_seconds"] >= READBACK_STABLE_SECONDS - 1e-8):
        record.update(verified=True, status="verified", verified_scenario_time=float(context.clock.now()))
        print("[TrafficLight] 신호 적용 확인: {} → {}".format(TARGET_LIGHT, actual["color"]))
        return True
    if not within_deadline:
        record.update(status="failed", error="좌회전 신호 적용 시간 초과: target={} / 실제={}".format(
            record["target_color"], record["after"]))
        raise RuntimeError(record["error"])
    return False


def _wait_signal_change(context, record):
    while not _poll_signal_change(context, record):
        time.sleep(READBACK_PERIOD_SECONDS)


def _command(context, color, *, wait=True):
    report = context.traffic_light_report
    if report["original"] is None:
        report["original"] = _read_signal(context)
        _persist(context)
    record = _issue_signal_change(context, color)
    report["commands"].append(record)
    context.traffic_light_pending_command = None if wait else record
    try:
        if wait:
            _wait_signal_change(context, record)
    finally:
        _persist(context)
    return record


def before_drive(context):
    if not enabled(context):
        return
    _inventory(context)
    _command(context, 1)
    context.traffic_light_report["status"] = "running"
    _persist(context)


def observe(context, elapsed, ego_state, current_s):
    if not enabled(context):
        return
    now = float(context.clock.now())
    previous = context.traffic_light_last_check
    if previous is not None and now - previous < CHECK_PERIOD_SECONDS:
        if now < previous:
            raise RuntimeError("신호 수집 시계가 역행했습니다")
        return
    if previous is not None and now - previous > 1.5:
        raise RuntimeError("실제 신호/차량 관측이 1.5초 이상 끊겼습니다")
    speed = float(ego_state["speed"]) / 3.6  # Runner/grpc/ROS motion contract: km/h.
    x, y = float(ego_state["x"]), float(ego_state["y"])
    if not all(math.isfinite(v) for v in (now, speed, x, y, float(elapsed), float(current_s))) or speed < 0.0:
        raise RuntimeError("실제 신호 수집 Ego 관측이 유효하지 않습니다")
    distance, progress = tunnel_baseline._projection(context.route_points, x, y)
    if distance > 1.5 or abs(progress - float(current_s)) > 2.0:
        raise RuntimeError("실제 Ego 위치와 수집 경로 진행량이 일치하지 않습니다")
    report = context.traffic_light_report
    prior = report["observations"][-1] if report["observations"] else report.get("startup_observation")
    if not report.get("start_pose_verified") or not isinstance(prior, dict):
        raise RuntimeError("실제 Ego START 관측이 없어 주행을 검증할 수 없습니다")
    dt = now - prior["scenario_time"]
    allowed_motion = 3.0 + 1.5 * max(speed, prior["speed_mps"]) * max(0.0, dt)
    if (dt < 0.0 or progress < prior["route_s"] - 1.0
            or abs(progress - prior["route_s"]) > allowed_motion
            or math.dist((x, y), prior["ego_xy"]) > allowed_motion):
        raise RuntimeError("연속 Ego 관측의 위치 이동이 실제 속도/시간과 일치하지 않습니다")
    collisions = [str(value) for value in ego_state.get("collision_objects", []) if str(value).strip()]
    report["collision_count"] += len(collisions)
    if collisions:
        _persist(context)
        raise RuntimeError("신호 수집 중 실제 충돌을 관측했습니다: " + repr(collisions))
    _inventory(context)
    signal = _read_signal(context, allow_unavailable=True)
    phase = report["phase"]
    row = dict(scenario_time=now, elapsed_seconds=float(elapsed), phase=phase,
               signal_id=signal["id"] if signal is not None else None,
               signal_color=signal["color"] if signal is not None else None, speed_mps=speed,
               route_s=progress, ego_xy=[x, y], cross_track_m=distance)
    report["observations"].append(row)
    context.traffic_light_last_check = now
    if progress >= report["route"]["right_exit_route_s_m"]:
        report["right_turn_completed"] = True
    stop_s, exit_s = (report["route"][key] for key in ("left_stop_route_s_m", "left_exit_route_s_m"))
    if signal is None:
        row["signal_read_unavailable"] = True
        report.setdefault("signal_read_gaps", []).append(dict(scenario_time=now, phase=phase))
        if context.traffic_light_signal_missing_since is None:
            context.traffic_light_signal_missing_since = now
        missing_seconds = now - context.traffic_light_signal_missing_since
        # No previous color is substituted, and missing time never extends a hold.
        if phase in ("red_approach", "red_hold", "straight_hold"):
            context.traffic_light_hold_since = None
            if phase == "red_hold":
                report["phase"] = "red_approach"
        if context.traffic_light_pending_command is not None:
            _poll_signal_change(context, context.traffic_light_pending_command, actual=None)
        _persist(context)
        if missing_seconds > MAXIMUM_SIGNAL_GAP_SECONDS:
            raise RuntimeError("좌회전 신호 실제 조회가 0.75초 이상 연속 누락됐습니다")
        return
    context.traffic_light_signal_missing_since = None
    if phase in ("red_approach", "red_hold", "straight_hold"):
        expected = 16 if phase == "straight_hold" else 1
        if signal["color"] != expected:
            raise RuntimeError("수집 단계 중 좌회전 신호가 다른 상태로 바뀌었습니다")
        if progress > stop_s + 0.5:
            raise RuntimeError("빨강/직진 초록에서 좌회전 정지선을 통과했습니다")
        stationary = speed <= 0.3 and 5.0 <= stop_s - progress <= 30.0
        if phase == "straight_hold":
            if not stationary or abs(progress - context.traffic_light_straight_start_s) > 0.5:
                raise RuntimeError("좌회전 대기 중 직진 초록에서 Expert가 출발했습니다")
            if context.traffic_light_hold_since is None:
                context.traffic_light_hold_since = now
            if now - context.traffic_light_hold_since >= STRAIGHT_HOLD_SECONDS:
                report["straight_green_stop_verified"] = True
                _command(context, 33, wait=False)
                report["phase"] = "arrow_pending"
                context.traffic_light_hold_since = None
        elif stationary:
            if context.traffic_light_hold_since is None:
                context.traffic_light_hold_since = now
                report["phase"] = "red_hold"
            if now - context.traffic_light_hold_since >= RED_HOLD_SECONDS:
                report["red_stop_verified"] = True
                context.traffic_light_straight_start_s = progress
                _command(context, 16, wait=False)
                report["phase"] = "straight_pending"
                context.traffic_light_hold_since = None
        else:
            context.traffic_light_hold_since = None
            report["phase"] = "red_approach"
    elif phase in ("straight_pending", "arrow_pending"):
        if phase == "straight_pending":
            if (speed > 0.3 or not 5.0 <= stop_s - progress <= 30.0
                    or abs(progress - context.traffic_light_straight_start_s) > 0.5):
                raise RuntimeError("좌회전 대기 중 직진 초록 적용 전에 Expert가 출발했습니다")
        record = context.traffic_light_pending_command
        if record is None:
            raise RuntimeError("신호 적용을 기다리는 명령 기록이 없습니다")
        if (phase == "arrow_pending" and signal["color"] not in record["allowed_colors"]
                and (speed > 0.3 or abs(progress - context.traffic_light_straight_start_s) > 0.5)):
            raise RuntimeError("좌회전 허용 신호가 실제로 적용되기 전에 Expert가 출발했습니다")
        if _poll_signal_change(context, record, actual=signal):
            report["phase"] = "straight_hold" if phase == "straight_pending" else "left_go"
            context.traffic_light_pending_command = None
            context.traffic_light_hold_since = None
    elif phase in ("left_go", "completed"):
        if signal["color"] not in (32, 33, 48):
            raise RuntimeError("좌회전 출발 단계에서 허용 신호가 사라졌습니다")
        if speed > 0.5 and progress > stop_s + 0.5:
            report["left_arrow_departure_verified"] = True
        if progress >= exit_s and report["left_arrow_departure_verified"]:
            report["left_turn_completed"] = True
            report["phase"] = "completed"
    else:
        raise RuntimeError("알 수 없는 신호 수집 단계: " + str(phase))
    _persist(context)


def finish(context, status, error=None):
    if not enabled(context) or not hasattr(context, "traffic_light_report"):
        return
    report = context.traffic_light_report
    requested_pass = status == "passed"
    flags = ("red_stop_verified", "straight_green_stop_verified", "left_arrow_departure_verified",
             "left_turn_completed", "expert_runtime_verified", "empty_road_verified",
             "start_pose_verified", "right_turn_completed")
    if status == "passed" and (report["status"] == "failed" or not all(report[key] for key in flags)
                               or report["actor_inventory_checks"] < 2 or report["collision_count"] != 0):
        status, error = "failed", report["failure_reason"] or "신호 대기/출발/좌회전 실제 관측이 완성되지 않았습니다"
    report["status"] = status
    if error:
        report.setdefault("failure_events", []).append(str(error))
        report["failure_reason"] = report["failure_reason"] or str(error)
    _persist(context)
    if requested_pass and status == "failed":
        raise RuntimeError(error)


def _stop_before_restore(context):
    """기록 종료 후 소유 Expert를 끄고 AUTO/D에서 실제 정지를 확인한다."""
    from . import tunnel_static_startup as startup
    record = dict(checks=[], final_verified=False, velocity_reset_used=False,
                  policy="common_auto_d_ros_brake")
    context.traffic_light_report["final_stop"] = record
    context.stop_drive_control()
    try:
        startup.hold_common_brake(context.grpc)
        def read(reference=None):
            return startup.common_stopped_readback(context.grpc, reference)[1]
        stopped = tunnel_baseline._wait_parking_stage(
            record, "physical_stop", read, now=time.monotonic, sleep=time.sleep,
            timeout_sec=20.0, minimum_seconds=1.5)
        startup.release_common_brake(context.grpc)
        startup._require_no_controller()
        final = tunnel_baseline._wait_parking_stage(
            record, "post_release", lambda: read(stopped), now=time.monotonic, sleep=time.sleep,
            timeout_sec=3.0, minimum_stable_seconds=0.5)
        record.update(final_verified=True, actual_final=final)
    finally:
        startup.release_common_brake(context.grpc)


def restore(context):
    if not enabled(context) or not hasattr(context, "traffic_light_report"):
        return
    report = context.traffic_light_report
    original = report.get("original")
    if original is None or (isinstance(report.get("restore"), dict)
                            and report["restore"].get("verified")):
        return
    failure = None
    try:
        _stop_before_restore(context)
    except (Exception, KeyboardInterrupt) as error:
        failure = error
    record = None
    try:
        record = _issue_signal_change(context, original["color"], restoring=True)
        record.update(original_color=original["color"], original=copy.deepcopy(original),
                      scope="exact_signal_color_only", automatic_schedule_restored=False)
        report["restore"] = record
        _wait_signal_change(context, record)
    except (Exception, KeyboardInterrupt) as error:
        if record is None:
            record = dict(id=TARGET_LIGHT, original_color=original["color"], original=copy.deepcopy(original),
                          scope="exact_signal_color_only", automatic_schedule_restored=False)
            report["restore"] = record
        record.update(verified=False, error=str(error) or type(error).__name__, status="failed")
        failure = failure or error
    finally:
        if failure is not None:
            detail = str(failure) or type(failure).__name__
            report.update(status="failed", restoration_error=detail)
            report["failure_reason"] = report["failure_reason"] or detail
            report.setdefault("failure_events", []).append(detail)
        _persist(context)
    if failure is not None:
        raise failure
