"""실제 Expert의 임시 경로·설정과 좌회전 신호 제어를 연결해 오프라인 검증한다.

MORAI/ROS 연결, Expert 프로세스 및 모델은 시작하지 않는다. 원본 경로와
Expert 코드는 읽기만 하며 생성한 runtime 파일은 /tmp에만 둔다.
"""
import copy
import hashlib
import importlib
import io
import json
import math
import os
import sys
import tempfile
from contextlib import contextmanager, redirect_stdout
from pathlib import Path
from types import SimpleNamespace

import yaml

from ..adapters.expert_controller import ExpertController
from ..scenarios.segment_modes import expert as mission_expert
from ..scenarios.segment_modes.modes import get_mission_mode
from ..scenarios.segment_modes.route import (
    RouteWindow, build_expert_route_points, validate_window)
from ..utils.map_loader import MGeoMapLoader
from ..utils.transform_utils import polyline_length


MODE = "full_loop_traffic_light"
RIGHT_SIGNAL = "C1256W000123"
LEFT_SIGNAL = "C1256W000052"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load_points(path):
    return tuple(tuple(map(float, row.split())) for row in Path(path).read_text(
        encoding="utf-8-sig").splitlines() if row.strip() and not row.lstrip().startswith("#"))


@contextmanager
def _expert_scope(source, environment):
    """호출자의 기존 Expert 모듈·환경·검색 경로를 그대로 되돌린다."""
    old_path = list(sys.path)
    old_modules = {key: value for key, value in sys.modules.items()
                   if key == "AIM2026.Expert" or key.startswith("AIM2026.Expert.")}
    old_environment = {key: os.environ.get(key) for key in environment}
    for key in old_modules:
        del sys.modules[key]
    sys.path.insert(0, str(Path(source).resolve().parents[2]))
    os.environ.update(environment)
    try:
        yield
    finally:
        for key in list(sys.modules):
            if key == "AIM2026.Expert" or key.startswith("AIM2026.Expert."):
                del sys.modules[key]
        sys.modules.update(old_modules)
        sys.path[:] = old_path
        for key, value in old_environment.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def verify_traffic_light_expert(global_cfg, scenario_cfg):
    """실제 mission crop을 export한 뒤 실제 Expert의 신호 판정 결과를 반환한다."""
    paths, cfg = copy.deepcopy(global_cfg["paths"]), copy.deepcopy(scenario_cfg)
    mode = get_mission_mode(MODE)
    points = _load_points(cfg["global_path"])
    validate_window(points, mode)
    first, last = mode.start.index, mode.end.index
    definition = yaml.safe_load(Path(cfg["route_definition_path"]).read_text(
        encoding="utf-8-sig"))[cfg.get("route_definition_key", "full_loop")]
    route_links = definition["route_links"]
    _require(len(route_links) == 39, "공식 순서 경로의 39개 링크 계약이 달라졌습니다.")
    _require(MODE in mission_expert.FILTERED_MISSIONS,
             "신호 mission이 실제 Expert speed-zone 필터에 등록되지 않았습니다.")
    _require(not any(vars(mode.zones).values()), "신호 수집에서 장애물/NPC zone이 켜져 있습니다.")
    expert_root = Path(paths["expert_root"]).resolve()
    expert_source = expert_root / "src"
    source_files = [expert_source / relative for relative in (
        "config/config.json", "config/config.py", "planning/traffic/stop_controller.py",
        "planning/traffic/route_targets.py", "path/mapping.py", "localization/point.py")]
    source_files += [Path(cfg["global_path"]), Path(cfg["route_definition_path"])]
    before = {str(path): _digest(path) for path in source_files}
    logs = io.StringIO()
    with tempfile.TemporaryDirectory(prefix="morai-traffic-light-expert-", dir="/tmp") as temporary:
        with redirect_stdout(logs):
            window = RouteWindow(points, polyline_length(points[:first + 1]), first, last)
            context = SimpleNamespace(
                mission_mode=mode, mission_route=window,
                route_points=list(points[first:last + 1]),
                map_loader=MGeoMapLoader(paths["mgeo_root"]),
            )
            runtime_points = build_expert_route_points(context)
            controller = ExpertController(
                expert_root=str(expert_root), runtime_root=temporary,
                map_name=cfg.get("expert_map_name", Path(paths["mgeo_root"]).name),
                mgeo_root=paths["mgeo_root"], route_points=runtime_points,
                route_signal_route_links=route_links,
                start_link=mode.start.link_id, end_link=mode.end.link_id,
                max_velocity_kmh=cfg.get("expert_max_velocity_kmh", 60.0),
                traffic_light_control=cfg.get("expert_traffic_light_control", False),
                is_closed_path=False,
                velocity_profile_window_size=cfg.get("expert_velocity_profile_window_size"),
            )
            context.expert_controller = controller
            mission_expert.prepare_speed_zones(context)
            _require(controller.process is None, "검증 중 Expert 프로세스가 생성되었습니다.")
            runtime_config = json.loads(Path(controller.runtime_config_path).read_text())
            _require(runtime_config["map"].get("route_signal_route_links") == route_links,
                     "임시 Expert 설정에 전체 순서 경로가 전달되지 않았습니다.")
            _require(runtime_config["map"]["is_closed_path"] is False,
                     "구간 Expert 경로가 closed path로 처리되었습니다.")
            environment = {
                "GT_BEV_CONFIG_PATH": controller.runtime_config_path,
                "GT_BEV_PATH_CSV": controller.runtime_path_csv,
                "GT_BEV_MAP_DIR": controller.mgeo_root,
                "GT_BEV_DIRECT_USE_ENV_CONFIG": "1",
            }
            with _expert_scope(expert_source, environment):
                Config = importlib.import_module("AIM2026.Expert.src.config.config").Config
                StopController = importlib.import_module(
                    "AIM2026.Expert.src.planning.traffic.stop_controller").TrafficLightStopController
                actual = Config()
                signal_document = actual["map"]["route_signal_targets"]
                policies = copy.deepcopy(actual["planning"]["traffic_light_stop"])
                _require(signal_document["route_slice_start_index"] == first,
                         "실제 Expert의 원본 route slice 시작점이 다릅니다.")
                _require(signal_document["path_points"] == len(runtime_points),
                         "실제 Expert 경로 포인트 수가 export와 다릅니다.")
                by_id = {row["signal_id"]: row for row in signal_document["targets"]}
                _require(set(by_id) == {RIGHT_SIGNAL, LEFT_SIGNAL},
                         "새 구간에 다른 정지선이 섞이거나 신호 정지선이 누락됐습니다.")
                right, left = by_id[RIGHT_SIGNAL], by_id[LEFT_SIGNAL]
                _require(right["node_id"] == "A1256W000757" and right["path_index"] == 1123 - first
                         and right["related_signal_raw"] == "right_unprotected",
                         "우회전 정지선의 원본 node/movement 바인딩이 다릅니다.")
                _require(left["node_id"] == "A1256W000742" and left["path_index"] == 1245 - first
                         and left["signal_movement"] == "left"
                         and left["signal_route_link_id"] == "A2256W000083",
                         "좌회전 정지선의 원본 node/movement 바인딩이 다릅니다.")
                _require(RIGHT_SIGNAL in policies.get("ignored_signal_ids", [])
                         and LEFT_SIGNAL in policies.get("left_arrow_only_signal_ids", []),
                         "현재 Expert의 우회전 ignore/좌회전 화살표 정책이 없습니다.")
                _require(float(policies["stop_margin"]) == 13.0,
                         "요청한 Expert의 13m stop margin 계약이 다릅니다.")

                def subject(**changes):
                    settings = dict(policies, **changes)
                    return StopController(
                        path=actual["map"]["path"],
                        stopline_mapper=actual["map"]["stopline_mapper"],
                        route_signal_targets=signal_document, is_closed_path=False, **settings)

                def sample(status, signal_id=LEFT_SIGNAL, source=100.0, reference=100.1):
                    return [signal_id, status, 2, 0.0, source, reference]

                planned = controller.max_velocity_kmh / 3.6
                index = left["path_index"]
                phase_results = []
                for status in (1, 4, 16, 32, 33, 36, 48):
                    cap = subject().get_target_velocity(sample(status), index, planned)
                    permitted = status in (32, 33, 36, 48)
                    _require(math.isclose(cap, planned if permitted else 0.0, abs_tol=1e-9),
                             "실제 Expert 좌회전 phase 판정이 다릅니다: raw=%s" % status)
                    phase_results.append({"raw_signal_bits": status,
                                          "left_permitted": permitted, "velocity_cap_mps": cap})
                unavailable_results = []
                for name, packet in (
                    ("미수신", None), ("빈 수신", []),
                    ("다른 신호 ID", sample(32, RIGHT_SIGNAL)),
                    ("오래된 신호", sample(32, reference=100.0 + float(policies["signal_timeout"]) + 1)),
                    ("미래 시각 신호", sample(32, source=101.0, reference=100.0)),
                ):
                    control = subject(hold_on_stale=False)
                    cap = control.get_target_velocity(packet, index, planned)
                    _require(cap == 0.0 and control.last_decision["signal_id"] == LEFT_SIGNAL,
                             "미확인 신호로 좌회전이 해제되었습니다: " + name)
                    unavailable_results.append({"case": name, "velocity_cap_mps": cap})
                control = subject()
                ignored_cap = control.get_target_velocity(sample(1, RIGHT_SIGNAL), right["path_index"], planned)
                _require(control.last_decision is None
                         or control.last_decision["signal_id"] != RIGHT_SIGNAL,
                         "우회전 ignore 대상이 정지 원인으로 적용되었습니다.")
                _require(ignored_cap > 0.0, "다음 좌회전 신호를 이유로 우회전 구간에서 즉시 정지합니다.")

                control = subject()
                path_s, target_s = control._path_s, control._path_s[index]
                margin = float(policies["stop_margin"])
                sample_indices = [min(range(index + 1),
                                      key=lambda i: abs(target_s - path_s[i] - distance))
                                  for distance in (105., 80., 60., 40., 20., 15.)]
                within_margin = next(i for i in range(index + 1) if target_s - path_s[i] <= margin)
                sample_indices.append(within_margin)
                approach = []
                for at in sorted(set(sample_indices)):
                    remaining = target_s - path_s[at]
                    cap = control.get_target_velocity(sample(1), at, planned)
                    expected = min(planned, math.sqrt(2.0 * float(policies["comfortable_deceleration"])
                                                     * max(0., remaining - margin)))
                    _require(math.isclose(cap, expected, abs_tol=1e-9),
                             "실제 경로상의 적신호 감속 곡선이 기존 계산과 다릅니다.")
                    approach.append({"local_path_index": at, "remaining_route_m": remaining,
                                     "velocity_cap_mps": cap})
                _require(approach[0]["remaining_route_m"] > 104.0
                         and approach[0]["velocity_cap_mps"] > 0.0
                         and approach[-1]["remaining_route_m"] <= margin
                         and approach[-1]["velocity_cap_mps"] == 0.0
                         and any(0.0 < row["velocity_cap_mps"] < planned for row in approach),
                         "원거리 즉시 정지가 아닌 점진 감속·13m 이내 정지를 확인하지 못했습니다.")
                _require(all(a["velocity_cap_mps"] >= b["velocity_cap_mps"]
                             for a, b in zip(approach, approach[1:])),
                         "정지선 접근 중 속도 상한이 증가합니다.")
                result = {
                    "schema": "traffic_light_expert_offline_verification_v1", "passed": True,
                    "scenario": MODE, "global_start_index": first, "global_end_index": last,
                    "mission_points": last - first + 1, "expert_points": len(runtime_points),
                    "expert_tail_m": polyline_length(runtime_points[last - first:]),
                    "mission_length_m": polyline_length(points[first:last + 1]),
                    "route_signal_targets": signal_document,
                    "traffic_light_policy": policies,
                    "runtime_speed_zones": runtime_config["planning"]["velocity_profile"].get("speed_limit_zones", []),
                    "phase_matrix": phase_results, "unavailable_signal_cases": unavailable_results,
                    "red_approach": approach, "stop_margin_m": margin,
                    "planned_velocity_mps": planned,
                    "runtime_config_sha256": _digest(controller.runtime_config_path),
                    "runtime_path_sha256": _digest(controller.runtime_path_csv),
                    "source_sha256": before, "source_files_unchanged": True,
                    "temporary_runtime_removed": True,
                    "expert_process_started": False, "ros_connected": False,
                    "morai_connected": False, "model_forwards": 0,
                    "verification_scope": "원본 경로·실제 Expert 임시 설정·신호 속도 상한의 오프라인 검증",
                    "limitation": "실제 차량 제동·센서 수신·신호 전환 readback·폐루프 주행 안전은 아직 검증하지 않았다.",
                }
    after = {str(path): _digest(path) for path in source_files}
    _require(after == before, "검증 전후 원본 Expert/경로 파일 hash가 달라졌습니다.")
    return result


def main():
    from ..main_runner import load_configuration
    with redirect_stdout(io.StringIO()):
        global_cfg, _, cfg = load_configuration()
    cfg["mission_mode"] = MODE
    print(json.dumps(verify_traffic_light_expert(global_cfg, cfg), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
