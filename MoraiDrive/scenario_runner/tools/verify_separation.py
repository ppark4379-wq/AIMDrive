import json
from pathlib import Path

import yaml

from .feature_contract import verify_feature_contract
from ..scenarios.full_loop_scenario import (
    FullLoopAfterStaticScenario,
    FullLoopHighwayTunnelTestScenario,
    FullLoopScenario,
)
from ..utils.link_graph import LinkGraph
from ..utils.map_loader import MGeoMapLoader
from ..utils.transform_utils import polyline_length
from ..scenarios.segment_modes.modes import MISSION_MODES
from ..scenarios.segment_modes.scenario import MissionScenario
from ..scenarios.full_loop_competition import FullLoopCompetitionScenario
from .verify_segment_modes import verify_mission_mode


EXPECTED_MAP = "R_KR_PR_K-city_2025"
EXPECTED_SCENARIOS = {
    "full_loop": FullLoopScenario,
    "full_loop_competition": FullLoopCompetitionScenario,
    "full_loop_after_static": FullLoopAfterStaticScenario,
    "full_loop_highway_tunnel_test": FullLoopHighwayTunnelTestScenario,
    **{name: MissionScenario for name in MISSION_MODES},
}
LEGACY_CONFIG_NAMES = {
    "urban.yaml",
    "urban_links.yaml",
    "highway.yaml",
    "highway_links.yaml",
    "roundabout.yaml",
    "roundabout_links.yaml",
    "roundabout2.yaml",
    "roundabout2_links.yaml",
    "suburban.yaml",
    "suburban_links.yaml",
}
FORBIDDEN_SOURCE_FRAGMENTS = {
    "from zones.urban_scenarios",
    "from scenario_runner.zones.urban_scenarios",
    "R_KR_PG_KATRI",
}
EXPERT_WRITE_FRAGMENTS = {
    'open(self.base_config_path, "w"',
    "os.makedirs(self.expert_root",
    "shutil.copyfile",
    "shutil.copytree",
}


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def load_path_points(path):
    points = []
    with Path(path).open("r", encoding="utf-8-sig") as source:
        for line in source:
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            fields = text.split()
            points.append(
                (
                    float(fields[0]),
                    float(fields[1]),
                    float(fields[2]) if len(fields) >= 3 else 0.0,
                )
            )
    return points


def verify_source_boundary(package_root):
    python_files = sorted(
        path
        for path in package_root.rglob("*.py")
        if path.name != "verify_separation.py"
        and ".runtime" not in path.relative_to(package_root).parts
    )
    require(python_files, "No Python files found in separated package")

    combined = "\n".join(
        path.read_text(encoding="utf-8")
        for path in python_files
    )
    for fragment in FORBIDDEN_SOURCE_FRAGMENTS:
        require(
            fragment not in combined,
            f"Legacy dependency found in separated source: {fragment}",
        )

    adapter_source = (
        package_root / "adapters" / "expert_controller.py"
    ).read_text(encoding="utf-8")
    for fragment in EXPERT_WRITE_FRAGMENTS:
        require(
            fragment not in adapter_source,
            f"External expert write operation found: {fragment}",
        )

    entrypoint_source = (
        package_root / "adapters" / "expert_process.py"
    ).read_text(encoding="utf-8")
    require(
        "from AIM2026.Expert.src.autonomous_driving import AutonomousDriving"
        in entrypoint_source,
        "AIMDrive expert is not imported by the scenario-owned entrypoint",
    )
    require(
        "from MoraiDrive.scenario_runner.legacy_expert.ros.ros_manager import RosManager"
        in entrypoint_source,
        "AIMDrive RosManager is not imported by the scenario-owned entrypoint",
    )

    config_names = {
        path.name for path in (package_root / "config").glob("*.yaml")
    }
    legacy_present = sorted(config_names & LEGACY_CONFIG_NAMES)
    require(
        not legacy_present,
        f"Legacy scenario configs were included: {legacy_present}",
    )


def verify_configuration(package_root, global_cfg, scenario_cfg):
    require(
        global_cfg["morai"]["map_name"] == EXPECTED_MAP,
        "Separated global config targets the wrong MORAI map",
    )
    manual_tunnel_trial = (
        scenario_cfg.get("drive_control_mode") == "morai_keyboard"
        and scenario_cfg.get("mission_mode") in ("full_loop_tunnel_dynamic", "full_loop_tunnel_vehicle")
        and not scenario_cfg.get("sync_runtime", {}).get("enabled", True)
    )
    require(
        scenario_cfg["drive_control_mode"] == "aimdrive_expert" or manual_tunnel_trial,
        "Separated runner must use the external AIMDrive expert, except for a manual tunnel trial",
    )

    internal_paths = [
        Path(global_cfg["paths"]["mgeo_root"]),
        Path(global_cfg["paths"]["runtime_root"]),
        Path(scenario_cfg["global_path"]),
        Path(scenario_cfg["route_definition_path"]),
        Path(scenario_cfg["profile_state_path"]),
        Path(scenario_cfg["route_debug_dir"]),
        Path(
            scenario_cfg["tunnel_sudden_brake"][
                "verification_report_path"
            ]
        ),
    ]
    package_root = package_root.resolve()
    for path in internal_paths:
        resolved = path.resolve()
        require(
            resolved == package_root or package_root in resolved.parents,
            f"Scenario-owned path escapes package boundary: {resolved}",
        )

    expert_root = Path(global_cfg["paths"]["expert_root"]).resolve()
    require(
        expert_root != package_root and package_root not in expert_root.parents,
        "External expert must not be copied into scenario_runner",
    )
    require(
        (
            expert_root
            / "src"
            / "config"
            / "config.json"
        ).is_file(),
        f"External expert config not found: {expert_root}",
    )


def verify_route(package_root, global_cfg, scenario_cfg):
    map_loader = MGeoMapLoader(global_cfg["paths"]["mgeo_root"])
    route_path = Path(scenario_cfg["route_definition_path"])
    route_data = yaml.safe_load(
        route_path.read_text(encoding="utf-8-sig")
    )["full_loop"]

    require(
        route_data["map_name"] == EXPECTED_MAP,
        "Route definition targets the wrong map",
    )
    route_links = list(route_data["route_links"])
    require(
        len(route_links) == 39,
        f"Expected 39 official route links, got {len(route_links)}",
    )
    missing = [
        link_id
        for link_id in route_links
        if link_id not in map_loader.link_set
    ]
    require(not missing, f"Route links missing from MGeo: {missing}")

    graph = LinkGraph(map_loader)
    disconnected = [
        (current, following)
        for current, following in zip(route_links[:-1], route_links[1:])
        if following not in graph.adj.get(current, [])
    ]
    require(
        not disconnected,
        f"Official route contains disconnected links: {disconnected}",
    )

    points = load_path_points(scenario_cfg["global_path"])
    require(
        len(points) == 4430,
        f"Expected 4430 official path points, got {len(points)}",
    )
    closure = (
        (points[-1][0] - points[0][0]) ** 2
        + (points[-1][1] - points[0][1]) ** 2
        + (points[-1][2] - points[0][2]) ** 2
    ) ** 0.5
    require(closure <= 0.5, f"Full-loop closure error is {closure:.6f}m")

    checkpoints = route_data["checkpoints"]
    require(
        all(0 <= int(item["path_index"]) < len(points) for item in checkpoints),
        "A checkpoint path_index is outside the official path",
    )

    return {
        "route_links": len(route_links),
        "route_points": len(points),
        "route_length_m": polyline_length(points),
        "closure_m": closure,
        "mgeo_links": len(map_loader.link_set),
        "mgeo_nodes": len(map_loader.node_set),
    }


def verify_package(
    package_root,
    global_cfg,
    scenario_cfg,
    scenario_name,
):
    package_root = Path(package_root)
    require(
        scenario_name in EXPECTED_SCENARIOS,
        f"Unsupported separated scenario: {scenario_name}",
    )
    verify_source_boundary(package_root)
    verify_configuration(package_root, global_cfg, scenario_cfg)
    verify_feature_contract(package_root, scenario_cfg)
    route = verify_route(package_root, global_cfg, scenario_cfg)

    report = {
        "status": "PASS",
        "map": EXPECTED_MAP,
        "supported_scenarios": sorted(EXPECTED_SCENARIOS),
        "selected_scenario": scenario_name,
        "expert_policy": "external import only; no expert writes",
        "route": route,
    }
    if scenario_name in MISSION_MODES:
        report["mission"] = verify_mission_mode(scenario_cfg, scenario_name)
    print("[KCityRunner] separation verification PASS")
    print(json.dumps(report, indent=2))
    return report
