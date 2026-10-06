import argparse
import copy
import os
import time
from pathlib import Path

import yaml

from .adapters.morai_grpc import MoraiGrpcClient
from .scenarios.full_loop_scenario import (
    FullLoopAfterStaticScenario,
    FullLoopHighwayTunnelTestScenario,
    FullLoopScenario,
)
from .utils.map_loader import MGeoMapLoader
from .scenarios.segment_modes.modes import MISSION_MODES, automatic_repeat_enabled
from .scenarios.segment_modes.scenario import MissionScenario


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
SUPPORTED_SCENARIOS = {
    "full_loop": FullLoopScenario,
    "full_loop_after_static": FullLoopAfterStaticScenario,
    "full_loop_highway_tunnel_test": FullLoopHighwayTunnelTestScenario,
    **{name: MissionScenario for name in MISSION_MODES},
}


def load_yaml(path):
    path = Path(path)
    print(f"[KCityRunner] loading yaml: {path}")
    with path.open("r", encoding="utf-8") as source:
        data = yaml.safe_load(source)
    if data is None:
        raise RuntimeError(f"YAML is empty: {path}")
    return data


def merge_dict(base, override):
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            merge_dict(base[key], value)
        else:
            base[key] = value


def package_path(value):
    path = Path(value)
    if not path.is_absolute():
        path = PACKAGE_ROOT / path
    return str(path.resolve())


def discover_external_path(
    configured,
    environment_name,
    candidates,
    marker,
    label,
):
    environment_value = os.environ.get(environment_name)
    if environment_value:
        candidates = [Path(environment_value)]
    elif configured and str(configured).lower() != "auto":
        configured_path = Path(configured)
        if not configured_path.is_absolute():
            configured_path = PACKAGE_ROOT / configured_path
        candidates = [configured_path]

    checked = []
    for candidate in candidates:
        candidate = candidate.resolve()
        checked.append(str(candidate))
        if (candidate / marker).is_file():
            print(f"[KCityRunner] {label}: {candidate}")
            return str(candidate)

    raise RuntimeError(
        f"Unable to locate {label}. Checked: {checked}. "
        f"Set {environment_name} or config/local.yaml."
    )


def resolve_runtime_paths(global_cfg, scenario_cfg):
    paths = global_cfg.setdefault("paths", {})
    paths["mgeo_root"] = package_path(
        paths.get("mgeo_root", "data/R_KR_PR_K-city_2025")
    )
    paths["runtime_root"] = package_path(
        paths.get("runtime_root", ".runtime")
    )

    paths["grpc_src"] = discover_external_path(
        configured=paths.get("grpc_src", "auto"),
        environment_name="MORAI_GRPC_SRC",
        candidates=[
            PROJECT_ROOT
            / "data_collection"
            / "third_party"
            / "grpc_inha_univ"
            / "src",
            PROJECT_ROOT / "third_party" / "grpc_inha_univ" / "src",
        ],
        marker=Path("api") / "morai_sim_client.py",
        label="MORAI gRPC SDK",
    )
    paths["expert_root"] = discover_external_path(
        configured=paths.get("expert_root", "auto"),
        environment_name="MORAI_EXPERT_ROOT",
        candidates=[
            PROJECT_ROOT / "expert",
            PROJECT_ROOT / "third_party" / "GT_BEV",
        ],
        marker=Path("src") / "expert" / "src" / "config" / "config.json",
        label="external AIMDrive expert",
    )

    for key in (
        "global_path",
        "route_definition_path",
        "profile_state_path",
        "route_debug_dir",
    ):
        if key in scenario_cfg:
            scenario_cfg[key] = package_path(scenario_cfg[key])

    tunnel_cfg = scenario_cfg.get("tunnel_sudden_brake", {})
    if "verification_report_path" in tunnel_cfg:
        tunnel_cfg["verification_report_path"] = package_path(
            tunnel_cfg["verification_report_path"]
        )


def load_configuration():
    global_cfg = load_yaml(PACKAGE_ROOT / "config" / "global.yaml")
    local_path = PACKAGE_ROOT / "config" / "local.yaml"
    if local_path.exists():
        merge_dict(global_cfg, load_yaml(local_path))

    zone_cfg = load_yaml(PACKAGE_ROOT / "config" / "full_loop.yaml")
    base_scenario_cfg = copy.deepcopy(zone_cfg["scenarios"]["full_loop"])
    resolve_runtime_paths(global_cfg, base_scenario_cfg)
    return global_cfg, zone_cfg, base_scenario_cfg


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "MORAI 26.R1 / R_KR_PR_K-city_2025 full-loop scenarios"
        )
    )
    parser.add_argument(
        "--zone",
        choices=("full_loop",),
        default="full_loop",
    )
    parser.add_argument(
        "--scenario",
        choices=tuple(SUPPORTED_SCENARIOS),
        default="full_loop",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run one scenario only even when automatic repeat is enabled.",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help=(
            "Stop after exactly this many completed scenario runs. "
            "Only applies while automatic repeat is enabled."
        ),
    )
    parser.add_argument(
        "--highway-object-audit",
        action="store_true",
        help=(
            "With full_loop_highway_tunnel_test, drive checkpoint 10 to "
            "checkpoint 13 before the tunnel and compare /Object_topic "
            "against live gRPC vehicle actors."
        ),
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate paths, imports, route, and configuration without MORAI.",
    )
    parser.add_argument(
        "--no-sync-runtime",
        action="store_true",
        help=(
            "Run an asynchronous scenario-only test without MORAI Sync "
            "Runtime or sensor capture. Ego progress is read through gRPC."
        ),
    )
    parser.add_argument(
        "--manual-ego",
        action="store_true",
        help=(
            "Drive Ego with MORAI keyboard controls in the tunnel dynamic trial. "
            "Requires --scenario full_loop_tunnel_dynamic --once."
        ),
    )
    parser.add_argument(
        "--tunnel-candidate-id",
        metavar="ID",
        help=(
            "Trial one route-relative tunnel pedestrian path by ID. "
            "Requires full_loop_tunnel_dynamic --once --manual-ego."
        ),
    )
    parser.add_argument(
        "--dynamic-review",
        action="append",
        metavar="CROSSING:MODEL:SPEED_MPS",
        help="Trial one catalogue crossing (repeat up to three times); never approves it.",
    )
    parser.add_argument(
        "--static-obstacle-count",
        type=int,
        choices=(1, 2),
        default=1,
        help=(
            "Spawn one or two verified static obstacles. Two is intended "
            "for scenario-only full_loop_static testing."
        ),
    )
    parser.add_argument(
        "--static-preset",
        default=None,
        help=(
            "Replay a static-obstacle preset saved by the live editor. "
            "Use a preset name from config/static_scenario_presets or a YAML path."
        ),
    )
    parser.add_argument(
        "--static-editor",
        action="store_true",
        help=(
            "Allow live MORAI obstacle repositioning for scenario design. "
            "This is a no-capture, asynchronous full_loop_static mode."
        ),
    )
    parser.add_argument(
        "--ego-trace",
        default=None,
        metavar="CSV",
        help=(
            "Write the Ego pose of every control tick to this CSV. "
            "Used by scenario-only static preset verification."
        ),
    )
    return parser


def resolve_static_preset(value):
    path = Path(value)
    if not path.suffix and path.name == str(path):
        path = PACKAGE_ROOT / "config" / "static_scenario_presets" / (str(path) + ".yaml")
    elif not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve()
    if not path.is_file():
        raise ValueError("Static obstacle preset not found: {}".format(path))
    return str(path)


def run_offline_check(global_cfg, scenario_cfg, scenario_name):
    from .tools.verify_separation import verify_package

    verify_package(
        package_root=PACKAGE_ROOT,
        global_cfg=global_cfg,
        scenario_cfg=scenario_cfg,
        scenario_name=scenario_name,
    )


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.max_runs is not None and args.max_runs < 1:
        parser.error("--max-runs must be at least 1")
    if args.manual_ego and (
        args.scenario != "full_loop_tunnel_dynamic" or not args.once
    ):
        parser.error("--manual-ego requires --scenario full_loop_tunnel_dynamic --once")
    if args.tunnel_candidate_id and not (
        args.scenario == "full_loop_tunnel_dynamic" and args.once and args.manual_ego
    ):
        parser.error(
            "--tunnel-candidate-id requires full_loop_tunnel_dynamic --once --manual-ego"
        )

    global_cfg, zone_cfg, base_scenario_cfg = load_configuration()
    if args.static_editor:
        if args.scenario != "full_loop_static":
            parser.error("--static-editor requires --scenario full_loop_static")
        if args.static_preset:
            parser.error("--static-editor cannot be combined with --static-preset")
        base_scenario_cfg["static_obstacle_editor_mode"] = True
        base_scenario_cfg.setdefault("sync_runtime", {})["enabled"] = False
        # Keep Ego fixed at the scenario start while retaining the Expert's
        # complete perception and planning graph. Only its final vehicle
        # command is redirected; /local_path and /lattice/paths remain live.
        base_scenario_cfg["expert_ros_remaps"] = [
            "/ctrl_cmd:=/static_scenario_editor/ctrl_cmd",
            "_static_editor_mode:=true",
        ]
    if args.no_sync_runtime:
        base_scenario_cfg.setdefault("sync_runtime", {})["enabled"] = False
    if args.manual_ego:
        base_scenario_cfg.setdefault("sync_runtime", {})["enabled"] = False
        base_scenario_cfg["drive_control_mode"] = "morai_keyboard"
    if args.tunnel_candidate_id:
        base_scenario_cfg.setdefault("tunnel_dynamic", {})[
            "trial_candidate_id"] = args.tunnel_candidate_id
    if args.dynamic_review:
        if args.scenario != "full_loop_dynamic" or not args.once or not args.no_sync_runtime:
            parser.error("--dynamic-review requires full_loop_dynamic --once --no-sync-runtime")
        if len(args.dynamic_review) > 3:
            parser.error("--dynamic-review supports at most three sequential events")
        review = []
        for selection in args.dynamic_review:
            try:
                crossing, model, speed = selection.rsplit(":", 2)
                review.append({"id": crossing, "model": model, "speed_mps": float(speed)})
            except ValueError:
                parser.error("--dynamic-review must be CROSSING:MODEL:SPEED_MPS")
        base_scenario_cfg["dynamic_obstacle_review"] = review
        base_scenario_cfg["route_debug_enabled"] = False
    if args.static_obstacle_count != 1:
        if args.scenario != "full_loop_static":
            parser.error(
                "--static-obstacle-count 2 requires "
                "--scenario full_loop_static"
            )
        base_scenario_cfg["static_obstacle_count"] = args.static_obstacle_count
    if args.static_preset:
        if args.scenario != "full_loop_static":
            parser.error("--static-preset requires --scenario full_loop_static")
        if args.static_obstacle_count != 1:
            parser.error("--static-preset cannot be combined with --static-obstacle-count")
        try:
            base_scenario_cfg["static_obstacle_preset"] = resolve_static_preset(
                args.static_preset
            )
        except ValueError as error:
            parser.error(str(error))
    if args.ego_trace:
        base_scenario_cfg["ego_trace_path"] = str(
            Path(args.ego_trace).expanduser().resolve()
        )
    if args.scenario in MISSION_MODES:
        base_scenario_cfg["mission_mode"] = args.scenario
    if (
        args.highway_object_audit
        and args.scenario != "full_loop_highway_tunnel_test"
    ):
        parser.error(
            "--highway-object-audit requires "
            "--scenario full_loop_highway_tunnel_test"
        )
    if args.highway_object_audit:
        base_scenario_cfg.setdefault("highway_object_audit", {})[
            "enabled"
        ] = True

    if args.check:
        run_offline_check(global_cfg, base_scenario_cfg, args.scenario)
        return

    print("[KCityRunner] connecting MORAI gRPC")
    grpc_client = MoraiGrpcClient(global_cfg)
    grpc_client.connect()
    map_loader = MGeoMapLoader(global_cfg["paths"]["mgeo_root"])

    scenario_class = SUPPORTED_SCENARIOS[args.scenario]
    automatic_repeat = automatic_repeat_enabled(
        args.scenario, base_scenario_cfg, args.once, args.highway_object_audit,
    )
    repeat_delay_sec = float(
        base_scenario_cfg.get("repeat_delay_sec", 1.0)
    )
    run_number = 1

    try:
        while True:
            scenario = scenario_class(
                grpc_client=grpc_client,
                map_loader=map_loader,
                global_cfg=global_cfg,
                scenario_cfg=copy.deepcopy(base_scenario_cfg),
            )
            scenario.zone_name = zone_cfg.get("zone", "full_loop")
            print(
                f"[KCityRunner] running {args.scenario} "
                f"sequence={run_number}"
            )
            scenario.run()
            if args.max_runs is not None and run_number >= args.max_runs:
                print(
                    f"[KCityRunner] completed requested runs: "
                    f"{run_number}/{args.max_runs}"
                )
                break
            if not automatic_repeat:
                break

            run_number += 1
            print(
                f"[KCityRunner] next scenario sequence={run_number} "
                f"in {repeat_delay_sec:.1f}s"
            )
            time.sleep(max(0.0, repeat_delay_sec))
    except KeyboardInterrupt:
        print("[KCityRunner] Ctrl+C received; stopping")
    finally:
        grpc_client.stop()


if __name__ == "__main__":
    main()
