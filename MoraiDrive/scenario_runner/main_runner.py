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


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
SUPPORTED_SCENARIOS = {
    "full_loop": FullLoopScenario,
    "full_loop_after_static": FullLoopAfterStaticScenario,
    "full_loop_highway_tunnel_test": FullLoopHighwayTunnelTestScenario,
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
        "--check",
        action="store_true",
        help="Validate paths, imports, route, and configuration without MORAI.",
    )
    return parser


def run_offline_check(global_cfg, scenario_cfg, scenario_name):
    from .tools.verify_separation import verify_package

    verify_package(
        package_root=PACKAGE_ROOT,
        global_cfg=global_cfg,
        scenario_cfg=scenario_cfg,
        scenario_name=scenario_name,
    )


def main():
    args = build_parser().parse_args()
    global_cfg, zone_cfg, base_scenario_cfg = load_configuration()

    if args.check:
        run_offline_check(global_cfg, base_scenario_cfg, args.scenario)
        return

    print("[KCityRunner] connecting MORAI gRPC")
    grpc_client = MoraiGrpcClient(global_cfg)
    grpc_client.connect()
    map_loader = MGeoMapLoader(global_cfg["paths"]["mgeo_root"])

    scenario_class = SUPPORTED_SCENARIOS[args.scenario]
    automatic_repeat = (
        args.scenario in {"full_loop", "full_loop_after_static"}
        and bool(base_scenario_cfg.get("auto_repeat", False))
        and not args.once
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
