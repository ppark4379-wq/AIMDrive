import argparse
import copy
import os
import random
import json
from types import SimpleNamespace
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
            PROJECT_ROOT.parent / "AIM2026" / "Expert",
            PROJECT_ROOT / "third_party" / "GT_BEV",
        ],
        marker=Path("src") / "config" / "config.json",
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
    parser.add_argument('--ve-objects', action='store_true',
                        help='기존 Expert 장애물 입력만 VE로 교체 (회전교차로/고주로, 비동기 실행)')
    parser.add_argument('--ve-signal-head', help='P1 3종 신호 head 경로; 신호 시험 구간에서만 사용')
    parser.add_argument('--ve-checkpoint', choices=('E3', 'E4', 'P1'), default='E3')
    parser.add_argument('--stop-before-tunnel', action='store_true',
                        help='VE 풀루트 시험의 종점을 고주로 종료 체크포인트로 제한')
    from .adapters.expert_controller import validate_max_velocity_kmh
    parser.add_argument('--expert-speed-cap-kmh', type=validate_max_velocity_kmh, default=None,
                        help='이번 실행의 Expert 목표속도 상한; 구간별 상한에도 적용 (km/h)')
    parser.add_argument('--ve-compare-gt', action='store_true',
                        help='별도 GT 기록기로 VE bbox·속도 비교; Expert는 VE 장애물만 사용')
    parser.add_argument(
        "--manual-ego",
        action="store_true",
        help=(
            "Drive Ego with MORAI keyboard controls in the tunnel dynamic trial. "
            "Requires a tunnel scenario and --once."
        ),
    )
    tunnel_selection = parser.add_mutually_exclusive_group()
    tunnel_selection.add_argument(
        "--tunnel-candidate-id",
        metavar="ID",
        help=(
            "Trial one route-relative tunnel pedestrian path by ID. "
            "Requires full_loop_tunnel_dynamic --once; uses Expert unless --manual-ego."
        ),
    )
    from .scenarios.full_loop_parts.tunnel_vehicle_catalog import BEHAVIORS
    tunnel_selection.add_argument(
        "--tunnel-vehicle-behavior", choices=BEHAVIORS,
        help="One continuous vehicle behavior; requires full_loop_tunnel_vehicle --once.",
    )
    tunnel_selection.add_argument(
        "--tunnel-vehicle-random", action="store_true",
        help="Select one supported whole-tunnel vehicle behavior using --seed.",
    )
    tunnel_selection.add_argument(
        "--tunnel-random-mix",
        action="store_true",
        help="Trial one seeded behavior/model mix with Expert or --manual-ego; requires --once.",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Set the scenario random seed to reproduce the selected scene.",
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
        "--static-verified-pair", action="store_true",
        help="Replay an exact two-obstacle layout previously passed by the GT Expert.",
    )
    parser.add_argument(
        "--no-highway-merge-static-vehicle", action="store_true",
        help="Disable only the scenario-owned stopped vehicle at the highway merge.",
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
    parser.add_argument(
        "--tunnel-static-library", type=Path, default=None,
        help="Optional approved layout library for full_loop_tunnel_static; defaults to the current curated list.",
    )
    parser.add_argument(
        "--tunnel-static-seed", type=int, default=None,
        help="Optional reproducible random seed for full_loop_tunnel_static.",
    )
    parser.add_argument(
        "--tunnel-static-output", type=Path, default=None,
        help="Optional private result directory for full_loop_tunnel_static.",
    )
    parser.add_argument(
        "--tunnel-static-revalidate", action="store_true",
        help="Verify historical tunnel layouts at the current inherited Expert settings; approve only successful fresh drives.",
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
    if scenario_cfg.get('ve_stop_before_tunnel'):
        context = FullLoopScenario(
            grpc_client=SimpleNamespace(make_transform=lambda *args: args),
            map_loader=MGeoMapLoader(global_cfg['paths']['mgeo_root']),
            global_cfg=global_cfg, scenario_cfg=copy.deepcopy(scenario_cfg))
        context.cfg['route_debug_enabled'] = False
        context.route_definition = context.load_route_definition()
        context.prepare_full_loop_route()
        if not context.route_length_m < context.tunnel_entry_s:
            raise RuntimeError('Before-tunnel route does not end before tunnel entry')
        from .adapters.ve_signal import build_signal_targets
        targets = build_signal_targets(context, {'map': {
            'data_dir': global_cfg['paths']['mgeo_root'],
            'route_signal_route_links': context.route_definition['route_links'],
            'is_closed_path': context.cfg['expert_is_closed_path']}})
        print('[BeforeTunnel] VE 신호 경로 연결 PASS: ' + repr(
            [(t['signal_id'], t['signal_movement']) for t in targets['targets']]))
        print('[BeforeTunnel] 실제 지도 경로·종점 검증 PASS')
    if scenario_name == "full_loop_traffic_light":
        from .tools.verify_traffic_light_expert import verify_traffic_light_expert
        print("[TrafficLight] 실제 Expert 경로·신호 정책 오프라인 검증:")
        print(json.dumps(verify_traffic_light_expert(global_cfg, scenario_cfg),
                         indent=2, ensure_ascii=False, allow_nan=False))
    if scenario_name == "full_loop_tunnel_vehicle":
        from .tools.generate_tunnel_dynamic_candidates import load_catalog_inputs
        from .scenarios.full_loop_parts.tunnel_vehicle_dynamic import prepare_tunnel_vehicle
        metadata, points, zone = load_catalog_inputs(PACKAGE_ROOT)
        configured = yaml.safe_load(Path(scenario_cfg["route_definition_path"]).read_text(encoding="utf-8-sig"))
        definition = configured[scenario_cfg["route_definition_key"]]
        zone = definition["zones"]["tunnel"]
        checkpoints = {str(c["id"]): int(c["path_index"]) for c in definition["checkpoints"]}
        from .utils.transform_utils import polyline_length
        points = [tuple(map(float, line.split()[:3])) for line in Path(scenario_cfg["global_path"]).read_text().splitlines()
                  if line.strip() and not line.lstrip().startswith("#")]
        entry = polyline_length(points[:checkpoints[str(zone["entry_checkpoint"])]+1])
        exit_s = polyline_length(points[:checkpoints[str(zone["exit_checkpoint"])]+1])
        seed = scenario_cfg.get("random_seed")
        seed = int(seed) if seed is not None else 0
        cfg = copy.deepcopy(scenario_cfg)
        cfg["tunnel_vehicle_dynamic"]["enabled"] = True
        context = SimpleNamespace(cfg=cfg, scenario_name=scenario_name, clean_road_mode=bool(cfg.get("clean_road_mode",False)),
            map_loader=MGeoMapLoader(global_cfg["paths"]["mgeo_root"]), route_points=points,
            route_definition=definition, tunnel_entry_s=entry, tunnel_exit_s=exit_s,
            run_seed=seed, random=random.Random(seed))
        prepare_tunnel_vehicle(context)
        print("[TunnelVehicle] offline validated (model availability/motion require MORAI):")
        print(json.dumps({"run_seed":seed,"plan":context.tunnel_vehicle_plan,
                          "unavailable_behaviors":context.tunnel_vehicle_catalog["unavailable_behaviors"]},indent=2))


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.max_runs is not None and args.max_runs < 1:
        parser.error("--max-runs must be at least 1")
    if args.scenario != "full_loop_tunnel_static" and (args.tunnel_static_revalidate or any(value is not None for value in (
            args.tunnel_static_library, args.tunnel_static_seed, args.tunnel_static_output))):
        parser.error("--tunnel-static-* options require --scenario full_loop_tunnel_static")
    if args.manual_ego and (
        args.scenario not in ("full_loop_tunnel_dynamic", "full_loop_tunnel_vehicle") or not args.once
    ):
        parser.error("--manual-ego requires a tunnel scenario and --once")
    if args.tunnel_candidate_id and not (
        args.scenario == "full_loop_tunnel_dynamic" and args.once
    ):
        parser.error(
            "--tunnel-candidate-id requires full_loop_tunnel_dynamic --once"
        )
    if args.tunnel_random_mix and not (
        args.scenario == "full_loop_tunnel_dynamic" and args.once
    ):
        parser.error("--tunnel-random-mix requires full_loop_tunnel_dynamic --once")
    if (args.tunnel_vehicle_behavior or args.tunnel_vehicle_random) and args.scenario != "full_loop_tunnel_vehicle":
        parser.error("Vehicle behavior options require full_loop_tunnel_vehicle --once")
    if args.scenario == "full_loop_tunnel_vehicle" and not args.once:
        parser.error("full_loop_tunnel_vehicle requires --once")

    global_cfg, zone_cfg, base_scenario_cfg = load_configuration()
    if args.stop_before_tunnel:
        if args.scenario != 'full_loop' or not args.ve_objects or not args.no_sync_runtime or not args.once:
            parser.error('--stop-before-tunnel requires full_loop --ve-objects --no-sync-runtime --once')
        base_scenario_cfg['ve_stop_before_tunnel'] = True
    if args.seed is not None:
        base_scenario_cfg["random_seed"] = args.seed
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
    if args.ve_objects:
        from .adapters.ve_objects import configure
        try:
            configure(base_scenario_cfg, args.scenario, args.ve_checkpoint)
        except ValueError as error:
            parser.error(str(error))
    if args.ve_signal_head:
        if not args.ve_objects or args.scenario not in ('full_loop_traffic_light', 'full_loop') or args.ve_checkpoint != 'P1':
            parser.error('--ve-signal-head requires traffic_light/full_loop, --ve-objects and P1')
        head_path = Path(args.ve_signal_head).expanduser().resolve()
        if not head_path.is_file():
            parser.error('신호 head 없음: ' + str(head_path))
        base_scenario_cfg['ve_signal_head'] = str(head_path)
        if args.scenario == 'full_loop':
            base_scenario_cfg['ve_full_route_signals'] = True
            base_scenario_cfg['clean_road_mode'] = False
            base_scenario_cfg.setdefault('traffic_light_wait_override', {})['enabled'] = False
        base_scenario_cfg['expert_ros_remaps'].append('/GetTrafficLightStatus:=/ve/GetTrafficLightStatus')
    if args.ve_objects and args.scenario == 'full_loop_traffic_light' and not args.ve_signal_head:
        parser.error('VE 신호 시험에는 --ve-signal-head가 필요합니다')
    if args.expert_speed_cap_kmh is not None:
        # GT-driven Expert at a raised cap is allowed on the highway only (VE can then be observed passively
        # with record_ve_manual.sh); other scenarios keep the VE-run requirement.
        gt_highway = args.scenario == "full_loop_highway" and not args.ve_objects
        if not gt_highway and (not args.ve_objects or args.scenario not in ("full_loop_highway", "full_loop_roundabout", "full_loop")):
            parser.error("--expert-speed-cap-kmh requires a VE highway/roundabout run (or a GT full_loop_highway run)")
        base_scenario_cfg["expert_speed_cap_kmh"] = args.expert_speed_cap_kmh
    if args.ve_compare_gt:
        if not args.ve_objects:
            parser.error('--ve-compare-gt requires --ve-objects')
        base_scenario_cfg['ve_compare_gt'] = True
    if args.manual_ego:
        base_scenario_cfg.setdefault("sync_runtime", {})["enabled"] = False
        base_scenario_cfg["drive_control_mode"] = "morai_keyboard"
    if args.tunnel_candidate_id:
        base_scenario_cfg.setdefault("tunnel_dynamic", {})[
            "trial_candidate_id"] = args.tunnel_candidate_id
    if args.tunnel_vehicle_behavior:
        base_scenario_cfg.setdefault("tunnel_vehicle_dynamic", {}).update(
            behavior=args.tunnel_vehicle_behavior, random_behavior=False)
    if args.tunnel_vehicle_random:
        base_scenario_cfg.setdefault("tunnel_vehicle_dynamic", {})["random_behavior"] = True
    if args.tunnel_random_mix:
        base_scenario_cfg.setdefault("tunnel_dynamic", {}).setdefault(
            "random_mix", {})["enabled"] = True
    if args.tunnel_candidate_id or args.tunnel_random_mix:
        # These explicit one-run trial options permit candidate paths with
        # either controller. Their catalogue status remains unchanged.
        base_scenario_cfg.setdefault("tunnel_dynamic", {})["candidate_policy"] = "trial"
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
    if args.no_highway_merge_static_vehicle:
        base_scenario_cfg.setdefault("highway_merge_static_vehicle", {})["enabled"] = False
        base_scenario_cfg["highway_merge_static_vehicle_explicitly_disabled"] = True
    if args.static_verified_pair:
        if args.scenario not in ("full_loop", "full_loop_static"):
            parser.error("--static-verified-pair requires full_loop or full_loop_static")
        if args.static_preset or args.static_editor or args.static_obstacle_count != 1:
            parser.error("--static-verified-pair cannot be combined with preset/editor/count")
        from .scenarios.full_loop_parts.static_obstacle import verified_pair_presets
        pool = verified_pair_presets()
        if not pool:
            parser.error("No verified two-obstacle layouts available")
        base_scenario_cfg["static_verified_pair"] = True
        print("[정적 배치] GT Expert 통과 2개 배치 {}종 · 원래 위치/각도/크기 재사용".format(len(pool)))
    if args.static_preset:
        if args.scenario not in ("full_loop", "full_loop_static"):
            parser.error("--static-preset requires full_loop or full_loop_static")
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
    if args.scenario == "full_loop_traffic_light":
        from .scenarios.segment_modes.traffic_light import configure
        configure(base_scenario_cfg)
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

    if args.scenario == "full_loop_tunnel_static":
        if args.ego_trace:
            parser.error("full_loop_tunnel_static records its own guarded Ego trace; --ego-trace is unavailable")
        from .scenarios.segment_modes.tunnel_static_replay import run_tunnel_static_scenario
        try:
            return run_tunnel_static_scenario(args, global_cfg, base_scenario_cfg)
        except (OSError, ValueError, RuntimeError) as error:
            parser.exit(2, str(error) + "\n")

    if args.check:
        if args.ve_objects:
            from inference.realtime_ve_objects import CHECKPOINTS
            selected = CHECKPOINTS[args.ve_checkpoint]
            if not Path(selected['path']).is_file():
                parser.error('VE 체크포인트 없음: '+selected['path'])
            print('[VE 설정] 체크포인트={} · Expert 장애물=/ve/Object_topic · GT 장애물 입력 없음 · 자차=기존 ROS · 신호={}'.format(args.ve_checkpoint, 'VE' if args.ve_signal_head else '기존 ROS'))
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
    raise SystemExit(main())
