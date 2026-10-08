"""Prepare tunnel candidates and inspect parked actors without changing Expert."""

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import time

import yaml

from ...main_runner import load_configuration, package_path
from ...utils.map_loader import MGeoMapLoader
from ...scenarios.segment_modes.modes import TUNNEL_NORMAL_START
from ...scenarios.full_loop_parts.tunnel_static import (
    generate_candidate,
    load_static_models,
    load_vehicle_models,
    screen_candidate,
    tunnel_geometry,
    validate_candidate,
)
from ...scenarios.full_loop_parts.tunnel_static_parts.screening import (
    screening_provenance, validate_screening_settings,
)


PACKAGE_ROOT = Path(__file__).resolve().parents[2]


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PACKAGE_ROOT / "config/tunnel_static.yaml")
    parser.add_argument("--seed", type=int, help="Repeat the same catalogue/layout with this seed")
    parser.add_argument("--count", type=int, choices=(1, 2, 3), help="Randomly choose 1 or 2 when omitted; three requires explicit opt-in")
    parser.add_argument("--experimental-three", action="store_true", help="Explicitly permit a three-actor candidate; every placement still requires measured screening and driving validation")
    parser.add_argument("--actor-kind", choices=("mixed", "obstacle", "vehicle"), default="mixed")
    parser.add_argument("--obstacle-model", action="append", help="Restrict to reviewed static models")
    parser.add_argument("--vehicle-model", action="append", help="Restrict to reviewed real NPC models")
    parser.add_argument("--candidates", type=int, default=1)
    parser.add_argument("--preview", action="store_true", help="Spawn one candidate with Ego in P, then clean up")
    parser.add_argument("--hold-seconds", type=float)
    parser.add_argument("--time-budget", type=float)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model-acceptance", type=Path,
                        help="Use only models whose parked live coverage still matches current sources")
    parser.add_argument("--runtime-catalog", type=Path,
                        help="Hash-bound native model measurements from the remaining-model audit")
    parser.add_argument("--pattern", help="Explicit expanded arrangement; the Expert still chooses the path")
    return parser


def _write(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


from ...scenarios.full_loop_parts.tunnel_static_parts.connection import _controller_publishers


def generation_options(configuration, pattern=None):
    """Pass one explicit placement policy to every candidate generator."""
    options = copy.deepcopy(configuration["layout"])
    selected = pattern or configuration.get("selected_pattern")
    if selected is not None:
        from ...scenarios.full_loop_parts.tunnel_static_parts.advanced_patterns import PATTERN_NAMES
        if selected not in PATTERN_NAMES:
            raise ValueError("Unknown expanded tunnel pattern: " + str(selected))
        if configuration.get("footprint_domain") != "two_lane_physical_road":
            raise ValueError("Expanded patterns require an explicit two-lane config")
        options["pattern"] = selected
        for key in ("footprint_domain", "static_yaw_offsets_deg", "station_window_m", "lateral_regions"):
            if key in configuration:
                options[key] = copy.deepcopy(configuration[key])
    return options


def run(args):
    if args.count == 3 and not args.experimental_three:
        raise ValueError("Count three requires --experimental-three; ordinary tunnel placement remains one or two")
    if args.candidates < 1 or (args.preview and args.candidates != 1):
        raise ValueError("Use at least one candidate; live preview accepts exactly one per run")
    config_bytes = args.config.read_bytes()
    configuration = yaml.safe_load(config_bytes)
    if configuration.get("schema") != "aimdrive_tunnel_static_config_v1":
        raise ValueError("Unsupported tunnel preparation config")
    if configuration.get("map") != "R_KR_PR_K-city_2025":
        raise ValueError("Tunnel preparation config map mismatch")
    validate_screening_settings(configuration["screening"])
    global_cfg, _, _ = load_configuration()
    loader = MGeoMapLoader(global_cfg["paths"]["mgeo_root"])
    options = configuration["models"]
    runtime = getattr(args, "runtime_catalog", None) or options.get("runtime_catalog")
    runtime_options = {"runtime_catalog_path": package_path(str(runtime))} if runtime else {}
    statics = load_static_models(package_path(options["static_catalog"]), **runtime_options)
    vehicles = load_vehicle_models(package_path(options["vehicle_registry"]), **runtime_options)
    placement_options = generation_options(configuration, getattr(args, "pattern", None))
    if args.model_acceptance is not None:
        from .models import accepted_model_pools
        if placement_options.get("pattern"):
            accepted = json.loads(args.model_acceptance.read_bytes())
            if accepted.get("config_sha256") != hashlib.sha256(config_bytes).hexdigest():
                raise ValueError("Expanded model acceptance belongs to a different placement config")
        statics, vehicles = accepted_model_pools(args.model_acceptance, statics, vehicles,
            source_geometry_sha256=tunnel_geometry(loader)["source_sha256"])
    if args.obstacle_model is not None:
        statics = {name: row for name, row in statics.items() if name in args.obstacle_model}
    allowed_vehicles = args.vehicle_model or options.get("vehicle_models")
    if allowed_vehicles is not None:
        vehicles = {name: row for name, row in vehicles.items() if name in allowed_vehicles}
    budget = args.time_budget if args.time_budget is not None else configuration["screening"]["time_budget_sec"]
    hold = args.hold_seconds if args.hold_seconds is not None else configuration["preview"]["hold_seconds"]
    if not math.isfinite(float(budget)) or budget <= 0:
        raise ValueError("Screening time budget must be finite and positive")
    if not math.isfinite(float(hold)) or not 0 <= hold <= 600:
        raise ValueError("Preview hold seconds must be finite and between 0 and 600")
    seed = args.seed if args.seed is not None else int.from_bytes(os.urandom(8), "big")
    output = args.output or Path(global_cfg["paths"]["runtime_root"]) / "tunnel_static" / (
        str(time.time_ns()) + "_" + str(seed)
    )
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary = {"schema": "aimdrive_tunnel_static_preparation_v1", "seed": seed,
               "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
               "expert_verified": False, "live_avoidance_verified": False,
               "candidate_count": args.candidates, "results": [], "output_dir": str(output),
               "reproduction": {"config_path": str(args.config.resolve()), "seed": seed,
                   "requested_count": args.count, "actor_kind": args.actor_kind,
                   "obstacle_models": args.obstacle_model,
                   "vehicle_models": args.vehicle_model or options.get("vehicle_models"),
                   "layout": placement_options, "time_budget_sec": budget,
                   "preview": args.preview, "hold_seconds": hold}}
    summary["screening_provenance"] = screening_provenance()
    if runtime:
        runtime_path = package_path(str(runtime))
        summary["runtime_catalog"] = {"path": str(runtime_path),
            "sha256": hashlib.sha256(runtime_path.read_bytes()).hexdigest()}
    if args.model_acceptance is not None:
        summary["model_acceptance"] = {
            "path": str(args.model_acceptance.resolve()),
            "sha256": hashlib.sha256(args.model_acceptance.read_bytes()).hexdigest(),
        }
    grpc = None
    try:
        if args.preview:
            publishers = _controller_publishers()
            if publishers:
                raise RuntimeError("Stop the active Ego controller before parked preview: " + ", ".join(publishers))
            from ...adapters.morai_grpc import MoraiGrpcClient
            from ...scenarios.full_loop_parts.tunnel_static import park_ego as _park

            connection_cfg = copy.deepcopy(global_cfg)
            connection_cfg["grpc"]["client_key"] = "tunnel_static_preview"
            grpc = MoraiGrpcClient(connection_cfg)
            grpc.connect()
            grpc.start_world(grpc.make_transform(*TUNNEL_NORMAL_START.xyz, TUNNEL_NORMAL_START.yaw_deg))
            grpc.resume_world()
            if not grpc.reset_actors():
                raise RuntimeError("Could not remove previous preview-owned actors")
            _park(grpc, grpc.get_ego(), {"ego_parking_commands": 0})
            from ...scenarios.segment_modes.tunnel_static_startup import block_and_park, align_parked_ego
            from ...scenarios.full_loop_parts.tunnel_static_parts.trial import settle_initial_ego
            block_and_park(grpc)
            settle_initial_ego(grpc)
            align_parked_ego(grpc, {}, settle_fn=settle_initial_ego)
            # This only prunes unavailable models; measured dimensions are never guessed.
            available_static = set(grpc.get_available_obstacle_models())
            available_vehicles = set(grpc.get_available_surround_vehicle_models())
            statics = {name: row for name, row in statics.items() if name in available_static}
            vehicles = {name: row for name, row in vehicles.items() if name in available_vehicles}
        summary["catalog_model_pools"] = {
            "obstacle": sorted(statics), "vehicle": sorted(vehicles),
        }
        for index in range(args.candidates):
            directory = output / ("candidate_%03d" % index)
            candidate = generate_candidate(loader, statics, vehicles, seed, count=args.count,
                candidate_index=index, actor_kind=args.actor_kind, allow_three=args.experimental_three, **placement_options)
            validate_candidate(candidate, loader)
            _write(directory / "candidate.json", candidate)
            screening = screen_candidate(candidate, loader, time_budget_sec=budget)
            _write(directory / "screening.json", screening)
            row = {"index": index, "count": candidate["count"],
                   "models": [item["actor_kind"] + ":" + item["model"] for item in candidate["obstacles"]],
                   "stations_m": [item["station_m"] for item in candidate["obstacles"]],
                   "screening_status": screening["status"], "path_found": screening["path_found"],
                   "candidate_path": str(directory / "candidate.json")}
            if args.preview:
                if not screening["path_found"]:
                    row["preview_status"] = "not_spawned_screening_failed"
                else:
                    from ...scenarios.full_loop_parts.tunnel_static import preview_candidate
                    result = preview_candidate(grpc, loader, candidate, hold_seconds=hold,
                        output_dir=directory, screen_fn=lambda data, map_loader, **kw: screen_candidate(
                            data, map_loader, time_budget_sec=budget, **kw))
                    row["preview_status"] = result["status"]
                    row["preview_failure_reason"] = result.get("failure_reason", "")
            summary["results"].append(row)
            summary["accepted_candidate_indices"] = [item["index"] for item in summary["results"] if item["path_found"]]
            summary["rejected_candidate_indices"] = [item["index"] for item in summary["results"] if not item["path_found"]]
            _write(output / "manifest.json", summary)
            print(json.dumps(row, ensure_ascii=False), flush=True)
        summary["status"] = "prepared" if all(item["path_found"] for item in summary["results"]) else "screening_rejected"
        if args.preview:
            summary["status"] = "valid_parked_preview" if summary["results"][0].get("preview_status") == "valid_preview" else "preview_failed"
        return summary
    except (Exception, KeyboardInterrupt) as error:
        summary.update(status="failed", failure_reason=str(error) or type(error).__name__)
        raise
    finally:
        cleanup_error = None
        try:
            if grpc is not None and grpc.world is not None:
                from ...scenarios.full_loop_parts.tunnel_static import park_ego as _park
                _park(grpc, grpc.get_ego(), {"ego_parking_commands": 0})
        except Exception as error:
            cleanup_error = error
            summary.update(status="failed", cleanup_failure=str(error))
        finally:
            try:
                if grpc is not None:
                    grpc.stop()
            finally:
                _write(output / "manifest.json", summary)
        if cleanup_error is not None:
            raise RuntimeError("Preview shutdown failed: " + str(cleanup_error)) from cleanup_error


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        summary = run(args)
    except (ValueError, RuntimeError) as error:
        parser.exit(2, str(error) + "\n")
    print("Result: " + summary["status"] + " | " + summary["output_dir"])
    print("Expert driving has not been tested or approved by this preparation.")
    return 0 if summary["status"] in ("prepared", "valid_parked_preview") else 1


if __name__ == "__main__":
    raise SystemExit(main())
