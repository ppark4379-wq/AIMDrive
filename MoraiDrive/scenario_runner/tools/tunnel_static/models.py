"""Audit native-scale tunnel models with one parked actor per model.

The source geometry catalogues remain sealed. Offline results describe nominal
geometry only. A separate acceptance file contains only successful live parked
previews, with report and catalogue hashes. This never approves Expert driving.
"""

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import time
import uuid

import yaml

from ...main_runner import load_configuration, package_path
from ...scenarios.segment_modes.modes import TUNNEL_NORMAL_START
from ...scenarios.full_loop_parts.tunnel_static import (
    check_tunnel_occupancy as _occupancy,
    generate_candidate,
    get_actor_id as _id,
    load_static_models,
    load_vehicle_models,
    park_ego as _park,
    preview_candidate,
    read_actor_inventory as _inventory,
    screen_candidate,
    tunnel_geometry,
)
from .candidates import PACKAGE_ROOT, _controller_publishers, _write
from ...scenarios.full_loop_parts.tunnel_static_parts.screening import (
    provenance_matches, screening_provenance, validate_screening_settings,
)


ACCEPTANCE_SCHEMA = "aimdrive_tunnel_model_acceptance_v1"


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PACKAGE_ROOT / "config/tunnel_static.yaml")
    parser.add_argument("--runtime-catalog", type=Path, help="Opt-in hash-bound campaign calibration; sealed catalogs stay unchanged")
    parser.add_argument("--live", action="store_true", help="Verify parked actors in the already loaded MORAI map")
    parser.add_argument("--actor-kind", choices=("mixed", "obstacle", "vehicle"), default="mixed")
    parser.add_argument("--model", action="append", help="Audit only these models; repeat for multiple models")
    parser.add_argument("--seed", type=int, default=20261006)
    parser.add_argument("--station-m", type=float, default=45.)
    parser.add_argument("--hold-seconds", type=float, default=5.)
    parser.add_argument("--time-budget", type=float)
    parser.add_argument("--offline-manifest", type=Path, help="Reuse a matching offline geometry audit; measured live screening still runs")
    parser.add_argument("--output", type=Path, help="New report directory; existing directories are refused")
    return parser


def _assert_controller_idle():
    publishers = _controller_publishers()
    if publishers:
        raise RuntimeError("Stop the active Ego controller before parked model verification: " + ", ".join(publishers))


class _GuardedEgo:
    """Stop issuing parking commands if another controller becomes active."""

    def __init__(self, actor):
        self.actor = actor

    def __getattr__(self, name):
        return getattr(self.actor, name)

    def set_velocity(self, value):
        _assert_controller_idle()
        return self.actor.set_velocity(value)

    def set_vehicle_gear(self, value):
        _assert_controller_idle()
        return self.actor.set_vehicle_gear(value)


class _BoundedStub:
    """Give owned SDK actor operations a deadline even when the SDK omits it."""

    def __init__(self, stub):
        self.stub = stub

    def __getattr__(self, name):
        method = getattr(self.stub, name)

        def invoke(*args, **kwargs):
            kwargs.setdefault("timeout", 1.)
            return method(*args, **kwargs)

        return invoke


def _connect_parked_world(global_cfg, geometry):
    """Attach without SDK constructor cleanup or simulation restart.

SimulationWorld.__init__ deletes actors for its client, and finalize stops the
simulation. Initialize only the SDK fields needed for owned spawns and Ego
handles; disconnect the transport directly on exit. No foreign actors are
deleted, moved, paused, or resumed.
"""
    from ...adapters.morai_grpc import MoraiGrpcClient
    connection = copy.deepcopy(global_cfg)
    connection["grpc"]["client_key"] = "tunnel_model_coverage_" + uuid.uuid4().hex
    grpc = MoraiGrpcClient(connection)
    try:
        from api.simulation_world import SimulationWorld
        from proto.morai.simulation.simulation_enum_pb2 import SYNC_MODE_TYPE_UNSPECIFIED

        grpc.connect()
        adapter = grpc.client._sim_adapter
        adapter._actor_stub = _BoundedStub(adapter._actor_stub)
        adapter._simulator_stub = _BoundedStub(adapter._simulator_stub)
        world = SimulationWorld.__new__(SimulationWorld)
        world._sim_adapter = adapter
        world._client_key = grpc.client_key
        world._actors = {}
        world._current_sync_mode = SYNC_MODE_TYPE_UNSPECIFIED
        world._ego_id = world._ego_label = "Ego"
        grpc.world = world
        grpc._patch_spawn_diagnostics()
        sdk_get_ego = grpc.get_ego
        grpc.get_ego = lambda: _GuardedEgo(sdk_get_ego())
        report = {"inventory_checks": 0, "ego_parking_commands": 0}
        ego = grpc.get_ego()
        from ...scenarios.segment_modes.tunnel_static_startup import block_and_park, align_parked_ego
        from ...scenarios.full_loop_parts.tunnel_static_parts.trial import settle_initial_ego
        _assert_controller_idle()
        block_and_park(grpc)
        state, _ = _park(grpc, ego, report)
        _occupancy(_inventory(grpc, report), _id(state), set(), geometry)
        _assert_controller_idle()
        if not ego.set_transform(grpc.make_transform(*TUNNEL_NORMAL_START.xyz, TUNNEL_NORMAL_START.yaw_deg)):
            raise RuntimeError("Ego parked baseline pose was rejected")
        block_and_park(grpc)
        time.sleep(.1)
        block_and_park(grpc)
        _park(grpc, ego, report)
        _assert_controller_idle()
        settle_initial_ego(grpc)
        align_parked_ego(grpc, {}, settle_fn=settle_initial_ego)
        _assert_controller_idle()
        return grpc
    except BaseException:
        grpc.client.disconnect()
        raise


def _available_models(grpc):
    from proto.morai.simulator.category_obstacles_pb2 import CategoryObstacles

    query = CategoryObstacles(vehicle=True, pedestrian=False, obstacle=True, spawn_point=False, map_object=False)
    response = grpc.world._sim_adapter._simulator_stub.GetAvailableObject(query, timeout=1.)
    if response is None or not hasattr(response, "obstacle") or not hasattr(response, "surround_vehicle"):
        raise RuntimeError("Available-model RPC is unavailable")
    return {"obstacle": set(response.obstacle), "vehicle": set(response.surround_vehicle)}


def _simulator_version(grpc):
    from proto.morai.common.type_pb2 import Empty

    response = grpc.world._sim_adapter._simulator_stub.GetSimulatorVersion(Empty(), timeout=1.)
    value = str(getattr(response, "value", ""))
    # Some MORAI builds expose the RPC while returning an empty version string.
    # Keep that absence explicit; actor availability/profile/hold checks provide
    # acceptance evidence independently of this descriptive metadata.
    return value or None


def _request_rows(statics, vehicles, actor_kind, requested, configured_vehicle_models=None):
    pools = {"obstacle": statics, "vehicle": vehicles}
    kinds = [actor_kind] if actor_kind != "mixed" else ["obstacle", "vehicle"]
    selected = set(requested or [])
    names = {"obstacle": set(statics), "vehicle": set(vehicles) | set(configured_vehicle_models or [])}
    rows = [(kind, name, pools[kind].get(name)) for kind in kinds for name in sorted(names[kind])
            if not selected or name in selected]
    missing = selected - {name for _, name, _ in rows}
    if missing:
        raise ValueError("Unknown model names in the selected pool: " + ", ".join(sorted(missing)))
    return rows


def _rpc_failed(preview):
    description = json.dumps(preview, ensure_ascii=False)
    return any(token in description for token in ("StatusCode.UNAVAILABLE", "StatusCode.DEADLINE_EXCEEDED",
                                                   "failed to connect", "Connection refused"))


def _offline_cache(path, summary):
    if path is None:
        return {}
    path = path.resolve()
    raw = path.read_bytes()
    manifest = json.loads(raw)
    if (manifest.get("schema") != "aimdrive_tunnel_model_coverage_v1"
            or manifest.get("status") != "completed_offline"
            or any(manifest.get(key) != summary.get(key) for key in (
                "config_sha256", "source_geometry_sha256", "seed", "station_m", "time_budget_sec"))):
        raise ValueError("Offline model audit inputs differ from the current verification")
    result = {}
    for row in manifest.get("results", []):
        directory = Path(row["report_dir"])
        candidate_path, screen_path = directory / "candidate.json", directory / "screening.json"
        if not candidate_path.is_file() or not screen_path.is_file():
            continue
        candidate = json.loads(candidate_path.read_bytes())
        screened_raw = screen_path.read_bytes()
        screened = json.loads(screened_raw)
        # A teammate may deliver a changed planner while models are being
        # audited. Never silently reuse an earlier implementation's result.
        if not provenance_matches(screened):
            continue
        result[(row["actor_kind"], row["model"])] = (candidate, screened, {
            "manifest_path": str(path), "manifest_sha256": hashlib.sha256(raw).hexdigest(),
            "screening_path": str(screen_path), "screening_sha256": hashlib.sha256(screened_raw).hexdigest()})
    return result


def _accepted_entry(row, preview, directory):
    return {"accepted": True, "actor_kind": row["actor_kind"], "model": row["model"],
            "scale": [1., 1., 1.], "geometry_source": copy.deepcopy(row["geometry_source"]),
            "hold_seconds": preview["hold_seconds"],
            "stationary_observed_seconds": preview["stationary_observed_seconds"],
            "preview_path": str((directory / "preview.json").resolve()),
            "preview_sha256": hashlib.sha256((directory / "preview.json").read_bytes()).hexdigest(),
            "station_m": row["station_m"], "api_profile_readback": row.get("api_profile_readback"),
            "simulator_version": row.get("simulator_version"),
            "screening_provenance": {key: preview.get("measured_route_screen", {}).get(key)
                for key in screening_provenance()},
            "expert_verified": False, "live_avoidance_verified": False,
            "physical_walls_verified": False, "physical_mesh_verified": False,
            "collision_behavior_verified": False}


def accepted_model_pools(acceptance_path, statics, vehicles, *, source_geometry_sha256=None):
    """Load a runtime allowlist only after current sources and reports match.

The result may restrict a random model pool, but cannot substitute for live
validation of a newly generated placement or for Expert driving validation.
"""
    path = Path(acceptance_path).resolve()
    document = json.loads(path.read_text())
    if document.get("schema") != ACCEPTANCE_SCHEMA or document.get("map") != "R_KR_PR_K-city_2025":
        raise ValueError("Unsupported tunnel model acceptance file")
    if source_geometry_sha256 is not None and document.get("source_geometry_sha256") != source_geometry_sha256:
        raise ValueError("Tunnel acceptance belongs to a different map geometry")
    original = {"obstacle": statics, "vehicle": vehicles}
    result = {kind: {} for kind in original}
    for entry in document.get("models", []):
        kind, name = entry.get("actor_kind"), entry.get("model")
        model = original.get(kind, {}).get(name)
        if (model is None or entry.get("accepted") is not True or entry.get("scale") != [1., 1., 1.]
                or entry.get("geometry_source") != model.get("geometry_source")):
            raise ValueError("Tunnel acceptance catalogue provenance mismatch: " + str(name))
        report_path = Path(entry["preview_path"])
        if not report_path.is_absolute():
            report_path = path.parent / report_path
        raw = report_path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != entry.get("preview_sha256"):
            raise ValueError("Tunnel acceptance preview hash mismatch: " + str(name))
        report = json.loads(raw)
        if not provenance_matches(report.get("measured_route_screen", {})):
            raise ValueError("Tunnel acceptance Jamming screening provenance mismatch: " + str(name))
        placements = report.get("planned", {}).get("obstacles", [])
        observed = report.get("observed", [])
        if (report.get("status") != "valid_preview" or report.get("cleanup", {}).get("verified") is not True
                or float(report.get("hold_seconds", 0)) < 5.
                or float(report.get("stationary_observed_seconds", 0)) < 5.
                or report.get("route_screened") is not True
                or report.get("measured_route_screen", {}).get("path_found") is not True
                or report.get("measured_route_screen", {}).get("geometry_checks", {}).get("passed") is not True
                or len(placements) != 1 or placements[0].get("model") != name
                or placements[0].get("actor_kind") != kind
                or placements[0].get("geometry_source") != model.get("geometry_source")
                or report.get("planned", {}).get("source_geometry", {}).get("sha256") != document.get("source_geometry_sha256")
                or len(observed) != 1 or observed[0].get("model") != name
                or observed[0].get("actor_kind") != kind
                or report.get("expert_verified") is not False
                or report.get("live_avoidance_verified") is not False):
            raise ValueError("Tunnel acceptance is not a verified parked preview: " + str(name))
        result[kind][name] = model
    return result["obstacle"], result["vehicle"]


def run(args):
    hold, station = float(args.hold_seconds), float(args.station_m)
    if not math.isfinite(hold) or not 5. <= hold <= 600.:
        raise ValueError("hold_seconds must be finite and in 5..600 for model acceptance")
    if not math.isfinite(station) or not 20. <= station <= 75.:
        raise ValueError("station_m must be finite and in 20..75")
    config_bytes = args.config.read_bytes()
    config = yaml.safe_load(config_bytes)
    if config.get("schema") != "aimdrive_tunnel_static_config_v1" or config.get("map") != "R_KR_PR_K-city_2025":
        raise ValueError("Unsupported tunnel model verification config")
    validate_screening_settings(config["screening"])
    budget = float(args.time_budget if args.time_budget is not None else config["screening"]["time_budget_sec"])
    if not math.isfinite(budget) or budget <= 0:
        raise ValueError("time_budget must be finite and positive")
    global_cfg, _, _ = load_configuration()
    from ...utils.map_loader import MGeoMapLoader
    loader = MGeoMapLoader(global_cfg["paths"]["mgeo_root"])
    geometry = tunnel_geometry(loader)
    runtime_path = args.runtime_catalog or config["models"].get("runtime_catalog")
    runtime_path = Path(package_path(runtime_path)).resolve() if runtime_path else None
    runtime_options = {"runtime_catalog_path": runtime_path} if runtime_path else {}
    statics = load_static_models(package_path(config["models"]["static_catalog"]), **runtime_options)
    vehicles = load_vehicle_models(package_path(config["models"]["vehicle_registry"]),
        None if runtime_path else config["models"].get("vehicle_models"), **runtime_options)
    requested = _request_rows(statics, vehicles, args.actor_kind, args.model,
                              config["models"].get("vehicle_models"))
    output = (args.output or Path(global_cfg["paths"]["runtime_root"]) / "tunnel_model_coverage" /
              (str(time.time_ns()) + "_" + uuid.uuid4().hex[:8])).resolve()
    output.mkdir(parents=True, exist_ok=False)
    summary = {"schema": "aimdrive_tunnel_model_coverage_v1", "map": config["map"],
               "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
               "source_geometry_sha256": geometry["source_sha256"], "seed": args.seed,
               "status": "started", "live_requested": args.live, "hold_seconds": hold,
               "station_m": station, "time_budget_sec": budget, "requested_model_count": len(requested),
               "expert_verified": False, "live_avoidance_verified": False,
               "physical_walls_verified": False, "results": [], "output_dir": str(output)}
    summary["screening_provenance"] = screening_provenance()
    if runtime_path:
        summary["runtime_catalog"] = {"path": str(runtime_path), "sha256": hashlib.sha256(runtime_path.read_bytes()).hexdigest()}
    acceptance = {"schema": ACCEPTANCE_SCHEMA, "map": config["map"],
                  "config_sha256": summary["config_sha256"], "source_geometry_sha256": geometry["source_sha256"],
                  "expert_verified": False, "live_avoidance_verified": False, "models": []}
    grpc = None
    try:
        cache = _offline_cache(args.offline_manifest, summary)
        if args.live:
            _assert_controller_idle()
            grpc = _connect_parked_world(global_cfg, geometry)
            client_key = getattr(grpc, "client_key", None)
            summary["owned_client_key"] = client_key if isinstance(client_key, str) else None
            available = _available_models(grpc)
            summary["simulator_version"] = _simulator_version(grpc)
            summary["simulator_version_verified"] = summary["simulator_version"] is not None
            summary["available_models"] = {key: sorted(value) for key, value in available.items()}
        for index, (kind, name, model) in enumerate(requested):
            directory = output / ("model_%03d_%s_%s" % (index, kind, hashlib.sha256(name.encode()).hexdigest()[:10]))
            directory.mkdir()
            row = {"index": index, "actor_kind": kind, "model": name, "scale": [1., 1., 1.],
                   "seed": args.seed, "station_m": station, "geometry_source": copy.deepcopy((model or {}).get("geometry_source")),
                   "geometry_verified": (model or {}).get("geometry_verified") is True,
                   "simulator_version": summary.get("simulator_version"),
                   "live_tested": False, "live_accepted": False, "report_dir": str(directory),
                   "nominal_footprint": {key: model[key] for key in ("width_m", "length_m", "center_offset_spawn_xy", "heading_offset_deg")} if model else None}
            summary["results"].append(row)
            if model is None:
                row.update(status="skipped_no_reviewed_geometry", reason="Configured model is absent from the evidence-backed geometry loader; no size was guessed")
            elif config.get("footprint_domain", "own_lane") == "own_lane" and model["width_m"] > float(config["layout"]["max_width_m"]):
                row.update(status="skipped_width_policy", reason="Reviewed width exceeds the configured one-lane placement policy")
            elif args.live and name not in available[kind]:
                row.update(status="skipped_unavailable", reason="Model absent from this simulator's available-model response")
            else:
                pool = {name: model}
                try:
                    layout_options = {}
                    if config.get("footprint_domain") == "two_lane_physical_road":
                        layout_options = dict(footprint_domain=config["footprint_domain"], pattern=None,
                            static_yaw_offsets_deg=config.get("static_yaw_offsets_deg"),
                            station_window_m=[station, station+.000001])
                    candidate = generate_candidate(loader, pool if kind == "obstacle" else {},
                        pool if kind == "vehicle" else {}, args.seed, count=1, candidate_index=0,
                        actor_kind=kind, station_min_m=station, station_max_m=station + .000001,
                        lateral_limit_m=0., max_width_m=float(config["layout"]["max_width_m"]), **layout_options)
                except ValueError as error:
                    row.update(status="skipped_nominal_footprint", reason=str(error))
                else:
                    row["station_m"] = candidate["obstacles"][0]["station_m"]
                    _write(directory / "candidate.json", candidate)
                    cached = cache.get((kind, name))
                    if cached is not None and cached[0] == candidate:
                        screened = copy.deepcopy(cached[1])
                        row["offline_screening_provenance"] = cached[2]
                    else:
                        screened = screen_candidate(candidate, loader, time_budget_sec=budget)
                    _write(directory / "screening.json", screened)
                    row.update(screening_status=screened["status"], nominal_path_found=screened["path_found"],
                               status="nominal_geometry_passed" if screened["path_found"] else "nominal_screening_inconclusive",
                               geometric_impossibility_proven=screened.get("geometric_impossibility_proven", False))
                    if args.live and screened["path_found"]:
                        _assert_controller_idle()
                        row["live_tested"] = True
                        row["status"] = "live_started"
                        _write(directory / "model.json", row)
                        _write(output / "manifest.json", summary)
                        preview = preview_candidate(grpc, loader, candidate, hold_seconds=hold,
                            output_dir=directory, screen_fn=lambda data, map_loader, **kw:
                                screen_candidate(data, map_loader, time_budget_sec=budget, **kw))
                        row.update(preview_status=preview["status"], failure_reason=preview.get("failure_reason", ""),
                                   cleanup=preview.get("cleanup", {}), stationary_samples=preview.get("stationary_samples", 0),
                                   stationary_observed_seconds=preview.get("stationary_observed_seconds", 0))
                        observed = preview.get("observed", [])
                        row["api_profile_readback"] = [item.get("reviewed_profile_readback") for item in observed]
                        row["actual_geometry"] = [{key: item.get(key) for key in (
                            "position_xyz", "rotation_xyz_deg", "world_bbox", "reviewed_envelope_at_observed_pose",
                            "width_m", "length_m", "heading_deg", "observed_transform_yaw_offset_deg",
                            "observed_bbox_yaw_offset_deg", "spawn_pose_readback")} for item in observed]
                        passed = (preview["status"] == "valid_preview" and preview.get("cleanup", {}).get("verified") is True
                                  and preview.get("stationary_observed_seconds", 0) >= hold
                                  and preview.get("route_screened") is True)
                        row.update(status="live_accepted" if passed else "live_failed", live_accepted=passed)
                        if passed:
                            acceptance["models"].append(_accepted_entry(row, preview, directory))
                        if preview.get("cleanup", {}).get("verified") is not True or _rpc_failed(preview):
                            summary.update(status="stopped", stop_reason="Owned cleanup could not be proven or a simulator RPC failed")
            _write(directory / "model.json", row)
            _write(output / "manifest.json", summary)
            _write(output / "acceptance.json", acceptance)
            print(json.dumps({key: row[key] for key in ("index", "actor_kind", "model", "status", "live_tested", "live_accepted")}), flush=True)
            if summary["status"] == "stopped":
                break
            if grpc is not None:
                _assert_controller_idle()
                # A health failure ends the run, rather than labelling the next model unsuitable.
                _inventory(grpc, {"inventory_checks": 0})
        if summary["status"] != "stopped":
            summary["status"] = "completed_live" if args.live else "completed_offline"
    except (Exception, KeyboardInterrupt) as error:
        summary.update(status="failed", failure_reason=str(error) or type(error).__name__)
        raise
    finally:
        try:
            if grpc is not None:
                try:
                    _assert_controller_idle()
                    _park(grpc, grpc.get_ego(), {"ego_parking_commands": 0})
                    summary["final_ego_parked"] = True
                except Exception as error:
                    summary.update(status="failed", final_ego_parked=False, shutdown_failure=str(error))
                finally:
                    grpc.client.disconnect()
        finally:
            summary["not_run_models"] = [{"actor_kind": kind, "model": name} for kind, name, _ in requested[len(summary["results"]):]]
            summary["accepted_models"] = [{"actor_kind": entry["actor_kind"], "model": entry["model"]} for entry in acceptance["models"]]
            statuses = [row.get("status", "interrupted") for row in summary["results"]]
            summary["status_counts"] = {status: statuses.count(status) for status in sorted(set(statuses))}
            _write(output / "manifest.json", summary)
            _write(output / "acceptance.json", acceptance)
    return summary


def main():
    parser = build_parser()
    try:
        summary = run(parser.parse_args())
    except (ValueError, RuntimeError, OSError) as error:
        parser.exit(2, str(error) + "\n")
    print("Result: " + summary["status"] + " | " + summary["output_dir"])
    return 0 if summary["status"] in ("completed_live", "completed_offline") else 1


if __name__ == "__main__":
    raise SystemExit(main())
