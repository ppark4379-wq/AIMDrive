"""Bounded offline use of the production Jamming planner and its road policy.

Each candidate runs in a disposable process. The nine recorded paths, their
connectors and return tail are selected by JammingPlanner itself. Passing here
is nominal geometry evidence, never physical-wall or driving approval.
"""

import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from scenario_runner.project_paths import EXPERT_ROOT, logical_source_path

from .layout import MAP_NAME, bbox_world, validate_candidate


PROJECT_ROOT = Path(__file__).resolve().parents[4]
EXPERT_CODE = EXPERT_ROOT / "src"
JAMMING_ROOT = EXPERT_ROOT / "data" / "recorded_paths" / "jamming"
ROUTE_PATH = PROJECT_ROOT / "scenario_runner/data/R_KR_PR_K-city_2025/2026_molit_comp_global_path.txt"
POLICY = {
    "planner": "production_JammingPlanner",
    "recorded_routes": 9,
    "mission": "full_loop_tunnel_normal",
    "start_original_route_index": 3723,
    "end_original_route_index": 4107,
    "expert_lookahead_tail_m": 35.0,
    "prediction_preview_points_at_spawn": 120,
    "local_reference_policy": "production_PathManager/config.local_path_size",
    "initial_path_position_tolerance_m": 0.25,
    "initial_path_heading_tolerance_deg": 15.0,
    "measured_pose_policy": "separate body/collision/alignment checks; never prepend a synthetic repositioning edge",
    "ego_body_projection_schema": "actor_flu_obb_roll_pitch_envelope_v1",
    "ego_body_max_abs_roll_pitch_deg": 3.0,
    "minimum_ego_body": {"rear_m": 0.85, "front_m": 3.85, "width_m": 2.17},
    "ego_body_projection_formula": "X_pad=sin(P)*(maxAbsY*sin(R)+maxAbsZ); Y_half=maxAbsY+sin(R)*maxAbsZ",
    "road_boundary_approach_m": 120.0,
    "road_boundary_departure_m": 70.0,
    "full_body_road_margin_m": 0.15,
    "planning_tunnel_body_road_margin_m": 0.35,
    "planning_road_reserve_basis": "paired tunnel interior only; connected approach/exit reference keeps runtime 0.15 m",
    "central_line_crossing_allowed": True,
    "worker_process_per_candidate": True,
}
WORKER_MODULE = "scenario_runner.scenarios.full_loop_parts.tunnel_static_parts.screening"
RESULT_MARKER = "AIMDRIVE_JAMMING_SCREENING_RESULT="


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _relative(path):
    return logical_source_path(path)


def screening_provenance():
    """Current source/library hashes, without importing Expert or ROS."""
    planning = EXPERT_CODE / "planning"
    sources = [Path(__file__), Path(__file__).with_name("live.py"),
               Path(__file__).with_name("layout.py"),
               Path(__file__).with_name("advanced_patterns.py"),
               planning.parent / "config/parameters/lattice.py",
               planning / "lattice/lattice_planner.py",
               planning / "common/reference_path.py",
               planning.parent / "autonomous_driving.py",
               planning.parent / "config/config.json",
               PROJECT_ROOT / "scenario_runner/scenarios/segment_modes/modes.py",
               PROJECT_ROOT / "scenario_runner/scenarios/segment_modes/route.py"]
    for directory in (planning / "jamming", planning / "static_obstacle",
                      planning / "common", planning.parent / "config/parameters"):
        sources.extend(sorted(directory.glob("*.py")))
    manifest = json.loads((JAMMING_ROOT / "routes.json").read_text(encoding="utf-8"))
    directory = manifest.get("directory", "recordings")
    if directory not in ("recordings", "prepared"):
        raise ValueError("Unsupported Jamming recording directory")
    names = manifest.get("routes", [])
    if len(names) != POLICY["recorded_routes"] or len(set(names)) != len(names):
        raise ValueError("Tunnel screening requires the nine production Jamming recordings")
    if any(not isinstance(name, str) or Path(name).name != name for name in names):
        raise ValueError("Invalid Jamming recording filename")
    recordings = [JAMMING_ROOT / "routes.json", EXPERT_CODE / "config/jamming_section.json"]
    recordings.extend(JAMMING_ROOT / directory / name for name in names)
    sources = {_relative(path): _file_hash(path) for path in sources}
    recordings = {_relative(path): _file_hash(path) for path in recordings}
    return {"planner_source_sha256": sources, "recorded_library_sha256": recordings,
            "original_route_sha256": _file_hash(ROUTE_PATH),
            "screening_policy": dict(POLICY), "screening_policy_sha256": _hash(POLICY)}


def provenance_matches(result):
    try:
        current = screening_provenance()
    except (OSError, ValueError, KeyError, TypeError):
        return False
    return all(result.get(key) == value for key, value in current.items())


def validate_screening_settings(settings):
    """Keep tool config and the fixed production policy from diverging."""
    aliases = {"planner": "planner", "recorded_routes": "recorded_routes",
               "road_boundary_approach_m": "road_boundary_approach_m",
               "road_boundary_departure_m": "road_boundary_departure_m",
               "central_line_crossing_allowed": "central_line_crossing_allowed"}
    for key, policy_key in aliases.items():
        if key in settings and settings[key] != POLICY[policy_key]:
            raise ValueError("Tunnel screening config differs from production policy: " + key)


def _expert_search():
    """Import only inside the disposable worker; no highway search fallback."""
    sys.path.insert(0, str(EXPERT_ROOT.parents[1]))
    importlib.import_module("AIM2026.Expert.src.planning.lattice")
    module = importlib.import_module("AIM2026.Expert.src.planning.jamming.recorded_planner")
    boundary = importlib.import_module("AIM2026.Expert.src.planning.jamming.road_boundary")
    point = importlib.import_module("AIM2026.Expert.src.interfaces.geometry").Point
    state = importlib.import_module("AIM2026.Expert.src.vehicle_state").VehicleState
    object_info = importlib.import_module("AIM2026.Expert.src.obstacle.object_info").ObjectInfo
    return module.JammingPlanner, boundary.RoadBoundary, point, state, object_info


def _finite(value, name, *, positive=False):
    if isinstance(value, bool):
        raise ValueError(name + " must be finite")
    value = float(value)
    if not math.isfinite(value) or (positive and value <= 0):
        raise ValueError(name + " must be finite" + (" and positive" if positive else ""))
    return value


def _boxes(candidate, measured_geometry):
    if measured_geometry is None:
        return [bbox_world(item) for item in candidate["obstacles"]]
    if len(measured_geometry) != candidate["count"]:
        raise ValueError("Measured geometry count differs from candidate")
    boxes = []
    for item in measured_geometry:
        center = item["center_xyz"]
        if len(center) != 3:
            raise ValueError("Measured box centre requires x, y, z")
        boxes.append({"center_xyz": [_finite(v, "measured centre") for v in center],
                      "heading_deg": _finite(item["heading_deg"], "measured heading"),
                      "length_m": _finite(item["length_m"], "measured length", positive=True),
                      "width_m": _finite(item["width_m"], "measured width", positive=True)})
    return boxes


def _ego_input(actual_ego_state):
    from ...segment_modes.modes import TUNNEL_NORMAL_START

    if actual_ego_state is None:
        return {"position_xyz": list(TUNNEL_NORMAL_START.xyz),
                "yaw_rad": math.radians(TUNNEL_NORMAL_START.yaw_deg), "speed_mps": 0.,
                "basis": "canonical_scenario_spawn"}
    position = actual_ego_state["position_xyz"]
    if len(position) != 3:
        raise ValueError("Measured Ego position requires x, y, z")
    yaw_deg = actual_ego_state.get("yaw_deg")
    if yaw_deg is None and "rotation_xyz_deg" in actual_ego_state:
        yaw_deg = actual_ego_state["rotation_xyz_deg"][2]
    yaw = (actual_ego_state["yaw_rad"] if "yaw_rad" in actual_ego_state
           else math.radians(_finite(yaw_deg, "measured Ego yaw")))
    return {"position_xyz": [_finite(v, "measured Ego position") for v in position],
            "yaw_rad": _finite(yaw, "measured Ego yaw"),
            "speed_mps": _finite(actual_ego_state.get("speed_mps", actual_ego_state.get("speed_norm_raw", 0.)), "measured Ego speed"),
            "raw_bbox_normalized": actual_ego_state.get("raw_bbox_normalized"),
            "basis": "observed_ego_state"}


def _body_input(actual_ego_footprint):
    if actual_ego_footprint is None:
        return None
    length = _finite(actual_ego_footprint["length_m"], "measured Ego length", positive=True)
    width = _finite(actual_ego_footprint["width_m"], "measured Ego width", positive=True)
    has_intrinsics = any(key in actual_ego_footprint for key in (
        "height_m", "center_actor_flu_m", "max_abs_roll_pitch_deg"))
    if has_intrinsics:
        height = _finite(actual_ego_footprint["height_m"], "measured Ego height", positive=True)
        center = actual_ego_footprint["center_actor_flu_m"]
        if len(center) != 3:
            raise ValueError("Intrinsic Ego box centre requires actor FLU x, y, z")
        forward, left, up = [_finite(v, "intrinsic Ego box centre") for v in center]
        maximum = _finite(actual_ego_footprint["max_abs_roll_pitch_deg"],
                          "Ego roll/pitch envelope", positive=True)
        if maximum != POLICY["ego_body_max_abs_roll_pitch_deg"]:
            raise ValueError("Ego roll/pitch envelope differs from the shared 3 degree policy")
        sine = math.sin(math.radians(maximum))
        max_abs_y = abs(left) + .5 * width
        max_abs_z = abs(up) + .5 * height
        # Bound the yaw-normalized Rz*Ry*Rx horizontal projection for every
        # roll/pitch combination in +/-3 degrees, including intrinsic centre
        # motion. Roads keep the same geometry; only the actual 3D body grows.
        longitudinal_pad = sine * (max_abs_y * sine + max_abs_z)
        projected = {"rear_m": .5 * length - forward + longitudinal_pad,
                     "front_m": .5 * length + forward + longitudinal_pad,
                     "width_m": 2. * (max_abs_y + sine * max_abs_z)}
    else:
        # Legacy offline callers have only a single 2D snapshot. New live
        # observations supply all intrinsic fields and always take the branch
        # above; incomplete 3D evidence raises instead of silently flattening.
        forward = _finite(actual_ego_footprint["center_forward_m"], "measured Ego forward offset")
        left = _finite(actual_ego_footprint.get("center_left_m", 0.), "measured Ego lateral offset")
        projected = {"rear_m": .5 * length - forward,
                     "front_m": .5 * length + forward,
                     "width_m": width + 2. * abs(left)}
    return {key: max(POLICY["minimum_ego_body"][key], value)
            for key, value in projected.items()}


def _mission_route():
    """Build exactly the canonical source window and Expert's private tail."""
    from ...segment_modes.modes import TUNNEL_NORMAL_START, TUNNEL_NORMAL_END
    from ...segment_modes.route import RouteWindow, build_expert_route_points, validate_boundary
    from types import SimpleNamespace

    points = [tuple(float(v) for v in line.split()[:3])
              for line in ROUTE_PATH.read_text().splitlines() if line.strip()]
    validate_boundary(points, TUNNEL_NORMAL_START)
    validate_boundary(points, TUNNEL_NORMAL_END)
    first, last = TUNNEL_NORMAL_START.index, TUNNEL_NORMAL_END.index
    context = SimpleNamespace(route_points=points[first:last + 1],
                              mission_route=RouteWindow(tuple(points), 0., first, last))
    return build_expert_route_points(context)


def _run_bounded(payload, budget):
    """A fresh process bounds imports, search, smoothing and full-body checks."""
    env = dict(os.environ)
    env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    process = subprocess.Popen([sys.executable, "-m", WORKER_MODULE, "--worker"],
                               cwd=PROJECT_ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True, env=env)
    try:
        stdout, stderr = process.communicate(json.dumps(payload, allow_nan=False), timeout=budget)
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate()
        return {"status": "search_timeout", "path_found": False,
                "reasons": ["Jamming planning exceeded the total worker deadline; impossibility is not proven."],
                "worker_terminated": True}
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
    for line in reversed(stdout.splitlines()):
        if line.startswith(RESULT_MARKER):
            return json.loads(line[len(RESULT_MARKER):])
    return {"status": "search_failed", "path_found": False,
            "reasons": ["Jamming worker exited without a result"],
            "worker_return_code": process.returncode, "worker_error": stderr[-2000:]}


def _complete_approach(selected_world, full_route, ego, np):
    """Keep PathManager's original-road approach before the preplanned entry.

The observed pose is checked separately. Connecting it to the nearest map
sample would invent a millimetre repositioning edge that the controller never
receives, and turn ordinary localization noise into extreme path curvature.
"""
    route = np.asarray(full_route, dtype=float)[:, :2]
    start = np.asarray(ego.position)
    entry = np.asarray(selected_world[0])
    first = int(np.argmin(np.linalg.norm(route - start, axis=1)))
    last = int(np.argmin(np.linalg.norm(route - entry, axis=1)))
    prefix = []
    if last > first:
        prefix.extend(row for row in route[first:last + 1]
                      if np.linalg.norm(row - entry) > 0.15)
    complete = np.asarray(prefix + [np.asarray(p) for p in selected_world], dtype=float)
    complete = complete[np.r_[True, np.linalg.norm(np.diff(complete, axis=0), axis=1) > 1e-5]]
    return complete


def _initial_alignment(complete, ego, np):
    """Fail closed on real start-pose disagreement with the commanded path."""
    delta = np.diff(complete, axis=0)
    lengths = np.linalg.norm(delta, axis=1)
    if not len(delta) or np.any(lengths <= 1e-7):
        return {"passed": False, "reason": "invalid_initial_reference"}
    # The controller starts with the nearest waypoint, not a later crossing
    # of the same map. Check only the first commanded segment at startup.
    tangent = delta[0] / lengths[0]
    difference = np.asarray(ego.position) - complete[0]
    along = float(difference @ tangent)
    lateral = float(difference @ np.array([-tangent[1], tangent[0]]))
    projected = complete[0] + np.clip(along, 0., lengths[0]) * tangent
    position_error = float(np.linalg.norm(np.asarray(ego.position) - projected))
    heading = math.atan2(tangent[1], tangent[0])
    heading_error = abs(math.atan2(math.sin(ego.yaw-heading), math.cos(ego.yaw-heading)))
    reason = ("initial_path_position_error" if position_error > POLICY["initial_path_position_tolerance_m"]
              else "initial_path_heading_error" if heading_error > math.radians(POLICY["initial_path_heading_tolerance_deg"])
              else "")
    return {"passed": not reason, "reason": reason,
            "position_error_m": position_error, "cross_track_error_m": lateral,
            "along_track_error_m": along, "heading_error_deg": math.degrees(heading_error),
            "position_tolerance_m": POLICY["initial_path_position_tolerance_m"],
            "heading_tolerance_deg": POLICY["initial_path_heading_tolerance_deg"],
            "actual_ego_position": list(ego.position), "commanded_path_start": complete[0].tolist()}


def _production_views(path, ego):
    """Use the ordinary PathManager's measured-pose local window."""
    manager_type = importlib.import_module("AIM2026.Expert.src.planning.common.reference_path").PathManager
    config = json.loads((EXPERT_CODE / "config/config.json").read_text())
    manager = manager_type(path, False, int(config["map"]["local_path_size"]))
    # Velocity is not an input to the fixed recorded-route selection. Supply
    # a benign profile only so the actual PathManager can return its window.
    manager.velocity_profile = [30. / 3.6] * len(path)
    reference, _ = manager.get_local_path(ego)
    prediction = path[manager.current_waypoint:
                      manager.current_waypoint + POLICY["prediction_preview_points_at_spawn"]]
    return reference, prediction, manager.current_waypoint


def _worker(payload):
    planner = None
    details = {}
    try:
        import numpy as np
        planner_type, boundary_type, point_type, state_type, object_type = _expert_search()
        boundary = boundary_type.from_links(payload["links"],
            approach_m=POLICY["road_boundary_approach_m"],
            departure_m=POLICY["road_boundary_departure_m"], body_policy=payload["body_policy"],
            planning_margin_m=POLICY["planning_tunnel_body_road_margin_m"],
            runtime_margin_m=POLICY["full_body_road_margin_m"])
        planner = planner_type(road_boundary=boundary, background=True)
        if planner.local.error:
            raise ValueError(planner.local.error)
        state = payload["ego"]
        ego = state_type(*state["position_xyz"][:2], state["yaw_rad"], state["speed_mps"])
        path = [point_type(*p[:2]) for p in payload["global_route"]]
        reference, prediction, waypoint = _production_views(path, ego)
        objects = [{"object_info": object_type(*box["center_xyz"][:2], 0.,
                    1 if entry["actor_kind"] == "vehicle" else 2,
                    name=entry["model"], is_static=True, size_x=box["length_m"],
                    size_y=box["width_m"], heading_deg=box["heading_deg"], unique_id=index)}
                   for index, (entry, box) in enumerate(zip(payload["entries"], payload["boxes"]))]
        details = {"road_boundary_policy": boundary.policy,
                   "road_boundary_policy_sha256": _hash(boundary.policy),
                   "road_boundary_source_sha256": boundary.source_sha256,
                   "obstacle_input": payload["boxes"], "ego_input": payload["ego"],
                   "local_reference_points": len(reference), "prediction_preview_points": len(prediction),
                   "reference_waypoint": waypoint,
                   "ego_body_projection_schema": POLICY["ego_body_projection_schema"],
                   "ego_body_projection_basis": ("intrinsic_RPY_envelope" if
                       payload["actual_ego_footprint"] is not None and
                       "center_actor_flu_m" in payload["actual_ego_footprint"] else
                       "legacy_2D_snapshot" if payload["actual_ego_footprint"] is not None else
                       "shared_default_body"),
                   "ego_body_input": payload["actual_ego_footprint"],
                   "measured_ego_body": payload["body_policy"]}
        planner.run(ego, reference, objects, prediction_path=prediction, global_path=path)
        if planner.local.pending is not None:
            worker = planner.local.pending.result()
            selected_local = worker.path
            selected_names = worker.selected_names
            search_details = worker.search_details
            rejections = worker.rejections
            # Invoke the ordinary planner again with the SAME measured state;
            # keep its actual approach outcome as evidence, not an oracle path.
            actual_result = planner.run(ego, reference, objects, prediction_path=prediction, global_path=path)
        else:
            worker = planner.local
            selected_local = worker.path
            selected_names = worker.selected_names
            search_details = worker.search_details
            rejections = worker.rejections
            actual_result = planner.run(ego, reference, objects, prediction_path=prediction, global_path=path)
        details.update(selected_routes=list(selected_names), search_details=dict(search_details),
                       edge_rejections=dict(rejections),
                       actual_planner_reason=getattr(actual_result, "reason", None),
                       planner_debug=getattr(actual_result, "debug_details", {}))
        if not selected_local or not selected_names:
            status = "search_timeout" if search_details.get("status") == "timeout" else "no_path_found"
            return dict(details, status=status, path_found=False,
                        reasons=["Production JammingPlanner found no complete recorded route; impossibility is not proven."])
        selected_world = [point_type(*planner.section.to_world(p)) for p in selected_local]
        complete = _complete_approach(selected_world, payload["global_route"], ego, np)
        checks = planner.validate_world_path(complete, [item["object_info"] for item in objects],
                                             ego=ego, global_path=path, planning=True)
        alignment = _initial_alignment(complete, ego, np)
        checks["initial_alignment"] = alignment
        if not alignment["passed"]:
            checks.update(passed=False, reason=alignment["reason"])
        details.update(geometry_checks=checks, planning_entry_world=list(selected_world[0]),
                       unchanged_approach_included=True, path_sample_count=len(complete))
        if not checks["passed"]:
            return dict(details, status="geometry_rejected", path_found=False,
                        reasons=[checks.get("reason", "production_Jamming_geometry_recheck_failed")])
        return dict(details, status="nominal_path_found", path_found=True, path=complete.tolist(),
                    cost=float(np.linalg.norm(np.diff(complete, axis=0), axis=1).sum()), reasons=[])
    except (ImportError, ModuleNotFoundError) as error:
        return dict(details, status="search_unavailable", path_found=False,
                    reasons=[type(error).__name__ + ": " + str(error)])
    except Exception as error:
        return dict(details, status="search_failed", path_found=False,
                    reasons=[type(error).__name__ + ": " + str(error)])
    finally:
        if planner is not None:
            planner.close()


def screen_candidate(candidate, map_loader, *, time_budget_sec=15., measured_geometry=None,
                     actual_ego_state=None, actual_ego_footprint=None,
                     start_station_m=None, rejoin_station_m=None):
    """Screen the actual Jamming maneuver, from mission spawn through return.

Measured obstacle boxes are passed unchanged into ObjectInfo. Measured Ego
state and body are optional for offline catalogue proposals and required by
the live seam. The overall budget includes subprocess import and final checks.
"""
    begin = time.monotonic()
    basis = ("observed conservative mesh/API/catalogue box union as ObjectInfo"
             if measured_geometry is not None else "catalogue footprint boxes as ObjectInfo")
    result = {"schema": "aimdrive_tunnel_static_screening_v1", "map": MAP_NAME,
              "candidate_seed": candidate.get("seed"), "candidate_index": candidate.get("candidate_index"),
              "status": "not_started", "path_found": False, "path": [], "cost": None, "reasons": [],
              "expert_verified": False, "live_avoidance_verified": False,
              "physical_walls_verified": False, "geometric_impossibility_proven": False,
              "geometry_basis": "production Jamming nine recorded paths + nominal MGeo 7 m road + " + basis}
    try:
        validate_candidate(candidate, map_loader)
        budget = _finite(time_budget_sec, "Screening time budget", positive=True)
        if start_station_m is not None or rejoin_station_m is not None:
            raise ValueError("Jamming screening uses the production scenario start and return; station overrides are unsupported")
        boxes = _boxes(candidate, measured_geometry)
        ego = _ego_input(actual_ego_state)
        body = _body_input(actual_ego_footprint)
        if actual_ego_footprint is not None and "center_actor_flu_m" in actual_ego_footprint:
            if actual_ego_state is None or "rotation_xyz_deg" not in actual_ego_state:
                raise ValueError("Intrinsic 3D Ego screening requires observed roll/pitch")
            raw_attitude = actual_ego_state["rotation_xyz_deg"]
            angles = [math.degrees(math.atan2(math.sin(math.radians(_finite(v, "Ego roll/pitch"))),
                                               math.cos(math.radians(float(v)))))
                      for v in raw_attitude[:2]]
            if len(angles) != 2 or any(abs(angle) > POLICY["ego_body_max_abs_roll_pitch_deg"] for angle in angles):
                raise ValueError("Measured Ego roll/pitch exceeds the shared 3 degree projection envelope")
            ego["roll_pitch_deg"] = angles
        result.update(screening_provenance(), time_budget_sec=budget)
        links = getattr(map_loader, "link_set", None)
        if not isinstance(links, dict):
            raise ValueError("Jamming screening requires the complete MGeo link set")
        payload = {"links": links, "boxes": boxes, "entries": candidate["obstacles"],
                   "ego": ego, "body_policy": body, "global_route": _mission_route(),
                   "actual_ego_footprint": actual_ego_footprint}
    except (ValueError, KeyError, TypeError, OSError) as error:
        result.update(status="invalid_candidate", reasons=[str(error)], elapsed_sec=time.monotonic() - begin)
        return result
    remaining = budget - (time.monotonic() - begin)
    if remaining <= 0.:
        result.update(status="search_timeout", reasons=["Screening deadline expired before planner startup."])
    else:
        result.update(_run_bounded(payload, remaining))
        if result["path_found"] and not provenance_matches(result):
            result.update(status="search_failed", path_found=False, path=[], cost=None,
                          reasons=["Production planner, recordings or route policy changed during screening."])
    result["elapsed_sec"] = time.monotonic() - begin
    return result


if __name__ == "__main__":
    if sys.argv[1:] != ["--worker"]:
        raise SystemExit("Use tunnel_static_candidates to invoke screening")
    print(RESULT_MARKER + json.dumps(_worker(json.load(sys.stdin)), allow_nan=False), flush=True)
