"""A curated tunnel fixture for an already-owned full-loop drive.

The ordinary scenario owns Ego, its single Expert and the complete route.  This
adapter owns only a separate NPC/obstacle namespace.  It neither parks Ego nor
starts a world, a controller or an offline planner.  Historical library proof
is checked at selection; fresh evidence here covers placement, stationary body
geometry and owned cleanup, not the segment trial's Expert-avoidance contract.
"""

import copy
import math
from pathlib import Path
import time
from types import SimpleNamespace
import uuid

from ...utils import safe_print as print
from .static_obstacle import lock_static_obstacle_transform
from .tunnel_static_parts import catalog, diversity, live, replay
from .tunnel_static_parts.layout import bbox_world, normalize_footprint, tunnel_geometry, validate_candidate
from .tunnel_static_parts.runtime import atomic_json


CHECK_PERIOD_SEC = 0.5
RETIRE_AFTER_EXIT_M = 30.0


def _save(context):
    path = getattr(context, "competition_static_report_path", None)
    if path is not None:
        atomic_json(Path(path), context.competition_static_result)


def _require_pins(state):
    for path, expected in state.pins.items():
        if replay.digest(path) != expected:
            raise RuntimeError("Competition static source/evidence changed: " + str(path))


def _environment_models(library, document, geometry, pins):
    """Keep sealed native shape/measurement proof, independent of planner code."""
    paths = {}
    for name in ("model_acceptance", "runtime_catalog"):
        meta = document[name]
        path = replay._resolve(meta["path"], library.parent)
        data, sha = replay._read(path, meta["sha256"])
        paths[name] = (path, data)
        pins[str(path)] = sha
    acceptance_path, acceptance = paths["model_acceptance"]
    runtime_path, runtime = paths["runtime_catalog"]
    if (acceptance.get("schema") != "aimdrive_tunnel_model_acceptance_v1"
            or acceptance.get("map") != "R_KR_PR_K-city_2025"
            or acceptance.get("source_geometry_sha256") != geometry["source_sha256"]):
        raise ValueError("Curated model acceptance has incompatible mapped geometry")
    pools = {"obstacle": catalog.load_static_models(runtime_catalog_path=runtime_path),
             "vehicle": catalog.load_vehicle_models(runtime_catalog_path=runtime_path)}
    contracts = {}
    for row in acceptance.get("models", []):
        identity = row.get("actor_kind"), row.get("model")
        profile = pools.get(identity[0], {}).get(identity[1])
        if (profile is None or identity in contracts or row.get("accepted") is not True
                or row.get("scale") != [1., 1., 1.]
                or row.get("geometry_source") != profile.get("geometry_source")):
            raise ValueError("Curated native model/profile mismatch: " + repr(identity))
        path = replay._resolve(row["preview_path"], acceptance_path.parent)
        preview, sha = replay._read(path, row["preview_sha256"])
        pins[str(path)] = sha
        planned, observed = preview.get("planned", {}).get("obstacles", []), preview.get("observed", [])
        screen = preview.get("measured_route_screen", {})
        if (preview.get("status") != "valid_preview" or preview.get("cleanup", {}).get("verified") is not True
                or float(preview.get("hold_seconds", 0.)) < 5.
                or float(preview.get("stationary_observed_seconds", 0.)) < 5.
                or preview.get("route_screened") is not True or screen.get("path_found") is not True
                or screen.get("geometry_checks", {}).get("passed") is not True
                or len(planned) != 1 or len(observed) != 1
                or (planned[0].get("actor_kind"), planned[0].get("model")) != identity
                or (observed[0].get("actor_kind"), observed[0].get("model")) != identity
                or planned[0].get("scale") != [1., 1., 1.]
                or planned[0].get("geometry_source") != profile.get("geometry_source")
                or normalize_footprint(planned[0]["planned_bbox"]) != normalize_footprint(profile)
                or preview.get("expert_verified") is not False or preview.get("live_avoidance_verified") is not False
                or preview.get("planned", {}).get("source_geometry", {}).get("sha256") != geometry["source_sha256"]):
            raise ValueError("Sealed native model placement/measurement proof incomplete: " + repr(identity))
        contracts[identity] = profile
    if len(contracts) != 64:
        raise ValueError("Curated native model acceptance must contain all 64 measured models")
    for row in runtime.get("models", []):
        if (row.get("actor_kind"), row.get("model")) in contracts:
            path = replay._resolve(row["measurement_path"], runtime_path.parent)
            _, pins[str(path)] = replay._read(path, row["measurement_sha256"])
    for path in (catalog.DEFAULT_STATIC_CATALOG,
                 *catalog.DEFAULT_VEHICLE_REGISTRY.parent.glob("*.json")):
        pins[str(path.resolve())] = replay.digest(path)
    return contracts


def _source_comparison(previous, current):
    old = previous.get("planner_source_sha256", {})
    new = current.get("planner_source_sha256", {})
    return {
        "planner_sources_equal": old == new,
        "changed_source_paths": sorted(path for path in set(old) | set(new) if old.get(path) != new.get(path)),
        "policy_equal": all(previous.get(key) == value for key, value in current.items()
                            if key != "planner_source_sha256"),
        "current_planner_acceptance_required": False,
        "historical_Expert_approval_transferred": False,
    }


def _select_layout(context):
    """Verify environment evidence without promoting historical Expert approval.

    The segment replay's source-compatibility and current-planner acceptance
    gates are deliberately not invoked: this mode provides a physical scene to
    the existing full-loop Expert, and claims no fresh planner/path proof.
    """
    from ..segment_modes.tunnel_static_replay import DEFAULT_LIBRARY, _curation_pins

    library = DEFAULT_LIBRARY.resolve()
    pins = _curation_pins(library)
    document, library_sha = replay._read(library, pins[str(library)])
    geometry = tunnel_geometry(context.map_loader)
    contracts = _environment_models(library, document, geometry, pins)
    provenance = replay.screening.screening_provenance()
    eligible, fingerprints = [], set()
    for row in document["candidates"]:
        paths, hashes, evidence = {}, {}, {}
        for name in ("candidate", "preview_report", "drive_report", "ego_trace",
                     "trial_manifest", "planner_diagnostics"):
            default = Path(row["drive_report_path"]).parent / {
                "trial_manifest": "manifest.json", "planner_diagnostics": "planner_diagnostics.json"}.get(name, "")
            path = replay._resolve(row.get(name + "_path", default), library.parent)
            expected = row.get(name + "_sha256")
            # The curated rows seal the first four files. As in the segment
            # loader, legacy auxiliary reports may have no historical hash;
            # pin their current bytes without inventing a historical signature.
            if expected is None and name in ("trial_manifest", "planner_diagnostics"):
                expected = replay.digest(path)
            if not isinstance(expected, str) or len(expected) != 64 or replay.digest(path) != expected:
                raise ValueError("Curated " + name + " evidence SHA256 mismatch")
            paths[name], hashes[name], pins[str(path)] = str(path), expected, expected
            if name != "ego_trace":
                evidence[name], _ = replay._read(path, expected)
        candidate, preview, drive = (evidence[key] for key in ("candidate", "preview_report", "drive_report"))
        manifest, planner = evidence["trial_manifest"], evidence["planner_diagnostics"]
        conditions = drive.get("completion_conditions", {})
        if (drive.get("status") != "passed" or not all(drive.get(key) is True for key in diversity.STRICT_FLAGS)
                or not all(conditions.get(key) is True for key in diversity.STRICT_CONDITIONS)
                or not all(value is True for value in conditions.values())
                or drive.get("failed_completion_conditions") or drive.get("collision_objects")
                or manifest.get("status") != "passed" or manifest.get("expert_verified") is not True
                or manifest.get("live_avoidance_verified") is not True
                or manifest.get("fixture_result", {}).get("status") != "valid_driving_trial"
                or manifest.get("fixture_result", {}).get("cleanup", {}).get("verified") is not True
                or preview.get("status") != "valid_driving_trial" or preview.get("cleanup", {}).get("verified") is not True
                or row.get("fixture_cleanup_verified") is not True or row.get("recovery_verified") is not True
                or planner.get("validated_path_seen") is not True or int(planner.get("calls", 0)) < 1):
            raise ValueError("Curated historical trial proof incomplete")
        screen = preview.get("measured_route_screen", {})
        if (screen.get("status") != "nominal_path_found" or screen.get("path_found") is not True
                or screen.get("geometry_checks", {}).get("passed") is not True
                or candidate.get("source_geometry", {}).get("sha256") != geometry["source_sha256"]):
            raise ValueError("Curated mapped placement/geometry proof incomplete")
        replay._road_body_compatible(screen, drive, planner, context.map_loader.link_set)
        validate_candidate(candidate, context.map_loader)
        for placement in candidate["obstacles"]:
            profile = contracts.get((placement.get("actor_kind"), placement.get("model")))
            if (profile is None or placement.get("scale") != [1., 1., 1.]
                    or placement.get("geometry_source") != profile.get("geometry_source")
                    or normalize_footprint(placement["planned_bbox"]) != normalize_footprint(profile)):
                raise ValueError("Curated candidate differs from its pinned native model shape")
        actual = diversity.path_features(replay._trace(paths["ego_trace"]), geometry,
                                        evidence_kind="historical_completed_actual_trace")
        sealed = diversity.path_features([point["position_xyz"] for point in drive.get("trajectory", [])],
                                        geometry, evidence_kind="historical_completed_actual_trace")
        if diversity.compare_paths(actual, sealed)["duplicate"] is not True:
            raise ValueError("Curated actual trace differs from its sealed historical trajectory")
        fingerprint = diversity.layout_fingerprint(candidate)
        if fingerprint in fingerprints:
            raise ValueError("Curated layout identity is duplicated")
        fingerprints.add(fingerprint)
        stations = [float(placement["station_m"]) for placement in candidate["obstacles"]]
        kinds = {placement["actor_kind"] for placement in candidate["obstacles"]}
        eligible.append(dict(id=row["candidate_sha256"], candidate=candidate, paths=paths, file_sha256=hashes,
            strata=dict(count=len(stations), longitudinal_third=min(2, int(sum(stations) / len(stations) / geometry["length_m"] * 3)),
                        actor_kind="mixed" if len(kinds) > 1 else next(iter(kinds)), actual_behavior=actual["realized_pattern"]),
            source_comparison=_source_comparison(screen, provenance)))
    selected, balance = replay.balanced_choice(eligible, context.run_seed)
    pins.update(replay.current_runtime_sources())
    project = Path(__file__).resolve().parents[3]
    for key in ("planner_source_sha256", "recorded_library_sha256"):
        for path, expected in provenance.get(key, {}).items():
            path = Path(path)
            pins[str(path if path.is_absolute() else project / path)] = expected
    pins[str(Path(__file__).resolve())] = replay.digest(__file__)
    return selected, geometry, pins, {
        "library_path": str(library), "library_sha256": library_sha,
        "eligible_count": len(eligible), "balance_decisions": balance,
        "historical_evidence_status": "sealed_curated_environment_only",
        "source_comparison": selected["source_comparison"],
        "historically_pinned_evidence": ["candidate", "preview_report", "drive_report", "ego_trace"],
        "legacy_auxiliary_reports_pinned_for_this_run": ["trial_manifest", "planner_diagnostics"],
    }


def prepare(context):
    """Select and pin one existing curated layout without contacting MORAI."""
    selected, geometry, pins, selection = _select_layout(context)
    candidate = copy.deepcopy(selected["candidate"])
    validate_candidate(candidate, context.map_loader)
    report = dict(
        schema="aimdrive_full_loop_competition_static_environment_v1",
        phase="prepared", run_seed=context.run_seed, selected_candidate_sha256=selected["id"],
        selection=selection, planned=candidate, environment_only=True,
        historical_library_selection_verified=True, fresh_placement_verified=False,
        expert_verified=False, live_avoidance_verified=False, route_screened=False,
        physical_mesh_verified=False, physical_walls_verified=False,
        collision_behavior_verified=False, tunnel_passed=False, completed=False,
        inventory_checks=0, observed=[], spawn_readbacks=[], stationary_samples=0,
        cleanup={"verified": False, "actors": []},
    )
    state = SimpleNamespace(candidate=candidate, geometry=geometry, pins=pins,
                            owned=[], references={}, grpc=None, baseline={},
                            ego_id=None, last_check=float("-inf"))
    context.competition_static_state = state
    context.competition_static_result = report
    _require_pins(state)
    _save(context)
    return True


def _fixture_client(parent):
    """Borrow transport without invoking SimulationWorld's deleting constructor."""
    if getattr(parent, "world", None) is None:
        raise RuntimeError("Competition static arm requires the existing started world")
    fixture = copy.copy(parent)
    fixture.client_key = "competition_static_" + uuid.uuid4().hex
    fixture.world = copy.copy(parent.world)
    fixture.world._client_key = fixture.client_key
    fixture.world._actors = {}
    return fixture


def _preserved_baseline(context, inventory, ego_id):
    """Parent-owned actors may retire normally; every other baseline actor stays."""
    parent = context.grpc
    owned_ids = set()
    for handle in getattr(context, "scenario_actors", []):
        info = handle.get_object_info()
        if getattr(info, "client_key", None) != parent.client_key:
            raise RuntimeError("Parent actor lacks its original namespace ownership pin")
        identity = str(handle.get_id())
        if identity not in inventory:
            identity = live._id(live._read_one(parent, handle))
        if identity not in inventory:
            raise RuntimeError("Parent actor cannot be bound to the setup inventory")
        owned_ids.add(identity)
    return {identity: value for identity, value in inventory.items()
            if identity == ego_id or identity not in owned_ids}


def _spawn_readback(grpc, entry, before, placement, report):
    """Reuse exact binding checks, without the preview's Ego P/hold requirement."""
    started = time.monotonic()
    deadline = started + live.SPAWN_READBACK_TIMEOUT_SEC
    diagnostic = {"request_id": entry["request_id"], "status": "waiting", "attempts": 0}
    report["spawn_readbacks"].append(diagnostic)
    for attempt in range(live.SPAWN_READBACK_ATTEMPTS):
        diagnostic["attempts"] = attempt + 1
        inventory = live._inventory(grpc, report, deadline=deadline)
        try:
            identity = live._resolve_raw(
                grpc, entry["handle"], inventory, set(before), placement["model"],
                deadline=deadline, placement=placement, owned_entry=entry,
                diagnostics=diagnostic,
            )
        except Exception as error:
            if isinstance(error, live._OwnedBaselineOverlap):
                entry["forbidden_baseline_actor_id"] = error.actor_id
            retry = isinstance(error, live._SpawnReadbackPending) or live._unsupported_actor_lookup(error)
            if not retry or attempt + 1 == live.SPAWN_READBACK_ATTEMPTS or time.monotonic() >= deadline:
                diagnostic.update(status="failed", failure_reason=str(error))
                raise
            time.sleep(min(live.SPAWN_READBACK_RETRY_SEC, max(0., deadline - time.monotonic())))
            continue
        diagnostic.update(status="verified", raw_actor_id=identity)
        return inventory, identity
    raise RuntimeError("Competition static spawn registration was not observed")


def _settle(grpc, entry, report):
    reference, stable = None, 0
    for attempt in range(40):
        inventory = live._inventory(grpc, report)
        if entry["raw_id"] not in inventory:
            raise RuntimeError("Competition static actor disappeared while settling")
        actor_state = inventory[entry["raw_id"]]
        observed = live._observation(actor_state)
        try:
            live._stationary(observed, reference, "Competition static actor")
        except RuntimeError:
            reference, stable = None, 0
            if attempt == 39:
                raise
        else:
            reference = reference or copy.deepcopy(observed)
            stable += 1
            if stable >= 8:
                observed["initial_stationary_samples"] = stable
                return actor_state, observed
        if attempt == 39:
            raise RuntimeError("Competition static actor did not settle across eight samples")
        time.sleep(0.1)


def _spawn_one(context, placement, index):
    state, report = context.competition_static_state, context.competition_static_result
    grpc = state.grpc
    before = live._inventory(grpc, report)
    live._occupancy(before, state.ego_id, set(state.references), state.geometry)
    request = "competition_static_" + uuid.uuid4().hex + "_" + str(index)
    kind = placement["actor_kind"]
    handle = live._handle_for_request(grpc, kind, request)
    entry = dict(handle=handle, request_id=request, raw_id=None, actor_kind=kind,
                 requested_model=placement["model"], placement=copy.deepcopy(placement),
                 source_handle_id=str(handle.get_object_info().id.value),
                 source_client_key=handle.get_object_info().client_key)
    live._fixture_namespace(grpc, [entry])
    state.owned.append(entry)
    transform = grpc.make_transform(*(placement["position"][axis] for axis in "xyz"), placement["yaw_deg"])
    try:
        if kind == "vehicle":
            created = grpc.world.spawn_vehicle(transform=transform, model_name=placement["model"],
                label=request, request_id=request, velocity=0., pause=False, multi_ego=False)
        else:
            from proto.morai.common.type_pb2 import Vector3
            created = grpc.world.spawn_obstacle(transform=transform, model_name=placement["model"],
                label=request, request_id=request, scale=Vector3(x=1., y=1., z=1.))
        if created is not None:
            if created.get_object_info().client_key != entry["source_client_key"]:
                raise RuntimeError("Static spawn returned a different fixture namespace")
            entry.update(handle=created, source_handle_id=str(created.get_object_info().id.value))
    except Exception as error:
        entry["spawn_rpc_error"] = str(error)
    handle = entry["handle"]
    fresh, identity = _spawn_readback(grpc, entry, before, placement, report)
    entry.update(raw_id=identity, raw_object_info=copy.deepcopy(fresh[identity].actor_info),
                 binding_phase="initial_before_parking")
    live._owned_raw_info(grpc, entry, entry["source_client_key"], report=report)
    controls = {}
    if kind == "obstacle":
        shim = SimpleNamespace(grpc=SimpleNamespace(
            get_all_obstacle_states=lambda: list(live._inventory(grpc, report).values())))
        frozen, diagnostics = lock_static_obstacle_transform(
            shim, handle, identity, transform,
            tuple(placement["position"][axis] for axis in "xyz"),
            attempts=8, stable_samples=2, retry_seconds=.1,
        )
        controls["static_transform_lock"] = diagnostics
        if frozen is None:
            raise RuntimeError("Competition static obstacle could not be fixed at its pose")
    else:
        live._check_initial_vehicle_dimensions(placement, live._observation(fresh[identity]))
        controls["speed_limit_zero"] = bool(handle.set_vehicle_dynamics_speed_limit(True, 0., reset=False))
        controls["velocity_zero"] = bool(handle.set_velocity(0.))
        route, waypoint = live._native_npc_route(context.map_loader, placement, state.geometry)
        controls["native_route"] = bool(grpc.set_vehicle_route(handle, route, decision_range=30.,
            route_waypoint_indices={route[0]: waypoint}, label="competition_static_parked_npc"))
        if hasattr(handle, "control"):
            from proto.morai.actor.actor_enum_pb2 import LONG_CMD_TYPE_SPEED
            controls["brake_zero_speed"] = bool(handle.control(long_cmd_type=LONG_CMD_TYPE_SPEED,
                throttle=0., brake=1., steer=0., velocity=0., acceleration=0., frame=0))
        if not all(controls[key] for key in ("speed_limit_zero", "velocity_zero", "native_route")):
            raise RuntimeError("Competition parked NPC rejected stationary commands")
    actor_state, observed = _settle(grpc, entry, report)
    requested = [float(placement["position"][axis]) for axis in "xyz"]
    z_limit = live.VEHICLE_GROUND_SNAP_Z_TOLERANCE_M if kind == "vehicle" else live.SPAWN_TOLERANCE_M
    if (math.dist(requested[:2], observed["position_xyz"][:2]) > live.SPAWN_TOLERANCE_M
            or abs(requested[2] - observed["position_xyz"][2]) > z_limit):
        raise RuntimeError("Competition static pivot differs from the selected placement")
    if live._model(grpc, actor_state) != placement["model"]:
        raise RuntimeError("Competition static model readback differs from the selected model")
    planned = bbox_world(placement)
    if live._angle_distance(planned["heading_deg"], observed["raw_bbox_normalized"]["heading_deg"]) > 3.:
        raise RuntimeError("Competition static heading differs from its reviewed convention")
    effective = copy.deepcopy(placement)
    effective["position"] = dict(zip("xyz", observed["position_xyz"]))
    effective["yaw_deg"] = observed["rotation_xyz_deg"][2] - float(placement["planned_bbox"]["heading_offset_deg"])
    box = live._conservative_box(live._conservative_box(planned, bbox_world(effective)), observed["raw_bbox_normalized"])
    if kind == "vehicle":
        collider = live._reviewed_collider_at_pose(placement, observed)
        if collider is not None:
            box = live._conservative_box(box, collider)
    entry.update(raw_observation=copy.deepcopy(observed), binding_phase="frozen")
    exact = live._observation(live.validated_global_actor_state(
        grpc, entry, entry["source_client_key"], report=report))
    if kind == "vehicle":
        observed["reviewed_profile_readback"] = live._check_npc_api_variant(placement, observed, exact)
    observed.update(box)
    observed.update(model=placement["model"], actor_kind=kind, request_id=request,
                    stationary_controls_accepted=controls)
    state.references[identity] = copy.deepcopy(observed)
    report["observed"].append(observed)


def arm(context):
    """Create and verify the fixture before the ordinary Expert is started."""
    state, report = context.competition_static_state, context.competition_static_result
    if report["phase"] == "active":
        return
    if report["phase"] != "prepared":
        raise RuntimeError("Competition static cannot arm from " + report["phase"])
    process = getattr(getattr(context, "expert_controller", None), "process", None)
    if process is not None and process.poll() is None:
        raise RuntimeError("Competition static arm must precede the ordinary Expert launch")
    _require_pins(state)
    state.grpc = _fixture_client(context.grpc)
    report["fixture_client_key"] = state.grpc.client_key
    try:
        before = live._inventory(state.grpc, report)
        ego_id = str(context.grpc.get_ego().get_id())
        if ego_id not in before:
            ego_id = live._id(live._read_one(context.grpc, context.grpc.get_ego()))
        if ego_id not in before:
            raise RuntimeError("Competition static setup cannot identify the existing Ego")
        state.ego_id = ego_id
        state.baseline = _preserved_baseline(context, before, ego_id)
        live._occupancy(before, ego_id, set(), state.geometry)
        for index, placement in enumerate(state.candidate["obstacles"]):
            _spawn_one(context, placement, index)
        fresh = live._inventory(state.grpc, report)
        live._geometry_checks(report["observed"], live._observation(fresh[ego_id])["raw_bbox_normalized"], state.geometry)
        _require_pins(state)
        report.update(phase="active", fresh_placement_verified=True,
                      owned_actor_ids=sorted(state.references))
        _save(context)
        print("[CompetitionStatic] prepared selected=" + report["selected_candidate_sha256"])
    except BaseException as error:
        report.update(phase="failed", failure_reason=str(error) or type(error).__name__)
        try:
            cleanup(context)
        except Exception as cleanup_error:
            report["cleanup_error"] = str(cleanup_error)
        _save(context)
        raise


def update(context, elapsed, ego_state, current_s):
    """Observe the owned stationary scene without changing Ego or its driver."""
    state, report = context.competition_static_state, context.competition_static_result
    if report["phase"] != "active":
        return
    now = float(elapsed)
    if now - state.last_check < CHECK_PERIOD_SEC:
        return
    state.last_check = now
    try:
        fresh = live._inventory(state.grpc, report)
        for entry in state.owned:
            identity = entry["raw_id"]
            actor = live.validated_global_actor_state(state.grpc, entry, state.grpc.client_key, report=report)
            live._stationary(live._observation(actor), state.references[identity], "Competition static actor")
        if state.ego_id not in fresh:
            raise RuntimeError("Competition static monitoring lost the existing Ego")
        live._occupancy(fresh, state.ego_id, set(state.references), state.geometry)
        # arm() validates placement against the parked Ego. During driving the
        # owned bodies remain pinned by the identity/stationary checks above;
        # the inherited full-loop driver handles MORAI-reported collisions.
        # Reapplying parked-Ego clearance here would reject valid close passes.
        report["stationary_samples"] += 1
        report["last_observation_elapsed_sec"] = now
        if float(current_s) >= float(context.tunnel_exit_s) + RETIRE_AFTER_EXIT_M:
            report.update(tunnel_passed=True, tunnel_passed_elapsed_sec=now)
            cleanup(context)
        _save(context)
    except BaseException as error:
        report.update(phase="failed", failure_reason=str(error) or type(error).__name__)
        try:
            cleanup(context)
        except Exception as cleanup_error:
            report["cleanup_error"] = str(cleanup_error)
        _save(context)
        raise


def cleanup(context):
    """Delete only the fixture namespace, preserving unrelated actor identities."""
    state = getattr(context, "competition_static_state", None)
    if state is None:
        return
    report = context.competition_static_result
    if report["cleanup"]["verified"]:
        return
    if state.grpc is None:
        report["cleanup"].update(verified=True, reason="no_fixture_namespace_created")
    else:
        try:
            live.cleanup_owned_actors(state.grpc, state.owned, state.baseline, report)
        except Exception as error:
            report.update(phase="failed", failure_reason=report.get("failure_reason") or str(error))
            _save(context)
            raise
    try:
        _require_pins(state)
    except Exception as error:
        report.update(phase="failed", failure_reason=report.get("failure_reason") or str(error))
        _save(context)
        raise
    report["completed"] = bool(report["tunnel_passed"] and report["fresh_placement_verified"]
                               and not report.get("failure_reason"))
    if report["phase"] != "failed":
        report["phase"] = "completed" if report["completed"] else "cleaned"
    _save(context)
