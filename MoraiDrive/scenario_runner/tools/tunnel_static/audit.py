"""Enumerate and calibrate every native API model with parked Ego.

Offline enumeration never calls the simulator. --live attaches to the existing
world and measures one owned actor in an explicitly evidenced quarantine area.
It never starts Expert, starts/stops the simulation, or deletes foreign actors.
Collider/nominal metadata alone cannot approve an unknown visible footprint.
"""
import argparse
import copy
import fcntl
import hashlib
import json
import math
from pathlib import Path
import signal
import time
import uuid

from ...main_runner import PACKAGE_ROOT, load_configuration
from ...scenarios.full_loop_parts.tunnel_static_parts import catalog
from ...scenarios.full_loop_parts.tunnel_static_parts.live import (
    _inventory, _observation, _model, _read_one, _handle_for_request, _stationary,
)
from .models import _assert_controller_idle, _available_models, _BoundedStub
from .batch import atomic_json, fingerprint, digest, GLOBAL_EXECUTION_LOCK

SCHEMA = "aimdrive_tunnel_model_audit_v1"
DEFAULT_BASELINE = PACKAGE_ROOT/".runtime/tunnel_model_coverage/expanded_20261007_v3/manifest.json"


class AuditInfrastructureError(RuntimeError):
    """Unusable evidence infrastructure is not a model suitability result."""


class ModelMetadataUnavailable(AuditInfrastructureError):
    pass


class ActorBindingPending(AuditInfrastructureError):
    pass


def hold_owned_vehicle(handle, speed_command):
    """Use the actual SDK signature; frame=0 is the async ignored tick."""
    return bool(handle.set_vehicle_dynamics_speed_limit(True, 0., reset=False)
        and handle.set_velocity(0.) and handle.control(long_cmd_type=speed_command,
            throttle=0., brake=1., steer=0., velocity=0., acceleration=0., frame=0))


def note_metadata_failure(state):
    count=state.get("consecutive_numeric_model_metadata_failures",0)+1
    state["consecutive_numeric_model_metadata_failures"]=count
    if count>=3:
        raise AuditInfrastructureError("Three consecutive numeric model metadata queries failed; stop instead of labelling more models unsuitable")
    return count


def api_models(document):
    value = document.get("available_models", document)
    if not isinstance(value, dict) or any(not isinstance(value.get(kind), list) for kind in ("obstacle", "vehicle")):
        raise ValueError("API model list requires obstacle and vehicle arrays")
    result = {}
    for kind in ("obstacle", "vehicle"):
        names = value[kind]
        if any(not isinstance(name, str) or not name.strip() for name in names) or len(set(names)) != len(names):
            raise ValueError("Missing or duplicate API model names")
        result[kind] = sorted(names)
    return result


def baseline_models(document):
    rows = document.get("results", document.get("models", []))
    values = [(row.get("actor_kind"), row.get("model")) for row in rows]
    if (not values or len(set(values)) != len(values)
            or any(kind not in ("obstacle", "vehicle") or not isinstance(name, str) or not name for kind, name in values)):
        raise ValueError("Baseline requires distinct exact model identities")
    return set(values)


def reviewed_models():
    # The broader profiles are opt-in; the default sealed static footprint
    # remains the preferred baseline where its name already exists.
    statics = catalog.load_reviewed_static_models()
    families = {("obstacle", name): "static_unit_3d" for name in statics}
    legacy = catalog.load_static_models()
    statics.update(legacy)
    families.update({("obstacle", name): "legacy_static" for name in legacy})
    vehicles = catalog.load_vehicle_models()
    families.update({("vehicle", name): "vehicle_native_3d" for name in vehicles})
    return {"obstacle": statics, "vehicle": vehicles}, families


def enumerate_models(available, baseline, pools, families, *, include_baseline=False):
    if baseline-set((kind, name) for kind, names in available.items() for name in names):
        raise ValueError("Baseline contains models absent from the pinned API list")
    rows = []
    for kind in ("obstacle", "vehicle"):
        for name in available[kind]:
            existing = (kind, name) in baseline
            if existing and not include_baseline:
                continue
            profile = pools[kind].get(name)
            rows.append(dict(index=len(rows), actor_kind=kind, model=name, baseline=existing,
                scale=[1., 1., 1.], available_in_pinned_api=True, live_tested=False,
                spawnability="not_tested", classification="unclassified", status="pending",
                geometry_verified=profile is not None, reviewed_family=families.get((kind, name)),
                reviewed_footprint_sha256=fingerprint(profile) if profile else None,
                reason=None if profile else "No independently reviewed native-scale surface footprint; collider/spec alone is insufficient",
                roof=dict(physical_roof_verified=False, status="unverified_provisional")))
    return rows


def validate_quarantine(document, requested):
    if document.get("schema") != "aimdrive_tunnel_model_quarantine_region_v1":
        raise ValueError("Live model audit requires an evidenced quarantine region")
    xyz = catalog._vector(document.get("position_xyz"), 3)
    yaw = catalog._vector([document.get("spawn_yaw_deg")], 1)[0]
    radius = catalog._vector([document.get("clear_radius_m")], 1, positive=True)[0]
    if radius < 20.:
        raise ValueError("Quarantine clear radius must be at least 20m")
    proof = document.get("evidence", {})
    evidence = Path(proof["original_route_path"]).resolve()
    if (digest(evidence) != proof.get("sha256") or str(document.get("map_road_type")) == "2"
            or proof.get("pose_exactly_matches_route_xyz") is not True
            or math.dist(xyz[:2], proof["tunnel_mapped_entry_xyz"][:2]) < 100.):
        raise ValueError("Quarantine route evidence changed or is not outside the tunnel")
    route = [line.replace(",", " ").split() for line in evidence.read_text().splitlines() if line.strip()]
    if math.dist(xyz, [float(v) for v in route[int(document["original_route_index"])][:3]]) > 1e-6:
        raise ValueError("Quarantine point is not the pinned exact route position")
    approved = baseline_models({"models": document.get("permitted_models", [])})
    if any((row["actor_kind"], row["model"]) not in approved for row in requested):
        raise ValueError("Quarantine region does not explicitly cover every requested model")
    return dict(document, position_xyz=list(xyz), yaw_deg=yaw, clear_radius_m=radius)


def bbox_agrees(direct, global_row):
    if math.dist(direct["position_xyz"], global_row["position_xyz"]) > .01:
        return False
    for key in ("center_xyz", "extent_xyz"):
        if math.dist(direct["world_bbox"][key], global_row["world_bbox"][key]) > .01:
            return False
    return all(abs((a-b+180.) % 360.-180.) <= .05 for a, b in zip(
        direct["world_bbox"]["rotation_xyz_deg"], global_row["world_bbox"]["rotation_xyz_deg"]))


def resolve_audit_actor(grpc, direct_state, inventory, before_ids):
    """Bind a UUID query to one new global numeric actor before model RPCs.

    GetActorState may echo the request alias. GetOptionName/GetVehicleSpec use
    the numeric ActorInfo from the global inventory. Zero extents are accepted
    only for identity resolution; physical geometry is validated afterwards.
    """
    def xyz(value):
        return catalog._vector([getattr(value, key) for key in ("x", "y", "z")], 3)
    def signature(state):
        return (xyz(state.transform.location), xyz(state.bounding_box.location),
                xyz(state.bounding_box.extent), xyz(state.bounding_box.rotation))
    reference = signature(direct_state)
    if any(value<0. for value in reference[2]):
        raise RuntimeError("Negative native box extents cannot bind actor identity")
    matches = []
    for identity, state in inventory.items():
        if identity in before_ids or not str(identity).isdecimal():
            continue
        if int(state.actor_info.object_type) != int(direct_state.actor_info.object_type):
            continue
        current = signature(state)
        if (all(math.dist(a,b)<=.01 for a,b in zip(reference[:3],current[:3]))
                and all(abs((a-b+180.) % 360.-180.)<=.05 for a,b in zip(reference[3],current[3]))):
            matches.append(identity)
    if not matches:
        raise ActorBindingPending("Owned vehicle pose and global snapshot have not yet converged")
    if len(matches)!=1:
        raise AuditInfrastructureError("Owned request cannot be uniquely bound to a new global numeric actor")
    identity = matches[0]
    try:
        observed_model = _model(grpc, inventory[identity])
    except Exception as error:
        raise ModelMetadataUnavailable("Resolved numeric actor model evidence is unavailable: "+str(identity)) from error
    return identity, observed_model


def measured_footprint(reviewed, observed, commanded_yaw, vertical):
    """Enclose the reviewed box and measured collider in the command frame."""
    return catalog.calibrate_reviewed_footprint(reviewed, observed, commanded_yaw, vertical)


def bind_fresh_actor(grpc, handle, before_ids, evidence, persist, stopped, *, timeout=5.):
    """Retry native suspension snapshots, preserving strict unique binding."""
    from google.protobuf.json_format import MessageToDict
    deadline=time.monotonic()+timeout
    attempts=[]
    while True:
        if stopped(): raise KeyboardInterrupt()
        _assert_controller_idle()
        direct=_read_one(grpc,handle)
        fresh=_inventory(grpc,{"inventory_checks":0})
        attempts.append(dict(receipt_monotonic=time.monotonic(),
            direct_raw_actor_state=MessageToDict(direct,preserving_proto_field_name=True),
            new_global_actor_candidates={identity:MessageToDict(state,preserving_proto_field_name=True)
                for identity,state in fresh.items() if identity not in before_ids}))
        evidence["binding_snapshots"]=attempts
        evidence["direct_raw_actor_state"]=attempts[-1]["direct_raw_actor_state"]
        persist()
        try: identity,name=resolve_audit_actor(grpc,direct,fresh,before_ids)
        except ActorBindingPending:
            if time.monotonic()>=deadline:
                raise AuditInfrastructureError("Bounded fresh direct/global native actor binding did not converge")
            time.sleep(.1)
            continue
        return direct,fresh,identity,name


def attach(global_cfg, client_key):
    from ...adapters.morai_grpc import MoraiGrpcClient
    cfg = copy.deepcopy(global_cfg); cfg["grpc"]["client_key"] = client_key
    grpc = MoraiGrpcClient(cfg)
    # The constructor installs the SDK's api/proto import paths. A clean
    # process cannot import those modules before this bootstrap occurs.
    from api.simulation_world import SimulationWorld
    from proto.morai.simulation.simulation_enum_pb2 import SYNC_MODE_TYPE_UNSPECIFIED
    grpc.connect()
    adapter = grpc.client._sim_adapter
    adapter._actor_stub = _BoundedStub(adapter._actor_stub)
    adapter._simulator_stub = _BoundedStub(adapter._simulator_stub)
    world = SimulationWorld.__new__(SimulationWorld)
    world._sim_adapter, world._client_key, world._actors = adapter, client_key, {}
    world._current_sync_mode = SYNC_MODE_TYPE_UNSPECIFIED
    world._ego_id = world._ego_label = "Ego"
    grpc.world = world
    return grpc


def baseline_identities(grpc, inventory):
    """Freeze exact ActorInfo/model identity, preserving foreign dynamics."""
    return {identity: {"actor_info_wire_hex": state.actor_info.SerializeToString(deterministic=True).hex(),
                       "model": _model(grpc, state)} for identity, state in inventory.items()}


def assert_baseline_identities(before, after):
    if set(before)!=set(after):
        raise RuntimeError("Audit global baseline actor IDs changed")
    if any(before[identity]!=after[identity] for identity in before):
        raise RuntimeError("Audit baseline ActorInfo or model identity was rebound")


def cleanup_owned(grpc, entry, before, identities):
    """Delete only the journaled request in this audit's owned namespace."""
    from proto.morai.common.enum_pb2 import STATUS_CODE_SUCCESS
    info = entry["handle"].get_object_info()
    if info.client_key != grpc.client_key or str(info.id.value) != entry["request_id"]:
        raise RuntimeError("Audit cleanup ownership identity changed")
    result = grpc.world._sim_adapter._actor_stub.DestroyActor(info, timeout=1.)
    after = _inventory(grpc, {"inventory_checks": 0})
    if set(after) != set(before):
        raise RuntimeError("Audit cleanup did not restore the exact global baseline")
    if result.status != STATUS_CODE_SUCCESS:
        # A rejected spawn may never have created an actor. Require both a
        # complete unchanged inventory and an explicit empty/NOT_FOUND owned
        # lookup; an unavailable/deadline RPC is never evidence of absence.
        try:
            state = grpc.world._sim_adapter._actor_stub.GetActorState(info, timeout=1.)
        except Exception as error:
            import grpc as grpc_module
            if not isinstance(error, grpc_module.RpcError) or error.code() != grpc_module.StatusCode.NOT_FOUND:
                raise RuntimeError("Owned audit cleanup absence cannot be proven") from error
        else:
            if str(state.actor_info.id.value):
                raise RuntimeError("Owned audit actor destroy rejected and its lookup remains present")
    assert_baseline_identities(identities, baseline_identities(grpc, after))
    if "Ego" not in before or "Ego" not in after:
        raise RuntimeError("Audit cleanup lost the known Ego")
    _stationary(_observation(after["Ego"]), _observation(before["Ego"]), "Ego baseline")
    return True


def validate_journal_owner(journal, client_key):
    if (not isinstance(client_key, str) or not client_key.startswith("tunnel_model_audit_")
            or journal.get("client_key") != client_key
            or not str(journal.get("request_id", "")).startswith("tunnel_audit_")
            or journal.get("actor_kind") not in ("obstacle", "vehicle")):
        raise RuntimeError("Interrupted audit request ownership is uncertain")


def recover_journal(grpc, journal):
    """Resolve interrupted cleanup using only the saved owned request ID."""
    from proto.morai.common.enum_pb2 import STATUS_CODE_SUCCESS
    validate_journal_owner(journal, grpc.client_key)
    handle = _handle_for_request(grpc, journal["actor_kind"], journal["request_id"])
    before_ids = set(journal["before_actor_ids"])
    current = _inventory(grpc, {"inventory_checks": 0})
    if set(current) != before_ids:
        result = grpc.world._sim_adapter._actor_stub.DestroyActor(handle.get_object_info(), timeout=1.)
        if result.status != STATUS_CODE_SUCCESS:
            raise RuntimeError("Interrupted owned audit cleanup was rejected")
        current = _inventory(grpc, {"inventory_checks": 0})
    if set(current) != before_ids:
        raise RuntimeError("Interrupted audit cleanup did not restore its exact global baseline")
    assert_baseline_identities(journal["baseline_identities"], baseline_identities(grpc, current))
    _stationary(_observation(current["Ego"]), journal["baseline_ego_observation"], "Resumed Ego baseline")
    return True


def measure_one(grpc, row, reviewed, region, directory, save_journal, stop_requested, ros=None):
    from ...scenarios.full_loop_parts.tunnel_static_parts.safety import park
    from proto.morai.actor.actor_enum_pb2 import LONG_CMD_TYPE_SPEED
    from proto.morai.common.type_pb2 import Vector3
    stub = grpc.world._sim_adapter._actor_stub
    _assert_controller_idle(); park(stub, grpc.get_ego())
    before = _inventory(grpc, {"inventory_checks": 0})
    identities = baseline_identities(grpc, before)
    for state in before.values():
        observation = _observation(state)
        extent = observation["world_bbox"]["extent_xyz"]
        distance = math.dist(observation["world_bbox"]["center_xyz"][:2], region["position_xyz"][:2])
        required = (float(region.get("ego_minimum_distance_m", 50.)) if observation["raw_actor_id"] == "Ego"
                    else region["clear_radius_m"]+float(region.get("foreign_actor_padding_m", 5.))+math.hypot(extent[0], extent[1]))
        if distance < required:
            raise RuntimeError("Quarantine contains an existing actor")
    request_id = "tunnel_audit_"+uuid.uuid4().hex
    entry = dict(request_id=request_id, handle=_handle_for_request(grpc, row["actor_kind"], request_id), raw_id=None)
    journal = dict(request_id=request_id, actor_kind=row["actor_kind"], model=row["model"],
                   before_actor_ids=sorted(before), client_key=grpc.client_key,
                   baseline_identities=identities, baseline_ego_observation=_observation(before["Ego"]))
    save_journal(journal)
    evidence = dict(schema="aimdrive_tunnel_model_measurement_v1", actor_kind=row["actor_kind"], model=row["model"],
        client_key=grpc.client_key, request_id=request_id, scale=[1., 1., 1.], samples=[],
        commanded_spawn_yaw_deg=region["yaw_deg"],
        reviewed_footprint_sha256=row["reviewed_footprint_sha256"], cleanup_verified=False,
        native_scale_contract_verified=True, physical_roof_verified=False,
        native_scale_contract="explicit_obstacle_unit_scale" if row["actor_kind"] == "obstacle" else "native_vehicle_spawn_without_scale")
    path = directory/"measurement.json"
    try:
        transform = grpc.make_transform(*region["position_xyz"], region["yaw_deg"])
        kwargs = dict(transform=transform, model_name=row["model"], label=request_id, request_id=request_id)
        if row["actor_kind"] == "vehicle":
            handle = grpc.world.spawn_vehicle(**kwargs, velocity=0., pause=False, multi_ego=False)
        else:
            handle = grpc.world.spawn_obstacle(**kwargs, scale=Vector3(x=1., y=1., z=1.))
        if handle is not None:
            entry["handle"] = handle
        if row["actor_kind"]=="vehicle" and not hold_owned_vehicle(entry["handle"],LONG_CMD_TYPE_SPEED):
            raise RuntimeError("Owned NPC stationary controls rejected")
        from google.protobuf.json_format import MessageToDict
        direct_state,fresh,raw_id,observed_name=bind_fresh_actor(grpc,entry["handle"],set(before),evidence,
            lambda:atomic_json(path,evidence),stop_requested)
        expected_type = int(entry["handle"].get_object_info().object_type)
        evidence["requested_object_type"] = expected_type
        if int(direct_state.actor_info.object_type) != expected_type:
            evidence["classification"] = "native_actor_type_mismatch"
            raise RuntimeError("Native actor type does not match requested API category")
        entry["raw_id"] = raw_id; journal["raw_actor_id"] = raw_id; save_journal(journal)
        evidence.update(raw_actor_id=raw_id, direct_reported_identifier=str(direct_state.actor_info.id.value),
            global_option_name=observed_name,
            model_query_identifier=raw_id, identifier_resolution_basis="unique new numeric global actor matched to owned UUID direct pose and complete raw box")
        evidence["global_raw_actor_state"] = MessageToDict(fresh[raw_id], preserving_proto_field_name=True)
        atomic_json(path, evidence)
        if observed_name != row["model"]:
            evidence["classification"] = "selector_or_alias_identity_mismatch"
            raise RuntimeError("Requested model resolved to a different native model: "+observed_name)
        evidence["native_spawn_identity_verified"] = True
        if int(fresh[raw_id].actor_info.object_type) != expected_type:
            raise RuntimeError("Global native actor type does not match requested API category")
        handle = entry["handle"]
        if row["actor_kind"] == "obstacle":
            if not handle.set_physics(False) or not handle.set_transform(transform):
                raise RuntimeError("Owned static transform lock rejected")
        else:
            spec = stub.GetVehicleSpec(fresh[raw_id].actor_info, timeout=1.)
            evidence["vehicle_spec_query_identifier"] = raw_id
            evidence["vehicle_spec"] = dict(length_m=float(spec.length), width_m=float(spec.width), height_m=float(spec.height))
            evidence["vehicle_spec_raw"] = MessageToDict(spec, preserving_proto_field_name=True)
            try: catalog._vector(list(evidence["vehicle_spec"].values()), 3, positive=True)
            except ValueError:
                evidence["classification"] = "nonpositive_or_invalid_native_vehicle_spec"
                raise
        settle_deadline, previous_stable = time.monotonic()+5., None
        while True:
            _assert_controller_idle(); park(stub, grpc.get_ego())
            state = _inventory(grpc, {"inventory_checks": 0})[raw_id]
            observed = _observation(state)
            try:
                _stationary(observed, previous_stable, "Owned model settling")
                if previous_stable is not None: break
                previous_stable = observed
            except RuntimeError:
                previous_stable = None
            if time.monotonic()>=settle_deadline:
                raise RuntimeError("Owned model did not settle within five seconds")
            if stop_requested(): raise KeyboardInterrupt()
            time.sleep(.2)
        started, reference = time.monotonic(), None
        while True:
            if time.monotonic()-started > 15.:
                raise RuntimeError("Bounded stationary measurement deadline exceeded")
            if stop_requested():
                raise KeyboardInterrupt()
            _assert_controller_idle(); park(stub, grpc.get_ego())
            direct_state = _read_one(grpc, handle)
            direct = _observation(direct_state)
            direct_reported_identifier = str(direct_state.actor_info.id.value)
            direct["raw_actor_id"] = raw_id
            direct["query_reported_identifier"] = direct_reported_identifier
            inventory = _inventory(grpc, {"inventory_checks": 0})
            if raw_id not in inventory or set(inventory) != set(before)|{raw_id}:
                raise RuntimeError("Global actor ownership baseline changed during measurement")
            global_row = _observation(inventory[raw_id])
            if _model(grpc, inventory[raw_id]) != row["model"] or not bbox_agrees(direct, global_row):
                raise RuntimeError("Model identity or direct/global bounding box disagrees")
            _stationary(global_row, reference, "Owned model")
            reference = reference or global_row
            evidence["samples"].append(dict(receipt_monotonic=time.monotonic(), direct=direct, global_actor=global_row,
                option_name=row["model"], direct_raw_actor_state=MessageToDict(direct_state, preserving_proto_field_name=True),
                direct_reported_identifier=direct_reported_identifier,
                global_raw_actor_state=MessageToDict(inventory[raw_id], preserving_proto_field_name=True)))
            atomic_json(path, evidence)
            if evidence["samples"][-1]["receipt_monotonic"]-evidence["samples"][0]["receipt_monotonic"] >= 5.:
                break
            time.sleep(.2)
        evidence.update(stationary_seconds=evidence["samples"][-1]["receipt_monotonic"]-evidence["samples"][0]["receipt_monotonic"],
                        direct_global_bbox_agreement=True)
        if ros is not None:
            snapshots = ros.object_snapshots(started)
            evidence["native_ros_frames"] = [{**{key: frame[key] for key in (
                "receipt_monotonic", "source_sequence", "source_stamp_ns", "publisher_node")}, "rows": [item for item in frame["rows"]
                if item["raw_actor_id"] == raw_id and item["actor_kind"] == row["actor_kind"]]} for frame in snapshots]
        vertical = catalog.reviewed_vertical_bounds(row["actor_kind"], row["model"]) if reviewed is not None else None
        if vertical is not None:
            top = max(vertical["top_offset_actor_m"], reference["world_bbox"]["center_xyz"][2]-reference["position_xyz"][2]
                      +reference["world_bbox"]["extent_xyz"][2])
            # Reserve for level-road RPY variation using the reviewed full box.
            top += .5*math.hypot(vertical["size_lwh_m"][0], vertical["size_lwh_m"][1])*math.sin(math.radians(3.))
            top += max(0., reference["position_xyz"][2]-region["position_xyz"][2])
            evidence["roof"] = dict(physical_roof_verified=False, policy="unverified_ceiling_policy",
                provisional_model_top_cap_m=4., top_above_map_floor_m=top,
                provisional_top_cap_passed=top<=4., reviewed_vertical_bounds=vertical)
            if top <= 4.:
                evidence["footprint"] = measured_footprint(reviewed, reference, region["yaw_deg"], vertical)
        return evidence
    finally:
        try:
            park(stub, grpc.get_ego())
            evidence["cleanup_verified"] = cleanup_owned(grpc, entry, before, identities)
            save_journal(None)
        finally:
            atomic_json(path, evidence)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--available-json", type=Path, default=DEFAULT_BASELINE)
    p.add_argument("--baseline-json", type=Path, default=DEFAULT_BASELINE)
    p.add_argument("--include-baseline", action="store_true")
    p.add_argument("--model", action="append")
    p.add_argument("--live", action="store_true")
    p.add_argument("--quarantine-region", type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--resume", action="store_true")
    return p


def run(args):
    available = api_models(json.loads(args.available_json.read_bytes()))
    baseline = baseline_models(json.loads(args.baseline_json.read_bytes()))
    pools, families = reviewed_models()
    rows = enumerate_models(available, baseline, pools, families, include_baseline=args.include_baseline)
    if args.model:
        wanted = set(args.model); rows = [row for row in rows if row["model"] in wanted]
        if wanted-{row["model"] for row in rows}:
            raise ValueError("Requested model absent from this exact audit scope")
    region = validate_quarantine(json.loads(args.quarantine_region.read_bytes()), rows) if args.live and args.quarantine_region else None
    if args.live and region is None:
        raise ValueError("--live requires --quarantine-region before any SDK attachment")
    contract = dict(available_sha256=digest(args.available_json), baseline_sha256=digest(args.baseline_json),
        include_baseline=args.include_baseline, live_mode=args.live,
        requested_models=[[row["actor_kind"], row["model"]] for row in rows],
        source_sha256=digest(__file__), catalog_source_sha256=digest(catalog.__file__),
        reviewed_models_sha256=fingerprint(pools), quarantine_sha256=digest(args.quarantine_region) if region else None)
    output = args.output.resolve(); manifest_path = output/"manifest.json"
    if args.resume:
        state = json.loads(manifest_path.read_bytes())
        if state.get("contract") != contract:
            raise ValueError("Frozen audit inputs changed")
    else:
        output.mkdir(parents=True, exist_ok=False)
        state = dict(schema=SCHEMA, status="started", contract=contract, available_models=available,
            baseline_counts={kind: sum(row[0]==kind for row in baseline) for kind in available}, results=rows,
            owned_client_key="tunnel_model_audit_"+uuid.uuid4().hex, inflight=None,
            expert_started=False, ego_driving_commanded=False, physical_roof_verified=False)
    runtime = dict(schema=catalog.RUNTIME_CATALOG_SCHEMA, scale_xyz=[1., 1., 1.], models=[], physical_roof_verified=False)
    old = output/"runtime_catalog.json"
    if args.resume and old.is_file():
        runtime = json.loads(old.read_bytes())
    stop = [False]
    def interrupted(signum, frame): stop[0] = True
    previous = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGINT, signal.SIGTERM)}
    grpc, ros, execution_lock = None, None, None
    def save_journal(entry): state["inflight"] = entry; atomic_json(manifest_path, state)
    try:
        atomic_json(manifest_path, state)
        if args.live:
            GLOBAL_EXECUTION_LOCK.parent.mkdir(parents=True, exist_ok=True)
            execution_lock = GLOBAL_EXECUTION_LOCK.open("a+")
            try: fcntl.flock(execution_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError("Another live tunnel campaign owns the global MORAI execution lock") from error
            _assert_controller_idle(); global_cfg, _, _ = load_configuration()
            grpc = attach(global_cfg, state["owned_client_key"])
            from ...scenarios.full_loop_parts.tunnel_static_parts.safety import park
            park(grpc.world._sim_adapter._actor_stub, grpc.get_ego())
            if state.get("inflight") is not None:
                recover_journal(grpc, state["inflight"])
                save_journal(None)
            if {key: sorted(value) for key, value in _available_models(grpc).items()} != available:
                raise RuntimeError("Current API model availability changed from pinned input")
            from ...scenarios.segment_modes.tunnel_static_trial import TrialRosEvidence
            import rospy
            from morai_msgs.msg import ObjectStatusList
            if not rospy.core.is_initialized(): rospy.init_node("tunnel_model_audit", anonymous=True, disable_signals=True)
            ros = TrialRosEvidence(); ros.subscribers = [rospy.Subscriber("/Object_topic", ObjectStatusList, ros.on_objects, queue_size=1)]
        for row in state["results"]:
            if row["status"] != "pending": continue
            if stop[0]: raise KeyboardInterrupt()
            if (digest(__file__) != contract["source_sha256"] or digest(catalog.__file__) != contract["catalog_source_sha256"]
                    or fingerprint(reviewed_models()[0]) != contract["reviewed_models_sha256"]
                    or (region and digest(args.quarantine_region) != contract["quarantine_sha256"])):
                raise RuntimeError("Frozen audit source, reviewed geometry or quarantine changed during the campaign")
            if not args.live:
                row.update(status="reviewed_geometry_available" if row["geometry_verified"] else "missing_geometry_needs_surface_review")
            else:
                row["live_tested"] = True; atomic_json(manifest_path, state)
                directory = output/("model_%03d_%s" % (row["index"], hashlib.sha256(row["model"].encode()).hexdigest()[:12]))
                try:
                    evidence = measure_one(grpc, row, pools[row["actor_kind"]].get(row["model"]), region,
                                           directory, save_journal, lambda: stop[0], ros)
                except Exception as error:
                    row.update(status="measurement_failed", spawnability="failed_or_unresolved", failure_reason=str(error))
                    proof_path = directory/"measurement.json"
                    if proof_path.is_file():
                        failed = json.loads(proof_path.read_bytes())
                        row.update(measurement_path=str(proof_path.resolve()), measurement_sha256=digest(proof_path))
                        if failed.get("native_spawn_identity_verified") is True:
                            row["spawnability"] = "exact_identity_spawn_verified"
                        if failed.get("classification") == "selector_or_alias_identity_mismatch":
                            row.update(status="unsupported_selector_or_alias", classification=failed["classification"],
                                spawnability="different_model_identity_observed", actual_option_name=failed.get("global_option_name"))
                        elif "Invalid world bounding box extent" in str(error):
                            row.update(status="unsupported_nonpositive_native_bbox", classification="nonpositive_bbox_observed",
                                spawnability="native_identity_verified_without_positive_bbox" if failed.get("global_option_name")==row["model"]
                                else "failed_or_unresolved")
                        elif failed.get("classification") == "native_actor_type_mismatch":
                            row.update(status="unsupported_native_actor_type", classification=failed["classification"])
                        elif failed.get("classification") == "nonpositive_or_invalid_native_vehicle_spec":
                            row.update(status="unsupported_native_vehicle_spec", classification=failed["classification"])
                    atomic_json(manifest_path, state)
                    if isinstance(error, ModelMetadataUnavailable):
                        row.update(status="numeric_model_metadata_unavailable", model_suitability_determined=False)
                        note_metadata_failure(state)
                        atomic_json(manifest_path,state)
                    elif proof_path.is_file() and json.loads(proof_path.read_bytes()).get("global_option_name"):
                        state["consecutive_numeric_model_metadata_failures"] = 0
                    transport_failure=any(token in (str(error)+" "+str(error.__cause__ or "")) for token in (
                        "StatusCode.UNAVAILABLE", "StatusCode.DEADLINE_EXCEEDED", "failed to connect", "Connection refused"))
                    if transport_failure or isinstance(error,(TypeError,AttributeError)) or (
                            isinstance(error, AuditInfrastructureError) and not isinstance(error,ModelMetadataUnavailable)):
                        row.update(status="audit_infrastructure_failure", model_suitability_determined=False)
                        atomic_json(manifest_path, state)
                        raise
                    if state["inflight"] is not None: raise
                else:
                    state["consecutive_numeric_model_metadata_failures"] = 0
                    row.update(spawnability="exact_identity_spawn_verified", classification="physical_bbox_positive",
                        status="measured_reviewed_geometry" if "footprint" in evidence else (
                            "provisional_height_policy_excluded" if evidence.get("roof", {}).get("provisional_top_cap_passed") is False
                            else "missing_geometry_needs_surface_review"),
                        geometry_verified="footprint" in evidence, measurement_path=str((directory/"measurement.json").resolve()),
                        measurement_sha256=digest(directory/"measurement.json"))
                    if "footprint" in evidence:
                        row["footprint"] = evidence["footprint"]
                        runtime["models"].append(copy.deepcopy(row))
            atomic_json(manifest_path, state); atomic_json(output/"runtime_catalog.json", runtime)
            print(json.dumps({key: row.get(key) for key in ("index", "actor_kind", "model", "status", "spawnability")}), flush=True)
        state["status"] = "completed_live" if args.live else "completed_offline"
    except BaseException as error:
        state.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", failure_reason=str(error) or type(error).__name__)
        raise
    finally:
        if ros is not None:
            try: ros.close()
            except Exception as error: state["subscriber_shutdown_failure"] = str(error)
        if grpc is not None:
            try:
                from ...scenarios.full_loop_parts.tunnel_static_parts.safety import park
                state["final_parking"] = park(grpc.world._sim_adapter._actor_stub, grpc.get_ego())
            except Exception as error:
                state.update(status="failed", final_parking_failure=str(error))
            finally:
                try: grpc.client.disconnect()
                except Exception as error: state["disconnect_failure"] = str(error)
        if execution_lock is not None:
            fcntl.flock(execution_lock.fileno(), fcntl.LOCK_UN)
            execution_lock.close()
        state["status_counts"] = {status: sum(row["status"]==status for row in state["results"])
                                  for status in sorted({row["status"] for row in state["results"]})}
        atomic_json(manifest_path, state); atomic_json(output/"runtime_catalog.json", runtime)
        for sig, handler in previous.items(): signal.signal(sig, handler)
    return state


def main():
    parser = build_parser(); args = parser.parse_args()
    try: run(args)
    except (ValueError, RuntimeError, OSError) as error: parser.exit(2, str(error)+"\n")
    return 0


if __name__ == "__main__": raise SystemExit(main())
