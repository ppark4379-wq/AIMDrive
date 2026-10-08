"""Bounded parking, owned-fixture cleanup and reversible scene relocation.

This helper attaches to the existing world without constructing SimulationWorld.
It never clears the global actor namespace or terminates an unrelated Expert.
"""

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import time
from types import SimpleNamespace

from ....main_runner import load_configuration
from .connection import _attach
from .connection import _controller_publishers
from .live import _observation, MAX_SPEED_RAW


OWNED_KEYS = ("tunnel_static_trial_fixture", "tunnel_static_trial_runner")
CONTROL_SCOPE = "no_competing_controllers; exact_active_owned_brake_holder_allowed"


def competing_controller_publishers():
    from ...segment_modes import tunnel_static_startup as startup
    return startup.competing_controller_publishers()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def actor_ids(value):
    items = value.split(",") if value else []
    if any(not item.isdecimal() for item in items) or len(items) != len(set(items)):
        raise ValueError("Relocation requires distinct numeric non-Ego actor IDs")
    return items


def expert_identity(pid, batch_root, proc_root=Path("/proc")):
    """Bind the exact entrypoint/configuration to a Linux process birth token."""
    entrypoint = Path(__file__).resolve().parents[3] / "adapters/expert_process.py"
    batch_root = Path(batch_root).resolve()
    process = proc_root/str(pid)
    try:
        arguments = process.joinpath("cmdline").read_bytes().split(b"\0")
        if os.fsencode(str(entrypoint)) not in arguments:
            return None
        environment = dict(item.split(b"=", 1) for item in process.joinpath("environ").read_bytes().split(b"\0") if b"=" in item)
        configuration = Path(os.fsdecode(environment.get(b"GT_BEV_CONFIG_PATH", b""))).resolve()
        configuration.relative_to(batch_root)
        data = json.loads(configuration.read_text())
        if data.get("planning", {}).get("jamming_trial_contract", {}).get("mode") != "full_loop_tunnel_static":
            return None
        fields = process.joinpath("stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] == "Z":
            return None
        return {"pid": pid, "process_group": int(fields[2]), "starttime_ticks": int(fields[19]),
                "configuration": str(configuration)}
    except (OSError, ValueError, KeyError, IndexError):
        return None


def owned_experts(batch_root, proc_root=Path("/proc")):
    """Identify subprocesses by exact entrypoint, private config and birth token."""
    found = []
    for process in proc_root.iterdir():
        if not process.name.isdecimal():
            continue
        if expert_identity(int(process.name), batch_root, proc_root) is not None:
            found.append(int(process.name))
    return found


def stop_owned_experts(batch_root):
    from ....adapters.expert_controller import ExpertController
    stopped = []
    for pid in owned_experts(batch_root):
        original = expert_identity(pid, batch_root)
        if original is None:
            continue
        try:
            group = original["process_group"]
            if group != pid:
                raise RuntimeError("Trial Expert does not own its process group")
            if expert_identity(pid, batch_root) == original:
                os.killpg(group, signal.SIGTERM)
            deadline = time.monotonic() + 10.
            while time.monotonic() < deadline and expert_identity(pid, batch_root) == original:
                time.sleep(.1)
            if expert_identity(pid, batch_root) == original:
                os.killpg(group, signal.SIGKILL)
                deadline = time.monotonic() + 5.
                while time.monotonic() < deadline and expert_identity(pid, batch_root) == original:
                    time.sleep(.1)
            if expert_identity(pid, batch_root) == original:
                raise RuntimeError("Owned Expert is still alive after bounded termination")
        except ProcessLookupError:
            pass
        controller = object.__new__(ExpertController)
        controller._clear_owned_ros_registrations(pid)
        stopped.append(pid)
    return stopped


def inventory(stub, client_key=""):
    from proto.morai.actor.actor_get_pb2 import GetAllActorsFilter
    return {state.actor_info.id.value: state for state in stub.GetAllActorsState(
        GetAllActorsFilter(client_key=client_key, vehicle=True, pedestrian=True, obstacle=True), timeout=1.).states}


def _parking_adapter(stub, ego):
    return SimpleNamespace(get_ego=lambda: ego, _tunnel_static_common_start=True,
        world=SimpleNamespace(_sim_adapter=SimpleNamespace(_actor_stub=stub)))


def park(stub, ego):
    from ...segment_modes import tunnel_static_startup as startup
    grpc = _parking_adapter(stub, ego)
    started = time.monotonic()
    startup.hold_common_brake(grpc)
    deadline = time.monotonic() + 3.
    last_error = None
    while True:
        try:
            state, observed = startup.common_stopped_readback(grpc)
        except RuntimeError as error:
            last_error = error
            if time.monotonic() >= deadline:
                raise RuntimeError("Ego brake-hold readback did not settle within three seconds: " + str(last_error))
            time.sleep(.1)
            continue
        observed_publishers = _controller_publishers()
        if competing_controller_publishers():
            raise RuntimeError("Control publisher appeared during Ego brake-hold verification")
        return {"parked": True, "external_input_disabled": False,
                "brake_hold_verified": True, "control_publishers_absent": True,
                "control_scope": CONTROL_SCOPE, "observed_control_publishers": observed_publishers,
                "parking_policy": "verified_common_brake_hold",
                "speed_norm_raw": observed["speed_norm_raw"], "gear_mode": int(state.vehicle_state.gear_mode),
                "control_mode": observed["control_mode"], "brake": float(state.vehicle_state.brake),
                "throttle": float(state.vehicle_state.throttle), "position_xyz": observed["position_xyz"],
                "elapsed_sec": time.monotonic()-started}


def recover(grpc, ego, batch_root):
    from proto.morai.common.type_pb2 import StringValue
    from proto.morai.common.enum_pb2 import STATUS_CODE_SUCCESS
    from ...segment_modes import tunnel_static_startup as startup
    stub = grpc.world._sim_adapter._actor_stub
    # Block input before terminating only birth-token-verified owned processes.
    # AUTO brake hold is admitted only after competing controllers disappeared.
    startup.block_common_input(_parking_adapter(stub, ego))
    stopped = stop_owned_experts(batch_root)
    if competing_controller_publishers():
        raise RuntimeError("A competing or unowned /ctrl_cmd publisher remains; batch cannot resume")
    report = {"initial_parking": park(stub, ego), "stopped_expert_pids": stopped}
    before = inventory(stub)
    baseline_path = Path(batch_root) / "foreign_actor_inventory.json"
    baseline = (set(json.loads(baseline_path.read_text())["actor_ids"]) if baseline_path.exists()
                else set(before) - {"Ego"})
    destroyed = {}
    for key in OWNED_KEYS:
        ids = set(before) - baseline - {"Ego"}
        response = stub.DestroyAllActors(StringValue(value=key), timeout=1.)
        if response.status != STATUS_CODE_SUCCESS:
            raise RuntimeError("Trial-owned actor cleanup rejected for " + key)
        destroyed[key] = sorted(ids)
    after = inventory(stub)
    # MORAI 26.R1.H3 ignores GetAllActorsFilter.client_key and returns global
    # IDs without owner keys. Use the frozen foreign baseline plus global
    # absence, rather than pretending that this read API proves ownership.
    if not baseline <= set(after):
        raise RuntimeError("Foreign actor disappeared during owned-fixture cleanup")
    if set(after) - baseline - {"Ego"}:
        raise RuntimeError("Unaccounted actors remain after namespaced cleanup; batch cannot resume")
    if not baseline_path.exists():
        write_json(baseline_path, {"actor_ids": sorted(baseline), "created_unix": time.time(),
                                  "read_filter_is_global": True})
    report.update(destroyed_owned_actors=destroyed, removed_actor_ids=sorted(set(before)-set(after)),
                  preserved_foreign_actor_ids=sorted(baseline), read_filter_is_global=True,
                  final_parking=park(stub, ego), observed_control_publishers=_controller_publishers(),
                  controller_publishers=competing_controller_publishers(), control_scope=CONTROL_SCOPE, status="recovered")
    if report["controller_publishers"]:
        raise RuntimeError("Control publisher appeared during recovery")
    return report


def set_transform(stub, info, target, *, restoring):
    from proto.morai.actor.actor_set_pb2 import SetTransformParam
    from proto.morai.common.enum_pb2 import STATUS_CODE_SUCCESS
    request = SetTransformParam(actor_info=info, transform=target)
    for attempt in range(6):
        response = stub.SetTransform(request, timeout=1.)
        if response.status != STATUS_CODE_SUCCESS:
            raise RuntimeError("Relocated actor transform command was rejected")
        time.sleep(.1)
        state = stub.GetActorState(info, timeout=1.)
        delta = [getattr(target.location, k)-getattr(state.transform.location, k) for k in "xyz"]
        angles = [(getattr(target.rotation, k)-getattr(state.transform.rotation, k)+180)%360-180 for k in "xyz"]
        if ((restoring and math.sqrt(sum(d*d for d in delta)) < .001 and max(map(abs, angles)) < .01)
                or (not restoring and math.hypot(*delta[:2]) < .01)):
            return {"position_error_m": math.sqrt(sum(d*d for d in delta)), "max_angle_error_deg": max(map(abs, angles)), "attempts": attempt+1}
        for k, displacement, angle in zip("xyz", delta, angles):
            setattr(request.transform.location, k, getattr(request.transform.location, k)+displacement)
            setattr(request.transform.rotation, k, getattr(request.transform.rotation, k)+angle)
    raise RuntimeError("Relocated actor transform readback did not converge")


def relocate(grpc, ego, output, ids, action):
    from proto.morai.actor.actor_get_pb2 import ActorState
    from google.protobuf.json_format import MessageToDict
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    snapshot_path = output / "scene_snapshot.json"
    stub = grpc.world._sim_adapter._actor_stub
    parking = park(stub, ego)
    if competing_controller_publishers():
        raise RuntimeError("Stop the Ego controller before moving or restoring the scene")
    rows = inventory(stub)
    if not snapshot_path.exists():
        if action == "restore":
            return {"status": "nothing_to_restore", "parking": parking}
        if not ids:
            return {"status": "nothing_to_move", "parking": parking}
        snapshot = {"schema": "aimdrive_tunnel_batch_scene_snapshot_v1", "actor_ids": ids, "actors": []}
        for identity in ids:
            state = rows[identity]
            if state.actor_info.client_key in OWNED_KEYS:
                raise RuntimeError("Relocation is only for explicitly selected foreign fixtures")
            model = stub.GetOptionName(state.actor_info, timeout=1.).value
            payload = state.SerializeToString()
            backup = output / (identity+".pb")
            backup.write_bytes(payload)
            snapshot["actors"].append({"id": identity, "model": model, "state_file": backup.name,
                                      "sha256": hashlib.sha256(payload).hexdigest(),
                                      "state": MessageToDict(state, preserving_proto_field_name=True)})
        # Commit all original states before changing the first actor.
        write_json(snapshot_path, snapshot)
    snapshot = json.loads(snapshot_path.read_text())
    if ids and ids != snapshot["actor_ids"]:
        raise RuntimeError("Relocation IDs differ from the durable original scene snapshot")
    evidence, errors = [], []
    for index, row in enumerate(snapshot["actors"]):
        try:
            identity = row["id"]
            payload = (output/row["state_file"]).read_bytes()
            if hashlib.sha256(payload).hexdigest() != row["sha256"]:
                raise RuntimeError("Original scene snapshot checksum mismatch")
            original = ActorState.FromString(payload)
            current = rows[identity]
            if stub.GetOptionName(current.actor_info, timeout=1.).value != row["model"] or current.actor_info != original.actor_info:
                raise RuntimeError("Foreign actor identity or model changed before relocation")
            target = copy.deepcopy(original.transform)
            if action == "move":
                target.location.x, target.location.y, target.location.z = -60., -150.-15.*index, 28.4
            result = set_transform(stub, current.actor_info, target, restoring=action == "restore")
            evidence.append(dict(id=identity, model=row["model"], **result))
        except Exception as error:
            if action != "restore":
                raise
            # Restore the other foreign actors even if one identity changed or
            # one readback failed; never leave them moved behind that failure.
            errors.append({"id": row["id"], "failure_reason": str(error)})
    report = {"status": "moved" if action == "move" else "restored", "parking": parking, "actors": evidence}
    if errors:
        report.update(status="failed", errors=errors)
    write_json(output/(action+".json"), report)
    if errors:
        raise RuntimeError("Some foreign actors could not be restored: " + json.dumps(errors))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", required=True, choices=("recover", "move", "restore"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-root", required=True, type=Path)
    parser.add_argument("--actor-ids", type=actor_ids, default=[])
    args = parser.parse_args()
    report = {"status": "failed", "action": args.action, "started_unix": time.time()}
    grpc = None
    try:
        configuration, _, _ = load_configuration()
        grpc, ego = _attach(configuration)
        report.update(recover(grpc, ego, args.batch_root) if args.action == "recover" else relocate(grpc, ego, args.output, args.actor_ids, args.action))
    except Exception as error:
        report["failure_reason"] = str(error)
        print("Batch safety failed: " + str(error), flush=True)
    finally:
        if grpc is not None:
            grpc.client.disconnect()
        report["finished_unix"] = time.time()
        write_json(args.output/(args.action+"_result.json"), report)
    print("Batch safety: " + report["status"], flush=True)
    return 0 if report["status"] != "failed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
