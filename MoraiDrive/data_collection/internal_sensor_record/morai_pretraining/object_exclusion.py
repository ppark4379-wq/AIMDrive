"""Measured, capture-scoped exclusion of a stationary actor outside the BEV.

No positive labels are generated for the excluded actor. Near, moving,
resized, or unmatched actors retain whole-frame invalidation.
"""
import hashlib
import json
import math
from pathlib import Path


def checked_json(path, digest):
    content = Path(path).read_bytes()
    if hashlib.sha256(content).hexdigest() != digest:
        raise ValueError("Object exclusion evidence changed: " + str(path))
    return json.loads(content)


def validate_exclusion_policy(policy, route):
    if policy is None:
        return None
    if policy.get("schema") != "morai_object_exclusion_v1":
        raise ValueError("Unknown object exclusion policy")
    allowed = {str(Path(p).resolve()) for p in policy["allowed_routes"]}
    if str(Path(route).resolve()) not in allowed:
        raise ValueError("Object evidence does not cover this recorded route")
    reference_path = Path(policy["reference_measurement_path"]).resolve()
    if str(reference_path.parent.parent) not in allowed or reference_path.parent.name != "measurements_full":
        raise ValueError("Reference measurement is outside the evidence scope")
    reference = checked_json(reference_path, policy["reference_measurement_sha256"])
    evidence = checked_json(policy["rpc_evidence_path"], policy["rpc_evidence_sha256"])
    actor_id = str(policy["actor_id"])
    if actor_id.lower() in ("ego", "0"):
        raise ValueError("Ego cannot be excluded as an external actor")
    raw = [o for o in reference["capture"]["objects_raw"] if str(o["id"]) == actor_id]
    states = [s for s in evidence["response"]["states"] if s["actor_info"]["id"]["value"] == actor_id]
    if len(raw) != 1 or len(states) != 1:
        raise ValueError("Actor identity is missing or ambiguous")
    raw, state = raw[0], states[0]
    box, origin = state["bounding_box"], state["transform"]["location"]
    extents = [box["extent"][k] for k in ("x", "y", "z")]
    offset = [box["location"][k] - origin[k] for k in ("x", "y", "z")]
    values = extents + offset + raw["size_xyz_raw_m"]
    if not all(math.isfinite(x) for x in values) or min(extents + raw["size_xyz_raw_m"]) <= 0:
        raise ValueError("Invalid enclosing geometry")
    if any(abs(a-b) > .05 for a, b in zip(sorted(2*x for x in extents), sorted(raw["size_xyz_raw_m"]))):
        raise ValueError("ROS and RPC sizes do not match the observed scale")
    # A 3D sphere encloses the box for any orientation/extent-axis ordering.
    radius = math.sqrt(sum(x*x for x in offset)) + math.sqrt(sum(x*x for x in extents)) + .1
    return dict(actor_id=actor_id, model_name=raw["model_name"], object_type=raw["type"],
                raw_size=raw["size_xyz_raw_m"], enclosing_radius_m=radius,
                bridge_instance=reference["capture"]["bridge_health"]["instance_id"])


def definitely_outside_bev(obj, frame, policy):
    if policy is None or str(obj.get("obj_id")) != policy["actor_id"]:
        return False
    cap = frame["capture"]
    if cap.get("bridge_health", {}).get("instance_id") != policy["bridge_instance"]:
        return False
    records = [o for o in cap["objects_raw"] if str(o["id"]) == policy["actor_id"]]
    if len(records) != 1:
        return False
    raw = records[0]
    if raw["model_name"] != policy["model_name"] or raw["type"] != policy["object_type"]:
        return False
    values = raw["size_xyz_raw_m"] + raw["position"] + raw["velocity_raw"]
    if not all(math.isfinite(x) for x in values):
        return False
    if (len(raw["size_xyz_raw_m"]) != 3 or
            any(abs(a-b) > .01 for a, b in zip(raw["size_xyz_raw_m"], policy["raw_size"])) or
            math.sqrt(sum(v*v for v in raw["velocity_raw"])) > .01):
        return False
    ego = frame["ego_raw"]
    dx, dy = raw["position"][0] - ego["x"], raw["position"][1] - ego["y"]
    c, s = math.cos(ego["yaw"]), math.sin(ego["yaw"])
    forward, left = c*dx + s*dy, -s*dx + c*dy
    # Native 50x50 one-metre cells: forward [-20,30], lateral [-25,25].
    distance = math.hypot(max(-20.-forward, 0., forward-30.), max(-25.-left, 0., left-25.))
    return distance > policy["enclosing_radius_m"]
