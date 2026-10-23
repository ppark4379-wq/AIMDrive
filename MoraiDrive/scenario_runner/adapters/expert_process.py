#!/usr/bin/env python3
"""Import and execute the external AIMDrive expert from scenario_runner."""

import os
import hashlib
import json
import math
from pathlib import Path
import sys
import time


def _json_finite(value):
    if isinstance(value, dict):
        return {str(key): _json_finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_finite(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "tolist"):
        return _json_finite(value.tolist())
    return value


def configure_tunnel_trial(driving, config, boundary_factory=None):
    """Install private measured trial inputs before the first ROS control tick."""
    planning = config.get("planning", {})
    rows = planning.get("jamming_scene_geometry")
    if rows is None:
        return
    contract = planning.get("jamming_trial_contract", {})
    if contract.get("mode") != "full_loop_tunnel_static" or not isinstance(rows, list) or not rows:
        raise RuntimeError("Measured Jamming geometry requires the guarded tunnel trial contract")
    if boundary_factory is None:
        from expert.src.planning.jamming.road_boundary import RoadBoundary
        boundary_factory = RoadBoundary.from_map_dir
    policy = contract["road_boundary_policy"]
    boundary = boundary_factory(config["map"]["data_dir"], body_policy=policy["body"],
                                approach_m=policy["approach_m"], departure_m=policy["departure_m"],
                                planning_margin_m=policy["planning_road_margin_m"],
                                runtime_margin_m=policy["runtime_road_margin_m"])
    if boundary.policy != policy or boundary.policy_sha256 != contract["road_boundary_policy_sha256"]:
        raise RuntimeError("Runtime Jamming road/body policy differs from measured screening")
    planner = driving.lattice_planning_process.recorded_jamming
    if planner is None:
        raise RuntimeError("Production JammingPlanner is unavailable for the trial")
    planner.set_road_boundary(boundary)
    planner.set_object_geometries(rows)
    scene_hash = hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":"),
                                          allow_nan=False).encode()).hexdigest()
    destination = Path(planning["jamming_trial_diagnostics_path"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    observed = {"schema": "aimdrive_jamming_trial_runtime_v1", "calls": 0, "produced_results": 0,
                "validated_path_seen": False, "selected_routes": [], "scene_geometry_sha256": scene_hash,
                "road_boundary_policy": boundary.policy, "road_boundary_policy_sha256": boundary.policy_sha256}
    original_run = planner.run

    def measured_run(*args, **kwargs):
        result = original_run(*args, **kwargs)
        observed["calls"] += 1
        if result is not None:
            observed["produced_results"] += 1
            details = getattr(result, "debug_details", {})
            static = details.get("static_avoidance", {})
            road = static.get("road_boundary", {})
            known_routes = set(observed["selected_routes"])
            known_routes.update(static.get("selected_routes", []))
            known_routes.update(getattr(planner.local, "selected_names", []))
            observed["selected_routes"] = sorted(known_routes)
            observed["validated_path_seen"] |= (static.get("path_validated") is True
                and road.get("passed") is True and road.get("policy_sha256") == boundary.policy_sha256
                and details.get("jamming_scene_geometry_sha256") == scene_hash)
            observed["last_result"] = {"mode": getattr(result, "mode", None),
                "reason": getattr(result, "reason", None), "active": bool(getattr(result, "active", False)),
                "debug_details": _json_finite(details)}
        observed["updated_monotonic"] = time.monotonic()
        temporary = destination.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(_json_finite(observed), indent=2, default=repr, allow_nan=False) + "\n", encoding="utf-8")
        os.replace(temporary, destination)
        return result
    planner.run = measured_run


def configure_ros_message_path(expert_root):
    """Use the message build shipped with this Expert, before inherited overlays."""
    message_path = Path(expert_root).resolve() / "devel/lib/python3/dist-packages"
    if not (message_path / "morai_msgs/msg/__init__.py").is_file():
        return None
    message_path_string = str(message_path)
    # An older ~/morai_ws overlay has a different EgoVehicleStatus MD5 and
    # silently prevents the Expert from receiving the bridge's /Ego_topic.
    if message_path_string in sys.path:
        sys.path.remove(message_path_string)
    sys.path.insert(0, message_path_string)
    return message_path


def main():
    expert_root = os.environ.get("MORAI_EXPERT_ROOT")
    if not expert_root:
        raise RuntimeError("MORAI_EXPERT_ROOT is not configured")

    configure_ros_message_path(expert_root)
    expert_source = os.path.join(os.path.abspath(expert_root), "src")
    if expert_source not in sys.path:
        sys.path.insert(0, expert_source)

    from expert.src.autonomous_driving import AutonomousDriving
    from expert.src.localization.path_manager import PathManager
    from expert.src.network.ros_manager import RosManager
    import morai_msgs.msg
    from morai_msgs.msg import CtrlCmd, EgoVehicleStatus

    root = Path(expert_root).resolve()
    modules = {
        "autonomous_driving": Path(
            sys.modules[AutonomousDriving.__module__].__file__
        ).resolve(),
        "path_manager": Path(
            sys.modules[PathManager.__module__].__file__
        ).resolve(),
        "ros_manager": Path(
            sys.modules[RosManager.__module__].__file__
        ).resolve(),
    }
    for name, module_path in modules.items():
        try:
            module_path.relative_to(root)
        except ValueError:
            raise RuntimeError(
                f"External Expert import escaped configured root: "
                f"{name}={module_path}, root={root}"
            )

    print(f"[Expert] mode={os.environ.get('MORAI_EXPERT_MODE', 'unknown')}", flush=True)
    print(f"[Expert] root={root}", flush=True)
    for name, module_path in modules.items():
        print(f"[Expert] {name} module={module_path}", flush=True)
    print(f"[Expert] morai_msgs={morai_msgs.msg.__file__}", flush=True)
    print(
        f"[Expert] ROS message MD5: EgoVehicleStatus={EgoVehicleStatus._md5sum} "
        f"CtrlCmd={CtrlCmd._md5sum}",
        flush=True,
    )

    driving = AutonomousDriving()
    config_path = os.environ.get("GT_BEV_CONFIG_PATH")
    if config_path:
        configure_tunnel_trial(driving, json.loads(Path(config_path).read_text(encoding="utf-8")))
    RosManager(driving).execute()


if __name__ == "__main__":
    main()
