"""Per-model choice of the most accurate recorded box source, proven by LiDAR.

MORAI publishes two boxes for the same actor and neither is right for every
model (static36 + full_route, native LiDAR within 12 m):

* gRPC ``bounding_box``: correct centre for vehicles and bicycles, but a
  bind-pose size for pedestrians (1.8 m) and a short collider for bicycles.
* ROS ``/Object_topic`` (what the Expert drives on): the actor pivot as
  position (1.1-1.5 m behind a vehicle centre), and a model size that fits
  pedestrians and bicycles; for some models its x/y are swapped.

Candidate boxes combine a centre (gRPC, the V4 catalog correction, or a
reviewed stationary-image fit offset from the pivot) with a
size (gRPC, ROS, ROS with x/y swapped).  A registry keeps, per model and
scale, the tightest candidate whose median LiDAR containment passes; the V5
audit applies it and masks models without such evidence.
"""
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics

import numpy as np

from async_capture.alignment import PayloadReader, decode, recorded_classes
from async_capture.journal import digest, verify_run
from async_capture.portable_paths import recorded_episode_directory


REGISTRY_SCHEMA = "morai_object_geometry_registry_v1"
DEFAULT_REGISTRY_POLICY = {
    "max_range_m": 12.0,
    "min_roi_points": 15,
    "containment_margin_m": .10,
    "min_observations": 10,
    "min_median_containment": .95,
    "max_observations_per_model_per_run": 40,
    # Conventional LiDAR ground removal: road contact, curbs and bases below.
    "height_above_ground_min_m": .30,
    # Points farther than this from every candidate box are background.
    "candidate_shell_m": .50,
    "ground_ring_width_m": 2.0,
    "ground_ring_min_points": 20,
    # base_link sits 0.30 m above the road for the recorded Ioniq5 Ego.
    "ground_fallback_above_base_link_m": -.30,
}
CENTERS = ("grpc", "v4", "fit")
SIZES = ("grpc", "ros", "ros_swap")
# base_link -> Lidar3D-5 in the pinned sensor contract (FLU, no rotation).
LIDAR_OFFSET_XZ = (1.0, 1.48)
ROS_LISTS = (("npc_list", "npc"), ("pedestrian_list", "pedestrian"), ("obstacle_list", "obstacle"))


def load_ros_object_sizes(episode, active, expected_raw_hash, ssd_root):
    """Constant /Object_topic size per actor id from every recorded message."""
    episode = Path(episode)
    index = json.loads((episode / "quality/INDEX_COMPLETE.json").read_text())
    raw = recorded_episode_directory(index["raw_path"], episode, "raw", ssd_root)
    manifest, _ = verify_run(raw, payloads=False)
    if digest(raw / "raw_manifest.json") != expected_raw_hash:
        raise ValueError("ROS object evidence belongs to a different raw recording")
    classes = recorded_classes(manifest)
    sizes = defaultdict(set)
    kinds = {}
    messages = 0
    reader = PayloadReader(raw)
    try:
        for meta in active:
            if meta["role"] != "objects":
                continue
            message = decode(meta, reader.read(meta), classes)
            messages += 1
            for field, kind in ROS_LISTS:
                for obj in getattr(message, field):
                    key = str(int(obj.unique_id))
                    sizes[key].add(tuple(round(float(v), 4) for v in (obj.size.x, obj.size.y, obj.size.z)))
                    kinds[key] = (kind, str(obj.name))
    finally:
        reader.close()
    actors = {}
    for key, values in sizes.items():
        size = sorted(values)[0]
        actors[key] = {"kind": kinds[key][0], "name": kinds[key][1],
                       "size_xyz_m": list(size), "constant": len(values) == 1,
                       "valid": len(values) == 1 and all(math.isfinite(v) and v > 0 for v in size)}
    return {"schema": "morai_ros_object_size_evidence_v1", "raw_manifest_sha256": expected_raw_hash,
            "messages": messages, "actors": actors}


def candidate_sizes(item, ros):
    """Length along heading and width across it for each size source."""
    width, length = float(item["size_wlh_m"][0]), float(item["size_wlh_m"][1])
    result = {"grpc": (length, width)}
    if ros and ros.get("valid"):
        x, y = float(ros["size_xyz_m"][0]), float(ros["size_xyz_m"][1])
        result["ros"], result["ros_swap"] = (x, y), (y, x)
    return result


def model_key(item, scale=None):
    """Registry key; an obstacle instance scale changes its geometry."""
    scale = [1.0, 1.0, 1.0] if scale is None else [round(float(v), 6) for v in scale]
    return "%s|%s|%s" % (item.get("kind"), str(item.get("model_name", "")).strip(),
                         ",".join("%g" % v for v in scale))


def containment(points_xy, center, heading_deg, length, width, margin):
    dx, dy = points_xy[:, 0] - center[0], points_xy[:, 1] - center[1]
    heading = math.radians(heading_deg)
    along = math.cos(heading) * dx + math.sin(heading) * dy
    across = -math.sin(heading) * dx + math.cos(heading) * dy
    return float(np.mean((np.abs(along) <= length / 2 + margin) & (np.abs(across) <= width / 2 + margin)))


def inside_box(points_xy, item, margin):
    """Points inside an actor's recorded gRPC box, inflated by ``margin``."""
    width, length = float(item["size_wlh_m"][0]), float(item["size_wlh_m"][1])
    dx = points_xy[:, 0] - float(item["position"][0])
    dy = points_xy[:, 1] - float(item["position"][1])
    heading = math.radians(float(item["heading_deg"]))
    along = math.cos(heading) * dx + math.sin(heading) * dy
    across = -math.sin(heading) * dx + math.cos(heading) * dy
    return (np.abs(along) <= length / 2 + margin) & (np.abs(across) <= width / 2 + margin)


def roi_points(world_xy, height, item, sizes, policy=DEFAULT_REGISTRY_POLICY, others=(), centers=None):
    """Points attributed to the actor: near any candidate box, above the local road.

    Only points within ``candidate_shell_m`` of some candidate box count, so a
    box that is too small still loses containment while curbs and roadside
    clutter further away do not.  Points inside a neighbouring actor's box
    belong to that actor.
    """
    centers = centers or [(float(item["position"][0]), float(item["position"][1]))]
    reach = max(max(pair) for pair in sizes.values()) / 2 + 1.5
    distance = np.hypot(world_xy[:, 0] - centers[0][0], world_xy[:, 1] - centers[0][1])
    # The local road height from a ring outside the actor, for slopes and any Ego base_link.
    ring = (distance > reach) & (distance < reach + policy["ground_ring_width_m"])
    ground = (float(np.percentile(height[ring], 10)) if ring.sum() >= policy["ground_ring_min_points"]
              else policy["ground_fallback_above_base_link_m"])
    keep = ((height > ground + policy["height_above_ground_min_m"])
            & (height < ground + float(item["size_wlh_m"][2]) + .3))
    near = np.zeros(len(world_xy), dtype=bool)
    for center in centers:
        for length, width in sizes.values():
            box = dict(item, position=[center[0], center[1], 0.], size_wlh_m=[width, length, 1.])
            near |= inside_box(world_xy, box, policy["candidate_shell_m"])
    keep &= near
    for other in others:
        keep &= ~inside_box(world_xy, other, .2)
    return world_xy[keep]


def fit_center(item, observed_offset_xy, fit_offset_xy):
    """Pivot + a pivot-relative fit offset, from the recorded box centre."""
    delta = (float(fit_offset_xy[0]) - float(observed_offset_xy[0]),
             float(fit_offset_xy[1]) - float(observed_offset_xy[1]))
    heading = math.radians(float(item["heading_deg"]))
    c, s = math.cos(heading), math.sin(heading)
    return (float(item["position"][0]) + c * delta[0] - s * delta[1],
            float(item["position"][1]) + s * delta[0] + c * delta[1])


def score_observation(world_xy, height, item, ros, extra_centers=None, policy=DEFAULT_REGISTRY_POLICY,
                      others=()):
    """Containment of every candidate box for one actor in one LiDAR frame.

    ``extra_centers`` maps "v4"/"fit" to alternative world X/Y centres.
    """
    sizes = candidate_sizes(item, ros)
    centers = {"grpc": (float(item["position"][0]), float(item["position"][1]))}
    for name, center in (extra_centers or {}).items():
        if center is not None:
            centers[name] = tuple(float(v) for v in center)
    points = roi_points(world_xy, height, item, sizes, policy, others, list(centers.values()))
    if len(points) < policy["min_roi_points"]:
        return None
    scores = {}
    for center_name, center in centers.items():
        for size_name, (length, width) in sizes.items():
            scores[center_name + "+" + size_name] = {
                "containment": containment(points, center, float(item["heading_deg"]),
                                           length, width, policy["containment_margin_m"]),
                "area_m2": length * width}
    return {"points": int(len(points)), "scores": scores}


def select_model_geometry(observations, policy=DEFAULT_REGISTRY_POLICY):
    """Tightest candidate whose median containment passes, else not approved."""
    summary = {}
    for name in sorted({name for row in observations for name in row["scores"]}):
        values = [row["scores"][name]["containment"] for row in observations if name in row["scores"]]
        areas = [row["scores"][name]["area_m2"] for row in observations if name in row["scores"]]
        summary[name] = {"observations": len(values),
                         "median_containment": statistics.median(values),
                         "area_m2": statistics.median(areas)}
    passing = [(row["area_m2"], name) for name, row in summary.items()
               if row["observations"] >= policy["min_observations"]
               and row["median_containment"] >= policy["min_median_containment"]]
    selected = min(passing)[1] if passing else None
    center, size = selected.split("+") if selected else (None, None)
    enough = any(row["observations"] >= policy["min_observations"] for row in summary.values())
    # Too little LiDAR is no evidence either way; contrary evidence is final.
    reason = (None if selected else "no_candidate_meets_lidar_containment" if enough
              else "insufficient_lidar_observations")
    return {"approved": selected is not None, "center_source": center, "size_source": size,
            "candidates": summary, "evidence_sufficient": enough, "reason": reason}


def lidar_world_points(points, ego):
    """Native FLU LiDAR points to world X/Y and height above base_link ground."""
    forward = points[:, 0] + LIDAR_OFFSET_XZ[0]
    left = points[:, 1]
    c, s = math.cos(ego["yaw"]), math.sin(ego["yaw"])
    world = np.column_stack((ego["x"] + c * forward - s * left, ego["y"] + s * forward + c * left))
    return world, points[:, 2] + LIDAR_OFFSET_XZ[1]


def check_fused_object_geometry(item, v4_result, registry, ros, scale, pivot_offset_xy=None):
    """V4 result refined by the registry: its evidence overrides the catalog.

    No registry entry, or too few LiDAR observations, keeps the V4 decision;
    sufficient contrary LiDAR evidence masks even a catalog-approved model.
    """
    result = dict(v4_result)
    if item.get("kind") == "obstacle" and scale is None:
        result["registry"] = {"key": None, "applied": False, "reason": "instance_scale_unknown"}
        return result
    key = model_key(item, scale)
    entry = registry.get("models", {}).get(key)
    if entry is None or entry.get("reason") == "insufficient_lidar_observations":
        reason = "no_registry_entry" if entry is None else entry["reason"]
        result["registry"] = {"key": key, "applied": False, "reason": reason}
        if item.get("kind") == "pedestrian":
            # Every measured human model reports a bind-pose gRPC box about as
            # wide as it is tall, which catalog containment cannot reject.
            result.pop("training_center_world_xy", None)
            result.update(reasons=["pedestrian_requires_lidar_registry"],
                          api_box_geometry_candidate=False)
        return result
    fused, reason = apply_registry(item, entry, ros, v4_result, pivot_offset_xy)
    result["registry"] = {"key": key, "applied": fused is not None, "reason": reason}
    result.pop("training_center_world_xy", None)
    if fused is None:
        result.update(reasons=[reason], api_box_geometry_candidate=False)
    else:
        result.update(fused, reasons=[], api_box_geometry_candidate=True)
    return result


def apply_registry(item, entry, ros, v4_result, pivot_offset_xy=None):
    """Training centre and size for a registry-approved model, or a reason."""
    if not entry.get("approved"):
        return None, entry.get("reason") or "registry_not_approved"
    if entry["center_source"] == "v4":
        center = (v4_result or {}).get("training_center_world_xy")
        if center is None:
            return None, "registry_v4_centre_unavailable"
    elif entry["center_source"] == "fit":
        if pivot_offset_xy is None or entry.get("center_offset_local_xy") is None:
            return None, "registry_fit_centre_unavailable"
        center = fit_center(item, pivot_offset_xy, entry["center_offset_local_xy"])
    else:
        center = [float(item["position"][0]), float(item["position"][1])]
    sizes = candidate_sizes(item, ros)
    if entry["size_source"] not in sizes:
        return None, "registry_ros_size_unavailable"
    length, width = sizes[entry["size_source"]]
    return {"training_center_world_xy": [float(center[0]), float(center[1])],
            "training_size_wl_m": [float(width), float(length)],
            "geometry_sources": {"center": entry["center_source"], "size": entry["size_source"]}}, None
