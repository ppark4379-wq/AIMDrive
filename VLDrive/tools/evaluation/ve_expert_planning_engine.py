"""Read-only snapshot comparison using the repository's actual lattice planner.

This is an input/geometry integration probe, not a driving or full Expert test.
No ROS, gRPC, controller, runtime Config, or simulator is instantiated. Unknown
VE categories stay unknown; the generic planner deliberately uses only boxes.
"""

import hashlib
import math
from pathlib import Path
import sys


SCHEMA = "ve_expert_generic_lattice_pair_v1"
_REPOSITORY = Path(__file__).resolve().parents[3]
_EXPERT_PACKAGE_ROOT = _REPOSITORY / "MoraiDrive/expert/src"


def _load_planner():
    # Imports the real repository implementation rather than a copied planner.
    root = str(_EXPERT_PACKAGE_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    import expert.src.planning.lattice.lattice_planner as implementation
    from expert.src.localization.point import Point
    from expert.src.obstacle.object_info import ObjectInfo
    from expert.src.vehicle_state import VehicleState

    expected = _EXPERT_PACKAGE_ROOT / "expert/src/planning/lattice/lattice_planner.py"
    if Path(implementation.__file__).resolve() != expected.resolve():
        raise RuntimeError("An unrelated Expert package shadows this repository")
    return implementation.LatticePlanner, Point, ObjectInfo, VehicleState, expected


def _number(value, name, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError("{} must be a finite numeric value".format(name))
    value = float(value)
    if not math.isfinite(value) or (nonnegative and value < 0.0):
        raise ValueError("{} must be a finite{} value".format(name, " nonnegative" if nonnegative else ""))
    return value


def _xy(value, name):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("{} must contain exactly two map coordinates".format(name))
    return (_number(value[0], name + ".x"), _number(value[1], name + ".y"))


def _validate_boxes(boxes, source, ObjectInfo):
    if not isinstance(boxes, (list, tuple)):
        raise ValueError("{} boxes must be an array".format(source))
    objects, metadata = [], []
    for index, box in enumerate(boxes):
        label = "{}[{}]".format(source, index)
        if box.get("source") != source:
            raise ValueError(label + " has an incorrect source identity")
        x, y = _xy(box.get("center_world"), label + ".center_world")
        length = _number(box.get("length_m"), label + ".length_m", True)
        width = _number(box.get("width_m"), label + ".width_m", True)
        if length <= 0.0 or width <= 0.0:
            raise ValueError(label + " has a degenerate footprint")
        heading = _number(box.get("heading_deg"), label + ".heading_deg")
        speed = box.get("speed_mps")
        if speed is not None:
            speed = _number(speed, label + ".speed_mps", True)
        confidence = _number(box.get("confidence", 1.0), label + ".confidence", True)
        if confidence > 1.0:
            raise ValueError(label + " confidence exceeds one")
        kind = box.get("kind", "unknown")
        if kind not in ("unknown", "npc", "pedestrian", "obstacle"):
            raise ValueError(label + " has an unsupported kind")
        static = box.get("is_static")
        if static is not None and not isinstance(static, bool):
            raise ValueError(label + " is_static must be a bool or null")
        # Generic lattice does not consult type, static status, IDs, or velocity.
        # In particular, never substitute a GT class or GT speed into VE boxes.
        objects.append(ObjectInfo(x, y, speed if speed is not None else 0.0, -1,
                                  is_static=False, size_x=length, size_y=width,
                                  heading_deg=heading, unique_id=box.get("track_id")))
        metadata.append(dict(kind=kind, velocity_unknown=speed is None,
                             planner_size_clamped=length < 1.0 or width < 1.0,
                             physical_length_m=length, physical_width_m=width))
    return objects, metadata


def _cumulative(path):
    total, distances = 0.0, [0.0]
    for first, second in zip(path[:-1], path[1:]):
        total += math.hypot(second[0] - first[0], second[1] - first[1])
        distances.append(total)
    return distances


def _point_at(path, distances, target):
    if target <= 0.0:
        return path[0]
    if target >= distances[-1]:
        return path[-1]
    import bisect
    index = bisect.bisect_right(distances, target) - 1
    length = distances[index + 1] - distances[index]
    ratio = (target - distances[index]) / length if length > 1e-12 else 0.0
    return tuple(path[index][axis] + ratio * (path[index + 1][axis] - path[index][axis])
                 for axis in (0, 1))


def _trim_official_path(path, position, closed=False, route_index=None):
    """Crop GP at its closest segment without using expert future trajectory GT."""
    if closed and path[0] != path[-1]:
        path = list(path) + [path[0]]
    nearest = None
    for index, (start, end) in enumerate(zip(path[:-1], path[1:])):
        if route_index is not None:
            separation = abs(index - route_index)
            if closed:
                separation = min(separation, abs(len(path) - 1 - separation))
            if separation > 3:
                continue
        dx, dy = end[0] - start[0], end[1] - start[1]
        length_sq = dx * dx + dy * dy
        if length_sq < 1e-12:
            continue
        ratio = max(0.0, min(1.0, ((position[0] - start[0]) * dx +
                                 (position[1] - start[1]) * dy) / length_sq))
        projection = (start[0] + ratio * dx, start[1] + ratio * dy)
        distance = math.hypot(position[0] - projection[0], position[1] - projection[1])
        if nearest is None or distance < nearest[0]:
            nearest = (distance, index, projection)
    if nearest is None:
        raise ValueError("Official path has no nonzero segment")
    distance, index, projection = nearest
    cropped = [projection]
    for point in path[index + 1:]:
        if math.hypot(point[0] - cropped[-1][0], point[1] - cropped[-1][1]) > 1e-9:
            cropped.append(point)
    if closed:
        for point in path[1:index + 2]:
            if math.hypot(point[0] - cropped[-1][0], point[1] - cropped[-1][1]) > 1e-9:
                cropped.append(point)
    if len(cropped) < 2:
        raise ValueError("Official path has no forward segment at the ego pose")
    return cropped, distance, index


def _truncate_path(path, horizon):
    distances = _cumulative(path)
    if distances[-1] <= horizon:
        return path
    result = [point for point, distance in zip(path, distances) if distance < horizon]
    result.append(_point_at(path, distances, horizon))
    return result


def _vertices(center, yaw, length, width):
    forward, left = (math.cos(yaw), math.sin(yaw)), (-math.sin(yaw), math.cos(yaw))
    return [(center[0] + a * length * 0.5 * forward[0] + b * width * 0.5 * left[0],
             center[1] + a * length * 0.5 * forward[1] + b * width * 0.5 * left[1])
            for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))]


def _point_segment_distance(point, start, end):
    dx, dy = end[0] - start[0], end[1] - start[1]
    length_sq = dx * dx + dy * dy
    ratio = 0.0 if length_sq <= 1e-12 else max(0.0, min(1.0,
        ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / length_sq))
    return math.hypot(point[0] - start[0] - ratio * dx,
                      point[1] - start[1] - ratio * dy)


def signed_polygon_clearance(first, second):
    """Euclidean separation when disjoint, negative SAT overlap when colliding."""
    overlap = float("inf")
    disjoint = False
    for polygon in (first, second):
        for start, end in zip(polygon, polygon[1:] + polygon[:1]):
            dx, dy = end[0] - start[0], end[1] - start[1]
            length = math.hypot(dx, dy)
            if length <= 1e-12:
                continue
            axis = (-dy / length, dx / length)
            projections = [[p[0] * axis[0] + p[1] * axis[1] for p in shape]
                           for shape in (first, second)]
            interval = min(max(projections[0]), max(projections[1])) - max(
                min(projections[0]), min(projections[1]))
            overlap = min(overlap, interval)
            if interval < -1e-10:
                disjoint = True
    if not disjoint:
        return -max(0.0, overlap)
    return min(_point_segment_distance(point, start, end)
               for points, edges in ((first, second), (second, first))
               for point in points
               for start, end in zip(edges, edges[1:] + edges[:1]))


def _assess_path(path, boxes, planner, sample_spacing=0.25, horizon_m=None):
    if len(path) < 2:
        return dict(checked=False, collision=None, minimum_clearance_m=None,
                    first_collision_distance_m=None, covered_distance_m=0.0,
                    samples=0, collision_box_indices=[])
    distances = _cumulative(path)
    covered = distances[-1] if horizon_m is None else min(distances[-1], horizon_m)
    count = max(1, int(math.ceil(covered / sample_spacing)))
    minimum, first_hit, indices = float("inf"), None, set()
    for index in range(count + 1):
        distance = covered * index / count
        point = _point_at(path, distances, distance)
        lower = _point_at(path, distances, max(0.0, distance - 0.1))
        upper = _point_at(path, distances, min(distances[-1], distance + 0.1))
        yaw = math.atan2(upper[1] - lower[1], upper[0] - lower[0])
        # Match the upstream lattice's rear-axle anchor and collision margin.
        center = (point[0] + planner.vehicle_length * 0.5 * math.cos(yaw),
                  point[1] + planner.vehicle_length * 0.5 * math.sin(yaw))
        vehicle = _vertices(center, yaw, planner.vehicle_length + 2.0 * planner.collision_margin,
                            2.0 * planner.vehicle_half_width + 2.0 * planner.collision_margin)
        for box_index, box in enumerate(boxes):
            obstacle = _vertices(box["center_world"], math.radians(box["heading_deg"]),
                                 box["length_m"], box["width_m"])
            clearance = signed_polygon_clearance(vehicle, obstacle)
            minimum = min(minimum, clearance)
            if clearance <= 1e-10:
                indices.add(box_index)
                first_hit = distance if first_hit is None else min(first_hit, distance)
    return dict(checked=True, collision=bool(indices), minimum_clearance_m=(
        minimum if math.isfinite(minimum) else None), first_collision_distance_m=first_hit,
        covered_distance_m=covered, samples=count + 1, collision_box_indices=sorted(indices))


def _json_result(result):
    decision = ("avoidance_candidate" if result.active else
                "all_candidates_blocked" if result.reason == "all_candidates_blocked" else
                "reference_path")
    return dict(decision=decision, reason=result.reason, mode=result.mode,
                active=result.active, obstacle_detected=result.obstacle_detected,
                long_obstacle_detected=result.long_obstacle_detected,
                longitudinal_speed_scale=result.longitudinal_speed_scale,
                selected_offset_m=result.selected_offset,
                candidate_count=len(result.candidates),
                valid_short_candidates=sum(c.valid and c.layer == "short" for c in result.candidates),
                path=[[float(point.x), float(point.y)] for point in result.path])


def evaluate_planning_pair(frame, planner_parameters=None):
    """Return a paired GT/VE lattice geometry report for the same official GP.

    Required frame keys: ego_pose={x,y,yaw(rad)}, speed_mps,
    official_reference_path=[[x,y],...], reference_path_source='official_global_path',
    gt_world_boxes and ve_world_boxes (the neutral adapter contract).
    """
    if frame.get("reference_path_source") != "official_global_path":
        raise ValueError("Only the official GP may be used as a planner reference")
    ego = frame.get("ego_pose", {})
    x, y = _number(ego.get("x"), "ego.x"), _number(ego.get("y"), "ego.y")
    yaw = _number(ego.get("yaw"), "ego.yaw")
    speed = _number(frame.get("speed_mps"), "ego.speed_mps", True)
    raw_path = frame.get("official_reference_path")
    if not isinstance(raw_path, (list, tuple)) or len(raw_path) < 2:
        raise ValueError("The official GP requires at least two points")
    path = [_xy(value, "official_reference_path") for value in raw_path]
    closed = frame.get("official_reference_closed", False)
    if not isinstance(closed, bool):
        raise ValueError("official_reference_closed must be a bool")
    route_index = frame.get("reference_path_start_index")
    if route_index is not None:
        if isinstance(route_index, bool) or not isinstance(route_index, int) or not 0 <= route_index < len(path):
            raise ValueError("reference_path_start_index must be an index in this exact official GP")
        if frame.get("reference_path_index_source") != "official_route_index":
            raise ValueError("A route occurrence hint requires official_route_index provenance")
    cropped, route_distance, closest_segment = _trim_official_path(path, (x, y), closed, route_index)
    cropped = _truncate_path(cropped, 40.0)
    LatticePlanner, Point, ObjectInfo, VehicleState, implementation_path = _load_planner()
    gt_objects, gt_metadata = _validate_boxes(frame.get("gt_world_boxes"), "gt", ObjectInfo)
    ve_objects, ve_metadata = _validate_boxes(frame.get("ve_world_boxes"), "ve", ObjectInfo)
    kwargs = dict(planner_parameters or {})
    # Independent instances ensure one arm cannot prime another's offset memory.
    planners = [LatticePlanner(**kwargs), LatticePlanner(**kwargs)]
    reference = [Point(*position) for position in cropped]
    vehicle = VehicleState(x=x, y=y, yaw=yaw, velocity=speed)
    results = [_json_result(planner.plan(vehicle, reference, objects, prefer_right=True,
                            mode="generic_geometry_snapshot"))
               for planner, objects in zip(planners, (gt_objects, ve_objects))]
    gt_boxes = frame["gt_world_boxes"]
    clamped_gt = [dict(box, length_m=max(1.0, box["length_m"]),
                       width_m=max(1.0, box["width_m"])) for box in gt_boxes]
    reference_check = _assess_path(cropped, gt_boxes, planners[0], horizon_m=35.0)
    common_horizon = min(_cumulative(result["path"])[-1] for result in results)
    common_horizon = min(35.0, common_horizon)
    for result, planner in zip(results, planners):
        result["gt_physical_footprint"] = _assess_path(result["path"], gt_boxes, planner)
        result["gt_planner_clamped_footprint"] = _assess_path(result["path"], clamped_gt, planner)
        result["common_horizon_gt_footprint"] = _assess_path(result["path"], gt_boxes, planner,
                                                            horizon_m=common_horizon)
        result["trajectory_is_snapshot_only"] = True
    gt_path, ve_path = (result["path"] for result in results)
    gt_s, ve_s = _cumulative(gt_path), _cumulative(ve_path)
    comparisons = max(1, int(math.ceil(common_horizon / 0.5)))
    deviations = [math.hypot(*(
        a - b for a, b in zip(_point_at(gt_path, gt_s, common_horizon * i / comparisons),
                             _point_at(ve_path, ve_s, common_horizon * i / comparisons))))
                  for i in range(comparisons + 1)]
    route_yaw = math.atan2(cropped[1][1] - cropped[0][1], cropped[1][0] - cropped[0][0])
    route_yaw_error = abs(math.atan2(math.sin(route_yaw - yaw), math.cos(route_yaw - yaw)))
    return dict(schema=SCHEMA, scope="actual_generic_lattice_static_footprint_snapshot",
                reference_path_source="official_global_path", route_projection_distance_m=route_distance,
                route_closest_segment_index=closest_segment, route_ego_yaw_error_rad=route_yaw_error,
                reference_path_closed=closed, reference_path_available_m=_cumulative(cropped)[-1],
                reference_path_world=[list(point) for point in cropped],
                reference_occurrence_selection=("official_route_index_neighborhood" if route_index is not None
                                               else "global_nearest_segment_unverified_occurrence"),
                reference_path_start_index=route_index,
                generic_gt=results[0], generic_ve=results[1],
                reference_path_gt_footprint=reference_check,
                comparison=dict(decision_agreement=results[0]["decision"] == results[1]["decision"],
                    obstacle_flag_agreement=results[0]["obstacle_detected"] == results[1]["obstacle_detected"],
                    selected_offset_delta_m=results[1]["selected_offset_m"] - results[0]["selected_offset_m"],
                    common_horizon_m=common_horizon, path_mean_separation_m=sum(deviations) / len(deviations),
                    path_max_separation_m=max(deviations),
                    gt_path_collision=results[0]["common_horizon_gt_footprint"]["collision"],
                    ve_path_collision=results[1]["common_horizon_gt_footprint"]["collision"]),
                input_coverage=dict(gt_boxes=len(gt_objects), ve_boxes=len(ve_objects),
                    unknown_ve_categories=sum(row["kind"] == "unknown" for row in ve_metadata),
                    unknown_ve_velocities=sum(row["velocity_unknown"] for row in ve_metadata),
                    clamped_gt_boxes=sum(row["planner_size_clamped"] for row in gt_metadata),
                    clamped_ve_boxes=sum(row["planner_size_clamped"] for row in ve_metadata)),
                planner_provenance=dict(path=str(implementation_path), sha256=hashlib.sha256(
                    implementation_path.read_bytes()).hexdigest(), parameters=kwargs,
                    vehicle_position_convention="upstream_lattice_rear_axle_anchor",
                    vehicle_length_m=planners[0].vehicle_length,
                    vehicle_width_m=2.0 * planners[0].vehicle_half_width,
                    collision_margin_m=planners[0].collision_margin,
                    obstacle_minimum_length_width_m=1.0),
                limitations=dict(full_autonomous_driving_validated=False,
                    dynamic_time_prediction_validated=False, traffic_signal_validated=False,
                    road_boundary_validated=False, tracking_validated=False,
                    ego_pose_rear_axle_reference_verified=False,
                    route_occurrence_disambiguated_by_caller=route_index is not None,
                    generic_lateral_offset_range_m=[-0.5 * (planners[0].num_offsets - 1) * planners[0].lateral_offset_step,
                                                    0.5 * (planners[0].num_offsets - 1) * planners[0].lateral_offset_step],
                    class_and_velocity_used=False, sensor_model_loaded=False,
                    snapshot_collision_sample_spacing_m=0.25,
                    missing_velocity_zero_placeholder_geometry_only=True,
                    collision_report_is_not_closed_loop_driving_success=True))
