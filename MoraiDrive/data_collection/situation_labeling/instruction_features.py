"""Evidence-backed instruction features from sealed GT and the official route.

Unknown measurements stay None. Planner collision prediction is deliberately not
reconstructed from a single GT frame.
"""
from collections import defaultdict
import math
from pathlib import Path

from MoraiDrive.data_collection.mgeo_navigation import MGeoMap
from .features import RouteProjector


def number(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return value if math.isfinite(value) else None


def actor_id(row):
    return row.get("actor_id")


def _side(value):
    value = number(value)
    if value is None:
        return None
    return "left" if value > 0.3 else "right" if value < -0.3 else "center"


def _mgeo(route):
    # The labeling pipeline resolves and capture-hash-verifies this root.
    # Its map geometry must not be replaced by a same-ID checkout copy.
    if "map_root" in route:
        root = route["map_root"]
        if not isinstance(root, (str, Path)) or not str(root).strip():
            return None
        root = Path(root).expanduser()
        if not root.is_absolute():
            return None
        path = root / "link_set.json"
        return MGeoMap(str(path)) if path.is_file() else None
    # Compatibility for constructed/legacy routes without a selected root.
    map_id = route.get("map_id")
    if not isinstance(map_id, str) or not map_id or Path(map_id).name != map_id:
        return None
    path = Path(__file__).resolve().parents[2] / "scenario_runner" / "data" / map_id / "link_set.json"
    return MGeoMap(str(path)) if path.is_file() else None


def _link(mgeo, x, y, heading=None):
    if mgeo is None or x is None or y is None:
        return None
    match = mgeo.match(x, y, None if heading is None else math.radians(heading))
    return match.link_id if match is not None and match.distance_m <= 3.0 else None


def _checkpoint_s(route, projector, identity):
    for checkpoint in route.get("checkpoints", []):
        if str(checkpoint.get("id")) == str(identity):
            return projector.project(checkpoint["x"], checkpoint["y"])[0]
    return None


def _projected_half_extent(length, width, heading_deg, axis_x, axis_y):
    heading = math.radians(heading_deg)
    hx, hy = math.cos(heading), math.sin(heading)
    return (0.5 * length * abs(hx * axis_x + hy * axis_y)
            + 0.5 * width * abs(-hy * axis_x + hx * axis_y))


def _link_profile(mgeo, link_id, profiles):
    """Cache each link's own point list and cumulative arc length."""
    profile = profiles.get(link_id)
    if profile is None:
        points = mgeo.links[link_id].get("points", [])
        xy = [(float(point[0]), float(point[1])) for point in points]
        cumulative = [0.0]
        for a, b in zip(xy, xy[1:]):
            cumulative.append(cumulative[-1] + math.hypot(b[0] - a[0], b[1] - a[1]))
        profile = (xy, cumulative)
        profiles[link_id] = profile
    return profile


def _link_s(mgeo, link_id, x, y, profiles):
    """Arc-length position of (x, y) projected onto one link's own polyline."""
    xy, cumulative = _link_profile(mgeo, link_id, profiles)
    if len(xy) < 2:
        return 0.0
    best_s, best_distance_sq = 0.0, None
    for index in range(len(xy) - 1):
        ax, ay = xy[index]
        bx, by = xy[index + 1]
        dx, dy = bx - ax, by - ay
        length_sq = dx * dx + dy * dy
        if length_sq < 1e-9:
            continue
        t = max(0.0, min(1.0, ((x - ax) * dx + (y - ay) * dy) / length_sq))
        px, py = ax + t * dx, ay + t * dy
        distance_sq = (x - px) ** 2 + (y - py) ** 2
        if best_distance_sq is None or distance_sq < best_distance_sq:
            best_distance_sq = distance_sq
            best_s = cumulative[index] + t * math.sqrt(length_sq)
    return best_s


def _nearest_link_among(mgeo, x, y, candidates, profiles):
    """Closest of a restricted link set to (x, y), regardless of match radius.

    Two roads can cross or touch almost exactly (a ring link ending at the
    same junction a different approach road passes through), so an
    unconstrained nearest-link match can silently return the wrong one.
    """
    best_link, best_distance_sq = None, None
    for link_id in candidates:
        xy, _ = _link_profile(mgeo, link_id, profiles)
        for index in range(len(xy) - 1):
            ax, ay = xy[index]
            bx, by = xy[index + 1]
            dx, dy = bx - ax, by - ay
            length_sq = dx * dx + dy * dy
            if length_sq < 1e-9:
                continue
            t = max(0.0, min(1.0, ((x - ax) * dx + (y - ay) * dy) / length_sq))
            px, py = ax + t * dx, ay + t * dy
            distance_sq = (x - px) ** 2 + (y - py) ** 2
            if best_distance_sq is None or distance_sq < best_distance_sq:
                best_distance_sq = distance_sq
                best_link = link_id
    return best_link


def _forward_gap(mgeo, start_link, start_xy, target_link, target_xy,
                 profiles, path_cache, max_distance=160.0):
    """Directed arc-length gap from start to target along the MGeo link graph.

    None when the target is behind start, unreachable within max_distance, or
    either endpoint failed to resolve. Reuses MGeoMap.shortest_link_path (a
    plain node-distance Dijkstra) across a tiny local cache, since only a
    handful of distinct links recur across one roundabout traversal.
    """
    if start_link is None or target_link is None:
        return None
    s0 = _link_s(mgeo, start_link, start_xy[0], start_xy[1], profiles)
    s1 = _link_s(mgeo, target_link, target_xy[0], target_xy[1], profiles)
    if start_link == target_link:
        gap = s1 - s0
        return max(0.0, gap) if gap >= -1.0 else None
    key = (start_link, target_link)
    if key not in path_cache:
        try:
            path_cache[key] = mgeo.shortest_link_path(start_link, target_link)
        except ValueError:
            path_cache[key] = None
    path = path_cache[key]
    if path is None:
        return None
    gap = (_link_profile(mgeo, start_link, profiles)[1][-1] - s0) + s1
    for mid in path[1:-1]:
        gap += _link_profile(mgeo, mid, profiles)[1][-1]
    return gap if -1.0 <= gap <= max_distance else None


def enrich(frames, objects, route):
    """Mutate fresh feature tables with instruction inputs and provenance."""
    projector = RouteProjector(route["nav_waypoints"])
    mgeo = _mgeo(route)
    grouped = defaultdict(list)
    for obj in objects:
        grouped[obj["index"]].append(obj)
    zone = (route.get("zones") or {}).get("roundabout") or {}
    entry_s = _checkpoint_s(route, projector, zone.get("ego_entry_checkpoint"))
    merge_xy = zone.get("merge_point_xy")
    merge_s = (projector.project(*merge_xy)[0]
               if isinstance(merge_xy, (list, tuple)) and len(merge_xy) == 2 else None)
    circulating_links = set(zone.get("internal_loop_links") or ())
    link_profiles = {}
    link_path_cache = {}
    has_merge_xy = isinstance(merge_xy, (list, tuple)) and len(merge_xy) == 2
    # Restricted to the ring: the merge point also sits almost exactly on
    # Ego's own approach link where it meets the junction, and an
    # unconstrained nearest-link match picks that private road instead of the
    # ring link circulating traffic is actually travelling on — which then
    # makes every circulating vehicle's distance-to-merge unreachable (no
    # forward path from the ring back into Ego's own lane).
    merge_link_id = (_nearest_link_among(mgeo, merge_xy[0], merge_xy[1], circulating_links, link_profiles)
                     if has_merge_xy and mgeo is not None and circulating_links else None)
    previous = None
    last_heading = None
    for frame in frames:
        index = frame["index"]
        x, y = number(frame.get("ego_world_x_m")), number(frame.get("ego_world_y_m"))
        speed = number(frame.get("ego_speed_mps"))
        frame.update(
            ego_heading_deg=None, ego_heading_source=None, ego_accel_mps2=None,
            ego_stopped=None if speed is None else speed <= 0.2,
            ego_decelerating=None, ego_hard_braking=None,
            current_link_id=None, ego_reference_path_offset_m=None,
            ego_reference_heading_error_deg=None, ego_lateral_heading_component=None,
            avoidance_active=None,
            returning_to_reference_path=None, avoid_direction=None,
            left_avoid_candidate=None, right_avoid_candidate=None,
            distance_to_entry_m=None, distance_to_merge_m=None,
            in_entry_section=None, in_merge_section=None,
            in_circulating_lane=None, roundabout_phase=None,
            ego_merging_candidate=None, ego_following_candidate=None,
            circulating_vehicle_ids=[], nearest_circulating_vehicle_id=None,
            circulating_vehicle_distance_m=None, circulating_vehicle_speed_mps=None,
            circulating_vehicle_heading_deg=None, conflict_actor_id=None,
            conflict_distance_m=None, collision_risk=None,
            future_collision_predicted=None, lead_actor_id=None,
            lead_distance_m=None, lead_bumper_gap_m=None,
            lead_bumper_gap_verified=False, lead_speed_mps=None,
            relative_speed_mps=None, safe_following_gap_m=None,
            lead_gap_insufficient=None, lead_gap_source=None,
            relevant_obstacle_ids=[], nearest_obstacle_id=None,
            nearest_obstacle_distance_m=None, nearest_obstacle_side=None,
            obstacle_count=0, obstacle_layout=None, left_obstacle_ids=[],
            right_obstacle_ids=[], corridor_gap_m=None,
            corridor_gap_lower_bound_m=None,
            corridor_passable_candidate=None, obstacle_passed=None,
            relevant_pedestrian_ids=[], pedestrian_count=0,
            nearest_pedestrian_distance_m=None, pedestrian_side=None,
            pedestrian_crossing_direction=None, pedestrian_v_d_mps=None,
            pedestrian_path_d_m=None, pedestrian_on_conflict_path=None,
            pedestrian_clearance_m=None, all_relevant_pedestrians_clear=None,
            ego_yielding_candidate=None, resume_candidate=None,
            dynamic_in_yield_family=None,
            dynamic_yield_transition_start=None, dynamic_resume_transition_start=None)
        if previous is not None and previous["index"] + 1 == index:
            px, py = number(previous.get("ego_world_x_m")), number(previous.get("ego_world_y_m"))
            ps = number(previous.get("ego_speed_mps"))
            dt = (int(frame["lidar_source_ns"]) - int(previous["lidar_source_ns"])) / 1e9
            if x is not None and y is not None and px is not None and py is not None and 0 < dt <= 0.25:
                distance = math.hypot(x - px, y - py)
                if distance >= 0.15:
                    last_heading = math.degrees(math.atan2(y - py, x - px))
                    frame["ego_heading_deg"] = last_heading
                    frame["ego_heading_source"] = "backward_gt_ego_motion"
                if speed is not None and ps is not None:
                    frame["ego_accel_mps2"] = (speed - ps) / dt
        if frame["ego_heading_deg"] is None and last_heading is not None:
            frame["ego_heading_deg"] = last_heading
            frame["ego_heading_source"] = "held_prior_motion"
        accel = frame["ego_accel_mps2"]
        if accel is not None:
            frame["ego_decelerating"] = accel <= -0.5
            frame["ego_hard_braking"] = accel <= -3.0
        if x is not None and y is not None:
            _, d, _, tx, ty = projector.project(x, y)
            frame["ego_reference_path_offset_m"] = d
            heading = frame["ego_heading_deg"]
            frame["current_link_id"] = _link(mgeo, x, y, heading)
            if heading is not None:
                error = (heading - math.degrees(math.atan2(ty, tx)) + 180) % 360 - 180
                frame["ego_reference_heading_error_deg"] = error
                frame["ego_lateral_heading_component"] = (
                    -math.sin(math.radians(heading)) * tx + math.cos(math.radians(heading)) * ty)
            previous_d = (number(previous.get("ego_reference_path_offset_m"))
                         if previous is not None and previous["index"] + 1 == index else None)
            # The sign-of-heading proxy above holds near zero once Ego has
            # already straightened out mid-avoidance, even while the position
            # offset itself keeps growing -- so "returning" is judged from the
            # actual recorded offset shrinking frame to frame, not from
            # instantaneous heading angle.
            frame["returning_to_reference_path"] = bool(
                abs(d) > 0.8 and previous_d is not None and abs(d) < abs(previous_d) - 0.01)
            frame["avoidance_active"] = abs(d) > 0.8
            if frame["avoidance_active"]:
                # The side Ego is currently offset to IS the side it is avoiding
                # via -- this is the actual recorded GT displacement, not a
                # heading-angle proxy (a gentle, already-car-straightened drift
                # can hold near-zero heading error while the position offset
                # itself still grows steadily to one side).
                frame["avoid_direction"] = _side(d)
                if frame["avoid_direction"] == "center":
                    frame["avoid_direction"] = None
                # A same-frame, current-evidence candidate: Ego's own sealed
                # position is, right now, actually displaced toward this side
                # and has not yet started converging back to the reference
                # path. This does not certify the far side of the obstacle is
                # clear of anything else -- only that the recorded trajectory
                # itself is consistent with avoiding this way at this instant.
                if frame["avoid_direction"] is not None and not frame["returning_to_reference_path"]:
                    frame[frame["avoid_direction"] + "_avoid_candidate"] = True
        if frame["zone_id"] == "roundabout":
            s = number(frame.get("ego_path_s_m"))
            if s is not None:
                frame["distance_to_entry_m"] = None if entry_s is None else entry_s - s
                frame["distance_to_merge_m"] = None if merge_s is None else merge_s - s
            link = frame["current_link_id"]
            if link is not None:
                frame["in_circulating_lane"] = link in circulating_links
            if s is not None and merge_s is not None:
                frame["in_entry_section"] = s < merge_s - 5.0
                frame["in_merge_section"] = abs(s - merge_s) <= 5.0
            # A matched ring link is direct current topology evidence. The
            # route projection can still be before its merge anchor while
            # Ego is already on that ring; treating its same-lane lead as
            # incoming entry traffic then invents a yield conflict.
            if frame["in_circulating_lane"] is True:
                frame["roundabout_phase"] = "circulating"
                frame["in_entry_section"] = False
                frame["in_merge_section"] = False
            elif s is not None and merge_s is not None:
                if frame["in_merge_section"]:
                    frame["roundabout_phase"] = "merge"
                elif frame["in_entry_section"]:
                    frame["roundabout_phase"] = "approach" if entry_s is None or s < entry_s else "entry"
                else:
                    frame["roundabout_phase"] = "exit"
            frame["ego_merging_candidate"] = (
                frame["in_merge_section"] is True and speed is not None and speed > 0.5
                and frame["ego_reference_heading_error_deg"] is not None
                and abs(frame["ego_reference_heading_error_deg"]) <= 30)
        for obj in grouped[index]:
            obj["actor_heading_deg"] = number(obj.get("actor_heading_deg"))
            ox, oy = number(obj.get("world_x_m")), number(obj.get("world_y_m"))
            obj["actor_link_id"] = _link(mgeo, ox, oy, obj["actor_heading_deg"])
            obj["actor_on_circulating_lane"] = (
                None if obj["actor_link_id"] is None else
                obj["actor_link_id"] in circulating_links)
            obj["is_lead_vehicle"] = False
            obj["is_conflict_vehicle"] = None
            obj["lead_bumper_gap_m"] = None
            obj["lead_bumper_gap_verified"] = False
            obj["actor_to_merge_m"] = None
            obj["rule_geometry_usable"] = (
                ox is not None and oy is not None
                and number(obj.get("actor_width_m")) is not None
                and number(obj.get("actor_width_m")) > 0
                and number(obj.get("actor_length_m")) is not None
                and number(obj.get("actor_length_m")) > 0
                and set(obj.get("invalid_reasons") or ()) <= {
                    "outside_training_bev_or_unknown", "object_geometry_not_candidate"})
            obj["distance_to_ego_m"] = (
                math.hypot(ox - x, oy - y) if None not in (ox, oy, x, y) else None)
            obj["target_side"] = _side(obj.get("left_m"))
        nearby = grouped[index]
        if frame["zone_id"] == "static_obstacle":
            candidates = []
            behind = []
            for obj in nearby:
                if obj.get("kind") != "obstacle":
                    continue
                delta, d = number(obj.get("delta_s_m")), number(obj.get("path_d_m"))
                width, length = number(obj.get("actor_width_m")), number(obj.get("actor_length_m"))
                if None in (delta, d, width, length):
                    continue
                radius = math.hypot(width, length) / 2
                # An obstacle can bound the corridor while its center is
                # outside the nominal path conflict band.
                if abs(d) > 4.0 + radius:
                    continue
                if 0 < delta <= 35:
                    candidates.append(obj)
                elif -15 <= delta <= 0:
                    behind.append(obj)
            candidates.sort(key=lambda row: row["delta_s_m"])
            frame["relevant_obstacle_ids"] = [actor_id(row) for row in candidates]
            frame["obstacle_count"] = len(candidates)
            frame["obstacle_passed"] = bool(behind) and not candidates
            frame["left_obstacle_ids"] = [actor_id(row) for row in candidates if row["path_d_m"] > 0]
            frame["right_obstacle_ids"] = [actor_id(row) for row in candidates if row["path_d_m"] < 0]
            if candidates:
                first = candidates[0]
                frame["nearest_obstacle_id"] = actor_id(first)
                frame["nearest_obstacle_distance_m"] = number(first["delta_s_m"])
                frame["nearest_obstacle_side"] = _side(first["path_d_m"])
                frame["obstacle_layout"] = (
                    "single" if len(candidates) == 1 else
                    "both_sides" if frame["left_obstacle_ids"] and frame["right_obstacle_ids"]
                    else "multiple")
            if frame["left_obstacle_ids"] and frame["right_obstacle_ids"]:
                left = next(row for row in candidates if actor_id(row) == frame["left_obstacle_ids"][0])
                right = next(row for row in candidates if actor_id(row) == frame["right_obstacle_ids"][0])
                if abs(left["delta_s_m"] - right["delta_s_m"]) <= 5:
                    tx, ty = number(left.get("path_tangent_x")), number(left.get("path_tangent_y"))
                    if tx is not None and ty is not None:
                        separation = abs((right["world_x_m"] - left["world_x_m"]) * -ty
                                         + (right["world_y_m"] - left["world_y_m"]) * tx)
                        lower = separation - sum(math.hypot(row["actor_width_m"], row["actor_length_m"]) / 2
                                                 for row in (left, right))
                        frame["corridor_gap_m"] = max(0.0, lower)
                        frame["corridor_gap_lower_bound_m"] = max(0.0, lower)
                        frame["corridor_passable_candidate"] = lower > 3.0
        elif frame["zone_id"] == "dynamic_obstacle":
            people = [row for row in nearby if row.get("kind") == "pedestrian"
                      and row.get("distance_to_ego_m") is not None
                      and row["distance_to_ego_m"] <= 35
                      and number(row.get("delta_s_m")) is not None
                      and -5 <= row["delta_s_m"] <= 35]
            for person in people:
                d, width, length, vd = (number(person.get(key)) for key in
                                        ("path_d_m", "actor_width_m",
                                         "actor_length_m", "v_d_mps"))
                clearance = (None if None in (d, width, length) else
                             abs(d) - math.hypot(width, length) / 2 - 1.0)
                person["pedestrian_clearance_m"] = clearance
                person["pedestrian_on_conflict_path"] = (
                    None if clearance is None else clearance <= 0)
                person["pedestrian_crossing_direction"] = (
                    "L2R" if vd is not None and vd < -0.2 else
                    "R2L" if vd is not None and vd > 0.2 else
                    "stationary" if vd is not None else "unknown")
            people.sort(key=lambda row: row["distance_to_ego_m"])
            frame["relevant_pedestrian_ids"] = [actor_id(row) for row in people]
            frame["pedestrian_count"] = len(people)
            frame["all_relevant_pedestrians_clear"] = not people
            if people:
                first = people[0]
                vd, d = number(first.get("v_d_mps")), number(first.get("path_d_m"))
                width = number(first.get("actor_width_m"))
                length = number(first.get("actor_length_m"))
                clearance = (None if None in (d, width, length) else
                             abs(d) - math.hypot(width, length) / 2 - 1.0)
                direction = ("L2R" if vd < -0.2 else "R2L" if vd > 0.2 else
                             "stationary" if vd is not None else "unknown")
                frame.update(nearest_pedestrian_distance_m=first["distance_to_ego_m"],
                             pedestrian_side=_side(first.get("left_m")),
                             pedestrian_crossing_direction=direction,
                             pedestrian_v_d_mps=vd, pedestrian_path_d_m=d,
                             pedestrian_clearance_m=clearance,
                             pedestrian_on_conflict_path=None if clearance is None else clearance <= 0,
                             all_relevant_pedestrians_clear=all(
                                 number(row.get("path_d_m")) is not None
                                 and number(row.get("actor_width_m")) is not None
                                 and number(row.get("actor_length_m")) is not None
                                 and abs(row["path_d_m"]) - math.hypot(
                                     row["actor_width_m"], row["actor_length_m"]) / 2 - 1.0 > 0
                                 for row in people))
                frame["pedestrian_on_conflict_path"] = (
                    True if any(row["pedestrian_on_conflict_path"] is True
                                for row in people) else
                    None if any(row["pedestrian_on_conflict_path"] is None
                                for row in people) else False)
            frame["ego_yielding_candidate"] = (
                frame["ego_decelerating"] is True or
                (speed is not None and 0.2 < speed <= 2.0))
        elif frame["zone_id"] == "roundabout":
            potential = [row for row in nearby if row.get("kind") in ("npc", "vehicle")
                         and row.get("distance_to_ego_m") is not None
                         and row["distance_to_ego_m"] <= 80]
            circulating = [row for row in potential
                           if row.get("actor_link_id") in circulating_links]
            circulating.sort(key=lambda row: row["distance_to_ego_m"])
            frame["circulating_vehicle_ids"] = [actor_id(row) for row in circulating]
            if circulating:
                first = circulating[0]
                frame.update(nearest_circulating_vehicle_id=actor_id(first),
                             circulating_vehicle_distance_m=first["distance_to_ego_m"],
                             circulating_vehicle_speed_mps=number(first.get("speed_mps")),
                             circulating_vehicle_heading_deg=first.get("actor_heading_deg"))

            ego_link = frame["current_link_id"]
            # Only the approach/merge phases need a yield decision; once
            # circulating or past the merge there is nothing left to enter.
            yield_phase = frame["roundabout_phase"] in ("approach", "entry", "merge")
            ego_eta_s = None
            if yield_phase:
                ego_to_merge = frame.get("distance_to_merge_m")
                if (ego_to_merge is not None and ego_to_merge >= -5.0
                        and speed is not None and speed > 0.3):
                    ego_eta_s = max(0.0, ego_to_merge) / speed

            conflicts, unresolved = [], []
            for row in circulating:
                ox, oy = number(row.get("world_x_m")), number(row.get("world_y_m"))
                actor_link = row.get("actor_link_id")
                to_merge = (_forward_gap(mgeo, actor_link, (ox, oy), merge_link_id, merge_xy,
                                         link_profiles, link_path_cache)
                            if None not in (ox, oy, actor_link, merge_link_id) else None)
                row["actor_to_merge_m"] = to_merge
                if not yield_phase or ego_link is None or to_merge is None:
                    # to_merge is None for a circulating vehicle the graph
                    # cannot reach forward from here (e.g. already well past
                    # the merge, looping back) — not a live entry conflict.
                    continue
                actor_speed = number(row.get("speed_mps"))
                if actor_speed is None:
                    unresolved.append(row)
                    continue
                if actor_speed > 0.3:
                    actor_eta_s = to_merge / actor_speed
                elif to_merge <= 3.0:
                    actor_eta_s = 0.0
                else:
                    continue  # stopped well short of the merge: not a live threat
                is_conflict = (actor_eta_s <= 4.0 if ego_eta_s is None
                              else abs(ego_eta_s - actor_eta_s) <= 3.0)
                if is_conflict:
                    conflicts.append((to_merge, row))
            if (frame["roundabout_phase"] is None or mgeo is None
                    or not circulating_links or merge_link_id is None or ego_link is None):
                # Missing topology/phase is not evidence of a clear entry.
                # In particular, None must not become False simply because
                # it is not one of the named approach/entry/merge phases.
                frame["collision_risk"] = None
            elif not yield_phase:
                frame["collision_risk"] = False
            elif unresolved:
                frame["collision_risk"] = None
            elif conflicts:
                conflicts.sort(key=lambda item: item[0])
                distance, conflict_row = conflicts[0]
                conflict_row["is_conflict_vehicle"] = True
                frame["collision_risk"] = True
                frame["conflict_actor_id"] = actor_id(conflict_row)
                frame["conflict_distance_m"] = distance
            else:
                frame["collision_risk"] = False

            if ego_link is not None and x is not None and y is not None:
                ego_heading = frame["ego_heading_deg"]
                tx = math.cos(math.radians(ego_heading)) if ego_heading is not None else 1.0
                ty = math.sin(math.radians(ego_heading)) if ego_heading is not None else 0.0
                leads = []
                for row in circulating:
                    ox, oy = number(row.get("world_x_m")), number(row.get("world_y_m"))
                    actor_link = row.get("actor_link_id")
                    heading = row.get("actor_heading_deg")
                    length, width = number(row.get("actor_length_m")), number(row.get("actor_width_m"))
                    if None in (ox, oy, actor_link, length, width) or not row["rule_geometry_usable"]:
                        continue
                    if ego_heading is not None and heading is not None:
                        error = abs((heading - ego_heading + 180) % 360 - 180)
                        # A curved ring shifts instantaneous heading faster than
                        # a straight road; a true lead can still differ by this
                        # much mid-curve, so this is looser than a straight-lane
                        # following check would use.
                        if error > 70:
                            continue
                    # Lane-graph distance (not a straight-line projection) so a
                    # lead just ahead on the next connected link, or on a
                    # different link than Ego's own approach link, is found too.
                    along = _forward_gap(mgeo, ego_link, (x, y), actor_link, (ox, oy),
                                         link_profiles, link_path_cache)
                    if along is None or along > 60:
                        continue
                    gap = along - _projected_half_extent(4.635, 2.0, ego_heading or 0.0, tx, ty)
                    gap -= _projected_half_extent(length, width, heading or 0.0, tx, ty)
                    leads.append((gap, along, row))
                if leads:
                    gap, along, lead = min(leads, key=lambda item: item[0])
                    lead["is_lead_vehicle"] = True
                    lead["lead_bumper_gap_m"] = max(0.0, gap)
                    lead["lead_bumper_gap_verified"] = True
                    frame.update(lead_actor_id=actor_id(lead), lead_distance_m=along,
                                 lead_bumper_gap_m=max(0.0, gap), lead_bumper_gap_verified=True,
                                 lead_speed_mps=number(lead.get("speed_mps")),
                                 safe_following_gap_m=None if speed is None else 3.0 + speed * 1.2,
                                 lead_gap_source="lane_graph_forward_distance")
                    if speed is not None and frame["lead_speed_mps"] is not None:
                        frame["relative_speed_mps"] = speed - frame["lead_speed_mps"]
                    if frame["safe_following_gap_m"] is not None:
                        frame["lead_gap_insufficient"] = gap < frame["safe_following_gap_m"]
                    frame["ego_following_candidate"] = True
        previous = frame
    saw_dynamic_conflict = False
    prior_dynamic_index = None
    for frame in frames:
        if (frame["zone_id"] != "dynamic_obstacle"
                or frame.get("valid") is not True
                or frame.get("object_inventory_usable") is not True):
            saw_dynamic_conflict = False
            prior_dynamic_index = None
            continue
        if prior_dynamic_index is not None and frame["index"] != prior_dynamic_index + 1:
            saw_dynamic_conflict = False
        prior_dynamic_index = frame["index"]
        if frame.get("pedestrian_on_conflict_path") is True:
            saw_dynamic_conflict = True
        speed = number(frame.get("ego_speed_mps"))
        frame["resume_candidate"] = bool(
            saw_dynamic_conflict and
            frame.get("all_relevant_pedestrians_clear") is True and
            speed is not None and speed <= 2.0)
        if frame["resume_candidate"] is False and (
                frame.get("all_relevant_pedestrians_clear") is True and
                speed is not None and speed > 2.0):
            saw_dynamic_conflict = False
    # Single-shot transition markers (true only on the first frame of a
    # change), independent of saw_dynamic_conflict/resume_candidate above,
    # which both stay true across the whole multi-frame window: dynamic_
    # obstacle_rule uses these to tell "just started yielding/resuming" apart
    # from "still yielding/already resumed".
    prev_in_yield_family = False
    prev_speed = None
    prior_dynamic_index = None
    for frame in frames:
        if (frame["zone_id"] != "dynamic_obstacle"
                or frame.get("valid") is not True
                or frame.get("object_inventory_usable") is not True):
            prev_in_yield_family = False
            prev_speed = None
            prior_dynamic_index = None
            continue
        if prior_dynamic_index is not None and frame["index"] != prior_dynamic_index + 1:
            prev_in_yield_family = False
            prev_speed = None
        prior_dynamic_index = frame["index"]
        conflict = frame.get("pedestrian_on_conflict_path")
        stopped = frame.get("ego_stopped")
        speed = number(frame.get("ego_speed_mps"))
        decel_or_yield = (frame.get("ego_decelerating") is True
                          or frame.get("ego_yielding_candidate") is True)
        strict_yield_now = bool(conflict is True and (
            stopped is True or (speed is not None and speed > 0.2 and decel_or_yield)))
        # ego_decelerating is a single-frame backward speed difference, so it
        # can blip above the threshold for one frame during an otherwise
        # continuous, monotonic slow-down. Don't drop Ego out of an already
        # -active yield episode over one noisy frame as long as it is still
        # not speeding back up.
        sustained_yield_now = bool(
            not strict_yield_now and prev_in_yield_family and conflict is True
            and speed is not None and prev_speed is not None and speed <= prev_speed)
        in_yield_now = strict_yield_now or sustained_yield_now
        frame["dynamic_in_yield_family"] = in_yield_now
        frame["dynamic_yield_transition_start"] = bool(in_yield_now and not prev_in_yield_family)
        frame["dynamic_resume_transition_start"] = bool(
            conflict is False and prev_in_yield_family
            and speed is not None and speed <= 2.0)
        prev_in_yield_family = in_yield_now
        prev_speed = speed
    from .highway_features import enrich_highway
    enrich_highway(frames, objects, route, mgeo)
    return frames, objects


def attach_static_avoid_episode(frames, hold_frames=5):
    """Mark the single-shot start, the initial-instruction hold window, and
    the ongoing span of a static-obstacle avoidance episode, driven by
    Expert's own follow/rejoined label transitions (expert_static_phase.py's
    RAW_STATE_TO_LABEL) -- the same state-machine debounce pattern as the
    dynamic_in_yield_family loop above, applied to Expert's label instead of
    a GT motion proxy, since static's own avoidance_active (GT lateral
    offset) can lag or lead the label by a frame or more and would otherwise
    force every held-FOLLOW frame to repeat a single "start avoiding"
    instruction (static_obstacle_rule previously returned "avoid"
    unconditionally for every expert_label == "follow" and blocking frame).

    `hold_frames` (default 5) is how many frames from and including the
    start frame still say "avoid" (회피해라) rather than "avoid_continue"
    (회피를 진행해라) -- e.g. a start at index 12 holds through index 16
    (12, 13, 14, 15, 16: 5 frames) and switches at index 17.

    Requires frame["expert_label"] already set (call after
    expert_static_phase.attach_static_phase); a no-op pass-through otherwise
    leaves all three new fields False.
    """
    active = False
    prev_label = None
    prior_index = None
    frames_since_start = 0
    for frame in frames:
        if frame.get("zone_id") != "static_obstacle" or frame.get("valid") is not True:
            active = False
            prev_label = None
            prior_index = None
            frames_since_start = 0
            frame["static_avoid_episode_active"] = False
            frame["static_avoid_transition_start"] = False
            frame["static_avoid_initial_hold"] = False
            continue
        if prior_index is not None and frame["index"] != prior_index + 1:
            active = False
            prev_label = None
            frames_since_start = 0
        prior_index = frame["index"]
        label = frame.get("expert_label")
        starting = bool(label == "follow" and prev_label != "follow")
        if starting:
            active = True
            frames_since_start = 1
        elif label != "follow":
            active = False
            frames_since_start = 0
        elif active:
            frames_since_start += 1
        frame["static_avoid_transition_start"] = starting
        frame["static_avoid_episode_active"] = active
        frame["static_avoid_initial_hold"] = bool(active and frames_since_start <= hold_frames)
        prev_label = label
    return frames
