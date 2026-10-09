"""Current highway motion, lane topology and nearby traffic from GT and MGeo.

A geometric clearance is an initial check, never a trajectory safety verdict.
"""
from collections import defaultdict
import math

from .highway_rules import highway_rule
from .instruction_features import (
    _forward_gap, _link_profile, _projected_half_extent, number,
)

RESPONDING_ACC_STATES = {"hard_braking", "stopped", "yielding", "decelerating"}
EGO_HALF_WIDTH_M = 1.0  # Captured Expert VEHICLE_WIDTH=2.0 m.
LANE_ASSOCIATION_MAX_HEADING_ERROR_DEG = 15.0


def _projection(mgeo, link_id, x, y, profiles):
    if link_id not in mgeo.links or x is None or y is None:
        return None
    points, cumulative = _link_profile(mgeo, link_id, profiles)
    best = None
    for i, (a, b) in enumerate(zip(points, points[1:])):
        dx, dy = b[0] - a[0], b[1] - a[1]
        length = math.hypot(dx, dy)
        if length < 1e-6:
            continue
        t = max(0.0, min(1.0, ((x-a[0])*dx + (y-a[1])*dy)/length**2))
        px, py = a[0]+t*dx, a[1]+t*dy
        distance = math.hypot(x-px, y-py)
        signed = (-(x-px)*dy + (y-py)*dx)/length
        candidate = (distance, cumulative[i]+t*length, signed, dx/length, dy/length)
        if best is None or candidate[0] < best[0]:
            best = candidate
    return best


def _neighbor(mgeo, link_id, side):
    field = side + "_lane_change_dst_link_idx"
    value = mgeo.links[link_id].get(field)
    return str(value) if value is not None and str(value) in mgeo.links else None


def _lane_id(mgeo, link_id):
    if link_id is None or link_id not in mgeo.links:
        return None
    value = mgeo.links[link_id].get("ego_lane")
    return "L" + str(value) if value is not None else link_id


def _heading_error(heading, projection):
    if heading is None or projection is None:
        return None
    return (heading - math.degrees(math.atan2(projection[4], projection[3])) + 180) % 360 - 180


def _association_neighbors(mgeo):
    """Only different lanes on one road sharing an actual endpoint node."""
    endpoints = defaultdict(set)
    for identity, link in mgeo.links.items():
        road = link.get("road_id")
        if road is None or road == "":
            continue
        for end in ("from_node_idx", "to_node_idx"):
            node = link.get(end)
            if node is not None and node != "":
                endpoints[(str(road), end, str(node))].add(identity)
    neighbors = defaultdict(set)
    for candidates in endpoints.values():
        for identity in candidates:
            neighbors[identity].update(other for other in candidates
                                       if _lane_id(mgeo, other) != _lane_id(mgeo, identity))
    return neighbors


def current_lane_association_ambiguity(mgeo, link, x, y, heading, profiles, neighbors=None):
    """Detect an unresolved current body overlap; never select a truer lane.

    The 1 m distance guard is half the captured ego width, not a confidence
    score tuned to neighboring labels. Both links must also be compatible with
    current recorded heading and share one endpoint on the same road.
    """
    receipt = {"schema": "current_highway_lane_association_v1",
               "selected_link_id": link, "ego_half_width_m": EGO_HALF_WIDTH_M,
               "ego_width_source": "captured_Expert_VEHICLE_WIDTH_2.0_m",
               "max_heading_error_deg": LANE_ASSOCIATION_MAX_HEADING_ERROR_DEG,
               "source": "current_recorded_gt_ego_xy_and_capture_verified_mgeo",
               "uses_future_ego": False, "uses_neighbor_sentence": False,
               "physical_lane_boundary_certified": False, "candidates": []}
    if link not in mgeo.links or any(number(v) is None for v in (x, y, heading)):
        return False, receipt
    projection = _projection(mgeo, link, x, y, profiles)
    error = _heading_error(heading, projection)
    if (projection is None or projection[0] > EGO_HALF_WIDTH_M
            or error is None or abs(error) > LANE_ASSOCIATION_MAX_HEADING_ERROR_DEG):
        return False, receipt
    selected = mgeo.links[link]
    receipt["candidates"].append({"link_id": link, "lane_id": _lane_id(mgeo, link),
        "distance_m": projection[0], "signed_left_m": projection[2],
        "heading_error_deg": error, "selected": True})
    neighbors = _association_neighbors(mgeo) if neighbors is None else neighbors
    for other in sorted(neighbors.get(link, ())):
        target = mgeo.links[other]
        # Recheck source relationships even when callers provide a cache.
        same_road = (selected.get("road_id") is not None and selected.get("road_id") != ""
                     and str(selected.get("road_id")) == str(target.get("road_id")))
        shared = [end for end in ("from_node_idx", "to_node_idx")
                  if selected.get(end) is not None and selected.get(end) != ""
                  and selected.get(end) == target.get(end)]
        if not same_road or not shared or _lane_id(mgeo, other) == _lane_id(mgeo, link):
            continue
        candidate = _projection(mgeo, other, x, y, profiles)
        error = _heading_error(heading, candidate)
        if (candidate is None or candidate[0] > EGO_HALF_WIDTH_M
                or error is None or abs(error) > LANE_ASSOCIATION_MAX_HEADING_ERROR_DEG):
            continue
        receipt["candidates"].append({"link_id": other, "lane_id": _lane_id(mgeo, other),
            "distance_m": candidate[0], "signed_left_m": candidate[2],
            "heading_error_deg": error, "selected": False,
            "shared_endpoint_fields": shared, "road_id": selected.get("road_id")})
    return len(receipt["candidates"]) > 1, receipt


def _bumper_gap(center_gap, heading, actor, tx, ty):
    length, width = number(actor.get("actor_length_m")), number(actor.get("actor_width_m"))
    actor_heading = number(actor.get("actor_heading_deg"))
    if None in (heading, length, width, actor_heading):
        return None
    ego_half = _projected_half_extent(4.635, 2.0, heading, tx, ty)
    actor_half = _projected_half_extent(length, width, actor_heading, tx, ty)
    return max(0.0, center_gap - ego_half - actor_half)


def _target_traffic(mgeo, target_link, ego_xy, heading, actors, profiles, paths):
    ep = _projection(mgeo, target_link, *ego_xy, profiles)
    if ep is None:
        return None, None
    lane = _lane_id(mgeo, target_link)
    front, rear = [], []
    for actor in actors:
        actor_link = actor.get("actor_link_id")
        if _lane_id(mgeo, actor_link) != lane:
            continue
        ax, ay = number(actor.get("world_x_m")), number(actor.get("world_y_m"))
        if None in (ax, ay):
            continue
        forward = _forward_gap(mgeo, target_link, ego_xy, actor_link,
                               (ax, ay), profiles, paths, 85)
        backward = _forward_gap(mgeo, actor_link, (ax, ay), target_link,
                                ego_xy, profiles, paths, 85)
        if forward is not None and 0 <= forward <= 80:
            front.append((forward, _bumper_gap(forward, heading, actor, ep[3], ep[4]), actor))
        elif backward is not None and 0 <= backward <= 80:
            rear.append((backward, _bumper_gap(backward, heading, actor, ep[3], ep[4]), actor))
    return (min(front, default=None, key=lambda item:item[0]),
            min(rear, default=None, key=lambda item:item[0]))


def enrich_highway(frames, objects, route, mgeo):
    grouped = defaultdict(list)
    for row in objects:
        grouped[row["index"]].append(row)
    profiles, paths = {}, {}
    association_neighbors = _association_neighbors(mgeo) if mgeo is not None else {}
    maneuver = None
    previous = None
    prev_acc_responding = False
    for frame in frames:
        if frame.get("zone_id") != "highway":
            maneuver = None
            previous = None
            prev_acc_responding = False
            continue
        frame.update(
            ego_current_link_id=frame.get("current_link_id"), ego_current_lane_id=None,
            ego_lane_center_offset_m=None, ego_heading_error_deg=None,
            left_lane_link_id=None, right_lane_link_id=None,
            left_lane_exists=None, right_lane_exists=None,
            left_lane_change_allowed=None, right_lane_change_allowed=None,
            lane_change_active=None, lane_change_direction=None,
            source_lane_id=None, target_lane_id=None, target_lane_link_id=None,
            lane_change_progress=None, lane_change_progress_ratio=None,
            source_lane_center_offset_m=None, target_lane_center_offset_m=None,
            lane_change_completed=False, lane_change_aborting=False,
            return_direction=None, highway_lateral_step_m=None,
            highway_lane_association_ambiguous=False,
            highway_lane_association_evidence=None,
            lead_actor_id=None, lead_distance_m=None,
            lead_bumper_gap_m=None, lead_speed_mps=None,
            relative_speed_to_lead_mps=None, lead_stopped=None, lead_slow=None,
            lead_ttc_s=None, following_gap_insufficient=None,
            target_front_actor_id=None, target_front_gap_m=None,
            target_front_bumper_gap_m=None, target_front_speed_mps=None,
            target_rear_actor_id=None, target_rear_gap_m=None,
            target_rear_bumper_gap_m=None, target_rear_speed_mps=None,
            target_front_clearance_ok=None, target_rear_clearance_ok=None,
            target_lane_initially_safe=None, target_lane_collision_safe=None,
            merging_actor_id=None, merging_actor_side=None,
            merging_actor_distance_m=None, merging_actor_link_id=None,
            merging_actor_target_lane_id=None, merging_actor_v_d_mps=None,
            is_merging_into_ego_lane=None, merge_conflict=None,
            yield_to_merging_vehicle=None, target_actor_id=None,
            target_kind=None, target_position_text=None,
            target_distance_m=None, response_reason=None,
            highway_acc_prev_responding=None)
        if mgeo is None or frame.get("valid") is not True:
            previous = None
            maneuver = None
            prev_acc_responding = False
            continue
        x, y = number(frame.get("ego_world_x_m")), number(frame.get("ego_world_y_m"))
        link = frame.get("current_link_id")
        if None in (x, y) or link not in mgeo.links:
            previous = None
            maneuver = None
            prev_acc_responding = False
            continue
        heading = number(frame.get("ego_heading_deg"))
        speed = number(frame.get("ego_speed_mps"))
        projection = _projection(mgeo, link, x, y, profiles)
        if projection is None:
            previous = None
            maneuver = None
            prev_acc_responding = False
            continue
        frame["ego_current_lane_id"] = _lane_id(mgeo, link)
        frame["ego_lane_center_offset_m"] = projection[2]
        frame["ego_heading_error_deg"] = _heading_error(heading, projection)
        ambiguous, association = current_lane_association_ambiguity(
            mgeo, link, x, y, heading, profiles, association_neighbors)
        association.update(index=frame["index"], lidar_source_ns=frame.get("lidar_source_ns"))
        frame["highway_lane_association_ambiguous"] = ambiguous
        frame["highway_lane_association_evidence"] = association
        for side in ("left", "right"):
            neighbor = _neighbor(mgeo, link, side)
            frame[side+"_lane_link_id"] = neighbor
            frame[side+"_lane_exists"] = neighbor is not None
            frame[side+"_lane_change_allowed"] = bool(
                neighbor and mgeo.links[link].get("can_move_"+side+"_lane"))
        consecutive = (previous is not None and previous["index"]+1 == frame["index"])
        if not consecutive:
            maneuver = None
            prev_acc_responding = False
        motion = None
        if consecutive:
            px, py = number(previous.get("ego_world_x_m")), number(previous.get("ego_world_y_m"))
            if None not in (px, py):
                # Positive is left of the current link tangent.
                motion = -(x-px)*projection[4] + (y-py)*projection[3]
        prior_motion = (number(previous.get("highway_lateral_step_m"))
                        if consecutive else None)
        start_motion = (motion is not None and
                        (abs(motion) >= 0.12 or
                         (abs(motion) >= 0.04 and prior_motion is not None and
                          abs(prior_motion) >= 0.04 and motion*prior_motion > 0)))
        if previous is not None and previous.get("lane_change_completed") is True:
            start_motion = False
        if maneuver is None and start_motion:
            side = "left" if motion > 0 else "right"
            neighbor = _neighbor(mgeo, link, side)
            link_source = link
            if previous is not None:
                prior_link = previous.get("current_link_id")
                if prior_link in mgeo.links and _neighbor(mgeo, prior_link, side) == link:
                    neighbor = link
                    link_source = prior_link
            if neighbor is not None:
                source_p = _projection(mgeo, link_source, x, y, profiles)
                target_p = _projection(mgeo, neighbor, x, y, profiles)
                if source_p and target_p and source_p[0] <= 2.5 and target_p[0] <= 4.5:
                    maneuver = {"source": link_source, "target": neighbor,
                                "direction": side, "last_ratio": 0.0}
        if maneuver is not None:
            source, target = maneuver["source"], maneuver["target"]
            sp = _projection(mgeo, source, x, y, profiles)
            tp = _projection(mgeo, target, x, y, profiles)
            if sp is None or tp is None:
                maneuver = None
            else:
                width = math.hypot(tp[0], sp[0])
                ratio = sp[0] / (sp[0]+tp[0]) if sp[0]+tp[0] > 0.1 else 0.5
                frame["source_lane_id"] = _lane_id(mgeo, source)
                frame["target_lane_id"] = _lane_id(mgeo, target)
                frame["target_lane_link_id"] = target
                frame["source_lane_center_offset_m"] = sp[2]
                frame["target_lane_center_offset_m"] = tp[2]
                frame["lane_change_progress_ratio"] = round(ratio, 3)
                frame["lane_change_progress"] = (
                    "source_lane" if ratio < 0.25 else
                    "target_lane" if ratio > 0.75 else "between_lanes")
                reversal = (ratio < maneuver["last_ratio"]-0.01 and
                            motion is not None and abs(motion) >= 0.04)
                if reversal:
                    maneuver["aborting"] = True
                if maneuver.get("aborting"):
                    frame["lane_change_aborting"] = True
                    frame["return_direction"] = (
                        "right" if maneuver["direction"] == "left" else "left")
                frame["lane_change_direction"] = (
                    frame["return_direction"] or maneuver["direction"])
                frame["lane_change_active"] = True
                if not maneuver.get("aborting") and ratio > 0.8 and tp[0] <= 0.55:
                    maneuver["target_stable"] = maneuver.get("target_stable", 0)+1
                    if maneuver["target_stable"] >= 2:
                        frame["lane_change_completed"] = True
                        frame["lane_change_active"] = False
                        frame["lane_change_direction"] = None
                        maneuver = None
                elif maneuver.get("aborting") and ratio < 0.2 and sp[0] <= 0.55:
                    maneuver["source_stable"] = maneuver.get("source_stable", 0)+1
                    if maneuver["source_stable"] >= 2:
                        frame["lane_change_completed"] = True
                        frame["lane_change_active"] = False
                        frame["lane_change_direction"] = None
                        maneuver = None
                else:
                    maneuver["target_stable"] = 0
                    maneuver["source_stable"] = 0
                if maneuver is not None:
                    maneuver["last_ratio"] = ratio
        if maneuver is None and frame["lane_change_active"] is None:
            # A centered, aligned car is positively known to be lane keeping.
            if projection[0] <= 0.7 and frame["ego_heading_error_deg"] is not None:
                if abs(frame["ego_heading_error_deg"]) <= 12 and not start_motion:
                    frame["lane_change_active"] = False
        rows = grouped[frame["index"]]
        for actor in rows:
            actor["highway_lane_id"] = _lane_id(mgeo, actor.get("actor_link_id"))
        ego_xy = (x, y)
        relevant = []
        unresolved = False
        for actor in rows:
            if actor.get("kind") != "npc":
                continue
            distance = number(actor.get("distance_to_ego_m"))
            if distance is None or distance > 85:
                continue
            if actor.get("rule_geometry_usable") is not True or actor.get("actor_link_id") is None:
                unresolved = True
                continue
            relevant.append(actor)
        frame["highway_nearby_geometry_unverified"] = unresolved
        lead_candidates = []
        for actor in relevant:
            actor_link = actor["actor_link_id"]
            if _lane_id(mgeo, actor_link) != frame["ego_current_lane_id"]:
                continue
            ax, ay = number(actor.get("world_x_m")), number(actor.get("world_y_m"))
            if None in (ax, ay):
                continue
            gap = _forward_gap(mgeo, link, ego_xy, actor_link, (ax, ay), profiles, paths, 85)
            if gap is not None and 0 < gap <= 80:
                lead_candidates.append((gap, actor))
        if lead_candidates:
            center_gap, lead = min(lead_candidates, key=lambda item: item[0])
            bumper = _bumper_gap(center_gap, heading, lead, projection[3], projection[4])
            lead_speed = number(lead.get("speed_mps"))
            frame.update(lead_actor_id=lead.get("actor_id"), lead_distance_m=center_gap,
                         lead_bumper_gap_m=bumper, lead_speed_mps=lead_speed,
                         lead_stopped=None if lead_speed is None else lead_speed <= 0.2,
                         lead_slow=None if speed is None or lead_speed is None else lead_speed < speed-1.0)
            if speed is not None and lead_speed is not None:
                closing = speed-lead_speed
                frame["relative_speed_to_lead_mps"] = closing
                frame["lead_ttc_s"] = (bumper/closing if bumper is not None and closing > 0.1 else None)
            if bumper is not None and speed is not None:
                frame["following_gap_insufficient"] = bumper < 5.0+0.25*speed
        side_data = {}
        for side in ("left", "right"):
            neighbor = frame[side+"_lane_link_id"]
            if neighbor is not None:
                side_data[side] = _target_traffic(
                    mgeo, neighbor, ego_xy, heading, relevant, profiles, paths)
        direction = frame["lane_change_direction"]
        if maneuver is not None and direction in ("left", "right"):
            # The map match can switch to target before lateral motion ends.
            target = maneuver["source"] if maneuver.get("aborting") else maneuver["target"]
            side_data[direction] = _target_traffic(
                mgeo, target, ego_xy, heading, relevant, profiles, paths)
        if direction in side_data:
            front, rear = side_data[direction]
            for key, item in (("front", front), ("rear", rear)):
                if item is not None:
                    frame["target_"+key+"_actor_id"] = item[2].get("actor_id")
                    frame["target_"+key+"_gap_m"] = item[0]
                    frame["target_"+key+"_bumper_gap_m"] = item[1]
                    frame["target_"+key+"_speed_mps"] = number(item[2].get("speed_mps"))
            if speed is not None and not unresolved:
                fs = frame["target_front_speed_mps"]
                rs = frame["target_rear_speed_mps"]
                frame["target_front_clearance_ok"] = (True if front is None else
                    None if fs is None else
                    front[0] >= 20.0+max(0.0, speed-fs)*1.5)
                frame["target_rear_clearance_ok"] = (True if rear is None else
                    None if rs is None else
                    rear[0] >= 12.0+max(0.0, rs-speed)*1.5)
                a, b = frame["target_front_clearance_ok"], frame["target_rear_clearance_ok"]
                frame["target_lane_initially_safe"] = None if None in (a, b) else a and b
        merge_candidates = []
        for side in ("left", "right"):
            neighbor = frame[side+"_lane_link_id"]
            if neighbor is None:
                continue
            for actor in relevant:
                if actor.get("actor_link_id") != neighbor:
                    continue
                ax, ay = number(actor.get("world_x_m")), number(actor.get("world_y_m"))
                vx, vy = number(actor.get("vx_mps")), number(actor.get("vy_mps"))
                ap = _projection(mgeo, neighbor, ax, ay, profiles)
                if ap is None or None in (vx, vy) or ap[0] > 2.0:
                    continue
                vd = -vx*ap[4]+vy*ap[3]
                toward = vd < -0.3 if side == "left" else vd > 0.3
                longitudinal = abs((ax-x)*projection[3]+(ay-y)*projection[4])
                if toward and longitudinal <= 25:
                    merge_candidates.append((longitudinal, side, vd, actor))
        if merge_candidates:
            longitudinal, side, vd, actor = min(merge_candidates, key=lambda item:item[0])
            distance = number(actor.get("distance_to_ego_m"))
            frame.update(merging_actor_id=actor.get("actor_id"), merging_actor_side=side,
                         merging_actor_distance_m=distance,
                         merging_actor_link_id=actor.get("actor_link_id"),
                         merging_actor_target_lane_id=frame["ego_current_lane_id"],
                         merging_actor_v_d_mps=vd,
                         is_merging_into_ego_lane=True,
                         merge_conflict=longitudinal <= 8.0)
            frame["yield_to_merging_vehicle"] = (frame["merge_conflict"] is True and
                (frame.get("ego_decelerating") is True or frame.get("ego_stopped") is True))
        elif not unresolved:
            frame["is_merging_into_ego_lane"] = False
            frame["merge_conflict"] = False
            frame["yield_to_merging_vehicle"] = False
        if frame["merge_conflict"] is True and frame["merging_actor_id"] is not None:
            frame.update(target_actor_id=frame["merging_actor_id"],
                         target_kind="vehicle",
                         target_position_text=frame["merging_actor_side"],
                         target_distance_m=frame["merging_actor_distance_m"],
                         response_reason="merging_vehicle")
        elif frame["lead_actor_id"] is not None:
            frame.update(target_actor_id=frame["lead_actor_id"],
                         target_kind="vehicle", target_position_text="front",
                         target_distance_m=frame["lead_distance_m"],
                         response_reason="lead_vehicle")
        frame["highway_lateral_step_m"] = motion
        # Reflects the prior frame's classification, set before classifying
        # this one, so highway_rule sees the same field here and later (in
        # evaluate_frames) and resolves identically both times -- single
        # source of truth for ACC_FOLLOW's responding sub-states instead of
        # re-deriving lead_cause/merge_cause separately for this debounce.
        frame["highway_acc_prev_responding"] = prev_acc_responding
        classification = highway_rule(frame, grouped[frame["index"]])
        prev_acc_responding = bool(
            classification.get("label_candidate") == "ACC_FOLLOW"
            and classification.get("behavior_state") in RESPONDING_ACC_STATES)
        previous = frame
