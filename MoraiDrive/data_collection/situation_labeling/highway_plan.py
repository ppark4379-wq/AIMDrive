"""Bind highway instructions to the Expert path actually published at this frame.

The path is a *selected plan*, not a future executed ego trajectory.  Body-frame
lateral displacement is not used to distinguish a road bend from a lane change.
Receive-bound publications are explicitly weaker than simultaneous sensor GT.
"""
from collections import Counter, defaultdict
from copy import deepcopy
from functools import lru_cache
import hashlib
import math
from pathlib import Path

from .highway_features import _lane_id, _neighbor, _projection
from .instruction_features import _mgeo, _link_profile, number
from .highway_rules import highway_rule

MARKER = "current_highway_selected_plan"
SCHEMA = "current_highway_selected_plan_v1"
MAX_LOOKAHEAD_M = 60.0
MIN_LOOKAHEAD_M = 15.0
LANE_CENTER_TOLERANCE_M = .75


@lru_cache(maxsize=4)
def _map_context(map_root, expected_sha):
    from MoraiDrive.data_collection.mgeo_navigation import MGeoMap
    path=Path(map_root)/'link_set.json'
    if hashlib.sha256(path.read_bytes()).hexdigest()!=expected_sha:
        raise ValueError('Captured highway plan map changed')
    return MGeoMap(str(path)),{}


def _heading(points, end=False):
    pairs = list(zip(points, points[1:]))
    for a, b in reversed(pairs) if end else pairs:
        dx, dy = b[0]-a[0], b[1]-a[1]
        if math.hypot(dx, dy) > .01:
            return math.atan2(dy, dx)
    return None


def _angle(a, b):
    return abs((a-b+math.pi) % (2*math.pi)-math.pi)


def lane_corridor(mgeo, start, profiles, max_length=150.0):
    """Follow unique forward lane continuations, never a lane-change edge.

    A shared graph node is not sufficient to choose a branch.  Same lane
    identity and tangent continuity are required when several successors exist;
    unresolved forks terminate the corridor rather than joining both lanes.
    """
    if start not in mgeo.links:
        return [], "missing_start_link"
    result, length, reason = [], 0.0, None
    current = start
    while current not in result and len(result) < 12:
        points, cumulative = _link_profile(mgeo, current, profiles)
        if len(points) < 2:
            return result, "missing_lane_polyline"
        result.append(current)
        length += cumulative[-1]
        if length >= max_length:
            break
        link = mgeo.links[current]
        candidates = [identity for identity, other in mgeo.links.items()
                      if identity not in result and link.get("to_node_idx") is not None
                      and other.get("from_node_idx") == link.get("to_node_idx")]
        heading = _heading(points, end=True)
        scored = []
        for candidate in candidates:
            path, _ = _link_profile(mgeo, candidate, profiles)
            incoming = _heading(path)
            if heading is None or incoming is None or _angle(heading, incoming) > math.radians(25):
                continue
            if math.dist(points[-1], path[0]) > 1.0:
                continue
            scored.append(candidate)
        same_lane = [candidate for candidate in scored
                     if _lane_id(mgeo, candidate) == _lane_id(mgeo, current)]
        if len(same_lane) == 1:
            current = same_lane[0]
        elif len(scored) == 1:
            current = scored[0]
        else:
            reason = "ambiguous_lane_continuation" if scored else "lane_corridor_ends"
            break
    return result, reason


def _corridor_projection(mgeo, links, x, y, profiles):
    candidates = [(link, _projection(mgeo, link, x, y, profiles)) for link in links]
    candidates = [(link, p) for link, p in candidates if p is not None]
    return min(candidates, key=lambda value:value[1][0]) if candidates else (None, None)


def _path_samples(points, x, y):
    """Use spatial samples of this one published path; never future ego rows."""
    if not isinstance(points, list) or len(points) < 2:
        return [], "published_path_missing"
    if any(not isinstance(p, (list, tuple)) or len(p) < 2
           or number(p[0]) is None or number(p[1]) is None for p in points):
        return [], "published_path_nonfinite"
    start = min(range(len(points)), key=lambda k:math.hypot(points[k][0]-x, points[k][1]-y))
    if math.hypot(points[start][0]-x, points[start][1]-y) > 5:
        return [], "published_path_not_near_current_ego"
    samples, arc, previous, next_arc = [], 0.0, points[start], 0.0
    for point in points[start:]:
        arc += math.hypot(point[0]-previous[0], point[1]-previous[1])
        previous = point
        if arc >= next_arc:
            samples.append((arc, float(point[0]), float(point[1])))
            next_arc = arc + 2.0
        if arc >= MAX_LOOKAHEAD_M:
            break
    if arc < MIN_LOOKAHEAD_M or len(samples) < 5:
        return [], "published_path_too_short"
    return samples, None


def _shared_endpoint_lane_change_support(mgeo, source, target, side, keep, destination):
    """Separate a lane change before a merge from a branch after that merge.

    Successors of two converging lanes may diverge again. Reaching one of
    those successors does not establish entry into the original adjacent
    lane. Require the existing geometric destination criteria to hold on
    both original links before their shared endpoint, for four consecutive
    samples. Ordinary parallel-lane changes do not need this extra guard.
    """
    node = mgeo.links[source].get("to_node_idx")
    if node is None or node == "" or node != mgeo.links[target].get("to_node_idx"):
        return None
    sign = 1 if side == "left" else -1
    supported, consecutive, maximum_consecutive = [], 0, 0
    for current, adjacent in zip(keep, destination):
        if (current["link_id"] == source and adjacent["link_id"] == target
                and adjacent["distance_m"] <= LANE_CENTER_TOLERANCE_M
                and current["distance_m"] >= 1.75
                and sign * current["signed_left_m"] >= 1.75):
            supported.append(current["arc_m"])
            consecutive += 1
            maximum_consecutive = max(maximum_consecutive, consecutive)
        else:
            consecutive = 0
    return {"shared_to_node": node, "source_link": source, "target_link": target,
            "premerge_destination_sample_arcs_m": supported,
            "maximum_consecutive_premerge_destination_samples": maximum_consecutive,
            "minimum_consecutive_samples": 4,
            "valid": maximum_consecutive >= 4,
            "physical_lane_boundary_certified": False}


def classify_selected_path(frame, publication, mgeo, profiles=None):
    """Map-relative Keep/Change classification with auditable current evidence."""
    profiles = {} if profiles is None else profiles
    receipt = {"schema":SCHEMA, "valid":False, "uses_future_ego":False,
               "uses_neighbor_sentence":False, "physical_synchrony_certified":False,
               "target_lane_safety_certified":False, "frame_index":frame.get("index"),
               "sensor_ns":frame.get("lidar_source_ns")}
    if publication.get("status") != "RECEIVE_BOUND_CURRENT_CONTEXT":
        receipt["reason"] = "expert_publication_context_unverified"
        return receipt
    path = publication.get("local_path") or {}
    ego = publication.get("ego") or {}
    if (publication.get("sensor_ns") != frame.get("lidar_source_ns")
            or not path.get("payload_sha256") or number(path.get("receive_age_ms")) is None
            or not 0 <= path["receive_age_ms"] <= 100):
        receipt["reason"] = "expert_path_frame_or_receipt_unverified"
        return receipt
    if (path.get("header") or {}).get("frame") != "map":
        receipt["reason"] = "expert_path_coordinate_frame_unverified"
        return receipt
    if mgeo is None or frame.get("current_link_id") not in mgeo.links:
        receipt["reason"] = "current_map_link_unverified"
        return receipt
    if frame.get("highway_lane_association_ambiguous") is True:
        receipt["reason"] = "current_highway_lane_association_ambiguous"
        return receipt
    x, y = number(frame.get("ego_world_x_m")), number(frame.get("ego_world_y_m"))
    if x is None or y is None:
        receipt["reason"] = "current_ego_position_unverified"
        return receipt
    xyz = ego.get("world_xyz") or []
    if len(xyz) < 2 or math.hypot(xyz[0]-x, xyz[1]-y) > 2.0:
        receipt["reason"] = "expert_context_ego_position_disagrees"
        return receipt
    samples, reason = _path_samples(path.get("world_xyz"), x, y)
    if reason:
        receipt["reason"] = reason
        return receipt
    source = frame["current_link_id"]
    corridors = {}
    for side, link in (("keep",source), ("left",_neighbor(mgeo, source, "left")),
                       ("right",_neighbor(mgeo, source, "right"))):
        if link:
            links, end_reason = lane_corridor(mgeo, link, profiles)
            corridors[side] = {"start_link":link, "links":links, "end_reason":end_reason}
    diagnostics = {}
    for side, corridor in corridors.items():
        projections = [_corridor_projection(mgeo,corridor["links"],a,b,profiles)
                       for _,a,b in samples]
        if any(p is None for _,p in projections):
            continue
        diagnostics[side] = {**corridor,
            "samples":[{"arc_m":sample[0], "link_id":link,
                        "distance_m":p[0], "signed_left_m":p[2]}
                       for sample,(link,p) in zip(samples,projections)]}
    receipt.update(path_payload_sha256=path["payload_sha256"], path_record_id=path.get("record_id"),
                   path_receive_age_ms=path["receive_age_ms"],
                   path_source_timestamp_ns=path.get("source_timestamp_ns"),
                   source_link_id=source, source_lane_id=_lane_id(mgeo,source),
                   lane_center_tolerance_m=LANE_CENTER_TOLERANCE_M,
                   maximum_path_lookahead_m=MAX_LOOKAHEAD_M, corridors=diagnostics)
    keep = diagnostics.get("keep",{}).get("samples",[])
    if not keep:
        receipt["reason"] = "current_lane_path_projection_missing"
        return receipt
    # A curved current lane can have arbitrarily large body-axis y offsets.
    # Only distance to its own forward polyline establishes lane keeping.
    if max(p["distance_m"] for p in keep) <= LANE_CENTER_TOLERANCE_M:
        receipt.update(valid=True, decision="keep_current_lane", direction=None,
                       target_link_id=source, target_lane_id=_lane_id(mgeo,source))
        return receipt
    selected = []
    tail_start = max(MIN_LOOKAHEAD_M, samples[-1][0] - 12)
    for side in ("left","right"):
        candidate = diagnostics.get(side)
        if not candidate:
            continue
        tail = [p for p in candidate["samples"] if p["arc_m"] >= tail_start]
        keep_tail = [p for p in keep if p["arc_m"] >= tail_start]
        sign = 1 if side == "left" else -1
        # Demand a sustained geometric destination, not a noisy steering tick.
        if (len(tail) >= 4 and max(p["distance_m"] for p in tail) <= LANE_CENTER_TOLERANCE_M
                and min(p["distance_m"] for p in keep_tail) >= 1.75
                and sign * keep_tail[-1]["signed_left_m"] >= 1.75
                and keep[0]["distance_m"] <= 2.5):
            selected.append(side)
    if len(selected) == 1:
        side=selected[0];target=corridors[side]["start_link"]
        support = _shared_endpoint_lane_change_support(
            mgeo, source, target, side, keep, diagnostics[side]["samples"])
        if support is not None:
            receipt["shared_endpoint_lane_change_support"] = support
            if not support["valid"]:
                receipt["reason"] = "published_path_only_postmerge_branch"
                return receipt
        receipt.update(valid=True, decision="select_adjacent_lane", direction=side,
                       target_link_id=target,target_lane_id=_lane_id(mgeo,target))
    else:
        receipt["reason"] = ("published_path_lane_destination_ambiguous" if selected
                             else "published_path_lane_destination_unresolved")
    return receipt


def _command_features(frame, publication):
    """Keep observations intact; rule inputs use current verified commands."""
    control=publication.get("control") or {};target=publication.get("target_velocity") or {}
    ego=publication.get("ego") or {}
    if (control.get("longlCmdType") != 1 or not control.get("payload_sha256")
            or number(control.get("receive_age_ms")) is None
            or not 0 <= control["receive_age_ms"] <= 100):
        return None
    accel,brake=number(control.get("accel")),number(control.get("brake"))
    speed=number(ego.get("speed_mps"));desired=number(target.get("value_mps"))
    if None in (accel,brake,speed) or not (0 <= accel <= 1 and 0 <= brake <= 1):
        return None
    if brake <= 1e-6 and accel > 1e-6 and desired is not None and desired >= speed-.1:
        return {"stopped":False,"decelerating":False,"hard_braking":False,
                "kind":"commanded_acceleration", "brake":brake,"accel":accel,
                "speed_mps":speed,"target_velocity_mps":desired,
                "control_payload_sha256":control["payload_sha256"]}
    if brake > 1e-6:
        return {"stopped":speed <= .2 and desired is not None and desired <= .2,
                "decelerating":speed > .2,"hard_braking":False,
                "kind":"commanded_braking", "brake":brake,"accel":accel,
                "speed_mps":speed,"target_velocity_mps":desired,
                "control_payload_sha256":control["payload_sha256"]}
    return None


def attach_current_highway_plans(frames, objects, route):
    """Attach captured-plan features before evaluate_frames; no GT masks change."""
    mgeo=_mgeo(route);profiles={};counts=Counter()
    map_root=str(Path(route['map_root']).resolve()) if route.get('map_root') else None
    map_sha=hashlib.sha256((Path(map_root)/'link_set.json').read_bytes()).hexdigest() if map_root else None
    for frame in frames:
        if frame.get("zone_id") != "highway":
            continue
        publication=frame.get("expert_publications") or {}
        proof=classify_selected_path(frame,publication,mgeo,profiles)
        proof["command"]=_command_features(frame,publication) if proof["valid"] else None
        proof['map_root']=map_root
        proof['map_link_set_sha256']=map_sha
        frame[MARKER]=proof
        if proof["valid"]:
            frame["observed_highway_lane_change_active"]=frame.get("lane_change_active")
            frame["observed_highway_lane_change_direction"]=frame.get("lane_change_direction")
            frame.update(lane_change_active=proof["decision"] == "select_adjacent_lane",
                         lane_change_direction=proof["direction"],
                         source_lane_id=proof["source_lane_id"],
                         target_lane_id=proof["target_lane_id"],
                         target_lane_link_id=proof["target_link_id"],
                         lane_change_aborting=False,return_direction=None)
            counts[proof["decision"]]+=1
        else:
            # Legacy observed motion may remain diagnostic, but cannot be
            # promoted into an earlier current plan by a future-ego pass.
            counts[proof["reason"]]+=1
    return dict(sorted(counts.items()))


def current_rule_features(frame):
    """Return a copy: commanded response never overwrites observed GT fields."""
    proof=frame.get(MARKER) or {};command=proof.get("command") or {}
    if not proof.get("valid") or not command:
        return frame
    result=dict(frame)
    for name in ("stopped","decelerating","hard_braking"):
        result["ego_"+name]=command[name]
    # A prior-frame observed braking pulse is not current plan history.
    if command["kind"] == "commanded_acceleration":
        result["highway_acc_prev_responding"]=False
    return result


def bind_current_highway_plan_rules(frames, objects, rules):
    """Reproduce current rules after any legacy pass, recording the selected plan."""
    indexed={f["index"]:f for f in frames};inventory=defaultdict(list);counts=Counter()
    for actor in objects:
        inventory[actor["index"]].append(actor)
    for rule in rules:
        frame=indexed.get(rule["index"],{});proof=frame.get(MARKER) or {}
        if rule.get("zone_id") != "highway":
            continue
        rule.setdefault("evidence",{})[MARKER]=deepcopy(proof)
        if not proof.get("valid"):
            label=rule.get('label_candidate') or ''
            state=rule.get('instruction_state')
            if (label.startswith('CHANGE_') or state in
                    ('hard_braking','decelerating','yielding','return_to_source_lane')):
                # No fresh selected plan means an observed past braking pulse
                # cannot remain a strong current Expert command. Preserve the
                # candidate/evidence for recovery; observations stay usable.
                rule['action_valid']=False
                reasons=rule.setdefault('unknown_reasons',[])
                if 'current_highway_selected_plan_unverified' not in reasons:
                    reasons.append('current_highway_selected_plan_unverified')
                counts['strong_action_requires_current_selected_plan']+=1
            counts['current_selected_plan_unavailable_keep_observed_rule']+=1
            continue
        expected=highway_rule(current_rule_features(frame),inventory[rule["index"]])
        for key in ("status","instruction_state","label_candidate","behavior_state",
                    "related_actors","variables","unknown_reasons","instruction_id"):
            rule[key]=deepcopy(expected.get(key))
        rule.setdefault("evidence",{}).update(deepcopy(expected.get("evidence") or {}))
        rule["evidence"][MARKER]=deepcopy(proof)
        if (expected.get('instruction_state') in ('hard_braking','decelerating','yielding')
                and not proof.get('command')):
            rule['action_valid']=False
            rule.setdefault('unknown_reasons',[]).append('current_highway_longitudinal_command_unverified')
            counts['strong_response_requires_current_command']+=1
        counts["bound_"+str(expected.get("instruction_state"))]+=1
    return dict(sorted(counts.items()))


def validate_current_highway_plan(frames, objects, rule):
    """Check final rendering against the same pinned current plan and inventory."""
    from .mapper import load_label_instruction_map
    from .renderer import load_catalog, render_instruction
    matches=[f for f in frames if f.get("index")==rule.get("index")]
    if len(matches)!=1:
        return False,["current_highway_plan_frame_missing_or_ambiguous"]
    frame=matches[0];proof=frame.get(MARKER) or {}
    if not proof.get("valid") or rule.get("evidence",{}).get(MARKER)!=proof:
        return False,["current_highway_plan_receipt_disagrees"]
    if not proof.get('map_root') or not proof.get('map_link_set_sha256'):
        return False,['current_highway_plan_map_pin_missing']
    mgeo,profiles=_map_context(proof['map_root'],proof['map_link_set_sha256'])
    publication=frame.get('expert_publications') or {}
    recomputed=classify_selected_path(frame,publication,mgeo,profiles)
    recomputed['command']=_command_features(frame,publication) if recomputed['valid'] else None
    recomputed.update(map_root=proof['map_root'],map_link_set_sha256=proof['map_link_set_sha256'])
    if recomputed!=proof:
        return False,['current_highway_plan_geometry_or_command_recomputation_disagrees']
    expected=highway_rule(current_rule_features(frame),[a for a in objects if a.get("index")==frame["index"]])
    if any(expected.get(k)!=rule.get(k) for k in
           ("status","label_candidate","instruction_state","related_actors")):
        return False,["current_highway_plan_rule_recomputation_disagrees"]
    if expected.get("unknown_reasons") or expected.get("status")!="matched":
        return False,["current_highway_plan_rule_unresolved"]
    if (expected.get('instruction_state') in ('hard_braking','decelerating','yielding')
            and not proof.get('command')):
        return False,['current_highway_longitudinal_command_unverified']
    for key,value in (expected.get('variables') or {}).items():
        if (rule.get('variables') or {}).get(key)!=value:
            return False,['current_highway_plan_variables_recomputation_disagrees']
    catalog=load_catalog();mapping=load_label_instruction_map(catalog=catalog)
    identifier=mapping.get(("highway",expected["label_candidate"],expected["instruction_state"]))
    if identifier is None:
        return False,["current_highway_plan_rendering_unavailable"]
    text,slots,missing=render_instruction(catalog,"highway",identifier,expected.get("variables") or {})
    if missing or (identifier,text,slots)!=(rule.get("instruction_id"),rule.get("instruction"),rule.get("instruction_variables")):
        return False,["current_highway_plan_final_rendering_disagrees"]
    return True,[]
