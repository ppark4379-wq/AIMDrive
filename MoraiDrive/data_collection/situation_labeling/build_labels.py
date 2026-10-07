"""Join sealed episode evidence and write one numeric feature row per GT actor."""

import argparse
from collections import Counter
import json
import math
from pathlib import Path

import yaml

from MoraiDrive.data_collection.mgeo_navigation import load_official_route
from .features import RouteProjector, numeric_features
from .feature_definitions import project_feature_rows, validate_feature_rows
from .instruction_features import enrich, number, attach_static_avoid_episode
from .rules import evaluate_frames, summarize_rules
from .expert_instructions import (apply_expert_instructions,
                                  hold_roundabout_instruction_across_brief_tracking_gaps)
from .expert_static_phase import attach_static_phase
from .expert_roundabout_phase import build_roundabout_observations
from .mapper import attach_rule_instructions
from .ego_position import update_static_positions
from .consistency import (HIGHWAY_KEEP_FOLLOW_IDS, HIGHWAY_MAX_ISLAND_FRAMES,
                          align_highway_lane_changes, align_static_avoidance,
                          drop_contradicted_instructions,
                          fill_short_instruction_gaps, hold_dynamic_wait_until_moving,
                          smooth_short_instruction_islands, use_stopped_roundabout_sentences)


FEATURE_NAMES = ("forward_m", "left_m", "actor_width_m", "actor_length_m",
                 "speed_mps", "vx_mps", "vy_mps",
                 "path_s_m", "ego_path_s_m", "delta_s_m", "path_d_m",
                 "path_segment_index", "world_x_m", "world_y_m",
                 "path_tangent_x", "path_tangent_y", "v_s_mps", "v_d_mps")


def _frames(path):
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"expected frame array: {path}")
    indexed = {}
    for row in rows:
        index = row["index"]
        if index in indexed:
            raise ValueError(f"duplicate frame index {index}: {path}")
        indexed[index] = row
    return indexed


def _match_actor(gt, geometry_rows):
    """Match unique instance keys; use actor ID only if both keys are absent."""
    key = gt.get("instance_key")
    actor_id = gt.get("id")
    if key:
        matches = [row for row in geometry_rows if row.get("instance_key") == key]
    elif actor_id is not None:
        matches = [row for row in geometry_rows
                   if not row.get("instance_key") and row.get("actor_id") == actor_id]
    else:
        return None, "missing_actor_identity"
    if len(matches) != 1:
        return None, "ambiguous_actor_match" if matches else "missing_geometry_actor"
    row = matches[0]
    if (row.get("actor_id") != actor_id or row.get("kind") != gt.get("kind")):
        return None, "actor_identity_mismatch"
    return row, None


def build_features(episode, route, gt_dir="gt_timeline_v1",
                   geometry_dir="object_geometry_audit_v2", admission_dir="admission_v2"):
    """Return feature rows and report; no source evidence is modified."""
    episode = Path(episode)
    gt_frames = _frames(episode / gt_dir / "frames.json")
    geometry_frames = _frames(episode / geometry_dir / "frames.json")
    admission_frames = _frames(episode / admission_dir / "frames.json")
    if gt_frames.keys() != geometry_frames.keys() or gt_frames.keys() != admission_frames.keys():
        raise ValueError("GT, geometry, and admission frame index sets differ")
    projector = RouteProjector(route["nav_waypoints"])
    records = []
    reasons = Counter()
    kinds = Counter()
    for index in sorted(gt_frames):
        gt, geometry, admission = (frames[index] for frames in
                                   (gt_frames, geometry_frames, admission_frames))
        stamp = gt["lidar_source_ns"]
        if (geometry.get("lidar_source_ns") != stamp or
                admission.get("lidar_source_time_ns") != stamp):
            raise ValueError(f"source timestamp mismatch at frame {index}")
        object_gt = gt["object_gt"]
        # The frame's GT ego position is projected once for all actors. Both
        # route distances use the same official path and metre convention.
        ego_xy = gt.get("ego_xy")
        ego_path_s_m = None
        ego_projection_error = None
        if not isinstance(ego_xy, list) or len(ego_xy) < 2:
            ego_projection_error = "missing_ego_xy"
        else:
            try:
                ego_path_s_m = projector.project(ego_xy[0], ego_xy[1])[0]
            except (TypeError, ValueError, OverflowError):
                ego_projection_error = "invalid_ego_xy"
        for actor in object_gt.get("objects", []):
            actor_id = actor.get("id")
            row = {"index": index, "lidar_source_ns": stamp,
                   "actor_id": actor_id, "instance_key": actor.get("instance_key"),
                   "kind": actor.get("kind"),
                   "actor_heading_deg": number(actor.get("heading_deg")),
                   "object_training_label_valid": admission.get("object_training_label_valid") is True,
                   **{name: None for name in FEATURE_NAMES}}
            row["ego_path_s_m"] = ego_path_s_m
            # Keep sealed GT position even if the BEV geometry join fails, so
            # rule relevance can distinguish a distant invalid actor.
            position = actor.get("position")
            if isinstance(position, list) and len(position) >= 2:
                row["world_x_m"] = number(position[0])
                row["world_y_m"] = number(position[1])
            invalid = []
            if ego_projection_error:
                invalid.append(ego_projection_error)
            for flag, reason in (("dataset_build_candidate", "frame_not_admitted_for_dataset_build"),
                                 ("object_gt_transport_timing_candidate", "object_time_association_failed"),
                                 ("object_model_identity_candidate", "actor_model_identity_failed")):
                if admission.get(flag) is not True:
                    invalid.append(reason)
            if object_gt.get("transport_timing_candidate") is not True:
                invalid.extend(object_gt.get("reasons") or ["object_time_association_failed"])
            if object_gt.get("model_identity_candidate") is not True:
                invalid.extend(object_gt.get("identity_reasons") or ["actor_model_identity_failed"])
            if object_gt.get("source") != "grpc_full_inventory" or object_gt.get("complete_inventory") is not True:
                invalid.append("incomplete_object_inventory")
            matched, match_reason = _match_actor(actor, geometry.get("objects", []))
            if match_reason:
                invalid.append(match_reason)
            else:
                if matched.get("training_bev_relevant") is not True:
                    invalid.append("outside_training_bev_or_unknown")
                if matched.get("api_box_geometry_candidate") is not True:
                    invalid.extend(matched.get("reasons") or ["object_geometry_not_candidate"])
                try:
                    row.update(numeric_features(actor, matched, projector))
                    if ego_path_s_m is not None:
                        row["delta_s_m"] = row["path_s_m"] - ego_path_s_m
                except (TypeError, ValueError, OverflowError) as exc:
                    invalid.append(str(exc) if str(exc) else "feature_calculation_failed")
            row["invalid_reasons"] = sorted(set(invalid))
            row["valid"] = not row["invalid_reasons"]
            reasons.update(row["invalid_reasons"])
            kinds[str(row["kind"])] += 1
            records.append(row)
    report = {"schema": "morai_situation_features_v1", "total_frames": len(gt_frames),
              "total_objects": len(records),
              "valid_feature_records": sum(row["valid"] for row in records),
              "invalid_feature_records": sum(not row["valid"] for row in records),
              "invalid_reason_counts": dict(sorted(reasons.items())),
              "object_kind_counts": dict(sorted(kinds.items())),
              "route_source": route.get("route_source"),
              "global_path_source": route.get("global_path_source"),
              "semantic_rules_applied": False}
    return records, report


def _zone_intervals(route):
    """Use only zone bounds explicitly anchored to official path indices."""
    points = route["nav_waypoints"]
    checkpoints = {}
    for checkpoint in route.get("checkpoints", []):
        index = int(checkpoint["path_index"])
        if not 0 <= index < len(points):
            raise ValueError("checkpoint path_index outside official path")
        if math.hypot(float(points[index][0]) - float(checkpoint["x"]),
                      float(points[index][1]) - float(checkpoint["y"])) > 0.01:
            raise ValueError("checkpoint coordinates differ from official path")
        checkpoints[str(checkpoint["id"])] = index
    zones = route.get("zones") or {}
    intervals = []
    dynamic = zones.get("dynamic_obstacle")
    if dynamic:
        intervals.append(("dynamic_obstacle", int(dynamic["start_path_index"]),
                          int(dynamic["end_path_index"])))
    static = zones.get("static_obstacle")
    if static:
        try:
            start, end = checkpoints["3"], checkpoints["4"]
        except KeyError as exc:
            raise ValueError("static_obstacle requires checkpoints 3 and 4") from exc
        obstacle_index = int(static["route_path_index"])
        if not start <= obstacle_index <= end:
            raise ValueError("static obstacle anchor lies outside checkpoints 3-4")
        intervals.append(("static_obstacle", start, end))
    for name in ("roundabout", "highway", "tunnel"):
        zone = zones.get(name)
        if not zone:
            continue
        if name == "tunnel":
            start_key, end_key = "entry_checkpoint", "exit_checkpoint"
        else:
            start_key, end_key = "ego_entry_checkpoint", "ego_exit_checkpoint"
        try:
            start = checkpoints[str(zone[start_key])]
            end = checkpoints[str(zone[end_key])]
        except KeyError as exc:
            raise ValueError(f"{name} checkpoint is absent from official route") from exc
        intervals.append((name, start, end))
    for name, start, end in intervals:
        if not 0 <= start <= end < len(points):
            raise ValueError(f"{name} zone indices outside official path")
    for i, (_, start, end) in enumerate(intervals):
        if any(start <= other_end and other_start <= end
               for _, other_start, other_end in intervals[i + 1:]):
            raise ValueError("overlapping official route zones")
    return intervals


def build_frame_features(episode, route, gt_dir="gt_timeline_v1",
                         admission_dir="admission_v2", max_speed_interval_s=0.25):
    """One row per sensor frame, including frames with zero GT objects.

    Ego speed is a backward XY finite difference in world metres per second.
    It uses only adjacent, admitted frames and original LiDAR source times.
    """
    if not math.isfinite(max_speed_interval_s) or max_speed_interval_s <= 0:
        raise ValueError("max_speed_interval_s must be positive and finite")
    episode = Path(episode)
    gt_frames = _frames(episode / gt_dir / "frames.json")
    admission_frames = _frames(episode / admission_dir / "frames.json")
    if gt_frames.keys() != admission_frames.keys():
        raise ValueError("GT and admission frame index sets differ")
    projector = RouteProjector(route["nav_waypoints"])
    intervals = _zone_intervals(route)
    rows = []
    previous = None
    for index in sorted(gt_frames):
        gt, admission = gt_frames[index], admission_frames[index]
        stamp = gt["lidar_source_ns"]
        if admission.get("lidar_source_time_ns") != stamp:
            raise ValueError(f"source timestamp mismatch at frame {index}")
        row = {"index": index, "lidar_source_ns": stamp,
               "ego_path_s_m": None, "ego_path_d_m": None, "ego_path_segment_index": None,
               "ego_speed_mps": None, "ego_speed_source": None,
               "ego_speed_invalid_reason": None, "zone_id": None,
               "ego_world_x_m": None, "ego_world_y_m": None,
               "dataset_build_candidate": admission.get("dataset_build_candidate") is True,
               "object_training_label_valid": admission.get("object_training_label_valid") is True,
               "object_inventory_usable": (
                   admission.get("object_gt_transport_timing_candidate") is True
                   and admission.get("object_model_identity_candidate") is True
                   and admission.get("full_actor_inventory_transport_candidate") is True
                   and gt.get("object_gt", {}).get("source") == "grpc_full_inventory"
                   and gt.get("object_gt", {}).get("complete_inventory") is True
                   and gt.get("object_gt", {}).get("transport_timing_candidate") is True
                   and gt.get("object_gt", {}).get("model_identity_candidate") is True)}
        invalid = []
        if not row["dataset_build_candidate"]:
            invalid.extend(admission.get("reasons") or ["frame_not_admitted_for_dataset_build"])
        xy = gt.get("ego_xy")
        current_xy = None
        if not isinstance(xy, list) or len(xy) < 2:
            invalid.append("missing_ego_xy")
        else:
            try:
                current_xy = (float(xy[0]), float(xy[1]))
                s, d, segment, _, _ = projector.project(*current_xy)
                row["ego_world_x_m"], row["ego_world_y_m"] = current_xy
                row["ego_path_s_m"] = s
                row["ego_path_d_m"] = d
                row["ego_path_segment_index"] = segment
                row["zone_id"] = next((name for name, start, end in intervals
                                       if start <= segment <= end), None)
            except (TypeError, ValueError, OverflowError):
                current_xy = None
                invalid.append("invalid_ego_xy")
        if (previous is None or previous[0] + 1 != index or
                not row["dataset_build_candidate"] or not previous[3] or
                current_xy is None or previous[2] is None):
            row["ego_speed_invalid_reason"] = "no_adjacent_admitted_ego_frame"
        else:
            dt = (int(stamp) - int(previous[1])) / 1e9
            if not 0 < dt <= max_speed_interval_s:
                row["ego_speed_invalid_reason"] = "invalid_ego_speed_interval"
            else:
                speed = math.dist(current_xy, previous[2]) / dt
                if math.isfinite(speed):
                    row["ego_speed_mps"] = speed
                    row["ego_speed_source"] = "backward_gt_ego_xy_difference"
                else:
                    row["ego_speed_invalid_reason"] = "nonfinite_ego_speed"
        row["invalid_reasons"] = sorted(set(invalid))
        row["valid"] = not row["invalid_reasons"]
        rows.append(row)
        previous = (index, stamp, current_xy, row["dataset_build_candidate"])
    return rows


def build_feature_tables(episode, route, gt_dir="gt_timeline_v1",
                         geometry_dir="object_geometry_audit_v2",
                         admission_dir="admission_v2"):
    """Build separate frame and object rows using the same sealed evidence."""
    objects, report = build_features(episode, route, gt_dir, geometry_dir, admission_dir)
    frames = build_frame_features(episode, route, gt_dir, admission_dir)
    frames, objects = enrich(frames, objects, route)
    frame_reasons = Counter(reason for row in frames for reason in row["invalid_reasons"])
    speed_reasons = Counter(row["ego_speed_invalid_reason"] for row in frames
                            if row["ego_speed_invalid_reason"])
    report.update(schema="morai_situation_features_v0_2",
                  valid_frame_records=sum(row["valid"] for row in frames),
                  invalid_frame_records=sum(not row["valid"] for row in frames),
                  frame_invalid_reason_counts=dict(sorted(frame_reasons.items())),
                  ego_speed_available_frames=sum(row["ego_speed_mps"] is not None
                                                 for row in frames),
                  ego_speed_invalid_reason_counts=dict(sorted(speed_reasons.items())),
                  zone_counts=dict(sorted(Counter(row["zone_id"] or "unassigned" for row in frames).items())),
                  zone_definition="official_path_indices; static_obstacle uses checkpoints 3-4")
    return frames, objects, report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--route-definition", type=Path, required=True)
    parser.add_argument("--global-path", type=Path, required=True)
    parser.add_argument("--route-key", default="full_loop")
    parser.add_argument("--gt-timeline-dir", default="gt_timeline_v1")
    parser.add_argument("--object-geometry-dir", default="object_geometry_audit_v2")
    parser.add_argument("--admission-dir", default="admission_v2")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expert-log", type=Path, action="append",
                        help="existing Expert diagnostic log or aligned state observations")
    parser.add_argument("--static-supervisor-log", type=Path, default=None,
                        help="supervisor.log to forward-fill Expert's static-obstacle "
                             "approach/follow/rejoined phase onto frames (optional)")
    parser.add_argument("--roundabout-supervisor-log", type=Path, default=None,
                        help="supervisor.log to recover Expert's roundabout_entry phase "
                             "(cruise/follow/yield/collision_slowdown/emergency) via "
                             "sim-time interpolation, for episodes collected before "
                             "Expert emitted a per-tick header-seq observation (optional)")
    args = parser.parse_args(argv)
    route = load_official_route(str(args.route_definition), str(args.global_path), args.route_key)
    with args.route_definition.open(encoding="utf-8-sig") as stream:
        route["zones"] = (yaml.safe_load(stream) or {})[args.route_key].get("zones", {})
    frames, objects, report = build_feature_tables(
        args.episode, route, args.gt_timeline_dir,
        args.object_geometry_dir, args.admission_dir)
    if args.static_supervisor_log is not None:
        attach_static_phase(frames, args.static_supervisor_log)
        attach_static_avoid_episode(frames)
        report["expert_static_phase_applied"] = True
    rules = evaluate_frames(frames, objects)
    roundabout_observations = None
    if args.roundabout_supervisor_log is not None:
        roundabout_observations = build_roundabout_observations(
            frames, args.roundabout_supervisor_log)
        report["expert_roundabout_phase_applied"] = True
        report["expert_roundabout_phase_frames"] = len(roundabout_observations)
    if args.expert_log is not None or roundabout_observations is not None:
        report["expert_instructions"] = apply_expert_instructions(
            frames, rules, args.episode, args.expert_log or [],
            index_observations=roundabout_observations, objects=objects)
    report["motion_alignment"] = {"static": align_static_avoidance(frames, rules),
                                  "dynamic": hold_dynamic_wait_until_moving(frames, rules),
                                  "highway": align_highway_lane_changes(frames, rules, objects)}
    report["instruction_mapping"] = attach_rule_instructions(rules)
    if args.expert_log is not None or roundabout_observations is not None:
        hold_roundabout_instruction_across_brief_tracking_gaps(rules)
    report["instruction_gap_frames_filled"] = fill_short_instruction_gaps(rules)
    report["instruction_island_frames_smoothed"] = (
        smooth_short_instruction_islands(rules)
        + smooth_short_instruction_islands(rules, HIGHWAY_MAX_ISLAND_FRAMES, "highway",
                                           HIGHWAY_KEEP_FOLLOW_IDS))
    report["roundabout_stopped_sentences"] = use_stopped_roundabout_sentences(frames, rules)
    from .side_view_hold import camera_views
    report["static_ego_position"] = update_static_positions(rules, objects, cameras=camera_views(args.episode))
    report["instruction_consistency"] = drop_contradicted_instructions(frames, rules)
    from .reviewed_overrides import apply_reviewed_overrides
    report['reviewed_override_frames'] = apply_reviewed_overrides(rules, args.episode)
    if report['reviewed_override_frames']:
        report['instruction_consistency'] = dict(Counter(r.get('instruction_consistency', 'null') for r in rules))
    report["rules"] = summarize_rules(rules)
    report["rules"]["instruction_templates_applied"] = (
        any(row.get("instruction") is not None for row in rules))
    report["rules"]["instruction_id_counts"] = dict(sorted(Counter(
        row["instruction_id"] for row in rules if row.get("instruction_id") is not None).items()))
    if args.expert_log is not None or roundabout_observations is not None:
        report["rules"]["expert_state_counts"] = dict(sorted(Counter(
            row["expert_state"] for row in rules
            if row.get("expert_state") is not None).items()))
    report["semantic_rules_applied"] = True
    report["schema"] = "morai_situation_features_v0_3_rule_inputs"
    persisted_frames, persisted_objects = project_feature_rows(frames, objects)
    report["feature_definitions"] = validate_feature_rows(
        persisted_frames, persisted_objects)
    args.output.mkdir(parents=True, exist_ok=False)
    for filename, rows in (("frame_features.jsonl", persisted_frames),
                           ("object_features.jsonl", persisted_objects),
                           ("situation_rules.jsonl", rules)):
        with (args.output / filename).open("w", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    (args.output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
