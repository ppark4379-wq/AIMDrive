"""Unchanged Expert decision diagnostics, called after control."""
import numpy as np
from ..config.parameters import highway as highway_parameters


def log_control(self, vehicle_state, planning_path, lattice_result, target_velocity,
                planned_velocity, profile_planned_velocity, curvature_bypass,
                acc_cmd, raw_steering_rad, front_steer_rad):
    self._debug_counter = getattr(self, "_debug_counter", 0) + 1
    if (
        highway_parameters.HIGHWAY_DEBUG_ENABLED
        and self._debug_counter
        >= highway_parameters.HIGHWAY_DEBUG_INTERVAL_CYCLES
    ):
        self._debug_counter = 0
        print(
            f"[CTRL] raw_steering={np.degrees(raw_steering_rad):+.2f}deg "
            f"front_steer={np.degrees(front_steer_rad):+.2f}deg "
            f"({front_steer_rad:+.3f}rad) acc_cmd={acc_cmd:+.3f} "
            f"target_v={target_velocity:.2f} planned_v={planned_velocity:.2f} "
            f"profile_v={profile_planned_velocity:.2f} "
            f"cur_v={vehicle_state.velocity:.2f}"
        )
        if curvature_bypass:
            print("[PLAN] curvature=off zone=second_merge reference_v={:.2f} base_v={:.2f}".format(
                profile_planned_velocity, planned_velocity))
        details = getattr(lattice_result, "debug_details", {})
        if details.get("lane_change_conflict"):
            print("[PLAN] lane_change_conflict={}".format(details["lane_change_conflict"]))
        if "static_avoidance" in details:
            print("[STATIC] {}".format(details["static_avoidance"]))
        if details.get('jamming_pedestrian', {}).get('checked'):
            print('[JAMMING_PERSON] {}'.format(details['jamming_pedestrian']))
        if details:
            def debug_value(value, suffix=""):
                return "inf" if np.isinf(value) else f"{value:.2f}{suffix}"

            print(
                "[PLAN] reason={} state={} link={} lane={} s={:.1f} -> target_link={} "
                "objects={}/{} path={} active={}".format(
                    lattice_result.reason,
                    getattr(lattice_result, "highway_state", "OFF"),
                    getattr(lattice_result, "current_link_id", ""),
                    details.get("ego_lane", "?"),
                    details.get("ego_s", 0.0),
                    getattr(lattice_result, "target_link_id", ""),
                    details.get("matched_vehicle_objects", 0),
                    details.get("total_vehicle_objects", 0),
                    len(planning_path),
                    lattice_result.active,
                )
            )
            print(
                "[PLAN] risk gap={} TTC={} front_acc={} merge_raw={} "
                "merge_filtered={} final_acc={} keep_valid={} keep_collision={}".format(
                    debug_value(lattice_result.nearest_vehicle_distance, "m"),
                    debug_value(lattice_result.ttc, "s"),
                    debug_value(details.get("front_acc_limit", float("inf")), "m/s"),
                    debug_value(details.get("raw_merge_limit", float("inf")), "m/s"),
                    debug_value(details.get("filtered_merge_limit", float("inf")), "m/s"),
                    debug_value(getattr(lattice_result, "acc_target_velocity", float("inf")), "m/s"),
                    details.get("keep_valid", True),
                    debug_value(details.get("keep_collision_time", float("inf")), "s"),
                )
            )
            for merge_check in details.get("merge_checks", ())[:6]:
                print("[PLAN] merge: {}".format(merge_check))
            for lane_check in details.get("lane_checks", ())[:6]:
                print("[PLAN] lane: {}".format(lane_check))
            for candidate in getattr(lattice_result, "candidates", ()):
                print(
                    "[PLAN] candidate dir={} link={} valid={} cost={} "
                    "collision={} curvature={:.4f}".format(
                        getattr(candidate, "direction", "?"),
                        getattr(candidate, "target_link_id", ""),
                        candidate.valid,
                        debug_value(candidate.cost),
                        debug_value(getattr(candidate, "collision_time", float("inf")), "s"),
                        getattr(candidate, "curvature_cost", 0.0),
                    )
                )
