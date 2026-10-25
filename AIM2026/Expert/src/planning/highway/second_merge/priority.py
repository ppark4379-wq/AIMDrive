"""두 번째 합류 우선권 판단. 기존 메서드와 플래너 상태를 그대로 사용한다."""
import math
from ....interfaces.geometry import Point
from ....config.parameters import highway as parameters


class SecondMergePriorityMixin:
    def _merge_zone_priority_speed_limit(
        self,
        vehicle_state,
        ego_link,
        object_matches,
        vehicle_entries,
        conservative_successors,
        merge_point,
    ):
        """Use the same future-OBB first-arrival rule as the first merge."""
        ego_match = self.lane_map.project(vehicle_state.position, ego_link)
        relevant = []
        for entry, object_match in object_matches:
            object_link = object_match.lane_link
            if object_link.id == ego_link.id:
                continue
            shared_successors = set(
                self.lane_map.successors.get(object_link.id, ())
            ).intersection(conservative_successors)
            if shared_successors:
                relevant.append((entry, object_match))

        touching = []
        for entry, object_match in relevant:
            object_info = entry["object_info"]
            object_speed = max(0.0, float(object_info.velocity))
            touch_distance = self._first_circle_obb_touch_distance(
                object_match,
                merge_point,
                parameters.CONSERVATIVE_MERGE_ENTRY_RADIUS,
                0.5 * max(1.0, float(getattr(object_info, "size_x", 0.0))),
                0.5 * max(1.0, float(getattr(object_info, "size_y", 0.0))),
                max_distance=object_speed * parameters.MERGE_PRIORITY_HORIZON,
            )
            touch_time = self._touch_time(
                touch_distance,
                object_speed,
                parameters.MERGE_PRIORITY_HORIZON,
            )
            if not math.isinf(touch_time):
                touching.append((entry, object_match, touch_time))

        ego_speed = max(0.0, float(vehicle_state.velocity))
        ego_entry_distance = self._first_circle_obb_touch_distance(
            ego_match,
            merge_point,
            parameters.CONSERVATIVE_MERGE_ENTRY_RADIUS,
            0.5 * parameters.VEHICLE_LENGTH,
            0.5 * parameters.VEHICLE_WIDTH,
            center_offset=0.5 * parameters.VEHICLE_LENGTH,
            max_distance=max(
                parameters.CONSERVATIVE_MERGE_EGO_RADIUS,
                ego_speed * parameters.MERGE_PRIORITY_HORIZON,
            ),
        )
        ego_inside = ego_entry_distance <= 1e-6
        ego_touch_time = self._touch_time(
            ego_entry_distance,
            ego_speed,
            parameters.MERGE_PRIORITY_HORIZON,
        )
        earliest_touch = (
            min(item[2] for item in touching) if touching else float("inf")
        )
        other_inside = earliest_touch <= 1e-9

        if self._merge_zone_priority is None:
            if other_inside:
                self._merge_zone_priority = "OTHER"
            elif ego_inside:
                self._merge_zone_priority = "EGO"
            elif touching and self._merge_arrivals_overlap(
                ego_touch_time,
                ego_speed,
                touching,
                parameters.CONSERVATIVE_MERGE_ENTRY_RADIUS,
                clear_time_margin=parameters.CONSERVATIVE_MERGE_GO_CLEAR_TIME_MARGIN,
                clear_distance=parameters.CONSERVATIVE_MERGE_CLOSE_GAP_YIELD_DISTANCE,
            ):
                self._merge_zone_priority = "OTHER"
            elif not math.isinf(ego_touch_time):
                self._merge_zone_priority = "EGO"

            if self._merge_zone_priority == "OTHER":
                winner_entry, winner_match, _ = min(
                    touching, key=lambda item: item[2]
                )
                self._merge_zone_other_key = self._merge_zone_object_key(
                    winner_entry, winner_match
                )
                self._merge_zone_clear_cycles = 0

        tracked_entry = None
        tracked_match = None
        for entry, object_match in object_matches:
            if (
                self._merge_zone_object_key(entry, object_match)
                == self._merge_zone_other_key
            ):
                tracked_entry = entry
                tracked_match = object_match
                break

        if self._merge_zone_priority == "OTHER" and ego_inside:
            tracked_has_entered = False
            if tracked_match is not None:
                tracked_link = tracked_match.lane_link
                if tracked_link.id in conservative_successors:
                    tracked_has_entered = True
                else:
                    tracked_circle_s = self.lane_map.project(
                        merge_point, tracked_link
                    ).s
                    tracked_has_entered = tracked_match.s >= tracked_circle_s
            if not tracked_has_entered and not other_inside:
                previous_key = self._merge_zone_other_key
                self._merge_zone_priority = "EGO"
                self._merge_zone_other_key = None
                self._merge_zone_clear_cycles = 0
                self._last_merge_debug.append(
                    "second predictive-zone actual entry override: EGO entered before {}".format(
                        previous_key or "predicted vehicle"
                    )
                )

        if self._merge_zone_priority == "EGO":
            self._merge_speed_limit_state = float("inf")
            for entry, object_match in object_matches:
                object_link = object_match.lane_link
                shared_successors = set(
                    self.lane_map.successors.get(object_link.id, ())
                ).intersection(conservative_successors)
                if object_link.id in conservative_successors or shared_successors:
                    self._merge_go_object_ids.add(id(entry["object_info"]))
            self._last_merge_debug.append(
                "second predictive-zone priority=EGO ego_touch={}s, late_vehicles_go={}".format(
                    "inf" if math.isinf(ego_touch_time) else "{:.2f}".format(ego_touch_time),
                    len(self._merge_go_object_ids),
                )
            )
            return float("inf")

        if self._merge_zone_priority != "OTHER":
            self._last_merge_debug.append(
                "second predictive-zone priority=PENDING (no predicted entry)"
            )
            return float("inf")

        tracked_cleared = (
            tracked_match is not None
            and tracked_match.lane_link.id in conservative_successors
            and tracked_match.s > parameters.CONSERVATIVE_MERGE_PASS_DISTANCE
        )
        if tracked_entry is not None and not tracked_cleared:
            self._merge_zone_clear_cycles = 0
        else:
            self._merge_zone_clear_cycles += 1
        if (
            self._merge_zone_clear_cycles
            >= self._cycles_for_reference_count(
                parameters.CONSERVATIVE_MERGE_CLEAR_CYCLES
            )
        ):
            self._merge_zone_priority = "EGO"
            self._merge_zone_other_key = None
            self._merge_zone_clear_cycles = 0
            self._merge_speed_limit_state = float("inf")
            self._last_merge_debug.append(
                "second predictive-zone other cleared -> priority=EGO"
            )
            return float("inf")

        tracked_speed = (
            max(0.0, float(tracked_entry["object_info"].velocity))
            if tracked_entry is not None
            else 0.0
        )
        if tracked_match is None or tracked_speed <= 0.1:
            clear_time = float("inf")
        elif tracked_match.lane_link.id in conservative_successors:
            clear_time = max(
                0.0,
                parameters.CONSERVATIVE_MERGE_PASS_DISTANCE - tracked_match.s,
            ) / tracked_speed
        else:
            clear_time = (
                max(0.0, tracked_match.lane_link.length - tracked_match.s)
                + parameters.CONSERVATIVE_MERGE_PASS_DISTANCE
            ) / tracked_speed

        if tracked_speed < parameters.CONSERVATIVE_MERGE_MOVING_SPEED:
            speed_limit = 0.0
        else:
            rolling_floor = parameters.MERGE_MIN_YIELD_SPEED
            desired_arrival_time = (
                clear_time + parameters.CONSERVATIVE_MERGE_FOLLOW_TIME_GAP
            )
            pacing_limit = (
                ego_entry_distance / max(0.1, desired_arrival_time)
                if not math.isinf(ego_entry_distance)
                else rolling_floor
            )
            braking_limit = math.sqrt(
                2.0
                * parameters.CONSERVATIVE_MERGE_COMFORT_DECEL
                * max(0.0, ego_entry_distance)
            )
            speed_limit = max(
                rolling_floor,
                min(
                    parameters.CONSERVATIVE_MERGE_MAX_SPEED,
                    pacing_limit,
                    braking_limit if braking_limit > 1e-6 else 0.0,
                ),
            )
        self._last_merge_debug.append(
            "second predictive-zone priority=OTHER key={} touch={}s ego_touch={}s clear_time={}s entry_distance={:.1f}m limit={:.2f}m/s".format(
                self._merge_zone_other_key or "unknown",
                "inf" if math.isinf(earliest_touch) else "{:.2f}".format(earliest_touch),
                "inf" if math.isinf(ego_touch_time) else "{:.2f}".format(ego_touch_time),
                "inf" if math.isinf(clear_time) else "{:.2f}".format(clear_time),
                0.0 if math.isinf(ego_entry_distance) else ego_entry_distance,
                speed_limit,
            )
        )
        return speed_limit

