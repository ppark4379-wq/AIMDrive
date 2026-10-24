"""예측 합류의 도착 시간·차체 영역·진입 판단.

HighwayLanePlanner의 동일 인스턴스 상태를 사용하는 메서드 묶음이다.
생성자나 별도 상태를 추가하지 않으며, 기존 메서드 본문과 호출 순서를 유지한다.
"""
import math
from ...interfaces.geometry import Point
from . import parameters


class PredictiveMergeMixin:
    def _predictive_merge_zone_speed_limit(
        self,
        vehicle_state,
        ego_match,
        object_matches,
        vehicle_entries,
        merge_successors,
        merge_point,
    ):
        """Give the merge circle to the OBB expected to touch it first."""
        ego_link = ego_match.lane_link
        relevant = []
        for entry, object_match in object_matches:
            object_link = object_match.lane_link
            if object_link.id == ego_link.id:
                continue
            if object_link.id not in parameters.PREDICTIVE_MERGE_OBJECT_LINK_IDS:
                continue
            relevant.append((entry, object_match))

        touching = []
        for entry, object_match in relevant:
            object_info = entry["object_info"]
            object_speed = max(0.0, float(object_info.velocity))
            object_half_length = 0.5 * max(
                1.0, float(getattr(object_info, "size_x", 0.0))
            )
            object_half_width = 0.5 * max(
                1.0, float(getattr(object_info, "size_y", 0.0))
            )
            object_touch_distance = self._first_circle_obb_touch_distance(
                object_match,
                merge_point,
                parameters.PREDICTIVE_MERGE_ENTRY_RADIUS,
                object_half_length,
                object_half_width,
                max_distance=(
                    object_speed * parameters.MERGE_PRIORITY_HORIZON
                ),
            )
            object_touch_time = self._touch_time(
                object_touch_distance,
                object_speed,
                parameters.MERGE_PRIORITY_HORIZON,
            )
            if not math.isinf(object_touch_time):
                touching.append((entry, object_match, object_touch_time))

        ego_speed = max(0.0, float(vehicle_state.velocity))
        ego_entry_distance = self._first_circle_obb_touch_distance(
            ego_match,
            merge_point,
            parameters.PREDICTIVE_MERGE_ENTRY_RADIUS,
            0.5 * parameters.VEHICLE_LENGTH,
            0.5 * parameters.VEHICLE_WIDTH,
            center_offset=0.5 * parameters.VEHICLE_LENGTH,
            max_distance=max(
                parameters.PREDICTIVE_MERGE_EGO_RADIUS,
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
            min(time_value for _, _, time_value in touching)
            if touching
            else float("inf")
        )
        other_inside = earliest_touch <= 1e-9
        # Actual occupancy wins over an earlier noisy ETA decision. Once Ego
        # touches the conflict core first, merge pacing must never brake it into
        # a side/rear impact.
        if self._predictive_merge_priority is None:
            # 현재 원 안에 먼저 들어온 차량을 우선한다. 둘 다 원 밖이면
            # 미래 경로가 단순히 원에 닿는지만 보지 않고 예상 진입시간을
            # 비교하여 상대 차량이 먼저 도착할 때만 선제 감속한다.
            if other_inside:
                self._predictive_merge_priority = "OTHER"
                self._predictive_merge_clear_cycles = 0
            elif ego_inside:
                self._predictive_merge_priority = "EGO"
            elif touching and self._merge_arrivals_overlap(
                ego_touch_time,
                ego_speed,
                touching,
                parameters.PREDICTIVE_MERGE_ENTRY_RADIUS,
            ):
                self._predictive_merge_priority = "OTHER"
                self._predictive_merge_clear_cycles = 0
            elif not math.isinf(ego_touch_time):
                # Use the same future path/OBB criterion for Ego, and lock GO
                # before its current rear-axle position reaches the circle.
                self._predictive_merge_priority = "EGO"

            if self._predictive_merge_priority == "OTHER":
                winner_entry, winner_match, _ = min(
                    touching, key=lambda item: item[2]
                )
                self._predictive_merge_other_key = self._merge_zone_object_key(
                    winner_entry, winner_match
                )

        if self._predictive_merge_priority == "OTHER" and ego_inside:
            # A future ETA is only a provisional right-of-way decision. Sensor
            # and velocity noise can swap two arrivals separated by a fraction
            # of a second. If Ego actually touches the circle before the
            # selected vehicle has ever reached it, Ego is the real winner and
            # must not keep braking into the merge.
            tracked_match = next(
                (
                    object_match
                    for entry, object_match in object_matches
                    if self._merge_zone_object_key(entry, object_match)
                    == self._predictive_merge_other_key
                ),
                None,
            )
            tracked_has_entered = False
            if tracked_match is not None:
                tracked_link = tracked_match.lane_link
                if tracked_link.id in merge_successors:
                    tracked_has_entered = True
                elif tracked_link.id in parameters.PREDICTIVE_MERGE_OBJECT_LINK_IDS:
                    tracked_circle_s = self.lane_map.project(
                        merge_point, tracked_link
                    ).s
                    tracked_has_entered = tracked_match.s >= tracked_circle_s
            if (
                tracked_match is not None
                and not tracked_has_entered
                and not other_inside
            ):
                previous_key = self._predictive_merge_other_key
                self._predictive_merge_priority = "EGO"
                self._predictive_merge_other_key = None
                self._predictive_merge_clear_cycles = 0
                self._last_merge_debug.append(
                    "predictive-zone actual entry override: EGO entered before {}".format(
                        previous_key or "predicted vehicle"
                    )
                )

        if self._predictive_merge_priority == "EGO":
            self._merge_speed_limit_state = float("inf")
            self._predictive_merge_other_key = None
            for entry, _ in relevant:
                self._merge_go_object_ids.add(id(entry["object_info"]))
            self._last_merge_debug.append(
                "predictive-zone priority=EGO ego_touch={}s (Ego OBB first), late_vehicles_go={}".format(
                    "inf" if math.isinf(ego_touch_time) else "{:.2f}".format(ego_touch_time),
                    len(self._merge_go_object_ids),
                )
            )
            return float("inf")

        if self._predictive_merge_priority != "OTHER":
            self._last_merge_debug.append(
                "predictive-zone priority=PENDING monitored={} (no predicted entry)".format(
                    ",".join(parameters.PREDICTIVE_MERGE_OBJECT_LINK_IDS)
                )
            )
            return float("inf")

        # Decide and lock the winner early, but do not disturb normal driving
        # before the user-designated first-merge braking point.
        brake_start = Point(*parameters.PREDICTIVE_MERGE_BRAKE_START_POINT)
        brake_start_s = self.lane_map.project(brake_start, ego_link).s
        if ego_match.s + 1e-6 < brake_start_s:
            self._merge_speed_limit_state = float("inf")
            self._last_merge_debug.append(
                "predictive-zone priority=OTHER waiting brake-start "
                "remaining={:.1f}m -> no decel".format(
                    brake_start_s - ego_match.s
                )
            )
            return float("inf")

        # Follow only the vehicle which actually won priority. A stream of
        # later approach vehicles must not extend OTHER forever after the
        # winner has cleared the overlap.
        tracked_entry = None
        tracked_match = None
        for entry, object_match in object_matches:
            if (
                self._merge_zone_object_key(entry, object_match)
                == self._predictive_merge_other_key
            ):
                tracked_entry = entry
                tracked_match = object_match
                break

        if tracked_entry is None and str(
            self._predictive_merge_other_key or ""
        ).startswith("link:"):
            # MORAI normally supplies stable NPC names. For unnamed objects,
            # bridge the one frame in which its link changes to the successor.
            successor_entries = [
                (entry, object_match)
                for entry, object_match in object_matches
                if object_match.lane_link.id in merge_successors
            ]
            if len(successor_entries) == 1:
                tracked_entry, tracked_match = successor_entries[0]
                self._predictive_merge_other_key = self._merge_zone_object_key(
                    tracked_entry, tracked_match
                )

        tracked_cleared = (
            tracked_match is not None
            and tracked_match.lane_link.id in merge_successors
            and tracked_match.s > parameters.PREDICTIVE_MERGE_PASS_DISTANCE
        )
        cycle_time = self.control_dt
        if tracked_entry is not None:
            tracked_speed = max(
                0.0, float(tracked_entry["object_info"].velocity)
            )
            clear_time = self._predictive_merge_clear_time(
                tracked_match, tracked_speed, merge_successors
            )
            self._predictive_merge_last_speed = tracked_speed
            self._predictive_merge_last_clear_time = clear_time
            predicted_hold = (
                clear_time + parameters.PREDICTIVE_MERGE_FOLLOW_TIME_GAP
                if not math.isinf(clear_time)
                else parameters.PREDICTIVE_MERGE_DROPOUT_MAX_HOLD_TIME
            )
            self._predictive_merge_dropout_hold_time = max(
                parameters.PREDICTIVE_MERGE_DROPOUT_MIN_HOLD_TIME,
                min(
                    parameters.PREDICTIVE_MERGE_DROPOUT_MAX_HOLD_TIME,
                    predicted_hold,
                ),
            )
        else:
            self._predictive_merge_dropout_hold_time = max(
                0.0,
                self._predictive_merge_dropout_hold_time - cycle_time,
            )
            if not math.isinf(self._predictive_merge_last_clear_time):
                self._predictive_merge_last_clear_time = max(
                    0.0,
                    self._predictive_merge_last_clear_time - cycle_time,
                )
            tracked_speed = self._predictive_merge_last_speed
            clear_time = self._predictive_merge_last_clear_time

        if tracked_entry is not None and not tracked_cleared:
            self._predictive_merge_clear_cycles = 0
        else:
            self._predictive_merge_clear_cycles += 1
        confirmed_clear = (
            tracked_cleared
            and self._predictive_merge_clear_cycles
            >= self._cycles_for_reference_count(
                parameters.PREDICTIVE_MERGE_CLEAR_CYCLES
            )
        )
        predicted_clear_after_dropout = (
            tracked_entry is None
            and self._predictive_merge_clear_cycles
            >= self._cycles_for_reference_count(
                parameters.PREDICTIVE_MERGE_CLEAR_CYCLES
            )
            and self._predictive_merge_dropout_hold_time <= 1e-9
        )
        if confirmed_clear or predicted_clear_after_dropout:
            self._predictive_merge_priority = "EGO"
            self._predictive_merge_other_key = None
            self._predictive_merge_clear_cycles = 0
            self._predictive_merge_last_speed = 0.0
            self._predictive_merge_last_clear_time = float("inf")
            self._predictive_merge_dropout_hold_time = 0.0
            self._merge_speed_limit_state = float("inf")
            for entry, _ in relevant:
                self._merge_go_object_ids.add(id(entry["object_info"]))
            self._last_merge_debug.append(
                "predictive-zone other cleared -> priority={}".format(
                    self._predictive_merge_priority or "PENDING"
                )
            )
            return float("inf")

        if tracked_entry is None:
            self._last_merge_debug.append(
                "predictive-zone winner dropout key={} hold={:.2f}s using_last_speed={:.2f}m/s".format(
                    self._predictive_merge_other_key or "unknown",
                    self._predictive_merge_dropout_hold_time,
                    tracked_speed,
                )
            )
        if tracked_speed < parameters.PREDICTIVE_MERGE_MOVING_SPEED:
            # A genuinely stopped priority vehicle cannot be followed safely.
            speed_limit = 0.0
        else:
            rolling_floor = parameters.MERGE_MIN_YIELD_SPEED
            desired_arrival_time = (
                clear_time + parameters.PREDICTIVE_MERGE_FOLLOW_TIME_GAP
            )
            pacing_limit = (
                ego_entry_distance / max(0.1, desired_arrival_time)
                if not math.isinf(ego_entry_distance)
                else rolling_floor
            )
            braking_limit = math.sqrt(
                2.0
                * parameters.PREDICTIVE_MERGE_COMFORT_DECEL
                * max(0.0, ego_entry_distance)
            )
            speed_limit = max(
                rolling_floor,
                min(
                    parameters.PREDICTIVE_MERGE_MAX_SPEED,
                    pacing_limit,
                    braking_limit if braking_limit > 1e-6 else 0.0,
                ),
            )
            speed_limit = min(
                speed_limit,
                parameters.PREDICTIVE_MERGE_MAX_SPEED,
            )
        self._last_merge_debug.append(
            "predictive-zone priority=OTHER key={} touch={}s ego_touch={}s clear_time={}s entry_distance={:.1f}m limit={:.2f}m/s".format(
                self._predictive_merge_other_key or "unknown",
                "inf" if math.isinf(earliest_touch) else "{:.2f}".format(earliest_touch),
                "inf" if math.isinf(ego_touch_time) else "{:.2f}".format(ego_touch_time),
                "inf" if math.isinf(clear_time) else "{:.2f}".format(clear_time),
                0.0 if math.isinf(ego_entry_distance) else ego_entry_distance,
                speed_limit,
            )
        )
        # Moving priority traffic is paced, not stopped for. The generic OBB
        # collision gate can still command zero for a real imminent collision.
        return speed_limit

    @staticmethod
    def _predictive_merge_clear_time(
        object_match, object_speed, merge_successors
    ):
        if object_match is None or object_speed <= 0.1:
            return float("inf")
        object_link = object_match.lane_link
        if object_link.id in merge_successors:
            remaining = max(
                0.0,
                parameters.PREDICTIVE_MERGE_PASS_DISTANCE - object_match.s,
            )
        elif object_link.id in parameters.PREDICTIVE_MERGE_OBJECT_LINK_IDS:
            remaining = (
                max(0.0, object_link.length - object_match.s)
                + parameters.PREDICTIVE_MERGE_PASS_DISTANCE
            )
        else:
            return float("inf")
        return remaining / object_speed

    def _first_circle_obb_touch_distance(
        self,
        lane_match,
        circle_center,
        circle_radius,
        half_length,
        half_width,
        center_offset=0.0,
        max_distance=0.0,
    ):
        """Return route distance until an OBB first intersects a circle."""
        max_distance = max(0.0, float(max_distance))

        def touches(distance):
            options = self.lane_map.advance_options(lane_match, distance)
            return any(
                self._obb_touches_circle(
                    point,
                    yaw,
                    center_offset,
                    half_length,
                    half_width,
                    circle_center,
                    circle_radius,
                )
                for point, yaw, _ in options
            )

        if touches(0.0):
            return 0.0
        if max_distance <= 1e-9:
            return float("inf")

        step = max(0.25, parameters.PATH_SAMPLE_SPACING)
        previous = 0.0
        distance = min(step, max_distance)
        while distance <= max_distance + 1e-9:
            if touches(distance):
                low = previous
                high = distance
                for _ in range(8):
                    middle = 0.5 * (low + high)
                    if touches(middle):
                        high = middle
                    else:
                        low = middle
                return high
            if distance >= max_distance - 1e-9:
                break
            previous = distance
            distance = min(distance + step, max_distance)
        return float("inf")

    @staticmethod
    def _touch_time(distance, speed, horizon=float("inf")):
        if distance <= 1e-6:
            return 0.0
        if math.isinf(distance) or speed <= 0.1:
            return float("inf")
        time_value = distance / speed
        return time_value if time_value <= horizon + 1e-9 else float("inf")

    @staticmethod
    def _merge_arrivals_overlap(ego_entry_time, ego_speed, touching, radius,
                                *, clear_time_margin=None, clear_distance=None):
        """Yield unless Ego can fully clear with both time and distance margin."""
        if clear_time_margin is None:
            clear_time_margin = parameters.MERGE_GO_CLEAR_TIME_MARGIN
        if clear_distance is None:
            clear_distance = parameters.MERGE_CLOSE_GAP_YIELD_DISTANCE
        if not touching:
            return False
        winner_entry, _, other_entry_time = min(touching, key=lambda item: item[2])
        if math.isinf(ego_entry_time):
            return True
        other_speed = max(
            0.0, float(winner_entry["object_info"].velocity)
        )
        # Conservative OBB traversal estimate from first touch until Ego's rear
        # leaves the conflict circle.
        occupancy_length = 2.0 * float(radius) + parameters.VEHICLE_LENGTH
        ego_exit_time = ego_entry_time + occupancy_length / max(0.1, ego_speed)
        clear_time = other_entry_time - ego_exit_time
        predicted_bumper_gap = max(0.0, clear_time) * other_speed
        return (
            clear_time < clear_time_margin
            or predicted_bumper_gap
            < clear_distance
        )

    @staticmethod
    def _obb_touches_circle(
        path_point,
        yaw,
        center_offset,
        half_length,
        half_width,
        circle_center,
        circle_radius,
    ):
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        center_x = path_point.x + center_offset * cos_yaw
        center_y = path_point.y + center_offset * sin_yaw
        delta_x = circle_center.x - center_x
        delta_y = circle_center.y - center_y
        local_x = delta_x * cos_yaw + delta_y * sin_yaw
        local_y = -delta_x * sin_yaw + delta_y * cos_yaw
        closest_x = max(-half_length, min(half_length, local_x))
        closest_y = max(-half_width, min(half_width, local_y))
        error_x = local_x - closest_x
        error_y = local_y - closest_y
        return error_x * error_x + error_y * error_y <= circle_radius ** 2
