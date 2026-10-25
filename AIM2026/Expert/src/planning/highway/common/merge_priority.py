"""합류 구간 속도 제한·우선권·안정화.

HighwayLanePlanner의 동일 인스턴스 상태를 사용하는 메서드 묶음이다.
생성자나 별도 상태를 추가하지 않으며, 기존 메서드 본문과 호출 순서를 유지한다.
"""
import math
from ....interfaces.geometry import Point
from ....config.parameters import highway as parameters


from ..second_merge.priority import SecondMergePriorityMixin


class MergeSpeedMixin(SecondMergePriorityMixin):
    def _merge_conflict_is_source_lead(self, ego_match, object_matches):
        """Separate following on the source lane from a converging-lane yield."""
        if not self._merge_guard_active:
            return False
        conflict = self.merge_safety.details.get("conflict") or {}
        if conflict.get("rear") or conflict.get("kind") not in ("gap", "obb"):
            return False
        return any(
            match.lane_link.id == ego_match.lane_link.id and match.s > ego_match.s
            and self._merge_zone_object_key(entry, match) == conflict.get("id")
            for entry, match in object_matches
        )

    def _merge_acc_speed_limit(
        self, vehicle_state, ego_match, object_matches, vehicle_entries=(),
        reference_path=None, desired_velocity=None, forward_speed_limit=float("inf"),
    ):
        """Return a speed ceiling when two different links reach one merge."""
        ego_link = ego_match.lane_link
        self._last_merge_debug = []
        self._merge_go_object_ids = set()
        limit = self.merge_safety.limit(
            vehicle_state, ego_match, object_matches, vehicle_entries,
            reference_path=reference_path, desired_velocity=desired_velocity,
            forward_speed_limit=forward_speed_limit,
        )
        self._merge_guard_active = limit is not None
        if self._merge_guard_active:
            details = self.merge_safety.details
            priority = "EGO" if details["decision"] == "GO" else "OTHER"
            if details["zone"] == "FIRST_PREDICTIVE":
                self._predictive_merge_priority = priority
                self._merge_zone_priority = None
            else:
                self._merge_zone_priority = priority
                self._predictive_merge_priority = None
            self._last_merge_debug.append(
                "{} {} conflict={} target={:.2f}m/s".format(
                    details["zone"], details["decision"], details.get("conflict"), limit
                ) + " cause={} stop_conflict={}".format(
                    details.get("cause", "none"), details.get("stop_conflict"))
            )
            return limit
        ego_distance = max(0.0, ego_link.length - ego_match.s)
        ego_speed = max(0.0, float(vehicle_state.velocity))

        ego_eta = ego_distance / max(0.1, ego_speed)
        speed_limit = float("inf")
        ego_successors = set(self.lane_map.successors.get(ego_link.id, ()))
        predictive_successors = ego_successors.intersection(
            parameters.PREDICTIVE_MERGE_SUCCESSORS
        )
        predictive_point = Point(*parameters.PREDICTIVE_MERGE_POINT)
        predictive_zone = (
            bool(predictive_successors)
            and vehicle_state.position.distance(predictive_point)
            <= parameters.PREDICTIVE_MERGE_EGO_RADIUS
        )
        if predictive_zone:
            return self._predictive_merge_zone_speed_limit(
                vehicle_state,
                ego_match,
                object_matches,
                vehicle_entries,
                predictive_successors,
                predictive_point,
            )
        if self._predictive_merge_priority is not None:
            self._predictive_merge_priority = None
            self._predictive_merge_clear_cycles = 0
            self._predictive_merge_other_key = None
            self._predictive_merge_last_speed = 0.0
            self._predictive_merge_last_clear_time = float("inf")
            self._predictive_merge_dropout_hold_time = 0.0

        conservative_successors = ego_successors.intersection(
            parameters.CONSERVATIVE_MERGE_SUCCESSORS
        )
        merge_point = Point(*parameters.CONSERVATIVE_MERGE_POINT)
        conservative_zone = (
            bool(conservative_successors)
            and vehicle_state.position.distance(merge_point)
            <= parameters.CONSERVATIVE_MERGE_EGO_RADIUS
        )
        if conservative_zone:
            priority_limit = self._merge_zone_priority_speed_limit(
                vehicle_state,
                ego_link,
                object_matches,
                vehicle_entries,
                conservative_successors,
                merge_point,
            )
            if self._merge_zone_priority == "EGO":
                # Ego entered first, so merge-opponent OBB predictions must not
                # brake it into a side/rear collision. Same-lane front ACC and
                # non-merge safety logic remain active outside this override.
                self._merge_speed_limit_state = float("inf")
                return float("inf")
            if math.isinf(priority_limit):
                return float("inf")
            return priority_limit

        if self._merge_zone_priority is not None and (
            not conservative_successors
            or vehicle_state.position.distance(merge_point)
            > parameters.CONSERVATIVE_MERGE_EGO_RADIUS
        ):
            self._reset_merge_zone_priority()

        self._merge_decisions = {
            key: decision
            for key, decision in self._merge_decisions.items()
            if key[0] in ego_successors
        }
        if (
            parameters.MERGE_MIN_REMAINING_DISTANCE
            <= ego_distance
            <= parameters.MERGE_LOOKAHEAD_DISTANCE
        ):
            for entry, object_match in object_matches:
                object_link = object_match.lane_link
                if object_link.id == ego_link.id:
                    continue
                shared_successors = ego_successors.intersection(
                    self.lane_map.successors.get(object_link.id, ())
                )
                if not shared_successors:
                    continue

                object_distance = max(0.0, object_link.length - object_match.s)
                if object_distance > parameters.MERGE_LOOKAHEAD_DISTANCE:
                    self._last_merge_debug.append(
                        "graph object={} link={} outside distance={:.1f}m".format(
                            getattr(entry["object_info"], "name", "") or "unnamed",
                            object_link.id,
                            object_distance,
                        )
                    )
                    continue
                successor_id = sorted(shared_successors)[0]
                conservative_yield = (
                    conservative_zone
                    and successor_id in conservative_successors
                    and entry["object_info"].position.distance(merge_point)
                    <= parameters.CONSERVATIVE_MERGE_OBJECT_RADIUS
                )
                object_speed = max(0.0, float(entry["object_info"].velocity))
                if object_speed < 0.1 and not conservative_yield:
                    continue
                object_eta = object_distance / max(0.1, object_speed)
                object_key = (
                    successor_id,
                    getattr(entry["object_info"], "name", "")
                    or object_link.id,
                )
                decision = self._merge_decisions.get(object_key)
                if conservative_yield:
                    # This merge is intentionally safety-first: even if Ego's
                    # ETA is smaller, let the neighboring approach vehicle pass.
                    decision = "YIELD"
                    self._merge_decisions[object_key] = decision
                elif decision is None:
                    decision = (
                        "GO"
                        if ego_eta + parameters.MERGE_EGO_PRIORITY_MARGIN
                        < object_eta
                        else "YIELD"
                    )
                    self._merge_decisions[object_key] = decision

                if decision == "GO":
                    self._merge_go_object_ids.add(id(entry["object_info"]))
                    self._last_merge_debug.append(
                        "graph object={} link={} successor={} ego_eta={:.2f}s obj_eta={:.2f}s -> MERGE_GO locked".format(
                            getattr(entry["object_info"], "name", "") or "unnamed",
                            object_link.id,
                            successor_id,
                            ego_eta,
                            object_eta,
                        )
                    )
                    continue
                if (
                    not conservative_yield
                    and
                    abs(ego_eta - object_eta)
                    > parameters.MERGE_ARRIVAL_TIME_WINDOW
                ):
                    self._last_merge_debug.append(
                        "graph object={} link={} ego_eta={:.2f}s obj_eta={:.2f}s -> time-clear".format(
                            getattr(entry["object_info"], "name", "") or "unnamed",
                            object_link.id,
                            ego_eta,
                            object_eta,
                        )
                    )
                    continue

                yield_time = object_eta + parameters.MERGE_YIELD_TIME_GAP
                speed_limit = min(
                    speed_limit,
                    ego_distance / max(0.1, yield_time),
                )
                if conservative_yield:
                    speed_limit = min(
                        speed_limit,
                        parameters.CONSERVATIVE_MERGE_MAX_SPEED,
                    )
                self._last_merge_debug.append(
                    "graph object={} link={} successor={} ego_eta={:.2f}s obj_eta={:.2f}s -> {} limit={:.2f}m/s".format(
                        getattr(entry["object_info"], "name", "") or "unnamed",
                        object_link.id,
                        successor_id,
                        ego_eta,
                        object_eta,
                        (
                            "SAFE_ZONE_YIELD"
                            if conservative_yield
                            else "MERGE_YIELD locked"
                        ),
                        speed_limit,
                    )
                )

        if conservative_zone:
            # Keep yielding while the other vehicle is crossing the beginning
            # of the shared successor. This closes the one-frame gap where its
            # lane match changes from an approach link to the merged link.
            matched_ids = {id(entry["object_info"]) for entry, _ in object_matches}
            for entry, object_match in object_matches:
                object_info = entry["object_info"]
                if object_match.lane_link.id not in conservative_successors:
                    continue
                if object_info.position.distance(merge_point) > parameters.CONSERVATIVE_MERGE_OBJECT_RADIUS:
                    continue
                if object_match.s > parameters.CONSERVATIVE_MERGE_PASS_DISTANCE:
                    continue
                object_speed = max(0.0, float(object_info.velocity))
                clear_distance = max(
                    0.0,
                    parameters.CONSERVATIVE_MERGE_PASS_DISTANCE
                    - object_match.s,
                )
                clear_time = (
                    clear_distance / max(0.1, object_speed)
                    + parameters.MERGE_YIELD_TIME_GAP
                )
                speed_limit = min(
                    speed_limit,
                    ego_distance / max(0.1, clear_time),
                    parameters.CONSERVATIVE_MERGE_MAX_SPEED,
                )
                self._last_merge_debug.append(
                    "successor object={} link={} s={:.1f}m -> SAFE_ZONE_YIELD limit={:.2f}m/s".format(
                        getattr(object_info, "name", "") or "unnamed",
                        object_match.lane_link.id,
                        object_match.s,
                        speed_limit,
                    )
                )

            # A vehicle between lane centerlines can temporarily fail map
            # matching. In this designated zone, proximity is enough to keep a
            # conservative speed until its lane association becomes stable.
            for entry in vehicle_entries:
                object_info = entry["object_info"]
                if id(object_info) in matched_ids:
                    continue
                if object_info.position.distance(merge_point) > parameters.CONSERVATIVE_MERGE_OBJECT_RADIUS:
                    continue
                speed_limit = min(
                    speed_limit,
                    parameters.CONSERVATIVE_MERGE_MAX_SPEED,
                )
                self._last_merge_debug.append(
                    "unmatched object={} near merge -> SAFE_ZONE_YIELD limit={:.2f}m/s".format(
                        getattr(object_info, "name", "") or "unnamed",
                        speed_limit,
                    )
                )
        trajectory_limit = self._trajectory_merge_speed_limit(
            vehicle_state, ego_match, object_matches, vehicle_entries
        )
        speed_limit = min(speed_limit, trajectory_limit)
        if math.isinf(speed_limit):
            return float("inf")
        return min(ego_speed, speed_limit)


    @staticmethod
    def _merge_zone_object_key(entry, lane_match):
        object_info = entry["object_info"]
        unique_id = getattr(object_info, "unique_id", None)
        if unique_id is not None:
            return "npc:{}".format(unique_id)
        name = str(getattr(object_info, "name", "") or "").strip()
        if name:
            return name
        if lane_match is not None:
            return "link:{}".format(lane_match.lane_link.id)
        return "anonymous:{:.1f}:{:.1f}".format(
            object_info.position.x, object_info.position.y
        )

    def _reset_merge_zone_priority(self):
        self._merge_zone_priority = None
        self._merge_zone_other_key = None
        self._merge_zone_clear_cycles = 0
        self._predictive_merge_priority = None
        self._predictive_merge_clear_cycles = 0
        self._predictive_merge_other_key = None
        self._predictive_merge_last_speed = 0.0
        self._predictive_merge_last_clear_time = float("inf")
        self._predictive_merge_dropout_hold_time = 0.0

    def _trajectory_merge_speed_limit(
        self, vehicle_state, ego_match, object_matches, vehicle_entries
    ):
        """Backstop merge ACC when an incoming vehicle misses lane matching."""
        ego_speed = max(0.0, float(vehicle_state.velocity))
        horizon = parameters.MERGE_TRAJECTORY_HORIZON
        step = parameters.MERGE_TRAJECTORY_TIME_STEP
        matched_object_ids = {
            id(entry["object_info"]) for entry, _ in object_matches
        }
        time_value = step
        earliest_collision = float("inf")
        unmatched_count = sum(
            1
            for entry in vehicle_entries
            if id(entry["object_info"]) not in matched_object_ids
        )
        while time_value <= horizon + 1e-9:
            ego_point, ego_yaw, _ = self.lane_map.advance_default(
                ego_match, ego_speed * time_value
            )
            for entry in vehicle_entries:
                object_info = entry["object_info"]
                if id(object_info) in self._merge_go_object_ids:
                    continue
                if id(object_info) in matched_object_ids:
                    continue
                if (
                    vehicle_state.position.distance(object_info.position)
                    > parameters.MERGE_TRAJECTORY_MAX_DISTANCE
                ):
                    continue
                local_position = entry.get("local_position")
                if (
                    local_position is not None
                    and local_position.x < 0.0
                    and abs(local_position.y)
                    <= parameters.ACC_FALLBACK_LATERAL_THRESHOLD
                ):
                    # Braking for a same-lane rear vehicle only worsens risk.
                    continue
                object_speed = max(0.0, float(object_info.velocity))
                object_yaw = math.radians(
                    float(getattr(object_info, "heading_deg", 0.0))
                )
                object_point = Point(
                    object_info.position.x
                    + math.cos(object_yaw) * object_speed * time_value,
                    object_info.position.y
                    + math.sin(object_yaw) * object_speed * time_value,
                )
                if self._obb_collision(
                    ego_point,
                    ego_yaw,
                    object_point,
                    object_yaw,
                    object_info,
                ):
                    earliest_collision = min(earliest_collision, time_value)
            time_value += step

        if math.isinf(earliest_collision):
            if unmatched_count:
                self._last_merge_debug.append(
                    "trajectory unmatched={} collision=none horizon={:.1f}s".format(
                        unmatched_count, horizon
                    )
                )
            return float("inf")
        if earliest_collision <= 1.0:
            self._last_merge_debug.append(
                "trajectory unmatched={} collision={:.2f}s -> STOP".format(
                    unmatched_count, earliest_collision
                )
            )
            return 0.0
        delay_ratio = max(
            0.0,
            (earliest_collision - 0.5)
            / (earliest_collision + parameters.MERGE_YIELD_TIME_GAP),
        )
        limit = ego_speed * delay_ratio
        self._last_merge_debug.append(
            "trajectory unmatched={} collision={:.2f}s -> limit={:.2f}m/s".format(
                unmatched_count, earliest_collision, limit
            )
        )
        return limit

    def _stabilize_merge_speed_limit(self, raw_limit, ego_speed):
        """Apply emergency reductions now and release transient ACC smoothly."""
        if self._merge_guard_active:
            # The new ceiling has already passed trajectory validation. Keeping
            # an obsolete lower cap would prolong braking after the car clears.
            self._merge_speed_limit_state = raw_limit
            return raw_limit
        previous = self._merge_speed_limit_state
        recovery_step = parameters.MERGE_ACC_RECOVERY_RATE * self.control_dt
        if not math.isinf(raw_limit):
            if math.isinf(previous):
                filtered = max(0.0, raw_limit)
            else:
                filtered = min(
                    max(0.0, raw_limit),
                    previous + recovery_step,
                )
            self._merge_speed_limit_state = filtered
            return filtered

        if math.isinf(previous):
            return float("inf")
        recovered = previous + recovery_step
        if recovered >= max(0.0, float(ego_speed)):
            self._merge_speed_limit_state = float("inf")
            return float("inf")
        self._merge_speed_limit_state = recovered
        return recovered
