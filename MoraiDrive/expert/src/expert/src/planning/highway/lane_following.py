"""기준 차로 매칭·전방 위험·ACC 속도 처리.

HighwayLanePlanner의 동일 인스턴스 상태를 사용하는 메서드 묶음이다.
생성자나 별도 상태를 추가하지 않으며, 기존 메서드 본문과 호출 순서를 유지한다.
"""
import math
from . import parameters


class LaneFollowingMixin:
    def _project_maneuver_route(self, position, link_id):
        """Project only onto the committed lane and its actual route successors."""
        matches, visited = [], set()
        for _ in range(9):
            if link_id not in self.lane_map.links or link_id in visited:
                break
            visited.add(link_id)
            matches.append(self.lane_map.project(position, self.lane_map.links[link_id]))
            link_id = self.lane_map.default_successor(link_id)
        return min(matches, key=lambda match: match.distance) if matches else None

    def _resolve_ego_lane(self, vehicle_state):
        if (self.state in (self.CHANGE_LEFT, self.CHANGE_RIGHT)
                and self.active_candidate is not None):
            candidate = self.active_candidate
            # A new road section is not a new lateral target. In particular,
            # 420 -> 408 must not fall back to source successor 434 merely
            # because Ego is still more than 2.4 m from its target centerline.
            match = self._project_maneuver_route(vehicle_state.position, candidate.target_link_id)
            maximum = max(parameters.MANEUVER_TARGET_MAX_DISTANCE,
                          abs(candidate.offset) + 1.0)
            if match is None or match.distance > maximum:
                return None
            self.target_link_id = match.lane_link.id
            candidate.target_link_id = match.lane_link.id
            candidate.target_lane = match.lane_link.lane_number
            return match

        global_match = self.lane_map.match(
            vehicle_state.position,
            heading=vehicle_state.yaw,
            preferred_ids=(self.target_link_id,) if self.target_link_id else (),
        )
        if self.target_link_id is None:
            return global_match

        target = self.lane_map.links.get(self.target_link_id)
        if target is not None:
            target_match = self.lane_map.project(vehicle_state.position, target)
            if target_match.s >= target.length - 0.1:
                # The endpoint projection remains within 4 m after crossing a
                # link boundary. Advance now so newly available neighbors are
                # considered before consuming the obstacle-avoidance distance.
                for successor_id in self.lane_map.successors.get(target.id, ()):
                    successor_match = self.lane_map.project(vehicle_state.position, self.lane_map.links[successor_id])
                    if (successor_match.s > 0.05
                            and successor_match.distance < target_match.distance
                            and successor_match.distance <= parameters.LANE_MATCH_MAX_DISTANCE):
                        self.target_link_id = successor_id
                        return successor_match
            target_max_distance = parameters.MANEUVER_TARGET_MAX_DISTANCE
            if (self.state in (self.CHANGE_LEFT, self.CHANGE_RIGHT)
                    and (getattr(self.active_candidate, "lane_change_count", 1) == 2
                         or getattr(self.active_candidate, 'static_corridor', False))):
                # The verified path spans two legal adjacent lanes. Retain its
                # final target while ego is still on the original/via lane.
                target_max_distance = max(target_max_distance, abs(self.active_candidate.offset) + 1.0)
            if target_match.distance <= target_max_distance:
                return target_match
            for successor_id in self.lane_map.successors.get(target.id, ()):
                successor = self.lane_map.links[successor_id]
                successor_match = self.lane_map.project(vehicle_state.position, successor)
                if successor_match.distance <= parameters.LANE_MATCH_MAX_DISTANCE:
                    self.target_link_id = successor_id
                    return successor_match
        return global_match

    def _match_objects(self, vehicle_entries):
        matches = []
        for entry in vehicle_entries:
            object_info = entry["object_info"]
            heading = math.radians(float(getattr(object_info, "heading_deg", 0.0)))
            lane_match = self.lane_map.match(object_info.position, heading=heading)
            if lane_match is not None:
                matches.append((entry, lane_match))
        return matches

    def _front_risk(
        self, vehicle_state, ego_match, object_matches, vehicle_entries=()
    ):
        nearest = float("inf")
        self._front_lead_speed = None
        minimum_ttc = float("inf")
        matched_object_ids = set()
        for entry, object_match in object_matches:
            matched_object_ids.add(id(entry["object_info"]))
            # Stationary body overlap is measured on the path, not by snapping
            # the obstacle center to a lane (a shoulder object may be harmless).
            if self.obstacle_motion.is_static(entry["object_info"]):
                continue
            center_distance = self.lane_map.forward_distance(ego_match, object_match)
            if center_distance is None or center_distance > parameters.FRONT_RELEVANCE_DISTANCE:
                continue
            object_info = entry["object_info"]
            object_length = max(1.0, float(getattr(object_info, "size_x", 0.0)))
            gap = max(
                0.0,
                center_distance
                - 0.5 * parameters.VEHICLE_LENGTH
                - 0.5 * object_length,
            )
            closing_speed = max(
                0.0, vehicle_state.velocity - float(object_info.velocity)
            )
            object_ttc = gap / closing_speed if closing_speed > 0.1 else float("inf")
            if gap < nearest:
                nearest = gap
                self._front_lead_speed = max(0.0, float(object_info.velocity))
            minimum_ttc = min(minimum_ttc, object_ttc)

        # Link matching can briefly fail at merges/splits. Keep ACC available
        # through an ego-local same-lane check so a real lead vehicle is not
        # ignored merely because its MGeo link ID differs for a few frames.
        for entry in vehicle_entries:
            if self.obstacle_motion.is_static(entry["object_info"]):
                continue
            if id(entry["object_info"]) in matched_object_ids:
                continue
            local_position = entry.get("local_position")
            if local_position is None or local_position.x <= 0.0:
                continue
            if abs(local_position.y) > parameters.ACC_FALLBACK_LATERAL_THRESHOLD:
                continue
            object_info = entry["object_info"]
            object_length = max(1.0, float(getattr(object_info, "size_x", 0.0)))
            gap = max(
                0.0,
                local_position.x
                - 0.5 * parameters.VEHICLE_LENGTH
                - 0.5 * object_length,
            )
            if gap > parameters.FRONT_RELEVANCE_DISTANCE:
                continue
            closing_speed = max(
                0.0, vehicle_state.velocity - float(object_info.velocity)
            )
            object_ttc = gap / closing_speed if closing_speed > 0.1 else float("inf")
            if gap < nearest:
                nearest = gap
                self._front_lead_speed = max(0.0, float(object_info.velocity))
            minimum_ttc = min(minimum_ttc, object_ttc)
        return nearest, minimum_ttc

    @staticmethod
    def _acc_speed_limit(ego_speed, gap, ttc, lead_speed=None):
        """Return an explicit longitudinal speed ceiling for ACC_FOLLOW."""
        if math.isinf(gap):
            return float("inf")
        closing_speed = 0.0 if math.isinf(ttc) else gap / max(0.1, ttc)
        lead_speed = max(0.0, ego_speed - closing_speed) if lead_speed is None else lead_speed
        desired_gap = (
            parameters.ACC_STANDSTILL_DISTANCE
            + max(0.0, ego_speed) * parameters.ACC_TIME_GAP
        )
        if math.isinf(ttc) and gap >= desired_gap:
            # Once the lead vehicle has opened a safe gap, remove the ceiling.
            # Capping by current Ego speed here would lock a stopped car at 0.
            return float("inf")
        speed_limit = (
            lead_speed
            + parameters.ACC_DISTANCE_GAIN * (gap - desired_gap)
        )
        if lead_speed > ego_speed + 0.1 and gap >= parameters.ACC_STANDSTILL_DISTANCE:
            speed_limit = max(ego_speed, speed_limit)
        return max(0.0, speed_limit)
