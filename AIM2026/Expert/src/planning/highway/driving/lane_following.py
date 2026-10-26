"""기준 차로 매칭·전방 위험·ACC 속도 처리.

HighwayLanePlanner의 동일 인스턴스 상태를 사용하는 메서드 묶음이다.
생성자나 별도 상태를 추가하지 않으며, 기존 메서드 본문과 호출 순서를 유지한다.
"""
import math
from copy import copy
from ....interfaces.geometry import Point
from ....config.parameters import highway as parameters


class LaneFollowingMixin:
    def _keep_lane_observations(self, ego, ego_match, entries):
        """Separate keep-lane braking evidence from raw lane-change evidence.

        Under the explicit no-cut-in assumption, a recently established
        parallel neighbor cannot jump sideways by a metre in one observation.
        Keep its measured longitudinal position and last lateral offset for at
        most 0.5 s. Only keep-lane checks see this view; all raw boxes remain in
        lane-change admission and committed-maneuver checks.
        """
        if not parameters.ASSUME_HIGHWAY_VEHICLES_KEEP_LANE:
            self._lane_observation_memory.clear()
            return entries
        self._lane_observation_clock += self.control_dt
        now = self._lane_observation_clock
        memory = self._lane_observation_memory
        for key in list(memory):
            if now-memory[key]['seen'] > .5:
                del memory[key]
        ordinary = self.state not in (self.CHANGE_LEFT, self.CHANGE_RIGHT)
        self._keep_lane_evidence = dict(corrected=[], duplicate=[], side_only=[])
        output = []
        neighbors = []
        for entry in entries:
            obj = entry['object_info']
            key = getattr(obj, 'unique_id', None)
            old = memory.get(key)
            stamp = (getattr(obj, 'observation_x', obj.position.x),
                     getattr(obj, 'observation_y', obj.position.y))
            match = self.lane_map.match(obj.position, heading=math.radians(obj.heading_deg))
            adjusted = obj
            if old is not None:
                lane = self.lane_map.links[old['lane']]
                projected = self.lane_map.project(obj.position, lane)
                # Reuse the existing topology/heading checks: exclude actual
                # convergence, our lane and crossing/non-moving traffic.
                neighbor = self._parallel_lane_margin(ego, ego_match, obj, projected) == 0.
                offset = (-(obj.position.x-projected.x)*math.sin(projected.yaw)
                          +(obj.position.y-projected.y)*math.cos(projected.yaw))
                if ordinary and neighbor:
                    if stamp != old['stamp'] and abs(offset-old['offset']) <= .6:
                        # A low-pass lateral anchor also rejects several small
                        # consecutive shifts that together fabricate a cut-in.
                        old['offset'] += .15*(offset-old['offset'])
                        old.update(seen=now, stamp=stamp)
                    adjusted = copy(obj)
                    adjusted.position = Point(projected.x-old['offset']*math.sin(projected.yaw),
                                              projected.y+old['offset']*math.cos(projected.yaw))
                    # An established non-converging neighbor belongs to the
                    # adjacent corridor while Ego is centered in its own lane.
                    # Small footprint inflation must not turn it into an ACC
                    # obstacle. Never carry this exception into our maneuver.
                    adjusted.keep_lane_side_only = ego_match.distance <= .5
                    if adjusted.keep_lane_side_only:
                        self._keep_lane_evidence['side_only'].append(key)
                    if abs(offset-old['offset']) > .05:
                        self._keep_lane_evidence['corrected'].append(key)
                    # Do not refresh the timeout from repeated bad observations.
                elif stamp != old['stamp']:
                    old = None
            if old is None and key is not None and match is not None and not getattr(obj, 'motion_uncertain', False):
                offset = (-(obj.position.x-match.x)*math.sin(match.yaw)
                          +(obj.position.y-match.y)*math.cos(match.yaw))
                memory[key] = dict(lane=match.lane_link.id, offset=offset, seen=now, stamp=stamp)
            view = dict(entry, object_info=adjusted)
            if adjusted is not obj and 'local_position' in view:
                dx, dy = adjusted.position.x-ego.position.x, adjusted.position.y-ego.position.y
                view['local_position'] = Point(dx*math.cos(ego.yaw)+dy*math.sin(ego.yaw),
                                              -dx*math.sin(ego.yaw)+dy*math.cos(ego.yaw))
            output.append(view)
            adjusted_match = self.lane_map.match(adjusted.position, heading=math.radians(adjusted.heading_deg))
            if (ordinary and not getattr(obj, 'motion_uncertain', False)
                    and self._parallel_lane_margin(ego, ego_match, adjusted, adjusted_match) == 0.):
                neighbors.append(adjusted)
        if not ordinary:
            return entries
        result = []
        for entry in output:
            obj = entry['object_info']
            supports = []
            if getattr(obj, 'motion_uncertain', False) and not getattr(obj, 'speed_unknown', False) and obj.velocity >= 2.:
                for other in neighbors:
                    angle = math.radians(other.heading_deg)
                    dx, dy = obj.position.x-other.position.x, obj.position.y-other.position.y
                    size_ratio = max(max(getattr(obj,k),getattr(other,k))/max(.1,min(getattr(obj,k),getattr(other,k)))
                                     for k in ('size_x','size_y'))
                    if (size_ratio <= 1.35
                            and abs(self.lane_map._angle_difference(math.radians(obj.heading_deg),angle)) < math.radians(10.)
                            and abs(obj.velocity-other.velocity) <= max(3., .3*other.velocity)
                            and abs(dx*math.cos(angle)+dy*math.sin(angle)) < .5*min(obj.size_x,other.size_x)
                            and abs(-dx*math.sin(angle)+dy*math.cos(angle)) < .8*min(obj.size_y,other.size_y)):
                        supports.append(other)
            if len(supports) == 1:
                self._keep_lane_evidence['duplicate'].append(obj.unique_id)
            else:
                result.append(entry)
        return result

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
                    and getattr(self.active_candidate, 'static_corridor', False)):
                # Retain the verified adjacent corridor target during motion.
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
            if getattr(entry['object_info'], 'keep_lane_side_only', False):
                continue
            # Stationary body overlap is measured on the path, not by snapping
            # the obstacle center to a lane (a shoulder object may be harmless).
            if self.obstacle_motion.is_static(entry["object_info"]):
                continue
            center_distance = self.lane_map.forward_distance(ego_match, object_match)
            if center_distance is None or center_distance > parameters.FRONT_RELEVANCE_DISTANCE:
                continue
            object_info = entry["object_info"]
            if (parameters.ASSUME_HIGHWAY_VEHICLES_KEEP_LANE
                    and self.state not in (self.CHANGE_LEFT, self.CHANGE_RIGHT)
                    and object_match.lane_link.id != ego_match.lane_link.id):
                # A reachable branch is not necessarily the lane Ego commands.
                # Once the object's body is clear of our actual lane corridor,
                # it must not remain an ACC lead merely through graph reachability.
                point, yaw, _ = self.lane_map.advance_default(ego_match, center_distance)
                _, _, _, overlaps = self.merge_safety.body_gap(
                    point, yaw, object_info.position, math.radians(object_info.heading_deg),
                    object_info, lateral_margin=0.)
                if not overlaps:
                    continue
            object_length = max(1.0, float(getattr(object_info, "size_x", 0.0)))
            gap = max(
                0.0,
                center_distance
                - parameters.VEHICLE_LENGTH
                - 0.5 * object_length,
            )
            # Use available speed consistently with trajectory prediction.
            # A missing speed still needs a conservative following bound.
            lead_speed = 0.0 if getattr(object_info, 'speed_unknown', False) else max(0.0, float(object_info.velocity))
            closing_speed = max(0.0, vehicle_state.velocity - lead_speed)
            object_ttc = gap / closing_speed if closing_speed > 0.1 else float("inf")
            if gap < nearest:
                nearest = gap
                self._front_lead_speed = lead_speed
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
                - parameters.VEHICLE_LENGTH
                - 0.5 * object_length,
            )
            if gap > parameters.FRONT_RELEVANCE_DISTANCE:
                continue
            lead_speed = 0.0 if getattr(object_info, 'speed_unknown', False) else max(0.0, float(object_info.velocity))
            closing_speed = max(0.0, vehicle_state.velocity - lead_speed)
            object_ttc = gap / closing_speed if closing_speed > 0.1 else float("inf")
            if gap < nearest:
                nearest = gap
                self._front_lead_speed = lead_speed
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
        # Ego position is the rear axle. Reserve room to shed relative speed
        # before reaching the minimum gap, including actuator response delay.
        # A short time-gap setting alone cannot stop a fast approach safely.
        closing = max(0.0, ego_speed - lead_speed)
        desired_gap += (closing * parameters.LANE_CHANGE_CONTROL_DELAY
                        + closing * closing / (2.0 * parameters.ACC_APPROACH_DECEL))
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
