#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Small lane graph compiled from the measured highway MGeo links."""

import io
import json
import math
import os
import bisect
from collections import deque

from ...localization.point import Point
from . import parameters
from .structures import HighwayLaneLink, LaneMatch


class HighwayLaneMap:
    def __init__(self, map_data_dir):
        link_path = os.path.join(map_data_dir, "link_set.json")
        with io.open(link_path, "r", encoding="utf-8-sig") as stream:
            raw_links = json.load(stream)
        section_by_road = dict(
            (str(road_id), int(section))
            for section, road_id in parameters.SECTION_ROADS
        )
        selected = []
        for raw in raw_links:
            link_id = str(raw.get("idx") or "")
            road_id = str(raw.get("road_id") or "")
            if link_id == parameters.ENTRY_LINK_ID:
                selected.append((raw, 0))
            elif road_id in section_by_road:
                selected.append((raw, section_by_road[road_id]))

        self.links = {}
        for raw, section_index in selected:
            points = [Point(float(p[0]), float(p[1])) for p in raw.get("points", [])]
            if len(points) < 2:
                continue
            cumulative_s = [0.0]
            for start, end in zip(points[:-1], points[1:]):
                cumulative_s.append(cumulative_s[-1] + start.distance(end))
            lane_number = int(raw.get("ego_lane") or 1)
            excluded = section_index > 0 and lane_number == 1
            lane_link = HighwayLaneLink(
                raw, section_index, points, cumulative_s, excluded
            )
            self.links[lane_link.id] = lane_link

        required_ids = set([parameters.ENTRY_LINK_ID])
        for source, destinations in parameters.SUCCESSOR_LINKS.items():
            required_ids.add(source)
            required_ids.update(destinations)
        missing = sorted(required_ids.difference(self.links))
        if missing:
            raise KeyError("Highway lane link(s) missing: {}".format(", ".join(missing)))

        self.successors = {}
        for link_id in self.links:
            self.successors[link_id] = tuple(
                target
                for target in parameters.SUCCESSOR_LINKS.get(link_id, ())
                if target in self.links
            )
        self.max_section_index = max(
            lane_link.section_index for lane_link in self.links.values()
        )

    def match(self, position, heading=None, preferred_ids=()):
        preferred = set(preferred_ids or ())
        best_match = None
        best_score = float("inf")
        for lane_link in self.links.values():
            match = self.project(position, lane_link)
            heading_cost = 0.0
            if heading is not None:
                heading_cost = 0.25 * abs(self._angle_difference(heading, match.yaw))
            preference_bonus = -0.35 if lane_link.id in preferred else 0.0
            score = match.distance + heading_cost + preference_bonus
            if score < best_score:
                best_score = score
                best_match = match
        if (
            best_match is None
            or best_match.distance > parameters.LANE_MATCH_MAX_DISTANCE
        ):
            return None
        return best_match

    def project(self, position, lane_link):
        best = None
        for index, (start, end) in enumerate(
            zip(lane_link.points[:-1], lane_link.points[1:])
        ):
            dx = end.x - start.x
            dy = end.y - start.y
            length_sq = dx * dx + dy * dy
            if length_sq < 1e-9:
                continue
            ratio = (
                (position.x - start.x) * dx + (position.y - start.y) * dy
            ) / length_sq
            ratio = max(0.0, min(1.0, ratio))
            x = start.x + ratio * dx
            y = start.y + ratio * dy
            distance = math.hypot(position.x - x, position.y - y)
            if best is None or distance < best.distance:
                segment_length = math.sqrt(length_sq)
                best = LaneMatch(
                    lane_link,
                    lane_link.cumulative_s[index] + ratio * segment_length,
                    distance,
                    x,
                    y,
                    math.atan2(dy, dx),
                )
        return best

    def legal_neighbors(self, lane_link):
        neighbors = []
        for direction, target_id, allowed in (
            ("left", lane_link.left_id, lane_link.can_move_left),
            ("right", lane_link.right_id, lane_link.can_move_right),
        ):
            target = self.links.get(str(target_id)) if target_id else None
            if target is None or not allowed or target.excluded:
                continue
            if target.section_index != lane_link.section_index:
                continue
            # 고속도로 중간에 끝나거나 빠져나가는 차선은 회피 목표로 사용하지 않습니다.
            if (
                target.section_index < self.max_section_index
                and not self.successors.get(target.id, ())
            ):
                continue
            neighbors.append((direction, target))
        return neighbors

    def default_successor(self, link_id):
        successors = self.successors.get(link_id, ())
        return successors[0] if successors else None

    def forward_distance(self, start_match, target_match, max_depth=6):
        if start_match.lane_link.id == target_match.lane_link.id:
            distance = target_match.s - start_match.s
            return distance if distance >= 0.0 else None

        queue = deque()
        initial = start_match.lane_link.length - start_match.s
        for successor in self.successors.get(start_match.lane_link.id, ()):
            queue.append((successor, initial, 1))
        visited = set()
        while queue:
            link_id, distance_to_start, depth = queue.popleft()
            if (link_id, depth) in visited or depth > max_depth:
                continue
            visited.add((link_id, depth))
            lane_link = self.links[link_id]
            if link_id == target_match.lane_link.id:
                return distance_to_start + target_match.s
            for successor in self.successors.get(link_id, ()):
                queue.append((successor, distance_to_start + lane_link.length, depth + 1))
        return None

    def advance_options(self, lane_match, distance, max_depth=6):
        return self._advance_from(
            lane_match.lane_link, lane_match.s + max(0.0, distance), 0, max_depth
        )

    def _advance_from(self, lane_link, target_s, depth, max_depth):
        if target_s <= lane_link.length or depth >= max_depth:
            return [self.point_at_s(lane_link, min(target_s, lane_link.length))]
        remaining = target_s - lane_link.length
        successors = self.successors.get(lane_link.id, ())
        if not successors:
            return [self.point_at_s(lane_link, lane_link.length)]
        results = []
        for successor in successors:
            results.extend(
                self._advance_from(self.links[successor], remaining, depth + 1, max_depth)
            )
        return results

    def route_points(self, lane_match, distance, spacing):
        points = []
        travelled = 0.0
        while travelled <= distance + 1e-6:
            # Ego 경로에는 분기에서 첫 번째(본선) successor를 사용합니다.
            points.append(self.advance_default(lane_match, travelled))
            travelled += spacing
        return points

    def advance_default(self, lane_match, distance):
        lane_link = lane_match.lane_link
        target_s = lane_match.s + max(0.0, distance)
        depth = 0
        while target_s > lane_link.length and depth < 8:
            target_s -= lane_link.length
            successor = self.default_successor(lane_link.id)
            if successor is None:
                target_s = lane_link.length
                break
            lane_link = self.links[successor]
            depth += 1
        return self.point_at_s(lane_link, min(target_s, lane_link.length))

    @staticmethod
    def point_at_s(lane_link, target_s):
        target_s = max(0.0, min(float(target_s), lane_link.length))
        index = bisect.bisect_left(lane_link.cumulative_s, target_s, lo=1)
        if index < len(lane_link.cumulative_s):
            start_s = lane_link.cumulative_s[index - 1]
            end_s = lane_link.cumulative_s[index]
            ratio = 0.0 if end_s <= start_s else (target_s - start_s) / (end_s - start_s)
            start = lane_link.points[index - 1]
            end = lane_link.points[index]
            point = Point(
                start.x + ratio * (end.x - start.x),
                start.y + ratio * (end.y - start.y),
            )
            return point, math.atan2(end.y - start.y, end.x - start.x), lane_link
        start = lane_link.points[-2]
        end = lane_link.points[-1]
        return Point(end.x, end.y), math.atan2(end.y - start.y, end.x - start.x), lane_link

    @staticmethod
    def _angle_difference(first, second):
        return math.atan2(math.sin(first - second), math.cos(first - second))
