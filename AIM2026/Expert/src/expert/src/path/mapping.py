#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TrafficLightStoplineMapper

mgeo 데이터( traffic_light_set / synced_traffic_light_set / link_set / node_set /
stoplane_marking_set )를 읽어 "신호등 idx -> 정지선" 매핑을 구성한다.

알고리즘은 src/bev_map/scripts/bev_map_generator.py 의
  - build_synced_signal_map
  - build_stopline_point_index
  - _find_stoplines_by_ids
  - _find_stoplines_for_link_ids
  - build_traffic_stopline_map
를 그대로 재사용한다.

ACC 사용 편의를 위해 get_stopline_center(traffic_light_idx) 를 제공한다.
- 매핑된 정지선들의 모든 점의 평균을 정지선 중심 Point(x, y) 로 반환
- 매핑이 없으면 None 반환
"""

import os
import json

import numpy as np

from ..localization.point import Point


TRAFFIC_LIGHT_TYPES = {"car", "bus"}

DIRECT_SIGNAL_STOPLINE_IDS = {
    "C119BS010063": ["B219BS010016"],
    "C119BS010064": ["B219BS010016"],
    "SSN000007":    ["B219BS010016"],
}


def _load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


class NodeTrafficLightStoplineMapper(object):
    """Build a traffic-light-to-stop-point map directly from MGeo v3 nodes.

    The newer KATRI map stores the applicable signal ID on each road node.  A
    node with both ``on_stop_line`` and ``traffic_light_id`` is therefore a
    lane-centre stop target; no traffic-light pole position or stop-line
    marking geometry is required for longitudinal control.
    """

    def __init__(self, map_dir):
        self.map_dir = os.path.abspath(map_dir)
        node_set = _load_json(os.path.join(self.map_dir, "node_set.json"))

        self._stop_points = {}
        self._node_count = 0
        for node in node_set:
            if not node.get("on_stop_line"):
                continue

            signal_id = node.get("traffic_light_id")
            point = node.get("point", [])
            if signal_id is None or str(signal_id).strip() == "" or len(point) < 2:
                continue

            signal_id = str(signal_id)
            target = {
                "idx": str(node.get("idx", "")),
                "point": [float(point[0]), float(point[1])]
                + ([float(point[2])] if len(point) >= 3 else []),
                "traffic_light_id": signal_id,
            }
            self._stop_points.setdefault(signal_id, []).append(target)
            self._node_count += 1

    def has_mapping(self, traffic_light_idx):
        return str(traffic_light_idx) in self._stop_points

    def get_stop_points(self, traffic_light_idx):
        return list(self._stop_points.get(str(traffic_light_idx), []))

    def mapped_signal_ids(self):
        return list(self._stop_points.keys())

    @property
    def stop_node_count(self):
        return self._node_count

    def __len__(self):
        return len(self._stop_points)


def _load_json_optional(path):
    if not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _point_key(point):
    if len(point) < 2:
        return None
    return (round(point[0], 3), round(point[1], 3))


def _build_stopline_point_index(stopline_set):
    point_index = {}
    for stopline in stopline_set:
        for point in stopline.get("points", []):
            key = _point_key(point)
            if key is not None:
                point_index.setdefault(key, []).append(stopline)
    return point_index


def _build_synced_signal_map(synced_traffic_light_set):
    mapping = {}
    for synced in synced_traffic_light_set:
        synced_id = synced.get("idx")
        signal_ids = [str(x) for x in synced.get("signal_id_list", [])]
        if synced_id:
            mapping[str(synced_id)] = signal_ids
    return mapping


def _find_stoplines_by_ids(stopline_ids, stopline_by_id):
    stoplines = []
    seen = set()
    for stopline_id in stopline_ids:
        stopline = stopline_by_id.get(stopline_id)
        if stopline is None or stopline_id in seen:
            continue
        seen.add(stopline_id)
        stoplines.append(stopline)
    return stoplines


def _find_stoplines_for_link_ids(link_ids, link_by_id, node_by_id, stopline_point_index):
    stoplines = []
    seen = set()
    for link_id in link_ids:
        link = link_by_id.get(link_id)
        if link is None:
            continue
        for node_id in (link.get("from_node_idx"), link.get("to_node_idx")):
            node = node_by_id.get(node_id)
            if node is None or not node.get("on_stop_line"):
                continue
            key = _point_key(node.get("point", []))
            if key is None:
                continue
            for stopline in stopline_point_index.get(key, []):
                stopline_id = stopline.get("idx")
                if stopline_id in seen:
                    continue
                seen.add(stopline_id)
                stoplines.append(stopline)
    return stoplines


def _build_traffic_stopline_map(traffic_light_set, synced_traffic_light_set,
                                link_set, node_set, stopline_set):
    link_by_id     = {link.get("idx"):     link     for link     in link_set}
    node_by_id     = {node.get("idx"):     node     for node     in node_set}
    stopline_by_id = {sl.get("idx"):       sl       for sl       in stopline_set}
    stopline_point_index = _build_stopline_point_index(stopline_set)

    mapping = {}

    for light in traffic_light_set:
        if light.get("type") not in TRAFFIC_LIGHT_TYPES:
            continue
        light_id = str(light.get("idx"))
        if light_id in DIRECT_SIGNAL_STOPLINE_IDS:
            stoplines = _find_stoplines_by_ids(
                DIRECT_SIGNAL_STOPLINE_IDS[light_id], stopline_by_id)
        else:
            stoplines = _find_stoplines_for_link_ids(
                light.get("link_id_list", []), link_by_id, node_by_id,
                stopline_point_index)
        if stoplines:
            mapping[light_id] = stoplines

    for synced in synced_traffic_light_set:
        synced_id = str(synced.get("idx"))
        if synced_id in DIRECT_SIGNAL_STOPLINE_IDS:
            stoplines = _find_stoplines_by_ids(
                DIRECT_SIGNAL_STOPLINE_IDS[synced_id], stopline_by_id)
            if stoplines:
                mapping[synced_id] = stoplines
            continue
        signal_ids = set(str(x) for x in synced.get("signal_id_list", []))
        if not signal_ids:
            continue
        if not any(light_id in mapping for light_id in signal_ids):
            continue
        stoplines = _find_stoplines_for_link_ids(
            synced.get("link_id_list", []), link_by_id, node_by_id,
            stopline_point_index)
        if stoplines:
            mapping[synced_id] = stoplines

    return mapping


class TrafficLightStoplineMapper(object):
    """신호등 idx -> 정지선 매핑을 보관하고 조회 인터페이스를 제공"""

    def __init__(self, map_dir):
        """map_dir: R_KR_PG_KATRI 폴더의 절대경로"""
        traffic_light_set        = _load_json_optional(os.path.join(map_dir, "traffic_light_set.json"))
        synced_traffic_light_set = _load_json_optional(os.path.join(map_dir, "synced_traffic_light_set.json"))
        link_set                 = _load_json(os.path.join(map_dir, "link_set.json"))
        node_set                 = _load_json(os.path.join(map_dir, "node_set.json"))
        stopline_set             = _load_json_optional(os.path.join(map_dir, "stoplane_marking_set.json"))

        self.synced_signal_map = _build_synced_signal_map(synced_traffic_light_set)
        self._stopline_map = _build_traffic_stopline_map(
            traffic_light_set, synced_traffic_light_set,
            link_set, node_set, stopline_set,
        )
        self._center_cache = {}

    @classmethod
    def for_map_name(cls, map_name):
        """Load one map from the workspace-level map_data directory."""
        current_path = os.path.dirname(os.path.realpath(__file__))
        workspace_root = os.path.normpath(
            os.path.join(current_path, "../../../../")
        )
        map_dir = os.path.join(workspace_root, "map_data", map_name)
        if not os.path.isdir(map_dir):
            map_dir = os.path.join(
                workspace_root, "map_data", "R_KR_PG_KATRI_2025"
            )
        return cls(map_dir)

    def has_mapping(self, traffic_light_idx):
        return str(traffic_light_idx) in self._stopline_map

    def get_stoplines(self, traffic_light_idx):
        """매핑된 정지선 dict 리스트 반환 (없으면 빈 리스트)"""
        return self._stopline_map.get(str(traffic_light_idx), [])

    def get_stopline_center(self, traffic_light_idx):
        """매핑된 정지선들의 모든 점 평균을 Point(x, y) 로 반환. 없으면 None."""
        key = str(traffic_light_idx)
        if key in self._center_cache:
            return self._center_cache[key]

        stoplines = self._stopline_map.get(key, [])
        if not stoplines:
            self._center_cache[key] = None
            return None

        xs, ys = [], []
        for stopline in stoplines:
            for p in stopline.get("points", []):
                if len(p) >= 2:
                    xs.append(p[0])
                    ys.append(p[1])

        if not xs:
            self._center_cache[key] = None
            return None

        center = Point(float(np.mean(xs)), float(np.mean(ys)))
        self._center_cache[key] = center
        return center

    def mapped_signal_ids(self):
        return list(self._stopline_map.keys())

    def __len__(self):
        return len(self._stopline_map)
