import json
import math
import os


class MGeoMapLoader:
    def __init__(self, mgeo_root: str):
        self.mgeo_root = mgeo_root
        self.link_set = self._load_json("link_set.json")
        self.node_set = self._load_json("node_set.json")
        self.singlecrosswalk_set = self._load_json_optional("singlecrosswalk_set.json")
        self.crosswalk_set = self._load_json_optional("crosswalk_set.json")

    def _load_json(self, filename: str):
        path = os.path.join(self.mgeo_root, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"MGeo file not found: {path}")

        with open(path, "r") as f:
            data = json.load(f)

        result = {}
        for item in data:
            idx = item.get("idx")
            if idx is not None:
                result[idx] = item
        return result

    def _load_json_optional(self, filename: str):
        path = os.path.join(self.mgeo_root, filename)
        if not os.path.exists(path):
            return {}
        with open(path, "r") as f:
            data = json.load(f)
        result = {}
        for item in data:
            idx = item.get("idx")
            if idx is not None:
                result[idx] = item
        return result

    def get_link(self, link_id: str):
        if link_id not in self.link_set:
            raise KeyError(f"Unknown link_id: {link_id}")
        return self.link_set[link_id]

    def get_link_points(self, link_id: str):
        link = self.get_link(link_id)
        return link["points"]

    def get_lane_group_link_ids(self, link_id: str):
        """link_id 자신 + 좌/우 차로 변경으로 넘어갈 수 있는 인접 링크 id."""
        links = {link_id}
        link = self.link_set.get(link_id)
        if link is None:
            return links

        left_id = link.get("left_lane_change_dst_link_idx")
        if left_id:
            links.add(left_id)

        right_id = link.get("right_lane_change_dst_link_idx")
        if right_id:
            links.add(right_id)

        return links

    def _crosswalk_spawn_info_from_points(self, pts, label: str):
        # 마지막 점이 첫 점과 같으면(polygon 닫힘) 제거
        if len(pts) >= 2 and pts[0][:2] == pts[-1][:2]:
            pts = pts[:-1]
        if len(pts) < 4:
            raise ValueError(f"Crosswalk {label} needs at least 4 points")

        # 4점 사각형에서 두 쌍의 마주보는 변을 비교해 이동 거리가 긴 쪽을 횡단 방향으로 선택
        # pair A: p0-p3 ↔ p1-p2  (인접한 첫/마지막 vs 나머지)
        # pair B: p0-p1 ↔ p2-p3  ([:2] vs [2:])
        def midpoint(pa, pb):
            return (pa[0] + pb[0]) / 2, (pa[1] + pb[1]) / 2, (pa[2] + pb[2]) / 2

        ma1 = midpoint(pts[0], pts[-1])
        mb1 = midpoint(pts[1], pts[-2])
        dist1 = math.hypot(mb1[0] - ma1[0], mb1[1] - ma1[1])

        ma2 = midpoint(pts[0], pts[1])
        mb2 = midpoint(pts[2], pts[-1])
        dist2 = math.hypot(mb2[0] - ma2[0], mb2[1] - ma2[1])

        if dist1 >= dist2:
            side_a_mid, side_b_mid = ma1, mb1
        else:
            side_a_mid, side_b_mid = ma2, mb2

        sx, sy, sz = side_a_mid
        ex, ey = side_b_mid[0], side_b_mid[1]

        dx, dy = ex - sx, ey - sy
        move_dist = math.hypot(dx, dy)
        if move_dist < 1e-3:
            raise ValueError(f"Crosswalk {label} has zero width")
        return sx, sy, ex, ey, sz, dx / move_dist, dy / move_dist, move_dist

    def get_singlecrosswalk_spawn_info(self, singlecrosswalk_id: str):
        """singlecrosswalk 양쪽 에지 중점을 반환한다."""
        if singlecrosswalk_id not in self.singlecrosswalk_set:
            raise KeyError(f"Unknown singlecrosswalk_id: {singlecrosswalk_id}")
        scw = self.singlecrosswalk_set[singlecrosswalk_id]
        return self._crosswalk_spawn_info_from_points(scw["points"], singlecrosswalk_id)

    def get_crosswalk_spawn_info(self, crosswalk_id: str):
        """횡단보도 양쪽 에지 중점을 반환: (start_x, start_y, end_x, end_y, z, dir_x, dir_y, move_dist)"""
        if crosswalk_id not in self.crosswalk_set:
            raise KeyError(f"Unknown crosswalk_id: {crosswalk_id}")
        cw = self.crosswalk_set[crosswalk_id]
        scw_ids = cw.get("single_crosswalk_list", [])
        if not scw_ids:
            raise ValueError(f"Crosswalk {crosswalk_id} has no singlecrosswalks")

        return self.get_singlecrosswalk_spawn_info(scw_ids[0])
