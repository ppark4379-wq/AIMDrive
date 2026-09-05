from collections import defaultdict, deque
import math


class LinkGraph:
    def __init__(self, map_loader):
        self.map_loader = map_loader
        self.links = map_loader.link_set
        self.adj = defaultdict(list)
        self._build_graph()

    def _as_list(self, value):
        if value is None:
            return []
        if isinstance(value, list):
            return value
        return [value]

    def _extract_link_id(self, item):
        if isinstance(item, str):
            return item
        if isinstance(item, dict):
            for k in ["idx", "id", "link_id", "value"]:
                if k in item:
                    return item[k]
        return None

    def _build_graph(self):
        """
        MGeo link_set에서 연결 그래프 생성.
        1순위: link 내부의 to_links류 필드
        2순위: from_node_idx / to_node_idx 기반 연결
        3순위: lane change dst link도 연결 후보로 추가
        """
        node_to_outgoing = defaultdict(list)

        for link_id, link in self.links.items():
            from_node = (
                link.get("from_node_idx")
                or link.get("from_node_id")
                or link.get("from_node")
            )
            if from_node:
                node_to_outgoing[from_node].append(link_id)

        for link_id, link in self.links.items():
            next_links = []

            # 직접 연결 필드가 있는 경우
            for key in [
                "to_links",
                "to_link",
                "to_link_list",
                "to_link_ids",
                "dst_links",
                "next_links",
            ]:
                if key in link:
                    for item in self._as_list(link.get(key)):
                        nxt = self._extract_link_id(item)
                        if nxt:
                            next_links.append(nxt)

            # node 기반 연결
            to_node = (
                link.get("to_node_idx")
                or link.get("to_node_id")
                or link.get("to_node")
            )
            if to_node:
                next_links.extend(node_to_outgoing.get(to_node, []))

            # 차선 변경 가능 link도 후보로 추가
            for key in [
                "left_lane_change_dst_link_idx",
                "right_lane_change_dst_link_idx",
                "lane_ch_link_left",
                "lane_ch_link_right",
            ]:
                nxt = link.get(key)
                if nxt:
                    next_links.append(nxt)

            # 중복/자기자신 제거
            clean = []
            for nxt in next_links:
                if nxt and nxt != link_id and nxt in self.links and nxt not in clean:
                    clean.append(nxt)

            self.adj[link_id] = clean

    def shortest_path(self, start_link, end_link, max_depth=5000):
        if start_link not in self.links:
            raise KeyError(f"Unknown start_link: {start_link}")
        if end_link not in self.links:
            raise KeyError(f"Unknown end_link: {end_link}")

        q = deque()
        q.append(start_link)

        parent = {start_link: None}
        depth = {start_link: 0}

        while q:
            cur = q.popleft()

            if cur == end_link:
                break

            if depth[cur] >= max_depth:
                continue

            for nxt in self.adj.get(cur, []):
                if nxt not in parent:
                    parent[nxt] = cur
                    depth[nxt] = depth[cur] + 1
                    q.append(nxt)

        if end_link not in parent:
            return None

        path = []
        cur = end_link
        while cur is not None:
            path.append(cur)
            cur = parent[cur]

        path.reverse()
        return path

    def link_length(self, link_id):
        points = self.links[link_id]["points"]
        total = 0.0
        for p0, p1 in zip(points[:-1], points[1:]):
            total += math.hypot(p1[0] - p0[0], p1[1] - p0[1])
        return total

    def path_length(self, route_links):
        return sum(self.link_length(lid) for lid in route_links)
