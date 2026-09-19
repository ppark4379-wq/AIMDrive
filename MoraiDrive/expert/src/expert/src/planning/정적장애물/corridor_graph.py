"""Whole-corridor A* with intermediate lateral and heading states.

Uses the recorded/global swept envelope, not unobserved road space. Inspired by
HL_VTD's precomputed gateway search: intermediate offsets, multi-length quintic
edges, and full-footprint validation before committing a complete route.
"""

import heapq
import math
import time

import numpy as np


# =============================================================================
# 정적 장애물 통로 플래닝: 탐색 격자와 방향 후보
# 기록 경로 연결로 해결되지 않을 때 통로 전체의 횡위치·방향을 탐색한다.
# =============================================================================
LONGITUDINAL_STEP = 1.0               # 탐색 상태를 배치할 전역 y 방향 간격 [m]
LATERAL_STEP = .1                     # 각 y 위치에서 생성할 횡방향 x 후보 간격 [m]
HEADING_SLOPES = tuple(np.arange(-.5, .51, .1)) # 전역 경로의 dx/dy 기울기에 더할 방향 후보 [무차원]

# 탐색 시간·우선순위와 차체 여유
SEARCH_TIMEOUT = 8.0                 # 횡방향 격자 탐색의 시간 제한; 선행 기록 경로 탐색 시간은 별도 [s]
HEURISTIC_WEIGHT = 1.15               # 남은 거리 추정 비용의 가중치; 클수록 목표 방향 탐색 우선
BODY_TRACKING_RESERVE = .15           # 기록된 통로 경계 안에 차체를 유지하기 위한 추종 여유 [m]


def search_recordings(owner, ego, obstacles, connector):
    """Preserve exact recorded-curve edges before expanding the lateral grid."""
    begin = time.monotonic()
    start, goal = float(ego.position.y), owner.rejoin_y
    if goal <= start:
        return None
    levels = np.unique(np.r_[np.arange(start, goal, 3.), goal])
    source = (-1, 0)
    queue, costs, previous = [(goal-start, 0., source)], {source: 0.}, {}
    while queue and time.monotonic()-begin < 2.:
        _, cost, node = heapq.heappop(queue)
        if cost != costs[node]:
            continue
        route_id, level = node
        if level == len(levels)-1:
            pieces, names = [], []
            while node != source:
                names.append(owner.records[node[0]][0])
                node, edge = previous[node]
                pieces.append(edge)
            pieces.reverse()
            tail = owner._edge(owner.global_curve, goal, owner.goal_y, obstacles)
            if tail is None:
                return None
            owner.selected_names = list(dict.fromkeys(reversed(names)))
            owner.search_details = dict(method='recorded_connections', status='complete',
                                        elapsed_s=time.monotonic()-begin)
            return np.vstack([pieces[0]]+[p[1:] for p in pieces[1:]]+[tail[1:]])
        a = owner._state(owner.curves[route_id], levels[level]) if route_id >= 0 else (
            start, ego.position.x, math.cos(ego.yaw)/max(.5, math.sin(ego.yaw)), 0.)
        for advance in (1, 2, 3, 4):
            target_level = level+advance
            if target_level >= len(levels):
                continue
            for target_id, curve in enumerate(owner.curves):
                if route_id == target_id and advance != 1:
                    continue
                target = (target_id, target_level)
                penalty = 1.5 if route_id >= 0 and route_id != target_id else 0.
                if cost+levels[target_level]-levels[level]+penalty >= costs.get(target, float('inf')):
                    continue
                transfer = curve if route_id == target_id else connector(a, owner._state(curve, levels[target_level]))
                edge = owner._edge(transfer, levels[level], levels[target_level], obstacles)
                if edge is None:
                    continue
                new_cost = cost+float(np.linalg.norm(np.diff(edge, axis=0), axis=1).sum())+penalty
                if new_cost < costs.get(target, float('inf')):
                    costs[target], previous[target] = new_cost, (node, edge)
                    heapq.heappush(queue, (new_cost+HEURISTIC_WEIGHT*(goal-levels[target_level]), new_cost, target))
    return None


def search_corridor(owner, ego, obstacles, connector, max_curvature):
    recorded = search_recordings(owner, ego, obstacles, connector)
    if recorded is not None:
        return recorded
    begin = time.monotonic()
    start, goal = float(ego.position.y), owner.rejoin_y
    if goal <= start:
        return None
    levels = np.r_[start, np.arange(start+LONGITUDINAL_STEP, goal-1., LONGITUDINAL_STEP), goal]
    nodes = [np.array([[start, ego.position.x, math.cos(ego.yaw)/max(.5, math.sin(ego.yaw)), 0.]])]
    for y in levels[1:]:
        base = owner._state(owner.global_curve, y)
        if y == goal:
            states = np.array([base])
        else:
            recorded = [owner._state(c, y) for c in owner.curves]
            low, high = min(v[1] for v in recorded+[base]), max(v[1] for v in recorded+[base])
            xs = np.unique(np.r_[np.arange(low, high, LATERAL_STEP), high, base[1], [v[1] for v in recorded]])
            states = np.array([(y, x, base[2]+slope, base[3]) for x in xs for slope in HEADING_SLOPES] + recorded)
        xy, yaw = states[:, [1, 0]], np.arctan2(1., states[:, 2])
        valid = owner._envelope_mask(xy, yaw, BODY_TRACKING_RESERVE) & owner._collision_mask(xy, yaw, obstacles, .12)
        nodes.append(states[valid])

    source = (0, 0)
    queue, costs, previous = [(HEURISTIC_WEIGHT*(goal-start), 0., source)], {source: 0.}, {}
    expanded = checked = 0
    end = None
    while queue:
        if time.monotonic()-begin > SEARCH_TIMEOUT:
            break
        _, cost, node = heapq.heappop(queue)
        if cost != costs[node]:
            continue
        level, index = node
        if level == len(levels)-1:
            end = node
            break
        expanded += 1
        owner.search_progress = max(owner.search_progress, level)
        a = nodes[level][index]
        for advance in (4, 6, 8, 12, 2, 3):
            target_level = level+advance
            if target_level >= len(levels) or not len(nodes[target_level]):
                continue
            targets = nodes[target_level]
            span = levels[target_level]-a[0]
            if span < 1.:
                continue
            dx = targets[:, 1]-a[1]
            estimated = cost + np.hypot(span, dx) + .15*np.abs(dx) + .15*np.abs(targets[:, 2]-a[2])
            eligible = np.array([estimated[i] < costs.get((target_level, i), float('inf'))
                                 for i in range(len(targets))])
            eligible &= np.abs(dx) <= .65*span
            indices = np.flatnonzero(eligible)
            if not len(indices):
                continue
            batch = targets[indices]
            curves = connector(tuple(a), (levels[target_level], batch[:, 1], batch[:, 2], batch[:, 3]))
            # Coarse probes only reject edges; acceptance always uses _edge's
            # dense swept OBB and envelope checks below.
            ys = np.linspace(a[0], levels[target_level], max(3, int(math.ceil(span/.7))+1))
            xs, slopes = curves(ys), curves(ys, 1)
            curvature = np.abs(curves(ys, 2))/(1+slopes*slopes)**1.5
            xy = np.column_stack((xs.ravel(), np.tile(ys, len(batch))))
            yaw = np.arctan2(1., slopes.ravel())
            valid = owner._envelope_mask(xy, yaw, BODY_TRACKING_RESERVE) & owner._collision_mask(xy, yaw, obstacles, .12)
            valid = valid.reshape(xs.shape).all(axis=1) & (curvature.max(axis=1) <= max_curvature)
            for i in indices[valid]:
                b = targets[i]
                edge = owner._edge(connector(tuple(a), tuple(b)), a[0], b[0], obstacles,
                                   body_margin=BODY_TRACKING_RESERVE)
                checked += 1
                if edge is None:
                    continue
                target = (target_level, int(i))
                # Weighted A* favors a timely feasible complete path. It does
                # not claim the shortest path; all accepted edges are checked.
                new_cost = float(estimated[i])
                if new_cost < costs.get(target, float('inf')):
                    costs[target], previous[target] = new_cost, (node, edge)
                    heapq.heappush(queue, (new_cost+HEURISTIC_WEIGHT*(goal-b[0]), new_cost, target))
    owner.search_details = dict(method='corridor_astar', expanded=expanded, checked_edges=checked,
                                elapsed_s=time.monotonic()-begin,
                                status='complete' if end is not None else ('timeout' if queue else 'blocked'))
    if end is None:
        return None
    pieces = []
    while end != source:
        end, edge = previous[end]
        pieces.append(edge)
    pieces.reverse()
    tail = owner._edge(owner.global_curve, goal, owner.goal_y, obstacles)
    if tail is None:
        owner.search_details['status'] = 'blocked_global_tail'
        return None
    owner.selected_names = ['corridor_astar']
    return np.vstack([pieces[0]]+[piece[1:] for piece in pieces[1:]]+[tail[1:]])
