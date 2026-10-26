"""Accumulate VE detections in the static-obstacle section into a persistent occupancy map.

Planning on instantaneous VE boxes fails in this section: boxes jitter by 10-30 cm, grow near range, vanish for a
frame and change track ID, so a path planned on one snapshot is invalidated by the next. Here the world is static and
GPS is accurate, so evidence is accumulated per map cell instead: a cell covered by a box gains log-odds, a visible
uncovered cell loses some, and only cells with repeated evidence are occupied. The planner receives one box per
occupied cluster, the same geometry for planning and for re-checking a committed path. No heading is trusted.

Misses are counted only inside the VE detection window and outside the shadow of the current boxes, so occluded
objects are not erased. The region is the static section of ``sections.py``; boxes elsewhere pass through unchanged.
"""
import math

import numpy as np
from scipy import ndimage
from scipy.spatial import ConvexHull, QhullError

from AIM2026.Expert.src.config.parameters.sections import STATIC_OBSTACLE_CERTAIN_X, STATIC_OBSTACLE_CERTAIN_Y

CELL_M = .1               # grid resolution
HIT = .85                 # log-odds added to a cell covered by a detection in one frame
MISS = .6                 # log-odds removed from a visible, unoccluded, uncovered cell in one frame;
                          # a cell must be covered in about 40 % of the frames it is visible to stay occupied
CLAMP = (-2., 4.)         # log-odds bounds: an object seen for long is released within about 1.5 s of absence
OCCUPIED = 1.5            # a cell becomes occupied at this log-odds (two net hits) ...
RELEASE = .4              # ... and stays occupied until it falls below this
VISIBLE_FORWARD = (2., 25.)  # ego-frame window where a missing detection counts as free space [m]
VISIBLE_LATERAL = 12.     # (inside the VE BEV extent: forward -20..30, lateral +-25)
MIN_CELLS = 3             # smaller clusters are ignored
MIN_FILL = .5             # a cluster filling less of its rectangle is split along its long axis (merged neighbours)
MAX_SPLIT_DEPTH = 3
MATCH_M = 1.5             # a cluster keeps its ID if its centre moved less than this
ID_BASE = 1_000_000       # positive IDs, disjoint from tracker serials (VE-unknown contract needs id > 0)


class StaticObstacleMap:
    def __init__(self, x_range=STATIC_OBSTACLE_CERTAIN_X, y_range=STATIC_OBSTACLE_CERTAIN_Y, cell=CELL_M):
        self.x0, self.y0, self.cell = float(x_range[0]), float(y_range[0]), float(cell)
        self.x1, self.y1 = float(x_range[1]), float(y_range[1])
        self.shape = (int(math.ceil((self.y1-self.y0)/cell)), int(math.ceil((self.x1-self.x0)/cell)))
        rows, cols = np.indices(self.shape)
        self.cx = self.x0 + (cols+.5)*cell
        self.cy = self.y0 + (rows+.5)*cell
        self.reset()

    def reset(self):
        self.log_odds = np.zeros(self.shape)
        self.occupied = np.zeros(self.shape, dtype=bool)
        self.clusters = {}          # id -> (x, y)
        self.next_id = 0
        self.generation = None
        self.version = 0            # increments whenever the occupied cells change
        self._field = None

    def distance(self, x, y):
        """Lower bound [m] of the distance from world points to the nearest occupied cell area.

        Points outside the map, or with no occupied cell, are at infinite distance.
        """
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        if self._field is None or self._field[0] != self.version:
            field = (ndimage.distance_transform_edt(~self.occupied)*self.cell if self.occupied.any()
                     else np.full(self.shape, np.inf))
            self._field = (self.version, field)
        col = np.floor((x-self.x0)/self.cell).astype(int)
        row = np.floor((y-self.y0)/self.cell).astype(int)
        inside = (row >= 0) & (row < self.shape[0]) & (col >= 0) & (col < self.shape[1])
        out = np.full(np.broadcast(x, y).shape, np.inf)
        # A point anywhere in its cell, to the edge of an occupied cell: subtract one cell diagonal.
        out[inside] = self._field[1][row[inside], col[inside]] - math.sqrt(2.)*self.cell
        return out

    def contains(self, x, y):
        return self.x0 <= x <= self.x1 and self.y0 <= y <= self.y1

    def _window(self, x0, x1, y0, y1):
        """Grid slices covering a world rectangle (clipped to the map)."""
        r0, r1 = (int(np.clip(math.floor((v-self.y0)/self.cell), 0, self.shape[0])) for v in (y0, y1+self.cell))
        c0, c1 = (int(np.clip(math.floor((v-self.x0)/self.cell), 0, self.shape[1])) for v in (x0, x1+self.cell))
        return slice(r0, r1), slice(c0, c1)

    def _cover(self, hits, box):
        x, y = box['center_world']
        a = math.radians(box['heading_deg'])
        reach = .5*math.hypot(box['length_m'], box['width_m'])
        window = self._window(x-reach, x+reach, y-reach, y+reach)
        dx, dy = self.cx[window]-x, self.cy[window]-y
        along = dx*math.cos(a)+dy*math.sin(a)
        across = -dx*math.sin(a)+dy*math.cos(a)
        hits[window] |= (np.abs(along) <= .5*box['length_m']) & (np.abs(across) <= .5*box['width_m'])

    def _visible(self, ego, boxes):
        """Visible map cells as (window, mask); only the window around the VE detection area is computed."""
        c, s = math.cos(ego['yaw_grid_rad']), math.sin(ego['yaw_grid_rad'])
        reach = math.hypot(VISIBLE_FORWARD[1], VISIBLE_LATERAL)
        window = self._window(ego['x']-reach, ego['x']+reach, ego['y']-reach, ego['y']+reach)
        dx, dy = self.cx[window]-ego['x'], self.cy[window]-ego['y']
        forward, left = c*dx+s*dy, -s*dx+c*dy
        visible = ((forward >= VISIBLE_FORWARD[0]) & (forward <= VISIBLE_FORWARD[1])
                   & (np.abs(left) <= VISIBLE_LATERAL))
        if not visible.any():
            return window, visible
        bearing, distance = np.arctan2(left, forward), np.hypot(forward, left)
        for box in boxes:
            # The shadow behind a detected object is unobserved, not free.
            x, y = box['center_world']
            a = math.radians(box['heading_deg'])
            hl, hw = .5*box['length_m'], .5*box['width_m']
            corners = [(x+u*hl*math.cos(a)-v*hw*math.sin(a), y+u*hl*math.sin(a)+v*hw*math.cos(a))
                       for u in (-1, 1) for v in (-1, 1)]
            f = [c*(px-ego['x'])+s*(py-ego['y']) for px, py in corners]
            l = [-s*(px-ego['x'])+c*(py-ego['y']) for px, py in corners]
            if min(f) <= 0.:
                continue
            angles = np.arctan2(l, f)
            near = min(math.hypot(a_, b_) for a_, b_ in zip(f, l))
            visible &= ~((bearing >= angles.min()) & (bearing <= angles.max()) & (distance > near))
        return window, visible

    def update(self, ego, boxes):
        hits = np.zeros(self.shape, dtype=bool)
        for box in boxes:
            self._cover(hits, box)
        delta = HIT*hits
        window, visible = self._visible(ego, boxes)
        delta[window] -= MISS*(visible & ~hits[window])
        self.log_odds = np.clip(self.log_odds + delta, *CLAMP)
        occupied = (self.log_odds >= OCCUPIED) | (self.occupied & (self.log_odds >= RELEASE))
        if not np.array_equal(occupied, self.occupied):
            self.version += 1
        self.occupied = occupied

    def _rectangles(self, xy, depth=0):
        """One rectangle per cluster; split a poorly filled one (merged neighbours) along its long axis."""
        h = .5*self.cell
        corners = np.vstack([xy+[u, v] for u in (-h, h) for v in (-h, h)])
        rect = _min_area_rect(corners)
        fill = len(xy)*self.cell*self.cell/max(rect[2]*rect[3], 1e-9)
        if fill >= MIN_FILL or depth >= MAX_SPLIT_DEPTH or len(xy) < 2*MIN_CELLS:
            return [rect]
        axis = np.array([math.cos(math.radians(rect[4])), math.sin(math.radians(rect[4]))])
        along = (xy-xy.mean(axis=0)) @ axis
        # Split at the largest gap in the cells' projection, where two objects meet.
        order = np.sort(along)
        cut = order[int(np.argmax(np.diff(order)))] if len(order) > 1 else 0.
        parts = [xy[along <= cut], xy[along > cut]]
        if min(len(p) for p in parts) < MIN_CELLS:
            return [rect]
        return [r for p in parts for r in self._rectangles(p, depth+1)]

    def boxes(self):
        labels, count = ndimage.label(self.occupied, structure=np.ones((3, 3)))
        out, clusters = [], {}
        rects = []
        for index in range(1, count+1):
            rows, cols = np.nonzero(labels == index)
            if len(rows) >= MIN_CELLS:
                rects += [(r, len(rows)) for r in self._rectangles(np.column_stack((self.cx[rows, cols],
                                                                                    self.cy[rows, cols])))]
        for (cx, cy, length, width, heading), cells in rects:
            identity = next((k for k, (x, y) in self.clusters.items()
                             if k not in clusters and math.hypot(x-cx, y-cy) < MATCH_M), None)
            if identity is None:
                identity, self.next_id = self.next_id, self.next_id+1
            clusters[identity] = (cx, cy)
            out.append(dict(center_world=[cx, cy], heading_deg=heading, planning_heading_deg=heading,
                            length_m=length, width_m=width, speed_mps=0., planning_speed_mps=0.,
                            observed_speed_mps=0., motion_state='stationary', planning_speed_source='static_map',
                            motion_direction_observed=False, track_id=ID_BASE+identity, map_cells=int(cells)))
        self.clusters = clusters
        return out

    def apply(self, packet):
        """Replace the section's detections in a tracked packet by the accumulated map boxes."""
        if packet.get('generation') != self.generation:
            self.reset()
            self.generation = packet.get('generation')
        inside = [b for b in packet['boxes'] if self.contains(*b['center_world'])]
        observed = [b for b in inside if b.get('planning_speed_source') != 'static_memory']
        ego = packet.get('ego_pose')
        if ego is not None:
            self.update(ego, observed)
        mapped = self.boxes()
        out = dict(packet, boxes=[b for b in packet['boxes'] if not self.contains(*b['center_world'])] + mapped)
        out['static_map'] = dict(raw_boxes=[[round(b['center_world'][0], 2), round(b['center_world'][1], 2),
                                              round(b['length_m'], 2), round(b['width_m'], 2),
                                              round(b['heading_deg'], 1), b.get('motion_state'), b.get('track_id')]
                                             for b in observed],
                                 occupied_cells=int(self.occupied.sum()), boxes=len(mapped))
        return out


def _min_area_rect(points):
    """Centre, length (>= width), width and heading [deg] of the minimum-area rectangle around the points."""
    try:
        hull = points[ConvexHull(points).vertices]
    except (QhullError, ValueError):
        hull = points
    best = None
    edges = np.diff(np.vstack((hull, hull[:1])), axis=0)
    for angle in np.unique(np.round(np.arctan2(edges[:, 1], edges[:, 0]) % (math.pi/2), 6)):
        c, s = math.cos(angle), math.sin(angle)
        u, v = hull @ np.array([c, s]), hull @ np.array([-s, c])
        area = (u.max()-u.min())*(v.max()-v.min())
        if best is None or area < best[0]:
            best = (area, angle, u.min(), u.max(), v.min(), v.max())
    _, angle, u0, u1, v0, v1 = best
    c, s = math.cos(angle), math.sin(angle)
    um, vm = .5*(u0+u1), .5*(v0+v1)
    cx, cy = um*c-vm*s, um*s+vm*c
    length, width, heading = u1-u0, v1-v0, angle
    if width > length:
        length, width, heading = width, length, angle+math.pi/2
    return float(cx), float(cy), float(length), float(width), math.degrees(heading)
