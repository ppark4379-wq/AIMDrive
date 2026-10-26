"""Accumulate VE detections in the static-obstacle section into a persistent occupancy map.

Planning on instantaneous VE boxes fails in this section: boxes jitter by 10-30 cm, grow near range, vanish for a
frame and change track ID, so a path planned on one snapshot is invalidated by the next. Here the world is static and
GPS is accurate, so evidence is accumulated per map cell instead: a cell covered by a box gains log-odds, a visible
uncovered cell loses some, and only cells with repeated evidence are occupied. The planner receives one box per
occupied cluster, the same geometry for planning and for re-checking a committed path. No heading is trusted.

Misses are counted only inside the VE detection window and outside the shadow of the current boxes, so occluded
objects are not erased. The region is the static section of ``sections.py``; boxes elsewhere pass through unchanged.
The same map in the Jamming section's rotated frame accumulates the tunnel's standing obstacles (tunnel_map.py).
"""
import math

import numpy as np
from scipy import ndimage
from scipy.spatial import ConvexHull, QhullError, cKDTree

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
EXACT_RANGE = 2.0         # below this the bound is replaced by the exact distance (speed saturates beyond) [m]
ID_BASE = 1_000_000       # positive IDs, disjoint from tracker serials (VE-unknown contract needs id > 0)
GRID_MARGIN = 2.          # the grid reaches this far past the evidence region (box extents spill over it) [m]
FIELD_PAD = 3.            # the distance transform covers the occupied cells' bounding box plus this; farther points
                          # get the distance to that box, a valid lower bound above EXACT_RANGE [m]


class StaticObstacleMap:
    def __init__(self, x_range=STATIC_OBSTACLE_CERTAIN_X, y_range=STATIC_OBSTACLE_CERTAIN_Y, cell=CELL_M, frame=None):
        """x_range/y_range bound the evidence region in the map frame. ``frame=(x, y, rotation)`` puts the map in a
        rotated frame (map = R(rotation)(world - (x, y))), as the Jamming section's forward-Y frame; distances and
        boxes are then in that frame. None keeps world coordinates."""
        self.frame = None if frame is None else (float(frame[0]), float(frame[1]), float(frame[2]))
        self.region = (float(x_range[0]), float(x_range[1]), float(y_range[0]), float(y_range[1]))  # evidence
        self.x0, self.x1 = self.region[0]-GRID_MARGIN, self.region[1]+GRID_MARGIN
        self.y0, self.y1, self.cell = self.region[2]-GRID_MARGIN, self.region[3]+GRID_MARGIN, float(cell)
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
        self._boxes = None          # (version, boxes)

    def to_map(self, x, y, yaw=0.):
        """World point(s) and heading [rad] in the map frame."""
        if self.frame is None:
            return x, y, yaw
        ox, oy, rotation = self.frame
        c, s = math.cos(rotation), math.sin(rotation)
        dx, dy = np.asarray(x, dtype=float)-ox, np.asarray(y, dtype=float)-oy
        return c*dx-s*dy, s*dx+c*dy, yaw+rotation

    def _in_frame(self, ego, boxes):
        if self.frame is None:
            return ego, boxes
        x, y, yaw = self.to_map(ego['x'], ego['y'], ego['yaw_grid_rad'])
        ego = dict(ego, x=float(x), y=float(y), yaw_grid_rad=float(yaw))
        moved = []
        for box in boxes:
            bx, by, heading = self.to_map(*box['center_world'], math.radians(box['heading_deg']))
            moved.append(dict(box, center_world=[float(bx), float(by)], heading_deg=math.degrees(heading)))
        return ego, moved

    def distance(self, x, y, exact_below=EXACT_RANGE):
        """Distance [m] from world points to the nearest occupied cell square (inf with no occupied cell).

        Never longer than the true distance. A cheap bound from the distance transform is used everywhere;
        within EXACT_RANGE it is replaced by the exact distance to the occupied squares (k-d tree over cell
        centres). A square can be nearest only if its centre lies within half a cell diagonal of the nearest
        centre distance, and the exact value is taken only when every such square was compared; otherwise the
        bound stays. A fixed cell-diagonal deduction everywhere would add about 0.14 m of hidden margin.
        exact_below=0 returns only the bound (at most 1.5 cell diagonals short), for screening many points.
        """
        x, y = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        if self._field is None or self._field[0] != self.version:
            self._field = (self.version, None)
            if self.occupied.any():
                # The transform only covers the occupied cells' bounding box plus FIELD_PAD; every occupied cell
                # lies inside it, so inside it the transform is the same as over the whole grid.
                rows, cols = np.nonzero(self.occupied)
                pad = int(math.ceil(FIELD_PAD/self.cell))
                r0, r1 = max(0, rows.min()-pad), min(self.shape[0], rows.max()+1+pad)
                c0, c1 = max(0, cols.min()-pad), min(self.shape[1], cols.max()+1+pad)
                field = ndimage.distance_transform_edt(~self.occupied[r0:r1, c0:c1])*self.cell
                tree = cKDTree(np.column_stack((self.cx[rows, cols], self.cy[rows, cols])))
                box = (self.x0+cols.min()*self.cell, self.x0+(cols.max()+1)*self.cell,
                       self.y0+rows.min()*self.cell, self.y0+(rows.max()+1)*self.cell)
                self._field = (self.version, (field, (r0, r1, c0, c1), box, tree))
        if self._field[1] is None:
            return np.full(x.shape, np.inf)
        field, (r0, r1, c0, c1), (bx0, bx1, by0, by1), tree = self._field[1]
        half = .5*math.sqrt(2.)*self.cell        # centre-to-corner of a cell
        px, py = x.reshape(-1), y.reshape(-1)
        # Outside the transform window: distance to the occupied cells' bounding box (a lower bound).
        flat = np.hypot(np.maximum(np.maximum(bx0-px, 0.), px-bx1), np.maximum(np.maximum(by0-py, 0.), py-by1))
        row = np.floor((py-self.y0)/self.cell).astype(int)
        col = np.floor((px-self.x0)/self.cell).astype(int)
        inside = (row >= r0) & (row < r1) & (col >= c0) & (col < c1)
        row, col = row[inside], col[inside]
        # Nearest centre from p >= nearest centre from p's cell centre - |p - that centre|; a square is at
        # least its centre distance minus half a diagonal.
        offset = np.hypot(px[inside]-self.cx[row, col], py[inside]-self.cy[row, col])
        flat[inside] = np.maximum(0., field[row-r0, col-c0]-offset-half)
        out = flat.reshape(x.shape)
        # The box bound is loose only where the window was clipped by the grid edge; measure those exactly.
        todo = np.flatnonzero((flat < exact_below) | (~inside & (flat < EXACT_RANGE)))
        for k in (8, 64):
            if not len(todo):
                break
            k = min(k, tree.n)
            d, index = tree.query(np.column_stack((px[todo], py[todo])), k=k)
            d, index = d.reshape(len(todo), k), index.reshape(len(todo), k)
            qx, qy = px[todo][:, None], py[todo][:, None]
            left, low = tree.data[index, 0]-.5*self.cell, tree.data[index, 1]-.5*self.cell
            dx = np.maximum(np.maximum(left-qx, 0.), qx-(left+self.cell))
            dy = np.maximum(np.maximum(low-qy, 0.), qy-(low+self.cell))
            candidate = np.hypot(dx, dy).min(axis=1)
            # Every square not compared is at least (k-th centre distance - half) away.
            complete = (k == tree.n) | (d[:, -1]-half >= candidate)
            flat[todo[complete]] = candidate[complete]
            todo = todo[~complete]           # left with the safe bound if still incomplete
        return out

    def contains(self, x, y):
        """Detections centred in the evidence region are mapped; the grid itself is larger."""
        x0, x1, y0, y1 = self.region
        return x0 <= x <= x1 and y0 <= y <= y1

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
        corners = [(ego['x']+f*c-l*s, ego['y']+f*s+l*c) for f in VISIBLE_FORWARD for l in (-VISIBLE_LATERAL, VISIBLE_LATERAL)]
        xs, ys = [p[0] for p in corners], [p[1] for p in corners]
        window = self._window(min(xs), max(xs), min(ys), max(ys))
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
        """One frame of evidence (map frame). Only the window covering the visible area and the boxes changes."""
        window, visible = self._visible(ego, boxes)
        rows, cols = [window[0].start, window[0].stop], [window[1].start, window[1].stop]
        for box in boxes:
            reach = .5*math.hypot(box['length_m'], box['width_m'])
            x, y = box['center_world']
            r, c = self._window(x-reach, x+reach, y-reach, y+reach)
            if r.stop > r.start and c.stop > c.start:
                rows += [r.start, r.stop]; cols += [c.start, c.stop]
        area = slice(min(rows), max(rows)), slice(min(cols), max(cols))
        if area[0].stop <= area[0].start or area[1].stop <= area[1].start:
            return
        hits = np.zeros(self.shape, dtype=bool)
        for box in boxes:
            self._cover(hits, box)
        delta = HIT*hits[area]
        seen = (slice(window[0].start-area[0].start, window[0].stop-area[0].start),
                slice(window[1].start-area[1].start, window[1].stop-area[1].start))
        delta[seen] -= MISS*(visible & ~hits[window])
        log_odds = np.clip(self.log_odds[area]+delta, *CLAMP)
        occupied = (log_odds >= OCCUPIED) | (self.occupied[area] & (log_odds >= RELEASE))
        self.log_odds[area] = log_odds
        if not np.array_equal(occupied, self.occupied[area]):
            self.version += 1
            self.occupied[area] = occupied

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
        """One stationary box per occupied cluster (map frame); recomputed only when the occupied cells change."""
        if self._boxes is None or self._boxes[0] != self.version:
            self._boxes = (self.version, self._clusters())
        return [dict(b) for b in self._boxes[1]]

    def observe(self, packet, boxes):
        """Accumulate `boxes` (world frame) of a packet without rewriting it; a new generation resets the map."""
        if packet.get('generation') != self.generation:
            self.reset()
            self.generation = packet.get('generation')
        ego = packet.get('ego_pose')
        if ego is not None:
            self.update(*self._in_frame(ego, boxes))

    def _clusters(self):
        out, clusters = [], {}
        rects = []
        rows, cols = np.nonzero(self.occupied)
        if len(rows):
            r0, c0 = rows.min(), cols.min()
            labels, _ = ndimage.label(self.occupied[r0:rows.max()+1, c0:cols.max()+1], structure=np.ones((3, 3)))
            label = labels[rows-r0, cols-c0]
            order = np.argsort(label, kind='stable')
            for group in np.split(order, np.flatnonzero(np.diff(label[order]))+1):
                r, c = rows[group], cols[group]
                if len(r) >= MIN_CELLS:
                    rects += [(rect, len(r)) for rect in self._rectangles(np.column_stack((self.cx[r, c],
                                                                                         self.cy[r, c])))]
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
        """Replace the section's detections in a tracked packet by the accumulated map boxes (world frame only)."""
        if self.frame is not None:
            raise ValueError('apply() rewrites world-frame packets; use observe() for a rotated map')
        if packet.get('generation') != self.generation:
            self.reset()
            self.generation = packet.get('generation')
        inside = [b for b in packet['boxes'] if self.contains(*b['center_world'])]
        observed = [b for b in inside if b.get('planning_speed_source') != 'static_memory']
        ego = packet.get('ego_pose')
        if ego is not None:
            self.update(*self._in_frame(ego, observed))
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
