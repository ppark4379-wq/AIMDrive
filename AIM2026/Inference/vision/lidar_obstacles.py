"""Class-agnostic observed extents, short tracks and conservative VE association.

Coordinates are base (right, forward, height above road). No vehicle templates,
semantic labels, hidden-body completion or prediction-only detections.
"""
import math
import numpy as np

from .postprocess import _rectangle, _intersection, _partial, to_world, lshape_axis


class LidarObstacleTracker:
    VOXEL = .2
    GAP = .55
    LONGITUDINAL_GAP = 2.0
    MIN_POINTS = 8
    MAX_GAP = .75
    CONFIRM_HITS = 3
    RAIL_MAX_WIDTH = .65
    RAIL_MAX_TOP = 1.3

    def __init__(self):
        self.tracks = []
        self.next_id = 1
        self.last_time = None
        self.observations = []

    def components(self, points):
        from scipy.spatial import cKDTree
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
        self.cluster_counts = dict(lidar_rail_rejected=0, lidar_fragment_merges=0)
        if points is None:
            return []
        p = np.asarray(points, dtype=float)
        if p.ndim != 2 or p.shape[1] != 3:
            return []
        # Flat-road base-frame ground band and a conservative ego footprint.
        # This is not a slope-aware ground estimator.
        keep = (np.isfinite(p).all(1) & (p[:, 2] > .3) & (p[:, 2] < 4.5)
                & (abs(p[:, 0]) < 25.) & (p[:, 1] > -20.) & (p[:, 1] < 30.))
        keep &= ~((abs(p[:, 0]) < 1.1) & (p[:, 1] > -1.3) & (p[:, 1] < 3.7))
        p = p[keep]
        if len(p) < self.MIN_POINTS:
            return []
        voxels, inverse = np.unique(np.floor(p[:, :2]/self.VOXEL).astype(int), axis=0,
                                    return_inverse=True)
        # First use the short radius. Remove low thin structures before they
        # can bridge a vehicle into a long roadside component.
        pairs = cKDTree(voxels*self.VOXEL).query_pairs(self.GAP, output_type='ndarray')
        graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])),
                           shape=(len(voxels), len(voxels)))
        _, labels = connected_components(graph, directed=False)
        labels = labels[inverse]
        order = np.argsort(labels)
        groups = np.split(order, np.flatnonzero(np.diff(labels[order]))+1)
        clouds = [p[g] for g in groups if len(g) >= self.MIN_POINTS]
        shapes = [self.measure(c) for c in clouds]
        thin = [i for i, (c, b) in enumerate(zip(clouds, shapes))
                if b is not None and b['width'] <= self.RAIL_MAX_WIDTH
                and np.quantile(c[:, 2], .95) <= self.RAIL_MAX_TOP
                and np.ptp(np.quantile(c[:, 2], [.05, .95])) <= .9]
        rejected = {i for i in thin if shapes[i]['length'] >= 2.5}
        for i in thin:
            a = shapes[i]; u = a['heading']; v = np.array([-u[1], u[0]])
            for j in thin:
                if j <= i:
                    continue
                b = shapes[j]; delta = b['center']-a['center']
                # Repeated roadside fragments along the same low thin line.
                if (abs(u@b['heading']) >= .98 and abs(delta@v) <= .5
                        and 4. <= abs(delta@u) <= 15.
                        and abs(delta@u)+(a['length']+b['length'])/2 >= 6.):
                    rejected.update((i, j))
        self.cluster_counts['lidar_rail_rejected'] = len(rejected)
        keep = [i for i, b in enumerate(shapes) if i not in rejected and b is not None]
        clouds, shapes = [clouds[i] for i in keep], [shapes[i] for i in keep]
        # Longer connections are allowed only between fragments. Two already
        # substantial footprints must not be joined by the 2 m chassis radius.
        scale = np.array([self.GAP, self.LONGITUDINAL_GAP])
        trees = [cKDTree(np.unique(c[:, :2], axis=0)/scale) for c in clouds]
        parents = list(range(len(clouds)))
        substantial = [int(b['length'] >= 3. and b['width'] >= 1.2) for b in shapes]

        def root(i):
            while parents[i] != i:
                i = parents[i]
            return i

        edges = []
        for i, a in enumerate(shapes):
            for j in range(i+1, len(shapes)):
                b = shapes[j]
                if (a['bounds'][1]+self.GAP < b['bounds'][0]
                        or b['bounds'][1]+self.GAP < a['bounds'][0]
                        or a['bounds'][3]+self.LONGITUDINAL_GAP < b['bounds'][2]
                        or b['bounds'][3]+self.LONGITUDINAL_GAP < a['bounds'][2]):
                    continue
                distance = float(trees[i].query(trees[j].data, k=1)[0].min())
                if distance <= 1.:
                    edges.append((distance, i, j))
        for _, i, j in sorted(edges):
            a, b = root(i), root(j)
            if a == b or substantial[a]+substantial[b] > 1:
                continue
            parents[b] = a
            substantial[a] += substantial[b]
            self.cluster_counts['lidar_fragment_merges'] += 1
        joined = {}
        for i, cloud in enumerate(clouds):
            joined.setdefault(root(i), []).append(cloud)
        return [np.concatenate(group) for group in joined.values()]

    @staticmethod
    def measure(cloud):
        xy = np.unique(cloud[:, :2], axis=0)
        if len(xy) < 3:
            return None
        # Fit visible rectangle edges rather than the covariance direction:
        # unequal returns on the two faces bias PCA toward the diagonal.
        # Use one representative per 10 cm cell to avoid scan density weighting
        # and bound angular-search work; final extents contain all input points.
        _, representatives = np.unique(np.floor(xy/.1).astype(np.int64), axis=0,
                                        return_index=True)
        fit = xy[representatives]
        axis, _, _ = lshape_axis(fit-fit.mean(0))
        u = np.array([math.cos(axis), math.sin(axis)])
        if u[1] < 0:
            u = -u
        v = np.array([-u[1], u[0]])
        along, across = xy@u, xy@v
        lo, hi, left, right = along.min(), along.max(), across.min(), across.max()
        # One voxel of thickness prevents a visible side becoming a zero-area box.
        length, width = max(.2, hi-lo), max(.2, right-left)
        if length > 29. or width > 29.:
            return None
        return _rectangle(u*(lo+hi)/2+v*(left+right)/2, u, length, width)

    def accepts(self, cloud, shape):
        return shape is not None

    def observe(self, points, pose, now):
        """Update tracks and observations without constructing output boxes."""
        if not np.isfinite([*pose, now]).all():
            raise ValueError('LiDAR observations require finite pose and time')
        self.observations = []
        if self.last_time is not None and now <= self.last_time:
            self.tracks.clear()
        self.last_time = now
        old = [t for t in self.tracks if 0 < now-t['time'] <= self.MAX_GAP]
        measurements = []
        for cloud in self.components(points):
            box = self.measure(cloud)
            if self.accepts(cloud, box):
                world, angle = to_world(box['center'], box['heading'], *pose)
                measurements.append(dict(box=box, cloud=cloud, world=np.asarray(world), angle=angle))
        # Prefer nearby observations; require mutual uniqueness rather than a
        # greedy assignment through a split/merge or two nearby vehicles.
        links = []
        for m in measurements:
            candidates = []
            for i, t in enumerate(old):
                dt = now-t['time']
                expected = t['center'] + (t['velocity']*dt if t['velocity'] is not None else 0.)
                gate = .6 + (6.*dt*dt if t['velocity'] is not None else 40.*dt)
                distance = float(np.linalg.norm(m['world']-expected))
                if distance <= gate:
                    candidates.append((distance, i))
            candidates.sort()
            # A distant alternative inside the speed gate is not an ambiguity;
            # close alternatives (within 0.6 m of the best match) are.
            links.append([i for distance, i in candidates
                          if distance <= candidates[0][0]+.6])
        counts = dict(lidar_candidates=len(measurements), lidar_confirmed=0,
                      lidar_backup=0, lidar_ambiguous=0, lidar_ve_duplicates=0)
        counts.update(self.cluster_counts)
        active, used_old = [], set()
        for m, links_here in zip(measurements, links):
            matched = (len(links_here) == 1 and sum(links_here[0] in row for row in links) == 1)
            t = old[links_here[0]] if matched else None
            ambiguous = bool(links_here) and not matched
            counts['lidar_ambiguous'] += int(ambiguous)
            speed, velocity = None, None
            if t is not None:
                used_old.add(links_here[0])
                dt = now-t['time']
                dims = np.array([m['box']['length'], m['box']['width']])
                stable = np.max(abs(dims-t['dims'])) < .35
                velocity = (m['world']-t['center'])/dt
                if not stable or np.linalg.norm(velocity) > 45. or dt < .03:
                    velocity = None
                hits = t['hits']+1
                if np.any(abs(dims-t['dims']) > np.maximum(.8, .25*t['dims'])):
                    # A substantial split/merge/visibility change starts a new
                    # confirmation window, even if its centre barely moved.
                    hits = 1
                    velocity = None
                if velocity is not None and t['velocity'] is not None and hits >= 3:
                    if np.linalg.norm(velocity-t['velocity']) <= 1.+10.*dt:
                        speed = float(np.linalg.norm(velocity))
                identity = t['id']
            else:
                identity = self.next_id
                self.next_id += 1
                hits = 1
            record = dict(id=identity, center=m['world'], time=now, velocity=velocity,
                          dims=np.array([m['box']['length'], m['box']['width']]), hits=hits)
            # Ambiguous frames are visible candidates but cannot sustain a track.
            if not ambiguous:
                active.append(record)
            m.update(id=identity, hits=hits, speed=speed, ambiguous=ambiguous)
            self.observations.append(dict(center_world=m['world'].tolist(), heading_deg=m['angle'],
                width_m=m['box']['width'], length_m=m['box']['length'], speed_mps=speed,
                track_id=identity, observation_time=now, lidar_points=len(m['cloud']),
                consecutive_hits=hits, confirmed=hits >= self.CONFIRM_HITS and not ambiguous,
                extent_kind='observed', source='lidar'))
        self.tracks = active + [dict(t, hits=0, velocity=None)
                                for i, t in enumerate(old) if i not in used_old]
        return measurements, counts

    def apply(self, boxes, points, pose, now):
        measurements, counts = self.observe(points, pose, now)
        # Match VE boxes by actual point support, never merely centroid distance.
        claims = []
        for m in measurements:
            ids = []
            for i, b in enumerate(boxes):
                u = b['heading']; v = np.array([-u[1], u[0]])
                delta = m['cloud'][:, :2]-b['center']
                support = ((abs(delta@u) <= b['length']/2+.2)
                           & (abs(delta@v) <= b['width']/2+.2)).sum()
                if support >= self.MIN_POINTS:
                    ids.append(i)
            claims.append(ids)
        suppressed, backups = set(), []
        for m, ids, observation in zip(measurements, claims, self.observations):
            confirmed = m['hits'] >= self.CONFIRM_HITS and not m['ambiguous']
            counts['lidar_confirmed'] += int(confirmed)
            unique = bool(ids) and all(sum(i in group for group in claims) == 1 for i in ids)
            # Separate VE objects sharing a connected cloud must not validate
            # that entire cloud (e.g. two cars joined by guardrail returns).
            duplicate_group = all(
                _intersection(boxes[a], boxes[b])/min(boxes[a]['area'], boxes[b]['area']) > .3
                for a in ids for b in ids if a < b)
            association_ok = unique and duplicate_group and not m['ambiguous']
            observation.update(ve_matched=bool(association_ok),
                               backup_eligible=confirmed and not ids
                               and not any(_intersection(m['box'], b) > 0 for b in boxes))
            if ids:
                # Suppress only overlapping duplicate detections, not distinct
                # vehicles that happen to be bridged by LiDAR points.
                if unique and len(ids) > 1:
                    kept = []
                    for i in sorted(ids, key=lambda i: -boxes[i]['score']):
                        if any(_intersection(boxes[i], boxes[k])/min(boxes[i]['area'], boxes[k]['area']) > .3
                               for k in kept):
                            suppressed.add(i)
                        else:
                            kept.append(i)
                continue
            # No VE support: publish only persistent, current observed extents.
            # Overlap without point support is uncertain, so do not double-count.
            if (not confirmed
                    or any(_intersection(m['box'], b) > 0 for b in boxes)):
                continue
            b = dict(m['box'], score=0., cell_index=-1, partial=_partial(m['box']),
                     tracked_speed=m['speed'], tracked_lidar_center=tuple(m['world']),
                     tracked_lidar_count=len(m['cloud']), track_id=m['id'],
                     observed_only=True, observation_time=now, lidar_hits=m['hits'])
            backups.append(b)
        counts['lidar_ve_duplicates'] = len(suppressed)
        # Stay within the existing UDP frame limit, prioritising VE detections.
        result = [b for i, b in enumerate(boxes) if i not in suppressed]
        backups = backups[:max(0, 256-len(result))]
        counts['lidar_backup'] = len(backups)
        return result+backups, counts
