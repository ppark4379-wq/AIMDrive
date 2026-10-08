"""Shared nominal tunnel road and conservative Ego body contract.

This boundary uses both MGeo lanes and their connected approach/departure
links. It does not turn a recorded sweep or a LiDAR observation into a road.
All geometry is immutable; shallow planner snapshots share this contract.
"""
import hashlib
import copy
import json
import math
from pathlib import Path
from threading import RLock
from types import MappingProxyType

import numpy as np
import shapely
from shapely.geometry import LineString, Polygon

if not hasattr(shapely, 'union_all') or not hasattr(shapely, 'polygons'):
    raise ImportError('Jamming road boundary requires Shapely 2; install the pinned Expert runtime dependencies')


OWN_LINK = "A2256W000126"
OPPOSING_LINK = "A2256W000131"
BODY_POLICY = {"rear_m": .85, "front_m": 3.85, "width_m": 2.17}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def _link_rows(links):
    if isinstance(links, dict) and "links" in links:
        links = links["links"]
    if isinstance(links, dict):
        rows = [(str(key), value) for key, value in links.items()]
    else:
        rows = [(str(value.get("idx", value.get("id", ""))), value) for value in links]
    result = {}
    for identity, row in rows:
        if not isinstance(row, dict) or not identity:
            raise ValueError("Invalid road link record")
        result[identity] = dict(row, idx=identity)
    return result


def _lane_points(row):
    points = np.asarray(row["points"], dtype=float)
    if points.ndim != 2 or points.shape[1] < 2 or len(points) < 2 or not np.isfinite(points).all():
        raise ValueError("Invalid road points: " + row["idx"])
    points = points[:, :2]
    keep = np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-7]
    points = points[keep]
    if len(points) < 2:
        raise ValueError("Empty road link: " + row["idx"])
    width = min(float(row["width_start"]), float(row["width_end"]))
    if not math.isfinite(width) or width <= 0.:
        raise ValueError("Invalid road width: " + row["idx"])
    return np.column_stack((points, np.full(len(points), width)))


def _length(points):
    return float(np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1).sum())


def _clip(points, distance, from_end=False):
    if from_end:
        return _clip(points[::-1], distance)[::-1]
    stations = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1))]
    if distance >= stations[-1] - 1e-8:
        return points.copy()
    index = int(np.searchsorted(stations, distance, side="right"))
    ratio = (distance-stations[index-1])/(stations[index]-stations[index-1])
    end = points[index-1] + ratio*(points[index]-points[index-1])
    return np.vstack((points[:index], end))


def _extend(rows, identity, before, after):
    """Follow connected links; never extrapolate a missing approach or exit."""
    points = _lane_points(rows[identity])
    used = {identity}
    for backwards, required in ((True, before), (False, after)):
        current = rows[identity]
        remaining = float(required)
        pieces = []
        while remaining > 1e-7:
            current_points = _lane_points(current)
            node = current.get("from_node_idx" if backwards else "to_node_idx")
            if node is None:
                raise ValueError("Road extension lacks connected nodes: " + current["idx"])
            tangent = (current_points[1, :2]-current_points[0, :2] if backwards else
                       current_points[-1, :2]-current_points[-2, :2])
            tangent /= np.linalg.norm(tangent)
            options = []
            for name, row in rows.items():
                if name in used or row.get("to_node_idx" if backwards else "from_node_idx") != node:
                    continue
                candidate = _lane_points(row)
                direction = (candidate[-1, :2]-candidate[-2, :2] if backwards else
                             candidate[1, :2]-candidate[0, :2])
                direction /= np.linalg.norm(direction)
                endpoint = candidate[-1, :2] if backwards else candidate[0, :2]
                anchor = current_points[0, :2] if backwards else current_points[-1, :2]
                alignment = float(np.dot(tangent, direction))
                if alignment > .5 and np.linalg.norm(endpoint-anchor) < .1:
                    options.append((-alignment, name, row, candidate))
            if not options:
                raise ValueError("Insufficient connected road extension: " + current["idx"])
            _, name, current, candidate = min(options, key=lambda item: (item[0], item[1]))
            used.add(name)
            part = _clip(candidate, min(remaining, _length(candidate)), from_end=backwards)
            pieces.append(part)
            remaining -= _length(part)
        if backwards:
            for part in pieces:
                points = np.vstack((part[:-1], points))
        else:
            for part in pieces:
                points = np.vstack((points, part[1:]))
    return points, used


class RoadBoundary:
    @classmethod
    def from_map_dir(cls, map_dir, **kwargs):
        path = Path(map_dir)/"link_set.json"
        return cls.from_links(json.loads(path.read_text(encoding="utf-8-sig")), **kwargs)

    @classmethod
    def from_links(cls, links, *, own_link=OWN_LINK, opposing_link=OPPOSING_LINK,
                   approach_m=120., departure_m=70., body_policy=None,
                   planning_margin_m=.35, runtime_margin_m=.15):
        rows = _link_rows(links)
        for identity in (own_link, opposing_link):
            if identity not in rows or str(rows[identity].get("road_type")) != "2":
                raise ValueError("Mapped tunnel link missing: " + identity)
        for distance in (approach_m, departure_m):
            if not math.isfinite(float(distance)) or float(distance) <= 0.:
                raise ValueError("Invalid road extension distance")
        if (not math.isfinite(float(runtime_margin_m)) or abs(float(runtime_margin_m)-.15)>1e-9
                or not math.isfinite(float(planning_margin_m)) or planning_margin_m<runtime_margin_m):
            raise ValueError("Runtime road margin is fixed at .15m; planning margin cannot be smaller")
        own = _lane_points(rows[own_link])
        opposite = _lane_points(rows[opposing_link])
        if any(abs(float(width)-3.5) > 1e-6 for width in np.r_[own[:, 2], opposite[:, 2]]):
            raise ValueError("Tunnel road contract requires two nominal 3.5m lanes")
        own_extended, own_used = _extend(rows, own_link, approach_m, departure_m)
        opposite_extended, opposite_used = _extend(rows, opposing_link, departure_m, approach_m)
        geometry = [{"idx": name, "from_node_idx": rows[name].get("from_node_idx"),
                     "to_node_idx": rows[name].get("to_node_idx"),
                     "points": _lane_points(rows[name]).tolist()}
                    for name in sorted(own_used | opposite_used)]
        policy = dict(BODY_POLICY)
        for key, value in (body_policy or {}).items():
            if key not in policy or not math.isfinite(float(value)) or float(value) <= 0.:
                raise ValueError("Invalid Ego body policy: " + str(key))
            policy[key] = max(policy[key], float(value))
        return cls(own, own_extended, opposite_extended[::-1], geometry, policy,
                   own_link, opposing_link, float(approach_m), float(departure_m), opposite[::-1],
                   float(planning_margin_m), float(runtime_margin_m))

    def __init__(self, own, own_extended, opposite_extended, geometry, body,
                 own_link, opposing_link, approach, departure, opposite_tunnel,
                 planning_margin, runtime_margin):
        # Shapely releases the GIL and GEOS prepared predicates maintain lazy
        # native indexes even when the Python geometry is immutable. The
        # runtime and background planner snapshots share one native polygon,
        # so they must also share its predicate lock.
        # https://github.com/libgeos/geos/issues/825
        self._predicate_lock = RLock()
        self.origin = own[0, :2].copy()
        self.forward = own[-1, :2]-own[0, :2]
        self.forward /= np.linalg.norm(self.forward)
        self.right = np.array([self.forward[1], -self.forward[0]])
        self._tunnel_length_m = float(np.linalg.norm(own[-1, :2]-own[0, :2]))
        self._planning_margin_m, self._runtime_margin_m = planning_margin, runtime_margin
        self._body_policy = MappingProxyType(dict(body))
        self.source_sha256 = _digest(geometry)
        self._policy = dict(schema="aimdrive_tunnel_road_boundary_v1", ego_link=own_link,
                           opposing_link=opposing_link, basis="nominal_mgeo_connected_lane_outer_edges",
                           physical_walls_verified=False, nominal_tunnel_width_m=7.,
                           approach_m=approach, departure_m=departure, body=dict(body),
                           planner_model={"rear_m": 0., "front_m": 4.64,
                                          "width_m": max(2.17, body['width_m'])},
                           static_collision_margin_m=.5, static_tracking_margin_m=.1,
                           moving_collision_margin_m=.25,
                           planning_road_margin_m=planning_margin, runtime_road_margin_m=runtime_margin,
                           planning_margin_domain="mapped_tunnel_pose_stations",
                           baseline_approach_return_road_margin_m=runtime_margin,
                           tracking_reserve_m=.15, source_sha256=self.source_sha256,
                           connected_link_ids=[row["idx"] for row in geometry])
        self.policy_sha256 = _digest(self._policy)
        # The two approaches diverge at a mapped junction. A min/max span
        # there would incorrectly fill the median and the intervening ground.
        # Union the actual connected lane strips; bridge only the two mapped
        # tunnel lanes where the road is explicitly one continuous corridor.
        strips = [LineString(lane[:, :2]).buffer(float(lane[:, 2].min())*.5,
                    cap_style=2, join_style=2, mitre_limit=2.)
                  for lane in (own_extended, opposite_extended)]
        def outer(points, sign):
            tangent = np.gradient(points[:, :2], axis=0)
            tangent /= np.linalg.norm(tangent, axis=1)[:, None]
            left = np.column_stack((-tangent[:, 1], tangent[:, 0]))
            return points[:, :2] + sign*points[:, 2, None]*.5*left
        core = opposite_tunnel
        tunnel = Polygon(np.vstack((outer(own, -1.), outer(core, 1.)[::-1])))
        if not tunnel.is_valid or tunnel.is_empty:
            raise ValueError("Invalid paired tunnel road polygon")
        self._polygon = shapely.union_all(strips+[tunnel])
        if not self._polygon.is_valid or self._polygon.is_empty:
            raise ValueError("Invalid connected road polygon")
        with self._predicate_lock:
            shapely.prepare(self._polygon)
        coverage = np.vstack((self.to_local(own_extended[:, :2]),
                              self.to_local(opposite_extended[:, :2])))
        self.low_y, self.high_y = float(coverage[:, 1].min()), float(coverage[:, 1].max())
        for value in (self.origin, self.forward, self.right):
            value.flags.writeable = False

    def __copy__(self):
        # Keep a separate wrapper (tests can replace a private polygon) while
        # immutable geometry, read-only arrays and their native lock stay
        # shared. A new lock on a shallow copy would reintroduce the race.
        result = object.__new__(type(self))
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo):
        # Planner snapshots share this immutable contract rather than trying
        # to pickle a MappingProxyType/RLock or duplicate prepared GEOS state.
        result = self.__copy__()
        memo[id(self)] = result
        return result

    def to_local(self, xy):
        delta = np.asarray(xy, dtype=float)-self.origin
        return np.column_stack((delta @ self.right, delta @ self.forward))

    @property
    def policy(self):
        return copy.deepcopy(self._policy)

    @property
    def body_policy(self):
        return dict(self._body_policy)

    @property
    def planning_margin_m(self):
        return self._planning_margin_m

    @property
    def runtime_margin_m(self):
        return self._runtime_margin_m

    def corners_world(self, xy, yaw, margin=.15):
        xy, yaw = np.asarray(xy, dtype=float), np.asarray(yaw, dtype=float)
        if xy.ndim != 2 or xy.shape[1] != 2 or yaw.shape != (len(xy),):
            raise ValueError("Body poses must be Nx2 points and N headings")
        if not math.isfinite(float(margin)) or margin < 0.:
            raise ValueError("Invalid body margin")
        along = np.array([-self._body_policy["rear_m"]-margin,
                          self._body_policy["front_m"]+margin])
        side = self._body_policy["width_m"]*.5+margin
        offsets = np.array([(along[0], -side), (along[1], -side),
                            (along[1], side), (along[0], side)])
        forward = np.column_stack((np.cos(yaw), np.sin(yaw)))
        left = np.column_stack((-np.sin(yaw), np.cos(yaw)))
        return xy[:, None, :] + offsets[None, :, 0, None]*forward[:, None, :] + offsets[None, :, 1, None]*left[:, None, :]

    def covers_world(self, xy, yaw, margin=.15):
        corners = self.corners_world(xy, yaw, margin)
        local = self.to_local(corners.reshape(-1, 2)).reshape(-1, 4, 2)
        return np.isfinite(local).all(axis=(1, 2)) & (local[:, :, 1].min(axis=1)>=self.low_y) & (local[:, :, 1].max(axis=1)<=self.high_y)

    def mask_world(self, xy, yaw, margin=.15):
        """Prepared, vectorized polygon containment of the entire Ego OBB.

        Testing rectangles, instead of only their corners, catches roadside
        intrusions on curves and prevents skipping a portal-straddling body.
        The polygon also preserves holes between diverging approach lanes.
        """
        corners = self.corners_world(xy, yaw, margin)
        valid = np.isfinite(corners).all(axis=(1, 2))
        indices = np.flatnonzero(valid)
        if len(indices):
            bodies = shapely.polygons(corners[indices])
            with self._predicate_lock:
                valid[indices] = shapely.covers(self._polygon, bodies)
        return valid

    def covers_geometry(self, geometry):
        """Test one observed polygon using the same shared native-cache lock."""
        with self._predicate_lock:
            return bool(shapely.covers(self._polygon, geometry))

    def outside_area_geometry(self, geometry):
        """Measure observed area outside the road without unguarded GEOS reads."""
        with self._predicate_lock:
            return float(shapely.difference(geometry, self._polygon).area)

    def planning_mask_world(self, xy, yaw, margin=None):
        """Reserve tracking room for tunnel maneuvers, preserving baseline.

        Every body, including portal-straddling bodies, still passes the
        whole-road runtime gate. Extra planning room applies to poses between
        the mapped portals; fixed approach and return keep the runtime margin.
        """
        xy, yaw = np.asarray(xy, dtype=float), np.asarray(yaw, dtype=float)
        reserve = self._planning_margin_m if margin is None else float(margin)
        if not math.isfinite(reserve) or reserve<self._runtime_margin_m:
            raise ValueError("Planning margin cannot weaken the runtime road gate")
        valid = self.mask_world(xy, yaw, self._runtime_margin_m)
        local = self.to_local(xy)
        interior = (local[:, 1]>=0.) & (local[:, 1]<=self._tunnel_length_m)
        if np.any(interior):
            valid[interior] &= self.mask_world(xy[interior], yaw[interior], reserve)
        return valid

    def collision_mask_world(self, xy, yaw, objects, margin=.72):
        """SAT for the whole rear/front union against normalized object OBBs."""
        xy, yaw = np.asarray(xy, dtype=float), np.asarray(yaw, dtype=float)
        valid = np.isfinite(xy).all(axis=1) & np.isfinite(yaw)
        forward = np.column_stack((np.cos(yaw), np.sin(yaw)))
        left = np.column_stack((-np.sin(yaw), np.cos(yaw)))
        rear, front = self._body_policy["rear_m"], self._body_policy["front_m"]
        center = xy + .5*(front-rear)*forward
        length, width = .5*(front+rear)+margin, .5*self._body_policy["width_m"]+margin
        for obj in objects:
            position = np.asarray(obj.position, dtype=float)
            values = (position[0], position[1], obj.heading_deg, obj.size_x, obj.size_y)
            if not all(math.isfinite(float(value)) for value in values) or obj.size_x <= 0. or obj.size_y <= 0.:
                return np.zeros(len(xy), dtype=bool)
            heading = math.radians(obj.heading_deg)
            of, ol = np.array([math.cos(heading), math.sin(heading)]), np.array([-math.sin(heading), math.cos(heading)])
            delta = position-center
            overlap = np.ones(len(xy), dtype=bool)
            for axis in (forward, left, np.broadcast_to(of, forward.shape), np.broadcast_to(ol, forward.shape)):
                radius = (length*np.abs(np.sum(forward*axis, axis=1)) + width*np.abs(np.sum(left*axis, axis=1)) +
                          .5*obj.size_x*np.abs(axis @ of) + .5*obj.size_y*np.abs(axis @ ol))
                overlap &= np.abs(np.sum(delta*axis, axis=1)) <= radius
            valid &= ~overlap
        return valid
