"""Explicit true-ENU orientation -> projected-map conversion.

Map positions remain unchanged. Never add an empirically fitted yaw constant,
rotate already-grid velocities, or infer a model's bbox origin convention.
"""
import itertools
import numpy as np
from pyproj import CRS, Proj, Transformer
from scipy.spatial.transform import Rotation


class MapGeometry:
    def __init__(self, crs, origin_xyz):
        self.crs = CRS.from_user_input(crs)
        self.origin = np.asarray(origin_xyz, dtype=float)
        if not self.crs.is_projected or self.origin.shape != (3,) or not np.isfinite(self.origin).all():
            raise ValueError('Explicit projected CRS and finite map origin required')
        if any(a.unit_name != 'metre' for a in self.crs.axis_info[:2]):
            raise ValueError('Map axes must use metres')
        self.proj = Proj(self.crs)
        self.inverse = Transformer.from_crs(self.crs, 'EPSG:4326', always_xy=True)

    def convergence_degrees(self, position):
        p = np.asarray(position, dtype=float)
        if p.shape != (3,) or not np.isfinite(p).all():
            raise ValueError('Finite map position required')
        lon, lat = self.inverse.transform(*(p[:2] + self.origin[:2]))
        angle = self.proj.get_factors(lon, lat).meridian_convergence
        if not np.isfinite(angle):
            raise ValueError('Projection convergence unavailable')
        return float(angle)

    def orientation(self, rotation, position, *, source_frame):
        p = np.asarray(position, dtype=float)
        if p.shape != (3,) or not np.isfinite(p).all():
            raise ValueError('Finite map position required')
        r = np.asarray(rotation, dtype=float)
        if r.shape != (3, 3) or not np.isfinite(r).all() or not np.allclose(r.T @ r, np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(r), 1.):
            raise ValueError('Proper rotation required')
        if source_frame == 'projected_grid':
            return r.copy()
        if source_frame != 'true_enu':
            raise ValueError('Orientation source must be explicit')
        # Left multiplication: change world reference, not the body mounting.
        return Rotation.from_euler('z', self.convergence_degrees(position), degrees=True).as_matrix() @ r

    def actor_box(self, actor, *, orientation_frame, center_semantics, local_offset=None):
        b = actor['bounding_box']
        center = np.array([b['location'][k] for k in 'xyz'], dtype=float)
        extent = np.array([b['extent'][k] for k in 'yxz'], dtype=float)
        if not np.isfinite(extent).all() or np.min(extent) <= 0:
            raise ValueError('Positive finite W/L/H half-extents required')
        raw_r = Rotation.from_euler('xyz', [b['rotation'][k] for k in 'xyz'], degrees=True).as_matrix()
        r = self.orientation(raw_r, center, source_frame=orientation_frame)
        if center_semantics == 'bbox_center':
            if local_offset is not None:
                raise ValueError('No implicit correction of a declared center')
        elif center_semantics == 'model_local_offset':
            d = np.asarray(local_offset, dtype=float)
            if d.shape != (3,) or not np.isfinite(d).all():
                raise ValueError('Model-local measured offset required')
            center = center + r @ d
        else:
            raise ValueError('Unknown bbox center semantics: do not guess by actor type')
        corners = np.array(list(itertools.product((-1., 1.), repeat=3))) * extent
        return center, r, extent, corners @ r.T + center

    def actor_box_from_pivot_profile(self, actor, model_name, profile, *, orientation_frame):
        """Reconstruct a center from a model reference, without modifying raw GT.

        This geometry operation is NOT training authorization. The caller must
        separately establish reference provenance, animation and motion scope.
        Never add a pivot offset to the potentially already-correct bbox center.
        """
        if profile.get('model_name') != model_name or not model_name:
            raise ValueError('Exact model reference required')
        if profile.get('object_type') != actor['actor_info']['object_type']:
            raise ValueError('Reference actor type differs')
        d = np.asarray(profile['pivot_center_offset_flu_m'], dtype=float)
        expected = np.asarray(profile['half_extent_lwh_m'], dtype=float)
        if (d.shape != (3,) or expected.shape != (3,) or
                not np.isfinite(d).all() or not np.isfinite(expected).all() or
                np.min(expected) <= 0):
            raise ValueError('Finite model reference required')
        _, r, extent, _ = self.actor_box(actor, orientation_frame=orientation_frame,
                                        center_semantics='bbox_center')
        if np.max(np.abs(extent-expected)/expected) > .05:
            raise ValueError('Scale outside measured reference')
        t = actor['transform']
        pivot = np.array([t['location'][k] for k in 'xyz'], dtype=float)
        ar = Rotation.from_euler('xyz', [t['rotation'][k] for k in 'xyz'], degrees=True).as_matrix()
        ar = self.orientation(ar, pivot, source_frame=orientation_frame)
        center = pivot + ar @ d
        corners = np.array(list(itertools.product((-1., 1.), repeat=3))) * extent
        return center, r, extent, corners @ r.T + center

    def world_to_ego(self, points, ego_position, ego_rotation, *, orientation_frame):
        """Use the same basis for objects, waypoints and navigation targets."""
        points = np.asarray(points, dtype=float)
        ego_position = np.asarray(ego_position, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError('Finite Nx3 map positions required')
        r = self.orientation(ego_rotation, ego_position, source_frame=orientation_frame)
        return (points - ego_position) @ r

    def actor_box_in_ego(self, actor, ego_position, ego_rotation, *,
                         actor_orientation_frame, ego_orientation_frame,
                         center_semantics, local_offset=None):
        """Shared static/pedestrian/vehicle geometry, with explicit center policy.

        Does not certify source timing, identity or the chosen model offset.
        Already-projected global velocity must not receive a second grid rotation.
        """
        center, r, extent, corners = self.actor_box(
            actor, orientation_frame=actor_orientation_frame,
            center_semantics=center_semantics, local_offset=local_offset)
        er = self.orientation(ego_rotation, ego_position, source_frame=ego_orientation_frame)
        ep = np.asarray(ego_position, dtype=float)
        return (center-ep) @ er, er.T @ r, extent, (corners-ep) @ er


BASIS_SCHEMA = 'morai_projected_grid_basis_v1'


def validate_map_basis(basis):
    if (basis.get('schema') != BASIS_SCHEMA
            or basis.get('position_frame') != 'projected_grid'
            or basis.get('source_orientation_frame') != 'true_enu'
            or basis.get('orientation_frame') != 'projected_grid'):
        raise ValueError('Explicit projected-position / true-ENU orientation contract required')
    return MapGeometry(basis['crs'], basis['origin_xyz'])


def grid_yaw(yaw, position, basis, *, source_frame):
    geo = validate_map_basis(basis)
    if not np.isfinite(yaw):
        raise ValueError('Finite yaw required')
    if source_frame == 'projected_grid':
        return float(yaw)
    if source_frame != basis['source_orientation_frame']:
        raise ValueError('Unknown orientation source')
    return float(yaw + np.radians(geo.convergence_degrees(position)))


def manifest_map_basis(manifests):
    """Prevent silently mixing old targets with corrected targets in one run."""
    import json
    bases = {json.dumps(doc.get('map_basis'), sort_keys=True) for _, doc in manifests}
    if len(bases) != 1:
        raise ValueError('Mixed manifest map orientation bases')
    basis = json.loads(bases.pop())
    if basis is not None:
        validate_map_basis(basis)
    return basis
