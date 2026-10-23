"""Projected-map position and meridian convergence contract for localization.

Extracted without changing the original validation or projection calculations.
No perception, vehicle-control, or UDP dependencies.
"""

import numpy as np
from pyproj import CRS, Proj, Transformer

class MapGeometry:

    def __init__(self, crs, origin_xyz):
        self.crs = CRS.from_user_input(crs)
        self.origin = np.asarray(origin_xyz, dtype=float)
        if not self.crs.is_projected or self.origin.shape != (3,) or (not np.isfinite(self.origin).all()):
            raise ValueError('Explicit projected CRS and finite map origin required')
        if any((a.unit_name != 'metre' for a in self.crs.axis_info[:2])):
            raise ValueError('Map axes must use metres')
        self.proj = Proj(self.crs)
        self.inverse = Transformer.from_crs(self.crs, 'EPSG:4326', always_xy=True)

    def convergence_degrees(self, position):
        p = np.asarray(position, dtype=float)
        if p.shape != (3,) or not np.isfinite(p).all():
            raise ValueError('Finite map position required')
        lon, lat = self.inverse.transform(*p[:2] + self.origin[:2])
        angle = self.proj.get_factors(lon, lat).meridian_convergence
        if not np.isfinite(angle):
            raise ValueError('Projection convergence unavailable')
        return float(angle)
BASIS_SCHEMA = 'morai_projected_grid_basis_v1'

def validate_map_basis(basis):
    if basis.get('schema') != BASIS_SCHEMA or basis.get('position_frame') != 'projected_grid' or basis.get('source_orientation_frame') != 'true_enu' or (basis.get('orientation_frame') != 'projected_grid'):
        raise ValueError('Explicit projected-position / true-ENU orientation contract required')
    return MapGeometry(basis['crs'], basis['origin_xyz'])
