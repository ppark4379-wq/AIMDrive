"""MORAI GNSS (WGS84) to MGeo local-coordinate conversion."""
import json
import os


class MGeoReference:
    def __init__(self, global_info_path: str):
        from pyproj import CRS, Transformer

        with open(global_info_path, encoding="utf-8") as f:
            info = json.load(f)
        self.origin = tuple(float(v) for v in info["local_origin_in_global"])
        self.transformer = Transformer.from_crs(
            CRS.from_epsg(4326),
            CRS.from_proj4(info["global_coordinate_system"]),
            always_xy=True,
        )

    @classmethod
    def from_map_dir(cls, map_dir: str):
        path = os.path.join(map_dir, "global_info.json") if map_dir else ""
        return cls(path) if path and os.path.isfile(path) else None

    @classmethod
    def from_optional_path(cls, global_info_path: str, map_dir: str = ""):
        if global_info_path:
            path = os.path.abspath(os.path.expanduser(global_info_path))
            if not os.path.isfile(path):
                raise FileNotFoundError(f"global_info.json not found: {path}")
            return cls(path)
        return cls.from_map_dir(map_dir)

    def to_local(self, latitude: float, longitude: float, altitude: float = 0.0):
        easting, northing = self.transformer.transform(longitude, latitude)
        return (
            easting - self.origin[0],
            northing - self.origin[1],
            altitude - self.origin[2],
        )
