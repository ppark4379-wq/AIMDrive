"""GPS/IMU -> planned-path geometry, Navigation text and map Zone ID.

No simulator object/traffic-light queries and no recorded ego/GT labels.
"""
import hashlib
import math
from pathlib import Path

import numpy as np
import yaml


ZONE_IDS = dict(static_obstacle=1, dynamic_obstacle=2, roundabout=3, highway=4, tunnel=5)


class RouteZones:
    """Same official checkpoint/index boundaries as Situation GT labeling."""
    def __init__(self, points, definition):
        self.points = np.asarray(points, dtype=np.float64)[:, :2]
        self.arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(self.points, axis=0), axis=1))]
        self.closed = np.linalg.norm(self.points[-1] - self.points[0]) <= .01
        checkpoints = {}
        for row in definition['checkpoints']:
            i = row['path_index']
            if (type(i) is not int or not 0 <= i < len(self.points)
                    or np.linalg.norm(self.points[i] - [row['x'], row['y']]) > .01):
                raise ValueError('Zone checkpoint does not match the official path')
            checkpoints[str(row['id'])] = i
        zones = definition['zones']
        indices = []
        for name, value in zones.items():
            if name not in ZONE_IDS:
                continue
            if name == 'static_obstacle':
                start, end = checkpoints['3'], checkpoints['4']
                if not start <= value['route_path_index'] <= end:
                    raise ValueError('Static obstacle anchor outside checkpoints 3-4')
            elif name == 'dynamic_obstacle':
                start, end = value['start_path_index'], value['end_path_index']
            else:
                prefix = '' if name == 'tunnel' else 'ego_'
                start = checkpoints[str(value[prefix + 'entry_checkpoint'])]
                end = checkpoints[str(value[prefix + 'exit_checkpoint'])]
            if not 0 <= start <= end < len(self.points):
                raise ValueError('Invalid official zone bounds')
            indices.append((ZONE_IDS[name], start, end))
        for i, (_, start, end) in enumerate(indices):
            if any(start <= e and s <= end for _, s, e in indices[i+1:]):
                raise ValueError('Overlapping official zone bounds')
        self.intervals = [(zone, float(self.arc[start]), float(self.arc[end]))
                          for zone, start, end in indices]

    def at(self, progress_m):
        if self.closed:
            progress_m %= self.arc[-1]
        for zone, start, end in self.intervals:
            if start <= progress_m <= end:
                return zone
        return 0


class RouteNavigation:
    def __init__(self, config, encoder_contract):
        from instruction.global_path.preprocess.process import process_path
        from instruction.global_path.encoder import PathEncoder
        path = Path(config['global_path'])
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        expected = encoder_contract.get('route_source_sha256s', [])
        if digest not in expected:
            raise ValueError('Official global path differs from the VE training route')
        points = np.loadtxt(path, ndmin=2)
        # Keep the training geometry algorithm and its defaults. No hand-tuned
        # text distances: numeric geometry still goes through the learned MLP.
        self.encoder = PathEncoder(process_path(path, **config.get('geometry', {})),
                                   **config.get('encoder', {}))
        self.instruction_options = config.get('instruction', {})
        definition = yaml.safe_load(Path(config['zone_definition']).read_text(encoding='utf-8'))[config['route_key']]
        self.zones = RouteZones(points, definition)
        self.previous_s = None

    def reset(self):
        self.previous_s = None

    def update(self, pose):
        from instruction.navigation.selector import select_instruction
        projection = self.encoder.project(pose, self.previous_s)
        if projection is None:
            self.reset()
            raise ValueError('GPS/IMU pose cannot be projected onto the planned route')
        self.previous_s = projection['progress_m']
        selection = select_instruction(self.encoder, [projection], **self.instruction_options)
        event = selection['event'] or self.encoder.next_event(self.previous_s)
        # A new planned bend is a new instruction segment even if the sentence
        # happens to be the same (e.g. two successive left bends).
        segment = (selection['text'], event['instance_id'] if event else 'straight')
        features = self.encoder.encode(projection, event)
        return dict(instruction=selection['text'], segment=segment, path_features=features,
                    zone_id=self.zones.at(self.previous_s), progress_m=self.previous_s)


class PoseProjector:
    def __init__(self, map_basis, gps_mount, imu_mount_deg):
        from pyproj import Transformer
        from data.src.loaders.morai_map_geometry import validate_map_basis
        from scipy.spatial.transform import Rotation
        self.geometry = validate_map_basis(map_basis)
        self.to_map = Transformer.from_crs('EPSG:4326', map_basis['crs'], always_xy=True)
        self.origin = np.asarray(map_basis['origin_xyz'], dtype=float)
        self.gps_mount = np.asarray(gps_mount, dtype=float)
        self.imu_mount = Rotation.from_euler('xyz', imu_mount_deg, degrees=True).as_matrix()

    def project(self, gps, imu):
        from scipy.spatial.transform import Rotation
        lat, lon = gps['latitude'], gps['longitude']
        if not (math.isfinite(lat) and math.isfinite(lon) and -90 <= lat <= 90 and -180 <= lon <= 180):
            raise ValueError('Invalid GPS fix')
        x, y = self.to_map.transform(lon, lat)
        position = np.array([x, y, gps.get('altitude') or 0.]) - self.origin
        rotation = Rotation.from_quat(imu['quaternion']).as_matrix() @ self.imu_mount.T
        rotation = self.geometry.orientation(rotation, position, source_frame='true_enu')
        position -= rotation @ self.gps_mount
        return dict(x=float(position[0]), y=float(position[1]),
                    yaw_rad=math.atan2(rotation[1, 0], rotation[0, 0]))
