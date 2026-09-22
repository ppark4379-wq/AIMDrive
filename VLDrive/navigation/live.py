"""Anchor-aligned navigation from shared GPS/status snapshots; owns no sockets."""
import math
import time

from navigation.engine import NavigationEngine
from navigation.route_source import load_live_route


class LiveNavigation:
    def __init__(self, document, *, max_anchor_skew_s=.15):
        from pyproj import Transformer
        if document.get('schema') != 'morai-navigation-observer-v1' or not document.get('route'):
            raise ValueError('Live VLDrive navigation requires the configured global path / map route')
        self.episode, manifest, _ = load_live_route(document['route'])
        self.policy = dict(manifest['policy'])
        self.position = document['position']
        self.runtime = document['runtime']
        # Live polling can skip an intermediate image while GPS remains fresh.
        # Match instruction continuity to the explicitly configured live pose
        # timeout, not the offline recording's adjacent-frame audit interval.
        self.policy['max_sensor_gap_ms'] = max(self.policy.get('max_sensor_gap_ms',150.),
                                             1000*self.runtime['max_age_s'])
        self.max_anchor_skew = max_anchor_skew_s
        self.converter = Transformer.from_crs('EPSG:4326', self.position['map_crs'], always_xy=True)
        self.reset()

    def reset(self):
        self.engine = NavigationEngine(self.episode['route'], self.policy, 'live:' + self.episode['episode_id'])
        self.last_stamp = None
        self.last_result = None

    def unavailable(self, reason):
        self.reset()
        return dict(valid=False, reason=reason, instruction=None, pose=None)

    def update(self, gps, status, *, anchor_stamp, now=None):
        now = time.monotonic() if now is None else now
        for name, snapshot in (('gps', gps), ('status', status)):
            payload, _, _, received = snapshot
            if payload is None or received is None:
                return self.unavailable(name + '_no_samples')
            if not 0 <= now - received <= self.runtime['max_age_s']:
                return self.unavailable(name + '_stale')
            if abs(received - anchor_stamp) > self.max_anchor_skew:
                return self.unavailable(name + '_anchor_skew')
        if abs(gps[3] - status[3]) > self.runtime['max_pair_skew_s']:
            return self.unavailable('gps_status_skew')
        fix, vehicle = gps[0], status[0]
        try:
            latitude, longitude = float(fix['latitude']), float(fix['longitude'])
            if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
                return self.unavailable('invalid_gps_coordinate')
            east, north = self.converter.transform(longitude, latitude)
            pose = dict(x=east-self.position['east_offset_m'], y=north-self.position['north_offset_m'],
                        yaw_rad=float(vehicle['yaw']), speed_mps=float(vehicle['speed']),
                        brake=float(vehicle['brake']), sim_time=float(vehicle['sim_time']))
        except (KeyError, TypeError, ValueError, OverflowError):
            return self.unavailable('invalid_navigation_pose')
        if not all(math.isfinite(v) for v in pose.values()) or pose['sim_time'] < 0:
            return self.unavailable('invalid_navigation_pose')
        stamp = round(pose['sim_time'] * 1e9)
        if stamp != self.last_stamp:
            self.last_result = self.engine.update(dict(pose, sensor_source_ns=stamp))
            self.last_stamp = stamp
        return dict(self.last_result, pose=pose, gps_sample_id=gps[2], status_sample_id=status[2],
                    gps_received_at=gps[3], status_received_at=status[3])


def navigation_for_frame(provider, cfg, meta):
    if cfg['navigation']['mode'] == 'fixed':
        return dict(valid=True, instruction=dict(text=cfg['navigation_instruction'], rule='fixed'),
                    pose=None, source='fixed_config')
    return provider.update(meta['gps_snapshot'], meta['status_snapshot'],
                           anchor_stamp=meta['received_at']['lidar'])
