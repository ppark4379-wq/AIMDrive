import math
import socket
from copy import deepcopy

from .Camera import CameraReceiver
from .GPS import GpsReceiver
from .IMU import ImuReceiver
from .VehicleStatus import VehicleStatusReceiver
from .LiDAR import LidarReceiver
from .Cmd import CmdSender, Command
from .Detection import DetectionSender
from .Pose import PoseSender, GridProjector, DEFAULT_HZ as POSE_HZ


class UDPManager:
    def __init__(self, config):
        self.config = deepcopy(config)
        self.sensor_config = {}
        cameras = {'Cam_Front': 'rgb_front', 'Cam_Left': 'rgb_left',
                   'Cam_Right': 'rgb_right', 'Cam_Rear': 'rgb_rear'}
        for key, name in cameras.items():
            self.sensor_config[name] = dict(config['Camera'][key])
        for key in ('GPS', 'IMU', 'LiDAR'):
            self.sensor_config[key.lower()] = dict(config[key])
        ego = config['Ego']
        self.sensor_config['vehicle_status'] = {
            'bind': ego.get('bind'), 'dest_port': ego.get('vehicle_status_port')}
        self.cmd_config = dict(ego)
        # Optional: VE detections -> planning/control (see Detection.py).
        self.detection_config = dict(config.get('Detection') or {})
        self._validate_geometry()
        self.receivers = {}
        self.cmd = None
        self.detection = None
        self.pose = None
        self.localization = None
        self.running = False

    def _validate_geometry(self):
        for name, cfg in self.sensor_config.items():
            if name == 'vehicle_status':
                continue
            for key in ('pos', 'rot_deg'):
                value = cfg.get(key)
                if (not isinstance(value, (list, tuple)) or len(value) != 3
                        or any(type(x) not in (int, float) or not math.isfinite(x) for x in value)):
                    raise ValueError(f'{name}.{key}: expected three finite numbers')
            if name.startswith('rgb_'):
                for key in ('width', 'height'):
                    if type(cfg.get(key)) is not int or cfg[key] <= 0:
                        raise ValueError(f'{name}.{key}: expected positive integer')
                fov = cfg.get('fov')
                if type(fov) not in (int, float) or not 0 < fov < 180:
                    raise ValueError(f'{name}.fov: horizontal FOV must be between 0 and 180 degrees')

    @staticmethod
    def _valid_port(port):
        return type(port) is int and 1 <= port <= 65535

    def _validate_ports(self):
        errors = []
        used = set()
        for name, cfg in self.sensor_config.items():
            port = cfg.get('dest_port')
            if not self._valid_port(port):
                errors.append(f'{name}.dest_port (Ego.vehicle_status_port for vehicle_status)')
            elif port in used:
                errors.append(f'{name}: duplicate receive port {port}')
            else:
                used.add(port)
        port = self.cmd_config.get('cmd_port')
        if port is not None:
            if not self._valid_port(port):
                errors.append('Ego.cmd_port')
            if not isinstance(self.cmd_config.get('ip'), str) or not self.cmd_config['ip'].strip():
                errors.append('Ego.ip: required when cmd_port is set')
        source = self.cmd_config.get('cmd_source_port')
        if source is not None:
            if not self._valid_port(source):
                errors.append('Ego.cmd_source_port')
            elif source in used:
                errors.append(f'Ego.cmd_source_port {source} is also a sensor receive port')
        port = self.detection_config.get('port')
        if port is not None:
            if not self._valid_port(port):
                errors.append('Detection.port')
            elif port in used:
                errors.append(f'Detection.port {port} is also a sensor receive port')
            if not isinstance(self.detection_config.get('checkpoint_sha256'), str):
                errors.append('Detection.checkpoint_sha256: required when Detection.port is set')
            hz = self.detection_config.get('pose_hz', POSE_HZ)
            if hz is not None and (type(hz) not in (int, float) or not 1 <= hz <= 100):
                errors.append('Detection.pose_hz: 1..100 Hz, or null to disable the control pose')
        if errors:
            raise ValueError('Missing/invalid UDP configuration: ' + '; '.join(errors))
        if (self.cmd_config.get('ctrl_mode', 2) != 2
                or self.cmd_config.get('cmd_type', 1) != 1
                or self.cmd_config.get('protocol', 'morai') != 'morai'):
            raise ValueError('Competition permits only MORAI cmd_type=1, ctrl_mode=2')

    def _probe_ports(self):
        """Fail if another process already owns a port.

        Receivers set SO_REUSEADDR, so a second consumer would otherwise bind
        silently and split the datagrams with the first one.
        """
        ports = [(name, cfg.get('bind') or '0.0.0.0', cfg['dest_port'])
                 for name, cfg in self.sensor_config.items()]
        if self.cmd_config.get('cmd_port') is not None and self.cmd_config.get('cmd_source_port') is not None:
            ports.append(('cmd_source', self.cmd_config.get('bind') or '0.0.0.0',
                          self.cmd_config['cmd_source_port']))
        busy = []
        for name, address, port in ports:
            probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                probe.bind((address, port))
            except OSError:
                busy.append(f'{name}:{port}')
            finally:
                probe.close()
        if busy:
            raise RuntimeError('UDP port already in use (stop the other receiver first): ' + ', '.join(busy))

    def start(self):
        if self.running:
            return self
        factories = dict(gps=GpsReceiver, imu=ImuReceiver, lidar=LidarReceiver,
                         vehicle_status=VehicleStatusReceiver)
        factories.update({name: CameraReceiver for name in
                          ('rgb_front', 'rgb_left', 'rgb_right', 'rgb_rear')})
        self._validate_ports()
        self._probe_ports()
        try:
            for name, cfg in self.sensor_config.items():
                options = {key: cfg[key] for key in ('elevations', 'distance_scale', 'frame_policy')
                           if name == 'lidar' and key in cfg}
                options['history_size'] = 200 if name in ('vehicle_status', 'imu', 'gps') else 16
                self.receivers[name] = factories[name](
                    cfg['dest_port'], bind=cfg.get('bind') or '0.0.0.0', name=name,
                    drop_audit=True, **options)
            if (self.config.get('Localization') or {}).get('enabled', False):
                from AIM2026.GPSJamming.gps_jamming.localization.service import create_localization
                self.localization = create_localization(self, self.config)
                if self.localization is not None:
                    self.localization.start()
            cfg = self.cmd_config
            if cfg.get('cmd_port') is not None:
                self.cmd = CmdSender(cfg['ip'], cfg['cmd_port'], protocol=cfg.get('protocol', 'morai'),
                                     source_port=cfg.get('cmd_source_port'),
                                     bind=cfg.get('bind') or '0.0.0.0')
            cfg = self.detection_config
            if cfg.get('port') is not None:
                host = cfg.get('ip') or '127.0.0.1'
                self.detection = DetectionSender(cfg['checkpoint_sha256'], host, cfg['port'],
                                                  generation=cfg.get('generation'))
                if cfg.get('pose_hz', POSE_HZ) is not None:
                    # Rule 2-6-2/2-8: GPS position, IMU heading, Competition Vehicle Status speed.
                    projector = GridProjector(self.sensor_config['gps']['pos'],
                                              self.sensor_config['imu']['rot_deg'],
                                              **(dict(crs=self.localization.basis['crs'],
                                                       origin_xyz=self.localization.basis['origin_xyz'])
                                                 if self.localization is not None else {}))
                    self.pose = PoseSender(self.localization or self.receivers['gps'], self.receivers['imu'],
                                           self.receivers['vehicle_status'], projector,
                                           cfg['checkpoint_sha256'], self.detection.generation,
                                           host, cfg['port'], hz=float(cfg.get('pose_hz', POSE_HZ))).start()
            self.running = True
            return self
        except BaseException:
            self.close()
            raise

    def snapshot(self):
        values = {name: receiver.snapshot() for name, receiver in self.receivers.items()}
        if self.localization is not None:
            values['gps'] = self.localization.snapshot()
        return values

    def recent(self):
        values = {name: receiver.recent() for name, receiver in self.receivers.items()}
        if self.localization is not None:
            values['gps'] = self.localization.recent()
        return values

    def diagnostics(self):
        """Per-receiver packet/sample counts and rejection reasons."""
        result = {name: receiver.diagnostics() for name, receiver in self.receivers.items()}
        if self.localization is not None:
            result['gps']['localization'] = self.localization.diagnostics()
        return result

    def send_control(self, accel, brake, steer):
        if not self.running or self.cmd is None:
            raise RuntimeError('UDP manager is not running or Cmd is not configured')
        cfg = self.cmd_config
        return self.cmd.send(Command(ctrl_mode=2,
                                     gear=cfg.get('gear', 4), cmd_type=1,
                                     accel=accel, brake=brake, steer=steer))

    def send_detection(self, boxes, ego_pose, sensor_received_at, **kwargs):
        """VE boxes to planning/control; see DetectionSender.send."""
        if not self.running or self.detection is None:
            raise RuntimeError('UDP manager is not running or Detection is not configured')
        return self.detection.send(boxes, ego_pose, sensor_received_at, **kwargs)

    def send_detection_not_ready(self, reason):
        if not self.running or self.detection is None:
            raise RuntimeError('UDP manager is not running or Detection is not configured')
        return self.detection.send_not_ready(reason)

    def close(self):
        if self.pose is not None:
            self.pose.close()
            self.pose = None
        if self.localization is not None:
            self.localization.close()
            self.localization = None
        for receiver in self.receivers.values():
            receiver.close()
        self.receivers.clear()
        if self.cmd is not None:
            self.cmd.close()
            self.cmd = None
        if self.detection is not None:
            self.detection.close()
            self.detection = None
        self.running = False

    stop = close

    def __enter__(self):
        return self.start()

    def __exit__(self, *args):
        self.close()
