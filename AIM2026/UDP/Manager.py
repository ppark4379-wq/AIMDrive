import math
from copy import deepcopy

from .Camera import CameraReceiver
from .GPS import GpsReceiver
from .IMU import ImuReceiver
from .VehicleStatus import VehicleStatusReceiver
from .LiDAR import LidarReceiver
from .Cmd import CmdSender, Command


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
        self._validate_geometry()
        self.receivers = {}
        self.cmd = None
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

    def _validate_ports(self):
        errors = []
        used = set()
        for name, cfg in self.sensor_config.items():
            port = cfg.get('dest_port')
            if type(port) is not int or not 1 <= port <= 65535:
                errors.append(f'{name}.dest_port (Ego.vehicle_status_port for vehicle_status)')
            elif port in used:
                errors.append(f'{name}: duplicate receive port {port}')
            else:
                used.add(port)
        port = self.cmd_config.get('cmd_port')
        if port is not None:
            if type(port) is not int or not 1 <= port <= 65535:
                errors.append('Ego.cmd_port')
            if not isinstance(self.cmd_config.get('ip'), str) or not self.cmd_config['ip'].strip():
                errors.append('Ego.ip: required when cmd_port is set')
        if errors:
            raise ValueError('Missing/invalid UDP configuration: ' + '; '.join(errors))

    def start(self):
        if self.running:
            return self
        factories = dict(gps=GpsReceiver, imu=ImuReceiver, lidar=LidarReceiver,
                         vehicle_status=VehicleStatusReceiver)
        factories.update({name: CameraReceiver for name in
                          ('rgb_front', 'rgb_left', 'rgb_right', 'rgb_rear')})
        self._validate_ports()
        try:
            for name, cfg in self.sensor_config.items():
                options = {key: cfg[key] for key in ('elevations', 'distance_scale')
                           if name == 'lidar' and key in cfg}
                self.receivers[name] = factories[name](
                    cfg['dest_port'], bind=cfg.get('bind') or '0.0.0.0', name=name, **options)
            cfg = self.cmd_config
            if cfg.get('cmd_port') is not None:
                self.cmd = CmdSender(cfg['ip'], cfg['cmd_port'], protocol=cfg.get('protocol', 'morai'))
            self.running = True
            return self
        except BaseException:
            self.close()
            raise

    def snapshot(self):
        return {name: receiver.snapshot() for name, receiver in self.receivers.items()}

    def send_control(self, accel, brake, steer):
        if not self.running or self.cmd is None:
            raise RuntimeError('UDP manager is not running or Cmd is not configured')
        cfg = self.cmd_config
        return self.cmd.send(Command(ctrl_mode=cfg.get('ctrl_mode', 2),
                                     gear=cfg.get('gear', 4), cmd_type=1,
                                     accel=accel, brake=brake, steer=steer))

    def close(self):
        for receiver in self.receivers.values():
            receiver.close()
        self.receivers.clear()
        if self.cmd is not None:
            self.cmd.close()
            self.cmd = None
        self.running = False

    stop = close

    def __enter__(self):
        return self.start()

    def __exit__(self, *args):
        self.close()
