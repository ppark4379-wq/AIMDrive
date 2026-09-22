"""CPU waypoint control worker; independent of GPU inference and rendering."""
import json
import math
import threading
import time
import traceback

from control.transport import BridgeClient
from control.waypoint_pid import current_waypoints, stop_command
from control.expert_controller import ExpertWaypointController


class ControllerSession:
    def __init__(self, cfg, config_path, output, *, view=None, clock=time.monotonic):
        self.cfg, self.config_path, self.output, self.view = cfg, config_path, output, view
        self.clock = clock
        self.pid = ExpertWaypointController(cfg)
        self.lock = threading.Lock()
        self.stopping = threading.Event()
        self.result = None
        self.reason = 'warming_up'
        self.client = self.thread = None
        self.error = None
        self.closed = False
        self.command = stop_command('warming_up')
        self.lateral_context = None
        self.steering_positive_direction = ('left' if
            cfg['expert_config']['common']['realtime_steering_sign']<0 else 'right')
        self.state = dict(enabled=True, status='waiting', command=self.command,
                          steering_positive_direction=self.steering_positive_direction)
        self.counts = dict(packets=0, pid_updates=0, lateral_updates=0, tracking_packets=0, hold_packets=0)

    def submit(self, result):
        # Prediction dictionaries are immutable after publishing in the pipeline.
        with self.lock:
            self.result = result

    def invalidate(self, reason):
        with self.lock:
            self.result, self.reason = None, reason

    def _pose(self, now):
        run = self.cfg['runtime']
        gps, status = self.source.gps.snapshot(), self.source.status.snapshot()
        # A receiver may publish a packet after the control tick started.
        now = max(now, self.clock())
        for name, snapshot, limit in [('gps', gps, run['max_pose_age_s']),
                                      ('status', status, run['max_status_age_s'])]:
            if snapshot[0] is None or snapshot[3] is None or not 0 <= now-snapshot[3] <= limit:
                raise ValueError(name+'_stale')
        if abs(gps[3]-status[3]) > run['max_pose_skew_s']:
            raise ValueError('gps_status_skew')
        fix, vehicle = gps[0], status[0]
        lat, lon = float(fix['latitude']), float(fix['longitude'])
        if not -90 <= lat <= 90 or not -180 <= lon <= 180:
            raise ValueError('invalid_gps')
        east, north = self.navigation.converter.transform(lon, lat)
        position = self.navigation.position
        pose = dict(x=east-position['east_offset_m'], y=north-position['north_offset_m'],
                    yaw_rad=float(vehicle['yaw']), speed_mps=float(vehicle['speed']),
                    sim_time=float(vehicle['sim_time']))
        if not all(math.isfinite(v) for v in pose.values()) or pose['sim_time'] < 0:
            raise ValueError('invalid_pose')
        # UDP Ego Vehicle Status reports steer in degrees, not normalized units.
        measured=vehicle.get('steer')
        pose['steering_measured_deg'] = float(measured) if measured is not None and math.isfinite(float(measured)) else None
        return pose

    def calculate(self, now):
        """Validate and calculate at the expert realtime rate (send_hz=pid_hz)."""
        run = self.cfg['runtime']
        with self.lock:
            result, reason = self.result, self.reason
        try:
            if result is None:
                raise ValueError(reason)
            age = now-float(result['anchor_received_at'])
            if not math.isfinite(age) or not 0 <= age <= run['max_prediction_age_s']:
                raise ValueError('prediction_stale')
            if result['history_generation'] != self.history.resets:
                raise ValueError('history_reset')
            nav = self.pipeline_state['navigation']
            if self.pipeline_state['error'] or self.stopping.is_set():
                raise ValueError('pipeline_stopped')
            if nav is None or not nav['valid']:
                raise ValueError('navigation_unavailable')
            if self.pipeline_state['missing']:
                raise ValueError('sensor_unavailable')
            # GPU inference can block the main loop. Check receivers here too.
            for name, _, _, samples, stream_age in self.source.stream_report():
                if name != 'imu' and (not samples or not 0 <= stream_age <= self.sensor_max_age):
                    raise ValueError(name+'_stale')
            self.client.poll()
            if self.client.last_ack is None or now-self.client.last_ack > self.cfg['bridge']['watchdog_s']:
                raise ValueError('bridge_ack_stale')
            pose = self._pose(now)
            if pose['sim_time'] < result['sim_time']:
                raise ValueError('sim_time_rewind')
            points, horizons, indices = current_waypoints(result, pose, age,
                max_pose_jump_m=run['max_pose_jump_m'],selection='expert')
            # Expert's realtime transport supplies fixed 1/realtime_sampling_rate
            # dt (network/synchronized_inputs.py), not measured wall-clock jitter.
            # Lateral PID has its own LMDrive cadence. Never let a new frame
            # renew the validity of steering held from an expired prediction.
            force = (self.lateral_context is not None and
                     now-self.lateral_context['anchor_received_at']>run['max_prediction_age_s'])
            if self.lateral_context is not None and self.cfg['lateral']['target']['mode']=='remaining_path':
                dx,dy = (self.lateral_context['aim_world'][0]-pose['x'],
                         self.lateral_context['aim_world'][1]-pose['y'])
                forward = math.cos(pose['yaw_rad'])*dx+math.sin(pose['yaw_rad'])*dy
                # Do not hold a command toward a point already passed between
                # PID updates. Re-select immediately using the current pose.
                force = force or forward<self.cfg['lateral']['target']['min_forward_m']
            self.command = self.pid.step(points, pose['speed_mps'], horizons,
                                         now=now,force_lateral=force)
            if self.command['lateral_updated']:
                left,forward = self.command['aim_left_forward']
                c,s = math.cos(pose['yaw_rad']),math.sin(pose['yaw_rad'])
                self.lateral_context = dict(frame=result['frame'],
                    anchor_received_at=result['anchor_received_at'],
                    aim_world=[pose['x']+c*forward-s*left,pose['y']+s*forward+c*left])
                self.counts['lateral_updates'] += 1
            lateral = self.lateral_context
            # A held steering target is displayed at its world position,
            # reprojected into the current pose (not yesterday's local axes).
            dx,dy = lateral['aim_world'][0]-pose['x'],lateral['aim_world'][1]-pose['y']
            c,s = math.cos(pose['yaw_rad']),math.sin(pose['yaw_rad'])
            self.command.update(aim_left_forward=[-s*dx+c*dy,c*dx+s*dy],
                aim_distance_m=math.hypot(dx,dy),aim_is_behind=c*dx+s*dy<=0,
                lateral_prediction_frame=lateral['frame'],
                lateral_prediction_age_s=now-lateral['anchor_received_at'],
                lateral_command_age_s=now-self.command['lateral_calculated_at'])
            self.command.update(prediction_frame=result['frame'],waypoint_indices=indices,
                                command_anchor_received_at=result['anchor_received_at'],
                                control_pose=pose,control_calculated_at=now)
            self.counts['pid_updates'] += 1
            return dict(self.command, prediction_age_s=now-self.command['command_anchor_received_at'],
                        current_pose=pose,steering_measured_deg=pose['steering_measured_deg'])
        except (ValueError, KeyError, TypeError, OverflowError) as exc:
            if self.command['reason'] == 'tracking':
                self.pid.reset()
            self.lateral_context = None
            self.command = stop_command(str(exc))
            return dict(self.command)

    def start(self, source, navigation, history, pipeline_state, sensor_max_age):
        self.source, self.navigation, self.history = source, navigation, history
        self.pipeline_state, self.sensor_max_age = pipeline_state, sensor_max_age
        self.client = BridgeClient(self.cfg, self.config_path, self.output)
        self.thread = threading.Thread(target=self._run, name='vldrive-pid', daemon=True)
        self.thread.start()
        print('LMDrive lateral PID '+str(self.cfg['lateral']['update_hz'])+' Hz | expert longitudinal '+str(self.cfg['runtime']['pid_hz'])+' Hz / UDP '+str(self.cfg['runtime']['send_hz'])+
              ' Hz | MORAI '+str(self.cfg['bridge']['target_port']), flush=True)

    def _run(self):
        period = 1/self.cfg['runtime']['send_hz']
        deadline = time.monotonic()
        try:
            with (self.output/'control.jsonl').open('x', encoding='utf-8', buffering=1) as log:
                while not self.stopping.is_set():
                    now = time.monotonic()
                    command = self.calculate(now)
                    if self.stopping.is_set():
                        break
                    self.client.send_command(command)
                    self.counts['packets'] += 1
                    self.counts['tracking_packets' if command['reason']=='tracking' else 'hold_packets'] += 1
                    ack_age = None if self.client.last_ack is None else max(0., now-self.client.last_ack)
                    self.state = dict(enabled=True, status=command['reason'], command=command,
                                      steering_positive_direction=self.steering_positive_direction,
                                      bridge_ack_age_s=ack_age, bridge_target=list(self.client.target),
                                      counts=dict(self.counts))
                    log.write(json.dumps(dict(monotonic_s=now, **self.state), allow_nan=False)+'\n')
                    if self.view is not None:
                        self.view.update(control=self.state)
                    if ack_age is None or ack_age > self.cfg['bridge']['idle_exit_s']:
                        raise RuntimeError('Control bridge acknowledgements lost')
                    deadline += period
                    if deadline < time.monotonic():
                        deadline = time.monotonic()+period
                    self.stopping.wait(max(0., deadline-time.monotonic()))
        except Exception:
            self.error = traceback.format_exc()
        finally:
            self.stopping.set()
            self.client.close()
            self.state = dict(enabled=True, status='stopped', command=stop_command('stopped'),
                              steering_positive_direction=self.steering_positive_direction,
                              error=self.error, counts=dict(self.counts))
            if self.view is not None:
                self.view.update(control=self.state)

    def check_error(self):
        if self.error:
            raise RuntimeError('PID worker failed:\n'+self.error)

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.stopping.set()
        if self.thread is not None:
            self.thread.join(timeout=2)
        elif self.client is not None:
            self.client.close()
