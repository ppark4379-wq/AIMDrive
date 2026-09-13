"""Explicit Simulation Time integration; never manufactures a ROS /clock."""
import time


def publish_final_brake(topic, observer):
    """Stop owned expert first; then verify the actual ROS control handoff."""
    import rospy
    from morai_msgs.msg import CtrlCmd
    if not rospy.core.is_initialized():
        rospy.init_node('scenario_review_brake',anonymous=True,disable_signals=True)
    code,message,state=rospy.get_master().getSystemState()
    if code!=1 or dict(state[0]).get(topic):
        raise RuntimeError('Cannot brake while another control publisher exists')
    publisher=rospy.Publisher(topic,CtrlCmd,queue_size=1)
    try:
        end=time.monotonic()+3
        while not publisher.get_num_connections() and time.monotonic()<end:time.sleep(.05)
        if not publisher.get_num_connections():raise RuntimeError('No brake control subscriber')
        for _ in range(15):
            publisher.publish(CtrlCmd(longlCmdType=1,accel=0,brake=1,front_steer=0))
            time.sleep(.1)
        observation=observer.actors()
        if not observation.get('available'):raise RuntimeError('Brake observation unavailable')
        ego=next(s for s in observation['response']['states'] if s['actor_info']['id']['value']=='Ego')
        state=ego['vehicle_state']
        if state['brake']<.99 or state['throttle']>.01:raise RuntimeError('Final brake was not applied')
        return dict(brake=state['brake'],throttle=state['throttle'],velocity=ego['velocity'],
                    brake_observation_receive_monotonic_ns=observation['receive_monotonic_ns'])
    finally:publisher.unregister()


class SimulationObservationClock:
    domain = 'grpc_simulation_timestamp'

    def __init__(self, observer):
        self.observer = observer
        self.previous_us = None
        self.last_progress_wall = time.monotonic()
        self.last_query_wall = float('-inf')

    def now(self):
        wall = time.monotonic()
        if self.previous_us is not None and wall-self.last_query_wall < .04:
            return self.previous_us / 1e6
        value = self.observer.actors()
        if not value.get('available'):
            raise RuntimeError('Simulation clock observation unavailable')
        stamp = int(value['response']['timestamp'])
        if self.previous_us is not None and stamp < self.previous_us:
            raise RuntimeError('Simulation clock reset within episode')
        if stamp != self.previous_us:
            self.last_progress_wall = wall
        elif wall-self.last_progress_wall > 10:
            raise RuntimeError('Simulation clock stopped for 10 wall seconds')
        self.previous_us, self.last_query_wall = stamp, wall
        return stamp / 1e6


def start_simulation_time_world(adapter, ego_transform):
    """Same public start contract as copied adapter, with explicit TIME_MANAGEMENT."""
    from proto.morai.simulation.start_param_pb2 import MapAndVehicle
    from proto.morai.actor.actor_set_pb2 import EgoCruiseControl
    from proto.morai.actor.actor_enum_pb2 import EGO_CRUISE_TYPE_LINK
    from proto.morai.simulation.sync_mode_pb2 import SyncMode
    from proto.morai.simulation.simulation_enum_pb2 import SYNC_MODE_TYPE_TIME_MANAGEMENT
    config = adapter.global_cfg['morai']
    adapter.client.start_simulation(
        map_and_vehicle=MapAndVehicle(map_name=config['map_name'],
                                     ego_vehicle_model=config['ego_vehicle_model']),
        ego_transform=ego_transform,
        ego_cruise_setting=EgoCruiseControl(cruise_on=False,cruise_type=EGO_CRUISE_TYPE_LINK,
                                          link_speed_ratio=0,constant_velocity=0),
        sync_mode=SyncMode(type=SYNC_MODE_TYPE_TIME_MANAGEMENT))
    adapter.world = adapter.client.get_simulation_world()
    if adapter.world is None:
        raise RuntimeError('Simulation Time world start failed')
    adapter._patch_spawn_diagnostics()


class RosClockGuard:
    """Observe the real /clock; wall-time deadlines never wait for simulated time."""
    def __init__(self):
        import rospy
        import threading
        from rosgraph_msgs.msg import Clock
        self.lock = threading.Lock()
        self.last_ns = None
        self.last_progress = None
        self.armed = False
        self.reset_seen = False
        self.count = 0
        self.duplicates = 0
        if not rospy.core.is_initialized():
            rospy.init_node('scenario_clock_guard', anonymous=True, disable_signals=True)
        self.subscriber = rospy.Subscriber('/clock', Clock, self.receive, queue_size=2000)

    def receive(self, message):
        stamp = message.clock.to_nsec()
        with self.lock:
            if self.last_ns is not None and stamp < self.last_ns and self.armed:
                self.reset_seen = True
            if stamp != self.last_ns:
                self.last_progress = time.monotonic()
            else:
                self.duplicates += 1
            self.last_ns = stamp
            self.count += 1

    def check(self):
        import rospy
        with self.lock:
            if self.reset_seen:
                raise RuntimeError('/clock reset during episode')
            if self.last_progress is None or time.monotonic()-self.last_progress > 3:
                raise RuntimeError('/clock missing or not advancing for 3 wall seconds')
            if rospy.get_param('/use_sim_time', False) is not True:
                raise RuntimeError('ROS simulation clock disabled during episode')
            return dict(clock_ns=self.last_ns, messages=self.count, duplicates=self.duplicates,
                        expert_clock='ros_clock', use_sim_time=True)

    def require_progress(self):
        with self.lock:
            before = self.last_ns
        deadline = time.monotonic()+5
        while time.monotonic() < deadline:
            with self.lock:
                advanced = self.last_ns is not None and before is not None and self.last_ns > before
                if before is None: before = self.last_ns
            if advanced: return self.check()
            time.sleep(.02)
        raise RuntimeError('No advancing /clock within 5 wall seconds')

    def arm(self):
        evidence = self.require_progress()
        with self.lock:
            self.armed = True
        return evidence

    def close(self):
        self.subscriber.unregister()
