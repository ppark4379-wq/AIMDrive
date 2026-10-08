"""Parked pose alignment and fresh-Expert handoff for tunnel trials only."""

import atexit
import copy
import os
import threading
import json
import math
import time

from ..full_loop_parts.tunnel_static_parts import live
from .modes import TUNNEL_NORMAL_START


def signed_angle(value):
    return (float(value)+180.) % 360.-180.


def _parts(grpc):
    ego = grpc.get_ego()
    return grpc.world._sim_adapter._actor_stub, ego.get_object_info()


def _accepted(operation, request):
    from proto.morai.common.enum_pb2 import STATUS_CODE_SUCCESS
    response = operation(request, timeout=.5)
    if response is None or response.status != STATUS_CODE_SUCCESS:
        raise RuntimeError("Tunnel startup command was rejected")


def block_and_park(grpc):
    from proto.morai.actor.actor_enum_pb2 import VEHICLE_CONTROL_KEYBOARD, GEAR_MODE_P
    from proto.morai.actor.actor_set_pb2 import VehicleControlModeParam, SetVelocityParam, VehicleGear
    stub, info = _parts(grpc)
    _accepted(stub.SetVehicleControlMode, VehicleControlModeParam(actor_info=info, mode=VEHICLE_CONTROL_KEYBOARD))
    _accepted(stub.SetVelocity, SetVelocityParam(actor_info=info, velocity=0.))
    _accepted(stub.SetVehicleGear, VehicleGear(actor_info=info, gear=GEAR_MODE_P))


def parked_readback(grpc, reference=None, *, expected_mode=None):
    from proto.morai.actor.actor_enum_pb2 import VEHICLE_CONTROL_KEYBOARD, GEAR_MODE_P
    stub, info = _parts(grpc)
    state = stub.GetActorState(info, timeout=.5)
    observed = live._observation(state)
    mode = int(stub.GetVehicleControlMode(info, timeout=.5).mode)
    expected_mode = int(VEHICLE_CONTROL_KEYBOARD) if expected_mode is None else int(expected_mode)
    if mode != expected_mode or int(state.vehicle_state.gear_mode) != int(GEAR_MODE_P):
        raise RuntimeError("Tunnel startup requires the expected control mode and observed P gear")
    live._stationary(observed, reference, "Tunnel startup Ego")
    observed.update(control_mode=mode, gear_mode=int(state.vehicle_state.gear_mode),
                    front_wheel_angle=float(state.vehicle_state.front_wheel_angle),
                    steer=float(state.vehicle_state.steer), throttle=float(state.vehicle_state.throttle),
                    brake=float(state.vehicle_state.brake), read_monotonic=time.monotonic())
    return observed


def align_parked_ego(grpc, evidence, *, settle_fn, now=None, sleep=None):
    """Measure yaw conversion instead of encoding this MORAI build's bias.

    Only the parked Ego is transformed, before any fixture measurements. The
    current settled physical height is retained; road/body margins are untouched.
    """
    from proto.morai.actor.actor_set_pb2 import SetTransformParam
    now, sleep = now or time.monotonic, sleep or time.sleep
    started = now()
    evidence.update(status="aligning", target_yaw_deg=TUNNEL_NORMAL_START.yaw_deg,
                    yaw_tolerance_deg=.15, xy_tolerance_m=.02, timeout_sec=15., corrections=[])
    command_yaw = TUNNEL_NORMAL_START.yaw_deg
    try:
        for attempt in range(4):
            remaining = 15.-(now()-started)
            if remaining <= .6:
                raise RuntimeError("Parked Ego start alignment exceeded its deadline")
            settling = {}
            settle_fn(grpc, settling, timeout_sec=min(5., remaining))
            observed = parked_readback(grpc)
            xy_error = math.dist(observed["position_xyz"][:2], TUNNEL_NORMAL_START.xyz[:2])
            yaw_error = signed_angle(observed["rotation_xyz_deg"][2]-TUNNEL_NORMAL_START.yaw_deg)
            evidence["last_readback"] = observed
            evidence["last_settle"] = settling
            evidence["yaw_error_deg"], evidence["xy_error_m"] = yaw_error, xy_error
            if now()-started >= 15.:
                raise RuntimeError("Parked Ego start alignment exceeded its deadline")
            if xy_error > .02:
                raise RuntimeError("Parked Ego XY does not match the canonical trial START")
            if abs(yaw_error) <= .15:
                evidence.update(status="aligned", elapsed_sec=now()-started,
                                measured_command_yaw_deg=command_yaw)
                return observed
            if attempt == 3:
                raise RuntimeError("Parked Ego start yaw did not converge within three corrections")
            # First send the requested canonical heading; subsequent measured
            # errors reveal the API-to-world angular conversion without assuming
            # a fixed prefab offset. Never alter a moving or unparked car.
            if attempt:
                command_yaw -= yaw_error
            transform = grpc.make_transform(TUNNEL_NORMAL_START.xyz[0], TUNNEL_NORMAL_START.xyz[1],
                                            observed["position_xyz"][2], command_yaw)
            stub, info = _parts(grpc)
            _accepted(stub.SetTransform, SetTransformParam(actor_info=info, transform=transform))
            # MORAI resets vehicle gear when applying SetTransform. Reassert
            # input blocking immediately and again after the transform tick.
            block_and_park(grpc)
            evidence["corrections"].append({"command_yaw_deg": command_yaw, "observed_error_before_deg": yaw_error})
            sleep(.1)
            block_and_park(grpc)
            if now()-started >= 15.:
                raise RuntimeError("Parked Ego start alignment exceeded its deadline")
        raise RuntimeError("Parked Ego start alignment failed")
    except BaseException as error:
        evidence.update(status="failed", failure_reason=str(error) or type(error).__name__, elapsed_sec=now()-started)
        raise


def common_brake_hold_enabled(grpc):
    return getattr(grpc, "_tunnel_static_common_start", False) is True


def _brake_command(info):
    from proto.morai.actor.actor_control_pb2 import VehicleCtrlCmd
    from proto.morai.actor.actor_enum_pb2 import LONG_CMD_TYPE_THROTTLE
    return VehicleCtrlCmd(actor_info=info, long_cmd_type=LONG_CMD_TYPE_THROTTLE,
                          throttle=0., brake=1., steer=0., velocity=0., acceleration=0., frame=0)


_ROS_BRAKE_HOLD = None


class _RosBrakeHold:
    """Use the same ROS actuator path as the shared collector's final brake."""
    def __init__(self, *, ros=None, message_type=None):
        if ros is None:
            import rospy as ros
        if message_type is None:
            from morai_msgs.msg import CtrlCmd as message_type
        if not ros.core.is_initialized():
            ros.init_node("tunnel_brake_hold", anonymous=True, disable_signals=True)
        self.owner_pid, self.node_name, self.ros = os.getpid(), ros.get_name(), ros
        self.topic = "/ctrl_cmd"
        self.message = message_type(longlCmdType=1, accel=0., brake=1., front_steer=0.)
        self.publisher = ros.Publisher(self.topic, message_type, queue_size=1, tcp_nodelay=True)
        self.stop_event, self.error = threading.Event(), None
        self.active = True
        self.thread = threading.Thread(target=self._run, name="tunnel-ros-brake-hold", daemon=True)
        self.thread.start()

    def _run(self):
        try:
            while not self.stop_event.is_set() and not self.ros.is_shutdown():
                self.publisher.publish(self.message)
                self.stop_event.wait(.05)
        except Exception as error:
            self.error = str(error)

    def require_connection(self):
        deadline = time.monotonic()+3.
        while time.monotonic() < deadline:
            if self.error or not self.thread.is_alive():
                raise RuntimeError("ROS brake publisher stopped: " + str(self.error))
            if self.publisher.get_num_connections() > 0:
                return
            self.stop_event.wait(.02)
        raise RuntimeError("No MORAI /ctrl_cmd subscriber connected to the ROS brake publisher")

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=2.)
        if self.thread.is_alive():
            raise RuntimeError("ROS brake publisher did not stop before Expert startup")
        self.publisher.unregister()
        self.active = False


def competing_controller_publishers():
    from ..full_loop_parts.tunnel_static_parts.connection import _controller_publishers
    publishers = _controller_publishers()
    owner = _ROS_BRAKE_HOLD
    own_node = (owner.node_name if isinstance(owner, _RosBrakeHold) and owner.active
                and owner.owner_pid == os.getpid() and owner.thread.is_alive() else None)
    return [node for node in publishers if node != own_node]


def _require_no_controller():
    publishers = competing_controller_publishers()
    if publishers:
        raise RuntimeError("Brake-held Ego requires exclusive control: " + ", ".join(publishers))


def _ensure_ros_brake():
    global _ROS_BRAKE_HOLD
    _require_no_controller()
    if _ROS_BRAKE_HOLD is None or not _ROS_BRAKE_HOLD.active:
        _ROS_BRAKE_HOLD = _RosBrakeHold()
    try:
        _ROS_BRAKE_HOLD.require_connection()
    except BaseException:
        release_common_brake()
        raise
    return _ROS_BRAKE_HOLD


def release_common_brake(grpc=None):
    global _ROS_BRAKE_HOLD
    if _ROS_BRAKE_HOLD is not None:
        _ROS_BRAKE_HOLD.close()
        _ROS_BRAKE_HOLD = None


atexit.register(release_common_brake)


def block_common_input(grpc):
    """Block external commands immediately while an owned Expert is stopped."""
    from proto.morai.actor.actor_enum_pb2 import VEHICLE_CONTROL_KEYBOARD
    from proto.morai.actor.actor_set_pb2 import VehicleControlModeParam, SetVelocityParam
    stub, info = _parts(grpc)
    _accepted(stub.SetVehicleControlMode, VehicleControlModeParam(actor_info=info, mode=VEHICLE_CONTROL_KEYBOARD))
    _accepted(stub.SetVelocity, SetVelocityParam(actor_info=info, velocity=0.))
    _accepted(stub.ControlVehicle, _brake_command(info))
    mode = int(stub.GetVehicleControlMode(info, timeout=.5).mode)
    if mode != int(VEHICLE_CONTROL_KEYBOARD):
        error = RuntimeError("Emergency external-input blocking was not observed")
        error.readback_pending = True
        error.commands_accepted = True
        error.control_mode = mode
        raise error
    return {"commands_accepted": True, "external_input_disabled": True, "control_mode": mode}


def hold_common_brake(grpc):
    """Use common AUTO/D and sustained ROS pedal commands before driving."""
    from proto.morai.actor.actor_enum_pb2 import VEHICLE_CONTROL_AUTO_MODE, GEAR_MODE_D
    from proto.morai.actor.actor_set_pb2 import VehicleControlModeParam, VehicleGear
    _require_no_controller()
    stub, info = _parts(grpc)
    _accepted(stub.SetVehicleControlMode, VehicleControlModeParam(actor_info=info, mode=VEHICLE_CONTROL_AUTO_MODE))
    _ensure_ros_brake()
    _accepted(stub.SetVehicleGear, VehicleGear(actor_info=info, gear=GEAR_MODE_D))


def common_stopped_readback(grpc, reference=None):
    from proto.morai.actor.actor_enum_pb2 import VEHICLE_CONTROL_AUTO_MODE, GEAR_MODE_D
    _require_no_controller()
    stub, info = _parts(grpc)
    state = stub.GetActorState(info, timeout=.5)
    observed = live._observation(state)
    mode = int(stub.GetVehicleControlMode(info, timeout=.5).mode)
    brake, throttle = float(state.vehicle_state.brake), float(state.vehicle_state.throttle)
    observed.update(control_mode=mode, gear_mode=int(state.vehicle_state.gear_mode),
                    brake=brake if math.isfinite(brake) else None,
                    throttle=throttle if math.isfinite(throttle) else None,
                    read_monotonic=time.monotonic())
    if (mode != int(VEHICLE_CONTROL_AUTO_MODE) or int(state.vehicle_state.gear_mode) != int(GEAR_MODE_D)
            or not math.isfinite(brake) or not math.isfinite(throttle) or brake < .99 or abs(throttle) > .01):
        error = RuntimeError("Common AUTO/D full-brake hold is not observed: mode=%s gear=%s brake=%s throttle=%s speed_raw=%s" % (
            mode, observed['gear_mode'], brake, throttle, observed['speed_norm_raw']))
        error.readback = observed
        raise error
    try:
        live._stationary(observed, reference, "Brake-held Ego")
    except RuntimeError as error:
        error.readback = observed
        raise
    observed['brake_hold_verified'] = True
    return state, observed


def finish_common_brake(grpc, evidence=None, *, now=None, sleep=None):
    """Shared final-brake duration followed by bounded physical readback.

    Keep the owned ROS hold for at least the common fifteen 0.1-second sends.
    Pending pedal/gear/motion observations may settle for three more seconds;
    rejected commands, unavailable observations and foreign controllers fail.
    """
    now, sleep = now or time.monotonic, sleep or time.sleep
    record = evidence if evidence is not None else {}
    record.update(status='braking', minimum_brake_seconds=1.5,
                  readback_timeout_sec=3., checks=[], final_verified=False)
    started = now()
    try:
        hold_common_brake(grpc)
        record['native_requests_accepted'] = True
        delivered = now()
        while now()-delivered < 1.5:
            _require_no_controller()
            owner = _ROS_BRAKE_HOLD
            if isinstance(owner, _RosBrakeHold) and (owner.error or not owner.thread.is_alive()):
                raise RuntimeError('Owned ROS brake publisher stopped during final braking')
            sleep(min(.1, max(0., 1.5-(now()-delivered))))
        record['brake_duration_sec'] = now()-delivered
        deadline = now()+3.
        while now() <= deadline:
            try:
                _, observed = common_stopped_readback(grpc)
            except RuntimeError as error:
                readback = getattr(error, 'readback', None)
                if readback is None:
                    raise
                if readback.get('brake') is None or readback.get('throttle') is None:
                    raise RuntimeError('Nonfinite final actuator observation') from error
                record['checks'].append({'status':'pending', 'reason':str(error), 'readback':copy.deepcopy(readback)})
                record['last_failure'] = str(error)
                if now() >= deadline:
                    break
                sleep(min(.1, max(0., deadline-now())))
            else:
                if now() > deadline:
                    break
                record['checks'].append({'status':'verified', 'readback':copy.deepcopy(observed)})
                record.update(status='verified', final_verified=True, elapsed_sec=now()-started,
                              actual_stop=copy.deepcopy(observed))
                return observed
        raise RuntimeError('Final Ego brake/zero-speed readback did not settle within three seconds')
    except Exception as error:
        record.update(status='failed', elapsed_sec=now()-started, failure_reason=str(error))
        raise


def prepare_common_start(grpc, evidence, *, timeout_sec=5., now=None, sleep=None):
    """Observe the same body tolerances in common AUTO/D; never request P."""
    now, sleep = now or time.monotonic, sleep or time.sleep
    if not math.isfinite(timeout_sec) or not .5 < timeout_sec <= 5.:
        raise ValueError("Common startup requires a bounded five-second observation window")
    grpc._tunnel_static_common_start = True
    started = now()
    evidence.update(status="preparing", stop_method="common_auto_d_ros_brake", samples=0, observed_samples=0,
                    timeout_sec=timeout_sec, minimum_stable_sec=.5, minimum_samples=3)
    hold_common_brake(grpc)
    reference, since, count = None, None, 0
    while now()-started < timeout_sec:
        evidence['observed_samples'] += 1
        try:
            _, observed = common_stopped_readback(grpc, reference)
        except RuntimeError as error:
            reference, since, count = None, None, 0
            evidence["last_failure"] = str(error)
            if getattr(error, 'readback', None) is not None:
                evidence['last_readback'] = error.readback
        else:
            evidence["samples"] += 1
            evidence["last_readback"] = observed
            if reference is None:
                reference, since = copy.deepcopy(observed), now()
            count += 1
            if count >= 3 and now()-since >= .5 and now()-started <= timeout_sec:
                if math.dist(observed["position_xyz"][:2], TUNNEL_NORMAL_START.xyz[:2]) > .02:
                    raise RuntimeError("Measured common START XY differs from the requested route START")
                evidence.update(status="ready", stable_samples=count, stable_duration_sec=now()-since,
                                elapsed_sec=now()-started, actual_start_pose=copy.deepcopy(observed))
                return observed
        sleep(min(.1, max(0., timeout_sec-(now()-started))))
    evidence.update(status="failed", stable_samples=count, elapsed_sec=now()-started)
    raise RuntimeError("Common AUTO/D Ego did not hold a continuous stationary window within five seconds: "
                       + evidence.get("last_failure", "insufficient stable duration"))


def readiness(snapshot, diagnostics, current, *, pid, started, now):
    """Accept only fresh commands from this exact Expert and matching Ego data."""
    prefix = "/morai_standard_"+str(int(pid))+"_"
    commands = [row for row in snapshot.get("commands", [])
                if row.get("publisher_node", "").startswith(prefix)
                and started < row.get("receipt_monotonic", 0.) <= now
                and now-row["receipt_monotonic"] <= .4]
    ego = snapshot.get("ego")
    if len({row["receipt_monotonic"] for row in commands}) < 3 or not ego:
        return None
    if not started < ego.get("receipt_monotonic", 0.) <= now or now-ego["receipt_monotonic"] > .25:
        return None
    if not all(math.isfinite(value) for value in (*ego["position_xyz"], ego["yaw_deg"])):
        raise RuntimeError("Fresh Ego startup observation is invalid")
    if (math.dist(ego["position_xyz"], current["position_xyz"]) > .05
            or abs(signed_angle(ego["yaw_deg"]-current["rotation_xyz_deg"][2])) > .2):
        return None
    if (diagnostics.get("calls", 0) < 3 or not started < diagnostics.get("updated_monotonic", 0.) <= now
            or now-diagnostics["updated_monotonic"] > .4):
        return None
    for command in commands:
        values = [command[key] for key in ("accel", "brake", "front_steer", "rear_steer")]
        if (not all(math.isfinite(value) for value in values) or command["longlCmdType"] != 1
                or not 0. <= command["accel"] <= 1. or not 0. <= command["brake"] <= 1.
                or abs(command["front_steer"]) > math.radians(45.) or abs(command["rear_steer"]) > 1e-6):
            raise RuntimeError("Fresh Expert startup command is invalid")
    return {"own_command_count": len(commands), "last_command": commands[-1], "ego": ego,
            "planner_calls": diagnostics["calls"], "planner_updated_monotonic": diagnostics["updated_monotonic"]}


def wait_expert_ready(context, trial, *, started, now=None, sleep=None):
    now, sleep = now or time.monotonic, sleep or time.sleep
    process = context.expert_controller.process
    report = trial.report["startup_handoff"]
    deadline = started+12.
    reference = None
    while now() < deadline:
        if process.poll() is not None:
            raise RuntimeError("Expert exited before its parked startup handshake")
        current = parked_readback(context.grpc, reference)
        if reference is None:
            reference = copy.deepcopy(current)
        trial.inspect(context, 0.)
        try:
            diagnostics = json.loads((trial.output_dir/"planner_diagnostics.json").read_text())
        except (OSError, ValueError):
            diagnostics = {}
        if diagnostics and diagnostics.get("road_boundary_policy_sha256") != trial.boundary.policy_sha256:
            raise RuntimeError("Startup Expert boundary policy differs from the measured trial")
        ready = readiness(trial.monitor.startup_snapshot(), diagnostics, current,
                          pid=process.pid, started=started, now=now())
        report.update(last_parked_readback=current, samples=report.get("samples", 0)+1)
        if now() >= deadline:
            break
        if ready:
            report.update(status="ready", readiness=ready, ready_monotonic=now(), elapsed_sec=now()-started)
            trial.persist()
            return current
        sleep(.02)
    raise RuntimeError("No three fresh owned Expert commands with matching parked Ego within twelve seconds")


def configure_drive(context):
    """Reuse common AUTO/D Expert startup without moving the measured scene."""
    from .tunnel_static_trial import required, immediate_hold
    trial = required(context)
    if context.clock.domain == "native_frame":
        raise RuntimeError("Native collector owns the Ego startup lifecycle")
    context.stop_expert_controller()
    reference = trial.preview_report["ego_observed"]
    trial.report["startup_handoff"] = {"status": "preparing", "startup_policy": "common_auto_d_brake",
        "second_teleport_issued": False, "timeout_sec": 12., "samples": 0}
    trial.persist()
    def held_scene_check():
        hold_common_brake(context.grpc)
        _, current = common_stopped_readback(context.grpc, reference)
        trial.inspect(context, 0.)
        trial.report["startup_handoff"]["last_held_readback"] = current
        trial.persist()
    try:
        def verify_and_release():
            held_scene_check()
            release_common_brake(context.grpc)
        configured = context.activate_expert_drive(before_expert_start=held_scene_check,
                                                   after_expert_setup=verify_and_release)
        if not configured:
            raise RuntimeError("Common Expert AUTO control activation failed")
        trial.report["startup_handoff"].update(status="released", expert_pid=context.expert_controller.process.pid,
            started_monotonic=time.monotonic(), released_monotonic=time.monotonic(),
            fresh_inputs_verified=False)
        trial.persist()
        return configured
    except BaseException as error:
        immediate_hold(context)
        trial.report["startup_handoff"].update(status="failed", failure_reason=str(error) or type(error).__name__)
        trial.persist()
        raise


def observe_common_startup(context, trial, current):
    handoff = trial.report.get("startup_handoff", {})
    if handoff.get("startup_policy") != "common_auto_d_brake" or handoff.get("fresh_inputs_verified"):
        return
    if handoff.get("status") != "released":
        return
    process = context.expert_controller.process
    if process.poll() is not None:
        raise RuntimeError("Current Expert exited during common startup")
    now = time.monotonic()
    diagnostics_path = trial.output_dir/"planner_diagnostics.json"
    try:
        diagnostics = json.loads(diagnostics_path.read_text())
    except (OSError, ValueError):
        diagnostics = {}
    if diagnostics and diagnostics.get("road_boundary_policy_sha256") != trial.boundary.policy_sha256:
        raise RuntimeError("Common startup Expert policy differs from the measured scene")
    ready = readiness(trial.monitor.startup_snapshot(), diagnostics, current, pid=process.pid,
                      started=handoff["started_monotonic"], now=now)
    if ready:
        handoff.update(fresh_inputs_verified=True, readiness=ready, ready_monotonic=now)
    elif now-handoff["started_monotonic"] >= handoff["timeout_sec"]:
        raise RuntimeError("No three fresh owned Expert commands and validated plans during common startup")
