"""Tunnel regression checks: startup."""

import copy, sys, threading, unittest, json, math, tempfile
from types import SimpleNamespace as NS, MethodType
from unittest.mock import Mock, patch
from ..core.full_loop_drive_base import FullLoopDriveBase
from ..scenarios.segment_modes import tunnel_static_startup as startup, tunnel_static_trial as trial_module, tunnel_static_trial as trial
from ..scenarios.segment_modes.modes import TUNNEL_NORMAL_START, TUNNEL_NORMAL_START as START
from ..scenarios.full_loop_parts.tunnel_static_parts import connection
from .test_tunnel_static_live import state, xyz
from ..tools import run_tunnel_static_trial as trial_startup
from pathlib import Path


# Common AUTO/D braking preserves measured fixtures without requiring P.

class CommonStartupClock:
    def __init__(self):self.value=0.
    def now(self):return self.value
    def sleep(self,seconds):self.value+=seconds


class CommonStartupTests(unittest.TestCase):
    def setUp(self):
        self.clock=CommonStartupClock()
        self.events=[]
        self.mode=3
        self.samples=[state('Ego',*TUNNEL_NORMAL_START.xyz[:2],width=2.16,length=4.64)]
        self.samples[0].vehicle_state.gear_mode=4
        self.samples[0].vehicle_state.brake=1.
        self.samples[0].vehicle_state.throttle=0.
        self.read_count=0
        self.commands=[]
        self.info=NS(id=NS(value='Ego'))
        self.stub=NS(GetActorState=self.read,GetVehicleControlMode=lambda info,timeout:NS(mode=self.mode),
                     SetVehicleControlMode=lambda value,timeout:self.command('mode',value),
                     SetVehicleGear=lambda value,timeout:self.command('gear',value),
                     ControlVehicle=lambda value,timeout:self.command('control',value),
                     SetVelocity=lambda value,timeout:self.command('velocity',value))
        self.grpc=NS(world=NS(_sim_adapter=NS(_actor_stub=self.stub)),
                     get_ego=lambda:NS(get_object_info=lambda:self.info))
        self.proto=patch.dict(sys.modules,{
            'proto.morai.actor.actor_enum_pb2':NS(VEHICLE_CONTROL_KEYBOARD=1,VEHICLE_CONTROL_AUTO_MODE=3,
                                                GEAR_MODE_D=4,GEAR_MODE_P=1,LONG_CMD_TYPE_SPEED=2,LONG_CMD_TYPE_THROTTLE=1),
            'proto.morai.actor.actor_set_pb2':NS(VehicleControlModeParam=NS,VehicleGear=NS,SetVelocityParam=NS),
            'proto.morai.actor.actor_control_pb2':NS(VehicleCtrlCmd=NS),
            'proto.morai.common.enum_pb2':NS(STATUS_CODE_SUCCESS=1),
        })
        self.proto.start();self.addCleanup(self.proto.stop)
        controllers=patch.object(connection,'_controller_publishers',return_value=[])
        controllers.start();self.addCleanup(controllers.stop)
        ros_hold=patch.object(startup,'_ensure_ros_brake',return_value=NS(active=True))
        self.ros_hold=ros_hold.start();self.addCleanup(ros_hold.stop)

    def command(self,name,value):
        self.commands.append((name,value))
        self.events.append(name)
        return NS(status=1)

    def read(self,info,timeout):
        self.assertIs(info,self.info)
        self.assertGreater(timeout,0.)
        self.assertLessEqual(timeout,.5)
        value=copy.deepcopy(self.samples[min(self.read_count,len(self.samples)-1)])
        self.read_count+=1
        return value

    def prepare(self,evidence=None):
        record={} if evidence is None else evidence
        result=startup.prepare_common_start(self.grpc,record,now=self.clock.now,sleep=self.clock.sleep)
        return result,record

    def test_common_hold_uses_auto_d_and_full_brake_without_p_or_pose_reset(self):
        startup.hold_common_brake(self.grpc)
        self.assertEqual([name for name,_ in self.commands],['mode','gear'])
        self.assertEqual(self.commands[0][1].mode,3)
        self.assertEqual(self.commands[1][1].gear,4)
        self.ros_hold.assert_called_once()
        self.assertNotIn('velocity',[name for name,_ in self.commands])

    def test_preparation_requires_continuous_stationarity_and_keeps_measured_heading(self):
        self.samples[0].transform.rotation.z=TUNNEL_NORMAL_START.yaw_deg+1.35
        self.samples[0].bounding_box.rotation.z=TUNNEL_NORMAL_START.yaw_deg+1.35
        observed,record=self.prepare()
        self.assertTrue(startup.common_brake_hold_enabled(self.grpc))
        self.assertEqual(record['status'],'ready')
        self.assertEqual(record['timeout_sec'],5.)
        self.assertGreaterEqual(record['stable_samples'],3)
        self.assertGreaterEqual(record['stable_duration_sec'],.5)
        self.assertEqual(observed['gear_mode'],4)
        self.assertTrue(observed['brake_hold_verified'])
        self.assertAlmostEqual(observed['rotation_xyz_deg'][2],TUNNEL_NORMAL_START.yaw_deg+1.35)

    def test_initial_motion_is_observed_until_stable_without_releasing_the_brake(self):
        moving=copy.deepcopy(self.samples[0]);moving.global_velocity=xyz(.19445,.37803,-.00306)
        self.samples=[moving]*4+[copy.deepcopy(self.samples[0])]*12
        observed,record=self.prepare()
        self.assertEqual(record['status'],'ready')
        self.assertGreaterEqual(record['elapsed_sec'],.9)
        self.assertGreaterEqual(record['stable_duration_sec'],.5)
        self.assertEqual(observed['speed_norm_raw'],0.)
        self.assertEqual(len(self.commands),2)

    def test_readback_rejects_wrong_mode_gear_brake_and_throttle(self):
        for field,value in (('mode',1),('gear_mode',1),('brake',.98),('throttle',.02),('brake',float('nan'))):
            with self.subTest(field=field):
                original=copy.deepcopy(self.samples[0]);mode=self.mode
                if field=='mode':self.mode=value
                else:setattr(self.samples[0].vehicle_state,field,value)
                with self.assertRaisesRegex(RuntimeError,'full-brake hold'):
                    startup.common_stopped_readback(self.grpc)
                self.samples[0]=original;self.mode=mode

    def test_persistent_motion_fails_at_the_existing_five_second_deadline(self):
        self.samples[0].global_velocity=xyz(0.,0.,.051)
        evidence={}
        with self.assertRaisesRegex(RuntimeError,'within five seconds'):
            self.prepare(evidence)
        self.assertEqual(evidence['status'],'failed')
        self.assertAlmostEqual(self.clock.value,5.)
        self.assertEqual(evidence['stable_samples'],0)

    def test_start_xy_mismatch_is_rejected_without_teleporting_to_hide_it(self):
        self.samples[0].transform.location.x+=.03
        self.samples[0].bounding_box.location.x+=.03
        with self.assertRaisesRegex(RuntimeError,'START XY differs'):
            self.prepare()
        self.assertEqual([name for name,_ in self.commands],['mode','gear'])

    def test_foreign_control_publisher_blocks_a_competing_brake_command(self):
        with patch.object(connection,'_controller_publishers',return_value=['/another_controller']), \
                self.assertRaisesRegex(RuntimeError,'exclusive control'):
            startup.hold_common_brake(self.grpc)
        self.assertEqual(self.commands,[])

    def test_expert_start_reuses_the_shared_auto_d_sequence_after_capture_setup(self):
        _,reference=startup.common_stopped_readback(self.grpc)
        trial=NS(preview_report={'ego_observed':reference},report={},persist=Mock(),inspect=Mock())
        self.events=[]
        context=NS(cfg={'drive_control_mode':'aimdrive_expert'},grpc=self.grpc,
                   clock=NS(domain='grpc_simulation_timestamp'),stop_expert_controller=Mock())
        self.grpc.set_ego_control_mode_auto=lambda:self.events.append('shared_auto') or True
        self.grpc.set_ego_gear_drive=lambda:self.events.append('shared_d') or True
        def setup():
            self.events.extend(['raw_ready','native_on'])
            context.expert_controller=NS(start_process=lambda:self.events.append('expert_started'),
                                         process=NS(pid=42,poll=lambda:None))
        context.setup_expert_controller=setup
        context.activate_expert_drive=MethodType(FullLoopDriveBase.activate_expert_drive,context)
        with patch.object(trial_module,'required',return_value=trial),patch.object(startup,'readiness') as readiness, \
                patch.object(startup,'release_common_brake',side_effect=lambda *a:self.events.append('brake_released')):
            self.assertTrue(startup.configure_drive(context))
        readiness.assert_not_called()
        self.assertEqual(self.events[:2],['shared_auto','shared_d'])
        self.assertLess(self.events.index('native_on'),self.events.index('expert_started'))
        self.assertLess(self.events.index('native_on'),self.events.index('brake_released'))
        self.assertLess(self.events.index('brake_released'),self.events.index('expert_started'))
        self.assertEqual(trial.inspect.call_count,2)
        self.assertEqual(trial.report['startup_handoff']['status'],'released')
        self.assertFalse(trial.report['startup_handoff']['second_teleport_issued'])


class ROSBrakeHoldTests(unittest.TestCase):
    def setUp(self):
        self.messages=[]
        self.published=threading.Event()
        self.three_messages=threading.Event()
        self.unregistered=False
        self.connections=1
        self.publish_error=False
        outer=self
        class Publisher:
            def publish(self,message):
                outer.published.set()
                if outer.publish_error:raise RuntimeError('isolated ROS send failure')
                outer.messages.append(copy.deepcopy(message))
                if len(outer.messages)>=3:outer.three_messages.set()
            def get_num_connections(self):return outer.connections
            def unregister(self):outer.unregistered=True
        self.publisher=Publisher()
        self.publisher_factory=Mock(return_value=self.publisher)
        self.ros=NS(core=NS(is_initialized=lambda:True),get_name=lambda:'/isolated_tunnel_brake',
                    is_shutdown=lambda:False,Publisher=self.publisher_factory)

    def holder(self):
        hold=startup._RosBrakeHold(ros=self.ros,message_type=NS)
        self.addCleanup(lambda:hold.close() if hold.active else None)
        return hold

    def test_sustained_ros_pedal_commands_use_the_shared_topic_and_stop_before_handoff(self):
        hold=self.holder()
        hold.require_connection()
        self.assertTrue(self.three_messages.wait(1.),'brake publication did not continue')
        self.publisher_factory.assert_called_once_with('/ctrl_cmd',NS,queue_size=1,tcp_nodelay=True)
        for message in self.messages:
            self.assertEqual(message.longlCmdType,1)
            self.assertEqual(message.accel,0.)
            self.assertEqual(message.brake,1.)
            self.assertEqual(message.front_steer,0.)
        hold.close()
        self.assertFalse(hold.thread.is_alive())
        self.assertFalse(hold.active)
        self.assertTrue(self.unregistered)

    def test_missing_morai_subscriber_fails_at_the_connection_deadline(self):
        self.connections=0
        hold=self.holder()
        with patch.object(startup.time,'monotonic',side_effect=[0.,0.,1.,2.,3.]), \
                self.assertRaisesRegex(RuntimeError,'No MORAI /ctrl_cmd subscriber'):
            hold.require_connection()

    def test_brake_worker_requests_twenty_hz_without_simulation_clock_sleep(self):
        waits=[]
        class Event:
            def __init__(self):self.stopped=False
            def is_set(self):return self.stopped
            def set(self):self.stopped=True
            def wait(self,seconds):
                waits.append(seconds)
                if len(waits)==3:self.stopped=True
        worker=NS(start=Mock(),join=Mock(),is_alive=lambda:False)
        with patch.object(startup.threading,'Event',Event),patch.object(startup.threading,'Thread',return_value=worker):
            hold=startup._RosBrakeHold(ros=self.ros,message_type=NS)
            try:
                hold._run()
                self.assertEqual(waits,[.05,.05,.05])
                self.assertEqual(len(self.messages),3)
            finally:
                hold.close()

    def test_ros_send_failure_is_reported_instead_of_claiming_a_brake_connection(self):
        self.publish_error=True
        hold=self.holder()
        self.assertTrue(self.published.wait(1.))
        hold.thread.join(timeout=1.)
        with self.assertRaisesRegex(RuntimeError,'isolated ROS send failure'):
            hold.require_connection()

    def test_only_this_process_active_brake_publisher_is_exempt_from_competition(self):
        hold=self.holder()
        with patch.object(startup,'_ROS_BRAKE_HOLD',hold), \
                patch.object(connection,'_controller_publishers',return_value=[hold.node_name,'/other_controller']):
            self.assertEqual(startup.competing_controller_publishers(),['/other_controller'])
            hold.close()
            self.assertEqual(startup.competing_controller_publishers(),[hold.node_name,'/other_controller'])

    def test_release_unregisters_the_publisher_and_clears_ownership(self):
        hold=self.holder()
        with patch.object(startup,'_ROS_BRAKE_HOLD',hold):
            startup.release_common_brake()
            self.assertIsNone(startup._ROS_BRAKE_HOLD)
        self.assertFalse(hold.thread.is_alive())
        self.assertTrue(self.unregistered)


class CommonTerminationTests(unittest.TestCase):
    def setUp(self):
        self.report={'status':'passed'}
        self.trial=NS(report=self.report,persist=Mock())
        self.grpc=NS(world=object(),_tunnel_static_common_start=True,get_ego=lambda:NS())
        self.context=NS(cfg={'mission_mode':'full_loop_tunnel_static'},grpc=self.grpc,
                        tunnel_static_trial_context=self.trial,
                        expert_controller=NS(process=NS(poll=lambda:None)),
                        stop_expert_controller=Mock(),stop_drive_control=Mock())
        self.stopped={'control_mode':3,'gear_mode':4,'speed_norm_raw':.0000167,
                      'position_xyz':[-102.9369,-550.4553,28.783],
                      'brake':1.,'throttle':0.}

    def test_normal_completed_exit_stops_the_expert_then_verifies_brake_without_keyboard_switch(self):
        with patch.object(startup,'block_common_input') as block, \
                patch.object(startup,'finish_common_brake',return_value=self.stopped) as finish:
            outcome=trial_module.immediate_hold(self.context)
        block.assert_not_called()
        self.context.stop_expert_controller.assert_called_once()
        finish.assert_called_once()
        self.assertTrue(outcome['final_verified'])
        self.assertTrue(self.report['final_stop_verified'])
        self.assertEqual(self.report['termination_hard_failures'],[])
        self.assertEqual(len(self.report['immediate_hold_history']),1)

    def test_old_transient_boolean_does_not_reject_a_new_verified_final_stop(self):
        self.report['all_immediate_hold_calls_accepted']=False
        self.report['immediate_hold_history']=[{'readback_error':'old deceleration sample','accepted_and_control_blocked':False}]
        with patch.object(startup,'finish_common_brake',return_value=self.stopped), \
                patch.object(trial_module.live,'_park'):
            trial_module.park(self.context)
        self.assertTrue(self.report['final_stop_verified'])
        self.assertTrue(self.report['ego_stopped_and_parked'])
        self.assertTrue(self.report['all_immediate_hold_calls_accepted'])
        self.assertEqual(len(self.report['immediate_hold_history']),2)

    def test_true_native_block_failure_remains_fatal_after_later_good_stop_observations(self):
        self.report['status']='running'
        with patch.object(startup,'block_common_input',side_effect=RuntimeError('native input blocking rejected')), \
                patch.object(startup,'finish_common_brake',return_value=self.stopped):
            first=trial_module.immediate_hold(self.context)
        self.assertFalse(first['final_verified'])
        self.assertTrue(self.report['termination_hard_failures'])
        self.report['status']='passed'
        with patch.object(startup,'finish_common_brake',return_value=self.stopped), \
                patch.object(trial_module.live,'_park'),self.assertRaises(RuntimeError):
            trial_module.park(self.context)
        self.assertTrue(any('native input blocking rejected' in value for value in self.report['termination_hard_failures']))
        self.assertFalse(self.report['all_immediate_hold_calls_accepted'])


# Startup settling does not weaken the parked fixture verifier.

class TrialStartupClock:
    def __init__(self):
        self.time = 0.

    def now(self):
        return self.time

    def sleep(self, seconds):
        self.time += seconds


class StartupSettleTest(unittest.TestCase):
    def setUp(self):
        self.clock = TrialStartupClock()
        self.calls = []
        self.mode_calls = []
        self.modes = [1]
        self.samples = [state("Ego", 0., 0., width=2., length=4.5)]
        self.samples[0].vehicle_state.gear_mode = 1
        self.stub = NS(GetActorState=self.read,GetVehicleControlMode=self.read_mode)
        self.grpc = NS(world=NS(_sim_adapter=NS(_actor_stub=self.stub)),
                       get_ego=lambda: NS(get_object_info=lambda: "Ego-info"))
        self.proto = patch.dict(sys.modules, {"proto.morai.actor.actor_enum_pb2": NS(
            GEAR_MODE_P=1,GEAR_MODE_D=4,VEHICLE_CONTROL_KEYBOARD=1)})
        self.proto.start()
        self.addCleanup(self.proto.stop)

    def read(self, request, timeout):
        self.assertEqual(request, "Ego-info")
        self.assertGreater(timeout, 0.)
        self.assertLessEqual(timeout, 1.)
        self.calls.append((self.clock.time, timeout))
        return copy.deepcopy(self.samples[min(len(self.calls)-1, len(self.samples)-1)])

    def settle(self, evidence=None):
        return trial_startup.settle_initial_ego(self.grpc, evidence, now=self.clock.now, sleep=self.clock.sleep)

    def read_mode(self,request,timeout):
        self.assertEqual(request,"Ego-info")
        self.assertGreater(timeout,0.)
        self.assertLessEqual(timeout,1.)
        self.mode_calls.append((self.clock.time,timeout))
        index=max(0,len(self.calls)-1)
        return NS(mode=self.modes[min(index,len(self.modes)-1)])

    def reassert_settle(self,evidence=None,timeout_sec=5.):
        return trial_startup.settle_initial_ego(self.grpc,evidence,reassert_park=True,
                                         timeout_sec=timeout_sec,now=self.clock.now,sleep=self.clock.sleep)

    def test_initial_single_zero_sample_then_vertical_bounce_resets_window_until_continuously_stable(self):
        stopped = self.samples[0]
        bouncing = copy.deepcopy(stopped)
        bouncing.global_velocity = xyz(.004, 0., .051)
        self.samples = [stopped, bouncing]+[copy.deepcopy(stopped) for _ in range(10)]
        result = self.settle()
        self.assertEqual(result["status"], "settled")
        self.assertEqual(result["window_resets"], 1)
        self.assertGreaterEqual(result["stable_duration_sec"], .5)
        self.assertGreaterEqual(result["stable_samples"], 3)
        self.assertGreaterEqual(result["elapsed_sec"], .7)
        self.assertIn("moving", result["last_failure"])
        self.assertLessEqual(result["elapsed_sec"], 5.)

    def test_collection_spawn_motion_waits_for_the_same_strict_stationary_window(self):
        moving = copy.deepcopy(self.samples[0])
        # Actual failed collection readback: the car was still settling after
        # StartSimulation. Waiting is permitted only before fixture capture.
        moving.global_velocity = xyz(.06577132642269135,.1492295116186142,-.06554288417100906)
        moving.angular_velocity = xyz(-.00634930282831192,-.000045421755203278735,.0007163670379668474)
        stopped = copy.deepcopy(moving)
        stopped.global_velocity = xyz(0.,0.,0.)
        stopped.angular_velocity = xyz(0.,0.,0.)
        self.samples = [copy.deepcopy(moving) for _ in range(4)]+[copy.deepcopy(stopped) for _ in range(10)]
        result = self.settle()
        self.assertEqual(result["status"],"settled")
        self.assertEqual(result["timeout_sec"],5.)
        self.assertEqual(result["window_resets"],4)
        self.assertGreaterEqual(result["stable_duration_sec"],.5)
        self.assertGreaterEqual(result["stable_samples"],3)
        self.assertGreaterEqual(result["elapsed_sec"],.9)
        self.assertLessEqual(result["elapsed_sec"],5.)
        self.assertEqual(result["last_observation"]["speed_norm_raw"],0.)

    def test_persistent_vertical_motion_times_out_without_any_relaxed_stationarity_claim(self):
        self.samples[0].global_velocity = xyz(0., 0., .051)
        evidence = {}
        with self.assertRaisesRegex(RuntimeError, "within five seconds"):
            self.settle(evidence)
        self.assertEqual(evidence["status"], "failed")
        self.assertEqual(evidence["stable_samples"], 0)
        self.assertAlmostEqual(self.clock.time, 5.)
        self.assertGreater(evidence["window_resets"], 3)

    def test_pose_and_bbox_orientation_changes_reset_even_if_velocity_is_zero(self):
        stopped = self.samples[0]
        shifted = copy.deepcopy(stopped)
        shifted.transform.location.z += .11
        shifted.bounding_box.location.z += .11
        shifted.bounding_box.rotation.z += .6
        self.samples = [stopped, shifted]+[copy.deepcopy(shifted) for _ in range(10)]
        result = self.settle()
        self.assertEqual(result["window_resets"], 1)
        self.assertGreaterEqual(result["stable_duration_sec"], .5)
        self.assertIn("moved", result["last_failure"])

    def test_non_p_gear_never_qualifies_as_stable(self):
        self.samples[0].vehicle_state.gear_mode = 3
        with self.assertRaisesRegex(RuntimeError, "P gear"):
            self.settle()

    def test_initial_accepted_request_but_d_readback_reasserts_before_a_stable_p_window(self):
        parked=copy.deepcopy(self.samples[0])
        drive=copy.deepcopy(parked);drive.vehicle_state.gear_mode=4
        self.samples=[drive]+[copy.deepcopy(parked) for _ in range(12)]
        with patch.object(trial_startup,"block_and_park") as parking:
            result=self.reassert_settle()
        self.assertEqual(result["status"],"settled")
        parking.assert_called_once()
        self.assertIs(parking.call_args.args[0],self.grpc)
        self.assertGreaterEqual(result["stable_samples"],3)
        self.assertGreaterEqual(result["stable_duration_sec"],.5)
        self.assertGreaterEqual(result["window_resets"],1)
        self.assertEqual(result["last_observed_gear_mode"],1)
        self.assertEqual(result["last_observed_control_mode"],1)
        self.assertEqual(result["parking_attempts"],2)
        retry=result["parking_reassertions"][0]
        self.assertEqual(retry["before"]["gear_mode"],4)
        self.assertTrue(retry["commands_accepted"])
        self.assertEqual(retry["after"]["gear_mode"],1)
        self.assertLessEqual(result["elapsed_sec"],5.)

    def test_late_d_reset_discards_the_previous_p_stable_samples(self):
        parked=copy.deepcopy(self.samples[0])
        drive=copy.deepcopy(parked);drive.vehicle_state.gear_mode=4
        self.samples=[parked,copy.deepcopy(parked),drive]+[copy.deepcopy(parked) for _ in range(12)]
        with patch.object(trial_startup,"block_and_park") as parking:
            result=self.reassert_settle()
        parking.assert_called_once()
        self.assertEqual(result["status"],"settled")
        self.assertGreaterEqual(result["elapsed_sec"],.75)
        self.assertGreaterEqual(result["stable_duration_sec"],.5)
        self.assertGreaterEqual(result["window_resets"],1)

    def test_persistent_d_reaches_the_bounded_parking_attempt_limit(self):
        self.samples[0].vehicle_state.gear_mode=4
        evidence={}
        with patch.object(trial_startup,"block_and_park") as parking, \
                self.assertRaisesRegex(RuntimeError,"parking|park|P gear"):
            self.reassert_settle(evidence)
        self.assertEqual(evidence["status"],"failed")
        self.assertEqual(parking.call_count,7)
        self.assertEqual(evidence["parking_attempts"],8)
        self.assertLessEqual(self.clock.time,5.)
        self.assertEqual(evidence["stable_samples"],0)

    def test_wrong_control_mode_reasserts_even_when_the_gear_is_already_p(self):
        self.modes=[3]+[1]*12
        with patch.object(trial_startup,"block_and_park") as parking:
            result=self.reassert_settle()
        parking.assert_called_once()
        self.assertEqual(result["status"],"settled")
        self.assertGreaterEqual(result["stable_duration_sec"],.5)

    def test_moving_p_keyboard_samples_wait_without_extra_control_commands(self):
        moving=copy.deepcopy(self.samples[0])
        moving.global_velocity=xyz(.19445394,.37803468,-.00306309)
        self.samples=[copy.deepcopy(moving) for _ in range(3)]+[copy.deepcopy(self.samples[0]) for _ in range(12)]
        with patch.object(trial_startup,"block_and_park") as parking:
            result=self.reassert_settle()
        parking.assert_not_called()
        self.assertEqual(result["status"],"settled")
        self.assertEqual(result["window_resets"],3)
        self.assertGreaterEqual(result["stable_duration_sec"],.5)

    def test_default_settle_does_not_query_control_mode_or_reassert_commands(self):
        with patch.object(trial_startup,"block_and_park") as parking:
            result=self.settle()
        self.assertEqual(result["status"],"settled")
        parking.assert_not_called()
        self.assertEqual(self.mode_calls,[])

    def test_insufficient_rpc_budget_does_not_issue_a_partial_parking_handshake(self):
        self.samples[0].vehicle_state.gear_mode=4
        evidence={}
        with patch.object(trial_startup,"block_and_park") as parking,self.assertRaises(RuntimeError):
            self.reassert_settle(evidence,timeout_sec=1.)
        parking.assert_not_called()
        self.assertEqual(evidence["status"],"failed")
        self.assertLessEqual(self.clock.time,1.)


# Fresh-command handoff and parked yaw feedback without simulator calls.

class StaticStartupClock:
    def __init__(self):
        self.value = 10.

    def now(self):
        return self.value

    def sleep(self, duration):
        self.value += duration


def observation(yaw=None):
    return {"position_xyz": [START.xyz[0], START.xyz[1], 30.],
            "rotation_xyz_deg": [0., 0., START.yaw_deg if yaw is None else yaw],
            "control_mode": 1, "gear_mode": 1}


class StartupTest(unittest.TestCase):
    def setUp(self):
        self.clock = StaticStartupClock()
        self.current = observation()
        self.pid = 12345
        self.snapshot = {"commands": [{"publisher_node": "/morai_standard_12345_100",
            "receipt_monotonic": at, "longlCmdType": 1, "accel": .1, "brake": 0.,
            "front_steer": .12, "rear_steer": 0.} for at in (9.85, 9.90, 9.95)],
            "ego": {"receipt_monotonic": 9.99, "position_xyz": self.current["position_xyz"],
                    "yaw_deg": START.yaw_deg}}
        self.diagnostics = {"calls": 3, "updated_monotonic": 9.98}
        modules = {"proto.morai.actor.actor_enum_pb2": NS(VEHICLE_CONTROL_KEYBOARD=1,
                        VEHICLE_CONTROL_AUTO_MODE=3, GEAR_MODE_P=1, GEAR_MODE_D=4),
                   "proto.morai.actor.actor_set_pb2": NS(
                        VehicleControlModeParam=lambda **args: NS(**args),
                        SetVelocityParam=lambda **args: NS(**args), VehicleGear=lambda **args: NS(**args),
                        SetTransformParam=lambda **args: NS(**args)),
                   "proto.morai.common.enum_pb2": NS(STATUS_CODE_SUCCESS=1)}
        self.proto = patch.dict(sys.modules, modules)
        self.proto.start()
        self.addCleanup(self.proto.stop)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def ready(self, snapshot=None, diagnostics=None):
        return startup.readiness(snapshot or self.snapshot, diagnostics or self.diagnostics,
            self.current, pid=self.pid, started=9.8, now=10.)

    def test_readiness_accepts_three_fresh_owned_commands_and_matching_ego(self):
        ready = self.ready()
        self.assertEqual(ready["own_command_count"], 3)
        self.assertEqual(ready["planner_calls"], 3)
        self.assertEqual(ready["last_command"]["front_steer"], .12)

    def test_readiness_excludes_other_pid_blank_and_prefix_spoofed_callers(self):
        for caller in ("/morai_standard_123456_100", "/foreign_controller", ""):
            with self.subTest(caller=caller):
                snapshot = copy.deepcopy(self.snapshot)
                for row in snapshot["commands"]:
                    row["publisher_node"] = caller
                self.assertIsNone(self.ready(snapshot))

    def test_readiness_requires_three_distinct_fresh_post_start_command_receipts(self):
        variations = [[9.85, 9.9], [9.9, 9.9, 9.9], [9.5, 9.9, 9.95], [9.8, 9.9, 9.95]]
        for receipts in variations:
            with self.subTest(receipts=receipts):
                snapshot = copy.deepcopy(self.snapshot)
                snapshot["commands"] = [dict(snapshot["commands"][0], receipt_monotonic=at) for at in receipts]
                self.assertIsNone(self.ready(snapshot))

    def test_readiness_rejects_stale_future_mismatched_or_nonfinite_ego_pose(self):
        changes = [{"receipt_monotonic": 9.7}, {"receipt_monotonic": 10.01},
                   {"position_xyz": [START.xyz[0]+.06, START.xyz[1], 30.]},
                   {"yaw_deg": START.yaw_deg+.3}]
        for change in changes:
            with self.subTest(change=change):
                snapshot = copy.deepcopy(self.snapshot)
                snapshot["ego"].update(change)
                self.assertIsNone(self.ready(snapshot))
        snapshot = copy.deepcopy(self.snapshot)
        snapshot["ego"]["yaw_deg"] = float("nan")
        with self.assertRaisesRegex(RuntimeError, "Ego startup observation is invalid"):
            self.ready(snapshot)

    def test_readiness_rejects_old_future_or_insufficient_dynamic_planner_proof(self):
        for change in ({"calls": 2}, {"updated_monotonic": 9.5}, {"updated_monotonic": 9.8},
                       {"updated_monotonic": 10.01}):
            with self.subTest(change=change):
                self.assertIsNone(self.ready(diagnostics=dict(self.diagnostics, **change)))

    def test_readiness_rejects_invalid_fresh_commands(self):
        changes = [{"longlCmdType": 2}, {"accel": 1.01}, {"brake": -.01},
                   {"front_steer": math.radians(46)}, {"rear_steer": .01}, {"accel": float("nan")}]
        for change in changes:
            with self.subTest(change=change):
                snapshot = copy.deepcopy(self.snapshot)
                snapshot["commands"][0].update(change)
                with self.assertRaisesRegex(RuntimeError, "command is invalid"):
                    self.ready(snapshot)

    def align_grpc(self, bias):
        current = observation(START.yaw_deg+bias)
        transforms = []
        def transform(request, timeout):
            self.assertEqual(timeout, .5)
            transforms.append(request.transform)
            current["rotation_xyz_deg"][2] = request.transform.rotation.z+bias
            return NS(status=1)
        stub = NS(SetTransform=transform)
        grpc = NS(get_ego=lambda: NS(get_object_info=lambda: "Ego-info"),
            world=NS(_sim_adapter=NS(_actor_stub=stub)),
            make_transform=lambda x, y, z, yaw: NS(location=NS(x=x, y=y, z=z), rotation=NS(z=yaw)))
        return grpc, current, transforms

    def test_alignment_measures_1347_and_other_biases_instead_of_encoding_constant(self):
        for bias in (1.347, 2.91, -.83):
            with self.subTest(bias=bias):
                grpc, current, transforms = self.align_grpc(bias)
                evidence = {}
                with patch.object(startup, "parked_readback", side_effect=lambda *args, **kwargs: copy.deepcopy(current)), patch.object(startup, "block_and_park"):
                    actual = startup.align_parked_ego(grpc, evidence, settle_fn=Mock(), now=self.clock.now, sleep=self.clock.sleep)
                self.assertEqual(evidence["status"], "aligned")
                self.assertAlmostEqual(actual["rotation_xyz_deg"][2], START.yaw_deg)
                self.assertAlmostEqual(transforms[-1].rotation.z, START.yaw_deg-bias)
                self.assertTrue(all(tf.location.z == 30. for tf in transforms))

    def test_already_aligned_parked_ego_is_not_transformed(self):
        grpc, current, transforms = self.align_grpc(0.)
        with patch.object(startup, "parked_readback", return_value=current):
            startup.align_parked_ego(grpc, {}, settle_fn=Mock(), now=self.clock.now, sleep=self.clock.sleep)
        self.assertEqual(transforms, [])

    def test_alignment_never_transforms_an_unparked_or_unsafe_ego(self):
        grpc, _, transforms = self.align_grpc(1.347)
        evidence = {}
        with patch.object(startup, "parked_readback", side_effect=RuntimeError("P gear is not observed")), self.assertRaisesRegex(RuntimeError, "P gear"):
            startup.align_parked_ego(grpc, evidence, settle_fn=Mock(), now=self.clock.now, sleep=self.clock.sleep)
        self.assertEqual(transforms, [])
        self.assertEqual(evidence["status"], "failed")

    def test_alignment_rejects_xy_error_without_repositioning_vehicle(self):
        grpc, current, transforms = self.align_grpc(1.347)
        current["position_xyz"][0] += .021
        with patch.object(startup, "parked_readback", return_value=current), self.assertRaisesRegex(RuntimeError, "XY does not match"):
            startup.align_parked_ego(grpc, {}, settle_fn=Mock(), now=self.clock.now, sleep=self.clock.sleep)
        self.assertEqual(transforms, [])

    def drive_context(self, *, gear_rejected=False, gear_not_observed=False):
        events = []

        def mode():
            events.append("AUTO")
            return True

        def select_gear():
            events.append("D")
            if gear_rejected:
                raise RuntimeError("D gear command rejected")
            return True

        grpc = NS(set_ego_control_mode_auto=mode, set_ego_gear_drive=select_gear,
                  set_ego_transform=Mock(), make_transform=Mock())
        controller = NS(process=NS(pid=self.pid, poll=lambda: None),
                        start_process=lambda: events.append("Expert-start"))
        context = NS(grpc=grpc, cfg={}, clock=NS(domain="host_monotonic"),
                     route_configured=False, route_setup_mode=None,
                     expert_controller=controller,
                     stop_expert_controller=lambda: events.append("stop-previous"),
                     setup_expert_controller=lambda: events.append("setup-Expert"))
        context.activate_expert_drive = lambda **hooks: FullLoopDriveBase.activate_expert_drive(context, **hooks)
        context._test_missing_d = gear_not_observed
        proof = NS(report={}, preview_report={"ego_observed": copy.deepcopy(self.current)},
                   persist=Mock(), inspect=lambda *args: events.append("guard"))
        return context, proof, events

    def run_configure(self, context, proof, events, *, held_error=None, release_error=None):
        reads = []

        def stopped(grpc, reference):
            events.append("held-readback")
            reads.append(reference)
            self.assertEqual(reference, proof.preview_report["ego_observed"])
            if context._test_missing_d:
                raise RuntimeError("Common AUTO/D full-brake hold is not observed")
            if held_error and len(reads) == 2:
                raise held_error
            return NS(), copy.deepcopy(self.current)

        def released(grpc):
            self.assertIs(grpc, context.grpc)
            events.append("release-brake")
            if release_error:
                raise release_error

        with patch.object(trial, "required", return_value=proof), \
                patch.object(trial, "immediate_hold", side_effect=lambda *args: events.append("immediate-hold")), \
                patch.object(startup, "hold_common_brake", side_effect=lambda *args: events.append("full-brake")), \
                patch.object(startup, "common_stopped_readback", side_effect=stopped), \
                patch.object(startup, "release_common_brake",
                             side_effect=released) as release, \
                patch.object(startup, "block_and_park") as park, \
                patch.object(startup, "parked_readback") as parked, \
                patch.object(startup, "wait_expert_ready") as wait:
            result = startup.configure_drive(context)
            park.assert_not_called()
            parked.assert_not_called()
            wait.assert_not_called()
            release.assert_called_once_with(context.grpc)
            return result

    def test_drive_handoff_reuses_auto_d_before_setup_and_verifies_both_held_gates(self):
        context, proof, events = self.drive_context()
        self.assertTrue(self.run_configure(context, proof, events))
        self.assertEqual(events, ["stop-previous", "AUTO", "D", "full-brake", "held-readback", "guard",
                                  "setup-Expert", "full-brake", "held-readback", "guard", "release-brake", "Expert-start"])
        context.grpc.set_ego_transform.assert_not_called()
        context.grpc.make_transform.assert_not_called()
        self.assertTrue(context.route_configured)
        handoff = proof.report["startup_handoff"]
        self.assertFalse(handoff["second_teleport_issued"])
        self.assertEqual(handoff["startup_policy"], "common_auto_d_brake")
        self.assertFalse(handoff["fresh_inputs_verified"])
        self.assertEqual(handoff["status"], "released")

    def test_post_setup_pose_failure_holds_immediately_before_launching_expert(self):
        context, proof, events = self.drive_context()
        with self.assertRaisesRegex(RuntimeError, "measured pose drift"):
            self.run_configure(context, proof, events, held_error=RuntimeError("measured pose drift"))
        self.assertEqual(events[-1], "immediate-hold")
        self.assertIn("AUTO", events)
        self.assertIn("D", events)
        self.assertIn("setup-Expert", events)
        self.assertNotIn("Expert-start", events)
        self.assertNotIn("release-brake", events)
        self.assertFalse(context.route_configured)
        self.assertEqual(proof.report["startup_handoff"]["status"], "failed")

    def test_d_gear_rejection_or_missing_held_readback_prevents_expert_start(self):
        for kwargs in ({"gear_rejected": True}, {"gear_not_observed": True}):
            with self.subTest(kwargs=kwargs):
                context, proof, events = self.drive_context(**kwargs)
                with self.assertRaises(RuntimeError):
                    self.run_configure(context, proof, events)
                self.assertEqual(events[-1], "immediate-hold")
                self.assertNotIn("Expert-start", events)
                self.assertNotIn("release-brake", events)
                self.assertFalse(context.route_configured)
                self.assertEqual(proof.report["startup_handoff"]["status"], "failed")

    def test_brake_publisher_release_failure_prevents_expert_start(self):
        context, proof, events = self.drive_context()
        with self.assertRaisesRegex(RuntimeError, "brake owner still active"):
            self.run_configure(context, proof, events,
                               release_error=RuntimeError("brake owner still active"))
        self.assertEqual(events[-1], "immediate-hold")
        self.assertIn("release-brake", events)
        self.assertNotIn("Expert-start", events)
        self.assertFalse(context.route_configured)
        self.assertEqual(proof.report["startup_handoff"]["status"], "failed")

    def test_wait_ready_reads_live_diagnostics_and_persists_owned_ready_proof(self):
        output = Path(self.tmp.name)
        (output/"planner_diagnostics.json").write_text(json.dumps(dict(self.diagnostics, road_boundary_policy_sha256="policy")))
        context = NS(grpc=NS(), expert_controller=NS(process=NS(pid=self.pid, poll=lambda: None)))
        proof = NS(report={"startup_handoff": {}}, output_dir=output, boundary=NS(policy_sha256="policy"),
                   monitor=NS(startup_snapshot=lambda: copy.deepcopy(self.snapshot)), inspect=Mock(), persist=Mock())
        with patch.object(startup, "parked_readback", return_value=self.current):
            startup.wait_expert_ready(context, proof, started=9.8, now=self.clock.now, sleep=self.clock.sleep)
        self.assertEqual(proof.report["startup_handoff"]["status"], "ready")
        self.assertEqual(proof.report["startup_handoff"]["readiness"]["own_command_count"], 3)
        proof.inspect.assert_called_once_with(context, 0.)


if __name__ == "__main__":
    unittest.main()
