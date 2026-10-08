"""Common AUTO/D braking preserves measured fixtures without requiring P."""

import copy
import sys
import threading
from types import SimpleNamespace as NS, MethodType
import unittest
from unittest.mock import Mock, patch

from ..core.full_loop_drive_base import FullLoopDriveBase
from ..scenarios.segment_modes import tunnel_static_startup as startup, tunnel_static_trial as trial_module
from ..scenarios.segment_modes.modes import TUNNEL_NORMAL_START
from ..scenarios.full_loop_parts.tunnel_static_parts import connection
from .test_tunnel_static_live import state, xyz


class Clock:
    def __init__(self):self.value=0.
    def now(self):return self.value
    def sleep(self,seconds):self.value+=seconds


class CommonStartupTests(unittest.TestCase):
    def setUp(self):
        self.clock=Clock()
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


if __name__=='__main__':unittest.main()
