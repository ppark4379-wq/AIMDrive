"""Shared brake holds retain preview immutability and bounded readbacks."""
import copy
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

from ..scenarios.full_loop_parts.tunnel_static_parts import live, safety
from ..scenarios.segment_modes import tunnel_static_startup as startup
from .test_tunnel_static_live import state


class CommonPreviewParkingTests(unittest.TestCase):
    def setUp(self):
        self.ego_state=state("Ego",0.,0.,width=2.,length=4.5)
        self.ego_state.vehicle_state=NS(gear_mode=4,brake=1.,throttle=0.)
        self.observed=live._observation(self.ego_state)
        self.observed["control_mode"]=3
        self.grpc=NS(_tunnel_static_common_start=True)
        self.ego=NS(set_velocity=Mock(),set_vehicle_gear=Mock())
        self.report={"ego_parking_commands":0}
        self.proto=patch.dict(sys.modules,{"proto.morai.actor.actor_enum_pb2":NS(GEAR_MODE_P=1,GEAR_MODE_D=4)})
        self.proto.start()
        self.addCleanup(self.proto.stop)

    def test_common_initial_hold_waits_for_readback_without_issuing_p_or_zero_velocity(self):
        with patch.object(startup,"hold_common_brake",autospec=True) as hold, \
                patch.object(startup,"common_stopped_readback",autospec=True,
                    side_effect=[RuntimeError("suspension still moving"),(self.ego_state,self.observed)]) as read, \
                patch.object(live.time,"sleep") as sleep:
            result=live._park(self.grpc,self.ego,self.report)
        self.assertEqual(result,(self.ego_state,self.observed))
        hold.assert_called_once_with(self.grpc)
        self.assertEqual(read.call_count,2)
        sleep.assert_called_once_with(.1)
        self.ego.set_velocity.assert_not_called()
        self.ego.set_vehicle_gear.assert_not_called()
        self.assertEqual(self.report["ego_parking_commands"],1)

    def test_reference_drift_is_rejected_before_any_brake_command(self):
        with patch.object(startup,"hold_common_brake",autospec=True) as hold, \
                patch.object(startup,"common_stopped_readback",autospec=True,
                    side_effect=RuntimeError("Ego pose moved")) as read:
            with self.assertRaisesRegex(RuntimeError,"pose moved"):
                live._park(self.grpc,self.ego,self.report,self.observed)
        read.assert_called_once_with(self.grpc,self.observed)
        hold.assert_not_called()
        self.assertEqual(self.report["ego_parking_commands"],0)

    def test_reference_checks_precede_hold_and_post_hold_readback_is_fail_fast(self):
        order=[]
        def read(_grpc,reference=None):
            order.append("read")
            if len(order)==3:
                raise RuntimeError("Brake hold lost")
            return self.ego_state,self.observed
        with patch.object(startup,"hold_common_brake",autospec=True,side_effect=lambda _:order.append("hold")), \
                patch.object(startup,"common_stopped_readback",autospec=True,side_effect=read), \
                patch.object(live.time,"sleep") as sleep:
            with self.assertRaisesRegex(RuntimeError,"Brake hold lost"):
                live._park(self.grpc,self.ego,self.report,self.observed)
        self.assertEqual(order,["read","hold","read"])
        sleep.assert_not_called()

    def test_initial_hold_keeps_twenty_read_limit(self):
        with patch.object(startup,"hold_common_brake",autospec=True) as hold, \
                patch.object(startup,"common_stopped_readback",autospec=True,
                    side_effect=RuntimeError("No verified stopped state")) as read, \
                patch.object(live.time,"sleep") as sleep:
            with self.assertRaisesRegex(RuntimeError,"No verified stopped state"):
                live._park(self.grpc,self.ego,self.report)
        self.assertEqual(read.call_count,20)
        self.assertEqual(sleep.call_count,19)
        hold.assert_called_once()

    def test_common_spawn_registration_verifies_hold_and_keeps_actual_ego_geometry(self):
        diagnostics={}
        inventory={"Ego":self.ego_state,"7":state("7",10.,3.5)}
        with patch.object(live,"_inventory",return_value=inventory), \
                patch.object(startup,"common_stopped_readback",autospec=True,return_value=(self.ego_state,self.observed)) as read, \
                patch.object(live,"_resolve_raw",return_value="7"):
            result=live._wait_spawn_readback(self.grpc,NS(),{"Ego"},"WoodBox",{},diagnostics,"Ego",self.observed)
        self.assertEqual(result,(inventory,"7"))
        read.assert_called_once_with(self.grpc,self.observed)
        self.assertEqual(diagnostics["status"],"verified")

    def test_spawn_inventory_cannot_hide_lost_gear_brake_throttle_or_motion(self):
        for field,value in (("gear_mode",1),("brake",.98),("throttle",.02),("throttle",-.02),("brake",float("nan"))):
            with self.subTest(field=field,value=value):
                wrong=copy.deepcopy(self.ego_state)
                setattr(wrong.vehicle_state,field,value)
                with patch.object(live,"_inventory",return_value={"Ego":wrong}), \
                        patch.object(startup,"common_stopped_readback",autospec=True,return_value=(self.ego_state,self.observed)), \
                        patch.object(live,"_resolve_raw") as resolve:
                    with self.assertRaisesRegex(RuntimeError,"brake hold"):
                        live._wait_spawn_readback(self.grpc,NS(),{"Ego"},"WoodBox",{}, {},"Ego",self.observed)
                resolve.assert_not_called()
        moving=copy.deepcopy(self.ego_state)
        moving.global_velocity.x=.051
        with patch.object(live,"_inventory",return_value={"Ego":moving}), \
                patch.object(startup,"common_stopped_readback",autospec=True) as read, \
                patch.object(live,"_resolve_raw") as resolve:
            with self.assertRaisesRegex(RuntimeError,"moving"):
                live._wait_spawn_readback(self.grpc,NS(),{"Ego"},"WoodBox",{}, {},"Ego",self.observed)
        read.assert_not_called()
        resolve.assert_not_called()


class CommonSafetyParkingTests(unittest.TestCase):
    def setUp(self):
        self.stub=NS()
        self.ego=NS()
        self.state=state("Ego",0.,0.,width=2.,length=4.5)
        self.state.vehicle_state=NS(gear_mode=4,brake=1.,throttle=0.)
        self.observed=dict(live._observation(self.state),control_mode=3)

    def test_shared_filter_excludes_only_exact_live_owned_brake_holder(self):
        from ..scenarios.full_loop_parts.tunnel_static_parts import connection
        owned="/tunnel_brake_hold_owned_pid"
        publishers=[owned,owned+"_other", "/foreign_controller"]
        holder=object.__new__(startup._RosBrakeHold)
        holder.owner_pid=startup.os.getpid()
        holder.node_name=owned
        holder.active=True
        holder.thread=NS(is_alive=lambda:True)
        with patch.object(connection,"_controller_publishers",return_value=publishers), \
                patch.object(startup,"_ROS_BRAKE_HOLD",holder):
            self.assertEqual(safety.competing_controller_publishers(),publishers[1:])
            holder.active=False
            self.assertEqual(safety.competing_controller_publishers(),publishers)
            holder.active=True
            holder.owner_pid+=1
            self.assertEqual(safety.competing_controller_publishers(),publishers)
            holder.owner_pid=startup.os.getpid()
            holder.thread=NS(is_alive=lambda:False)
            self.assertEqual(safety.competing_controller_publishers(),publishers)
        impostor=NS(owner_pid=startup.os.getpid(),node_name=owned,active=True,thread=NS(is_alive=lambda:True))
        with patch.object(connection,"_controller_publishers",return_value=publishers), \
                patch.object(startup,"_ROS_BRAKE_HOLD",impostor):
            self.assertEqual(safety.competing_controller_publishers(),publishers)

    def test_safety_adapter_verifies_common_hold_and_reports_actual_auto_mode(self):
        with patch.object(startup,"hold_common_brake",autospec=True) as hold, \
                patch.object(safety,"_controller_publishers",return_value=[]), \
                patch.object(safety,"competing_controller_publishers",return_value=[]), \
                patch.object(startup,"common_stopped_readback",autospec=True,return_value=(self.state,self.observed)) as read:
            result=safety.park(self.stub,self.ego)
        adapter=hold.call_args.args[0]
        self.assertIs(adapter.get_ego(),self.ego)
        self.assertIs(adapter.world._sim_adapter._actor_stub,self.stub)
        read.assert_called_once_with(adapter)
        self.assertTrue(result["parked"])
        self.assertTrue(result["brake_hold_verified"])
        self.assertTrue(result["control_publishers_absent"])
        self.assertFalse(result["external_input_disabled"])
        self.assertEqual((result["control_mode"],result["gear_mode"]),(3,4))
        self.assertEqual((result["brake"],result["throttle"]),(1.,0.))
        self.assertEqual(result["control_scope"],safety.CONTROL_SCOPE)

    def test_publisher_appearing_after_hold_cannot_be_reported_absent(self):
        with patch.object(startup,"hold_common_brake",autospec=True), \
                patch.object(startup,"common_stopped_readback",autospec=True,return_value=(self.state,self.observed)), \
                patch.object(safety,"_controller_publishers",return_value=["/foreign_controller"]), \
                patch.object(safety,"competing_controller_publishers",return_value=["/foreign_controller"]):
            with self.assertRaisesRegex(RuntimeError,"publisher appeared"):
                safety.park(self.stub,self.ego)

    def test_exact_owned_brake_publisher_is_disclosed_without_becoming_a_competitor(self):
        owned="/tunnel_static_brake_hold_owned_pid"
        with patch.object(startup,"hold_common_brake",autospec=True), \
                patch.object(startup,"common_stopped_readback",autospec=True,return_value=(self.state,self.observed)), \
                patch.object(safety,"_controller_publishers",return_value=[owned]), \
                patch.object(safety,"competing_controller_publishers",return_value=[]):
            result=safety.park(self.stub,self.ego)
        self.assertTrue(result["control_publishers_absent"])
        self.assertFalse(result["external_input_disabled"])
        self.assertEqual(result["observed_control_publishers"],[owned])
        self.assertEqual(result["control_scope"],safety.CONTROL_SCOPE)

    def test_rejected_or_contended_hold_cannot_become_verified_parking(self):
        with patch.object(startup,"hold_common_brake",autospec=True,side_effect=RuntimeError("Competing controller")), \
                patch.object(startup,"common_stopped_readback",autospec=True) as read:
            with self.assertRaisesRegex(RuntimeError,"Competing controller"):
                safety.park(self.stub,self.ego)
        read.assert_not_called()

    def test_unverified_safety_readback_retains_three_second_deadline(self):
        clock={"time":0.}
        def sleep(seconds):
            clock["time"]+=seconds
        with patch.object(startup,"hold_common_brake",autospec=True) as hold, \
                patch.object(startup,"common_stopped_readback",autospec=True,side_effect=RuntimeError("Brake not applied")) as read, \
                patch.object(safety.time,"monotonic",side_effect=lambda:clock["time"]), \
                patch.object(safety.time,"sleep",side_effect=sleep):
            with self.assertRaisesRegex(RuntimeError,"within three seconds"):
                safety.park(self.stub,self.ego)
        hold.assert_called_once()
        self.assertGreater(read.call_count,1)
        self.assertLessEqual(clock["time"],3.1)


if __name__=="__main__":
    unittest.main()
