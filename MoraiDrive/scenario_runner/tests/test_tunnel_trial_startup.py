"""Startup settling does not weaken the parked fixture verifier."""

import copy
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from ..tools import run_tunnel_static_trial as startup
from .test_tunnel_static_live import state, xyz


class Clock:
    def __init__(self):
        self.time = 0.

    def now(self):
        return self.time

    def sleep(self, seconds):
        self.time += seconds


class StartupSettleTest(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
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
        return startup.settle_initial_ego(self.grpc, evidence, now=self.clock.now, sleep=self.clock.sleep)

    def read_mode(self,request,timeout):
        self.assertEqual(request,"Ego-info")
        self.assertGreater(timeout,0.)
        self.assertLessEqual(timeout,1.)
        self.mode_calls.append((self.clock.time,timeout))
        index=max(0,len(self.calls)-1)
        return NS(mode=self.modes[min(index,len(self.modes)-1)])

    def reassert_settle(self,evidence=None,timeout_sec=5.):
        return startup.settle_initial_ego(self.grpc,evidence,reassert_park=True,
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
        with patch.object(startup,"block_and_park") as parking:
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
        with patch.object(startup,"block_and_park") as parking:
            result=self.reassert_settle()
        parking.assert_called_once()
        self.assertEqual(result["status"],"settled")
        self.assertGreaterEqual(result["elapsed_sec"],.75)
        self.assertGreaterEqual(result["stable_duration_sec"],.5)
        self.assertGreaterEqual(result["window_resets"],1)

    def test_persistent_d_reaches_the_bounded_parking_attempt_limit(self):
        self.samples[0].vehicle_state.gear_mode=4
        evidence={}
        with patch.object(startup,"block_and_park") as parking, \
                self.assertRaisesRegex(RuntimeError,"parking|park|P gear"):
            self.reassert_settle(evidence)
        self.assertEqual(evidence["status"],"failed")
        self.assertEqual(parking.call_count,7)
        self.assertEqual(evidence["parking_attempts"],8)
        self.assertLessEqual(self.clock.time,5.)
        self.assertEqual(evidence["stable_samples"],0)

    def test_wrong_control_mode_reasserts_even_when_the_gear_is_already_p(self):
        self.modes=[3]+[1]*12
        with patch.object(startup,"block_and_park") as parking:
            result=self.reassert_settle()
        parking.assert_called_once()
        self.assertEqual(result["status"],"settled")
        self.assertGreaterEqual(result["stable_duration_sec"],.5)

    def test_moving_p_keyboard_samples_wait_without_extra_control_commands(self):
        moving=copy.deepcopy(self.samples[0])
        moving.global_velocity=xyz(.19445394,.37803468,-.00306309)
        self.samples=[copy.deepcopy(moving) for _ in range(3)]+[copy.deepcopy(self.samples[0]) for _ in range(12)]
        with patch.object(startup,"block_and_park") as parking:
            result=self.reassert_settle()
        parking.assert_not_called()
        self.assertEqual(result["status"],"settled")
        self.assertEqual(result["window_resets"],3)
        self.assertGreaterEqual(result["stable_duration_sec"],.5)

    def test_default_settle_does_not_query_control_mode_or_reassert_commands(self):
        with patch.object(startup,"block_and_park") as parking:
            result=self.settle()
        self.assertEqual(result["status"],"settled")
        parking.assert_not_called()
        self.assertEqual(self.mode_calls,[])

    def test_insufficient_rpc_budget_does_not_issue_a_partial_parking_handshake(self):
        self.samples[0].vehicle_state.gear_mode=4
        evidence={}
        with patch.object(startup,"block_and_park") as parking,self.assertRaises(RuntimeError):
            self.reassert_settle(evidence,timeout_sec=1.)
        parking.assert_not_called()
        self.assertEqual(evidence["status"],"failed")
        self.assertLessEqual(self.clock.time,1.)


if __name__ == "__main__":
    unittest.main()
