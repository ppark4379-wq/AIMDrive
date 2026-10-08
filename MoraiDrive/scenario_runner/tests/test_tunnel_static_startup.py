"""Fresh-command handoff and parked yaw feedback without simulator calls."""

import copy
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

from ..core.full_loop_drive_base import FullLoopDriveBase
from ..scenarios.segment_modes import tunnel_static_startup as startup
from ..scenarios.segment_modes import tunnel_static_trial as trial
from ..scenarios.segment_modes.modes import TUNNEL_NORMAL_START as START


class Clock:
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
        self.clock = Clock()
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
