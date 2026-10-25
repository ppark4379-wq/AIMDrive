"""Offline proof of separate-process VE/Planning/Control incident evidence."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ..validation.event_monitor import EgoObservation, ValidationEvent
from ..validation.incident import IncidentRecorder
from ..validation.pipeline_diagnosis import diagnose_pipeline
from ..validation.pipeline_log import PipelineWriter, read_window
from .test_incident_recorder import RUN_ID, fake_scenario


def stages():
    common = dict(schema="aimdrive_pipeline_v1", run_id=RUN_ID,
                  clock_domain="host_monotonic_ns", timestamp_ns=30_000_000_000,
                  sequence=1, cycle_id=5, coordinate_frame="projected_grid")
    box = dict(track_id=11, center_world=[5., 0.], position_xy=[5., 0.],
               length_m=2., width_m=2., heading_deg=0., motion_state="stationary",
               coordinate_frame="projected_grid")
    perception = dict(common, stage="perception", ve_valid=True, ve_sequence=8,
                      ve_generation="gen-1",
                      ve_received_ns=29_900_000_000, objects=[box])
    planning = dict(common, stage="planning", ve_sequence=8, started_ns=29_990_000_000,
                    input_objects=[box], planned_trajectory=[[0., 3.], [10., 3.]],
                    target_velocity_mps=2.)
    control = dict(common, stage="control", planning_cycle_id=5,
                   input_path=planning["planned_trajectory"], target_velocity_mps=2.,
                   calculated_command=dict(accel=.2, brake=0., front_steer_rad=0.),
                   max_steering_rad=1.,
                   final_wire_command=dict(accel=.2, brake=0., steer_normalized=0.),
                   sent=True, sent_ns=30_010_000_000)
    return perception, planning, control


def check(rows, name):
    return next(row for row in rows if row["check_id"] == name)


class PipelineDiagnosisTests(unittest.TestCase):
    def analyze(self, perception=None, planning=None, control=None, errors=None):
        p, l, c = stages()
        return diagnose_pipeline(RUN_ID, {"captured_monotonic_sec": 31.},
                                 [perception or p] if perception is not False else None,
                                 [planning or l] if planning is not False else None,
                                 [control or c] if control is not False else None,
                                 errors or {})

    def test_normal_chain_and_no_unfounded_ego_blame(self):
        checks = self.analyze()
        for name in ("perception.delivery", "planning.path_generated", "control.handoff", "control.command"):
            self.assertEqual(check(checks, name)["status"], "OBSERVED")
        self.assertEqual(check(checks, "control.ego_response")["status"], "INSUFFICIENT_EVIDENCE")

    def test_ve_track_missing_from_planner_input(self):
        p, l, c = stages()
        l["input_objects"] = []
        row = check(self.analyze(p, l, c), "perception.delivery")
        self.assertEqual(row["status"], "SUSPECTED")
        self.assertIn("11", row["finding"])

    def test_static_footprint_on_planned_path(self):
        p, l, c = stages()
        l["planned_trajectory"] = [[0., 0.], [10., 0.]]
        c["input_path"] = l["planned_trajectory"]
        self.assertEqual(check(self.analyze(p, l, c), "planning.footprint")["status"], "SUSPECTED")
        l["input_objects"][0]["motion_state"] = "moving"
        self.assertEqual(check(self.analyze(p, l, c), "planning.footprint")["status"], "INSUFFICIENT_EVIDENCE")

    def test_stop_plan_with_wrong_control(self):
        p, l, c = stages()
        l["target_velocity_mps"] = c["target_velocity_mps"] = 0.
        self.assertEqual(check(self.analyze(p, l, c), "control.stop_target")["status"], "SUSPECTED")

    def test_ego_response_requires_explicit_ack_and_contract(self):
        p, l, c = stages()
        c["observed_ego"] = dict(applied_command_cycle_id=5, longitudinal_acceleration_mps2=-1.)
        c["observed_ego_received_ns"] = 30_020_000_000
        c["response_contract"] = dict(min_acceleration_mps2=0., max_acceleration_mps2=2.)
        self.assertEqual(check(self.analyze(p, l, c), "control.ego_response")["status"], "SUSPECTED")
        del c["observed_ego"]["applied_command_cycle_id"]
        self.assertEqual(check(self.analyze(p, l, c), "control.ego_response")["status"], "INSUFFICIENT_EVIDENCE")

    def test_clock_mismatch_and_missing_stage(self):
        p, l, c = stages()
        p["timestamp_ns"] = 32_000_000_000
        self.assertEqual(check(self.analyze(p, l, c), "pipeline.time_link")["status"], "INSUFFICIENT_EVIDENCE")
        self.assertEqual(check(self.analyze(planning=False), "perception.delivery")["status"], "INSUFFICIENT_EVIDENCE")

    def test_cross_host_window_uses_verified_anchor_and_shared_stage_clock(self):
        p, l, c = stages()
        for row in (p, l, c):
            row['clock_host'] = 'windows'
        failure = dict(captured_monotonic_sec=10., pipeline_anchor_ns=31_000_000_000,
                       pipeline_window_basis='cross-host wall window; monotonic clocks not aligned')
        rows = diagnose_pipeline(RUN_ID, failure, [p], [l], [c], {})
        self.assertEqual(check(rows, 'pipeline.time_link')['status'], 'OBSERVED')
        self.assertEqual(check(rows, 'perception.delivery')['status'], 'OBSERVED')
        c['clock_host'] = 'linux'
        rows = diagnose_pipeline(RUN_ID, failure, [p], [l], [c], {})
        self.assertEqual(check(rows, 'pipeline.time_link')['status'], 'INSUFFICIENT_EVIDENCE')


class PipelineIncidentTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_thirty_second_window_survives_collision_and_rediagnosis(self):
        journal = self.root / "pipeline.jsonl"
        p, l, c = stages()
        old = dict(p, timestamp_ns=0)
        journal.write_text("".join(json.dumps(row) + "\n" for row in (old, p, l, c)))
        (self.root / "performance.jsonl").write_text(json.dumps(dict(
            run_id=RUN_ID, generation="gen-1", clock_domain="host_monotonic_ns",
            event="sent", sent_ns=30_000_000_000, sequence=8)) + "\n")
        scenario = fake_scenario()
        scenario._validation_controller = "ve_gps_udp"
        recorder = IncidentRecorder(self.root / "incidents", scenario)
        recorder.connect_pipeline(journal)
        observation = EgoObservation(RUN_ID, 31., (1., 2., 3.), 4., ("npc-7",),
                                     yaw_deg=0., clock_domain="host_monotonic")
        event = ValidationEvent(31., "collision_started", RUN_ID, "npc-7")
        recorder.observe(0, observation, [event], ego_state={}, captured_at=31.)
        bundle = recorder.bundles[0]
        self.assertEqual(len((bundle / "perception_trace.jsonl").read_text().splitlines()), 1)
        self.assertEqual(len((bundle / "planning_trace.jsonl").read_text().splitlines()), 1)
        self.assertEqual(len((bundle / "control_trace.jsonl").read_text().splitlines()), 1)
        self.assertEqual(len((bundle / "ve_audit_trace.jsonl").read_text().splitlines()), 1)
        from ..validation.diagnose import diagnose_incident
        report = diagnose_incident(bundle)
        self.assertEqual(check(report["checks"], "perception.delivery")["status"], "OBSERVED")

    def test_corrupt_tail_and_wrong_run_are_reported(self):
        path = self.root / "pipeline.jsonl"
        p, _, _ = stages()
        path.write_text(json.dumps(p) + "\n{" )
        rows, error = read_window(path, RUN_ID, 31_000_000_000)
        self.assertEqual(len(rows), 1)
        self.assertEqual(error, "truncated final line")
        rows, error = read_window(path, "other", 31_000_000_000)
        self.assertEqual(rows, [])
        self.assertIn("identity", error)

    def test_writer_failure_never_raises_into_drive_cycle(self):
        writer = PipelineWriter(self.root / "pipeline.jsonl", RUN_ID)
        class Broken:
            def write(self, _):
                raise OSError("disk full")
        writer.stream.close()
        writer.stream = Broken()
        self.assertFalse(writer.write("control", cycle_id=1))
        self.assertEqual(writer.error, "disk full")

    def test_close_flush_failure_is_reported_without_raising(self):
        writer = PipelineWriter(self.root / "pipeline.jsonl", RUN_ID)
        class Broken:
            closed = False
            def flush(self):
                raise OSError("flush failed")
            def close(self):
                self.closed = True
        writer.stream.close()
        broken = Broken()
        writer.stream = broken
        writer.close()
        self.assertEqual(writer.error, "flush failed")
        self.assertTrue(broken.closed)
        self.assertIn("flush failed", (self.root / "pipeline_error.json").read_text())


if __name__ == "__main__":
    unittest.main()
