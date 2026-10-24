"""Tunnel speed comes from the unchanged Expert source, with no fixed cap."""

import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from contextlib import redirect_stdout
from contextlib import ExitStack
from unittest.mock import Mock, patch

from ..adapters.expert_controller import (
    ExpertController, expert_max_velocity_contract, resolve_expert_max_velocity,
    validate_max_velocity_kmh,
)
from ..core.full_loop_drive_base import FullLoopDriveBase
from ..scenarios.segment_modes import events
from ..scenarios.full_loop_parts.tunnel_static_parts import trial as trial_cli
from ..tools.tunnel_static import batch


class TunnelSpeedInheritanceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.expert = self.root/"fake_expert"
        self.config_path = self.expert/"src/config/config.json"
        self.config_path.parent.mkdir(parents=True)
        self.config = {"map":{"name":"original", "data_dir":"original-map", "use_mgeo_path":True,
                              "traffic_light_control":True,"is_closed_path":True},
                       "planning":{"velocity_profile":{"max_velocity":37,"window_size":50,"road_friction":.15,
                                                           "terminal_deceleration":2.,"speed_limit_zones":[]}},
                       "control":{"controller_gain":123}}
        self.write_config()

    def write_config(self):
        self.config_path.write_text(json.dumps(self.config))

    def controller(self, speed=None, **kwargs):
        with redirect_stdout(io.StringIO()):
            return ExpertController(str(self.expert),str(self.root/"runtime"),"test-map",str(self.root/"map"),
                                    [[0.,0.],[1.,0.]],max_velocity_kmh=speed,**kwargs)

    def test_none_inherits_non_60_original_and_preserves_the_original_field_and_file(self):
        before = self.config_path.read_bytes()
        controller = self.controller()
        runtime = json.loads(Path(controller.runtime_config_path).read_text())
        self.assertEqual(runtime["planning"],self.config["planning"])
        self.assertIs(type(runtime["planning"]["velocity_profile"]["max_velocity"]),int)
        self.assertEqual(controller.max_velocity_kmh,37.)
        self.assertTrue(controller.speed_contract["inherited"])
        self.assertEqual(controller.speed_contract["expert_config_sha256"],hashlib.sha256(before).hexdigest())
        self.assertEqual(runtime["control"],self.config["control"])
        self.assertEqual(self.config_path.read_bytes(),before)

    def test_explicit_other_zone_override_keeps_existing_export_semantics(self):
        controller = self.controller(60.,velocity_profile_window_size=15)
        runtime = json.loads(Path(controller.runtime_config_path).read_text())
        self.assertEqual(runtime["planning"]["velocity_profile"]["max_velocity"],60.)
        self.assertEqual(runtime["planning"]["velocity_profile"]["window_size"],15)
        self.assertEqual(runtime["planning"]["velocity_profile"]["road_friction"],.15)
        self.assertEqual(runtime["control"],self.config["control"])
        self.assertFalse(controller.speed_contract["inherited"])
        self.assertEqual(json.loads(self.config_path.read_text()),self.config)

    def test_each_invocation_reads_current_original_and_pins_its_source(self):
        first = expert_max_velocity_contract(self.expert)
        self.config["planning"]["velocity_profile"]["max_velocity"] = 43
        self.write_config()
        second = expert_max_velocity_contract(self.expert)
        self.assertEqual(first["resolved_max_velocity_kmh"],37.)
        self.assertEqual(second["resolved_max_velocity_kmh"],43.)
        self.assertNotEqual(first["expert_config_sha256"],second["expert_config_sha256"])
        self.assertEqual(resolve_expert_max_velocity(self.expert),43.)

    def test_inherited_and_explicit_values_require_positive_finite_numbers(self):
        for value in (0,-1,float("nan"),float("inf"),True,"bad"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):validate_max_velocity_kmh(value)
                self.config["planning"]["velocity_profile"]["max_velocity"] = value
                self.write_config()
                with self.assertRaises(ValueError):self.controller()
        self.assertEqual(validate_max_velocity_kmh(37),37.)
        self.assertEqual(validate_max_velocity_kmh(100),100.)

    def context(self,mode,speed="missing"):
        cfg={"mission_mode":mode,"profile_state_path":str(self.root/"profile.json")}
        if speed != "missing":cfg["expert_max_velocity_kmh"]=speed
        return SimpleNamespace(cfg=cfg,global_cfg={"paths":{"expert_root":str(self.expert),"runtime_root":str(self.root),
                                                           "mgeo_root":str(self.root/"map")}},
                               route_points=[[0.,0.],[1.,0.]],start_link="start",end_link="end")

    def test_core_passes_none_for_tunnel_and_keeps_other_zone_explicit_values(self):
        for mode,speed,expected in (("full_loop_tunnel_static",None,None),
                                   ("full_loop_tunnel_static","missing",None),
                                   ("full_loop_static",60,60.),("full_loop_dynamic",37,37.),
                                   ("full_loop_roundabout","missing",60.)):
            with self.subTest(mode=mode,speed=speed),patch("scenario_runner.core.full_loop_drive_base.ExpertController") as constructor:
                context=self.context(mode,speed)
                FullLoopDriveBase.setup_expert_controller(context)
                self.assertEqual(constructor.call_args.kwargs["max_velocity_kmh"],expected)

    def test_tunnel_events_keep_all_guard_settings_without_clamping_speed(self):
        context=self.context("full_loop_tunnel_static",None)
        with redirect_stdout(io.StringIO()):events.prepare_mission(context)
        self.assertIsNone(context.cfg["expert_max_velocity_kmh"])
        self.assertEqual(context.cfg["timeout_sec"],180.)
        self.assertEqual(context.cfg["expert_max_cross_track_error_m"],5.)
        self.assertEqual(context.cfg["off_route_timeout_sec"],3.)
        self.assertEqual(context.cfg["progress_watchdog"],{"enabled":True,"timeout_sec":45.,
                                                         "startup_grace_sec":30.,"minimum_progress_m":2.})
        context=self.context("full_loop_tunnel_static",37)
        with redirect_stdout(io.StringIO()):events.prepare_mission(context)
        self.assertEqual(context.cfg["expert_max_velocity_kmh"],37)

    def test_normal_events_do_not_change_normal_speed(self):
        for name in ("full_loop_static","full_loop_dynamic","full_loop_roundabout"):
            context=self.context(name,60)
            before=copy.deepcopy(context.cfg)
            with redirect_stdout(io.StringIO()):events.prepare_mission(context)
            self.assertEqual(context.cfg["expert_max_velocity_kmh"],before["expert_max_velocity_kmh"])

    def trial_args(self,extra=()):
        candidate=self.root/"candidate.json"
        candidate.write_text(json.dumps({"count":1,"obstacles":[{"actor_kind":"obstacle","model":"Box"}]}))
        return trial_cli.build_parser().parse_args(["--candidate",str(candidate),"--output",str(self.root/"trial"),*extra])

    def test_trial_cli_defaults_to_inheritance_and_accepts_original_speed_above_15(self):
        args=self.trial_args()
        self.assertIsNone(args.expert_max_velocity_kmh)
        trial_cli.validate_trial_settings(args)
        args=self.trial_args(("--expert-max-velocity-kmh","37"))
        trial_cli.validate_trial_settings(args)
        args.expert_max_velocity_kmh=float("nan")
        with self.assertRaises(ValueError):trial_cli.validate_trial_settings(args)

    def run_mock_trial(self,args,preview_effect=None):
        grpc=Mock()
        grpc.world=object()
        grpc.get_available_obstacle_models.return_value=["Box"]
        grpc.get_available_surround_vehicle_models.return_value=[]
        callback=Mock(return_value=lambda *a:None)
        configuration={"paths":{"expert_root":str(self.expert),"mgeo_root":str(self.root/"map"),
                                "runtime_root":str(self.root)},"grpc":{}}
        base={"expert_max_velocity_kmh":60.,"profile_state_path":"legacy.json","random_seed":99}
        with ExitStack() as stack:
            stack.enter_context(patch.object(trial_cli,"load_configuration",return_value=(configuration,{},base)))
            stack.enter_context(patch.object(trial_cli,"MGeoMapLoader"))
            stack.enter_context(patch.object(trial_cli,"validate_candidate"))
            stack.enter_context(patch.object(trial_cli,"_controller_publishers",return_value=[]))
            stack.enter_context(patch("scenario_runner.adapters.morai_grpc.MoraiGrpcClient",return_value=grpc))
            for name in ("block_and_park", "park_ego"):
                stack.enter_context(patch.object(trial_cli, name))
            stack.enter_context(patch.object(trial_cli, "prepare_common_start", autospec=True))
            stack.enter_context(patch.object(trial_cli,"make_drive_callback",callback))
            stack.enter_context(patch.object(trial_cli,"preview_candidate",side_effect=preview_effect,
                return_value={"status":"valid_driving_trial","expert_verified":True,"live_avoidance_verified":True}))
            report=trial_cli.run(args)
        return report,callback.call_args.args[3]

    def test_cli_delivers_inheritance_seed_and_persistent_profile_to_callback(self):
        profile=self.root/"shared/full_loop_tunnel_static_profile.json"
        args=self.trial_args(("--run-seed","731","--profile-state-path",str(profile),"--console-output"))
        report,cfg=self.run_mock_trial(args)
        self.assertIsNone(cfg["expert_max_velocity_kmh"])
        self.assertEqual(cfg["random_seed"],731)
        self.assertEqual(cfg["profile_state_path"],str(profile.resolve()))
        self.assertTrue(cfg["tunnel_static_profile_state_path_explicit"])
        self.assertTrue(cfg["tunnel_static_console_output"])
        self.assertEqual(report["expert_speed_source"]["resolved_max_velocity_kmh"],37.)
        self.assertTrue(report["expert_speed_source"]["inherited"])

    def test_legacy_cli_keeps_seed_and_history_untouched_when_not_supplied(self):
        _,cfg=self.run_mock_trial(self.trial_args())
        self.assertEqual(cfg["random_seed"],99)
        self.assertEqual(cfg["profile_state_path"],"legacy.json")
        self.assertNotIn("tunnel_static_profile_state_path_explicit",cfg)

    def test_changed_original_config_during_trial_cannot_be_reported_passed(self):
        def changed(*args,**kwargs):
            self.config["planning"]["velocity_profile"]["max_velocity"]=43
            self.write_config()
            return {"status":"valid_driving_trial","expert_verified":True,"live_avoidance_verified":True}
        with self.assertRaisesRegex(RuntimeError,"speed source changed"):
            self.run_mock_trial(self.trial_args(),changed)
        saved=json.loads((self.root/"trial/manifest.json").read_text())
        self.assertEqual(saved["status"],"failed")

    def test_grouped_batch_omits_inherited_speed_and_preserves_explicit_source_value(self):
        args=SimpleNamespace(max_velocity=None,hold_seconds=5.,time_budget=15.)
        context=SimpleNamespace(args=args)
        command=batch.Batch.trial_command(context,Path("candidate.json"),Path("trial"))
        self.assertNotIn("--expert-max-velocity-kmh",command)
        args.max_velocity=37.
        command=batch.Batch.trial_command(context,Path("candidate.json"),Path("trial"))
        self.assertEqual(command[command.index("--expert-max-velocity-kmh")+1],"37.0")
        self.assertIsNone(batch.build_parser().get_default("max_velocity"))


if __name__ == "__main__":
    unittest.main()
