"""Full-course competition coverage using the existing scene and Expert code."""

import copy
import json
import os
from pathlib import Path
import random
import time
from types import SimpleNamespace

from ..core.full_loop_drive_base import FullLoopDriveBase
from .full_loop_scenario import FullLoopScenario
from .full_loop_parts import competition_static, tunnel_dynamic, tunnel_dynamic_timing, tunnel_vehicle_dynamic
from .segment_modes.modes import TUNNEL_START, TUNNEL_END


TUNNEL_SCENES = ("legacy", "static", "pedestrian", "vehicle")


def _atomic_json(path, document):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        json.dump(document, output, indent=2, ensure_ascii=False, allow_nan=False)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(str(temporary), str(path))


class FullLoopCompetitionScenario(FullLoopScenario):
    """Change scene selection only; inherit the entire full-loop drive lifecycle."""

    scenario_name = "full_loop_competition"

    def __init__(self, grpc_client, map_loader, global_cfg, scenario_cfg, *, clock=None):
        cfg = copy.deepcopy(scenario_cfg)
        profile = Path(cfg.get("profile_state_path", ".runtime/full_loop_profile.json"))
        cfg["profile_state_path"] = str(profile.with_name("full_loop_competition_profile.json"))
        super().__init__(grpc_client, map_loader, global_cfg, cfg, clock=clock)
        self._competition_lock = None
        self._competition_report = None
        self._competition_error = None
        self._competition_route_finished = False
        self._competition_setup_entered = False
        self._competition_cleanup_verified = False

    def read_profile_state(self):
        """A damaged rotation record must not silently restart at the first scene."""
        path = Path(self.profile_state_path())
        if not path.exists():
            return {}
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
            competition = state["competition"]
            sequence = competition["sequence"]
            if (competition["schema"] != 1 or type(sequence) is not int or sequence < 1
                    or competition["tunnel_scene"] != TUNNEL_SCENES[(sequence - 1) % 4]
                    or state["weather"] not in ("SUNNY", "FOGGY")
                    or type(state["time_hour"]) is not int):
                raise ValueError("inconsistent competition profile")
            return state
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise RuntimeError("Invalid competition profile; rotation was not advanced: " + str(path)) from error

    def _select_scene(self):
        previous = self.read_profile_state()
        self.competition_sequence = previous.get("competition", {}).get("sequence", 0) + 1
        self.competition_tunnel_scene = TUNNEL_SCENES[(self.competition_sequence - 1) % 4]
        self._configure_scene()

    def _configure_scene(self):
        scene = self.competition_tunnel_scene
        self.cfg["competition_tunnel_scene"] = scene
        self.cfg.setdefault("tunnel_sudden_brake", {})["enabled"] = scene == "legacy"
        pedestrians = self.cfg.setdefault("tunnel_dynamic", {})
        pedestrians["enabled"] = scene == "pedestrian"
        vehicles = self.cfg.setdefault("tunnel_vehicle_dynamic", {})
        vehicles["enabled"] = scene == "vehicle"
        if scene == "pedestrian":
            # Same explicit random-mix paths as the existing tunnel trial command.
            pedestrians.pop("trial_candidate_id", None)
            pedestrians["candidate_policy"] = "trial"
            pedestrians.setdefault("random_mix", {})["enabled"] = True
        if scene == "vehicle":
            vehicles["random_behavior"] = True
        # The legacy event report must also leave ordinary full_loop history alone.
        if "verification_report_path" in self.cfg["tunnel_sudden_brake"]:
            original = Path(self.cfg["tunnel_sudden_brake"]["verification_report_path"])
            self.cfg["tunnel_sudden_brake"]["verification_report_path"] = str(
                original.with_name("competition_tunnel_sudden_brake_latest.json"))
        print("[Competition] sequence={} tunnel={}".format(self.competition_sequence, scene))

    def _acquire_lock(self):
        if self._competition_lock is not None:
            return
        # POSIX-only; imported here so the Windows Expert launcher can still import the scenario registry.
        import fcntl
        profile = Path(self.profile_state_path())
        profile.parent.mkdir(parents=True, exist_ok=True)
        self._competition_lock = profile.with_suffix(".lock").open("a+")
        try:
            fcntl.flock(self._competition_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self._competition_lock.close()
            self._competition_lock = None
            raise RuntimeError("Another full_loop_competition run owns the rotation state") from None

    def run(self):
        # Lock before even starting the inherited runtime bridge or Expert.
        self._acquire_lock()
        try:
            return super().run()
        except BaseException as error:
            self._competition_error = str(error) or type(error).__name__
            self._write_report("interrupted")
            raise
        finally:
            if self._competition_lock is not None:
                self._competition_lock.close()
                self._competition_lock = None

    def setup(self):
        self._acquire_lock()
        try:
            self._select_scene()
            self._competition_setup_entered = True
            super().setup()
            self._write_report("running")
        except BaseException as error:
            self._competition_error = str(error) or type(error).__name__
            raise

    def prepare_full_loop_route(self):
        super().prepare_full_loop_route()
        # Keep the dedicated tunnel's geographic timing window, not the full
        # course's origin/goal. This does not change Ego's route or destination.
        self.competition_tunnel_start_s = self.path_s_at_index(TUNNEL_START.index)
        end = self.path_s_at_index(TUNNEL_END.index)
        self.competition_tunnel_goal_s = self.competition_tunnel_start_s + (
            tunnel_dynamic_timing.goal_progress_s(end - self.competition_tunnel_start_s, self.cfg))
        if self.competition_tunnel_scene == "static":
            if hasattr(self, "run_instance_id"):
                self.competition_static_report_path = (
                    Path(self.profile_state_path()).parent / "competition_runs"
                    / (self.run_instance_id + "_static.json"))
            competition_static.prepare(self)

    def configure_drive_with_retries(self):
        if self.competition_tunnel_scene == "static":
            # Environment only, verified before the inherited Expert starts.
            competition_static.arm(self)
        return super().configure_drive_with_retries()

    def profile_state_metadata(self):
        return {"competition": {"schema": 1, "sequence": self.competition_sequence,
                                "tunnel_scene": self.competition_tunnel_scene}}

    def on_expert_timeline_tick(self, elapsed, ego_state, current_s, remaining_s):
        super().on_expert_timeline_tick(elapsed, ego_state, current_s, remaining_s)
        if self.competition_tunnel_scene == "static":
            competition_static.update(self, elapsed, ego_state, current_s)

    def run_timeline(self):
        try:
            result = super().run_timeline()
            self._competition_route_finished = True
            return result
        except BaseException as error:
            self._competition_error = str(error) or type(error).__name__
            raise

    def _write_report(self, status):
        if not hasattr(self, "run_instance_id"):
            return
        report = {
            "schema": "aimdrive_full_loop_competition_v1",
            "sequence": self.competition_sequence,
            "run_instance_id": self.run_instance_id,
            "run_seed": self.run_seed,
            "tunnel_scene": self.competition_tunnel_scene,
            "weather": getattr(self, "profile", {}).get("weather"),
            "time_hour": getattr(self, "scenario_hour", None),
            "roundabout_speed_kph": getattr(self, "roundabout_common_speed_kph", None),
            "tunnel_vehicle_behavior": (getattr(self, "tunnel_vehicle_plan", None) or {}).get("behavior"),
            "tunnel_pedestrian_paths": [p["path_id"] for p in
                                        (getattr(self, "tunnel_dynamic_plans", None) or [])],
            "tunnel_static_candidate": getattr(self, "competition_static_result", {}).get(
                "selected_candidate_sha256"),
            "tunnel_static_report": str(getattr(self, "competition_static_report_path", "")) or None,
            "expert_root": self.global_cfg["paths"]["expert_root"],
            "status": status, "error": self._competition_error,
            "route_finished": self._competition_route_finished,
            "cleanup_verified": self._competition_cleanup_verified,
            "updated_at_unix": time.time(),
        }
        # Runtime evidence stays inside the already ignored .runtime tree.
        path = Path(self.profile_state_path()).parent / "competition_runs" / (self.run_instance_id + ".json")
        _atomic_json(path, report)
        self._competition_report = report

    def cleanup(self):
        cleanup_error = None
        try:
            try:
                if self._competition_setup_entered:
                    super().cleanup()
                else:
                    # The inherited runtime may have started before setup failed.
                    FullLoopDriveBase.cleanup(self)
            finally:
                if getattr(self, "competition_tunnel_scene", None) == "static":
                    competition_static.cleanup(self)
            self._competition_cleanup_verified = True
        except BaseException as error:
            cleanup_error = error
            self._competition_error = str(error) or type(error).__name__
            raise
        finally:
            try:
                status = ("completed" if self._competition_route_finished and not cleanup_error
                          and not self._competition_error else "interrupted")
                self._write_report(status)
            finally:
                if self._competition_lock is not None:
                    self._competition_lock.close()
                    self._competition_lock = None


def verify_competition(global_cfg, scenario_cfg, map_loader):
    """Check all four full-course plans without advancing state or touching MORAI."""
    plans = []
    for index, scene in enumerate(TUNNEL_SCENES, 1):
        context = FullLoopCompetitionScenario(
            SimpleNamespace(make_transform=lambda *args: args), map_loader, global_cfg, scenario_cfg)
        context.competition_sequence = index
        context.competition_tunnel_scene = scene
        context._configure_scene()
        context.cfg["route_debug_enabled"] = False
        context.run_seed = int(scenario_cfg.get("random_seed") or 0)
        context.random = random.Random(context.run_seed)
        context.clean_road_mode = bool(context.cfg.get("clean_road_mode", False))
        context.route_definition = context.load_route_definition()
        context.prepare_full_loop_route()
        tunnel_dynamic.prepare_tunnel_dynamic(context)
        tunnel_vehicle_dynamic.prepare_tunnel_vehicle(context)
        plans.append({"scene": scene, "route_points": len(context.route_points),
                      "route_length_m": context.route_length_m,
                      "pedestrian_paths": [p["path_id"] for p in
                                           (getattr(context, "tunnel_dynamic_plans", None) or [])],
                      "vehicle_behavior": (getattr(context, "tunnel_vehicle_plan", None) or {}).get("behavior")})
    print("[Competition] four full-course plans verified offline (live motion not checked)")
    print(json.dumps(plans, indent=2, ensure_ascii=False))
    return plans
