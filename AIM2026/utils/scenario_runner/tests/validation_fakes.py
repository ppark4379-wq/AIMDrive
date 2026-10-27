"""Fake Ego drive used by validation tests; no MORAI process."""

from unittest.mock import Mock

from ..core.full_loop_drive_base import FullLoopDriveBase


class FakeClock:
    domain = "fake_scenario_clock"

    def __init__(self):
        self.value = 0.0

    def now(self):
        self.value += 0.1
        return self.value


class FakeDrive(FullLoopDriveBase):
    plans = []

    def __init__(self, grpc_client, map_loader, global_cfg, scenario_cfg):
        super().__init__(grpc_client, map_loader, global_cfg, scenario_cfg,
                         clock=FakeClock())
        self.plan = type(self).plans.pop(0)
        self.goal_x, self.goal_y = 100.0, 0.0
        self.route_length_m = 10.0
        self.expert_controller = Mock()
        self.expert_controller.is_running.return_value = True
        self.stop_drive_control = Mock()
        self.project_on_route_near_progress = Mock(
            side_effect=[(value, 0.0) for value in self.plan["progress"]]
        )
        self.grpc.get_ego_motion_state.side_effect = self.plan["states"]

    def run(self):
        self.run_instance_id = self.plan["run_id"]
        self.run_expert_timeline()


class FakeRecordingDrive(FakeDrive):
    def observe_collision(self, elapsed, ego_state, current_s):
        pass


def ego(collisions):
    return dict(x=0.0, y=0.0, z=0.0, yaw_deg=0.0, speed=3.6,
                speed_mps=1.0, current_link="link-1", collision_objects=collisions)
