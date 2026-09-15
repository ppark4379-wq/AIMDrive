import math
import os
import time

from .base_scenario import BaseScenario
from ..adapters.expert_controller import ExpertController
from ..utils.transform_utils import (
    dist_xy,
    nearest_point_index,
)


class FullLoopDriveBase(BaseScenario):
    """Shared route-following lifecycle for the three K-City full-loop modes."""

    zone_name = "full_loop"
    scenario_name = "full_loop_base"

    def dump_route_debug(self):
        debug_dir = self.cfg["route_debug_dir"]
        os.makedirs(debug_dir, exist_ok=True)

        route_id = time.strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(debug_dir, f"full_loop_route_{route_id}.csv")
        png_path = os.path.join(debug_dir, f"full_loop_route_{route_id}.png")

        with open(csv_path, "w", encoding="utf-8") as output:
            output.write("idx,x,y,z\n")
            for index, point in enumerate(self.route_points):
                z = point[2] if len(point) >= 3 else 0.0
                output.write(f"{index},{point[0]},{point[1]},{z}\n")

        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            figure, axes = plt.subplots(figsize=(10, 10))
            for link_id in self.zone_allowed_links:
                points = self.map_loader.get_link_points(link_id)
                axes.plot(
                    [point[0] for point in points],
                    [point[1] for point in points],
                    color="#bbbbbb",
                    linewidth=0.7,
                    alpha=0.7,
                )

            xs = [point[0] for point in self.route_points]
            ys = [point[1] for point in self.route_points]
            axes.plot(xs, ys, color="#d62728", linewidth=2.5)
            axes.scatter([xs[0]], [ys[0]], color="#2ca02c", s=80, label="start")
            axes.scatter(
                [self.goal_x],
                [self.goal_y],
                color="#1f77b4",
                s=80,
                label="goal",
            )
            axes.set_aspect("equal", adjustable="box")
            axes.set_title(
                f"{self.start_link} -> {self.end_link}, "
                f"{len(self.route_links)} links, {self.route_length_m:.1f}m"
            )
            axes.legend()
            axes.grid(True, linewidth=0.3)
            figure.tight_layout()
            figure.savefig(png_path, dpi=150)
            plt.close(figure)
            print(f"[KCityFullLoop] route debug png: {png_path}")
        except Exception as error:
            print(f"[KCityFullLoop] route debug plot skipped: {error}")

        print(f"[KCityFullLoop] route debug csv: {csv_path}")

    def build_route_waypoint_indices(self):
        waypoint_indices = []
        for link_id in self.route_links:
            points = self.map_loader.get_link_points(link_id)
            waypoint_indices.append(len(points) - 1)

        if self.route_links:
            end_points = self.map_loader.get_link_points(self.route_links[-1])
            waypoint_indices[-1] = nearest_point_index(
                end_points,
                self.goal_x,
                self.goal_y,
            )
        return waypoint_indices

    def configure_drive_from_start(self):
        """Place Ego at START and run the external AIMDrive expert."""
        if self.clock.domain == 'native_frame':
            raise RuntimeError('Native collector owns Ego placement, gear and control lifecycle')
        self.route_configured = False
        self.route_setup_mode = None

        self.grpc.set_ego_transform(self.start_tf)
        if (
            self.cfg.get("reset_ego_motion_on_start", False)
            and hasattr(self.grpc, "reset_ego_motion")
        ):
            self.grpc.reset_ego_motion()
            time.sleep(float(self.cfg.get("reset_ego_motion_wait_sec", 0.5)))
        else:
            time.sleep(0.2)

        drive_control_mode = str(
            self.cfg.get("drive_control_mode", "aimdrive_expert")
        ).lower()
        supported_modes = {"aimdrive_expert"}
        if drive_control_mode not in supported_modes:
            raise RuntimeError(
                "The separated K-City runner supports only the external "
                f"AIMDrive expert; got drive_control_mode={drive_control_mode!r}"
            )

        self.stop_expert_controller()
        if hasattr(self.grpc, "stop_ego_cruise"):
            self.grpc.stop_ego_cruise()
        auto_ok = self.grpc.set_ego_control_mode_auto()
        if hasattr(self.grpc, "set_ego_gear_drive"):
            self.grpc.set_ego_gear_drive()

        self.setup_expert_controller()
        self.expert_controller.start_process()
        self.route_configured = auto_ok
        self.route_setup_mode = "aimdrive_expert"
        print("[KCityFullLoop] using external AIMDrive expert process")
        return auto_ok

    def configure_drive_with_retries(self):
        attempts = int(self.cfg.get("route_setup_attempts", 5))
        for attempt in range(1, attempts + 1):
            if self.configure_drive_from_start():
                if attempt > 1:
                    print(
                        "[KCityFullLoop] expert control setup recovered "
                        f"on attempt={attempt}"
                    )
                return
            print(
                f"[KCityFullLoop] expert control setup retry "
                f"{attempt}/{attempts}"
            )
            time.sleep(0.5)

        raise RuntimeError(
            f"Failed to configure expert control after {attempts} attempts"
        )

    def setup_expert_controller(self):
        paths = self.global_cfg["paths"]
        self.expert_controller = ExpertController(
            expert_root=paths["expert_root"],
            runtime_root=os.path.join(paths["runtime_root"], "expert"),
            map_name=self.cfg.get(
                "expert_map_name",
                os.path.basename(os.path.normpath(paths["mgeo_root"])),
            ),
            mgeo_root=paths["mgeo_root"],
            route_points=self.route_points,
            start_link=self.start_link,
            end_link=self.end_link,
            max_velocity_kmh=float(
                self.cfg.get("expert_max_velocity_kmh", 60.0)
            ),
            traffic_light_control=bool(
                self.cfg.get("expert_traffic_light_control", False)
            ),
            python_executable=self.cfg.get("expert_python_executable"),
            ros_remaps=self.cfg.get("expert_ros_remaps", []),
            is_closed_path=bool(
                self.cfg.get("expert_is_closed_path", False)
            ),
            velocity_profile_window_size=self.cfg.get(
                "expert_velocity_profile_window_size"
            ),
        )

    def stop_expert_controller(self):
        controller = getattr(self, "expert_controller", None)
        if controller is not None:
            controller.stop_process()
            self.expert_controller = None

    def stop_drive_control(self):
        self.stop_expert_controller()
        if hasattr(self.grpc, "stop_ego_control"):
            self.grpc.stop_ego_control()

    def project_on_route_near_progress(self, x, y, previous_s=None):
        if len(self.route_points) < 2:
            raise ValueError("Route must have at least two points")

        back_window_m = float(
            self.cfg.get("route_projection_back_window_m", 8.0)
        )
        front_window_m = float(
            self.cfg.get("route_projection_front_window_m", 40.0)
        )
        minimum_s = 0.0
        maximum_s = float("inf")
        if previous_s is not None:
            minimum_s = max(0.0, previous_s - back_window_m)
            maximum_s = min(
                self.route_length_m,
                previous_s + front_window_m,
            )

        best_s = 0.0
        best_distance = float("inf")
        cumulative = 0.0
        for point_0, point_1 in zip(
            self.route_points[:-1],
            self.route_points[1:],
        ):
            dx = point_1[0] - point_0[0]
            dy = point_1[1] - point_0[1]
            segment_length_squared = dx * dx + dy * dy
            if segment_length_squared < 1e-9:
                continue

            segment_length = math.sqrt(segment_length_squared)
            segment_start_s = cumulative
            segment_end_s = cumulative + segment_length
            cumulative = segment_end_s
            if (
                segment_end_s < minimum_s
                or segment_start_s > maximum_s
            ):
                continue

            ratio = (
                (x - point_0[0]) * dx + (y - point_0[1]) * dy
            ) / segment_length_squared
            ratio = max(0.0, min(1.0, ratio))
            projected_s = segment_start_s + ratio * segment_length
            if projected_s < minimum_s or projected_s > maximum_s:
                continue

            projected_x = point_0[0] + ratio * dx
            projected_y = point_0[1] + ratio * dy
            distance = dist_xy(x, y, projected_x, projected_y)
            if distance < best_distance:
                best_distance = distance
                best_s = projected_s

        if best_distance == float("inf"):
            return self.project_on_route_near_progress(
                x,
                y,
                previous_s=None,
            )
        return best_s, best_distance

    def run_timeline(self):
        self.run_expert_timeline()

    def run_expert_timeline(self):
        if self.clock.domain == "native_frame":
            raise RuntimeError("Native scenarios must be advanced by the capture tick owner")
        timeout_sec = self.cfg.get("timeout_sec", 0.0)
        use_timeout = timeout_sec is not None and float(timeout_sec) > 0.0
        goal_tolerance_m = float(self.cfg.get("goal_tolerance_m", 4.0))
        check_period_sec = float(self.cfg.get("check_period_sec", 0.05))
        maximum_cross_track_error_m = float(
            self.cfg.get("expert_max_cross_track_error_m", 10.0)
        )
        arrival_stop_distance_m = float(
            self.cfg.get(
                "expert_arrival_stop_distance_m",
                max(goal_tolerance_m, 6.0),
            )
        )
        off_route_timeout_sec = float(
            self.cfg.get("off_route_timeout_sec", 3.0)
        )
        off_route_grace_sec = float(
            self.cfg.get("off_route_grace_sec", 2.0)
        )
        max_laps = int(self.cfg.get("max_laps", 0))
        progress_watchdog = dict(self.cfg.get("progress_watchdog", {}))
        progress_watchdog_enabled = bool(
            progress_watchdog.get("enabled", True)
        )
        progress_watchdog_timeout_sec = max(
            1.0,
            float(progress_watchdog.get("timeout_sec", 120.0)),
        )
        progress_watchdog_grace_sec = max(
            0.0,
            float(progress_watchdog.get("startup_grace_sec", 30.0)),
        )
        progress_watchdog_min_progress_m = max(
            0.1,
            float(progress_watchdog.get("minimum_progress_m", 2.0)),
        )

        print(
            "[KCityFullLoop] AIMDrive expert running. "
            f"arrival_stop={arrival_stop_distance_m}m, "
            f"timeout={'disabled' if not use_timeout else str(timeout_sec) + 's'}, "
            f"progress_watchdog="
            f"{'disabled' if not progress_watchdog_enabled else str(progress_watchdog_timeout_sec) + 's'}"
        )

        lap = 0
        lap_start_time = self.clock.now()
        last_print_time = 0.0
        off_route_enter_time = None
        self.expert_last_s = 0.0
        progress_anchor_s = None
        last_progress_time = lap_start_time

        while True:
            elapsed = self.clock.now() - lap_start_time
            controller = getattr(self, "expert_controller", None)
            if controller is None or not controller.is_running():
                raise RuntimeError("AIMDrive expert process stopped unexpectedly")

            ego_state = self.grpc.get_ego_motion_state()
            ego_x = ego_state["x"]
            ego_y = ego_state["y"]
            distance_to_goal = dist_xy(
                ego_x,
                ego_y,
                self.goal_x,
                self.goal_y,
            )

            current_s, cross_track_error = self.project_on_route_near_progress(
                ego_x,
                ego_y,
                previous_s=getattr(self, "expert_last_s", None),
            )
            current_s = max(
                getattr(self, "expert_last_s", 0.0),
                current_s,
            )
            self.expert_last_s = current_s
            remaining_s = max(0.0, self.route_length_m - current_s)

            if progress_anchor_s is None:
                progress_anchor_s = current_s
                last_progress_time = self.clock.now()
            elif (
                current_s - progress_anchor_s
                >= progress_watchdog_min_progress_m
            ):
                progress_anchor_s = current_s
                last_progress_time = self.clock.now()

            tick_hook = getattr(self, "on_expert_timeline_tick", None)
            if tick_hook is not None:
                tick_hook(
                    elapsed=elapsed,
                    ego_state=ego_state,
                    current_s=current_s,
                    remaining_s=remaining_s,
                )

            near_route_end = remaining_s <= arrival_stop_distance_m
            route_end_reached = (
                near_route_end
                and cross_track_error <= maximum_cross_track_error_m
            )
            minimum_completion_s = self.route_length_m * float(
                self.cfg.get("minimum_completion_progress_ratio", 0.0)
            )
            progress_ready = current_s >= minimum_completion_s
            goal_reached = progress_ready and (
                route_end_reached
                or distance_to_goal <= goal_tolerance_m
            )

            if (
                elapsed >= off_route_grace_sec
                and cross_track_error > maximum_cross_track_error_m
            ):
                if off_route_enter_time is None:
                    off_route_enter_time = self.clock.now()
            else:
                off_route_enter_time = None

            if elapsed - last_print_time >= 1.0:
                print(
                    f"[KCityFullLoop] lap={lap + 1} t={elapsed:.1f}s "
                    f"ego=({ego_x:.2f}, {ego_y:.2f}) "
                    f"yaw={ego_state['yaw_deg']:.1f} "
                    f"speed={ego_state['speed']:.1f} "
                    f"link={ego_state['current_link']} "
                    f"s={current_s:.1f}/{self.route_length_m:.1f} "
                    f"cte={cross_track_error:.2f} "
                    f"remain={remaining_s:.1f} "
                    f"dist_to_goal={distance_to_goal:.2f}m"
                )
                last_print_time = elapsed

            if (
                off_route_enter_time is not None
                and self.clock.now() - off_route_enter_time
                >= off_route_timeout_sec
            ):
                self.stop_drive_control()
                raise RuntimeError(
                    "AIMDrive expert left the official route: "
                    f"cte={cross_track_error:.2f}m, "
                    f"link={ego_state['current_link']}, "
                    f"elapsed={elapsed:.1f}s"
                )

            stalled_sec = self.clock.now() - last_progress_time
            if (
                progress_watchdog_enabled
                and elapsed >= progress_watchdog_grace_sec
                and stalled_sec >= progress_watchdog_timeout_sec
                and remaining_s > arrival_stop_distance_m
            ):
                dynamic_phase = getattr(
                    self,
                    "dynamic_obstacle_phase",
                    "not_available",
                )
                tunnel_phase = getattr(
                    self,
                    "tunnel_event_phase",
                    "not_available",
                )
                reason = (
                    "reason=no_route_progress "
                    f"stalled={stalled_sec:.1f}s "
                    f"threshold={progress_watchdog_timeout_sec:.1f}s "
                    f"ego=({ego_x:.2f},{ego_y:.2f}) "
                    f"link={ego_state['current_link']} "
                    f"s={current_s:.1f}/{self.route_length_m:.1f} "
                    f"remain={remaining_s:.1f}m "
                    f"speed={ego_state['speed']:.1f} "
                    f"dynamic_phase={dynamic_phase} "
                    f"tunnel_phase={tunnel_phase}"
                )
                print(
                    f"[KCityFullLoop] FATAL_STALL {reason}",
                    flush=True,
                )
                self.stop_drive_control()
                raise RuntimeError(f"EGO progress watchdog abort: {reason}")

            if goal_reached:
                lap += 1
                print(
                    f"[KCityFullLoop] GOAL REACHED lap={lap}, "
                    f"link={ego_state['current_link']}, "
                    f"dist={distance_to_goal:.2f}m, "
                    f"s={current_s:.1f}/{self.route_length_m:.1f}, "
                    f"elapsed={elapsed:.1f}s"
                )
                self.stop_drive_control()
                if max_laps > 0 and lap >= max_laps:
                    break
                self.notify_lap_end(lap)
                self.restart_to_start_and_drive()
                lap_start_time = self.clock.now()
                last_print_time = 0.0
                off_route_enter_time = None
                self.expert_last_s = 0.0
                progress_anchor_s = None
                last_progress_time = lap_start_time
                continue

            if use_timeout and elapsed >= float(timeout_sec):
                self.stop_drive_control()
                raise TimeoutError(
                    "K-City full-loop timeout: "
                    f"dist={distance_to_goal:.2f}m, elapsed={elapsed:.1f}s"
                )

            time.sleep(check_period_sec)

    def cleanup(self):
        self.stop_expert_controller()
