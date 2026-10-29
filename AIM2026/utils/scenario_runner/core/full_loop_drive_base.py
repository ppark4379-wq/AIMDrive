import json
import math
import os
import sys
import threading
import time
from ..validation.incident import snapshot_nearby_actors

from .base_scenario import BaseScenario
from ..adapters.expert_controller import ExpertController
from ..adapters.sync_runtime import SyncRuntimeBridge
from ..utils.transform_utils import (
    dist_xy,
    nearest_point_index,
)


class ScenarioCollisionError(RuntimeError):
    """Verification ended because an actual monitored contact was reported."""


class FullLoopDriveBase(BaseScenario):
    """Shared route-following lifecycle for the three K-City full-loop modes."""

    zone_name = "full_loop"
    scenario_name = "full_loop_base"

    def start_static_editor_session(self):
        """Attach the long-lived editor trial handshake to ROS."""
        import rospy
        from std_msgs.msg import Bool, String

        if not rospy.core.is_initialized():
            rospy.init_node(
                "static_scenario_runner", anonymous=True, disable_signals=True,
            )
        self._static_editor_lock = threading.RLock()
        self._static_editor_drive_enabled = False
        self._static_editor_event_type = String
        self._static_editor_event_publisher = rospy.Publisher(
            "/static_scenario_editor/trial_event",
            String,
            queue_size=10,
            latch=True,
        )

        def on_drive_enabled(message):
            with self._static_editor_lock:
                self._static_editor_drive_enabled = bool(message.data)

        self._static_editor_drive_subscriber = rospy.Subscriber(
            "/static_scenario_editor/drive_enabled",
            Bool,
            on_drive_enabled,
            queue_size=1,
        )

    def close_static_editor_session(self):
        subscriber = getattr(self, "_static_editor_drive_subscriber", None)
        if subscriber is not None:
            subscriber.unregister()
            self._static_editor_drive_subscriber = None

    def static_editor_drive_enabled(self):
        lock = getattr(self, "_static_editor_lock", None)
        if lock is None:
            # Direct unit tests do not start the ROS bridge.
            return bool(getattr(self, "_static_editor_drive_enabled", True))
        with lock:
            return bool(self._static_editor_drive_enabled)

    def publish_static_editor_event(self, event, **payload):
        document = {
            "event": str(event),
            "timestamp": time.time(),
        }
        document.update(payload)
        publisher = getattr(self, "_static_editor_event_publisher", None)
        message_type = getattr(self, "_static_editor_event_type", None)
        if publisher is not None and message_type is not None:
            publisher.publish(message_type(data=json.dumps(document)))
        print("[STATIC EDITOR] " + json.dumps(document, ensure_ascii=False))

    def finish_static_editor_trial(
        self, trial, outcome, reason, collisions, reset_for_next=True,
    ):
        """Brake, report one trial, then rebuild only Ego/Expert state."""
        if hasattr(self.grpc, "stop_ego_control"):
            self.grpc.stop_ego_control()
        self.publish_static_editor_event(
            "trial_finished",
            trial=int(trial),
            outcome=str(outcome),
            reason=str(reason),
            collisions=sorted(set(str(value) for value in collisions)),
        )

        if not reset_for_next:
            return

        # The editor receives trial_finished, closes its command relay and
        # publishes drive_enabled=False. Never teleport while commands are
        # still being forwarded to MORAI.
        deadline = time.monotonic() + 3.0
        while self.static_editor_drive_enabled() and time.monotonic() < deadline:
            time.sleep(0.02)
        if self.static_editor_drive_enabled():
            raise RuntimeError(
                "Static editor did not close the Expert command relay after "
                "trial completion"
            )

        self.publish_static_editor_event(
            "trial_resetting", trial=int(trial), outcome=str(outcome),
        )
        self.configure_drive_with_retries()
        self.publish_static_editor_event(
            "trial_ready", trial=int(trial), previous_outcome=str(outcome),
        )

    def require_async_ros_control_path(self):
        """Fail before actor setup when Expert commands cannot reach MORAI."""
        import rosgraph

        master = rosgraph.Master("/scenario_runner_async_preflight")
        try:
            _, subscribers, _ = master.getSystemState()
        except Exception as error:
            raise RuntimeError(
                "Asynchronous scenario mode requires a running ROS master "
                "and MORAI bridge. Start tools/preflight_ros_control.sh "
                "in another terminal; Sensor Record and Synchronization Mode "
                "may remain OFF."
            ) from error

        control_topic = str(
            self.cfg.get("async_control_topic", "/ctrl_cmd")
        )
        control_subscribers = next(
            (nodes for topic, nodes in subscribers if topic == control_topic),
            [],
        )
        if not control_subscribers:
            raise RuntimeError(
                "MORAI bridge is not subscribed to {}. Start the 01 bridge "
                "and connect MORAI vehicle information to its address before "
                "running the scenario.".format(control_topic)
            )
        print(
            "[KCityRunner] asynchronous ROS control ready: "
            "{} subscribers={}".format(
                control_topic, list(control_subscribers)
            )
        )

    def run(self):
        runtime_cfg = self.cfg.get("sync_runtime", {})
        use_sync_runtime = bool(runtime_cfg.get("enabled", True))
        self.runtime_bridge = (
            SyncRuntimeBridge(runtime_cfg) if use_sync_runtime else None
        )
        editor_mode = bool(self.cfg.get("static_obstacle_editor_mode", False))
        try:
            if editor_mode:
                self.start_static_editor_session()
            if self.runtime_bridge is not None:
                self.runtime_bridge.start()
            else:
                print(
                    "[KCityRunner] asynchronous scenario test mode: "
                    "Sync Runtime and sensor capture are disabled"
                )
                if str(self.cfg.get("drive_control_mode", "aimdrive_expert")).lower() != "morai_keyboard":
                    self.require_async_ros_control_path()
            if self.cfg.get('ve_objects_checkpoint'):
                from ..adapters.ve_objects import VeObjectInput
                self.ve_object_input = VeObjectInput(self)
                self.ve_object_input.start()
            self.setup()
            if editor_mode:
                self.publish_static_editor_event("session_ready", trial=0)
            if self.runtime_bridge is not None:
                self.runtime_bridge.wait_until_ready(self.expert_controller)
            self.run_timeline()
        finally:
            try:
                self.cleanup()
            finally:
                if editor_mode:
                    self.close_static_editor_session()

    def get_ego_motion_state(self):
        bridge = getattr(self, "runtime_bridge", None)
        if bridge is None:
            # Collection adapters own their runtime and historically consumed
            # the gRPC result (and its failure) directly.
            return self.grpc.get_ego_motion_state()
        try:
            state = self.grpc.get_ego_motion_state()
            if state:
                return state
        except Exception as error:
            grpc_error = error
        else:
            grpc_error = RuntimeError("gRPC returned no Ego state")
        state = bridge.latest_state()
        if state is None:
            raise RuntimeError(
                "No Ego state: gRPC failed and fresh ROS state is unavailable"
            ) from grpc_error
        return state

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
        if self.clock.domain == "native_frame":
            raise RuntimeError(
                "Native collector owns Ego placement, gear and control lifecycle"
            )
        self.route_configured = False
        self.route_setup_mode = None

        runtime_bridge = getattr(self, "runtime_bridge", None)
        asynchronous_test_mode = not bool(
            self.cfg.get("sync_runtime", {}).get("enabled", True)
        )
        if runtime_bridge is not None or asynchronous_test_mode:
            self.stop_expert_controller()
            if hasattr(self.grpc, "stop_ego_cruise"):
                self.grpc.stop_ego_cruise()

        self.grpc.set_ego_transform(self.start_tf)
        if (
            (runtime_bridge is not None or asynchronous_test_mode)
            and self.cfg.get("reset_ego_motion_on_start", False)
            and hasattr(self.grpc, "reset_ego_motion")
        ):
            self.grpc.reset_ego_motion()
            time.sleep(float(self.cfg.get("reset_ego_motion_wait_sec", 0.5)))
        else:
            time.sleep(0.2)

        if runtime_bridge is None and not asynchronous_test_mode:
            # Preserve the proven collection call order while the adapter owns
            # Sync Runtime and Sensor Record coordination.
            self.stop_expert_controller()
            if hasattr(self.grpc, "stop_ego_cruise"):
                self.grpc.stop_ego_cruise()

        return self.activate_expert_drive()

    def activate_expert_drive(
        self, *, before_expert_start=None, after_expert_setup=None,
        after_expert_start=None,
    ):
        """Select AUTO/D and start the shared Expert without moving Ego.

        Optional hooks keep measured fixtures stationary while collection and
        current-Expert readiness gates run. Ordinary zones use no hooks and
        retain their existing control and process call order.
        """
        drive_control_mode = str(
            self.cfg.get("drive_control_mode", "aimdrive_expert")
        ).lower()
        supported_modes = {"aimdrive_expert", "morai_keyboard"}
        if drive_control_mode not in supported_modes:
            raise RuntimeError(
                f"Unsupported drive_control_mode={drive_control_mode!r}"
            )

        if drive_control_mode == "morai_keyboard":
            if any(hook is not None for hook in (
                before_expert_start, after_expert_setup, after_expert_start,
            )):
                raise RuntimeError("Keyboard control cannot bypass guarded Expert startup")
            if self.cfg.get("sync_runtime", {}).get("enabled", True):
                raise RuntimeError("MORAI keyboard mode requires asynchronous runtime")
            if not self.grpc.set_ego_control_mode_keyboard():
                raise RuntimeError("Failed to switch Ego to MORAI keyboard control")
            if hasattr(self.grpc, "set_ego_gear_drive"):
                self.grpc.set_ego_gear_drive()
            self.route_configured = True
            self.route_setup_mode = "morai_keyboard"
            print("[KCityFullLoop] MORAI keyboard controls Ego; no expert process")
            return True

        auto_ok = self.grpc.set_ego_control_mode_auto()
        if hasattr(self.grpc, "set_ego_gear_drive"):
            self.grpc.set_ego_gear_drive()

        if before_expert_start is not None:
            before_expert_start()
        self.setup_expert_controller()
        if after_expert_setup is not None:
            after_expert_setup()
        ve_input = getattr(self, 've_object_input', None)
        if ve_input is not None:
            ve_input.before_drive()
        self.expert_controller.start_process()
        if ve_input is not None:
            ve_input.after_start()
        if after_expert_start is not None:
            after_expert_start()
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
            route_signal_route_links=getattr(self, "route_definition", {}).get("route_links"),
            start_link=self.start_link,
            end_link=self.end_link,
            max_velocity_kmh=self.cfg.get("expert_max_velocity_kmh"),
            speed_cap_kmh=self.cfg.get("expert_speed_cap_kmh"),
            uncapped_speed_zones=self.cfg.get("expert_uncapped_speed_zones", ()),
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
        FullLoopDriveBase.cleanup(self)
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

    def on_goal_reached_before_stop(self, *, elapsed, ego_state, current_s, remaining_s):
        """Notify a specialized scenario before the normal GOAL stop only."""
        return None

    def run_timeline(self):
        self.run_expert_timeline()

    def run_expert_timeline(self):
        if self.clock.domain == "native_frame":
            raise RuntimeError(
                "Native scenarios must be advanced by the capture tick owner"
            )
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

        manual_ego = str(self.cfg.get("drive_control_mode", "aimdrive_expert")).lower() == "morai_keyboard"
        print(
            f"[KCityFullLoop] {'MORAI keyboard' if manual_ego else 'AIMDrive expert'} running. "
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
        editor_collision_signature = None
        progress_anchor_s = None
        last_progress_time = lap_start_time
        editor_mode = bool(self.cfg.get("static_obstacle_editor_mode", False))
        editor_trial_active = False
        editor_trial = 0
        editor_baseline_remaining = set()
        editor_trial_collisions = set()
        editor_max_trials = max(
            0, int(self.cfg.get("static_editor_max_trials", 0))
        )

        while True:
            elapsed = self.clock.now() - lap_start_time
            controller = getattr(self, "expert_controller", None)
            if not manual_ego and (controller is None or not controller.is_running()):
                raise RuntimeError("AIMDrive expert process stopped unexpectedly")

            if editor_mode:
                maintain_obstacles = getattr(
                    self, "maintain_static_obstacles_for_editor", None,
                )
                if callable(maintain_obstacles):
                    maintain_obstacles()

            if editor_mode and not self.static_editor_drive_enabled():
                if editor_trial_active:
                    should_reset = not (
                        editor_max_trials > 0
                        and editor_trial >= editor_max_trials
                    )
                    self.finish_static_editor_trial(
                        editor_trial,
                        "cancelled",
                        "operator_stopped",
                        editor_trial_collisions,
                        reset_for_next=should_reset,
                    )
                    if not should_reset:
                        break
                    editor_trial_active = False
                    lap_start_time = self.clock.now()
                    last_print_time = 0.0
                    off_route_enter_time = None
                    self.expert_last_s = 0.0
                    progress_anchor_s = None
                    last_progress_time = lap_start_time
                time.sleep(check_period_sec)
                continue

            ego_state = self.get_ego_motion_state()
            validation_log = getattr(self, "_runner_validation_log", None)
            if validation_log is not None:
                try:
                    validation_log.observe(
                        self.run_instance_id, elapsed, ego_state,
                        clock_domain=self.clock.domain, lap_index=lap,
                        actor_provider=lambda: snapshot_nearby_actors(self.grpc, ego_state),
                        route_progress_provider=(lambda: self.project_on_route_near_progress(
                            ego_state["x"], ego_state["y"],
                            previous_s=getattr(self, "expert_last_s", None),
                        )[0]) if (not editor_mode and not callable(
                            getattr(self, "observe_collision", None))) else None,
                    )
                except Exception as error:
                    print("[Validation] Ego observation failed: " + str(error),
                          file=sys.stderr, flush=True)
            self.record_ego_trace(elapsed, ego_state)
            ego_x = ego_state["x"]
            ego_y = ego_state["y"]
            collision_objects = sorted(set(
                str(value).strip()
                for value in ego_state.get("collision_objects", [])
                if str(value).strip()
            ))
            if editor_mode and not editor_trial_active:
                editor_trial += 1
                lap = editor_trial - 1
                editor_trial_active = True
                editor_baseline_remaining = set(collision_objects)
                editor_trial_collisions = set()
                lap_start_time = self.clock.now()
                elapsed = 0.0
                last_print_time = 0.0
                off_route_enter_time = None
                self.expert_last_s = 0.0
                progress_anchor_s = None
                last_progress_time = lap_start_time
                self.publish_static_editor_event(
                    "trial_started", trial=editor_trial,
                )
            if editor_mode:
                current_collisions = set(collision_objects)
                editor_baseline_remaining.intersection_update(
                    current_collisions
                )
                editor_trial_collisions.update(
                    current_collisions - editor_baseline_remaining
                )
            # The interactive editor must stay alive so the operator can see
            # the actual contact and driven trace in MORAI/the browser. The
            # separate replay verifier remains fail-closed on every contact.
            if bool(self.cfg.get("static_obstacle_editor_mode", False)):
                signature = tuple(collision_objects)
                if signature != editor_collision_signature:
                    if signature:
                        print(
                            "[STATIC EDITOR] MORAI collision currently reported: "
                            + str(list(signature))
                        )
                    elif editor_collision_signature:
                        print("[STATIC EDITOR] MORAI collision report cleared")
                    editor_collision_signature = signature
            # Collection subclasses own a richer collision policy and event
            # journal through observe_collision(). Plain replay verification
            # fails immediately instead of accepting a touched obstacle.
            elif getattr(self, 'collision_verification_record_only', False):
                try:
                    self.record_collision_observation(ego_state, elapsed)
                except Exception as exc:
                    print('[Collision] Observation write failed: ' + str(exc), flush=True)
            elif collision_objects and not callable(
                getattr(self, "observe_collision", None)
            ):
                self.stop_drive_control()
                recorder = getattr(self, "record_collision_stop", None)
                if callable(recorder):
                    try:
                        recorder(ego_state, elapsed)
                    except Exception as exc:
                        # Diagnostics must never replace the collision failure
                        # or prevent the existing scenario cleanup.
                        print("[Collision] Evidence write failed: " + str(exc))
                raise ScenarioCollisionError(
                    ("Manual Ego collision during tunnel trial: " if manual_ego else
                     "AIMDrive expert collision during scenario verification: ")
                    + str(collision_objects)
                )
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
            if validation_log is not None:
                validation_log.note_route_progress(current_s)
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
                if editor_mode:
                    should_reset = not (
                        editor_max_trials > 0
                        and editor_trial >= editor_max_trials
                    )
                    self.finish_static_editor_trial(
                        editor_trial,
                        "failed",
                        "off_route",
                        editor_trial_collisions,
                        reset_for_next=should_reset,
                    )
                    if not should_reset:
                        break
                    editor_trial_active = False
                    lap_start_time = self.clock.now()
                    last_print_time = 0.0
                    off_route_enter_time = None
                    self.expert_last_s = 0.0
                    progress_anchor_s = None
                    last_progress_time = lap_start_time
                    continue
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
                if editor_mode:
                    should_reset = not (
                        editor_max_trials > 0
                        and editor_trial >= editor_max_trials
                    )
                    self.finish_static_editor_trial(
                        editor_trial,
                        "failed",
                        "no_route_progress",
                        editor_trial_collisions,
                        reset_for_next=should_reset,
                    )
                    if not should_reset:
                        break
                    editor_trial_active = False
                    lap_start_time = self.clock.now()
                    last_print_time = 0.0
                    off_route_enter_time = None
                    self.expert_last_s = 0.0
                    progress_anchor_s = None
                    last_progress_time = lap_start_time
                    continue
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
                if editor_mode:
                    outcome = (
                        "passed" if not editor_trial_collisions else "failed"
                    )
                    reason = (
                        "goal_reached"
                        if outcome == "passed"
                        else "goal_reached_with_collision"
                    )
                    should_reset = not (
                        editor_max_trials > 0
                        and editor_trial >= editor_max_trials
                    )
                    self.finish_static_editor_trial(
                        editor_trial,
                        outcome,
                        reason,
                        editor_trial_collisions,
                        reset_for_next=should_reset,
                    )
                    if not should_reset:
                        break
                    editor_trial_active = False
                    lap_start_time = self.clock.now()
                    last_print_time = 0.0
                    off_route_enter_time = None
                    self.expert_last_s = 0.0
                    progress_anchor_s = None
                    last_progress_time = lap_start_time
                    continue
                self.on_goal_reached_before_stop(
                    elapsed=elapsed, ego_state=ego_state,
                    current_s=current_s, remaining_s=remaining_s,
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
                if editor_mode:
                    should_reset = not (
                        editor_max_trials > 0
                        and editor_trial >= editor_max_trials
                    )
                    self.finish_static_editor_trial(
                        editor_trial,
                        "failed",
                        "timeout",
                        editor_trial_collisions,
                        reset_for_next=should_reset,
                    )
                    if not should_reset:
                        break
                    editor_trial_active = False
                    lap_start_time = self.clock.now()
                    last_print_time = 0.0
                    off_route_enter_time = None
                    self.expert_last_s = 0.0
                    progress_anchor_s = None
                    last_progress_time = lap_start_time
                    continue
                self.stop_drive_control()
                raise TimeoutError(
                    "K-City full-loop timeout: "
                    f"dist={distance_to_goal:.2f}m, elapsed={elapsed:.1f}s"
                )

            time.sleep(check_period_sec)

    def record_ego_trace(self, elapsed, ego_state):
        """Append the Ego pose of every control tick when --ego-trace is set."""
        ve_input = getattr(self, 've_object_input', None)
        if ve_input is not None:
            ve_input.inspect()
        path = self.cfg.get("ego_trace_path")
        if not path:
            return
        stream = getattr(self, "_ego_trace_stream", None)
        if stream is None:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            stream = open(path, "w", encoding="utf-8", buffering=1)
            stream.write("elapsed_s,x,y,yaw_deg,speed\n")
            self._ego_trace_stream = stream
        stream.write("{:.3f},{:.4f},{:.4f},{:.3f},{:.3f}\n".format(
            float(elapsed), float(ego_state["x"]), float(ego_state["y"]),
            float(ego_state["yaw_deg"]), float(ego_state.get("speed", 0.0)),
        ))

    def cleanup(self):
        try:
            bridge = getattr(self, "runtime_bridge", None)
            if bridge is not None:
                bridge.close()
        finally:
            self.stop_expert_controller()
            ve_input = getattr(self, 've_object_input', None)
            if ve_input is not None:
                self.ve_object_input = None
                ve_input.close()
            stream = getattr(self, "_ego_trace_stream", None)
            if stream is not None:
                self._ego_trace_stream = None
                stream.close()
