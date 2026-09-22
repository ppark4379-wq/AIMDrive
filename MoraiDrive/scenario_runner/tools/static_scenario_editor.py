#!/usr/bin/env python3
"""Live browser editor and gated Expert drive for static-obstacle presets.

Operators move obstacles in MORAI while Expert commands are held by a brake
relay. The browser can then release the real Expert commands to ``/ctrl_cmd``
so the planned path and the driven Ego trace remain visible in one view.
"""

import argparse
import json
import math
import os
from pathlib import Path
import re
import socket
import struct
import sys
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PRESET_DIR = ROOT / "scenario_runner" / "config" / "static_scenario_presets"
WEB_PATH = Path(__file__).with_name("static_scenario_editor_web.html")
SCHEMA = "morai_static_obstacle_preset_v1"
NAME_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$")


def windows_host_gateway():
    try:
        with open("/proc/net/route", "r", encoding="ascii") as stream:
            next(stream)
            for line in stream:
                fields = line.split()
                if len(fields) >= 3 and fields[1] == "00000000":
                    return socket.inet_ntoa(struct.pack("<L", int(fields[2], 16)))
    except (OSError, ValueError, StopIteration, struct.error):
        pass
    raise RuntimeError("Windows MORAI gRPC host was not found")


def load_editor_policy():
    config_path = ROOT / "scenario_runner" / "config" / "full_loop.yaml"
    with config_path.open("r", encoding="utf-8") as source:
        config = yaml.safe_load(source)["scenarios"]["full_loop"]
    models = [str(value) for value in config.get("static_obstacle_models", [])]
    scales = {
        str(key): [float(item) for item in value]
        for key, value in config.get("static_obstacle_model_scales", {}).items()
    }
    default_scale = [
        float(value)
        for value in config.get("static_obstacle_default_scale", [2.0, 3.0, 2.0])
    ]
    # Preserve the established route-zone default when the explicit key is absent.
    if "static_obstacle_default_scale" not in config:
        route_path = Path(config["route_definition_path"])
        if not route_path.is_absolute():
            route_path = ROOT / "scenario_runner" / route_path
        with route_path.open("r", encoding="utf-8") as source:
            route = yaml.safe_load(source)["full_loop"]
        default_scale = [
            float(value)
            for value in route["zones"]["static_obstacle"].get("scale", default_scale)
        ]
    return models, scales, default_scale


class LiveState:
    def __init__(self, preset_dir):
        self.lock = threading.RLock()
        self.preset_dir = Path(preset_dir).resolve()
        self.global_path = []
        self.local_path = []
        self.candidates = []
        self.expert_objects = []
        self.ego = None
        self.ego_trace = []
        self.status = "Expert ROS data waiting"
        self.obstacles = []
        self.collision_objects = []
        self.collision_error = None
        self.initial_collision_objects = []
        self.drive_baseline_collision_objects = []
        self.drive_baseline_remaining = set()
        self.new_collision_objects = []
        self.collision_initialized = False
        self.collision_updated_at = None
        self.drive_enabled = False
        self.drive_session_recording = False
        self.drive_started_at = None
        self.control_updated_at = None
        self.drive_handler = None
        self.trial_number = 0
        self.trial_status = "waiting_for_scenario"
        self.trial_result = None
        self.trial_reset_ready = False
        self.verified_obstacles = []
        self.verified_obstacle_signature = None
        self.layout_verified = False
        self.saved_presets = []
        self.grpc_error = None
        self.ros_updated_at = None
        self.local_updated_at = None
        self.lattice_updated_at = None
        self.expert_objects_updated_at = None
        self.grpc_updated_at = None
        self.models, self.model_scales, self.default_scale = load_editor_policy()

    @staticmethod
    def _obstacle_signature(obstacles):
        return tuple(sorted(
            (
                str(item.get("actor_id", "")),
                round(float(item.get("x", 0.0)), 3),
                round(float(item.get("y", 0.0)), 3),
                round(float(item.get("z", 0.0)), 3),
                round(float(item.get("yaw_deg", 0.0)), 2),
                round(float(item.get("width_m", 0.0)), 3),
                round(float(item.get("length_m", 0.0)), 3),
            )
            for item in obstacles
        ))

    @staticmethod
    def _points(path_message):
        return [
            [float(pose.pose.position.x), float(pose.pose.position.y)]
            for pose in path_message.poses
        ]

    def on_global_path(self, message):
        with self.lock:
            self.global_path = self._points(message)
            self.ros_updated_at = time.time()

    def on_local_path(self, message):
        with self.lock:
            self.local_path = self._points(message)
            self.ros_updated_at = time.time()
            self.local_updated_at = self.ros_updated_at

    def on_lattice_paths(self, message):
        candidates = []
        ego = None
        status = self.status
        for marker in message.markers:
            namespace = str(marker.ns)
            if namespace.startswith("candidate_paths_") and marker.points:
                selected = marker.color.r > 0.8 and marker.color.g < 0.3
                valid = selected or marker.color.g > 0.7
                candidates.append({
                    "layer": namespace.replace("candidate_paths_", ""),
                    "selected": selected,
                    "valid": valid,
                    "points": [[float(p.x), float(p.y)] for p in marker.points],
                })
            elif namespace == "ego_shape" and marker.action != marker.DELETE:
                ego = {
                    "x": float(marker.pose.position.x),
                    "y": float(marker.pose.position.y),
                    "yaw_quaternion": [
                        float(marker.pose.orientation.x),
                        float(marker.pose.orientation.y),
                        float(marker.pose.orientation.z),
                        float(marker.pose.orientation.w),
                    ],
                }
            elif namespace == "lattice_status" and marker.text:
                status = str(marker.text)
        with self.lock:
            self.candidates = candidates
            if ego is not None:
                self.ego = ego
                if self.drive_enabled:
                    point = [ego["x"], ego["y"]]
                    if (
                        not self.ego_trace
                        or math.hypot(
                            point[0] - self.ego_trace[-1][0],
                            point[1] - self.ego_trace[-1][1],
                        ) >= 0.05
                    ):
                        self.ego_trace.append(point)
                        if len(self.ego_trace) > 20000:
                            self.ego_trace = self.ego_trace[-20000:]
            self.status = status
            self.ros_updated_at = time.time()
            self.lattice_updated_at = self.ros_updated_at

    def on_lattice_objects(self, message):
        objects = []
        for marker in message.markers:
            if str(marker.ns) != "lattice_objects" or marker.action == marker.DELETE:
                continue
            quaternion = marker.pose.orientation
            yaw = math.atan2(
                2.0 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y),
                1.0 - 2.0 * (quaternion.y * quaternion.y + quaternion.z * quaternion.z),
            )
            objects.append({
                "x": float(marker.pose.position.x),
                "y": float(marker.pose.position.y),
                "yaw_deg": math.degrees(yaw),
                "size_x_m": float(marker.scale.x),
                "size_y_m": float(marker.scale.y),
            })
        with self.lock:
            self.expert_objects = objects
            self.ros_updated_at = time.time()
            self.expert_objects_updated_at = self.ros_updated_at

    def update_obstacles(self, obstacles, error=None):
        with self.lock:
            self.obstacles = obstacles
            self.grpc_error = error
            self.grpc_updated_at = time.time()
            if (
                self.layout_verified
                and self.verified_obstacle_signature is not None
                and self._obstacle_signature(obstacles)
                != self.verified_obstacle_signature
            ):
                self.layout_verified = False
                self.verified_obstacles = []
                self.verified_obstacle_signature = None
                self.trial_result = None
                self.trial_status = "editing"
                self.trial_reset_ready = True

    def update_collisions(self, collision_objects):
        current = sorted(set(
            str(value).strip() for value in collision_objects
            if str(value).strip()
        ))
        with self.lock:
            if not self.collision_initialized:
                self.initial_collision_objects = list(current)
                self.collision_initialized = True
            self.collision_objects = current
            self.collision_error = None
            if self.drive_session_recording:
                current_set = set(current)
                self.drive_baseline_remaining.intersection_update(current_set)
                newly_observed = current_set - self.drive_baseline_remaining
                self.new_collision_objects = sorted(
                    set(self.new_collision_objects) | newly_observed
                )
            elif self.drive_started_at is None:
                self.new_collision_objects = []
            self.collision_updated_at = time.time()

    def update_collision_error(self, error):
        with self.lock:
            self.collision_error = str(error)
            self.collision_updated_at = time.time()

    def attach_drive_handler(self, handler):
        with self.lock:
            self.drive_handler = handler

    def mark_control_received(self):
        with self.lock:
            self.control_updated_at = time.time()

    def on_trial_event(self, message):
        try:
            event = json.loads(message) if isinstance(message, str) else dict(message)
        except (TypeError, ValueError, json.JSONDecodeError):
            return
        event_name = str(event.get("event", ""))
        handler = None
        if event_name == "trial_finished":
            with self.lock:
                handler = self.drive_handler
            if handler is not None:
                handler(False)

        with self.lock:
            if event_name == "session_ready":
                self.trial_status = "editing"
                self.trial_reset_ready = True
            elif event_name == "trial_started":
                self.trial_number = int(event.get("trial", self.trial_number + 1))
                self.trial_status = "driving"
                self.trial_result = None
                self.trial_reset_ready = False
                self.layout_verified = False
                self.verified_obstacles = []
                self.verified_obstacle_signature = None
            elif event_name == "trial_finished":
                outcome = str(event.get("outcome", "failed"))
                self.trial_number = int(event.get("trial", self.trial_number))
                self.trial_status = outcome
                self.trial_result = dict(event)
                self.drive_enabled = False
                self.drive_session_recording = False
                self.trial_reset_ready = False
                if outcome == "passed":
                    self.verified_obstacles = [
                        dict(item) for item in self.obstacles
                    ]
                    self.verified_obstacle_signature = self._obstacle_signature(
                        self.verified_obstacles
                    )
                    self.layout_verified = True
                else:
                    self.verified_obstacles = []
                    self.verified_obstacle_signature = None
                    self.layout_verified = False
            elif event_name == "trial_resetting":
                self.trial_reset_ready = False
                self.local_path = []
                self.candidates = []
                self.expert_objects = []
                self.local_updated_at = None
                self.lattice_updated_at = None
                self.expert_objects_updated_at = None
                self.control_updated_at = None
            elif event_name == "trial_ready":
                self.trial_reset_ready = True

    def set_drive(self, enabled):
        enabled = bool(enabled)
        with self.lock:
            now = time.time()
            if enabled:
                if not self.trial_reset_ready:
                    raise ValueError(
                        "Scenario runner is not ready for the next trial"
                    )
                local_age = (
                    None if self.local_updated_at is None
                    else now - self.local_updated_at
                )
                lattice_age = (
                    None if self.lattice_updated_at is None
                    else now - self.lattice_updated_at
                )
                control_age = (
                    None if self.control_updated_at is None
                    else now - self.control_updated_at
                )
                selected = any(
                    item.get("selected") and item.get("valid")
                    for item in self.candidates
                )
                if local_age is None or local_age > 1.5:
                    raise ValueError("Fresh /local_path is required before driving")
                if lattice_age is None or lattice_age > 1.5 or not selected:
                    raise ValueError("A selected valid Expert path is required before driving")
                if control_age is None or control_age > 1.5:
                    raise ValueError("Fresh Expert control commands are required before driving")
                if self.drive_handler is None:
                    raise ValueError("Expert command relay is unavailable")
                self.ego_trace = []
                if self.ego is not None:
                    self.ego_trace.append([self.ego["x"], self.ego["y"]])
                self.drive_started_at = now
                self.drive_session_recording = True
                self.drive_baseline_collision_objects = list(
                    self.collision_objects
                )
                self.drive_baseline_remaining = set(self.collision_objects)
                self.new_collision_objects = []
                self.trial_status = "starting"
                self.trial_result = None
                self.trial_reset_ready = False
                self.layout_verified = False
                self.verified_obstacles = []
                self.verified_obstacle_signature = None
            handler = self.drive_handler
        if handler is None:
            raise ValueError("Expert command relay is unavailable")
        handler(enabled)
        with self.lock:
            self.drive_enabled = enabled
            self.drive_session_recording = enabled
        return self.snapshot()

    def snapshot(self):
        with self.lock:
            return {
                "global_path": list(self.global_path),
                "local_path": list(self.local_path),
                "candidates": list(self.candidates),
                "expert_objects": list(self.expert_objects),
                "ego": self.ego,
                "ego_trace": list(self.ego_trace),
                "status": self.status,
                "obstacles": list(self.obstacles),
                "allowed_models": list(self.models),
                "default_scales": self.model_scales,
                "fallback_scale": self.default_scale,
                "vehicle_footprint": {
                    "length_m": 4.635,
                    "width_m": 2.0,
                    # Same baseline clearance used by the recorded static
                    # planner: collision + tracking + dense-edge reserve.
                    "margin_m": 0.72,
                    "display_only": True,
                },
                "grpc_error": self.grpc_error,
                "drive_enabled": self.drive_enabled,
                "drive_started_at": self.drive_started_at,
                "trial_number": self.trial_number,
                "trial_status": self.trial_status,
                "trial_result": self.trial_result,
                "trial_reset_ready": self.trial_reset_ready,
                "layout_verified": self.layout_verified,
                "saved_presets": list(self.saved_presets),
                "collision_objects": list(self.collision_objects),
                "collision_error": self.collision_error,
                "initial_collision_objects": list(self.initial_collision_objects),
                "drive_baseline_collision_objects": list(
                    self.drive_baseline_collision_objects
                ),
                "new_collision_objects": list(self.new_collision_objects),
                "ros_age_sec": None if self.ros_updated_at is None else time.time() - self.ros_updated_at,
                "local_age_sec": None if self.local_updated_at is None else time.time() - self.local_updated_at,
                "lattice_age_sec": None if self.lattice_updated_at is None else time.time() - self.lattice_updated_at,
                "expert_objects_age_sec": None if self.expert_objects_updated_at is None else time.time() - self.expert_objects_updated_at,
                "grpc_age_sec": None if self.grpc_updated_at is None else time.time() - self.grpc_updated_at,
                "collision_age_sec": None if self.collision_updated_at is None else time.time() - self.collision_updated_at,
                "control_age_sec": None if self.control_updated_at is None else time.time() - self.control_updated_at,
            }

    def save(self, request):
        name = str(request.get("name", "")).strip()
        if not NAME_PATTERN.fullmatch(name):
            raise ValueError("Name must use 1-64 ASCII letters, digits, '_' or '-'")
        selected = request.get("obstacles", [])
        if not isinstance(selected, list) or not 1 <= len(selected) <= 8:
            raise ValueError("Select between one and eight obstacles")

        with self.lock:
            if not self.layout_verified or not self.trial_result:
                raise ValueError(
                    "Run and pass this exact obstacle layout before saving"
                )
            if str(self.trial_result.get("outcome")) != "passed":
                raise ValueError(
                    "Only a collision-free completed trial can be saved"
                )
            inventory = {
                item["actor_id"]: dict(item)
                for item in self.verified_obstacles
            }
            route = list(self.global_path)
        if len(route) < 2:
            raise ValueError("/global_path is unavailable; run the Expert first")

        output_obstacles = []
        for selected_item in selected:
            actor_id = str(selected_item.get("actor_id", ""))
            if actor_id not in inventory:
                raise ValueError("Obstacle {} disappeared before save".format(actor_id))
            item = inventory[actor_id]
            model = str(selected_item.get("model") or item.get("model") or "").strip()
            if model not in self.models:
                raise ValueError("Choose an allowed replay model for actor {}".format(actor_id))
            scale = selected_item.get("scale") or self.model_scales.get(model) or self.default_scale
            if not isinstance(scale, list) or len(scale) != 3 or any(float(v) <= 0 for v in scale):
                raise ValueError("Scale for actor {} must contain three positive values".format(actor_id))
            nearest_index = min(
                range(len(route)),
                key=lambda index: math.hypot(
                    route[index][0] - item["x"], route[index][1] - item["y"]
                ),
            )
            route_distance = math.hypot(
                route[nearest_index][0] - item["x"],
                route[nearest_index][1] - item["y"],
            )
            output_obstacles.append({
                "model": model,
                "position": {
                    "x": round(float(item["x"]), 6),
                    "y": round(float(item["y"]), 6),
                    "z": round(float(item["z"]), 6),
                },
                "yaw_deg": round(float(item["yaw_deg"]), 6),
                "scale": [float(value) for value in scale],
                "route_path_index": int(nearest_index),
                "route_distance_m": round(route_distance, 4),
                "source_actor_id": actor_id,
            })

        document = {
            "schema": SCHEMA,
            "name": name,
            "map_name": "R_KR_PR_K-city_2025",
            "scenario": "full_loop_static",
            "saved_at_unix": time.time(),
            "source": "live_morai_static_scenario_editor",
            "obstacles": output_obstacles,
        }
        self.preset_dir.mkdir(parents=True, exist_ok=True)
        output = (self.preset_dir / (name + ".yaml")).resolve()
        if output.parent != self.preset_dir:
            raise ValueError("Preset path escapes the configured directory")
        if output.exists() and not bool(request.get("overwrite", False)):
            raise FileExistsError("Preset already exists; enable overwrite to replace it")
        temporary = output.with_suffix(".yaml.tmp")
        temporary.write_text(
            yaml.safe_dump(document, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        os.replace(str(temporary), str(output))
        with self.lock:
            self.saved_presets.append(str(output))
        return {"saved": str(output), "preset": document}


class MoraiObstacleReader(threading.Thread):
    def __init__(self, state, sdk_root, target, poll_hz):
        super().__init__(name="morai-obstacle-reader", daemon=True)
        self.state = state
        self.sdk_root = Path(sdk_root).resolve()
        self.target = target
        self.period = 1.0 / max(0.2, float(poll_hz))
        self.stop_event = threading.Event()
        self.model_cache = {}
        self.channel = None

    def _connect(self):
        proto_path = str(self.sdk_root / "proto")
        if proto_path not in sys.path:
            sys.path.insert(0, proto_path)
        import grpc
        import morai.actor.actor_pb2_grpc as actor_grpc
        from morai.actor.actor_get_pb2 import GetAllActorsFilter
        self.grpc = grpc
        self.filter_type = GetAllActorsFilter
        self.channel = grpc.insecure_channel(self.target)
        self.stub = actor_grpc.ActorStub(self.channel)

    def _option_name(self, actor_info):
        actor_id = str(actor_info.id.value)
        try:
            value = self.stub.GetOptionName(actor_info, timeout=0.3).value
        except Exception:
            # Actor IDs are reused by MORAI after a scenario restart. Keep a
            # previous value only as a short RPC-failure fallback; successful
            # reads always refresh the current actor's model name.
            return self.model_cache.get(actor_id, "")
        value = str(value)
        self.model_cache[actor_id] = value
        return value

    @staticmethod
    def _obstacle_payload(actor, model):
        """Keep replay pose separate from the physical world OBB.

        Actor transform is the pose accepted by SpawnObstacle. MORAI's
        bounding_box fields are the world-space physical box used for drawing
        and dataset geometry; some barricade assets have a model-axis offset.
        """
        transform = actor.transform
        box = actor.bounding_box
        extent = box.extent
        return {
            "actor_id": str(actor.actor_info.id.value),
            "model": str(model),
            "x": float(transform.location.x),
            "y": float(transform.location.y),
            "z": float(transform.location.z),
            "yaw_deg": float(transform.rotation.z),
            "footprint_x": float(box.location.x),
            "footprint_y": float(box.location.y),
            "footprint_z": float(box.location.z),
            "footprint_yaw_deg": float(box.rotation.z),
            "width_m": 2.0 * abs(float(extent.x)),
            "length_m": 2.0 * abs(float(extent.y)),
            "geometry_source": "GetAllActorsState.bounding_box",
        }

    def read(self):
        response = self.stub.GetAllActorsState(
            self.filter_type(client_key="", vehicle=False, pedestrian=False, obstacle=True),
            timeout=0.75,
        )
        obstacles = []
        for actor in response.states:
            obstacles.append(self._obstacle_payload(
                actor, self._option_name(actor.actor_info)
            ))
        return obstacles

    def read_ego_collisions(self):
        response = self.stub.GetAllActorsState(
            self.filter_type(
                client_key="", vehicle=True, pedestrian=False, obstacle=False
            ),
            timeout=0.75,
        )
        ego = next((
            actor for actor in response.states
            if str(actor.actor_info.id.value).casefold() == "ego"
        ), None)
        if ego is None:
            raise RuntimeError("GetAllActorsState did not return Ego")
        return list(ego.vehicle_state.collision_objects)

    def run(self):
        while not self.stop_event.is_set():
            started = time.monotonic()
            try:
                if self.channel is None:
                    self._connect()
                self.state.update_obstacles(self.read())
                try:
                    self.state.update_collisions(self.read_ego_collisions())
                except Exception as error:
                    self.state.update_collision_error(
                        "{}: {}".format(type(error).__name__, error)
                    )
            except Exception as error:
                self.state.update_obstacles([], "{}: {}".format(type(error).__name__, error))
                if self.channel is not None:
                    self.channel.close()
                self.channel = None
            self.stop_event.wait(
                max(0.0, max(self.period, 1.0 if self.channel is None else 0.0) - (time.monotonic() - started))
            )

    def close(self):
        self.stop_event.set()
        if self.channel is not None:
            self.channel.close()


class ExpertCommandRelay:
    """Hold Expert commands with braking, then release them from the browser."""

    def __init__(self, state, publisher, control_type, mode_publisher=None,
                 arming_delay_sec=0.0):
        self.state = state
        self.publisher = publisher
        self.control_type = control_type
        self.mode_publisher = mode_publisher
        self.arming_delay_sec = max(0.0, float(arming_delay_sec))
        self.arming_until = 0.0
        self.lock = threading.RLock()
        self.enabled = False
        self.last_command = None
        state.attach_drive_handler(self.set_enabled)
        self._publish_mode(False)

    def _publish_mode(self, enabled):
        if self.mode_publisher is not None:
            self.mode_publisher.publish(bool(enabled))

    def _brake(self):
        command = self.control_type()
        command.longlCmdType = 1
        command.accel = 0.0
        command.brake = 1.0
        command.front_steer = 0.0
        self.publisher.publish(command)

    def on_command(self, command):
        with self.lock:
            self.last_command = command
            self.state.mark_control_received()
            if self.enabled and time.monotonic() >= self.arming_until:
                self.publisher.publish(command)
            else:
                self._brake()

    def set_enabled(self, enabled):
        with self.lock:
            # Commit the accepted route inside Expert before any command can
            # reach MORAI. The short arm window lets the ROS subscriber apply
            # that mode transition across processes.
            self._publish_mode(enabled)
            self.enabled = bool(enabled)
            self.arming_until = (
                time.monotonic() + self.arming_delay_sec
                if self.enabled else 0.0
            )
            if (self.enabled and self.last_command is not None
                    and self.arming_delay_sec == 0.0):
                self.publisher.publish(self.last_command)
            else:
                self._brake()

    def close(self):
        self.set_enabled(False)


def make_handler(state):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            sys.stdout.write("[EditorHTTP] " + fmt % args + "\n")

        def send_json(self, payload, status=HTTPStatus.OK):
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/api/state":
                self.send_json(state.snapshot())
                return
            if self.path in ("/", "/index.html"):
                body = WEB_PATH.read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self):
            if self.path not in ("/api/save", "/api/drive"):
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 1024 * 1024:
                    raise ValueError("Invalid request size")
                request = json.loads(self.rfile.read(length).decode("utf-8"))
                if self.path == "/api/drive":
                    self.send_json(state.set_drive(bool(request.get("enabled"))))
                else:
                    self.send_json(state.save(request), HTTPStatus.CREATED)
            except FileExistsError as error:
                self.send_json({"error": str(error)}, HTTPStatus.CONFLICT)
            except (ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
                self.send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
            except Exception as error:
                self.send_json(
                    {"error": "{}: {}".format(type(error).__name__, error)},
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                )

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--listen", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--grpc-host", default=os.environ.get("MORAI_GRPC_HOST", "auto"))
    parser.add_argument("--grpc-port", type=int, default=int(os.environ.get("MORAI_GRPC_PORT", "7789")))
    parser.add_argument("--poll-hz", type=float, default=4.0)
    parser.add_argument("--preset-dir", default=str(DEFAULT_PRESET_DIR))
    parser.add_argument(
        "--auto-drive",
        action="store_true",
        help=(
            "Release Expert control automatically as soon as fresh control "
            "and a selected valid path are available."
        ),
    )
    args = parser.parse_args()
    if not 1 <= args.port <= 65535 or not 1 <= args.grpc_port <= 65535:
        parser.error("ports must be in 1..65535")

    import rospy
    from morai_msgs.msg import CtrlCmd
    from nav_msgs.msg import Path as RosPath
    from std_msgs.msg import Bool, String
    from visualization_msgs.msg import MarkerArray

    state = LiveState(args.preset_dir)
    rospy.init_node("static_scenario_editor", anonymous=True, disable_signals=True)
    rospy.Subscriber("/global_path", RosPath, state.on_global_path, queue_size=1)
    rospy.Subscriber("/local_path", RosPath, state.on_local_path, queue_size=1)
    rospy.Subscriber("/lattice/paths", MarkerArray, state.on_lattice_paths, queue_size=1)
    rospy.Subscriber("/lattice/objects", MarkerArray, state.on_lattice_objects, queue_size=1)
    rospy.Subscriber(
        "/static_scenario_editor/trial_event",
        String,
        lambda message: state.on_trial_event(message.data),
        queue_size=10,
    )
    control_publisher = rospy.Publisher("/ctrl_cmd", CtrlCmd, queue_size=1)
    mode_publisher = rospy.Publisher(
        "/static_scenario_editor/drive_enabled", Bool, queue_size=1, latch=True
    )
    relay = ExpertCommandRelay(
        state,
        control_publisher,
        CtrlCmd,
        mode_publisher=mode_publisher,
        arming_delay_sec=0.15,
    )
    rospy.Subscriber(
        "/static_scenario_editor/ctrl_cmd",
        CtrlCmd,
        relay.on_command,
        queue_size=1,
    )

    grpc_host = windows_host_gateway() if args.grpc_host.casefold() == "auto" else args.grpc_host
    sdk_root = ROOT / "data_collection" / "third_party" / "grpc_inha_univ" / "src"
    reader = MoraiObstacleReader(
        state, sdk_root, "{}:{}".format(grpc_host, args.grpc_port), args.poll_hz
    )
    reader.start()
    auto_drive_stop = threading.Event()
    auto_drive_thread = None
    if args.auto_drive:
        def enable_drive_when_ready():
            last_error = None
            while not auto_drive_stop.wait(0.2):
                try:
                    state.set_drive(True)
                    print("[Editor] valid path ready; actual Expert drive enabled")
                    return
                except ValueError as error:
                    message = str(error)
                    if message != last_error:
                        print("[Editor] auto-drive waiting: " + message)
                        last_error = message

        auto_drive_thread = threading.Thread(
            target=enable_drive_when_ready,
            name="static-editor-auto-drive",
            daemon=True,
        )
        auto_drive_thread.start()
    server = ThreadingHTTPServer((args.listen, args.port), make_handler(state))
    print("[Editor] MORAI gRPC: {}:{}".format(grpc_host, args.grpc_port))
    print("[Editor] browser: http://{}:{}".format(args.listen, args.port))
    print("[Editor] presets: {}".format(state.preset_dir))
    print("[Editor] Move obstacles while drive is stopped; use the browser to release Expert control.")
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        print("[Editor] stopping")
    finally:
        server.shutdown()
        server.server_close()
        auto_drive_stop.set()
        if auto_drive_thread is not None:
            auto_drive_thread.join(timeout=1.0)
        relay.close()
        reader.close()
        reader.join(timeout=2.0)


if __name__ == "__main__":
    main()
