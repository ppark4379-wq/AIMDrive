import json
import math
import os
import threading
import time


class ObjectGhostAudit:
    """Compare live ROS object reports with MORAI gRPC actor states."""

    def __init__(
        self,
        grpc_client,
        output_dir,
        run_instance_id,
        sample_period_sec=0.5,
        match_distance_m=5.0,
        ros_max_age_sec=1.0,
        candidate_samples=3,
    ):
        self.grpc = grpc_client
        self.output_dir = os.path.abspath(output_dir)
        self.run_instance_id = str(run_instance_id)
        self.sample_period_sec = max(0.1, float(sample_period_sec))
        self.match_distance_m = max(0.1, float(match_distance_m))
        self.ros_max_age_sec = max(0.1, float(ros_max_age_sec))
        self.candidate_samples = max(2, int(candidate_samples))

        self._lock = threading.Lock()
        self._latest_objects = None
        self._latest_ctrl = None
        self._last_sample_monotonic = float("-inf")
        self._subscribers = []
        self._output = None
        self.output_path = None
        self.summary_path = None
        self._metrics = {
            "samples": 0,
            "fresh_ros_samples": 0,
            "stale_or_missing_ros_samples": 0,
            "grpc_error_samples": 0,
            "ros_npc_observations": 0,
            "ros_object_observations": 0,
            "matched_observations": 0,
            "ros_only_observations": 0,
        }
        self._ros_only_tracks = {}

    @staticmethod
    def _ros_object(item):
        return {
            "unique_id": int(item.unique_id),
            "name": str(item.name),
            "type": int(item.type),
            "x": float(item.position.x),
            "y": float(item.position.y),
            "z": float(item.position.z),
            "velocity_mps": float(item.velocity.x),
            "heading_deg": float(item.heading),
            "size_x": float(item.size.x),
            "size_y": float(item.size.y),
        }

    @staticmethod
    def _grpc_actor(state):
        actor_info = getattr(state, "actor_info", None)
        identifier = getattr(getattr(actor_info, "id", None), "value", "")
        location = state.transform.location
        velocity = getattr(state, "global_velocity", None)
        if velocity is None:
            velocity = getattr(state, "velocity", None)
        return {
            "actor_id": str(identifier),
            "object_type": int(getattr(actor_info, "object_type", 0)),
            "x": float(location.x),
            "y": float(location.y),
            "z": float(location.z),
            "velocity_x_mps": float(getattr(velocity, "x", 0.0)),
            "velocity_y_mps": float(getattr(velocity, "y", 0.0)),
        }

    @staticmethod
    def compare_snapshots(ros_npcs, grpc_vehicles, match_distance_m):
        """Return one-to-one spatial matches and unmatched objects."""
        remaining = set(range(len(grpc_vehicles)))
        matches = []
        ros_only = []
        for ros_object in ros_npcs:
            best_index = None
            best_distance = float("inf")
            for index in remaining:
                grpc_actor = grpc_vehicles[index]
                distance = math.hypot(
                    float(ros_object["x"]) - float(grpc_actor["x"]),
                    float(ros_object["y"]) - float(grpc_actor["y"]),
                )
                if distance < best_distance:
                    best_distance = distance
                    best_index = index
            if best_index is not None and best_distance <= match_distance_m:
                remaining.remove(best_index)
                matches.append(
                    {
                        "ros_unique_id": ros_object["unique_id"],
                        "ros_name": ros_object["name"],
                        "grpc_actor_id": grpc_vehicles[best_index]["actor_id"],
                        "distance_m": best_distance,
                    }
                )
            else:
                candidate = dict(ros_object)
                candidate["nearest_grpc_distance_m"] = (
                    None if not math.isfinite(best_distance) else best_distance
                )
                ros_only.append(candidate)
        grpc_only = [grpc_vehicles[index] for index in sorted(remaining)]
        return matches, ros_only, grpc_only

    def _object_callback(self, data):
        snapshot = {
            "receipt_monotonic": time.monotonic(),
            "header_seq": int(data.header.seq),
            "header_stamp_sec": float(data.header.stamp.to_sec()),
            "npcs": [self._ros_object(item) for item in data.npc_list],
            "pedestrians": [
                self._ros_object(item) for item in data.pedestrian_list
            ],
            "obstacles": [
                self._ros_object(item) for item in data.obstacle_list
            ],
        }
        with self._lock:
            self._latest_objects = snapshot

    def _ctrl_callback(self, data):
        snapshot = {
            "receipt_monotonic": time.monotonic(),
            "longitudinal_command_type": int(data.longlCmdType),
            "accel": float(data.accel),
            "brake": float(data.brake),
            "front_steer": float(data.front_steer),
            "velocity": float(data.velocity),
            "acceleration": float(data.acceleration),
        }
        with self._lock:
            self._latest_ctrl = snapshot

    def _write(self, record):
        if self._output is None:
            return
        self._output.write(json.dumps(record, sort_keys=True) + "\n")
        self._output.flush()

    def start(self):
        try:
            import rospy
            from morai_msgs.msg import CtrlCmd, ObjectStatusList
        except Exception as error:
            raise RuntimeError(
                "Object audit requires the AIMDrive ROS workspace: "
                "source expert/devel/setup.bash"
            ) from error

        os.makedirs(self.output_dir, exist_ok=True)
        base_name = f"highway_object_audit_{self.run_instance_id}"
        self.output_path = os.path.join(
            self.output_dir,
            base_name + ".jsonl",
        )
        self.summary_path = os.path.join(
            self.output_dir,
            base_name + ".summary.json",
        )
        self._output = open(self.output_path, "w", encoding="utf-8")

        if not rospy.core.is_initialized():
            rospy.init_node(
                "scenario_runner_object_audit",
                anonymous=True,
                disable_signals=True,
            )
        self._subscribers = [
            rospy.Subscriber(
                "/Object_topic",
                ObjectStatusList,
                self._object_callback,
                queue_size=1,
            ),
            rospy.Subscriber(
                "/ctrl_cmd",
                CtrlCmd,
                self._ctrl_callback,
                queue_size=1,
            ),
        ]
        self._write(
            {
                "record_type": "metadata",
                "run_instance_id": self.run_instance_id,
                "sample_period_sec": self.sample_period_sec,
                "match_distance_m": self.match_distance_m,
                "ros_max_age_sec": self.ros_max_age_sec,
                "candidate_samples": self.candidate_samples,
                "started_unix": time.time(),
            }
        )
        print(f"[ObjectAudit] recording: {self.output_path}")

    def _grpc_snapshot(self):
        if self.grpc.world is None:
            raise RuntimeError("MORAI gRPC world is not available")
        response = self.grpc.world.get_all_actors_state(
            vehicle=True,
            pedestrian=True,
            obstacle=True,
        )
        return [
            self._grpc_actor(state)
            for state in list(getattr(response, "states", []))
        ]

    def _update_ros_only_tracks(self, ros_only):
        present = set()
        for item in ros_only:
            key = str(item["unique_id"])
            present.add(key)
            track = self._ros_only_tracks.setdefault(
                key,
                {
                    "unique_id": item["unique_id"],
                    "name": item["name"],
                    "observations": 0,
                    "consecutive": 0,
                    "max_consecutive": 0,
                    "first_position": [item["x"], item["y"]],
                    "last_position": [item["x"], item["y"]],
                },
            )
            track["observations"] += 1
            track["consecutive"] += 1
            track["max_consecutive"] = max(
                track["max_consecutive"],
                track["consecutive"],
            )
            track["last_position"] = [item["x"], item["y"]]
            if (
                track["consecutive"] == self.candidate_samples
                or track["consecutive"] % 10 == 0
            ):
                print(
                    "[ObjectAudit] ROS_ONLY candidate "
                    f"id={item['unique_id']} name={item['name']} "
                    f"pos=({item['x']:.2f},{item['y']:.2f}) "
                    f"consecutive={track['consecutive']}"
                )
        for key, track in self._ros_only_tracks.items():
            if key not in present:
                track["consecutive"] = 0

    def sample(self, elapsed, ego_state, current_s):
        now = time.monotonic()
        if now - self._last_sample_monotonic < self.sample_period_sec:
            return
        self._last_sample_monotonic = now
        with self._lock:
            objects = self._latest_objects
            ctrl = self._latest_ctrl

        self._metrics["samples"] += 1
        record = {
            "record_type": "sample",
            "sample_index": self._metrics["samples"],
            "wall_time_unix": time.time(),
            "elapsed_sec": float(elapsed),
            "route_s_m": float(current_s),
            "ego": dict(ego_state),
            "ctrl_cmd": ctrl,
        }
        if objects is None:
            self._metrics["stale_or_missing_ros_samples"] += 1
            record["status"] = "ros_object_topic_missing"
            self._write(record)
            return

        ros_age = now - float(objects["receipt_monotonic"])
        record["ros_age_sec"] = ros_age
        record["ros_header_seq"] = objects["header_seq"]
        record["ros_header_stamp_sec"] = objects["header_stamp_sec"]
        record["ros_npcs"] = objects["npcs"]
        record["ros_pedestrians"] = objects["pedestrians"]
        record["ros_obstacles"] = objects["obstacles"]
        if ros_age > self.ros_max_age_sec:
            self._metrics["stale_or_missing_ros_samples"] += 1
            record["status"] = "ros_object_topic_stale"
            self._write(record)
            return

        self._metrics["fresh_ros_samples"] += 1
        try:
            grpc_vehicles = self._grpc_snapshot()
        except Exception as error:
            self._metrics["grpc_error_samples"] += 1
            record["status"] = "grpc_snapshot_error"
            record["grpc_error"] = repr(error)
            self._write(record)
            return

        ros_objects = (
            list(objects["npcs"])
            + list(objects["pedestrians"])
            + list(objects["obstacles"])
        )
        matches, ros_only, grpc_only = self.compare_snapshots(
            ros_objects,
            grpc_vehicles,
            self.match_distance_m,
        )
        self._metrics["ros_npc_observations"] += len(objects["npcs"])
        self._metrics["ros_object_observations"] += len(ros_objects)
        self._metrics["matched_observations"] += len(matches)
        self._metrics["ros_only_observations"] += len(ros_only)
        self._update_ros_only_tracks(ros_only)
        record.update(
            {
                "status": "ok",
                "grpc_vehicles": grpc_vehicles,
                "matches": matches,
                "ros_only": ros_only,
                "grpc_only": grpc_only,
            }
        )
        self._write(record)

    def stop(self):
        for subscriber in self._subscribers:
            try:
                subscriber.unregister()
            except Exception:
                pass
        self._subscribers = []

        candidates = [
            dict(track)
            for track in self._ros_only_tracks.values()
            if track["max_consecutive"] >= self.candidate_samples
        ]
        denominator = max(1, self._metrics["ros_object_observations"])
        summary = {
            "run_instance_id": self.run_instance_id,
            "metrics": dict(self._metrics),
            "grpc_match_coverage": (
                self._metrics["matched_observations"] / denominator
            ),
            "ros_only_candidates": candidates,
            "ended_unix": time.time(),
            "classification_note": (
                "A ROS-only candidate is strong evidence only when overall "
                "gRPC match coverage is high; it is not based on rendering."
            ),
        }
        if self.summary_path is not None:
            with open(self.summary_path, "w", encoding="utf-8") as output:
                json.dump(summary, output, indent=2, sort_keys=True)
                output.write("\n")
        self._write({"record_type": "summary", **summary})
        if self._output is not None:
            self._output.close()
            self._output = None
        if self.summary_path is not None:
            print(
                f"[ObjectAudit] summary: {self.summary_path}, "
                f"coverage={summary['grpc_match_coverage']:.3f}, "
                f"ros_only_candidates={len(candidates)}"
            )
