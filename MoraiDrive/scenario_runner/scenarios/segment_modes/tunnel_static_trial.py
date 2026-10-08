"""One guarded Expert drive through an already owned tunnel fixture."""

import copy
from contextlib import nullcontext
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import threading
import time
from types import SimpleNamespace

from ...adapters.morai_grpc import MoraiGrpcClient
from ..full_loop_parts.tunnel_static_parts import live, screening
from ..full_loop_parts.tunnel_static_parts.catalog import load_static_models
from ..full_loop_parts.tunnel_static_parts.layout import (
    boxes_overlap, footprint_corners, normalize_footprint, project_polyline, route_pose, validate_candidate,
)
from .modes import TUNNEL_NORMAL_START


MODE = "full_loop_tunnel_static"


def enabled(context):
    return (getattr(context, "cfg", {}).get("mission_mode") == MODE
            or getattr(getattr(context, "mission_mode", None), "name", None) == MODE)


def _write(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def mapped_original_lane(map_loader):
    """The actual own tunnel/departure lane, independent of the trial END."""
    from shapely.geometry import LineString
    own_id, departure_id = "A2256W000126", "A2256W000128"
    own, departure = (map_loader.get_link(identity) for identity in (own_id, departure_id))
    own_points, departure_points = own["points"], departure["points"]
    if (not own.get("to_node_idx") or own.get("to_node_idx") != departure.get("from_node_idx")
            or math.dist(own_points[-1][:2], departure_points[0][:2]) > .05):
        raise RuntimeError("Original tunnel lane and mapped departure are not connected")
    widths = [float(row[key]) for row in (own, departure) for key in ("width_start", "width_end")]
    if any(not math.isfinite(width) or width <= 0 for width in widths):
        raise RuntimeError("Invalid actual mapped original-lane width")
    points = [point[:2] for point in own_points]+[point[:2] for point in departure_points[1:]]
    polygon = LineString(points).buffer(min(widths)*.5, cap_style=2, join_style=2)
    if polygon.is_empty or not polygon.is_valid:
        raise RuntimeError("Invalid original mapped lane polygon")
    return polygon, {"link_ids": [own_id, departure_id], "nominal_minimum_width_m": min(widths),
                     "start_xyz": own_points[0], "end_xyz": departure_points[-1],
                     "basis": "whole connected MGeo lane links; mission END does not clip the road"}


def lane_return_checks(boundary, polygon, observed):
    import numpy as np
    from shapely.geometry import Polygon
    heading = math.radians(observed["rotation_xyz_deg"][2])
    corners = boundary.corners_world(np.array([observed["position_xyz"][:2]]), np.array([heading]), margin=.10)[0]
    raw = observed["raw_bbox_normalized"]
    raw_corners = footprint_corners(raw["center_xyz"], raw["width_m"]+.20, raw["length_m"]+.20, raw["heading_deg"])
    policy_passed = bool(polygon.covers(Polygon(corners)))
    raw_passed = bool(polygon.covers(Polygon(raw_corners)))
    return {"policy_body_in_original_lane": policy_passed, "raw_bbox_in_original_lane": raw_passed,
            "whole_body_in_original_lane": policy_passed and raw_passed,
            "policy_body_corners_world": corners.tolist(), "raw_bbox_corners_world": raw_corners}


def classify_lane_maneuver(points, geometry, *, hysteresis_m=.25, minimum_run_m=2.):
    """Describe measured lane excursions without naming a long bypass an S."""
    own, opposite = geometry["own_points"], geometry["opposing_points"]
    transitions, lateral_values = [], []
    pending, pending_station = None, None
    length = geometry.get("length_m") or sum(math.dist(a[:2], b[:2]) for a, b in zip(own, own[1:]))
    for point in points:
        station, lateral, _ = project_polyline(own, *point[:2])
        if not .1 < station < length-.1:
            continue
        center, _ = route_pose(own, station)
        opposite_station = project_polyline(opposite, *center[:2])[0]
        opposite_center, _ = route_pose(opposite, opposite_station)
        separation = project_polyline(own, *opposite_center[:2])[1]
        sign = 1. if separation >= 0 else -1.
        coordinate, divider = sign*lateral, abs(separation)*.5
        lateral_values.append(lateral)
        lane = ("opposing" if coordinate > divider+hysteresis_m else
                "own" if coordinate < divider-hysteresis_m else None)
        if lane is None:
            continue
        if lane != pending:
            pending, pending_station = lane, station
        if abs(station-pending_station) >= minimum_run_m and (not transitions or transitions[-1]["lane"] != lane):
            transitions.append({"lane": lane, "station_m": station})
    sequence = [row["lane"] for row in transitions]
    excursions = sum(sequence[index:index+3] == ["own", "opposing", "own"] for index in range(len(sequence)-2))
    pattern = ("alternating_lane_detours" if excursions >= 2 else
               "single_opposing_lane_excursion" if excursions == 1 else
               "rejoined_original_lane" if sequence and sequence[-1] == "own" and "opposing" in sequence else
               "opposing_lane_without_verified_return" if "opposing" in sequence else "within_original_lane")
    return {"pattern": pattern, "lane_sequence": sequence, "transitions": transitions,
            "completed_opposing_excursions": excursions, "s_maneuver_observed": excursions >= 2,
            "hysteresis_m": hysteresis_m, "minimum_run_m": minimum_run_m,
            "minimum_lateral_m": min(lateral_values) if lateral_values else None,
            "maximum_lateral_m": max(lateral_values) if lateral_values else None}


class TrialRosEvidence:
    """Require fresh simulator objects and capture the Expert's selected path."""
    def __init__(self):
        self.lock = threading.Lock()
        self.objects = None
        self.selected_path = None
        self.subscribers = []
        self.object_messages = self.path_messages = 0
        self.ego = None
        self.commands = []
        self.object_history = []

    def start(self):
        import rospy
        from morai_msgs.msg import ObjectStatusList, EgoVehicleStatus, CtrlCmd
        from visualization_msgs.msg import MarkerArray
        if not rospy.core.is_initialized():
            rospy.init_node("tunnel_static_trial_evidence", anonymous=True, disable_signals=True)
        self.subscribers = [
            rospy.Subscriber("/Object_topic", ObjectStatusList, self.on_objects, queue_size=1),
            rospy.Subscriber("/lattice/paths", MarkerArray, self.on_paths, queue_size=1),
            rospy.Subscriber("/Ego_topic", EgoVehicleStatus, self.on_ego, queue_size=1),
            rospy.Subscriber("/ctrl_cmd", CtrlCmd, self.on_command, queue_size=1),
        ]

    def close(self):
        for subscriber in self.subscribers:
            subscriber.unregister()
        self.subscribers = []

    def on_objects(self, message):
        rows = []
        for kind, objects in (("vehicle", message.npc_list), ("obstacle", message.obstacle_list)):
            for item in objects:
                rows.append({"raw_actor_id": str(item.unique_id), "actor_kind": kind, "name": str(item.name),
                    "position_xyz": [float(item.position.x), float(item.position.y), float(item.position.z)],
                    "heading_deg": float(item.heading), "size_xyz": [float(item.size.x), float(item.size.y), float(item.size.z)],
                    "velocity_raw": float(item.velocity.x)})
        with self.lock:
            stamp = getattr(message.header, "stamp", None)
            self.objects = {"receipt_monotonic": time.monotonic(), "rows": rows,
                "source_sequence": int(message.header.seq),
                "source_stamp_ns": (int(stamp.secs)*1_000_000_000+int(stamp.nsecs)) if stamp is not None else 0,
                "publisher_node": (getattr(message, "_connection_header", None) or {}).get("callerid", "")}
            self.object_history.append(self.objects)
            self.object_history = self.object_history[-64:]
            self.object_messages += 1

    def object_snapshots(self, after):
        with self.lock:
            return copy.deepcopy([item for item in self.object_history if item["receipt_monotonic"] > after])

    def on_paths(self, message):
        selected = [marker for marker in message.markers
                    if marker.ns.startswith("candidate_paths_") and marker.action == 0
                    and marker.color.r >= .8 and marker.color.g <= .2 and len(marker.points) >= 3]
        if not selected:
            return
        marker = max(selected, key=lambda item: len(item.points))
        points = [[float(point.x), float(point.y)] for point in marker.points]
        with self.lock:
            self.selected_path = {"receipt_monotonic": time.monotonic(), "namespace": marker.ns,
                                  "points": points, "sha256": hashlib.sha256(json.dumps(points).encode()).hexdigest()}
            self.path_messages += 1

    def snapshot(self):
        with self.lock:
            return copy.deepcopy(self.objects), copy.deepcopy(self.selected_path)

    def on_ego(self, message):
        row = {"receipt_monotonic": time.monotonic(), "source_sequence": int(message.header.seq),
               "position_xyz": [float(message.position.x), float(message.position.y), float(message.position.z)],
               "yaw_deg": float(message.heading)}
        with self.lock:
            self.ego = row

    def on_command(self, message):
        row = {"receipt_monotonic": time.monotonic(),
               "publisher_node": (getattr(message, "_connection_header", None) or {}).get("callerid", ""),
               "longlCmdType": int(message.longlCmdType)}
        row.update({key: float(getattr(message, key)) for key in ("accel", "brake", "front_steer", "rear_steer")})
        with self.lock:
            self.commands.append(row)
            self.commands = self.commands[-12:]

    def startup_snapshot(self):
        with self.lock:
            return {"ego": copy.deepcopy(self.ego), "commands": copy.deepcopy(self.commands)}


class BorrowedTrialClient(MoraiGrpcClient):
    """Reuse the fixture world while isolating Runner cleanup by client key."""
    def __init__(self, fixture_client):
        self.__dict__.update(fixture_client.__dict__)
        self.client_key = "tunnel_static_trial_runner"
        self.fixture_client_key = fixture_client.client_key
        if self.client_key == self.fixture_client_key:
            raise RuntimeError("Trial Runner and fixture must use distinct client keys")
        self.world_reused = False

    def start_world(self, transform):
        if self.world is None:
            raise RuntimeError("Trial requires an already started owned fixture world")
        expected = TUNNEL_NORMAL_START
        xyz = live._xyz(transform.location, "trial requested START")
        if math.dist(xyz, expected.xyz) > .01:
            raise RuntimeError("Trial Runner requested a different START")
        # No StartSimulation or world reconstruction: those could reset an
        # actor while the preview helper still owns its stationary reference.
        self.world_reused = True

    def stop(self):
        raise RuntimeError("The outer fixture helper owns MORAI finalization")


@dataclass
class TrialContext:
    candidate: dict
    preview_report: dict
    global_cfg: dict
    output_dir: Path
    boundary: object
    monitor: object

    def __post_init__(self):
        self.references = {str(item["raw_actor_id"]): copy.deepcopy(item)
                           for item in self.preview_report.get("observed", [])}
        if len(self.references) != self.candidate.get("count") or "Ego" in self.references:
            raise RuntimeError("Trial requires every exact owned fixture observation")
        measured = self.preview_report.get("measured_route_screen", {})
        if (measured.get("status") != "nominal_path_found" or measured.get("path_found") is not True
                or measured.get("geometry_checks", {}).get("passed") is not True
                or not screening.provenance_matches(measured)):
            raise RuntimeError("Trial requires a current measured production Jamming path")
        if measured.get("road_boundary_policy") != self.boundary.policy:
            raise RuntimeError("Offline and live road/body policies differ")
        self.report = {"schema": "aimdrive_tunnel_static_drive_trial_v1", "mode": MODE,
            "status": "armed", "failure_reason": "", "expert_verified": False,
            "live_avoidance_verified": False, "physical_walls_verified": False,
            "mapped_entry_observed": False, "mapped_exit_observed": False,
            "ego_lane_return_verified": False, "goal_reached": False, "inventory_checks": 0,
            "trajectory_checks": 0, "observed_object_ids": [], "selected_path_observed": False,
            "road_boundary_policy": self.boundary.policy,
            "road_boundary_policy_sha256": self.boundary.policy_sha256,
            "fixture_raw_actor_ids": sorted(self.references), "collision_objects": [], "trajectory": []}
        self.previous_pose = None
        self.roll_pitch_limit_deg = float(self.preview_report.get("actual_ego_footprint", {}).get("max_abs_roll_pitch_deg", 3.))
        if not math.isfinite(self.roll_pitch_limit_deg) or not 0 < self.roll_pitch_limit_deg <= 3.:
            raise RuntimeError("Trial Ego projection requires an explicit roll/pitch limit up to three degrees")
        self.report["ego_roll_pitch_limit_deg"] = self.roll_pitch_limit_deg
        self.report["requested_pattern"] = self.candidate.get("placement", {}).get("requested_pattern")
        self.report["realized_pattern"] = None
        self.report["s_avoidance_verified"] = False
        self.geometry = None
        self.last_selected_hash = None
        self.saved_plan_keys = set()
        package = Path(__file__).resolve().parents[2]
        files = [Path(__file__), package/"adapters/expert_process.py", package/"adapters/expert_controller.py",
                 package/"core/full_loop_drive_base.py", Path(__file__).with_name("scenario.py"),
                 Path(__file__).with_name("expert.py"), Path(__file__).with_name("tunnel_static_startup.py")]
        expert_root = self.global_cfg.get("paths", {}).get("expert_root")
        if expert_root:
            files.append(Path(expert_root)/"src/expert/src/network/ros_manager.py")
        self.runtime_sources = {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
        self.report["runtime_source_sha256"] = self.runtime_sources
        from shapely.geometry import mapping
        with getattr(self.boundary, "_predicate_lock", nullcontext()):
            self.report["road_boundary_geometry"] = mapping(self.boundary._polygon)
        self.persist()

    def persist(self):
        self.report["owned_actor_geometry"] = copy.deepcopy(list(self.references.values()))
        _write(self.output_dir / "drive_trial.json", self.report)

    def _objects(self):
        return [SimpleNamespace(position=item["center_xyz"][:2], heading_deg=item["heading_deg"],
                                size_x=item["length_m"], size_y=item["width_m"])
                for item in self.references.values()]

    def prepare(self, context):
        self.geometry = validate_candidate(self.candidate, context.map_loader)
        self.route_points = context.route_points
        self.original_lane_polygon, lane_source = mapped_original_lane(context.map_loader)
        self.report["original_lane_geometry"] = lane_source
        self.entry_s = project_polyline(self.route_points, *self.geometry["own_points"][0][:2])[0]
        self.exit_s = project_polyline(self.route_points, *self.geometry["own_points"][-1][:2])[0]
        self.report["route_length_m"] = context.route_length_m
        self.report["mapped_entry_route_s_m"] = self.entry_s
        self.report["mapped_exit_route_s_m"] = self.exit_s
        self.persist()

    def bind_static_surfaces(self, context, *, timeout_sec=3., monotonic_fn=None, sleep_fn=None):
        """Pin fresh native metadata without treating it as collision geometry."""
        now = monotonic_fn or time.monotonic
        sleep = sleep_fn or time.sleep
        if not math.isfinite(timeout_sec) or not 0 < timeout_sec <= 5.:
            raise ValueError("Static native metadata capture requires a bounded timeout up to five seconds")
        started = now()
        deadline = started+timeout_sec
        statics = {identity: row for identity, row in self.references.items()
                   if row["actor_kind"] == "obstacle"}
        if not statics:
            return
        planned = self.candidate.get("obstacles", [])
        observed = self.preview_report.get("observed", [])
        if len(planned) != len(observed) or len(planned) != len(self.references):
            raise RuntimeError("Static surface contract requires every original candidate placement")
        placements = {}
        for placement, reference in zip(planned, observed):
            identity = str(reference["raw_actor_id"])
            if (identity not in self.references or identity in placements
                    or placement.get("actor_kind") != reference["actor_kind"]
                    or placement.get("model") != reference["model"]):
                raise RuntimeError("Static surface original placement identity/model differs from the observed scene")
            placements[identity] = placement
        catalog = load_static_models(allowed_models={row["model"] for row in statics.values()})
        for placement in placements.values():
            source = placement.get("geometry_source", {})
            runtime_path = source.get("runtime_catalog_path")
            if runtime_path:
                if hashlib.sha256(Path(runtime_path).read_bytes()).hexdigest() != source.get("runtime_catalog_sha256"):
                    raise RuntimeError("Static runtime catalog source hash changed")
                catalog.update(load_static_models(allowed_models={placement["model"]}, runtime_catalog_path=runtime_path))
        inventory = live._inventory(context.grpc, self.report)
        stub = context.grpc.world._sim_adapter._actor_stub
        for identity, reference in statics.items():
            placement = placements[identity]
            model = catalog.get(reference["model"])
            if (model is None or model.get("geometry_verified") is not True
                    or placement.get("geometry_source") != model["geometry_source"]
                    or normalize_footprint(placement.get("planned_bbox")) != normalize_footprint(model)):
                raise RuntimeError("Static surface candidate does not match the current sealed catalog: " + identity)
            state = inventory.get(identity)
            if state is None or live._id(state) != identity:
                raise RuntimeError("Static surface contract requires the exact current owned actor: " + identity)
            live._stationary(live._observation(state), reference, "Static surface fixture " + identity)
            if stub.GetOptionName(state.actor_info, timeout=.5).value != reference["model"]:
                raise RuntimeError("Static surface fixture model changed: " + identity)
            if now() >= deadline:
                raise RuntimeError("Fresh native static metadata capture timed out before binding")
        from proto.morai.actor.actor_enum_pb2 import GEAR_MODE_P
        ego_id = str(context.grpc.get_ego().get_id())
        ego_state = inventory.get(ego_id)
        from .tunnel_static_startup import common_brake_hold_enabled, common_stopped_readback
        if ego_state is None:
            raise RuntimeError("Native static metadata requires the exact stopped Ego")
        if common_brake_hold_enabled(context.grpc):
            common_stopped_readback(context.grpc, self.preview_report.get("ego_observed"))
        elif int(ego_state.vehicle_state.gear_mode) != int(GEAR_MODE_P):
            raise RuntimeError("Native static metadata must be captured while Ego is parked in P")
        ego_reference = self.preview_report.get("ego_observed") or live._observation(ego_state)
        captured = {identity: [] for identity in statics}
        cursor = started
        last_sequence, last_stamp, publisher = None, None, None
        while any(len(frames) < 3 for frames in captured.values()):
            current_time = now()
            if current_time >= deadline:
                raise RuntimeError("Fresh native static metadata capture timed out; every owned fixture needs three verified ROS frames")
            frames = self.monitor.object_snapshots(cursor)
            for frame in frames:
                current_time = now()
                receipt = frame.get("receipt_monotonic")
                sequence, stamp = frame.get("source_sequence"), frame.get("source_stamp_ns")
                caller = frame.get("publisher_node")
                if (not isinstance(receipt, (int, float)) or not math.isfinite(receipt)
                        or not cursor < receipt <= current_time or current_time-receipt > .5
                        or not isinstance(sequence, int) or isinstance(sequence, bool) or sequence < 0
                        or not isinstance(stamp, int) or isinstance(stamp, bool) or stamp < 0
                        or not isinstance(caller, str) or not caller):
                    raise RuntimeError("Native static metadata frame is stale or lacks its actual source identity")
                publisher_changed = publisher is not None and caller != publisher
                sequence_regressed = last_sequence is not None and sequence < last_sequence
                stamp_regressed = last_stamp is not None and stamp != 0 and last_stamp != 0 and stamp < last_stamp
                if publisher_changed or sequence_regressed or stamp_regressed:
                    reason = ("publisher_changed" if publisher_changed else
                        "source_sequence_and_stamp_regression" if sequence_regressed and stamp_regressed else
                        "source_sequence_regression" if sequence_regressed else "source_stamp_regression")
                    failure = {"schema": "aimdrive_native_static_metadata_failure_v1",
                        "stage": "preflight_static_binding", "actor_kind": "obstacle", "reason": reason,
                        "owned_actor_ids": sorted(statics), "previous_publisher": publisher,
                        "current_publisher": caller, "previous_source_sequence": last_sequence,
                        "current_source_sequence": sequence, "previous_source_stamp_ns": last_stamp or 0,
                        "current_source_stamp_ns": stamp,
                        "retryable_same_publisher_ordering": not publisher_changed}
                    self.preview_report["native_metadata_failure"] = copy.deepcopy(failure)
                    self.report["native_metadata_failure"] = copy.deepcopy(failure)
                    self.persist()
                    raise RuntimeError("Native static metadata publisher changed during binding" if publisher_changed
                        else "Native static metadata sequence/stamp regressed with the same publisher")
                sequence_duplicate = last_sequence is not None and sequence == last_sequence
                stamp_duplicate = last_stamp is not None and stamp != 0 and stamp == last_stamp
                previous_sequence, previous_stamp = last_sequence, last_stamp
                # Advance every observed source watermark, including skipped
                # duplicates. A later decrease must not be hidden by skipping
                # a frame whose sequence progressed but simulation time did not.
                cursor, last_sequence, publisher = receipt, sequence, caller
                if stamp:
                    last_stamp = stamp
                if sequence_duplicate or stamp_duplicate:
                    reason = ("source_sequence_and_stamp_duplicate" if sequence_duplicate and stamp_duplicate else
                        "source_sequence_duplicate" if sequence_duplicate else "source_stamp_duplicate")
                    duplicates = copy.deepcopy(self.report.get("native_metadata_duplicates", {
                        "schema": "aimdrive_native_static_metadata_duplicates_v1",
                        "stage": "preflight_static_binding", "status": "skipped",
                        "source_identity_stable": True, "verified_frames_added": 0,
                        "skipped_frame_count": 0, "skip_reasons": {}, "frames": []}))
                    duplicates["skipped_frame_count"] += 1
                    duplicates["skip_reasons"][reason] = duplicates["skip_reasons"].get(reason, 0)+1
                    duplicates["frames"].append({"reason": reason, "publisher_node": caller,
                        "receipt_monotonic": receipt,
                        "previous_source_sequence": previous_sequence, "current_source_sequence": sequence,
                        "previous_source_stamp_ns": previous_stamp or 0, "current_source_stamp_ns": stamp})
                    # Match the monitor's bounded history while retaining the
                    # total skipped count. Raw ROS messages remain unchanged.
                    duplicates["frames"] = duplicates["frames"][-64:]
                    self.report["native_metadata_duplicates"] = duplicates
                    self.preview_report["native_metadata_duplicates"] = copy.deepcopy(duplicates)
                    continue
                fresh = live._inventory(context.grpc, self.report)
                if now() >= deadline:
                    raise RuntimeError("Fresh native static metadata capture timed out during bounded SDK verification")
                current_ego = fresh.get(ego_id)
                if current_ego is None:
                    raise RuntimeError("Native static metadata capture lost the exact Ego")
                if common_brake_hold_enabled(context.grpc):
                    common_stopped_readback(context.grpc, ego_reference)
                elif int(current_ego.vehicle_state.gear_mode) != int(GEAR_MODE_P):
                    raise RuntimeError("Native static metadata capture lost parked Ego P gear")
                live._stationary(live._observation(current_ego), ego_reference, "Native metadata Ego")
                for identity, reference in statics.items():
                    state = fresh.get(identity)
                    if state is None or live._id(state) != identity:
                        raise RuntimeError("Native static metadata actor disappeared: " + identity)
                    live._stationary(live._observation(state), reference, "Native metadata fixture " + identity)
                    matches = [row for row in frame["rows"] if row.get("raw_actor_id") == identity]
                    if not matches:
                        continue
                    if len(matches) != 1:
                        raise RuntimeError("Native static metadata contains duplicate owned actor IDs: " + identity)
                    row = matches[0]
                    values = row.get("size_xyz", [])
                    position = row.get("position_xyz", [])
                    heading, velocity = row.get("heading_deg"), row.get("velocity_raw")
                    if (row.get("actor_kind") != "obstacle" or row.get("name") != reference["model"]
                            or len(position) != 3 or len(values) != 3
                            or any(not math.isfinite(value) for value in position)
                            or any(not math.isfinite(value) or value <= 0. for value in values)
                            or not isinstance(heading, (int, float)) or not math.isfinite(heading)
                            or not isinstance(velocity, (int, float)) or not math.isfinite(velocity)
                            or abs(velocity) > live.MAX_SPEED_RAW
                            or math.dist(position[:2], reference["position_xyz"][:2]) > .15
                            or live._angle_distance(heading, reference["rotation_xyz_deg"][2]) > 2.):
                        raise RuntimeError("Native static metadata does not match the exact stationary SDK actor: " + identity)
                    evidence = {key: frame[key] for key in (
                        "receipt_monotonic", "source_sequence", "source_stamp_ns", "publisher_node")}
                    evidence["row"] = copy.deepcopy(row)
                    if captured[identity] and any(abs(value-baseline) > .01 for value, baseline in zip(
                            values, captured[identity][0]["row"]["size_xyz"])):
                        raise RuntimeError("Native static metadata dimensions changed between verified frames: " + identity)
                    if len(captured[identity]) < 3:
                        captured[identity].append(evidence)
                if all(len(items) == 3 for items in captured.values()):
                    break
            if any(len(frames) < 3 for frames in captured.values()):
                sleep(min(.02, max(0., deadline-now())))
        for identity, reference in statics.items():
            values = list(captured[identity][0]["row"]["size_xyz"])
            validated = now()
            if validated >= deadline or any(not 0 <= validated-frame["receipt_monotonic"] <= .5
                                            for frame in captured[identity]):
                raise RuntimeError("Native static metadata evidence exceeded its fresh bounded capture window: " + identity)
            contract = {"source": "verified_native_ROS_static_metadata", "raw_actor_id": identity,
                        "model": reference["model"], "size_lw_m": values[:2], "size_lwh_m": values,
                        "binding_monotonic": started, "validated_monotonic": validated,
                        "geometry_source": copy.deepcopy(catalog[reference["model"]]["geometry_source"]),
                        "native_frames": captured[identity]}
            existing = reference.get("static_surface_contract")
            if existing is not None:
                if ({key: value for key, value in existing.items() if key not in (
                        "native_frames", "binding_monotonic", "validated_monotonic")} != {
                        key: value for key, value in contract.items() if key not in (
                            "native_frames", "binding_monotonic", "validated_monotonic")}
                        or existing["native_frames"][0]["publisher_node"] != publisher):
                    raise RuntimeError("Bound native static metadata contract changed: " + identity)
            else:
                reference["static_surface_contract"] = contract
        self.report["static_surface_contracts"] = {
            identity: copy.deepcopy(reference["static_surface_contract"])
            for identity, reference in statics.items()}
        self.persist()

    def bind_vehicle_specs(self, context):
        """Pin native ROS dimensions separately from the observed collider."""
        vehicles = {identity: row for identity, row in self.references.items()
                    if row["actor_kind"] == "vehicle"}
        if not vehicles:
            return
        inventory = live._inventory(context.grpc, self.report)
        stub = context.grpc.world._sim_adapter._actor_stub
        for identity, reference in vehicles.items():
            state = inventory.get(identity)
            if state is None or live._id(state) != identity:
                raise RuntimeError("Vehicle spec requires the exact current owned actor: " + identity)
            live._stationary(live._observation(state), reference, "Vehicle spec fixture " + identity)
            model = stub.GetOptionName(state.actor_info, timeout=.5).value
            if model != reference["model"]:
                raise RuntimeError("Vehicle spec fixture model changed: " + identity)
            spec = stub.GetVehicleSpec(state.actor_info, timeout=.5)
            values = [float(spec.length), float(spec.width), float(spec.height)]
            if any(not math.isfinite(value) or value <= 0. for value in values):
                raise RuntimeError("Invalid native vehicle specification: " + identity)
            # Native ROS height is nominal vehicle metadata, rather than the
            # renderer/collider height (DEFAULT_SEDAN reports 1.5 vs 1.3475 m).
            # Keep it bound and checked against ROS, while the physical box
            # remains independently measured and guarded. Planar dimensions
            # must still fit the conservative collision envelope.
            maximum = [reference["length_m"], reference["width_m"]]
            if any(value > limit+.1 for value, limit in zip(values[:2], maximum)):
                raise RuntimeError("Native vehicle specification exceeds the conservative observed envelope: " + identity)
            contract = {"source": "bounded_GetVehicleSpec", "raw_actor_id": identity,
                        "model": reference["model"], "size_lwh_m": values}
            existing = reference.get("vehicle_spec_contract")
            if existing is not None and existing != contract:
                raise RuntimeError("Bound native vehicle specification changed: " + identity)
            reference["vehicle_spec_contract"] = contract
        self.report["vehicle_spec_contracts"] = {
            identity: copy.deepcopy(reference["vehicle_spec_contract"])
            for identity, reference in vehicles.items()}
        self.persist()

    def inspect(self, context, elapsed, ego_state=None):
        import numpy as np
        from shapely.geometry import Polygon
        self.report["last_guard_elapsed_sec"] = float(elapsed)
        self.report["last_guard_stage"] = "inventory"
        inventory = live._inventory(context.grpc, self.report)
        ego_id = str(context.grpc.get_ego().get_id())
        if ego_id not in inventory:
            raise RuntimeError("Trial global inventory is missing Ego")
        current = live._observation(inventory[ego_id])
        self.report["last_ego_observation"] = current
        self.report["last_guard_stage"] = "ego_roll_pitch"
        tilt = [live._angle_distance(value, 0.) for value in current["rotation_xyz_deg"][:2]]
        self.report["last_ego_roll_pitch_abs_deg"] = tilt
        if max(tilt) > self.roll_pitch_limit_deg:
            raise RuntimeError("Actual Ego roll/pitch exceeds the three-degree conservative projection contract")
        self.report["last_fixture_observations"] = {}
        self.report["last_guard_stage"] = "fixture_stationarity"
        for identity, reference in self.references.items():
            if identity not in inventory:
                raise RuntimeError("Owned trial fixture disappeared: " + identity)
            observed = live._observation(inventory[identity])
            self.report["last_fixture_observations"][identity] = observed
            live._stationary(observed, reference, "Trial fixture " + identity)
            if live._model(context.grpc, inventory[identity]) != reference["model"]:
                raise RuntimeError("Trial fixture model changed: " + identity)
        self.report["last_guard_stage"] = "foreign_occupancy"
        live._occupancy(inventory, ego_id, set(self.references), self.geometry)
        self.report["last_guard_stage"] = "reported_collision"
        collisions = list(inventory[ego_id].vehicle_state.collision_objects)
        if collisions:
            self.report["collision_objects"] = [str(item) for item in collisions]
            raise RuntimeError("Collision during tunnel static trial: " + str(collisions))
        x, y = current["position_xyz"][:2]
        yaw = math.radians(current["rotation_xyz_deg"][2])
        poses = [(x, y, yaw)]
        if self.previous_pose is not None:
            px, py, pyaw = self.previous_pose
            delta = (yaw-pyaw+math.pi) % (2*math.pi)-math.pi
            count = max(1, math.ceil(math.hypot(x-px, y-py)/.10), math.ceil(abs(delta)/math.radians(1.)))
            if count > 100:
                raise RuntimeError("Ego pose jump exceeds the bounded swept-body guard")
            poses = [(px+(x-px)*i/count, py+(y-py)*i/count, pyaw+delta*i/count) for i in range(1, count+1)]
        xy = np.array([item[:2] for item in poses])
        headings = np.array([item[2] for item in poses])
        self.report["last_guard_stage"] = "policy_body_road"
        policy_corners = self.boundary.corners_world(xy, headings, margin=.15)
        road_mask = self.boundary.mask_world(xy, headings, margin=.15)
        raw_box = current["raw_bbox_normalized"]
        raw_corners = footprint_corners(raw_box["center_xyz"], raw_box["width_m"], raw_box["length_m"], raw_box["heading_deg"])
        inflated_corners = footprint_corners(raw_box["center_xyz"], raw_box["width_m"]+.30,
                                            raw_box["length_m"]+.30, raw_box["heading_deg"])
        raw_polygon = Polygon(inflated_corners)
        delta = np.asarray(raw_corners)-np.asarray(current["position_xyz"][:2])
        forward, left = np.array([math.cos(yaw), math.sin(yaw)]), np.array([-math.sin(yaw), math.cos(yaw)])
        along, lateral = delta @ forward, delta @ left
        self.report["last_geometry_check"] = {
            "swept_ego_poses_xy_yaw_rad": poses,
            "policy_body_corners_world": policy_corners.tolist(),
            "policy_road_mask": road_mask.tolist(),
            "raw_bbox_corners_world": raw_corners,
            "raw_bbox_corners_with_margin_world": inflated_corners,
            "raw_bbox_road_passed": bool(self.boundary.covers_geometry(raw_polygon)),
            "raw_bbox_without_margin_road_passed": bool(self.boundary.covers_geometry(Polygon(raw_corners))),
            "raw_bbox_outside_area_m2": float(self.boundary.outside_area_geometry(raw_polygon)),
            "raw_heading_difference_deg": live._angle_distance(raw_box["heading_deg"], current["rotation_xyz_deg"][2]),
            "observed_raw_bbox_actor_envelope": {"rear_m": max(0., float(-along.min())),
                "front_m": max(0., float(along.max())), "width_m": float(2*np.abs(lateral).max())},
            "road_tracking_margin_m": .15,
        }
        if not road_mask.all():
            raise RuntimeError("Actual Ego body left the shared mapped road boundary")
        self.report["last_guard_stage"] = "policy_body_collision"
        collision_mask = self.boundary.collision_mask_world(xy, headings, self._objects(), margin=.12)
        self.report["last_geometry_check"]["policy_collision_mask"] = collision_mask.tolist()
        if not collision_mask.all():
            raise RuntimeError("Actual swept Ego body intersects a conservative fixture envelope")
        self.report["last_guard_stage"] = "raw_bbox_road"
        if not self.boundary.covers_geometry(raw_polygon):
            raise RuntimeError("Actual returned Ego bounding box leaves the mapped road")
        self.report["last_guard_stage"] = "raw_bbox_collision"
        if any(boxes_overlap(raw_box, fixture, margin_m=.05) for fixture in self.references.values()):
            raise RuntimeError("Actual returned Ego box overlaps a fixture envelope")
        self.previous_pose = (x, y, yaw)
        station, lateral, _ = project_polyline(self.route_points, x, y)
        self.report["mapped_entry_observed"] |= station >= self.entry_s
        self.report["mapped_exit_observed"] |= station >= self.exit_s
        self.report["last_route_s_m"] = station
        self.report["last_lateral_m"] = lateral
        self.report["last_ego_xyz"] = current["position_xyz"]
        self.report["trajectory_checks"] += 1
        self.report["trajectory"].append({"elapsed_sec": float(elapsed), "position_xyz": current["position_xyz"],
            "yaw_deg": current["rotation_xyz_deg"][2], "route_s_m": station, "lateral_m": lateral})
        self.report["last_guard_stage"] = "ros_evidence"
        self.inspect_ros(current, allow_missing=elapsed == 0.)
        self.report["last_guard_stage"] = "passed"
        from .tunnel_static_startup import observe_common_startup
        observe_common_startup(context, self, current)
        self.persist()
        return current

    def inspect_ros(self, ego, allow_missing=False):
        objects, selected = self.monitor.snapshot()
        self.report["last_ros_objects"] = objects
        now = time.monotonic()
        if objects is None or now-objects["receipt_monotonic"] > 1.:
            if allow_missing:
                return
            raise RuntimeError("Actual /Object_topic is missing or stale during trial")
        observed_ids = set(self.report["observed_object_ids"])
        for identity, reference in self.references.items():
            if math.dist(ego["position_xyz"][:2], reference["position_xyz"][:2]) > 80.:
                continue
            matches = [item for item in objects["rows"] if item["raw_actor_id"] == identity]
            if len(matches) != 1 or matches[0]["actor_kind"] != reference["actor_kind"]:
                raise RuntimeError("ROS Object_topic does not identify the exact trial fixture: " + identity)
            row = matches[0]
            if (math.dist(row["position_xyz"][:2], reference["position_xyz"][:2]) > .5
                    or live._angle_distance(row["heading_deg"], reference["rotation_xyz_deg"][2]) > 3.):
                raise RuntimeError("ROS fixture pose does not match bounded gRPC readback: " + identity)
            raw = reference["raw_bbox_normalized"]
            spec = reference.get("vehicle_spec_contract") if reference["actor_kind"] == "vehicle" else None
            surface = reference.get("static_surface_contract") if reference["actor_kind"] == "obstacle" else None
            if spec is not None:
                if (spec.get("source") != "bounded_GetVehicleSpec" or spec.get("raw_actor_id") != identity
                        or spec.get("model") != reference["model"] or row.get("name") != reference["model"]):
                    raise RuntimeError("ROS fixture does not match the bound native vehicle specification: " + identity)
                actual, expected = row["size_xyz"], spec["size_lwh_m"]
                if (len(actual) != 3 or len(expected) != 3
                        or any(not math.isfinite(value) or value <= 0. for value in actual+expected)
                        or any(abs(value-native) > .1 for value, native in zip(actual, expected))):
                    raise RuntimeError("ROS fixture size changed from its bound native vehicle specification: " + identity)
            elif surface is not None:
                if (surface.get("source") not in ("sealed_lidar_static_catalog", "verified_native_ROS_static_metadata")
                        or surface.get("raw_actor_id") != identity
                        or surface.get("model") != reference["model"] or row.get("name") != reference["model"]):
                    raise RuntimeError("ROS fixture does not match the bound sealed static surface: " + identity)
                native = surface["source"] == "verified_native_ROS_static_metadata"
                if native and objects.get("publisher_node") != surface["native_frames"][0]["publisher_node"]:
                    raise RuntimeError("ROS static metadata publisher changed from its bound native source: " + identity)
                actual, expected = (row["size_xyz"], surface["size_lwh_m"]) if native else (
                    row["size_xyz"][:2], surface["size_lw_m"])
                if (len(actual) != (3 if native else 2) or len(expected) != (3 if native else 2)
                        or any(not math.isfinite(value) or value <= 0. for value in actual+expected)
                        or any(abs(value-sealed) > .1 for value, sealed in zip(actual, expected))):
                    raise RuntimeError("ROS fixture size changed from its bound sealed static surface: " + identity)
            elif abs(row["size_xyz"][0]-raw["length_m"]) > .15 or abs(row["size_xyz"][1]-raw["width_m"]) > .15:
                raise RuntimeError("ROS fixture size axes disagree with the actual gRPC contract: " + identity)
            observed_ids.add(identity)
        self.report["observed_object_ids"] = sorted(observed_ids)
        if selected is not None and now-selected["receipt_monotonic"] <= 1.:
            self.report["selected_path_observed"] = True
            self.report["selected_path_updates"] = self.report.get("selected_path_updates", 0)+1
            self.report["last_selected_path_summary"] = {"namespace": selected["namespace"],
                "sha256": selected["sha256"], "point_count": len(selected["points"]),
                "first_xy": selected["points"][0], "last_xy": selected["points"][-1]}
            try:
                diagnostics = json.loads((self.output_dir/"planner_diagnostics.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                diagnostics = {}
            static = diagnostics.get("last_result", {}).get("debug_details", {}).get("static_avoidance", {})
            identity = {"namespace": selected["namespace"], "selected_routes": static.get("selected_routes", []),
                        "plan_count": static.get("plan_count"), "scene_geometry_sha256": diagnostics.get("scene_geometry_sha256")}
            key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
            if diagnostics.get("validated_path_seen") is True and key not in self.saved_plan_keys:
                self.saved_plan_keys.add(key)
                self.report["distinct_selected_plans"] = len(self.saved_plan_keys)
                if len(self.report.get("selected_paths", [])) < 8:
                    self.report.setdefault("selected_paths", []).append(dict(selected, plan_identity=identity))
                else:
                    self.report["additional_full_paths_omitted"] = True
                road = static.get("road_boundary", {})
                if (self.geometry is not None and static.get("path_validated") is True
                        and road.get("passed") is True and road.get("policy_sha256") == self.boundary.policy_sha256):
                    shape = classify_lane_maneuver(selected["points"], self.geometry)
                    self.report.setdefault("validated_selected_path_maneuvers", []).append(
                        {"path_sha256": selected["sha256"], "plan_identity": identity, "maneuver": shape})
                    self.report["validated_selected_path_maneuvers"] = self.report["validated_selected_path_maneuvers"][-32:]

    def finish(self, context):
        current = self.inspect(context, self.report.get("trajectory", [{}])[-1].get("elapsed_sec", 0.)+.001)
        lane_checks = lane_return_checks(self.boundary, self.original_lane_polygon, current)
        self.report["lane_return_checks"] = lane_checks
        actual_shape = classify_lane_maneuver([row["position_xyz"] for row in self.report["trajectory"]], self.geometry)
        self.report["actual_trajectory_maneuver"] = actual_shape
        self.report["realized_pattern"] = actual_shape["pattern"]
        _, tangent = route_pose(self.route_points, self.report["last_route_s_m"])
        heading_error = live._angle_distance(current["rotation_xyz_deg"][2], math.degrees(tangent))
        self.report["final_heading_error_deg"] = heading_error
        conditions = {"mapped_entry_observed": self.report["mapped_entry_observed"] is True,
                      "mapped_exit_observed": self.report["mapped_exit_observed"] is True,
                      "minimum_route_progress": self.report["last_route_s_m"] >= .95*context.route_length_m,
                      "center_rejoined_original_route": abs(self.report["last_lateral_m"]) <= .6,
                      "whole_body_in_original_lane": lane_checks["whole_body_in_original_lane"],
                      "heading_aligned": heading_error <= 8.,
                      "all_owned_ros_objects_verified": set(self.report["observed_object_ids"]) == set(self.references),
                      "selected_path_observed": self.report["selected_path_observed"] is True,
                      "collision_free": not bool(self.report["collision_objects"]),
                      "final_fixture_and_road_checks_passed": self.report.get("last_guard_stage") == "passed"}
        if self.report.get("startup_handoff", {}).get("startup_policy") == "common_auto_d_brake":
            conditions["current_expert_startup_verified"] = self.report["startup_handoff"].get("fresh_inputs_verified") is True
        diagnostics_path = self.output_dir / "planner_diagnostics.json"
        try:
            diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            diagnostics = {}
            self.report["planner_diagnostics_read_error"] = str(error)
        self.report["planner_diagnostics"] = diagnostics
        expected_scene_hash = hashlib.sha256(json.dumps(list(self.references.values()), sort_keys=True,
            separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        conditions.update(actual_jamming_recorded_path_validated=diagnostics.get("validated_path_seen") is True and bool(diagnostics.get("selected_routes")),
                          scene_geometry_matches=diagnostics.get("scene_geometry_sha256") == expected_scene_hash,
                          planner_boundary_policy_matches=diagnostics.get("road_boundary_policy_sha256") == self.boundary.policy_sha256,
                          planner_recording_and_policy_sources_current=screening.provenance_matches(self.preview_report["measured_route_screen"]))
        try:
            conditions["runtime_handoff_sources_current"] = all(hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
                                                                for path, digest in self.runtime_sources.items())
        except OSError:
            conditions["runtime_handoff_sources_current"] = False
        self.report["completion_conditions"] = conditions
        failed = [name for name, passed in conditions.items() if not passed]
        self.report["failed_completion_conditions"] = failed
        if failed:
            raise RuntimeError("Tunnel trial completion conditions failed: " + ", ".join(failed))
        self.report.update(status="passed", goal_reached=True, ego_lane_return_verified=True,
                           expert_verified=True, live_avoidance_verified=True, completed=True,
                           collision_free=True, returned_to_original_lane=True, fixture_stationary=True)
        planned_s = any(row["maneuver"]["s_maneuver_observed"]
                        for row in self.report.get("validated_selected_path_maneuvers", []))
        self.report["s_avoidance_verified"] = actual_shape["s_maneuver_observed"] and planned_s
        self.report["pattern_classification"] = {"requested_pattern": self.report["requested_pattern"],
            "actual_trajectory": actual_shape, "validated_selected_path_s_observed": planned_s,
            "s_avoidance_verified": self.report["s_avoidance_verified"],
            "basis": "bounded gRPC Ego lane transitions plus production-validated selected paths; images do not classify maneuvers"}
        self.persist()

    def fail(self, error):
        reason = str(error) or type(error).__name__
        primary = self.report.get('failure_reason') if self.report.get('status') == 'failed' else None
        if primary and primary != reason:
            self.report.setdefault('subsequent_failures', []).append(reason)
        self.report.update(status="failed", failure_reason=primary or reason,
                           expert_verified=False, live_avoidance_verified=False, s_avoidance_verified=False)
        self.persist()


def required(context):
    trial = getattr(context, "tunnel_static_trial_context", None)
    if not isinstance(trial, TrialContext):
        raise RuntimeError("Use run_tunnel_static_trial with a measured, screened owned fixture; bare tunnel static mode cannot drive")
    return trial


def request_normal_goal_stop(context, *, elapsed, ego_state, current_s, remaining_s):
    """Declare GOAL braking intent without approving the completed trial."""
    trial = required(context)
    report = trial.report
    process = getattr(getattr(context, 'expert_controller', None), 'process', None)
    if (report.get('status') != 'running' or report.get('last_guard_stage') != 'passed'
            or report.get('collision_objects') or ego_state.get('collision_objects')
            or report.get('termination_hard_failures') or report.get('failed_completion_conditions')
            or process is None or process.poll() is not None):
        raise RuntimeError('Normal GOAL braking requires an active guarded collision-free trial')
    report['normal_goal_stop_intent'] = {
        'pending': True, 'expert_pid': int(process.pid),
        'requested_monotonic': time.monotonic(), 'elapsed_sec': float(elapsed),
        'route_s_m': float(current_s), 'remaining_route_m': float(remaining_s),
        'completion_verification_pending': True,
    }
    trial.persist()


def cancel_normal_goal_stop(context, error):
    """A failed or interrupted timeline must use the emergency stop path."""
    current = getattr(context, 'tunnel_static_trial_context', None)
    intent = getattr(current, 'report', {}).get('normal_goal_stop_intent')
    if isinstance(intent, dict) and intent.get('pending') is True:
        intent.update(pending=False, cancelled_reason=str(error) or type(error).__name__)
        current.persist()


def immediate_hold(context):
    """Disable external Ego input and park before waiting for Expert teardown."""
    if not enabled(context) or getattr(context.grpc, "world", None) is None:
        return None
    from .tunnel_static_startup import (
        common_brake_hold_enabled, block_common_input, hold_common_brake, common_stopped_readback,
    )
    if common_brake_hold_enabled(context.grpc):
        from .tunnel_static_startup import finish_common_brake
        current = getattr(context, 'tunnel_static_trial_context', None)
        report = getattr(current, 'report', {})
        outcome = {"started_monotonic": time.monotonic(), "stop_method": "common_auto_d_ros_brake",
                   "control_scope": "owned_expert_stopped_and_exclusive_ros_brake_hold", "hard_failures": []}
        controller = getattr(context, 'expert_controller', None)
        process = getattr(controller, 'process', None)
        intent = report.get('normal_goal_stop_intent')
        normal_goal = (isinstance(intent, dict) and intent.get('pending') is True
            and process is not None and intent.get('expert_pid') == process.pid
            and report.get('status') == 'running' and report.get('last_guard_stage') == 'passed'
            and not report.get('collision_objects') and not report.get('termination_hard_failures')
            and not report.get('failed_completion_conditions'))
        if isinstance(intent, dict) and intent.get('pending') is True:
            intent.update(pending=False, consumed_monotonic=time.monotonic(),
                          consumed_as_normal_goal=bool(normal_goal))
        outcome['normal_goal_stop'] = bool(normal_goal)
        emergency = (process is not None and process.poll() is None
                     and report.get('status') != 'passed' and not normal_goal)
        if emergency:
            try:
                outcome['input_block_phase'] = block_common_input(context.grpc)
            except Exception as error:
                if getattr(error, 'readback_pending', False) and getattr(error, 'commands_accepted', False):
                    outcome['input_block_phase'] = {'commands_accepted':True, 'external_input_disabled':False,
                        'observation_pending':True, 'control_mode':getattr(error, 'control_mode', None)}
                    outcome['input_block_observation_pending'] = str(error)
                else:
                    outcome['hard_failures'].append('input_block: '+str(error))
        try:
            context.stop_expert_controller()
        except Exception as error:
            outcome['hard_failures'].append('stop_owned_expert: '+str(error))
        stop_evidence = {}
        outcome['final_brake_checks'] = stop_evidence
        try:
            stopped = finish_common_brake(context.grpc, stop_evidence)
            outcome.update(observed_control_mode=stopped['control_mode'], observed_gear_mode=stopped['gear_mode'],
                observed_speed_norm_raw=stopped['speed_norm_raw'], observed_position_xyz=stopped['position_xyz'],
                observed_brake=stopped['brake'], observed_throttle=stopped['throttle'],
                observed_parked_and_zero_speed=True, brake_hold_verified=True,
                control_publishers_absent=True, external_input_disabled=False)
        except Exception as error:
            outcome['readback_error'] = str(error)
            outcome['hard_failures'].append('final_brake: '+str(error))
        outcome['final_verified'] = outcome.get('brake_hold_verified') is True and not outcome['hard_failures']
        # Compatibility field: this policy proves exclusive ROS control and
        # physical stop, rather than using KEYBOARD/P as the final actuator state.
        outcome['accepted_and_control_blocked'] = outcome['final_verified']
        outcome['finished_monotonic'] = time.monotonic()
        if current is not None:
            report.setdefault('immediate_hold_history', []).append(copy.deepcopy(outcome))
            report.setdefault('termination_hard_failures', []).extend(outcome['hard_failures'])
            report['immediate_hold'] = outcome
            report['final_stop_verified'] = outcome['final_verified']
            report['all_immediate_hold_calls_accepted'] = not report['termination_hard_failures'] and outcome['final_verified']
        return outcome
    from proto.morai.actor.actor_enum_pb2 import VEHICLE_CONTROL_KEYBOARD, GEAR_MODE_P
    from proto.morai.actor.actor_set_pb2 import VehicleControlModeParam, SetVelocityParam, VehicleGear
    from proto.morai.common.enum_pb2 import STATUS_CODE_SUCCESS
    ego = context.grpc.get_ego()
    info = ego.get_object_info()
    stub = context.grpc.world._sim_adapter._actor_stub
    request = VehicleControlModeParam()
    request.actor_info.CopyFrom(info)
    request.mode = VEHICLE_CONTROL_KEYBOARD
    velocity = SetVelocityParam()
    velocity.actor_info.CopyFrom(info)
    velocity.velocity = 0.
    gear = VehicleGear()
    gear.actor_info.CopyFrom(info)
    gear.gear = GEAR_MODE_P
    outcome = {"started_monotonic": time.monotonic(), "rpc_timeout_sec": .5}
    for label, operation, argument in (("external_input_disabled", stub.SetVehicleControlMode, request),
                                       ("velocity_zero", stub.SetVelocity, velocity),
                                       ("gear_p", stub.SetVehicleGear, gear)):
        try:
            result = operation(argument, timeout=.5)
            outcome[label] = result is not None and result.status == STATUS_CODE_SUCCESS
        except Exception as error:
            outcome[label] = False
            outcome[label+"_error"] = str(error)
    try:
        outcome["observed_control_mode"] = int(stub.GetVehicleControlMode(info, timeout=.5).mode)
        state = stub.GetActorState(info, timeout=.5)
        observed = live._observation(state)
        outcome["observed_gear_mode"] = int(state.vehicle_state.gear_mode)
        outcome["observed_speed_norm_raw"] = observed["speed_norm_raw"]
        outcome["observed_position_xyz"] = observed["position_xyz"]
        outcome["observed_parked_and_zero_speed"] = (outcome["observed_gear_mode"] == int(GEAR_MODE_P)
                                                      and observed["speed_norm_raw"] <= live.MAX_SPEED_RAW)
    except Exception as error:
        outcome["readback_error"] = str(error)
    outcome["finished_monotonic"] = time.monotonic()
    outcome["accepted_and_control_blocked"] = all(outcome.get(key) is True for key in (
        "external_input_disabled", "velocity_zero", "gear_p")) and outcome.get("observed_control_mode") == int(VEHICLE_CONTROL_KEYBOARD)
    current = getattr(context, "tunnel_static_trial_context", None)
    if current is not None:
        current.report["immediate_hold"] = outcome
        current.report["all_immediate_hold_calls_accepted"] = current.report.get("all_immediate_hold_calls_accepted", True) and outcome["accepted_and_control_blocked"]
    return outcome


def park(context):
    if not enabled(context):
        return
    immediate_hold(context)
    try:
        context.stop_drive_control()
    finally:
        live._park(context.grpc, context.grpc.get_ego(), {"ego_parking_commands": 0})
    current = getattr(context, "tunnel_static_trial_context", None)
    if current is not None:
        current.report["ego_stopped_and_parked"] = True
        from .tunnel_static_startup import common_brake_hold_enabled
        if common_brake_hold_enabled(context.grpc):
            confirmed = (current.report.get('final_stop_verified') is True
                and current.report.get('immediate_hold', {}).get('final_verified') is True
                and not current.report.get('termination_hard_failures'))
        else:
            confirmed = current.report.get("all_immediate_hold_calls_accepted") is True
        if not confirmed:
            raise RuntimeError("Immediate trial Ego input block / parking handshake failed")
        current.persist()


def make_drive_callback(fixture_grpc, map_loader, global_cfg, scenario_cfg, output_dir, *,
                        scenario_factory=None, scenario_kwargs=None, configure_scenario=None,
                        result_sink=None, collection_episode=None):
    """Return the synchronous scene callback; the outer helper owns cleanup."""
    def drive(report, owned):
        from .scenario import MissionScenario
        _, boundary_type, _, _, _ = screening._expert_search()
        measured = report["measured_route_screen"]
        boundary = boundary_type.from_map_dir(global_cfg["paths"]["mgeo_root"],
            approach_m=measured["road_boundary_policy"]["approach_m"],
            departure_m=measured["road_boundary_policy"]["departure_m"],
            planning_margin_m=measured["road_boundary_policy"]["planning_road_margin_m"],
            runtime_margin_m=measured["road_boundary_policy"]["runtime_road_margin_m"],
            body_policy=measured["measured_ego_body"])
        monitor = TrialRosEvidence()
        trial = TrialContext(report["planned"], report, global_cfg, Path(output_dir), boundary, monitor)
        if set(trial.references) != {str(item["raw_id"]) for item in owned}:
            raise RuntimeError("Trial callback fixture ownership differs from the observed scene")
        runner_cfg = copy.deepcopy(scenario_cfg)
        runner_cfg.update(mission_mode=MODE, auto_repeat=False, max_laps=1, timeout_sec=180.,
                          route_debug_enabled=False, ego_trace_path=str(Path(output_dir)/"ego_trace.csv"))
        runner_cfg.setdefault("sync_runtime", {})["enabled"] = False
        cfg = copy.deepcopy(global_cfg)
        cfg["paths"]["runtime_root"] = str(Path(output_dir)/"runner_runtime")
        if collection_episode is not None:
            cfg["paths"]["runtime_root"] = str(Path(collection_episode)/"tunnel_static"/"runner_runtime")
        if not runner_cfg.get("tunnel_static_profile_state_path_explicit", False):
            runner_cfg["profile_state_path"] = str(Path(output_dir)/"profile.json")
        client = BorrowedTrialClient(fixture_grpc)
        scenario = (scenario_factory or MissionScenario)(client, map_loader, cfg, runner_cfg,
                                                        **(scenario_kwargs or {}))
        if collection_episode is not None:
            # The existing RawScenario constructor owns creation of the episode.
            # Only then create its tunnel evidence directory.
            trial.output_dir = Path(collection_episode) / "tunnel_static"
            trial.output_dir.mkdir(parents=True, exist_ok=True)
            runner_cfg["ego_trace_path"] = str(trial.output_dir / "ego_trace.csv")
            scenario.cfg["ego_trace_path"] = runner_cfg["ego_trace_path"]
        scenario.tunnel_static_trial_context = trial
        if configure_scenario is not None:
            configure_scenario(scenario)
        if collection_episode is not None:
            def verified_collection_brake():
                observed = immediate_hold(scenario)
                if (not observed or observed.get("accepted_and_control_blocked") is not True
                        or observed.get("observed_parked_and_zero_speed") is not True
                        or observed.get('final_verified') is not True
                        or trial.report.get('termination_hard_failures')):
                    raise RuntimeError("Collected tunnel final parking was not verified")
                return observed
            scenario.final_brake = verified_collection_brake
        try:
            monitor.start()
            outcome = scenario.run()
            if result_sink is not None:
                result_sink["outcome"] = outcome
                result_sink["scenario"] = scenario
            if collection_episode is not None:
                if (not isinstance(outcome, dict) or outcome.get("status") != "mission_completed"
                        or outcome.get("full_mission_completed") is not True or trial.report.get("status") != "passed"):
                    raise RuntimeError("Collected tunnel mission did not complete all actual-drive conditions: " + str(outcome))
                park(scenario)
            if not client.world_reused:
                raise RuntimeError("Trial did not reuse the exact parked fixture world")
            return trial.report
        except BaseException as error:
            trial.fail(error)
            raise
        finally:
            monitor.close()
    return drive
