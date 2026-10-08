"""Collect stationary full-pose LiDAR evidence without starting Expert.

Run as python -m scenario_runner.tools.survey_tunnel_boundaries. Existing Ego
must already be parked with no /ctrl_cmd publisher. No StartSimulation, actor
deletion, network change, sensor reload, FOV change, or moving Ego command is
issued. Only parked teleports within three checked map links are used.
"""

import argparse
import copy
import csv
import json
import math
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import numpy as np

from ...main_runner import load_configuration
from ...scenarios.full_loop_parts.tunnel_static import (
    SURVEY_SCHEMA as SCHEMA,
    analyze_clouds,
    cloud_to_world,
    observe_actor as _observation,
    park_ego as _park,
    project_cloud,
    read_actor as _read_one,
    sensor_calibration,
    station_pose,
    survey_route,
    validate_exposure,
)
from ...utils.map_loader import MGeoMapLoader
from .candidates import _controller_publishers, _write


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sensor-settings", type=Path, default=Path(
        "/mnt/c/MoraiLauncher_Win/MoraiLauncher_Win_Data/SaveFile/Sensor/26.R1.H3/aim_sensorset.json"))
    parser.add_argument("--sensor-id", type=int, default=5)
    parser.add_argument("--stations", default="-10,-5,0,5,10,30,60,90,110,115,119,124,129")
    parser.add_argument("--frames", type=int, default=3)
    parser.add_argument("--settle-seconds", type=float, default=2.)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--analyze", type=Path, help="Reanalyze an existing manifest without simulator connection")
    return parser


class CloudBuffer:
    def __init__(self):
        self.condition = threading.Condition()
        self.rows = []

    def receive(self, message):
        with self.condition:
            self.rows.append((message, time.monotonic(), time.time()))
            self.rows = self.rows[-50:]
            self.condition.notify_all()

    def clear(self):
        with self.condition:
            self.rows.clear()

    def fresh(self, after_monotonic, after_stamp, timeout=8.):
        deadline = time.monotonic()+timeout
        with self.condition:
            while True:
                for message, receipt, unix in self.rows:
                    if receipt > after_monotonic and message.header.stamp.to_sec() > after_stamp:
                        return message, receipt, unix
                remaining = deadline-time.monotonic()
                if remaining <= 0:
                    raise RuntimeError("No fresh LiDAR cloud during stationary pose bracket")
                self.condition.wait(min(.25, remaining))


def _pose(grpc, ego):
    state = _read_one(grpc, ego)
    result = _observation(state)
    result.update(gear_mode=int(state.vehicle_state.gear_mode), read_monotonic=time.monotonic(), read_unix=time.time())
    return result


def _clock(adapter):
    from proto.morai.common.type_pb2 import Empty
    result = adapter._simulator_stub.GetTimestamp(Empty(), timeout=1.)
    # This MORAI API reports Unix microseconds. PointCloud stamps use seconds.
    # Keep the conversion explicit rather than accepting arbitrary clock shifts.
    return float(result.value)*.000001


def _teleport_station(grpc, ego, route, station):
    from proto.morai.actor.actor_set_pb2 import SetTransformParam
    from proto.morai.common.enum_pb2 import STATUS_CODE_SUCCESS

    position, yaw = station_pose(route, station)
    target_yaw = math.degrees(yaw)
    command_yaw = target_yaw
    report = {"ego_parking_commands": 0}
    for attempt in range(3):
        transform = grpc.make_transform(*position, command_yaw)
        param = SetTransformParam(actor_info=ego.get_object_info(), transform=transform)
        result = grpc.world._sim_adapter._actor_stub.SetTransform(param, timeout=2.)
        if result.status != STATUS_CODE_SUCCESS:
            raise RuntimeError("Parked Ego teleport was rejected")
        _park(grpc, ego, report)
        time.sleep(.35)
        observed = _pose(grpc, ego)
        yaw_error = (observed["rotation_xyz_deg"][2]-target_yaw+180.) % 360.-180.
        if math.dist(observed["position_xyz"][:2], position[:2]) > .1:
            raise RuntimeError("Parked Ego teleport XY readback differs from map station")
        if abs(yaw_error) <= .15:
            return {"station_m": station, "command_yaw_deg": command_yaw,
                    "desired_map_pose_xyz": list(position), "desired_yaw_deg": target_yaw,
                    "attempts": attempt+1, "observed": observed}
        command_yaw -= yaw_error
    raise RuntimeError("Parked Ego teleport yaw did not converge")


def _attach(global_cfg):
    """Attach using raw handle: SDK SimulationWorld constructor deletes actors."""
    from ...adapters.morai_grpc import MoraiGrpcClient

    cfg = copy.deepcopy(global_cfg)
    cfg["grpc"]["client_key"] = "tunnel_stationary_survey"
    grpc = MoraiGrpcClient(cfg)
    grpc.connect()
    from api.vehicle import Vehicle
    adapter = grpc.client._sim_adapter
    ego = Vehicle(adapter, grpc.client_key, "Ego")
    grpc.world = SimpleNamespace(_sim_adapter=adapter, get_ego=lambda: ego)
    return grpc, ego


def analyze(manifest_path):
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest["schema"] != SCHEMA:
        raise ValueError("Unsupported stationary survey schema")
    points, views = [], []
    for frame in manifest["frames"]:
        cloud = np.load(manifest_path.parent/frame["file"], allow_pickle=False)["points"]
        world = cloud_to_world(cloud, frame["before"], manifest["calibration"])
        projected = project_cloud(world, manifest["route"])
        points.append(projected)
        views.extend([frame["station_index"]]*len(projected))
    if not points:
        raise RuntimeError("No stationary survey frames available")
    projected = np.vstack(points)
    np.savez_compressed(manifest_path.parent/"projected_cloud.npz", station_lateral_height=projected, view_ids=views)
    result = analyze_clouds(projected, views, manifest["route"], calibration=manifest["calibration"])
    result.update(source_manifest=str(manifest_path.resolve()),
                  acquisition_success=manifest.get("status") == "collected_restored_parked",
                  ego_restoration=manifest.get("restoration"))
    _write(manifest_path.parent/"boundaries.json", result)
    with (manifest_path.parent/"roof_bins.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result["roof_bins"][0]))
        writer.writeheader(); writer.writerows(result["roof_bins"])
    with (manifest_path.parent/"structure_bins.csv").open("w", newline="") as stream:
        rows = result["structure_extent"]["bins"]
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    plot_analysis(projected, manifest, result, manifest_path.parent)
    return result


def plot_analysis(projected, manifest, result, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
    stride = max(1, len(projected)//160000)
    scatter = projected[::stride]
    axes[0].scatter(scatter[:, 0], scatter[:, 2], s=.25, alpha=.25)
    axes[0].set(xlabel="Station from mapped entrance (m)", ylabel="Height above MGeo (m)", ylim=(-1, 10))
    mask = (scatter[:, 2] > .7) & (scatter[:, 2] < 3.5)
    axes[1].scatter(scatter[mask, 0], scatter[mask, 1], s=.25, alpha=.25)
    axes[1].axhspan(-1.75, 5.25, alpha=.12, color="green", label="Nominal mapped carriageway, physical curb pending")
    axes[1].set(xlabel="Station from mapped entrance (m)", ylabel="Lateral from Ego lane centre (m)", ylim=(-9, 13))
    for axis in axes:
        axis.set_xlim(-15, manifest["route"]["length_m"]+15)
        for value in (0., manifest["route"]["length_m"]):
            axis.axvline(value, linestyle="--", color="black", linewidth=1)
        for estimate in result["portal_estimates"].values():
            axis.axvspan(*estimate["station_interval_m"], alpha=.2, color="orange")
    axes[1].legend(loc="upper left")
    figure.suptitle("Stationary full-RPY LiDAR evidence; structure edges estimated, drivable boundary unverified")
    figure.savefig(Path(output)/"boundaries.png", dpi=160)
    plt.close(figure)


def collect(args):
    stations = [float(value) for value in args.stations.split(",")]
    if (not stations or not all(math.isfinite(value) for value in stations)
            or not 1 <= args.frames <= 20 or not math.isfinite(args.settle_seconds) or args.settle_seconds < 1.):
        raise ValueError("Finite stations, 1..20 frames, and at least 1s settling are required")
    calibration = sensor_calibration(args.sensor_settings, args.sensor_id)
    global_cfg, _, _ = load_configuration()
    route = survey_route(MGeoMapLoader(global_cfg["paths"]["mgeo_root"]))
    if min(stations) < -15. or max(stations) > route["length_m"]+15.:
        raise ValueError("Survey stations exceed bounded tunnel approach/departure")
    for station in stations:
        station_pose(route, station)
    if _controller_publishers():
        raise RuntimeError("Stop competing /ctrl_cmd publishers before stationary survey")
    output = (args.output or Path(global_cfg["paths"]["runtime_root"])/"tunnel_static"/("boundary_survey_"+str(time.time_ns()))).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"schema": SCHEMA, "status": "collecting", "calibration": calibration, "route": route,
                "stations": [], "frames": [], "expert_started": False, "ego_driving_commanded": False,
                "sensor_settings_changed": False, "created_unix": time.time()}
    grpc = None; ego = None; initial = None; subscriber = None
    try:
        import rospy
        from sensor_msgs.msg import PointCloud2
        from sensor_msgs import point_cloud2
        rospy.init_node("tunnel_stationary_boundary_survey", anonymous=True, disable_signals=True)
        grpc, ego = _attach(global_cfg)
        from proto.morai.actor.actor_enum_pb2 import GEAR_MODE_P
        initial_observation = _pose(grpc, ego)
        if initial_observation["gear_mode"] != int(GEAR_MODE_P) or initial_observation["speed_norm_raw"] > .05:
            raise RuntimeError("Existing Ego must already be stationary in P")
        initial = initial_observation
        from proto.morai.actor.actor_set_pb2 import EgoCruiseControl
        from proto.morai.common.enum_pb2 import STATUS_CODE_SUCCESS
        adapter = grpc.world._sim_adapter
        infos = adapter._sensor_stub.GetSensorInfos(ego.get_object_info(), timeout=2.)
        matches = [value for value in infos.values if int(value.sensor_id.value) == args.sensor_id]
        if len(matches) != 1 or int(matches[0].sensor_type) != 2:
            raise RuntimeError("Pinned 3D LiDAR identity/type not uniquely present on active Ego")
        manifest["runtime_sensor_infos"] = [{"id": int(value.sensor_id.value), "type": int(value.sensor_type)} for value in infos.values]
        manifest["initial_ego"] = initial
        # Disable existing cruise; do not select AI/Auto or issue a drive command.
        cruise = EgoCruiseControl(actor_info=ego.get_object_info(), cruise_on=False)
        result = adapter._actor_stub.SetVehicleEgoCruise(cruise, timeout=2.)
        if result.status != STATUS_CODE_SUCCESS:
            raise RuntimeError("Could not disable Ego cruise")
        buffer = CloudBuffer()
        subscriber = rospy.Subscriber(calibration["topic"], PointCloud2, buffer.receive, queue_size=1)
        previous_stamp = None
        for index, station in enumerate(stations):
            row = _teleport_station(grpc, ego, route, station)
            manifest["stations"].append(row)
            time.sleep(args.settle_seconds)
            buffer.clear()
            for number in range(args.frames):
                accepted = False
                for attempt in range(8):
                    before = _pose(grpc, ego)
                    stamp_before = _clock(adapter)
                    threshold = max(stamp_before, previous_stamp or 0.)
                    buffer.clear()
                    message, receipt, unix = buffer.fresh(before["read_monotonic"], threshold)
                    after = _pose(grpc, ego)
                    stamp_after = _clock(adapter)
                    stamp = message.header.stamp.to_sec()
                    try:
                        validate_exposure(before, after, cloud_stamp=stamp, stamp_before=stamp_before,
                            stamp_after=stamp_after, receipt_monotonic=receipt,
                            pose_before_monotonic=before["read_monotonic"], pose_after_monotonic=after["read_monotonic"],
                            frame_id=message.header.frame_id, expected_frame=calibration["frame_id"], previous_stamp=previous_stamp)
                        if before["gear_mode"] != int(GEAR_MODE_P) or after["gear_mode"] != int(GEAR_MODE_P):
                            raise RuntimeError("Ego left P during collection")
                    except RuntimeError:
                        if attempt == 7:
                            raise
                        continue
                    names = tuple(value.name for value in message.fields)
                    columns = ("x", "y", "z", "intensity") if "intensity" in names else ("x", "y", "z")
                    points = np.asarray(list(point_cloud2.read_points(message, field_names=columns, skip_nans=True)), dtype=np.float32)
                    name = "station_%02d_frame_%02d.npz" % (index, number)
                    np.savez_compressed(output/name, points=points)
                    manifest["frames"].append({"file": name, "station_index": index, "station_m": station,
                        "point_count": len(points), "cloud_stamp_sec": stamp, "received_monotonic": receipt,
                        "received_unix": unix, "frame_id": message.header.frame_id, "columns": list(columns),
                        "sim_stamp_before_sec": stamp_before, "sim_stamp_after_sec": stamp_after,
                        "before": before, "after": after})
                    previous_stamp = stamp; accepted = True
                    _write(output/"manifest.json", manifest)
                    break
                if not accepted:
                    raise RuntimeError("Could not bracket stationary LiDAR frame")
            print(json.dumps({"station_m": station, "frames_total": len(manifest["frames"])}), flush=True)
        manifest["status"] = "collected"
    except BaseException as error:
        manifest.update(status="failed", failure_reason=str(error) or type(error).__name__)
        raise
    finally:
        if subscriber is not None:
            subscriber.unregister()
        try:
            if grpc is not None and initial is not None:
                from proto.morai.actor.actor_set_pb2 import SetTransformParam
                from proto.morai.common.enum_pb2 import STATUS_CODE_SUCCESS
                desired = initial["rotation_xyz_deg"]
                command = list(desired)
                for attempt in range(3):
                    transform = grpc.make_transform(*initial["position_xyz"], command[2])
                    transform.rotation.x, transform.rotation.y = command[:2]
                    param = SetTransformParam(actor_info=ego.get_object_info(), transform=transform)
                    if grpc.world._sim_adapter._actor_stub.SetTransform(param, timeout=2.).status != STATUS_CODE_SUCCESS:
                        raise RuntimeError("Ego initial transform restoration rejected")
                    _, observed = _park(grpc, ego, {"ego_parking_commands": 0})
                    error = [(actual-target+180.) % 360.-180. for actual, target in zip(observed["rotation_xyz_deg"], desired)]
                    if max(map(abs, error)) < .2:
                        break
                    command = [value-difference for value, difference in zip(command, error)]
                final = _pose(grpc, ego)
                if math.dist(final["position_xyz"][:2], initial["position_xyz"][:2]) > .1:
                    raise RuntimeError("Ego restoration XY changed")
                if max(abs((actual-target+180.) % 360.-180.) for actual, target in zip(final["rotation_xyz_deg"], desired)) > .2:
                    raise RuntimeError("Ego restoration RPY changed")
                manifest["restoration"] = {"status": "initial_pose_restored_verified_P", "observed": final,
                    "cruise": "disabled; P retained", "control_mode": "unchanged; no mode write issued"}
                if manifest["status"] == "collected":
                    manifest["status"] = "collected_restored_parked"
        except BaseException as error:
            manifest.update(status="failed", restoration={"status": "failed", "reason": str(error)})
            raise
        finally:
            _write(output/"manifest.json", manifest)
            if grpc is not None:
                grpc.client.disconnect()
    return analyze(output/"manifest.json")


def main():
    parser = build_parser(); args = parser.parse_args()
    try:
        result = analyze(args.analyze) if args.analyze else collect(args)
    except (ValueError, RuntimeError) as error:
        parser.exit(2, str(error)+"\n")
    print(json.dumps({key: result[key] for key in ("source_manifest", "portal_estimates", "observed_wall_span_m", "physical_portal_edge_verified", "physical_drivable_width_verified")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
