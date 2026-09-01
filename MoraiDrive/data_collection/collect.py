"""
Entrypoint: listens on MORAI's UDP topics, synchronizes them into frames
at DATASET.collect_hz, and writes out a CARLA/LMDrive-style route folder.

Run: python collect.py --route route_moraiA_scenario1 --duration 120

Requires udp/parsers.py's parse_ego_info / parse_object_info to be filled
in with MORAI's real packet layout first -- see that file's TODOs.
"""
import argparse
import time

from config import NETWORK, DATASET
from udp.socket_receiver import UdpTopicReceiver
from udp.parsers import parse_ego_info, parse_object_info, parse_camera_jpeg, parse_lidar_points
from sync.frame_sync import FrameSynchronizer
from writer.dataset_writer import DatasetWriter
from writer.image_writer import ImageWriter
from writer.lidar_writer import LidarWriter
from schema import EgoState, ActorData, build_measurement_dict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--route", default=DATASET.route_name)
    ap.add_argument("--duration", type=float, default=120.0, help="seconds")
    args = ap.parse_args()

    ego_recv = UdpTopicReceiver("ego", NETWORK.bind_ip, NETWORK.ego_info_port,
                                 NETWORK.recv_buffer_size, parse_ego_info).start()
    obj_recv = UdpTopicReceiver("objects", NETWORK.bind_ip, NETWORK.object_info_port,
                                 NETWORK.recv_buffer_size, parse_object_info).start()
    cam_recvs = {
        name: UdpTopicReceiver(f"cam_{name}", NETWORK.bind_ip, port,
                                NETWORK.recv_buffer_size, parse_camera_jpeg).start()
        for name, port in NETWORK.camera_port.items()
    }
    lidar_recv = UdpTopicReceiver("lidar", NETWORK.bind_ip, NETWORK.lidar_port,
                                   NETWORK.recv_buffer_size, parse_lidar_points).start()

    sync = FrameSynchronizer(
        ego_receiver=ego_recv,
        object_receiver=obj_recv,
        camera_receivers=cam_recvs,
        lidar_receiver=lidar_recv,
        max_staleness_sec=NETWORK.max_staleness_sec,
    )

    ds_writer = DatasetWriter(DATASET.output_root, args.route)
    img_writer = ImageWriter(ds_writer.output_dir, view_names=list(NETWORK.camera_port.keys()),
                              ext=DATASET.image_ext, quality=DATASET.image_quality)
    lidar_writer = LidarWriter(ds_writer.output_dir)

    period = 1.0 / DATASET.collect_hz
    frame_id = 0
    t_end = time.time() + args.duration

    try:
        while time.time() < t_end:
            t0 = time.time()
            frame = sync.tick()
            if frame is not None:
                ego = EgoState(
                    x=frame.ego["x"], y=frame.ego["y"],
                    theta=frame.ego["theta"], speed=frame.ego["speed"],
                    x_command=frame.ego.get("x_command", 0.0),
                    y_command=frame.ego.get("y_command", 0.0),
                )
                actors = {
                    aid: ActorData(**adata) for aid, adata in frame.actors.items()
                } if frame.actors else None

                measurement = build_measurement_dict(ego, actors)
                ds_writer.write_frame(frame_id, measurement)
                img_writer.write(frame_id, frame.images)
                lidar_writer.write(frame_id, frame.lidar)
                frame_id += 1
            else:
                print("frame skipped: waiting for fresh sensor data")

            dt = time.time() - t0
            if dt < period:
                time.sleep(period - dt)
    finally:
        ds_writer.finalize()
        for r in [ego_recv, obj_recv, lidar_recv, *cam_recvs.values()]:
            r.stop()
        print(f"done. {frame_id} frames written to {ds_writer.output_dir}")


if __name__ == "__main__":
    main()
