"""Shared configuration, serialization and CLI execution for LiDAR-only runs."""
import argparse
from contextlib import nullcontext, ExitStack
from dataclasses import asdict
import json
from pathlib import Path
import time
import numpy as np
import yaml
from ..interfaces.lidar_udp import LidarAssembler, LidarUDPReceiver, LidarUDPStream, decode_packet
from ..pipeline import LidarPipeline
from ..timing.timestamps import TimestampMapper

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "perception.yaml"


def load_config(path):
    with open(path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict) or not all(k in config for k in ("lidar", "timing")):
        raise ValueError("Expected lidar and timing configuration")
    return config


def make_assembler(config):
    return LidarAssembler(TimestampMapper(**config["timing"]["lidar"]),
                          **config["lidar"]["assembly"])


def frame_to_dict(frame):
    def convert(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, dict):
            return {k: convert(v) for k, v in value.items()}
        if isinstance(value, list):
            return [convert(v) for v in value]
        return value
    return convert(asdict(frame))


def main(offline=False):
    parser = argparse.ArgumentParser(description="Independent LiDAR candidate perception")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, help="Optional frame JSONL, includes XYZI points")
    parser.add_argument("--direct-view", action="store_true", help="ROS-free Open3D stage viewer")
    parser.add_argument("--roi-only", action="store_true",
                        help="Open3D ROI preview; skip voxel, ground removal and clustering")
    parser.add_argument("--voxel-only", action="store_true",
                        help="Open3D voxel preview with ROI comparison; skip ground removal and clustering")
    parser.add_argument("--measure-hz", action="store_true", help="Report final pipeline completion Hz every 5 seconds")
    parser.add_argument("--ground-only", action="store_true",
                        help="Open3D RANSAC preview: before, removed ground and remaining points; skip clustering")
    if offline:
        parser.add_argument("--replay-hz", type=float, help="Replay frame rate; default 10 with direct viewer, otherwise unlimited")
        parser.add_argument("--input", type=Path, required=True,
                            help="Packet JSONL: payload_hex and receive_monotonic_time (ns)")
    else:
        parser.add_argument("--duration", type=float, default=0, help="Seconds; 0 runs until Ctrl+C")
        parser.add_argument("--port", type=int)
        parser.add_argument("--record-packets", type=Path, help="Record accepted packets for replay")
    args = parser.parse_args()
    if sum((args.roi_only, args.voxel_only, args.ground_only)) > 1:
        parser.error("Choose one of --roi-only, --voxel-only or --ground-only")
    preview_only = args.roi_only or args.voxel_only or args.ground_only
    if preview_only:
        args.direct_view = True
    if not offline and args.duration < 0:
        parser.error("--duration must be nonnegative")
    replay_hz = (args.replay_hz if args.replay_hz is not None else (10 if args.direct_view else 0)) if offline else 0
    if replay_hz < 0:
        parser.error("--replay-hz must be nonnegative")
    config = load_config(args.config)
    assembler = make_assembler(config)
    pipeline = LidarPipeline(config["lidar"], capture_debug=args.direct_view,
                             roi_only=args.roi_only, voxel_only=args.voxel_only, ground_only=args.ground_only)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    runtime = config["lidar"].get("runtime", {})
    start = time.monotonic()
    with ExitStack() as resources:
        output = resources.enter_context(args.output.open("x", encoding="utf-8") if args.output else nullcontext(None))
        direct_viewer = None
        if args.direct_view:
            from ..visualization.direct_viewer import DirectStageViewer
            direct_viewer = DirectStageViewer(config["lidar"]["preprocessing"],
                                              roi_only=args.roi_only, voxel_only=args.voxel_only, ground_only=args.ground_only)
            resources.callback(direct_viewer.close)
        start = time.monotonic()
        hz_start = start
        hz_count = 0
        hz_complete = 0
        processing_ms = []
        def report_hz(now):
            nonlocal hz_start, hz_count, hz_complete, processing_ms
            elapsed = now - hz_start
            if args.measure_hz and elapsed >= 5:
                mean_ms = float(np.mean(processing_ms)) if processing_ms else 0.0
                p95_ms = float(np.percentile(processing_ms, 95)) if processing_ms else 0.0
                print(f"[Hz] final_output={hz_count / elapsed:.2f} Hz "
                      f"complete_output={hz_complete / elapsed:.2f} Hz "
                      f"process_mean={mean_ms:.2f} ms p95={p95_ms:.2f} ms "
                      f"frames={hz_count} window={elapsed:.2f}s", flush=True)
                hz_start, hz_count, hz_complete, processing_ms = now, 0, 0, []
        def emit(frames):
            nonlocal count, hz_count, hz_complete
            for frame in frames:
                if preview_only and direct_viewer.stop.is_set():
                    break
                frame_start = time.monotonic()
                result = pipeline.process(frame)
                completed_at = time.monotonic()
                if args.measure_hz:
                    # Count at pipeline completion, before file output and rendering.
                    hz_count += 1
                    hz_complete += int(result.complete)
                    processing_ms.append((completed_at - frame_start) * 1000)
                    report_hz(completed_at)
                if output:
                    output.write(json.dumps(frame_to_dict(result), allow_nan=False) + "\n")
                    output.flush()
                if direct_viewer:
                    direct_viewer.update(pipeline.last_debug)
                count += 1
                print(f"frame={frame.frame_id} raw={frame.raw_timestamp} complete={frame.complete} "
                      f"points={len(frame.points)} clusters={len(result.clusters)} "
                      f"time={frame.timestamp_status} issues={','.join(frame.issues)}", flush=True)
                if replay_hz:
                    time.sleep(max(0, 1 / replay_hz - (time.monotonic() - frame_start)))
        try:
            if offline:
                with args.input.open(encoding="utf-8") as source:
                    for line_number, line in enumerate(source, 1):
                        try:
                            row = json.loads(line)
                            packet = decode_packet(bytes.fromhex(row["payload_hex"]),
                                                   **config["lidar"]["decoder"])
                            emit(assembler.feed(packet, int(row["receive_monotonic_time"])))
                            if preview_only and direct_viewer.stop.is_set():
                                break
                        except (ValueError, KeyError) as exc:
                            raise ValueError(f"Invalid replay line {line_number}: {exc}") from exc
            else:
                udp = dict(config["lidar"]["udp"])
                if args.port is not None:
                    udp["port"] = args.port
                with LidarUDPReceiver(assembler, decoder=config["lidar"]["decoder"], **udp) as receiver:
                    packet_file = None
                    try:
                        if args.record_packets:
                            args.record_packets.parent.mkdir(parents=True, exist_ok=True)
                            packet_file = args.record_packets.open("x", encoding="utf-8")
                            receiver.packet_sink = lambda payload, now: packet_file.write(json.dumps(
                                {"payload_hex": payload.hex(), "receive_monotonic_time": now}) + "\n")
                        print(f"Listening on {udp['bind']}:{udp['port']}", flush=True)
                        next_status = time.monotonic() + 5
                        with LidarUDPStream(receiver, runtime.get("frame_queue_size", 4)) as stream:
                            try:
                                while args.duration == 0 or time.monotonic() - start < args.duration:
                                    if preview_only and direct_viewer.stop.is_set():
                                        break
                                    frames = stream.read()
                                    for frame in frames:
                                        frame.diagnostics["processing_dropped_frames"] = stream.dropped_frames
                                    emit(frames)
                                    report_hz(time.monotonic())
                                    if time.monotonic() >= next_status:
                                        print(f"frames={count} invalid_packets={receiver.invalid_packets} "
                                              f"dropped_frames={stream.dropped_frames} "
                                              f"last_error={receiver.last_error}", flush=True)
                                        next_status = time.monotonic() + 5
                            except KeyboardInterrupt:
                                print("Stopping receiver.")
                        emit(stream.drain())
                        print(f"processing_dropped_frames={stream.dropped_frames}")
                    finally:
                        if packet_file:
                            packet_file.close()
                        print(f"invalid_packets={receiver.invalid_packets} last_error={receiver.last_error}")
        except KeyboardInterrupt:
            print("Stopping; checking coverage of pending frames before flushing.")
        finally:
            emit(assembler.flush())
            if direct_viewer:
                print(f"viewer_skipped_frames={direct_viewer.dropped_frames}")
    print(f"Processed {count} frames" + (f"; wrote {args.output}" if args.output else ""))
