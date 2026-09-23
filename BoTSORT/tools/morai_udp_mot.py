#!/usr/bin/env python3
"""MORAI UDP camera -> YOLOv7/YOLO26 weights -> BoT-SORT -> JSONL."""
import argparse
import contextlib
import json
import logging
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "yolov7"))

from tools.morai_udp import UdpCamera

LOG = logging.getLogger("morai_mot")


def make_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--detector", choices=("yolov7", "yolo26"), default="yolov7")
    p.add_argument("--weights", required=True, help="Existing .pt checkpoint for the selected detector")
    p.add_argument("--bind-ip", default="0.0.0.0",
                   help="Local listen address; accepts localhost and local interface traffic")
    p.add_argument("--port", type=int, default=15001, help="MORAI camera Destination Port (15001)")
    p.add_argument("--sender-ip", help="Optional simulator IP filter")
    p.add_argument("--index-base", type=int, choices=(0, 1), default=0)
    p.add_argument("--device", default="0", help="CUDA device index, or cpu")
    p.add_argument("--img-size", type=int, default=1280,
                   help="YOLO inference size; default matches the 1280x720 camera width")
    p.add_argument("--fp16", action="store_true", help="Use FP16 on CUDA")
    p.add_argument("--conf-thres", type=float, default=0.09)
    p.add_argument("--iou-thres", type=float, default=0.7)
    p.add_argument("--classes", nargs="+", type=int)
    p.add_argument("--agnostic-nms", action="store_true")
    p.add_argument("--track-high-thresh", type=float, default=0.3)
    p.add_argument("--track-low-thresh", type=float, default=0.05)
    p.add_argument("--new-track-thresh", type=float, default=0.4)
    p.add_argument("--track-buffer", type=int, default=30,
                   help="Original 30 Hz reference buffer; 30 becomes 20 updates at 20 Hz")
    p.add_argument("--match-thresh", type=float, default=0.7)
    p.add_argument("--min-box-area", type=float, default=10)
    p.add_argument("--cmc-method", choices=("sparseOptFlow", "orb", "ecc", "none"),
                   default="sparseOptFlow")
    # Preserve upstream's option semantics for reproducibility.
    p.add_argument("--fuse-score", dest="mot20", action="store_true",
                   help="Same flag/behavior as upstream mc_demo_yolov7.py")
    p.add_argument("--view", action="store_true")
    p.add_argument("--show-detections", action="store_true",
                   help="Draw orange YOLO boxes in addition to green tracks")
    p.add_argument("--output", help="JSONL result file (stdout if omitted)")
    p.add_argument("--save-images", help="Directory for original JPEG frames, named by output frame_id")
    p.add_argument("--max-frames", type=int, default=0, help="Stop after N processed frames; 0 unlimited")
    return p


def track_records(tracks, width, height, min_area):
    records = []
    for t in tracks:
        x1, y1, x2, y2 = [float(v) for v in t.tlbr]
        x1, x2 = max(0.0, min(width, x1)), max(0.0, min(width, x2))
        y1, y2 = max(0.0, min(height, y1)), max(0.0, min(height, y2))
        if x2 <= x1 or y2 <= y1 or (x2 - x1) * (y2 - y1) <= min_area:
            continue
        records.append({"track_id": int(t.track_id), "bbox_xyxy": [x1, y1, x2, y2],
                        "class_id": int(t.cls), "score": float(t.score),
                        "confirmed": bool(t.is_activated)})
    return records


def run(args, stdout=None):
    # Lazy imports keep --help and UDP tests independent of torch/ROS.
    import cv2
    import numpy as np
    from tools.morai_detectors import create_detector
    from tracker.mc_bot_sort import BoTSORT

    detector = create_detector(args)
    args.with_reid = False
    args.proximity_thresh, args.appearance_thresh = 0.5, 0.25
    args.name, args.ablation = "morai_udp", False
    tracker = BoTSORT(args, frame_rate=20)
    detector.warmup()
    image_dir = Path(args.save_images) if args.save_images else None
    if image_dir:
        image_dir.mkdir(parents=True, exist_ok=True)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    sink = open(args.output, "w", buffering=1) if args.output else contextlib.nullcontext(stdout if stdout is not None else sys.stdout)
    count, last_stamp, decode_errors, stale_frames = 0, None, 0, 0
    camera = None
    try:
        with sink as output:
            camera = UdpCamera(args.bind_ip, args.port, args.sender_ip, args.index_base)
            LOG.info("Listening on %s:%s; camera=20 Hz, ReID=off", *camera.address)
            while args.max_frames == 0 or count < args.max_frames:
                frame = camera.get()
                if frame is None:
                    LOG.warning("No complete JPEG for 1 s; check MORAI IP/port/protocol. %s", camera.stats())
                    continue
                if last_stamp is not None and frame.stamp <= last_stamp:
                    stale_frames += 1
                    LOG.warning("Ignoring old/duplicate timestamp %s. Restart this script after simulator time reset.",
                                frame.stamp)
                    continue
                started = time.monotonic()
                image = cv2.imdecode(np.frombuffer(frame.jpeg, np.uint8), cv2.IMREAD_COLOR)
                if image is None:
                    decode_errors += 1
                    LOG.warning("JPEG decode failed")
                    continue
                if count == 0:
                    LOG.info("Camera image: %dx%d (configured MORAI camera: 1280x720)",
                             image.shape[1], image.shape[0])
                gap = None if last_stamp is None else (
                    frame.stamp[0] - last_stamp[0] + (frame.stamp[1] - last_stamp[1]) * 1e-9
                )
                if gap is not None and gap > 0.075:
                    LOG.warning("Camera timestamp gap %.3f s (expected 0.050 s); MOT baseline affected", gap)
                last_stamp = frame.stamp
                detections = detector.detect(image)
                # Call even on an empty detection frame, preserving upstream lifecycle.
                tracks = tracker.update(detections, image)
                records = track_records(tracks, image.shape[1], image.shape[0], args.min_box_area)
                count += 1
                processing_ms = (time.monotonic() - started) * 1000
                record = {
                    "frame_id": count, "received_frame_id": frame.sequence,
                    "detector": args.detector, "weights": Path(args.weights).name,
                    "timestamp": {"sec": frame.stamp[0], "nanosec": frame.stamp[1]},
                    "camera_hz": 20, "image_size": [image.shape[1], image.shape[0]],
                    "bbox_format": "xyxy_pixels", "tracks": records,
                    "detections": [{"bbox_xyxy": [float(v) for v in d[:4]],
                                    "score": float(d[4]), "class_id": int(d[5])} for d in detections],
                    "processing_ms": processing_ms, "source_dt": gap,
                    "receiver": camera.stats(), "decode_errors": decode_errors,
                    "stale_frames": stale_frames,
                }
                output.write(json.dumps(record, allow_nan=False) + "\n")
                if image_dir:
                    (image_dir / ("%06d.jpg" % count)).write_bytes(frame.jpeg)
                if args.view:
                    if args.show_detections:
                        for d in detections:
                            x1, y1, x2, y2 = [int(v) for v in d[:4]]
                            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 165, 255), 1)
                    for r in records:
                        x1, y1, x2, y2 = [int(v) for v in r["bbox_xyxy"]]
                        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = "ID %d cls %d %.2f" % (r["track_id"], r["class_id"], r["score"])
                        cv2.putText(image, label, (x1, max(15, y1 - 5)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.imshow("MORAI " + args.detector + " + BoT-SORT", image)
                    if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                        break
                if count % 20 == 0:
                    LOG.info("frame=%d tracks=%d processing=%.1f ms %s",
                             count, len(records), processing_ms, camera.stats())
                    if processing_ms > 50:
                        LOG.warning("Processing exceeds the 50 ms camera period")
    finally:
        if camera is not None:
            camera.close()
        if args.view:
            cv2.destroyAllWindows()


def main():
    parser = make_parser()
    args = parser.parse_args()
    if not Path(args.weights).expanduser().is_file():
        parser.error("--weights must point to an existing .pt file for the selected detector")
    if not 1 <= args.port <= 65535 or args.img_size <= 0 or args.max_frames < 0:
        parser.error("Invalid port, image size, or max frames")
    for name in ("conf_thres", "iou_thres", "track_high_thresh", "track_low_thresh",
                 "new_track_thresh", "match_thresh"):
        if not 0 <= getattr(args, name) <= 1:
            parser.error(name + " must be in [0, 1]")
    if args.track_low_thresh >= args.track_high_thresh or args.track_buffer <= 0:
        parser.error("Require track_low_thresh < track_high_thresh and a positive track_buffer")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.conf_thres > args.track_low_thresh:
        LOG.warning("YOLO conf %.2f discards some low-score boxes (tracker low %.2f); upstream demo defaults",
                    args.conf_thres, args.track_low_thresh)
    try:
        # Keep initialization/progress prints from upstream libraries out of JSONL.
        original_stdout = sys.stdout
        with contextlib.redirect_stdout(sys.stderr):
            run(args, stdout=original_stdout)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
