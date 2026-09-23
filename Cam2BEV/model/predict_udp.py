#!/usr/bin/env python3
"""Receive four MORAI RGB streams over UDP; optionally run Cam2BEV."""
import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np

from morai_udp import CAMERA_ORDER, CameraReceiver, select_frames


def load_config(path):
    config = json.loads(path.read_text(encoding="utf-8"))
    if set(config["cameras"]) != set(CAMERA_ORDER):
        raise ValueError("Exactly front, rear, left and right cameras are required")
    ports = []
    for name in CAMERA_ORDER:
        camera = config["cameras"][name]
        port = camera["port"]
        if not isinstance(port, int) or not 1 <= port <= 65535:
            raise ValueError("Invalid port for " + name)
        if any(not isinstance(camera[k], int) or camera[k] <= 0 for k in ("width", "height")):
            raise ValueError("Invalid resolution for " + name)
        ports.append(port)
    if len(set(ports)) != 4:
        raise ValueError("Use a distinct destination port for each camera")
    for key in ("max_skew_ms", "max_age_ms"):
        if not np.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(key + " must be positive")
    return config


def decode_batch(batch, cameras):
    images = {}
    for name in CAMERA_ORDER:
        image = cv2.imdecode(np.frombuffer(batch[name].jpeg, np.uint8), cv2.IMREAD_COLOR)
        expected = (cameras[name]["height"], cameras[name]["width"])
        if image is None or image.shape[:2] != expected:
            raise ValueError(name + ": JPEG decode failed or resolution does not match config")
        images[name] = image
    return images


class BevPredictor:
    """An external segmenter supplies native-resolution Cam2BEV class IDs."""
    def __init__(self, config, config_dir):
        settings = config["inference"]
        paths = {}
        for key in ("segmenter", "homographies", "weights"):
            value = settings.get(key)
            if not value:
                raise ValueError("inference." + key + " is required for --infer")
            path = Path(value).expanduser()
            path = path if path.is_absolute() else config_dir / path
            if not path.is_file():
                raise ValueError("File not found: " + str(path))
            paths[key] = path.resolve()
        # Keep capture mode independent of TensorFlow and model weights.
        import utils
        self.utils = utils
        self.shape = tuple(settings["image_shape"])
        if len(self.shape) != 2 or any(not isinstance(s, int) or s <= 0 or s % 16 for s in self.shape):
            raise ValueError("image_shape must contain two positive multiples of 16")
        self.threshold = float(settings.get("road_threshold", 0.7))
        if not 0 <= self.threshold <= 1:
            raise ValueError("road_threshold must be between 0 and 1")
        base = Path(__file__).resolve().parent
        palette = utils.parse_convert_xml(str(base / "one_hot_conversion/convert_9+occl.xml"))
        self.palette = palette
        homographies = utils.load_module(str(paths["homographies"])).H
        if len(homographies) != 4 or any(np.shape(h) != (3, 3) or not np.isfinite(h).all() for h in homographies):
            raise ValueError("H must contain four finite 3x3 STN homographies in front/rear/left/right order")
        self.segmenter = utils.load_module(str(paths["segmenter"]))
        if not callable(getattr(self.segmenter, "segment", None)):
            raise ValueError("Segmenter must expose segment(rgb_image, camera_name)")
        architecture = utils.load_module(str(base / "architecture/uNetXST.py"))
        self.model = architecture.get_network((*self.shape, 10), len(palette), n_inputs=4, thetas=homographies)
        self.model.load_weights(str(paths["weights"]))

    def predict(self, images):
        inputs = []
        for name in CAMERA_ORDER:
            rgb = cv2.cvtColor(images[name], cv2.COLOR_BGR2RGB)
            ids = np.asarray(self.segmenter.segment(rgb, name))
            if (ids.shape != rgb.shape[:2] or not np.issubdtype(ids.dtype, np.integer)
                    or ids.min() < 0 or ids.max() >= 10):
                raise ValueError("segment() must return native HxW integer class IDs 0..9 for " + name)
            # Each camera has its own source resolution. Center-crop, then resize,
            # matching utils.resize_image_op and homography_converter.
            ids = self.utils.resize_image_op(
                ids[..., None].astype(np.float32), ids.shape, self.shape,
                interpolation="nearest").numpy()[..., 0].astype(np.int32)
            inputs.append(np.eye(10, dtype=np.float32)[ids][None])
        prediction = self.model.predict(inputs)[0]
        mask = ((prediction.argmax(-1) == 0) & (prediction[..., 0] >= self.threshold))
        bev = self.utils.one_hot_decode_image(prediction, self.palette)
        return cv2.cvtColor(bev, cv2.COLOR_RGB2BGR), mask.astype(np.uint8) * 255


def save_batch(directory, count, batch, images, result):
    stem = "{:08d}".format(count)
    for name in CAMERA_ORDER:
        folder = directory / name
        folder.mkdir(parents=True, exist_ok=True)
        # Preserve received RGB JPEG without recompression.
        (folder / (stem + ".jpg")).write_bytes(batch[name].jpeg)
    if result is not None:
        for name, image in zip(("bev", "drivable"), result):
            folder = directory / name
            folder.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(folder / (stem + ".png")), image):
                raise IOError("Could not write " + name)
    metadata = {
        "pairing": "host_arrival_time_approximate",
        "cameras": {name: {
            "received_monotonic_s": batch[name].received_at,
            "simulator_timestamp_hex": batch[name].timestamp.hex(),
            "sequence": batch[name].sequence,
            "shape": list(images[name].shape[:2]),
        } for name in CAMERA_ORDER},
    }
    folder = directory / "metadata"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / (stem + ".json")).write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.morai_udp.json"))
    parser.add_argument("--preview", action="store_true", help="Show received RGB and optional BEV")
    parser.add_argument("--output", type=Path, help="Save paired RGB frames and optional predictions")
    parser.add_argument("--diagnostics", action="store_true", help="Print first UDP packet header and length for protocol diagnosis")
    parser.add_argument("--infer", action="store_true", help="Requires segmenter, homographies and weights")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N batches; 0 means unlimited")
    args = parser.parse_args()
    if args.max_frames < 0:
        parser.error("--max-frames must be non-negative")
    if not args.preview and args.output is None:
        parser.error("Specify --preview and/or --output")
    config = load_config(args.config)
    predictor = BevPredictor(config, args.config.resolve().parent) if args.infer else None
    if args.output is not None:
        args.output.mkdir(parents=True, exist_ok=True)
        if any(args.output.iterdir()):
            parser.error("--output must be empty to avoid overwriting captures")
    receivers = {}
    previous = {}
    diagnosed = set()
    count = 0
    report_at = time.monotonic()
    try:
        for name in CAMERA_ORDER:
            port = config["cameras"][name]["port"]
            receivers[name] = CameraReceiver(config["bind_ip"], port, config.get("protocol", "auto"))
            print("{}: UDP {}:{}".format(name, config["bind_ip"], port), flush=True)
        print("Pairing uses host arrival times, not synchronized exposure timestamps.", flush=True)
        while not args.max_frames or count < args.max_frames:
            queues = {name: receiver.snapshot() for name, receiver in receivers.items()}
            batch = select_frames(queues, previous, config["max_skew_ms"] / 1000,
                                  config["max_age_ms"] / 1000)
            if batch:
                previous = {name: frame.sequence for name, frame in batch.items()}
                try:
                    images = decode_batch(batch, config["cameras"])
                except ValueError as exc:
                    print("Dropped batch: " + str(exc), flush=True)
                    continue
                result = predictor.predict(images) if predictor else None
                if args.output is not None:
                    save_batch(args.output, count, batch, images, result)
                if args.preview:
                    for name, image in images.items():
                        cv2.imshow(name, image)
                    if result:
                        cv2.imshow("BEV", result[0])
                        cv2.imshow("Drivable candidate", result[1])
                count += 1
            else:
                time.sleep(0.005)
            if args.preview and cv2.waitKey(1) & 0xff in (27, ord("q")):
                break
            if time.monotonic() - report_at >= 5:
                totals = {name: (frames[-1].sequence if frames else 0) for name, frames in queues.items()}
                stats = {name: receiver.diagnostics() for name, receiver in receivers.items()}
                print("UDP packets: {}".format({name: item["packets"] for name, item in stats.items()}), flush=True)
                if args.diagnostics:
                    for name, item in stats.items():
                        if item["first_packet"] is not None and name not in diagnosed:
                            print("UDP diagnostic {}: {}".format(name, item["first_packet"]), flush=True)
                            if name == "front":
                                for sample in item["packet_samples"]:
                                    print("Front fragment: {}".format(sample), flush=True)
                            diagnosed.add(name)
                print("Complete JPEGs: {}; paired batches: {}".format(totals, count), flush=True)
                report_at = time.monotonic()
    except KeyboardInterrupt:
        pass
    finally:
        for receiver in receivers.values():
            receiver.close()
        if args.preview:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
