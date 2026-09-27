#!/usr/bin/env python3
"""Run the frozen stage-1 vision encoder on the live MORAI UDP streams.

    python inference/realtime_vision_encoder.py --config configs/vldrive_realtime.json#presets/vision_encoder_realtime

Receive-only: the encoder watches the simulator and draws what its prediction
heads say, and nothing in this process ever sends to the simulator, so it cannot
drive the car.

Each iteration assembles one frame from the newest sample of every stream and
runs the encoder.  Inference never waits for a picture: the figures are drawn in
a child process at ``output.render_hz``, and ``render_hz: 0`` turns them off
entirely so the loop runs the encoder and nothing else.

``output.window`` puts the panels side by side in one window -- the raw front
camera next to the figures the heads produce -- which is the view to watch while
driving.  They are the same figures that go to the files, rasterised rather than
saved, so the live view and the paper figures cannot drift apart.  Pressing "s"
on the window saves the frame on screen at full quality.

``output.write_files`` writes them into ``output.dir`` instead of, or as well as,
the window:

    latest_bev_detection.png   oriented boxes over the objectness map
    latest_bev_features.png    BEV token activation VLDrive pools
    latest_traffic_light.png   red / not-red probability beside the camera
    latest_waypoints.png       the 0.5 ... 2.5 s predicted path

Open those in any viewer that reloads on change.  Pass ``--history`` to also keep
a numbered copy of every frame.

A status line reports the measured rate and the staleness of each stream, so a
sensor that stops publishing is visible immediately rather than silently
freezing the picture.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import queue as queue_module
import shutil
import signal
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.common.runtime import configure_runtime

configure_runtime()

import numpy as np
import torch

from inference.live_frame import LiveFrameSource
from data.src.loaders.morai_encoder_contract import checkpoint_encoder_contract
from inference.live_view import DisplayUnavailable, LiveView, camera_panel
from models.ve.vision_encoder import create_vision_encoder
from utils.visualization import vision_encoder_visualization as viz
from utils.visualization.vision_encoder_visualization import plt
from utils.training.vision_encoder_training import MODEL_NAME, sha256_file

# Torch sizes its CPU thread pool for training-sized tensors.  The per-frame
# work here is tiny -- a 224x224 ToTensor, a Normalize -- and fanning that out
# across every core costs far more than it saves: measured on this machine,
# ToTensor of one camera frame takes 19.05 ms on 20 threads and 0.07 ms on four,
# which alone was 232 ms of every frame.  The encoder forward runs on the GPU
# and is unaffected.
CPU_THREADS = 4

RUNNING = True


def _stop(_signum, _frame):
    global RUNNING
    RUNNING = False


def load_encoder(checkpoint, contract_sidecar=None):
    """Load a checkpoint into the head-bearing encoder and freeze it."""
    path = Path(checkpoint)
    if not path.is_file():
        raise SystemExit("Checkpoint not found: {}".format(path))
    # pretrained_backbone=False: the strict load below replaces every weight.
    model = create_vision_encoder({"name": MODEL_NAME}, pretrained_backbone=False)
    blob = torch.load(str(path), map_location="cpu", weights_only=False)
    state = blob.get("state_dict", blob)
    result = model.load_state_dict(state, strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    checkpoint_hash = sha256_file(path)
    sidecar = json.loads(Path(contract_sidecar).read_text(encoding="utf-8")) if contract_sidecar else None
    contract = checkpoint_encoder_contract(blob, checkpoint_hash, sidecar)
    return model, contract, {
        "checkpoint": str(path),
        "checkpoint_sha256": checkpoint_hash,
        "strict_load": str(result),
        "steps": blob.get("steps"),
        "encoder_contract": contract or None,
    }


def resolve_adapter(contract, override):
    """Reject overrides that would silently change checkpoint preprocessing."""
    adapter = dict(contract["adapter"])
    for key, value in (override or {}).items():
        if key == "note" or value is None:
            continue
        if key not in adapter or json.dumps(value) != json.dumps(adapter[key]):
            raise ValueError("encoder.{} contradicts the checkpoint contract".format(key))
    return adapter, dict.fromkeys(adapter, "checkpoint contract")


# What the live window shows, left to right, top to bottom.  "camera" is the
# raw front image; the rest are the prediction heads.
DEFAULT_PANELS = ("camera", "bev_detection", "traffic_light", "waypoints")
FIGURE_PANELS = ("bev_detection", "bev_features", "traffic_light", "waypoints")


def panel_names(config):
    """Which panels this run produces, honouring the older bev_source setting."""
    requested = config["output"].get("panels")
    if requested:
        return [name for name in requested]
    source = config["visualization"]["bev_source"]
    names = ["camera"]
    if source in ("detection", "both"):
        names.append("bev_detection")
    if source in ("features", "both"):
        names.append("bev_features")
    return names + ["traffic_light", "waypoints"]


def build_figures(config, snapshot, names):
    """The matplotlib figures for one snapshot, in panel order.

    Called only from the figure process; ``snapshot`` is already plain numpy.
    This is the single place the renderers are called, so the window and the
    saved files are the same picture by construction.
    """
    detection, waypoints, traffic_light, bev_features, meta = snapshot
    subtitle = "live  {:.1f} m/s   yaw {:.1f} deg".format(meta["speed"], meta["yaw_degrees"])
    if meta.get("velocity_contract_mismatch"):
        subtitle += " | DIAGNOSTIC: status speed != training GPS"
    visualization = config["visualization"]
    overlay = meta.get("lidar_xy") if visualization.get("lidar_overlay", True) else None

    makers = {
        "bev_detection": lambda: viz.render_bev_detection(
            detection, threshold=visualization["detection_threshold"],
            lidar_points=overlay, subtitle=subtitle,
            detection_axes=meta["detection_output_axes"]),
        "bev_features": lambda: viz.render_bev_features(bev_features, subtitle=subtitle),
        "traffic_light": lambda: viz.render_traffic_light(
            traffic_light,
            camera_image=meta["front_image"] if visualization["show_camera"] else None,
            subtitle=subtitle),
        "waypoints": lambda: viz.render_waypoints(
            waypoints, speed=meta["speed"],
            target_point=np.asarray(meta["target_point"]) if any(meta["target_point"]) else None,
            subtitle=subtitle, waypoint_axes=meta["waypoint_output_axes"]),
    }
    return [(name, makers[name]()) for name in names if name in makers]


def write_figures(figures, output_dir, dpi, history_index=None, formats=("png",)):
    """Save the figures, leaving them open for whoever else wants them."""
    written = []
    for name, figure in figures:
        pending = output_dir / ("pending_" + name)
        paths = viz.save_figure(figure, pending, formats, dpi, close=False)
        for path in paths:
            final = output_dir / ("latest_" + name + path.suffix)
            # Replace atomically so a viewer never reads a half-written file.
            path.replace(final)
            written.append(final)
            if history_index is not None:
                shutil.copyfile(final, output_dir / "history" /
                                "{:06d}_{}{}".format(history_index, name, path.suffix))
    return written


def _figure_worker(config, output_dir, history, channel, counter, finished):
    """Child process: the only place matplotlib and the window ever run."""
    # The parent owns Ctrl+C and will shut this down through the queue.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    names = panel_names(config)
    write_files = config["output"].get("write_files", True)
    view = None
    if config["output"].get("window", False):
        try:
            view = LiveView([name for name in names],
                            columns=config["output"].get("window_columns", 2),
                            scale=config["output"].get("window_scale", 1.0))
        except DisplayUnavailable as error:
            print("\nno display for the live window ({}); writing files instead".format(error))
            write_files = True
    if not write_files and view is None:
        finished.set()
        return

    try:
        while True:
            snapshot = channel.get()
            if snapshot is None:
                return
            index = counter.value
            figures = build_figures(config, snapshot, names)
            if write_files:
                write_figures(figures, output_dir, config["output"]["dpi"],
                              index if history else None)
            if view is not None:
                meta = snapshot[4]
                panels = [(name, viz.figure_to_bgr(figure, close=False))
                          for name, figure in figures]
                if "camera" in names and meta.get("front_image") is not None:
                    panels.insert(0, ("camera", camera_panel(meta["front_image"])))
                status = ("frame {:d}   {:.1f} m/s   yaw {:.1f} deg   "
                          "lidar {:d} pts").format(index, meta["speed"], meta["yaw_degrees"],
                                                   int(meta.get("lidar_points", 0)))
                if not view.show(panels, status):
                    finished.set()
                    return
                if view.take_save_request():
                    saved = write_figures(figures, output_dir, config["output"]["dpi"],
                                          index, formats=("pdf", "png"))
                    print("\nsaved frame {} at full quality -> {}".format(
                        index, saved[0].parent if saved else output_dir))
            for _name, figure in figures:
                plt.close(figure)
            with counter.get_lock():
                counter.value += 1
    finally:
        if view is not None:
            view.close()


class FigureWriter:
    """Draw the figures in a child process, so inference never waits for them.

    A thread is not enough.  Four matplotlib figures cost about 780 ms and that
    work is nearly all interpreted Python, so a drawing thread holds the GIL and
    stalls the loop just as badly as drawing inline: measured on the live
    streams, inference ran at 4.0 Hz with figures off and 1.8 Hz with a drawing
    thread, whatever rate the thread was throttled to.  In a separate process
    the drawing runs on its own core and the loop keeps its full rate.

    The queue holds one frame.  A frame offered while the child is still drawing
    is dropped rather than queued, so what appears on screen is always the
    newest the child could manage -- which is what someone watching a live view
    wants anyway.
    """

    def __init__(self, config, output_dir, history=False):
        # fork: the child draws and never touches CUDA, and forking costs
        # nothing, where spawn would re-import torch in the child.
        context = multiprocessing.get_context("fork" if hasattr(os, "fork") else "spawn")
        self.period = 1.0 / config["output"]["render_hz"]
        self.submitted = 0
        self.dropped = 0
        self._next = 0.0
        self._channel = context.Queue(maxsize=1)
        self._counter = context.Value("i", 0)
        # Set when the window is closed, so "q" stops the run and not just the
        # picture.
        self.finished = context.Event()
        self._process = context.Process(
            target=_figure_worker,
            args=(config, output_dir, history, self._channel, self._counter, self.finished),
            name="figures", daemon=True)
        self._process.start()

    @property
    def drawn(self):
        return int(self._counter.value)

    def submit(self, outputs, meta):
        """Offer one frame; returns immediately whether or not it is taken."""
        now = time.time()
        if now < self._next:
            return False
        detection, waypoints, traffic_light, _stop_sign, bev_features = outputs
        # Copy off the GPU here: the child must not touch CUDA state, and these
        # are small (the detection map is the largest at 2500x8).
        snapshot = (
            detection[0].float().cpu().numpy(),
            waypoints[0].float().cpu().numpy(),
            traffic_light[0].float().cpu().numpy(),
            bev_features[0].float().cpu().numpy(),
            {"speed": meta["speed"], "yaw_degrees": meta["yaw_degrees"],
             "target_point": meta["target_point"], "front_image": meta["front_image"],
             "lidar_xy": meta.get("lidar_xy"), "lidar_points": meta.get("lidar_points", 0),
             "velocity_contract_mismatch": meta.get("velocity_contract_mismatch", False),
             "waypoint_output_axes": meta["waypoint_output_axes"],
             "detection_output_axes": meta["detection_output_axes"]},
        )
        try:
            self._channel.put_nowait(snapshot)
        except queue_module.Full:
            self.dropped += 1
            return False
        self.submitted += 1
        self._next = now + self.period
        return True

    def close(self, timeout=10.0):
        try:
            self._channel.put(None, timeout=1.0)
        except queue_module.Full:
            pass
        self._process.join(timeout=timeout)
        if self._process.is_alive():
            self._process.terminate()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--contract", type=Path,
                        help="SHA-pinned contract JSON exported for older SensorRecord checkpoints")
    parser.add_argument("--seconds", type=float, default=0.0,
                        help="stop after this long; 0 runs until Ctrl+C")
    parser.add_argument("--history", action="store_true",
                        help="also keep a numbered copy of every rendered frame")
    args = parser.parse_args()

    from utils.config.config_files import read_document, document_bytes
    config = read_document(args.config)
    signal.signal(signal.SIGINT, _stop)

    torch.set_num_threads(config["runtime"].get("cpu_threads", CPU_THREADS))

    device = torch.device(config["runtime"]["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("runtime.device=cuda but no CUDA device is available")

    output_dir = ROOT / config["output"]["dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.history:
        (output_dir / "history").mkdir(exist_ok=True)
    sidecar = args.contract or config.get("contract_sidecar")
    model, contract, provenance = load_encoder(ROOT / config["checkpoint"],
                                               (ROOT / sidecar) if sidecar else None)
    adapter, decided_by = resolve_adapter(contract, config.get("encoder"))
    provenance["adapter"] = adapter
    provenance["adapter_decided_by"] = decided_by
    print("checkpoint : {}".format(provenance["checkpoint"]))
    print("contract   : {} steps, {}".format(
        provenance.get("steps"),
        "token order " + str(contract.get("token_order")) if contract
        else "no morai_encoder_contract in this checkpoint"))
    for key in sorted(adapter):
        print("  {:<26} {:<34} ({})".format(key, str(adapter[key]), decided_by[key]))

    # Forked before moving the model to CUDA. Invalid contracts fail before the child starts.
    writer = (FigureWriter(config, output_dir, history=args.history)
              if config["output"].get("render_hz", 0) else None)
    if writer:
        destination = []
        if config["output"].get("window", False):
            destination.append("window")
        if config["output"].get("write_files", True):
            destination.append(str(output_dir))
        print("figures    : {:g} Hz -> {}".format(
            config["output"]["render_hz"], " + ".join(destination) or "nowhere"))
    else:
        print("figures    : off (output.render_hz = 0)")

    model.to(device)
    route_csv = config["route"]["csv"]
    try:
        source = LiveFrameSource(
            ports=config["ports"],
            route_csv=(ROOT / route_csv) if route_csv else None,
            lidar_elevations=config["lidar"].get("elevations"),
            lidar_axis_policy=config["lidar"]["axis_policy"],
            bind=config["runtime"]["bind"],
            encoder_contract=contract,
            lidar_frame=config["lidar"].get("frame"),
            lidar_to_base=config["lidar"].get("lidar_to_base"),
            velocity_source=config["runtime"].get("velocity_source"),
            allow_velocity_mismatch=config["runtime"].get("allow_velocity_mismatch", False),
            route_map_root=(ROOT / config["route"]["map_root"]) if config["route"].get("map_root") else None,
            route_definition=(ROOT / config["route"]["definition"]) if config["route"].get("definition") else None,
        )
    except Exception:
        if writer is not None:
            writer.close()
        raise
    offsets = (config["route"]["utm_east_offset"], config["route"]["utm_north_offset"])
    provenance["live_lidar_frame"] = config["lidar"]["frame"]
    provenance["live_lidar_to_base"] = config["lidar"].get("lidar_to_base")
    provenance["velocity_contract_mismatch"] = source.velocity_mismatch
    provenance["live_velocity_source"] = source.velocity_source
    provenance["live_pose_source"] = "GPS position + Competition Vehicle Status yaw; teacher-pose parity not verified"
    if source.velocity_mismatch:
        print("[DIAGNOSTIC] vehicle_status speed differs from training causal GPS speed; "
              "this run does not establish scalar/time input parity")
    print("checkpoint SHA256: " + provenance["checkpoint_sha256"])

    print("waiting for every stream to publish...", flush=True)
    ready, missing = source.wait_ready(timeout=config["runtime"]["startup_timeout_s"])
    if not ready:
        print("streams never arrived: {}".format(", ".join(missing)))
        print("Check the relay is running and the simulator's Network Settings match "
              "the ports in this config.")
        source.close()
        if writer is not None:
            writer.close()
        return 1
    print("all streams live -> {}".format(output_dir), flush=True)

    period = 1.0 / max(config["runtime"]["target_hz"], 0.1)
    started = time.time()
    inferred = 0
    skipped = 0
    next_due = time.time()
    while RUNNING:
        if args.seconds and time.time() - started >= args.seconds:
            break
        if writer is not None and writer.finished.is_set():
            print("\nwindow closed")
            break
        now = time.time()
        if now < next_due:
            time.sleep(min(next_due - now, 0.02))
            continue
        next_due = now + period

        inputs, meta = source.build(device, utm_offsets=offsets,
                                    max_age=config["runtime"]["max_age_s"])
        if inputs is None:
            skipped += 1
            print("\r  waiting for {:<40}".format(", ".join(meta["missing"])[:40]),
                  end="", flush=True)
            continue
        with torch.no_grad():
            outputs = model(inputs)
        inferred += 1
        if writer is not None:
            writer.submit(outputs, meta)

        logits = outputs[2][0].float().cpu().numpy()
        shifted = np.exp(logits - logits.max())
        red = float((shifted / shifted.sum())[0])
        objectness = float(torch.sigmoid(outputs[0][0, :, 0]).max())
        forward = float(-outputs[1][0, -1, 1])
        elapsed = time.time() - started
        print("\r  {:5d} infer {:4.1f} Hz | {:4d} drawn | {:5.2f} m/s | p(red) {:.3f} | "
              "obj {:.3f} | fwd@2.5s {:5.1f} m | lidar {:5d} pts   ".format(
                  inferred, inferred / max(elapsed, 1e-6),
                  writer.drawn if writer else 0, meta["speed"], red,
                  objectness, forward, meta["lidar_points"]), end="", flush=True)

    elapsed = time.time() - started
    if writer is not None:
        writer.close()
    print()
    print("stopped after {:.1f} s: {} frames inferred ({:.1f} Hz), {} skipped".format(
        elapsed, inferred, inferred / max(elapsed, 1e-6), skipped))
    if writer is not None:
        print("  figures: {} drawn, {} offered while the last was still drawing".format(
            writer.drawn, writer.dropped))
    for name, port, packets, samples, age in source.stream_report():
        print("  {:<16} port {:<6} {:>8} packets {:>8} samples  last {:.2f} s ago".format(
            name, port, packets, samples, min(age, 999)))
    manifest = {
        "schema": "morai-vision-encoder-realtime-report-v1",
        "config": config,
        "encoder": provenance,
        "forced_inputs": [key for key in ("velocity_input", "target_point_input")
                          if contract[key] == "constant_zero"],
        "cpu_threads": torch.get_num_threads(),
        "frames_inferred": inferred,
        "frames_rendered": writer.drawn if writer else 0,
        "frames_skipped": skipped,
        "inference_hz": inferred / max(elapsed, 1e-6),
        "elapsed_s": elapsed,
        "streams": [{"name": n, "port": p, "packets": k, "samples": s}
                    for n, p, k, s, _ in source.stream_report()],
    }
    (output_dir / "realtime_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    source.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
