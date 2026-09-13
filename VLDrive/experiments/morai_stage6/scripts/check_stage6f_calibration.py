#!/usr/bin/env python3
"""Audit a ROS calibration snapshot against a preserved MORAI sensor preset."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from aimdrive.competition.sensor_calibration import audit_snapshot, transform_points, project_body


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True, help="Probe output directory")
    parser.add_argument("--preset", required=True)
    parser.add_argument("--output", required=True, help="New report JSON")
    parser.add_argument("--bridge-url", default="ws://172.18.40.28:9090")
    parser.add_argument("--confirmed-fovs", help="Operator-checked Camera-1..4 FOVs, comma separated")
    parser.add_argument("--confirmation-note", default="")
    parser.add_argument("--plots", action="store_true", help="Static projection hypotheses, not calibration approval")
    args = parser.parse_args()
    output = Path(args.output).expanduser().resolve()
    if ROOT not in output.parents or output.exists():
        parser.error("Report must be a new file inside this checkout")
    confirmed = None
    if args.confirmed_fovs:
        values = [float(x) for x in args.confirmed_fovs.split(",")]
        if len(values) != 4 or not args.confirmation_note.strip():
            parser.error("Four FOVs and an explicit confirmation note are required")
        confirmed = dict(zip(("Camera-1", "Camera-2", "Camera-3", "Camera-4"), values))
    directory = Path(args.snapshot).expanduser().resolve()
    raw = Path(args.preset).expanduser().read_bytes()
    snap_raw = (directory / "snapshot.json").read_bytes()
    snapshot = json.loads(snap_raw)
    report = audit_snapshot(snapshot, json.loads(raw.decode("utf-8-sig")), args.bridge_url, confirmed)
    report.update(preset_sha256=hashlib.sha256(raw).hexdigest(),
                  snapshot_sha256=hashlib.sha256(snap_raw).hexdigest(),
                  confirmation_note=args.confirmation_note,
                  clock=snapshot.get("clock"), rates=snapshot["rates"])
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.plots and not report["errors"]:
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from PIL import Image
        lidar = np.load(directory / "lidar.npy", allow_pickle=False)[:, :3]
        lidar_frame = next(k for k in report["sensors"] if k.startswith("Lidar3D"))
        base_from_lidar = np.array(report["sensors"][lidar_frame]["measured_child_to_base"])
        points_base = transform_points(lidar, base_from_lidar)
        fig, axes = plt.subplots(4, 2, figsize=(16, 16))
        for row, (frame, camera) in enumerate(sorted(report["cameras"].items())):
            camera_from_base = np.linalg.inv(report["sensors"][frame]["measured_child_to_base"])
            points_camera = transform_points(points_base, camera_from_base)
            width, height = camera["size"]
            for col, axis in enumerate(("horizontal", "vertical")):
                uv, valid = project_body(points_camera, width, height, camera["saved_fov_deg"], axis)
                ax = axes[row, col]
                ax.imshow(Image.open(directory / (frame + ".jpg")))
                ax.scatter(*uv[valid].T, c=points_camera[valid, 0], s=1, cmap="viridis", vmin=0, vmax=30)
                ax.set_title("%s: FOV %s = %s deg (hypothesis)" % (frame, axis, camera["saved_fov_deg"]))
                ax.set_xlim(0, width); ax.set_ylim(height, 0)
        fig.suptitle("Measured TF; body +x forward / +y left / +z up; no motion compensation")
        fig.tight_layout(rect=(0, 0, 1, .98))
        plot_path = output.with_suffix(".projection.png")
        if plot_path.exists():
            raise FileExistsError(plot_path)
        fig.savefig(plot_path, dpi=130)
        plt.close(fig)
        report["projection_hypotheses"] = plot_path.name
    with output.open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    print(json.dumps(dict(report=str(output), errors=report["errors"],
                          measured_setup_matches_preset=report["measured_setup_matches_preset"],
                          calibration_complete=report["calibration_complete"])))
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
