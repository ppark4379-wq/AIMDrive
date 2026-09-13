#!/usr/bin/env python3
"""Compare the traffic-light response of one encoder on CARLA and on MORAI.

Top rows show the camera frame immediately before and after a red-to-green
change.  The bar chart contrasts how much the traffic-light token moves at that
change against how much it moves between two frames of the same colour, which is
the only fair noise floor for a stationary vehicle.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

TRAFFIC_TOKEN = 100


def relative_change(tokens, before, after, token=None):
    sl = slice(None) if token is None else slice(token, token + 1)
    x, y = tokens[str(before)][sl], tokens[str(after)][sl]
    return float(np.linalg.norm(x - y) / max(np.linalg.norm(x), 1e-9))


def read_frame(spec, frame):
    root, route, mode = spec
    if mode == "carla_full":
        path = Path(root) / "data" / route / "rgb_full" / "{:04d}.jpg".format(frame)
        return Image.open(path).convert("RGB").crop((0, 0, 800, 600))
    path = Path(root) / "data" / route / "rgb_front" / "{:04d}.jpg".format(frame)
    return Image.open(path).convert("RGB")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-root", type=Path, required=True)
    parser.add_argument("--carla-probe", required=True)
    parser.add_argument("--carla-root", type=Path, required=True)
    parser.add_argument("--carla-route", required=True)
    parser.add_argument("--carla-frames", required=True,
                        help="red_before,green_after,red_control")
    parser.add_argument("--morai-probes", required=True, help="Comma separated probe names")
    parser.add_argument("--morai-root", type=Path, required=True)
    parser.add_argument("--morai-route", default="full_loop_basic_drive_2026_09_09_01_44_15")
    parser.add_argument("--morai-frames", required=True,
                        help="red_before,green_after,red_control")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    def tokens(name):
        return dict(np.load(args.probe_root / name / "tokens.npz"))

    cr, cg, cc = (int(x) for x in args.carla_frames.split(","))
    mr, mg, mc = (int(x) for x in args.morai_frames.split(","))
    morai_probes = [x.strip() for x in args.morai_probes.split(",") if x.strip()]

    bars = []
    ct = tokens(args.carla_probe)
    bars.append(("CARLA\n" + args.carla_probe.replace("carla_red_", ""),
                 relative_change(ct, cr, cg, TRAFFIC_TOKEN),
                 relative_change(ct, cc, cr, TRAFFIC_TOKEN)))
    for name in morai_probes:
        mt = tokens(name)
        bars.append(("MORAI\n" + name,
                     relative_change(mt, mr, mg, TRAFFIC_TOKEN),
                     relative_change(mt, mc, mr, TRAFFIC_TOKEN)))

    fig = plt.figure(figsize=(13, 7.5))
    grid = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.15], hspace=0.3, wspace=0.25)

    specs = [
        (0, "CARLA red (frame {})".format(cr), (args.carla_root, args.carla_route, "carla_full"), cr),
        (1, "CARLA green (frame {})".format(cg), (args.carla_root, args.carla_route, "carla_full"), cg),
        (2, "MORAI red (frame {})".format(mr), (args.morai_root, args.morai_route, "morai"), mr),
        (3, "MORAI green (frame {})".format(mg), (args.morai_root, args.morai_route, "morai"), mg),
    ]
    for column, title, spec, frame in specs:
        axis = fig.add_subplot(grid[0, column])
        axis.imshow(read_frame(spec, frame))
        axis.set_title(title, fontsize=10)
        axis.set_xticks([]); axis.set_yticks([])

    axis = fig.add_subplot(grid[1, :])
    positions = np.arange(len(bars))
    width = 0.38
    axis.bar(positions - width / 2, [b[1] for b in bars], width,
             label="red to green change", color="#c0392b")
    axis.bar(positions + width / 2, [b[2] for b in bars], width,
             label="same colour control", color="#7f8c8d")
    for index, (_, change, control) in enumerate(bars):
        axis.text(index, max(change, control) * 1.05 + 0.02,
                  "{:.2f}x".format(change / max(control, 1e-9)), ha="center", fontsize=10)
    axis.set_xticks(positions)
    axis.set_xticklabels([b[0] for b in bars], fontsize=9)
    axis.set_ylabel("relative change of the traffic-light token")
    axis.set_title("A colour change should move the token far more than two frames of the same colour",
                   fontsize=10)
    axis.legend()
    axis.grid(axis="y", alpha=0.3)

    fig.suptitle("Traffic-light signal in the visual tokens: CARLA versus MORAI", fontsize=12)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=130, bbox_inches="tight")
    print(json.dumps({name: dict(change=c, control=k, ratio=c / max(k, 1e-9))
                      for name, c, k in bars}, indent=2))


if __name__ == "__main__":
    main()
