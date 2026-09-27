#!/usr/bin/env python3
"""Render visual-token maps produced by inspect_vision_tokens.py.

One row per frame: the MORAI front camera, then each encoder's BEV token norm
map and PCA colour map.  The PCA basis is shared across all frames of a single
encoder so colours are comparable between frames of that encoder.
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

BEV_TOKENS = 100


def load_probe(directory):
    directory = Path(directory)
    summary = json.loads((directory / "inspection.json").read_text(encoding="utf-8"))
    tokens = {int(k): v for k, v in np.load(directory / "tokens.npz").items()}
    return summary, tokens


def shared_pca_basis(tokens, frames):
    stacked = np.concatenate([tokens[f][:BEV_TOKENS] for f in frames], axis=0).astype(np.float64)
    centred = stacked - stacked.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    return vt[:3], stacked.mean(axis=0, keepdims=True)


def pca_image(token, basis, mean):
    projected = (token[:BEV_TOKENS].astype(np.float64) - mean) @ basis.T
    lo, hi = projected.min(axis=0, keepdims=True), projected.max(axis=0, keepdims=True)
    scaled = (projected - lo) / np.maximum(hi - lo, 1e-9)
    return scaled.reshape(10, 10, 3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-root", type=Path, required=True)
    parser.add_argument("--labels", required=True, help="Comma separated encoder labels")
    parser.add_argument("--frames", required=True, help="Comma separated frame ids to draw")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--route", default="full_loop_basic_drive_2026_09_09_01_44_15")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--titles", default="", help="Optional per-frame captions, ; separated")
    parser.add_argument("--image-mode", choices=("morai", "carla_full"), default="morai",
                        help="carla_full reads rgb_full and takes the leading front view")
    args = parser.parse_args()

    labels = [x.strip() for x in args.labels.split(",") if x.strip()]
    frames = [int(x) for x in args.frames.split(",") if x.strip()]
    captions = [c.strip() for c in args.titles.split(";")] if args.titles else [""] * len(frames)

    probes = {label: load_probe(args.probe_root / label) for label in labels}
    bases = {}
    for label in labels:
        summary, tokens = probes[label]
        bases[label] = shared_pca_basis(tokens, [f for f in frames if f in tokens])

    columns = 1 + 2 * len(labels)
    fig, axes = plt.subplots(len(frames), columns,
                             figsize=(2.5 * columns, 2.7 * len(frames)), squeeze=False)
    for row, frame in enumerate(frames):
        if args.image_mode == "carla_full":
            image_path = args.dataset_root / "data" / args.route / "rgb_full" / "{:04d}.jpg".format(frame)
            picture = Image.open(image_path).convert("RGB").crop((0, 0, 800, 600))
        else:
            image_path = args.dataset_root / "data" / args.route / "rgb_front" / "{:04d}.jpg".format(frame)
            picture = Image.open(image_path)
        axis = axes[row][0]
        axis.imshow(picture)
        axis.set_ylabel("frame {}\n{}".format(frame, captions[row] if row < len(captions) else ""),
                        fontsize=8)
        axis.set_xticks([]); axis.set_yticks([])
        if row == 0:
            axis.set_title("MORAI front RGB", fontsize=9)

        for index, label in enumerate(labels):
            summary, tokens = probes[label]
            token = tokens[frame]
            grid = np.linalg.norm(token[:BEV_TOKENS], axis=1).reshape(10, 10)
            ax_norm = axes[row][1 + 2 * index]
            image = ax_norm.imshow(grid, cmap="viridis")
            ax_norm.set_xticks([]); ax_norm.set_yticks([])
            fig.colorbar(image, ax=ax_norm, fraction=0.046)
            ax_pca = axes[row][2 + 2 * index]
            ax_pca.imshow(pca_image(token, *bases[label]))
            ax_pca.set_xticks([]); ax_pca.set_yticks([])
            if row == 0:
                ax_norm.set_title("{}\nBEV token norm".format(label), fontsize=9)
                ax_pca.set_title("{}\nBEV token PCA".format(label), fontsize=9)

    source = "CARLA" if args.image_mode == "carla_full" else "MORAI"
    fig.suptitle("Vision encoder only: {} BEV visual tokens (10x10 grid, 256-d each)".format(source),
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=130)
    print(json.dumps(dict(output=str(args.output), frames=frames, labels=labels), indent=2))


if __name__ == "__main__":
    main()
