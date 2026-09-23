"""Publication-quality figures for the stage-1 vision encoder's prediction heads.

Three renderers, one per head, each returning its own matplotlib Figure so the
caller saves them as separate files:

    render_bev_detection    traffic_pred_head   -> BEV occupancy + oriented boxes
    render_bev_features     BEV decoder tokens  -> feature activation map
    render_traffic_light    traffic_light_pred_head -> red / not-red probability
    render_waypoints        waypoints_generator -> 0.5 ... 2.5 s future path

Matplotlib only, Agg backend, vector-safe: every figure is drawn with real
metric axes so a PDF/SVG stays sharp at any zoom.

Tensor layouts, verified against models/vision_encoder.py
--------------------------------------------------------
detection ``[2500, 8]`` reshapes to a 50x50 grid whose channels, once
multiplied by :data:`DETECTION_SCALE`, are::

    0 objectness   1,2 offset dx,dy from the cell centre (m)
    3,4 heading cos,sin        5,6 box half-length, half-width (m)
    7 speed (m/s)

Grid cell (row i, col j) sits at ``(lateral, forward) = (j - 24.5, 29.5 - i)``
metres, matching LMDrive's ``convert_grid_to_xy``, so the map covers
lateral [-25, 25] m and longitudinal [-20, 30] m at 1 m per cell.

waypoints ``[5, 2]`` use the LMDrive head convention
``lmdrive_left_negative_forward``: axis 0 is +left, axis 1 is -forward, both
confirmed against real motion (see :func:`forward_right`).  Every figure here
plots the vehicle's right on the right of the page, the usual bird's-eye view,
which is also how the BEV grid column runs.

traffic light ``[2]`` are logits; class 0 means a red light applies to the ego
vehicle, class 1 means it does not.
"""
from __future__ import annotations

import numpy as np

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Polygon, Rectangle

# Undo the per-channel scaling the detection targets were stored with.
DETECTION_SCALE = np.array([1.0, 3.5, 3.5, 1.0, 1.0, 3.5, 2.0, 8.0])
BEV_GRID = 50
# (left, right, bottom, top) in metres, from convert_grid_to_xy on a 50x50 grid.
BEV_EXTENT = (-25.0, 25.0, -20.0, 30.0)
HORIZONS_S = (0.5, 1.0, 1.5, 2.0, 2.5)
# VLDrive pools the 50x50 BEV grid to 10x10, giving 100 of its 106 visual tokens.
VLDRIVE_POOL = 10
# A passenger car footprint, drawn to scale as the ego reference.
EGO_LENGTH_M = 4.5
EGO_WIDTH_M = 1.8

INK = "#1a1a1a"
MUTED = "#8a8a8a"
GRID = "#d9d9d9"
ACCENT = "#1f5fa9"
RED = "#c0392b"
GREEN = "#1e8449"


def use_paper_style():
    """Typography and line weights tuned for a two-column paper figure."""
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.linewidth": 0.8,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "axes.grid": False,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.fontsize": 8,
        "legend.frameon": False,
        "text.color": INK,
        "image.interpolation": "nearest",
    })


def forward_right(waypoints, waypoint_axes="left_negative_forward"):
    """Decode explicitly: legacy LMDrive callers default to their left axis.

    SensorRecord callers must pass right_negative_forward from the checkpoint.
    """
    waypoints = np.asarray(waypoints, dtype=np.float64).reshape(-1, 2)
    if waypoint_axes == "right_negative_forward":
        return -waypoints[:, 1], waypoints[:, 0]
    if waypoint_axes == "left_negative_forward":
        return -waypoints[:, 1], -waypoints[:, 0]
    raise ValueError("Unknown waypoint axes: " + str(waypoint_axes))


def denormalize_image(image):
    """ImageNet-normalised CHW tensor -> HWC float array in [0, 1] for imshow."""
    array = np.asarray(image, dtype=np.float32)
    if array.ndim != 3:
        raise ValueError("Expected a CHW image, got shape {}".format(array.shape))
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    return np.clip(array * std + mean, 0.0, 1.0).transpose(1, 2, 0)


def _as_grid(detection):
    """[2500, 8] or [50, 50, 8] raw head output -> decoded 50x50 grid.

    Channel 0 is trained with ``nn.BCEWithLogitsLoss`` (utils/losses.py), so it
    leaves the head as a logit and a sigmoid turns it into the objectness
    probability the thresholds expect.  LMDrive's own renderer compared the raw
    logit against 0.4/0.9 and so reported nothing on a real checkpoint whose
    logits sit near -1; measured max logits of -0.27 (official) and -0.01 (ours)
    become 0.43 and 0.50 once squashed.  Channels 1-7 stay linear regressions.
    """
    array = np.asarray(detection, dtype=np.float64)
    if array.shape == (BEV_GRID * BEV_GRID, 8):
        array = array.reshape(BEV_GRID, BEV_GRID, 8)
    if array.shape != (BEV_GRID, BEV_GRID, 8):
        raise ValueError("Detection map must be [2500, 8] or [50, 50, 8], got {}".format(
            np.asarray(detection).shape))
    decoded = array * DETECTION_SCALE
    decoded[:, :, 0] = 1.0 / (1.0 + np.exp(-array[:, :, 0]))
    return decoded


def _cell_centre(row, col):
    """LMDrive convert_grid_to_xy, as (lateral, forward) metres."""
    return col - 24.5, 29.5 - row


def decode_detection_map(detection, threshold=0.4, detection_axes="legacy_renderer_v1"):
    """Peak cells of the objectness channel -> oriented boxes in metres.

    Keeps a cell whose objectness is very high, or which is a local maximum
    above ``threshold``; the same rule LMDrive's renderer used, so a figure from
    this module and one from the original agree on what counts as an object.
    """
    if detection_axes not in ("legacy_renderer_v1",
                               "right_forward_offsets_right_negative_forward_heading"):
        raise ValueError("Unknown detection axes: " + str(detection_axes))
    grid = _as_grid(detection)
    probability = grid[:, :, 0]
    padded = np.zeros((BEV_GRID + 2, BEV_GRID + 2))
    padded[1:-1, 1:-1] = probability

    boxes = []
    for row in range(BEV_GRID):
        for col in range(BEV_GRID):
            value = probability[row, col]
            i, j = row + 1, col + 1
            peak = (value > 0.9) or (
                value > threshold
                and value > padded[i, j - 1] and value > padded[i, j + 1]
                and value > padded[i - 1, j] and value > padded[i + 1, j])
            if not peak:
                continue
            centre_lateral, centre_forward = _cell_centre(row, col)
            heading = np.array([grid[row, col, 3], grid[row, col, 4]], dtype=np.float64)
            extent = np.array([grid[row, col, 5], grid[row, col, 6]], dtype=np.float64)
            if detection_axes == "legacy_renderer_v1":
                if extent[0] < 1.5:
                    extent = extent * 1.5
            else:
                # Head orientation is [right, -forward]; plot is [right, forward].
                heading[1] *= -1.
                if not np.isfinite(extent).all() or np.any(extent <= 0):
                    continue
            boxes.append({
                "objectness": float(value),
                "lateral": float(centre_lateral + grid[row, col, 1]),
                "forward": float(centre_forward + grid[row, col, 2]),
                "heading": heading,
                "half_extent": extent,
                "speed": float(grid[row, col, 7]),
                "cell": (row, col),
            })
    boxes.sort(key=lambda box: box["objectness"], reverse=True)
    return boxes


def _box_corners(box):
    """Rotated rectangle corners in (lateral, forward) metres."""
    heading = box["heading"]
    norm = float(np.linalg.norm(heading))
    heading = heading / norm if norm > 1e-6 else np.array([0.0, 1.0])
    along = heading * box["half_extent"][0]
    across = np.array([-heading[1], heading[0]]) * box["half_extent"][1]
    centre = np.array([box["lateral"], box["forward"]])
    return np.array([centre + along + across, centre + along - across,
                     centre - along - across, centre - along + across])


def _draw_ego(axes, color=INK):
    """Ego footprint drawn to scale; pass a light colour over a dark colormap."""
    axes.add_patch(Rectangle((-EGO_WIDTH_M / 2, -EGO_LENGTH_M / 2), EGO_WIDTH_M, EGO_LENGTH_M,
                             facecolor="none", edgecolor=color, linewidth=1.1, zorder=5))
    axes.annotate("", xy=(0, EGO_LENGTH_M / 2 + 2.2), xytext=(0, EGO_LENGTH_M / 2),
                  arrowprops=dict(arrowstyle="-|>", color=color, linewidth=1.1), zorder=5)


def _titles(axes, title):
    axes.set_title(title, loc="left", pad=6)


def _caption(figure, text, color=MUTED, size=7.5):
    """Captions go in the figure's supxlabel slot, which constrained_layout
    reserves space for; free-floating axes text overlapped the axes instead."""
    if text:
        figure.supxlabel(text, fontsize=size, color=color, ha="left", x=0.01)


def _style_bev_axes(axes):
    axes.set_xlim(BEV_EXTENT[0], BEV_EXTENT[1])
    axes.set_ylim(BEV_EXTENT[2], BEV_EXTENT[3])
    axes.set_aspect("equal")
    # The PointPillars grid column grows towards the vehicle's right, so the
    # page's x-axis is +right; confirmed by the guardrail in a real frame, which
    # the LiDAR puts on the left (376 points) and the map renders on the left.
    axes.set_xlabel("lateral (m), + right")
    axes.set_ylabel("longitudinal (m), + forward")
    axes.set_xticks(np.arange(-25, 26, 10))
    axes.set_yticks(np.arange(-20, 31, 10))
    axes.grid(True, color=GRID, linewidth=0.4, alpha=0.8)
    axes.set_axisbelow(True)


def render_bev_detection(detection, *, threshold=0.4, title="BEV map", subtitle=None,
                         lidar_points=None, figsize=(5.0, 5.2),
                         detection_axes="legacy_renderer_v1"):
    """traffic_pred_head -> objectness heat map with the decoded oriented boxes.

    ``lidar_points`` is an optional [N, 2] array of (forward, right) metres in
    the ego frame, drawn as a faint scatter under the boxes.  It is what the
    sensor actually returned, so a box with no points beneath it, or a wall of
    points the head ignored, is visible at a glance -- which is the whole point
    of watching this while driving.
    """
    use_paper_style()
    grid = _as_grid(detection)
    boxes = decode_detection_map(detection, threshold=threshold, detection_axes=detection_axes)

    figure, axes = plt.subplots(figsize=figsize, constrained_layout=True)
    image = axes.imshow(np.clip(grid[:, :, 0], 0.0, 1.0), extent=BEV_EXTENT, origin="upper",
                        cmap="magma", vmin=0.0, vmax=1.0, zorder=1)
    if lidar_points is not None and len(lidar_points):
        cloud = np.asarray(lidar_points, dtype=np.float32).reshape(-1, 2)
        # The BEV page is +right across and +forward up; the cloud arrives as
        # (forward, right), matching datasets/lidar and the live adapter.
        axes.scatter(cloud[:, 1], cloud[:, 0], s=0.4, c="#5ec8f0", alpha=0.45,
                     linewidths=0, zorder=2, rasterized=True)
    for box in boxes:
        axes.add_patch(Polygon(_box_corners(box), closed=True, facecolor="none",
                               edgecolor=ACCENT, linewidth=1.0, zorder=4))
    _draw_ego(axes, color="white")
    _style_bev_axes(axes)
    _titles(axes, title)

    bar = figure.colorbar(image, ax=axes, fraction=0.046, pad=0.02)
    bar.set_label("objectness", rotation=90)
    bar.outline.set_linewidth(0.6)

    handles = [
        Line2D([], [], color=ACCENT, linewidth=1.0,
               label="detected object (n={})".format(len(boxes))),
        Line2D([], [], color="white", linewidth=1.1, label="ego vehicle"),
    ]
    if lidar_points is not None and len(lidar_points):
        handles.append(Line2D([], [], color="#5ec8f0", marker=".", linestyle="none",
                              label="LiDAR return"))
    axes.legend(handles=handles, loc="lower left", ncol=1, framealpha=0.0,
                labelcolor="white")
    _caption(figure, subtitle)
    return figure


def render_bev_features(bev_features, *, title="BEV feature activation", subtitle=None,
                        show_pooling=True, figsize=(5.0, 5.2)):
    """BEV decoder tokens -> per-cell L2 norm, the signal VLDrive actually pools.

    ``bev_features`` is the encoder's [2500, C] BEV output.  VLDrive average-pools
    this 50x50 grid to 10x10 to form 100 of its 106 visual tokens, so the dashed
    overlay shows exactly which cells become one token.

    The statistic is each cell's distance from the frame's mean BEV token,
    ``||f_i - mean(f)||``.  Plain ``||f_i||`` would be flat: the decoder ends in a
    LayerNorm, which pins every token to ``sqrt(C)`` (measured: mean 16.0015,
    std 0.0003 at C=256), so it carries no spatial information at all.
    """
    use_paper_style()
    array = np.asarray(bev_features, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] != BEV_GRID * BEV_GRID:
        raise ValueError("BEV features must be [2500, C], got {}".format(array.shape))
    deviation = array - array.mean(axis=0, keepdims=True)
    magnitude = np.linalg.norm(deviation, axis=1).reshape(BEV_GRID, BEV_GRID)

    figure, axes = plt.subplots(figsize=figsize, constrained_layout=True)
    image = axes.imshow(magnitude, extent=BEV_EXTENT, origin="upper", cmap="viridis", zorder=1)
    if show_pooling:
        step_x = (BEV_EXTENT[1] - BEV_EXTENT[0]) / VLDRIVE_POOL
        step_y = (BEV_EXTENT[3] - BEV_EXTENT[2]) / VLDRIVE_POOL
        for index in range(1, VLDRIVE_POOL):
            axes.axvline(BEV_EXTENT[0] + index * step_x, color="white", linewidth=0.4,
                         alpha=0.45, linestyle=(0, (3, 3)), zorder=3)
            axes.axhline(BEV_EXTENT[2] + index * step_y, color="white", linewidth=0.4,
                         alpha=0.45, linestyle=(0, (3, 3)), zorder=3)
    _draw_ego(axes, color="white")
    _style_bev_axes(axes)
    _titles(axes, title)

    bar = figure.colorbar(image, ax=axes, fraction=0.046, pad=0.02)
    bar.set_label("distance from the frame's mean BEV token (L2)", rotation=90)
    bar.outline.set_linewidth(0.6)

    if show_pooling:
        axes.legend(handles=[Line2D([], [], color=MUTED, linewidth=0.6, linestyle=(0, (3, 3)),
                                    label="10x10 pooling -> 100 VLDrive tokens")],
                    loc="lower left", framealpha=0.0, labelcolor="white")
    _caption(figure, subtitle)
    return figure


def render_traffic_light(logits, *, camera_image=None, title="Traffic light state",
                         subtitle=None, figsize=None):
    """traffic_light_pred_head -> class probabilities, beside the front camera.

    Class 0 is "a red light applies to the ego vehicle", class 1 is "it does
    not".  The predicted class is named in text as well as coloured, so the
    figure does not rely on red/green discrimination alone.
    """
    use_paper_style()
    values = np.asarray(logits, dtype=np.float64).reshape(-1)
    if values.size != 2:
        raise ValueError("Traffic-light logits must have 2 entries, got {}".format(values.size))
    shifted = values - values.max()
    probability = np.exp(shifted) / np.exp(shifted).sum()
    labels = ("red applies", "not red")
    colors = (RED, GREEN)
    predicted = int(np.argmax(probability))

    with_camera = camera_image is not None
    if figsize is None:
        figsize = (6.2, 2.8) if with_camera else (3.8, 2.6)
    if with_camera:
        figure, (camera_axes, axes) = plt.subplots(
            1, 2, figsize=figsize, constrained_layout=True, gridspec_kw={"width_ratios": [1.3, 1]})
        camera_axes.imshow(camera_image)
        camera_axes.set_xticks([])
        camera_axes.set_yticks([])
        for spine in camera_axes.spines.values():
            spine.set_linewidth(0.8)
        camera_axes.set_title("front camera", pad=6)
    else:
        figure, axes = plt.subplots(figsize=figsize, constrained_layout=True)

    positions = np.arange(2)
    axes.barh(positions, probability, height=0.55,
              color=[colors[i] if i == predicted else "#cfcfcf" for i in positions],
              edgecolor="none", zorder=2)
    for index, value in enumerate(probability):
        axes.text(min(value + 0.03, 0.97), index, "{:.3f}".format(value),
                  va="center", ha="left", fontsize=8,
                  color=INK if index == predicted else MUTED)
    axes.set_yticks(positions)
    axes.set_yticklabels(labels)
    axes.invert_yaxis()
    axes.set_xlim(0.0, 1.0)
    axes.set_xlabel("softmax probability")
    _titles(axes, title)
    axes.grid(True, axis="x", color=GRID, linewidth=0.4)
    axes.set_axisbelow(True)
    for side in ("top", "right", "left"):
        axes.spines[side].set_visible(False)

    caption = "predicted: {}  (p = {:.3f})".format(labels[predicted], probability[predicted])
    if subtitle:
        caption = caption + "          " + subtitle
    _caption(figure, caption, color=colors[predicted], size=8)
    return figure


def render_waypoints(waypoints, *, speed=None, target_point=None, title="Future waypoints",
                     subtitle=None, figsize=(3.8, 5.4), waypoint_axes="left_negative_forward"):
    """waypoints_generator -> the 0.5 ... 2.5 s path in the ego frame."""
    use_paper_style()
    forward, right = forward_right(waypoints, waypoint_axes=waypoint_axes)
    if forward.size != len(HORIZONS_S):
        raise ValueError("Expected {} waypoints, got {}".format(len(HORIZONS_S), forward.size))

    path_right = np.concatenate([[0.0], right])
    path_forward = np.concatenate([[0.0], forward])

    figure, axes = plt.subplots(figsize=figsize, constrained_layout=True)
    axes.plot(path_right, path_forward, color=ACCENT, linewidth=1.4, zorder=3,
              solid_capstyle="round")
    scatter = axes.scatter(right, forward, c=HORIZONS_S, cmap="viridis", s=34, zorder=4,
                           edgecolor="white", linewidth=0.6)
    for index, horizon in enumerate(HORIZONS_S):
        axes.annotate("{:.1f}s".format(horizon), (right[index], forward[index]),
                      textcoords="offset points", xytext=(6, -2), fontsize=7.5, color=INK)
    if target_point is not None:
        target = np.asarray(target_point, dtype=np.float64).reshape(2)
        target_forward, target_right = forward_right(target, waypoint_axes=waypoint_axes)
        axes.scatter(target_right, target_forward, marker="*", s=90, color=MUTED, zorder=4,
                     label="route target point")

    axes.add_patch(Rectangle((-EGO_WIDTH_M / 2, -EGO_LENGTH_M / 2), EGO_WIDTH_M, EGO_LENGTH_M,
                             facecolor="none", edgecolor=INK, linewidth=1.0, zorder=5))
    axes.add_patch(Circle((0, 0), 0.25, color=INK, zorder=6))

    # Equal aspect keeps the path metrically honest, so widen the lateral view
    # until the axes box has a drawable shape instead of a sliver.
    forward_span = max(float(forward.max()) + 5.0, 12.0) - min(-6.0, float(forward.min()) - 2.0)
    half_width = max(float(np.abs(right).max()) * 1.6, forward_span / 3.2, 5.0)
    axes.set_xlim(-half_width, half_width)
    axes.set_ylim(min(-6.0, float(forward.min()) - 2.0), max(float(forward.max()) + 5.0, 12.0))
    axes.set_aspect("equal")
    axes.set_xlabel("lateral (m), + right")
    axes.set_ylabel("longitudinal (m), + forward")
    _titles(axes, title)
    axes.grid(True, color=GRID, linewidth=0.4)
    axes.set_axisbelow(True)

    bar = figure.colorbar(scatter, ax=axes, fraction=0.046, pad=0.02, ticks=HORIZONS_S)
    bar.set_label("prediction horizon (s)", rotation=90)
    bar.outline.set_linewidth(0.6)

    travelled = float(np.hypot(np.diff(path_right), np.diff(path_forward)).sum())
    caption = "path length {:.1f} m".format(travelled)
    if speed is not None:
        caption = "ego speed {:.2f} m/s     ".format(float(speed)) + caption
    if subtitle:
        caption = caption + "\n" + subtitle
    _caption(figure, caption)
    if target_point is not None:
        axes.legend(loc="upper left", framealpha=0.0)
    return figure


def figure_to_bgr(figure, close=True):
    """Rasterise a figure to a BGR array, for showing instead of saving.

    The window and the PNG therefore come from the same renderer call, so what
    is watched live and what goes in the paper cannot drift apart.  Writing the
    file is what costs: on this machine a BEV figure takes about 93 ms to draw
    and rasterise, against 369 ms to draw and save with ``bbox="tight"``.
    """
    figure.canvas.draw()
    rgba = np.asarray(figure.canvas.buffer_rgba())
    bgr = np.ascontiguousarray(rgba[:, :, [2, 1, 0]])
    if close:
        plt.close(figure)
    return bgr


def save_figure(figure, stem, formats=("pdf", "png"), dpi=200, close=True):
    """Write one figure to every requested format, closing it unless asked not to.

    Pass ``close=False`` when the caller still needs the figure -- the live
    window rasterises the same figure it saves, so that the two can never show
    different things.
    """
    written = []
    for suffix in formats:
        path = stem.with_suffix("." + suffix)
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=dpi)
        written.append(path)
    if close:
        plt.close(figure)
    return written
