"""Left-arrow check on the front camera for the left-turn signal (C1256W000052).

The three-phase head reads that signal from the encoder's traffic-light token, which keeps the lit red lamp but little of
the green arrow beside it: driven to the stop line it read the arrow as red (10-28). The lamp itself is unambiguous in the
front image at the stop position (11-16 m before the line): red + arrow lights a red and a green lamp side by side in one
housing, straight green a green lamp alone, yellow and red one lamp each. Checked against 1,777 labelled frames of 15
Expert-driven approaches (SUNNY/FOGGY, 11-15 h): at the stop position every disagreement with the read-back colour lay
within 1 s of a colour change, where the rendered lamp and the label are a moment apart.

Only the upper part of the image is searched (the lamp hangs above the road; the vehicle's own bonnet and the road are
left out), and a red and a green blob must sit level with each other and close together, so a red lamp and a green lamp
of two different heads are not taken for an arrow.
"""
import cv2
import numpy as np

ROI = (0, 260, 300, 1280)          # rows, columns of a 1280x720 front image
MIN_BLOB = 8                       # lit pixels of one lamp: 54+ at 20 m, 190 at 13 m
MAX_DX, MAX_DY = 140, 14           # red and arrow lamps of one housing: about 60-80 px apart, same row
LAMP_TOP_MIN = 50                  # a lamp whose top is above this row is leaving the image (vehicle past ~11 m):
                                   # at the stop position (13 m) the lamp top is at rows 54-96, at 9-10 m at 0-9


def _blobs(mask):
    count, _, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    return [centroids[i] for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= MIN_BLOB]


def _tops(mask):
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    return [int(stats[i, cv2.CC_STAT_TOP]) + ROI[0] for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= MIN_BLOB]


def lamp_masks(bgr):
    rows, cols = slice(ROI[0], ROI[1]), slice(ROI[2], ROI[3])
    hsv = cv2.cvtColor(bgr[rows, cols], cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    lit = (s > 120) & (v > 170)
    return dict(red=lit & ((h < 8) | (h > 172)), green=lit & (h > 50) & (h < 90), yellow=lit & (h > 18) & (h < 35))


def left_arrow(bgr):
    """BGR front image -> {'arrow': bool, 'lamp_in_view': bool, 'lamp_top': row|None, 'red'/'green'/'yellow': px}.

    ``lamp_in_view``: some lamp is lit with its top below LAMP_TOP_MIN, i.e. fully in the image, so a missing arrow
    is a reading (stop) rather than a lamp leaving the image.
    """
    if bgr is None or bgr.ndim != 3 or bgr.shape[0] < ROI[1] or bgr.shape[1] < ROI[3]:
        return dict(arrow=False, lamp_in_view=False, lamp_top=None, red=0, green=0, yellow=0, valid=False)
    masks = lamp_masks(bgr)
    reds, greens = _blobs(masks['red']), _blobs(masks['green'])
    arrow = any(abs(r[1] - g[1]) <= MAX_DY and 0 < g[0] - r[0] <= MAX_DX for r in reds for g in greens)
    tops = [t for m in masks.values() for t in _tops(m)]
    top = min(tops) if tops else None
    return dict(arrow=bool(arrow), lamp_in_view=top is not None and top >= LAMP_TOP_MIN, lamp_top=top, valid=True,
                **{k: int(m.sum()) for k, m in masks.items()})
