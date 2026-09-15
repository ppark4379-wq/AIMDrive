"""One window showing the paper figures as they are produced, live.

    from inference.live_view import LiveView
    view = LiveView(["camera", "bev_detection", "traffic_light", "waypoints"])
    view.show([("camera", bgr), ("bev_detection", bgr), ...])

A panel here is the identical figure that goes to the PNG, rasterised to pixels
instead of written to a file, so what is watched while driving and what goes in
the paper cannot drift apart.  Writing the file is what was slow: a BEV figure
takes about 93 ms to draw and rasterise and 369 ms to draw and save.

Keys, while the window has focus:

    q / Esc   close the window and stop the run
    s         save the frame on screen at full quality (PDF and PNG)
    space     freeze on the current frame; press again to follow along

Needs a display.  Under WSL that means WSLg, which Windows 11 provides; if
there is none, construction raises and the caller can fall back to files.
"""
from __future__ import annotations

import cv2
import numpy as np

BACKGROUND = (250, 250, 250)
INK = (40, 40, 40)
MUTED = (150, 150, 150)


class DisplayUnavailable(RuntimeError):
    """No display to open a window on."""


class LiveView:
    """A tiled window over the rendered panels, refreshed in place."""

    def __init__(self, names, title="VLDrive vision encoder", columns=2,
                 cell=(560, 580), scale=1.0):
        self.names = list(names)
        self.title = title
        self.columns = max(1, int(columns))
        self.cell = (int(cell[0] * scale), int(cell[1] * scale))
        # Bounds on the per-row height, so one very wide or very tall panel
        # cannot stretch its row out of proportion with the others.
        self.min_row = int(self.cell[1] * 0.55)
        self.max_row = int(self.cell[1] * 1.25)
        self.frozen = False
        self.save_requested = False
        self._closed = False
        self._last = None
        try:
            cv2.namedWindow(self.title, cv2.WINDOW_NORMAL)
            rows = -(-len(self.names) // self.columns)
            cv2.resizeWindow(self.title, self.cell[0] * self.columns,
                             self.cell[1] * rows + 28)
            cv2.imshow(self.title, self._placeholder())
            cv2.waitKey(1)
        except cv2.error as error:                       # no DISPLAY, no WSLg
            raise DisplayUnavailable(str(error))

    # ------------------------------------------------------------------ layout

    def _placeholder(self):
        rows = -(-len(self.names) // self.columns)
        canvas = np.full((self.cell[1] * rows + 28, self.cell[0] * self.columns, 3),
                         BACKGROUND, np.uint8)  # replaced by the first real frame
        cv2.putText(canvas, "waiting for the first frame...", (24, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, INK, 1, cv2.LINE_AA)
        return canvas

    def _scale(self, panel, height):
        source_h, source_w = panel.shape[:2]
        width = max(1, int(round(source_w * height / source_h)))
        # INTER_AREA downscales text and thin lines without the shimmer that
        # INTER_LINEAR gives on a figure full of 1px strokes.
        return cv2.resize(panel, (width, height), interpolation=(
            cv2.INTER_AREA if height < source_h else cv2.INTER_LINEAR))

    def _row(self, panels, target_width):
        """Lay a row out at one common height, filling the width exactly.

        A wide 16:9 camera beside a square figure would waste a third of the
        window if every panel were letterboxed into an identical cell, so the
        row picks the height at which the panels happen to fill the width.
        """
        aspects = [panel.shape[1] / panel.shape[0] for panel in panels]
        height = int(target_width / sum(aspects))
        height = max(self.min_row, min(height, self.max_row))
        scaled = [self._scale(panel, height) for panel in panels]
        row = np.hstack(scaled)
        if row.shape[1] > target_width:
            # The height clamp can push a row past the window; shrink the whole
            # row rather than cropping it, which would cut off an axis.
            row = cv2.resize(row, (target_width,
                                   max(1, int(round(row.shape[0] * target_width / row.shape[1])))),
                             interpolation=cv2.INTER_AREA)
        elif row.shape[1] < target_width:
            left = (target_width - row.shape[1]) // 2
            row = cv2.copyMakeBorder(row, 0, 0, left, target_width - row.shape[1] - left,
                                     cv2.BORDER_CONSTANT, value=BACKGROUND)
        return row

    def _compose(self, panels, status):
        lookup = dict(panels)
        target_width = self.cell[0] * self.columns
        rows = []
        for start in range(0, len(self.names), self.columns):
            present = [lookup[name] for name in self.names[start:start + self.columns]
                       if lookup.get(name) is not None]
            rows.append(self._row(present, target_width) if present else
                        np.full((self.min_row, target_width, 3), BACKGROUND, np.uint8))
        canvas = np.vstack(rows)

        bar = np.full((28, canvas.shape[1], 3), BACKGROUND, np.uint8)
        cv2.line(bar, (0, 0), (bar.shape[1], 0), (225, 225, 225), 1)
        cv2.putText(bar, status, (12, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.48, INK, 1, cv2.LINE_AA)
        hint = "q quit   s save   space freeze" + ("   [FROZEN]" if self.frozen else "")
        (text_w, _), _ = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.putText(bar, hint, (bar.shape[1] - text_w - 12, 19), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, MUTED if not self.frozen else INK, 1, cv2.LINE_AA)
        return np.vstack([canvas, bar])

    # ------------------------------------------------------------------- frame

    def show(self, panels, status=""):
        """Draw one frame and read the keyboard.  Returns False once closed."""
        if self._closed:
            return False
        if not self.frozen or self._last is None:
            self._last = self._compose(panels, status)
        else:
            # Keep the status bar live even while the picture is held, so a
            # frozen window is never mistaken for a stalled one.
            held = self._last[:-28]
            self._last = np.vstack([held, self._compose([], status)[-28:]])
        cv2.imshow(self.title, self._last)
        return self._pump()

    def _pump(self):
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            self.close()
            return False
        if key == ord("s"):
            self.save_requested = True
        if key == ord(" "):
            self.frozen = not self.frozen
        # The window manager's close button leaves the window gone but waitKey
        # happy, so ask the property directly.
        try:
            if cv2.getWindowProperty(self.title, cv2.WND_PROP_VISIBLE) < 1:
                self.close()
                return False
        except cv2.error:
            self.close()
            return False
        return True

    def take_save_request(self):
        requested, self.save_requested = self.save_requested, False
        return requested

    @property
    def closed(self):
        return self._closed

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            cv2.destroyWindow(self.title)
            cv2.waitKey(1)
        except cv2.error:
            pass


def camera_panel(image_rgb, label="front camera (raw)"):
    """The undecorated front camera, captioned to match the figure panels."""
    panel = cv2.cvtColor(np.ascontiguousarray(image_rgb), cv2.COLOR_RGB2BGR)
    # Deep enough that the caption survives being scaled into a row.
    scale = panel.shape[1] / 640.0
    bar = np.full((int(46 * scale), panel.shape[1], 3), BACKGROUND, np.uint8)
    cv2.putText(bar, label, (int(12 * scale), int(32 * scale)), cv2.FONT_HERSHEY_SIMPLEX,
                0.72 * scale, INK, max(1, int(round(1.4 * scale))), cv2.LINE_AA)
    return np.vstack([bar, panel])
