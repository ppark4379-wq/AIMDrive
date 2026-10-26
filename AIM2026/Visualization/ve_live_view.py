"""Live bird's-eye view of what the VE sends to the Expert (no ROS, no simulator access, read-only).

It follows the newest run's ``inputs.jsonl`` (written by ``Expert/main.py --visualize``) and draws, ego-centred with the
vehicle driving up the screen:
  box fill   blue = stationary, orange = uncertain, red = moving (arrow = planning speed along the planning heading)
  label      track id, speed (m/s), motion state
  grey frame the VE BEV window (right +-25 m, forward -20..+30 m)
  top left   frame number, age of the data, ego speed, boxes

Run:   python AIM2026/Visualization/ve_live_view.py                 # newest run, opens a window
       python AIM2026/Visualization/ve_live_view.py --run <run dir>  # a specific run folder (contains inputs.jsonl)
       python AIM2026/Visualization/ve_live_view.py --png /tmp/ve.png  # no window: rewrite one PNG a few times a second
Needs matplotlib only. Close the window or press Ctrl+C to stop.
"""
import argparse
import glob
import json
import math
import os
import time
from pathlib import Path

import matplotlib

REPO = Path(__file__).resolve().parents[2]
BEV = (-25., 25., -20., 30.)            # right min/max, forward min/max (VE BEV window)
COLORS = {'stationary': '#1f6fd1', 'uncertain': '#e8851a', 'moving': '#d8372c'}


def newest_run():
    patterns = [str(REPO / 'AIM2026/outputs/expert/*/inputs.jsonl'),
                str(REPO / 'MoraiDrive/scenario_runner/.runtime/ve_gps_full_route/*/driver/inputs.jsonl')]
    files = [f for p in patterns for f in glob.glob(p)]
    return Path(max(files, key=os.path.getmtime)).parent if files else None


def last_line(path):
    """Last complete JSON line of a growing file without reading all of it."""
    with open(path, 'rb') as stream:
        stream.seek(0, os.SEEK_END)
        size = stream.tell()
        stream.seek(max(0, size - 400_000))
        lines = stream.read().splitlines()
    for raw in reversed(lines):
        try:
            return json.loads(raw)
        except ValueError:
            continue
    return None


def to_ego(point, ego):
    """World (x, y) -> (right, forward) in the ego frame."""
    dx, dy = point[0] - ego['x'], point[1] - ego['y']
    c, s = math.cos(ego['yaw_grid_rad']), math.sin(ego['yaw_grid_rad'])
    return -(-dx * s + dy * c), dx * c + dy * s


def rotated_rect(center, heading_deg, length, width, ego):
    """Corners (right, forward) of a box whose long side follows ``heading_deg`` (world, degrees)."""
    rel = math.radians(heading_deg) - ego['yaw_grid_rad']
    u = (-math.sin(rel), math.cos(rel))            # along the box, in (right, forward)
    v = (u[1], -u[0])
    cx, cy = center
    return [(cx + a * u[0] * length / 2 + b * v[0] * width / 2, cy + a * u[1] * length / 2 + b * v[1] * width / 2)
            for a, b in ((1, 1), (1, -1), (-1, -1), (-1, 1))]


def draw(ax, row, age):
    from matplotlib.patches import Polygon, Rectangle
    ax.clear()
    ax.add_patch(Rectangle((BEV[0], BEV[2]), BEV[1] - BEV[0], BEV[3] - BEV[2], fill=False, ec='#999', lw=1, ls='--'))
    for r in (10, 20, 30):
        ax.add_patch(plt_circle(r))
    ego = row['ego_pose']
    ax.add_patch(Polygon([(-.95, -1.1), (.95, -1.1), (.95, 3.5), (-.95, 3.5)], closed=True, fc='#333', ec='k', zorder=5))
    boxes = row['boxes']
    for b in boxes:
        state = b.get('motion_state') or 'uncertain'
        colour = COLORS.get(state, '#888')
        centre = to_ego(b['center_world'], ego)
        corners = rotated_rect(centre, b['heading_deg'], b['length_m'], b['width_m'], ego)
        ax.add_patch(Polygon(corners, closed=True, fc=colour, alpha=.28, ec=colour, lw=1.8, zorder=3))
        speed = b.get('planning_speed_mps')
        if speed and speed > .3 and state == 'moving':
            rel = math.radians(b.get('planning_heading_deg', b['heading_deg'])) - ego['yaw_grid_rad']
            ax.annotate('', xy=(centre[0] - math.sin(rel) * speed, centre[1] + math.cos(rel) * speed), xytext=centre,
                        arrowprops=dict(arrowstyle='-|>', color=colour, lw=1.6), zorder=4)
        ax.text(centre[0], centre[1], '%s\n%s%s' % (b.get('track_id'), state[:4], '' if speed is None else ' %.1f' % speed),
                fontsize=7, ha='center', va='center', color='k', zorder=6)
    ax.set_xlim(BEV[0] - 2, BEV[1] + 2)
    ax.set_ylim(BEV[2] - 2, BEV[3] + 2)
    ax.set_aspect('equal')
    ax.set_xlabel('right [m]', fontsize=8)
    ax.set_ylabel('forward [m]', fontsize=8)
    ax.tick_params(labelsize=7)
    ready = row.get('ready', True)
    ax.set_title('frame %s   age %.1f s   ego %.1f m/s   boxes %d   %s' % (
        row.get('sequence', '?'), age, abs(ego.get('speed_mps') or 0.), len(boxes), '' if ready else 'NOT READY: %s' % row.get('reason')),
        fontsize=9, color='k' if ready and age < 1.5 else '#c00', loc='left')


def plt_circle(radius):
    from matplotlib.patches import Circle
    return Circle((0, 0), radius, fill=False, ec='#ccc', lw=.7)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run', type=Path, help='run folder holding inputs.jsonl (default: the newest run)')
    parser.add_argument('--png', type=Path, help='no window: keep rewriting this PNG')
    parser.add_argument('--hz', type=float, default=8.)
    args = parser.parse_args()
    if args.png:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 7.4))
    if not args.png:
        plt.ion()
        fig.show()
    folder, shown_sequence, last_seen = None, None, time.time()
    print('[VE view] waiting for a run ...', flush=True)
    while args.png or plt.fignum_exists(fig.number):
        target = args.run or newest_run()
        if target and target != folder and (target / 'inputs.jsonl').is_file():
            folder, shown_sequence = target, None
            print('[VE view] following %s' % folder, flush=True)
        row = last_line(folder / 'inputs.jsonl') if folder else None
        if row and row.get('ego_pose') and row.get('sequence') != shown_sequence:
            shown_sequence = row.get('sequence')
            last_seen = time.time()
            draw(ax, row, 0.)
        if row and row.get('ego_pose') and time.time() - last_seen > 1.5:
            ax.set_title('NO NEW FRAME for %.0f s (the run ended or VE stalled)' % (time.time() - last_seen),
                         fontsize=9, color='#c00', loc='left')
        if args.png:
            fig.savefig(str(args.png), dpi=90)
            time.sleep(1. / args.hz)
        else:
            fig.canvas.draw_idle()
            plt.pause(1. / args.hz)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
