"""Hold the footprint of a parked object steady: report the median of its recent boxes instead of the latest one.

VE's box on a parked object changes from frame to frame: the centre moves by up to 0.8 m and the heading by tens of
degrees when the box flips between cells, so the planner sees an obstacle that slides and turns. Once a track is
labelled stationary its footprint cannot change, so each such box is replaced by the median centre, length and width
and the mean axis of the track's last HISTORY boxes (at least MIN_BOXES). Any other label clears the history, so a
starting vehicle is reported where it is as soon as the tracker or the resolver releases it.

Same-drive replay of 47 drives (45 tunnel cases, 2 full loops): frame-to-frame centre jump p90 0.15 -> 0.04 m, boxes
turning more than 10 degrees 2.7 -> 2.1 %, median centre error against the GT box 0.26 -> 0.22 m; moving vehicles
unchanged.
"""
import math

HISTORY = 15           # boxes (about 1.5 s)
MIN_BOXES = 5
GAP_S = 1.             # a track missing longer than this starts again
SKIP_SOURCES = ('static_map',)


class StationaryGeometry:
    def __init__(self):
        self.tracks = {}           # id -> [(x, y, axis rad mod pi, length, width)]
        self.last = {}             # id -> time of its last box
        self.generation = None

    def apply(self, packet):
        """Replace the footprint of stationary boxes of a tracked packet (in place) and return it."""
        now = packet['sent_ns'] / 1e9
        if packet.get('generation') != self.generation:
            self.tracks.clear()
            self.last.clear()
            self.generation = packet.get('generation')
        for identity in [k for k, t in self.last.items() if now - t > GAP_S]:
            self.tracks.pop(identity, None)
            del self.last[identity]
        for box in packet['boxes']:
            identity = box.get('track_id')
            if identity is None or identity < 0 or box.get('planning_speed_source') in SKIP_SOURCES:
                continue
            self.last[identity] = now
            if box.get('motion_state') != 'stationary':
                self.tracks.pop(identity, None)
                continue
            length, width, axis = box['length_m'], box['width_m'], math.radians(box['heading_deg'])
            if width > length:
                length, width, axis = width, length, axis + math.pi / 2
            history = self.tracks.setdefault(identity, [])
            history.append((*box['center_world'], axis % math.pi, length, width))
            del history[:-HISTORY]
            if len(history) < MIN_BOXES:
                continue
            median = [sorted(h[k] for h in history)[len(history) // 2] for k in (0, 1, 3, 4)]
            heading = math.degrees(.5 * math.atan2(sum(math.sin(2 * h[2]) for h in history),
                                                   sum(math.cos(2 * h[2]) for h in history)))
            box.update(center_world=median[:2], heading_deg=heading, planning_heading_deg=heading,
                       length_m=median[2], width_m=median[3])
        return packet
