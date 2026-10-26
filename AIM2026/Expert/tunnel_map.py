"""Accumulated occupancy map of standing obstacles in the Jamming (tunnel) section.

The tunnel planner judged every frame's VE boxes pass/fail, so a box that grew, changed track ID or flipped to
``uncertain`` blocked a path found a moment earlier, and every flip reset the plan. Standing obstacles are accumulated
here as in the static-obstacle section (static_map.py), in the section's forward-Y frame, while moving traffic keeps
the existing moving-object handling.

What is a mover: the tracker labels standing objects ``uncertain`` or ``moving`` on single frames (box jitter, heading
flips), so the label alone is not used. A track is a mover once its centre has travelled MOVER_DISPLACEMENT from
where it was first seen, or the tracker called it moving for MOVER_FRAMES frames in a row. Movers do not enter the map;
the cells of a mover seen before it was recognised are cleared by the map's free-space evidence.
The packet is not rewritten; the map is read by the Jamming planner.
"""
import math

from AIM2026.Expert.static_map import StaticObstacleMap

MOVER_DISPLACEMENT = 2.0  # a track whose centre moved this far from its first position is a mover [m]
MOVER_FRAMES = 5          # ... or one labelled moving this many frames in a row
TRACK_TIMEOUT_S = 5.      # forget a track not seen for this long
REGION_LATERAL = (-8.5, 3.)  # section-frame lateral extent of the evidence region (road: -6.5 .. +0.5) [m]
REGION_BEFORE = 25.       # evidence starts this far before the section start line [m]
REGION_AFTER = 15.        # ... and ends this far after its end line [m]


class TrackMotion:
    """Tracks that have shown sustained motion."""
    def __init__(self):
        self.tracks = {}

    def reset(self):
        self.tracks = {}

    def update(self, boxes, now):
        for box in boxes:
            identity = box.get('track_id')
            if identity is None:
                continue
            x, y = box['center_world']
            track = self.tracks.setdefault(identity, dict(first=(x, y), run=0, mover=False))
            track['run'] = track['run']+1 if box.get('motion_state') == 'moving' else 0
            track['seen'] = now
            if (math.hypot(x-track['first'][0], y-track['first'][1]) >= MOVER_DISPLACEMENT
                    or track['run'] >= MOVER_FRAMES):
                track['mover'] = True
        self.tracks = {k: v for k, v in self.tracks.items() if now-v['seen'] <= TRACK_TIMEOUT_S}

    def is_mover(self, identity):
        return bool(self.tracks.get(identity, {}).get('mover'))


class TunnelObstacleMap(StaticObstacleMap):
    def __init__(self, section=None):
        if section is None:
            from AIM2026.Expert.src.planning.jamming.section import JammingSection
            section = JammingSection()
        end = section.length+section.global_tail_m+REGION_AFTER
        super().__init__(x_range=REGION_LATERAL, y_range=(-REGION_BEFORE, end),
                         frame=(section.start[0], section.start[1], section.rotation))
        self.motion = TrackMotion()

    def reset(self):
        super().reset()
        if hasattr(self, 'motion'):
            self.motion.reset()

    def observe_packet(self, packet):
        """Accumulate the packet's standing obstacles inside the section (the packet is not changed)."""
        if packet.get('generation') != self.generation:
            self.reset()
            self.generation = packet.get('generation')
        boxes = []
        for box in packet['boxes']:
            if box.get('planning_speed_source') in ('static_memory', 'static_map'):
                continue
            x, y, _ = self.to_map(*box['center_world'])
            if self.contains(float(x), float(y)):
                boxes.append(box)
        self.motion.update(boxes, packet.get('sent_ns', 0)/1e9)
        self.observe(packet, [b for b in boxes if not self.motion.is_mover(b.get('track_id'))])
