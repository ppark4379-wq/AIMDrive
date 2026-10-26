"""Give one track id to the two box hypotheses VE alternates between on one parked object.

On thin or long parked objects (barriers, bollards) VE often reports a piece of the object in one frame and the whole
object in the next, 0.6-2 m apart and up to 4x different in size. Each frame holds only one of them, but the tracker keeps
both tracks alive for its 0.75 s gap and gives every frame to the nearer one, so the id flips 1-2-1-2. In the 15-case
tunnel benchmark this was 76 % of the id changes, and every change restarted the stationary decision: 43 % of the
remaining 'uncertain' frames came within 1.5 s after one.

Here a track whose box overlaps the last box of an older, recently seen track (both footprints grown by MARGIN_M) takes
that track's id, provided neither moves and the two were never reported in the same frame side by side: two real
neighbours (a row of barrels) are seen together without touching and are never merged, while VE's piece and whole box
of one object, when they do come in one frame, overlap. The original id stays in 'source_track_id'.
"""
import math

from AIM2026.Expert.box_dedup import _footprint
from AIM2026.Inference.vision.postprocess import _intersection, _rectangle

KEEP_S = 1.            # an older track is a candidate this long after its last box (tracker gap .75 s)
MARGIN_M = .5          # footprints grown by this on every side must overlap
MAX_SPEED_MPS = 1.     # VE speed above this is not merged
SKIP_SOURCES = ('static_memory', 'static_map')


def _grown(box):
    a = math.radians(box['heading_deg'])
    return _rectangle(box['center_world'], [math.cos(a), math.sin(a)], box['length_m'] + 2 * MARGIN_M,
                      box['width_m'] + 2 * MARGIN_M)


def _touch(a, b):
    try:
        return _intersection(_footprint(a), _footprint(b)) > 0.
    except (ValueError, KeyError, TypeError):
        return False


def _calm(box):
    speed = box.get('speed_mps')
    return box.get('motion_state') != 'moving' and (speed is None or not math.isfinite(speed) or speed <= MAX_SPEED_MPS)


class TrackAliaser:
    def __init__(self):
        self.alias = {}        # track id -> id it is reported as
        self.tracks = {}       # reported id -> dict(first, last, box, calm)
        self.together = set()  # frozenset({a, b}) of reported ids seen in one frame
        self.generation = None
        self.merged = 0

    def apply(self, packet):
        """Rewrite track ids of a tracked packet in place and return it."""
        now = packet['sent_ns'] / 1e9
        if packet.get('generation') != self.generation:
            self.alias.clear(); self.tracks.clear(); self.together.clear()
            self.generation = packet.get('generation')
        gone = {k for k, v in self.tracks.items() if now - v['last'] > KEEP_S}
        for k in gone:
            del self.tracks[k]
        self.alias = {k: v for k, v in self.alias.items() if v not in gone}
        boxes = [b for b in packet['boxes'] if (b.get('track_id') if b.get('track_id') is not None else -1) >= 0
                 and b.get('planning_speed_source') not in SKIP_SOURCES]
        reported = [self.alias.get(b['track_id'], b['track_id']) for b in boxes]
        # Seen in one frame without touching = two real objects. Two boxes that overlap are one object reported twice.
        for i, a in enumerate(reported):
            for j in range(i + 1, len(boxes)):
                if a != reported[j] and not _touch(boxes[i], boxes[j]):
                    self.together.add(frozenset((a, reported[j])))
        present = set(reported)
        for box, current in zip(boxes, reported):
            ident = box['track_id']
            if current == ident and _calm(box):
                first = self.tracks[ident]['first'] if ident in self.tracks else now
                try:
                    shape = _grown(box)
                except (ValueError, KeyError, TypeError):
                    continue
                candidates = [k for k, v in self.tracks.items()
                              if k != ident and k not in present and v['calm'] and v['first'] < first
                              and frozenset((k, ident)) not in self.together and _intersection(shape, v['shape']) > 0.]
                if candidates:
                    root = min(candidates, key=lambda k: self.tracks[k]['first'])
                    self.alias[ident] = root
                    self.tracks.pop(ident, None)
                    self.merged += 1
        # two boxes of one frame must never share an id: the larger keeps it, the other goes back to its own
        owner = {}
        for box in sorted(boxes, key=lambda b: -b['length_m'] * b['width_m']):
            ident = box['track_id']
            reported_id = self.alias.get(ident, ident)
            if reported_id in owner:
                self.alias.pop(ident, None)
                self.together.add(frozenset((reported_id, ident)))
                reported_id = ident
            owner[reported_id] = ident
            if reported_id != ident:
                box['source_track_id'] = ident
                box['track_id'] = reported_id
            try:
                shape = _grown(box)
            except (ValueError, KeyError, TypeError):
                continue
            entry = self.tracks.get(reported_id)
            # a track once labelled moving never takes others in; a single high raw VE speed (often on the first frame)
            # only keeps that frame's box from being merged
            self.tracks[reported_id] = dict(first=entry['first'] if entry else now, last=now, box=box, shape=shape,
                                            calm=box.get('motion_state') != 'moving' and (entry['calm'] if entry else True))
        return packet
