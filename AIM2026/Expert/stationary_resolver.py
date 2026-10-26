"""Resolve 'uncertain' tracks that are parked, using a longer position history than the tracker's.

The tracker calls a box stationary only from the last 0.8 s of centres (radius <= 0.3 m, speed < 0.6 m/s) and clears
that history whenever the detector misses the object for more than 0.3 s. Far or partly seen parked objects have box
centres that wander by a few decimetres and are missed now and then, so most of their frames stay 'uncertain', and the
planner treats an uncertain box as a possible mover. Measured on a 5-minute drive, 59 % of the uncertain boxes belonged
to tracks older than 2 s whose whole-life displacement was about 0.2 m.

Here every track keeps up to WINDOW_S of world centres through gaps of up to GAP_S. An uncertain box becomes stationary
when, over at least MIN_SPAN_S, the median centre of the oldest third and of the newest third differ by at most
MAX_NET_M, the newest centres still sit at the window median, and the detector's own speed is low. A 'moving' label is
never changed, and the newest-centre test drops an object that starts to move within a few tenths of a second.
"""
import math
import statistics

WINDOW_S = 3.0         # centres kept per track
GAP_S = 1.0            # a track missing for longer starts a new history
MIN_SPAN_S = .6        # history needed before a decision (1.0 s / 5 samples: 0.8 %p more uncertain in the tunnel
MIN_SAMPLES = 4        # benchmark; 0.6 s labelled 1 of 897 moving-vehicle frames stationary on a full loop)
MAX_NET_M = .4         # oldest-third vs newest-third median centre
LATEST_SAMPLES = 3     # the newest centres ...
MAX_LATEST_M = .35     # ... must stay this close to the window median (or 10 % of the box length, if larger)
MAX_RAW_SPEED_MPS = 1.0  # median detector speed over the window
RELEASE_NET_M = .6       # once resolved, a track stays stationary until it moves this far (window ends) ...
RELEASE_LATEST_M = .45   # ... or its two newest centres leave the window median by this much: a small object (a pedestrian
                         # starting at 1.4 m/s) is released within about 0.4 s. Vehicle-sized boxes jitter more and a
VEHICLE_SIZE_M = 2.5     # starting vehicle is caught by the tracker's own 'moving' label, so they need 15 % of their
VEHICLE_RELEASE_SHARE = .15  # length (about 0.7 m)


def _median_point(points):
    return statistics.median(p[0] for p in points), statistics.median(p[1] for p in points)


class StationaryResolver:
    def __init__(self):
        self.tracks = {}       # id -> dict(last=t, samples=[(t, x, y, raw_speed)], resolved=bool)
        self.generation = None
        self.resolved = 0

    def apply(self, packet):
        """Relabel parked 'uncertain' boxes of a tracked packet (in place) and return it."""
        now = packet['sent_ns'] / 1e9
        if packet.get('generation') != self.generation:
            self.tracks.clear()
            self.generation = packet.get('generation')
        for identity in [k for k, v in self.tracks.items() if now - v['last'] > GAP_S]:
            del self.tracks[identity]
        for box in packet['boxes']:
            identity = box.get('track_id')
            if identity is None or identity < 0 or box.get('planning_speed_source') in ('static_memory', 'static_map'):
                continue
            track = self.tracks.setdefault(identity, dict(last=now, samples=[]))
            track['last'] = now
            if box.get('motion_state') == 'moving':
                track['samples'].clear()          # a mover starts a fresh history when it stops
                track['resolved'] = False
                continue
            samples = track['samples']
            samples.append((now, *box['center_world'], box.get('speed_mps')))
            while samples and now - samples[0][0] > WINDOW_S:
                samples.pop(0)
            if box.get('motion_state') == 'stationary':
                track['resolved'] = True          # the tracker agrees; keep it through its next uncertain frames
                continue
            if box.get('motion_state') != 'uncertain':
                continue
            # Hysteresis: deciding again every frame would flip the label back and forth on the same jitter.
            if track.get('resolved'):
                vehicle = max(box['length_m'], box['width_m']) >= VEHICLE_SIZE_M
                track['resolved'] = self._parked(samples, box, RELEASE_NET_M, RELEASE_LATEST_M,
                                                 VEHICLE_RELEASE_SHARE if vehicle else 0., latest_n=2)
            else:
                track['resolved'] = self._parked(samples, box)
            if not track['resolved']:
                continue
            box.update(motion_state='stationary', planning_speed_mps=0., speed_mps=0.,
                       planning_speed_source='stationary_long_window', motion_direction_observed=False)
            self.resolved += 1
        return packet

    @staticmethod
    def _parked(samples, box, max_net=MAX_NET_M, max_latest=MAX_LATEST_M, latest_share=.1, latest_n=LATEST_SAMPLES):
        if len(samples) < MIN_SAMPLES or samples[-1][0] - samples[0][0] < MIN_SPAN_S:
            return False
        third = max(1, len(samples) // 3)
        points = [s[1:3] for s in samples]
        old, new, middle = _median_point(points[:third]), _median_point(points[-third:]), _median_point(points)
        if math.dist(old, new) > max_net:
            return False
        latest = _median_point(points[-latest_n:])
        if math.dist(latest, middle) > max(max_latest, latest_share * max(box['length_m'], box['width_m'])):
            return False
        raw = [s[3] for s in samples if s[3] is not None and math.isfinite(s[3])]
        return not raw or statistics.median(raw) <= MAX_RAW_SPEED_MPS
