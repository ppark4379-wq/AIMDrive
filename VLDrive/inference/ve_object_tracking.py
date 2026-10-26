"""Short-lived world-space tracks from VE boxes, without simulator labels.

Only observed boxes are emitted. Missing detections may retain an association
for 0.75 s but never produce a phantom obstacle. Box geometry is not smoothed
unless ``static_geometry_fusion`` is enabled; then only confirmed stationary
tracks publish a fused footprint (see ``_StaticGeometry``).
Motion is estimated from publisher-time samples, not synchronized sensor time.
VE scalar speed alone cannot establish motion or predict association positions.
Unconfirmed tracks retain a valid VE speed as a provisional estimate. Motion
confidence and speed are separate: lack of history must not turn a closing car
into a stationary prediction. Missing speed remains None until observed.
"""
from collections import deque
import math

import numpy as np
from scipy.optimize import linear_sum_assignment


def _position_motion(history):
    """Require sustained, coherent displacement before trusting a velocity.

    A large speed regressed by VE is not independent motion evidence. Neither
    is the distance travelled by a box that oscillates around a fixed object.
    The path/displacement and line-fit checks reject that latter case. This
    cannot distinguish actual motion from a sustained, coherent localization
    drift, which must be addressed in sensor/model geometry separately.
    """
    values = np.asarray(history, dtype=float)
    times = values[:, 0] - values[0, 0]
    xy = values[:, 1:]
    span = float(times[-1])
    center = np.median(xy, axis=0)
    radius = float(np.linalg.norm(xy - center, axis=1).max())
    latest_deviation = float(np.linalg.norm(xy[-1]-np.median(xy[:-1], axis=0))) if len(xy) > 1 else 0.
    result = dict(span_s=span, radius_m=radius, speed_mps=0., velocity=(0., 0.),
                  coherent=False, residual_m=0., path_efficiency=0., displacement_m=0.,
                  latest_deviation_m=latest_deviation)
    if len(values) < 4 or span < .3 - 1e-9:
        return result
    centered_times = times - times.mean()
    velocity = (centered_times[:, None] * (xy - xy.mean(axis=0))).sum(axis=0) / (centered_times @ centered_times)
    fitted = xy.mean(axis=0) + centered_times[:, None] * velocity
    residual = float(np.sqrt(np.mean(np.sum((xy-fitted)**2, axis=1))))
    net = float(np.linalg.norm(xy[-1]-xy[0]))
    path = float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())
    efficiency = net / max(path, 1e-9)
    speed = float(np.linalg.norm(velocity))
    coherent = (speed >= .8 and net >= .35 and efficiency >= .8
                and residual <= max(.12, .15*net))
    result.update(speed_mps=speed, velocity=tuple(map(float, velocity)),
                  coherent=bool(coherent), residual_m=residual, path_efficiency=efficiency,
                  displacement_m=net)
    return result


class _StaticGeometry:
    """World-frame footprint fusion for one non-moving track.

    Per-frame VE boxes of a static object scatter more than they are biased,
    so a robust fusion over distinct viewpoints reduces footprint error and
    frame-to-frame jitter. Samples are taken per ego travel, not per frame,
    so a stopped ego cannot let one view dominate. Near observations weigh
    more, but below ``FREEZE_RANGE_M`` boxes are partial/unstable and no new
    samples are taken. Boxes are canonicalized (length >= width, axis mod
    180 deg) since an OBB footprint is invariant to those changes.
    """
    STEP_M = .5
    TIME_STEP_S = .3       # sampling without an ego pose
    FREEZE_RANGE_M = 8.
    MAX_SAMPLES = 60
    MIN_SAMPLES = 3

    def __init__(self):
        self.samples = []
        self.last_ego = self.last_time = None

    def add(self, box, now, ego):
        center = np.asarray(box['center_world'], dtype=float)
        length, width = float(box['length_m']), float(box['width_m'])
        axis = math.radians(box['heading_deg'])
        if width > length:
            length, width, axis = width, length, axis + math.pi/2
        weight = 1.
        if ego is not None:
            ego = np.asarray(ego, dtype=float)
            distance = float(np.linalg.norm(center - ego))
            if self.samples and distance < self.FREEZE_RANGE_M:
                return
            if self.last_ego is not None and np.linalg.norm(ego - self.last_ego) < self.STEP_M:
                return
            weight = 1. / max(distance, 5.)**2
            self.last_ego = ego
        else:
            if self.last_time is not None and now - self.last_time < self.TIME_STEP_S:
                return
            self.last_time = now
        self.samples.append((center, axis, length, width, weight))
        del self.samples[:-self.MAX_SAMPLES]

    def fused(self):
        if len(self.samples) < self.MIN_SAMPLES:
            return None
        centers = np.array([s[0] for s in self.samples])
        axes = np.array([s[1] for s in self.samples])
        weights = np.array([s[4] for s in self.samples])

        def median(values):
            order = np.argsort(values)
            cumulative = np.cumsum(weights[order])
            return float(values[order][np.searchsorted(cumulative, cumulative[-1] / 2)])
        axis = .5 * math.atan2(float(weights @ np.sin(2*axes)), float(weights @ np.cos(2*axes)))
        return dict(center_world=[median(centers[:, 0]), median(centers[:, 1])],
                    heading_deg=math.degrees(axis),
                    length_m=median(np.array([s[2] for s in self.samples])),
                    width_m=median(np.array([s[3] for s in self.samples])))


def _ego_xy(packet):
    pose = packet.get('ego_pose')
    if not pose:
        return None
    try:
        xy = (float(pose['x']), float(pose['y']))
    except (KeyError, TypeError, ValueError):
        return None
    return xy if all(math.isfinite(v) for v in xy) else None


STATIC_HOLD_RADIUS_M = .6
STATIC_BREAK_RADIUS_M = .9           # an established stationary object must travel this far from its anchor to count as moving
STATIC_HOLD_MIN_STATIONARY_S = 3.   # only an object that has stood still this long is held through weak evidence


class VeObjectTracker:
    @staticmethod
    def _moving_fragment_match(track, box, predicted, dt):
        """A partial body must not discard a coherent moving track's history.

        This only relaxes the size-ratio gate. Independent displacement still
        determines motion/speed below, and raw geometry remains untouched.
        Tight centre/axis/containment gates restrict this exception to recent
        observations along the measured trajectory, rather than nearby boxes.
        """
        if (track.get('state') != 'moving' or dt > .25
                or np.linalg.norm(track.get('velocity', (0., 0.))) < 2.):
            return False
        old = track['box']
        if min(old['length_m'], box['length_m']) < 1. or min(old['width_m'], box['width_m']) < .8:
            return False
        yaw = math.atan2(track['velocity'][1], track['velocity'][0])
        if any(abs(math.cos(math.radians(b['heading_deg'])-yaw)) < math.cos(math.radians(15.))
               for b in (old, box)):
            return False
        delta = np.asarray(box['center_world']) - predicted
        if np.linalg.norm(delta) > 1.5:
            return False
        along = abs(delta[0]*math.cos(yaw) + delta[1]*math.sin(yaw))
        across = abs(-delta[0]*math.sin(yaw) + delta[1]*math.cos(yaw))
        # Either a full box shrank to a fragment, or the full body reappeared.
        # Require containment on both axes, not simply touching rectangles.
        return all(offset + .5*min(old[key], box[key]) <= .5*max(old[key], box[key]) + .25
                   for offset, key in ((along, 'length_m'), (across, 'width_m')))

    def _support_overlapping_motion(self, boxes, now):
        """Retain both footprints of a duplicate detection, without inventing a stop.

        Only substantial overlap with a separately confirmed moving footprint,
        aligned axes, matching size and agreeing model speeds supplies evidence.
        Nearby distinct cars and a low-speed/confirmed stationary box are excluded.
        This does not remove geometry or mark motion as independently observed.
        """
        if not self.moving_carry_s:
            return
        moving = [b for b in boxes if b['motion_state'] == 'moving'
                  and b['track_span_s'] >= .5 and b['planning_speed_mps'] >= 2.]
        for box in boxes:
            if (box['motion_state'] != 'uncertain' or box['planning_speed_source'] != 've_provisional'
                    or box['speed_mps'] < 2. or box['length_m'] < 3.):
                continue
            for other in moving:
                speed = other['planning_speed_mps']
                if abs(box['speed_mps']-speed) > max(1.5, .25*speed):
                    continue
                yaw = math.radians(other['planning_heading_deg'])
                if abs(math.cos(math.radians(box['heading_deg'])-yaw)) < math.cos(math.radians(5.)):
                    continue
                if any(max(box[k],other[k])/max(.1,min(box[k],other[k])) > 1.35
                       for k in ('length_m','width_m')):
                    continue
                dx,dy = (box['center_world'][i]-other['center_world'][i] for i in range(2))
                along = abs(dx*math.cos(yaw)+dy*math.sin(yaw))
                across = abs(-dx*math.sin(yaw)+dy*math.cos(yaw))
                overlap = max(0.,min(box['length_m'],other['length_m'],
                                    (box['length_m']+other['length_m'])/2.-along)) * max(
                    0.,min(box['width_m'],other['width_m'],(box['width_m']+other['width_m'])/2.-across))
                if overlap < .4*min(box['length_m']*box['width_m'],other['length_m']*other['width_m']):
                    continue
                box['planning_speed_source'] = 'overlap_supported'
                box['motion_support_track_id'] = other['track_id']
                box['planning_heading_deg'] = other['planning_heading_deg']
                velocity = (box['speed_mps']*math.cos(yaw),box['speed_mps']*math.sin(yaw))
                track = self.tracks[box['track_id']]
                track['velocity'] = velocity
                track['carry'] = (now,box['speed_mps'],velocity,other['planning_heading_deg'],velocity)
                break

    def __init__(self, max_gap=.75, static_geometry_fusion=False, moving_carry_s=0., slow_window_s=0., stationary_hold_s=0., provisional_static_mps=0.):
        """``moving_carry_s``: a track that was confirmed moving and only lost its evidence (no stationary
        evidence) keeps its last planning speed/velocity for this long. Slow vehicles (1-2 m/s) otherwise flip
        to ``uncertain`` and fall back to the provisional model speed."""
        if not 0. <= provisional_static_mps <= .6:
            raise ValueError('provisional_static_mps must be within 0..0.6 m/s')
        if not 0. <= stationary_hold_s <= 6.:
            raise ValueError('stationary_hold_s must be within 0..6 s')
        if not 0. <= moving_carry_s <= 2.:
            raise ValueError('moving_carry_s must be within 0..2 s')
        if slow_window_s and not .8 < slow_window_s <= 2.4:
            raise ValueError('slow_window_s must be 0 (off) or within 0.8..2.4 s')
        self.moving_carry_s = float(moving_carry_s)
        # ``slow_window_s``: second, longer evidence window used only for slow movers (0.8-3 m/s). A vehicle 25-30 m
        # away at the edge of the BEV has box-centre scatter of a few decimetres, so a 0.8 s window cannot confirm
        # 1-2 m/s; over ~1.6 s the same scatter is small against the displacement. Slow vehicles also stop in a
        # few metres, so the longer confirmation delay does not cost safety margin.
        self.slow_window_s = float(slow_window_s)
        # ``stationary_hold_s``: a track confirmed stationary keeps that state through short stretches of weak evidence
        # (box-centre jitter, a re-created track) while the box stays within ``STATIC_HOLD_RADIUS_M`` of its anchor and no
        # coherent motion is seen. The Expert's Jamming planner treats every non-stationary VE box as a mover, drops it
        # from the static scene and resets its plan on each change, so flicker alone can stop a bypass from ever finishing.
        self.stationary_hold_s = float(stationary_hold_s)
        # ``provisional_static_mps``: a young track (>= 0.2 s) whose raw VE speed is at most this and whose centre has not
        # moved is called stationary at once instead of waiting 0.5 s as ``uncertain``. The Expert treats ``uncertain`` as a
        # mover and resets its plan when the label later flips, so each newly seen parked object delays the avoidance plan.
        self.provisional_static_mps = float(provisional_static_mps)
        self.static_memory = []
        self.max_gap = max_gap
        self.static_geometry_fusion = static_geometry_fusion
        self.tracks = {}
        self.serial = 0
        self.generation = None
        self.last_time = None

    def _recall_static(self, box, now):
        center = np.asarray(box['center_world'], dtype=float)
        best = None
        for index, memory in enumerate(self.static_memory):
            old, anchor = memory['track'], np.asarray(memory['track']['anchor'], dtype=float)
            if memory['identity'] in self.tracks:
                continue
            size = max((max(box[k], old['box'][k]) + .5) / (min(box[k], old['box'][k]) + .5) for k in ('length_m', 'width_m'))
            distance = float(np.linalg.norm(center - anchor))
            if distance <= STATIC_HOLD_RADIUS_M and size <= 2.:
                if best is None or distance < best[0]:
                    best = (distance, index)
        if best is None:
            return None
        memory = self.static_memory.pop(best[1])
        track = memory['track']
        track.update(history=deque(maxlen=20), state='uncertain', velocity=(0., 0.), speed=0.)
        self.tracks[memory['identity']] = track
        return memory['identity']

    def _established_static(self, track, box, now):
        anchor = track.get('anchor')
        return (anchor is not None and track.get('stationary_since') is not None
                and track.get('anchor_seen', -1e9) - track['stationary_since'] >= STATIC_HOLD_MIN_STATIONARY_S
                and now - track['anchor_seen'] <= self.stationary_hold_s
                and float(np.linalg.norm(np.asarray(box['center_world'], dtype=float) - np.asarray(anchor))) <= STATIC_BREAK_RADIUS_M)

    def _hold_stationary(self, track, box, now, spatial_moving, measured_speed):
        """Keep a confirmed stationary track stationary through weak evidence; returns True when held."""
        center = np.asarray(box['center_world'], dtype=float)
        if track['state'] == 'stationary':
            anchor = track.get('anchor')
            track['anchor'] = tuple(center) if anchor is None else tuple(.9 * np.asarray(anchor) + .1 * center)
            track['anchor_seen'] = now
            track.setdefault('stationary_since', now)
            return False
        anchor = track.get('anchor')
        if (spatial_moving or anchor is None or measured_speed >= 2.5
                or now - track.get('anchor_seen', -1e9) > self.stationary_hold_s
                or float(np.linalg.norm(center - np.asarray(anchor))) > STATIC_BREAK_RADIUS_M):
            if spatial_moving or anchor is not None and float(np.linalg.norm(center - np.asarray(anchor))) > STATIC_BREAK_RADIUS_M:
                for key in ('anchor', 'stationary_since'):      # real motion or a different place: forget the anchor
                    track.pop(key, None)
            return False
        return track['anchor_seen'] - track.get('stationary_since', now) >= STATIC_HOLD_MIN_STATIONARY_S

    def update(self, packet):
        now = packet['sent_ns'] / 1e9
        if packet['generation'] != self.generation or (self.last_time is not None and now <= self.last_time):
            self.tracks.clear()
        self.generation, self.last_time = packet['generation'], now
        if self.stationary_hold_s:
            # A stationary track that timed out is remembered, so the same object re-seen a moment later keeps its identity.
            for k, t in self.tracks.items():
                if now - t['seen'] > self.max_gap and t.get('anchor') is not None and t['state'] != 'moving':
                    self.static_memory.append(dict(identity=k, track=t, seen=t['seen']))
            self.static_memory = [m for m in self.static_memory if now - m['seen'] <= self.stationary_hold_s]
        self.tracks = {k: t for k, t in self.tracks.items() if now - t['seen'] <= self.max_gap}
        boxes, ids = packet['boxes'], list(self.tracks)
        ego = _ego_xy(packet)
        assignments = {}
        if boxes and ids:
            # Private unmatched columns allow every box to start a fresh track.
            cost = np.full((len(boxes), len(ids) + len(boxes)), 1e6)
            for i, box in enumerate(boxes):
                cost[i, len(ids) + i] = 1.
                for j, identity in enumerate(ids):
                    track = self.tracks[identity]
                    dt = now - track['seen']
                    predicted = np.asarray(track['xy']) + np.asarray(track['velocity']) * dt
                    distance = float(np.linalg.norm(np.asarray(box['center_world']) - predicted))
                    previous = track['box']
                    # Small footprints can change by a few decimetres between
                    # frames. A raw ratio (e.g. .28 -> .67 m) would reject the
                    # same object despite an unchanged centre. Use an absolute
                    # tolerance in the association cost only; emitted geometry
                    # and the one-to-one centre gate remain unchanged.
                    ratio = max((max(box[k], previous[k]) + .5) /
                                (min(box[k], previous[k]) + .5)
                                for k in ('length_m', 'width_m'))
                    # Raw speed can widen the search but must not move the
                    # predicted centre: a stopped car can have a large VE speed.
                    gate = max(.8, min(2., previous['length_m'] / 2.)) + min(
                        3., max(track['speed'], box['speed_mps'] or 0.) * dt)
                    fragment_match = (2. < ratio <= 4.
                                      and self._moving_fragment_match(track, box, predicted, dt))
                    if (ratio <= 2. or fragment_match) and distance <= gate:
                        # Prefer recent observations over a stale duplicate when
                        # their geometry is similarly plausible. Keep gap tracks
                        # eligible so a genuine short occlusion can reconnect.
                        gap_cost = .15 * min(1., max(0., dt-.15)/.6)
                        # Size is unreliable for a verified partial footprint.
                        # Penalizing its shrink/growth again would let a young
                        # duplicate steal the full body on the next frame.
                        size_cost = 0. if fragment_match else .15 * (ratio - 1.)
                        cost[i, j] = .8 * distance / gate + size_cost + gap_cost
            rows, columns = linear_sum_assignment(cost)
            assignments = {i: ids[j] for i, j in zip(rows, columns) if j < len(ids) and cost[i, j] < 1.}
        output = []
        for index, box in enumerate(boxes):
            identity = assignments.get(index)
            if identity is None and self.stationary_hold_s:
                reuse = self._recall_static(box, now)
                if reuse is not None:
                    identity = reuse
            if identity is None:
                self.serial += 1
                identity = self.serial
                self.tracks[identity] = dict(history=deque(maxlen=20), state='uncertain',
                                             velocity=(0., 0.), speed=0.)
            track = self.tracks[identity]
            history = track['history']
            continues = False
            # Do not infer motion across a long detection gap.
            if history and now - history[-1][0] > .3:
                dt = now - history[-1][0]
                expected = np.asarray(track.get('velocity', (0., 0.))) * dt
                observed = np.asarray(box['center_world']) - np.asarray(history[-1][1:])
                continues = (self.moving_carry_s > 0. and track.get('carry')
                             and dt <= self.max_gap and np.linalg.norm(expected) >= .35
                             and np.linalg.norm(observed-expected) <= max(.3, .2*np.linalg.norm(expected)))
                if not continues:
                    history.clear()
                    track['state'] = 'uncertain'
            history.append((now, *box['center_world']))
            window = max(.8, self.slow_window_s)
            while len(history) > 2 and now - history[0][0] > window:
                history.popleft()
            motion = _position_motion([h for h in history if now - h[0] <= .8] if self.slow_window_s else history)
            if continues and not motion['coherent']:
                # A confirmed mover observed farther along its predicted path
                # supplies new displacement evidence even at 2--3 Hz. Repeated
                # 0.4 s intervals must not erase all history at every frame.
                velocity = observed / dt
                motion.update(coherent=True, velocity=tuple(map(float, velocity)),
                              speed_mps=float(np.linalg.norm(velocity)),
                              displacement_m=float(np.linalg.norm(observed)))
            span, measured_speed = motion['span_s'], motion['speed_mps']
            spatial_moving = motion['coherent']
            vx, vy = motion['velocity']
            raw_speed = box['speed_mps']
            raw_supported = (raw_speed is not None and raw_speed >= .8
                             and abs(raw_speed-measured_speed) <= max(1.5, .35*measured_speed))
            # Hysteresis for an already confirmed moving track. Small lateral
            # box jitter should not zero a supported speed at every frame.
            if not spatial_moving and self.slow_window_s and len(history) >= 8:
                slow = _position_motion(history)
                # Not the path-efficiency test: box-centre scatter adds path length at every step, so a longer window
                # makes it fail more. Use the regression instead: the fitted travel must stand clear of the scatter,
                # and a vehicle-sized box must move along its own axis.
                fitted = slow['speed_mps'] * slow['span_s']
                heading_rad = math.radians(box['heading_deg'])
                aligned = (box['length_m'] < 3. or slow['speed_mps'] > 0 and
                           abs(slow['velocity'][0]*math.cos(heading_rad) + slow['velocity'][1]*math.sin(heading_rad))
                           / slow['speed_mps'] >= .8)
                if (.8 <= slow['speed_mps'] <= 3.0 and slow['span_s'] >= self.slow_window_s - .3 and aligned
                        and fitted >= max(1.0 if box['length_m'] >= 3. else 1.2,
                                          (3. if box['length_m'] >= 3. else 4.) * slow['residual_m'])):
                    motion, span, measured_speed = slow, slow['span_s'], slow['speed_mps']
                    vx, vy = slow['velocity']
                    spatial_moving = True
                    raw_supported = (raw_speed is not None and raw_speed >= .8
                                     and abs(raw_speed-measured_speed) <= max(1.5, .35*measured_speed))
            if (self.stationary_hold_s and spatial_moving and measured_speed < 2.5
                    and self._established_static(track, box, now)):
                # Passing close to a parked object changes the LiDAR viewpoint, so its box centre slides by decimetres.
                # That apparent drift is not motion: demand real displacement from the anchor before calling it moving.
                spatial_moving = False
            old_velocity = np.asarray(track['velocity'])
            direction_agreement = float(old_velocity @ np.asarray((vx, vy))) / max(
                float(np.linalg.norm(old_velocity))*measured_speed, 1e-9)
            if (track['state'] == 'moving' and raw_supported and span >= .3
                    and measured_speed >= .8 and motion['displacement_m'] >= .35
                    and motion['path_efficiency'] >= .6 and direction_agreement >= .9
                    and motion['residual_m'] <= max(.2, .2*motion['displacement_m'])):
                spatial_moving = True
            speed = (float(raw_speed) if raw_speed is not None
                     and math.isfinite(raw_speed) and raw_speed >= 0. else None)
            source = 've_provisional' if speed is not None else 'speed_unavailable'
            track['state'] = 'uncertain'
            if spatial_moving:
                track['state'] = 'moving'
                speed, source = measured_speed, 'position_history'
                # Use the model only when independently observed motion agrees.
                # A low raw speed also must not hide a moving vehicle.
                if raw_supported:
                    speed, source = raw_speed, 've_supported_by_position'
            elif (len(history) >= 4 and span >= .5 - 1e-9 and motion['radius_m'] <= .3
                  and motion['displacement_m'] < .3 and motion['latest_deviation_m'] <= .25
                  and measured_speed < .6):
                track['state'] = 'stationary'
                speed, source = 0., 'stationary_history'
            elif (self.provisional_static_mps and len(history) >= 2 and span >= .2 - 1e-9
                  and raw_speed is not None and raw_speed <= self.provisional_static_mps
                  and motion['radius_m'] <= .5 and motion['displacement_m'] < .4 and measured_speed < .8):
                track['state'] = 'stationary'
                speed, source = 0., 'provisional_stationary'
            if self.stationary_hold_s:
                held = self._hold_stationary(track, box, now, spatial_moving, measured_speed)
                if held:
                    speed, source = 0., 'stationary_hold'
                    track['state'] = 'stationary'
            carried = False
            if spatial_moving:
                track['carry'] = (now, speed, (vx, vy), box['heading_deg'], motion['velocity'])
            elif (self.moving_carry_s > 0. and track['state'] == 'uncertain' and track.get('carry')
                    and now - track['carry'][0] <= self.moving_carry_s
                    and (raw_speed is None or raw_speed >= .5 * track['carry'][1])):
                # Evidence is weak, not contradictory: stationary evidence would have set state 'stationary'.
                speed, source, carried = track['carry'][1], 'carry_over', True
            heading = box['heading_deg']
            if spatial_moving or carried:
                # An OBB has a 180-degree ambiguity; resolve its forward sign
                # from observed motion without rotating/changing the footprint.
                angle = math.radians(heading)
                direction = (vx, vy) if spatial_moving else track['carry'][2]
                if direction[0] * math.cos(angle) + direction[1] * math.sin(angle) < 0.:
                    heading = (heading + 360.) % 360. - 180.
            velocity = (vx, vy) if spatial_moving else (track['carry'][2] if carried else (0., 0.))
            # Association and motion keep using the raw box below.
            track.update(seen=now, xy=box['center_world'], box=box, velocity=velocity,
                         speed=speed if speed is not None else 0.)
            emitted = box
            if self.static_geometry_fusion:
                if spatial_moving:
                    track.pop('static_geometry', None)
                else:
                    geometry = track.setdefault('static_geometry', _StaticGeometry())
                    geometry.add(box, now, ego)
                    fused = geometry.fused() if track['state'] == 'stationary' else None
                    if fused is not None:
                        heading = fused['heading_deg']
                        emitted = dict(box, **fused, raw_center_world=box['center_world'],
                                       raw_heading_deg=box['heading_deg'], raw_length_m=box['length_m'],
                                       raw_width_m=box['width_m'], geometry_samples=len(geometry.samples))
            output.append(dict(emitted, track_id=identity, motion_state=track['state'],
                               planning_speed_mps=speed, planning_heading_deg=heading,
                               motion_direction_observed=spatial_moving, track_span_s=span,
                               planning_speed_source=source, observed_speed_mps=measured_speed,
                               motion_fit_residual_m=motion['residual_m'],
                               motion_path_efficiency=motion['path_efficiency']))
        self._support_overlapping_motion(output, now)
        policy = 'world_position_supported_speed_v4_provisional_ve+moving_fragment_association_v1'
        if self.static_geometry_fusion:
            policy += '+stationary_geometry_fusion_v1'
        return dict(packet, boxes=output, tracking_policy=policy)
