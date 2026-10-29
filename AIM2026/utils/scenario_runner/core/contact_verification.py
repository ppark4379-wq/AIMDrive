"""Narrow runtime exception for a newly spawned, provably distant NPC report.

Uses simulator evidence ONLY in the scenario evaluator, never in Expert inputs.
An exception means 'report requires review', not 'collision-free run'.
"""
from collections import OrderedDict
import math


class SpawnContactVerifier:
    # Generous bounds, not tuning for the driving controller. A vehicle may
    # move 60 m/s and the pair may close 120 m/s between observations.
    SPEED_BOUND = 60.
    CONTACT_ENVELOPE = 25.
    MAX_QUERY_S = .15
    MAX_BIRTH_AGE_S = 2.

    def __init__(self):
        self.births = OrderedDict()
        self.continuing = {}
        self.review_ids = set()

    @staticmethod
    def point(value):
        try:
            point = tuple(float(v) for v in value)
            return point if len(point) == 2 and all(math.isfinite(v) for v in point) else None
        except (ValueError, TypeError):
            return None

    def record_spawn(self, identity, started, ego_time, ego_xy, requested_xy, actual_xy):
        ego, requested, actual = map(self.point, (ego_xy, requested_xy, actual_xy))
        # A failed/misplaced spawn gets NO exception. Recording the earliest
        # request time also covers time spent in a slow SpawnVehicle RPC.
        if (None in (ego, requested, actual) or not all(math.isfinite(v) for v in (started, ego_time))
                or ego_time > started or math.dist(requested, actual) > 2.):
            return
        identity = str(identity)
        self.births[identity] = dict(time=ego_time, started=started, ego=ego,
                                     target=requested, distance=math.dist(ego, requested))
        while len(self.births) > 128:
            self.births.popitem(last=False)

    def assess(self, reports, now, query_s, ego_xy, targets, *, entry_review=False):
        """Return per-ID evidence; defer only when every check proves separation.

        targets contains {actor_id: (x,y)} from one fresh all-actor query. Unknown
        IDs and unavailable/stale queries stop. During the pre-merge approach,
        entry_review permits a continuing distant spawn label across a history
        gap; this is monitoring policy, not proof of collision-free travel.
        """
        reports = list(dict.fromkeys(map(str, reports)))
        for identity in set(self.continuing) - set(reports):
            self.continuing.pop(identity)
            self.births.pop(identity, None)  # a cleared label cannot inherit a later exemption
        ego = self.point(ego_xy)
        decisions = {}
        for identity in reports:
            target = self.point(targets.get(identity))
            previous = self.continuing.get(identity)
            birth = self.births.get(identity)
            evidence = dict(action='stop', reason='no_verified_live_spawn')
            if previous or birth:
                anchor = previous or birth
                age = now - anchor['time']
                birth_age = now - birth['started'] if birth else None
                evidence.update(interval_s=age, birth_age_s=birth_age)
                if (not math.isfinite(now) or not math.isfinite(query_s) or not 0 <= query_s <= self.MAX_QUERY_S
                        or ego is None or target is None or age < 0.):
                    evidence['reason'] = 'missing_or_stale_geometry'
                elif birth_age is not None and birth_age < 0.:
                    evidence['reason'] = 'invalid_time'
                elif not previous and (birth_age is None or birth_age > self.MAX_BIRTH_AGE_S):
                    evidence['reason'] = 'spawn_too_old'
                else:
                    distance = math.dist(ego, target)
                    # Geometry must also be continuous: teleporting from a
                    # nearby impact to a distant endpoint cannot clear contact.
                    bound = self.SPEED_BOUND * age + 2.
                    # Bound the WHOLE interval using both observed endpoints.
                    # A hidden contact would require closing from d0 and then
                    # separating again to d1 within the same time budget:
                    # min separation >= (d0 + d1 - relative_speed_bound*dt)/2.
                    # Using only d0 falsely stopped a run after a 0.396 s spawn
                    # RPC, although both observations placed the NPC ~74 m away.
                    travel_budget = 2. * self.SPEED_BOUND * age
                    lower = min(anchor['distance'], distance,
                                (anchor['distance'] + distance - travel_budget) / 2.) - 2.
                    evidence.update(distance_m=distance, previous_distance_m=anchor['distance'],
                                    relative_travel_budget_m=travel_budget,
                                    separation_lower_bound_m=lower)
                    if (math.dist(ego, anchor['ego']) > bound or math.dist(target, anchor['target']) > bound):
                        evidence['reason'] = 'discontinuous_geometry'
                    elif min(distance, lower) <= self.CONTACT_ENVELOPE:
                        if (entry_review and previous
                                and min(distance, anchor['distance']) > self.CONTACT_ENVELOPE):
                            evidence.update(action='continue_with_review',
                                            reason='entry_distant_report_review')
                        else:
                            evidence['reason'] = 'contact_possible'
                    else:
                        evidence.update(action='continue_with_review', reason='verified_distant_spawn_report')
                    if evidence['action'] == 'continue_with_review':
                        # Snapshot capture can occur anywhere inside the RPC.
                        # Anchor at its start so the next check also budgets
                        # the preceding query's uncertainty in observation time.
                        self.continuing[identity] = dict(time=now-query_s, ego=ego, target=target, distance=distance)
                        self.review_ids.add(identity)
            if evidence['action'] == 'stop':
                self.continuing.pop(identity, None)
                self.births.pop(identity, None)
            decisions[identity] = evidence
        return decisions
