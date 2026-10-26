"""Contact labels MORAI reports before the drive starts (own-vehicle / anonymous records).

Some MORAI builds list the Ego itself (or an anonymous '-100') in ``collision_objects`` before any driving. The
shared drive loop stops on any non-empty list, so those scenarios end at the first tick. The baseline journals
what was reported before the drive and exempts only those labels; every new contact still stops the run.
This is an explicit monitoring exception, not proof of collision-free driving.

The highway/roundabout launcher (AIM2026/Expert/run_highway.py) has its own copy of this logic, written first and
deliberately left untouched here. The GPS+VE launcher uses this module; the two copies can be unified later.
"""
from datetime import datetime
import json
import time


class StartupContactBaseline:
    """Journal pre-drive self/anonymous records and verified removed NPCs (same rules as run_highway.py's copy).

    This is an explicit monitoring exception, not proof of collision-free driving.
    Once absent, the label loses its exemption for the rest of this run.
    """
    def __init__(self, state, live_actor_ids=None):
        initial = {str(value).strip() for value in state.get('collision_objects', [])
                   if str(value).strip()}
        self.initial = initial.copy()
        self.initial_live_actor_ids = (sorted(str(v) for v in live_actor_ids)
                                       if live_actor_ids is not None else None)
        self.seen = initial.copy()
        self.previous = initial.copy()
        numeric = {value for value in initial if value.isdecimal()}
        absent = (numeric - set(live_actor_ids)) if live_actor_ids is not None else set()
        allowed = {'Ego', '-100'} | absent
        self.ignored = initial if 'Ego' in initial and initial <= allowed else set()
        self.removed_actor_ids = self.ignored & absent
        self.active = bool(self.ignored)

    def apply(self, state, live_actor_ids=None):
        state = dict(state)
        raw = list(dict.fromkeys(str(value).strip() for value in state.get('collision_objects', [])
                                 if str(value).strip()))
        previously_ignored = self.ignored.copy()
        # Each exemption can only disappear, never grow after the baseline.
        # A cleared label that later reappears is a NEW contact, including -100.
        self.ignored.intersection_update(raw)
        # Reused actor IDs must not hide a new contact. If the live actor query
        # fails, numeric exemptions are revoked instead of assuming an empty scene.
        if self.removed_actor_ids & self.ignored:
            self.ignored.difference_update(self.removed_actor_ids if live_actor_ids is None
                                           else self.removed_actor_ids & set(live_actor_ids))
        self.active = bool(self.ignored)
        state['raw_collision_objects'] = raw
        state['startup_contact_baseline_active'] = self.active
        state['startup_contact_ignored'] = sorted(self.ignored)
        state['collision_objects'] = [item for item in raw if item not in self.ignored]
        state['contact_classification'] = {
            item: ('startup_residual' if item in self.ignored else
                   'startup_exemption_revoked' if item in previously_ignored else
                   'new_report' if item not in self.seen else
                   'reappeared_report' if item not in self.previous else
                   'preexisting_unverified' if item in self.initial else 'continuing_report')
            for item in raw
        }
        self.seen.update(raw)
        self.previous = set(raw)
        return state


class StartupContactExemption:
    """Scenario mixin placed before the drive-loop scenario class in the bases.

    Needs ``grpc`` and a place for evidence files (``_contact_output_dir``). ``contact_label`` only names the
    launcher in console lines.
    """
    contact_label = 'Scenario'

    def _contact_output_dir(self):
        return self.output

    def _strict_contact_mode(self):
        return bool(self.args.strict_collisions)

    def live_actor_ids(self):
        """Scenario verification only; never feed actor states to the Expert."""
        world = getattr(getattr(self, 'grpc', None), 'world', None)
        if world is None:
            return None
        try:
            response = world.get_all_actors_state(vehicle=True, pedestrian=True, obstacle=True)
            if response is None or not hasattr(response, 'states'):
                return None
            return {str(item.actor_info.id.value) for item in response.states}
        except Exception as exc:
            print('[{}] Cannot verify retired collision actors: {}'.format(self.contact_label, exc), flush=True)
            return None

    def get_ego_motion_state(self):
        state = super().get_ego_motion_state()
        baseline = getattr(self, '_startup_contact_baseline', None)
        if baseline is None:
            return state
        actor_ids = (self.live_actor_ids() if baseline.removed_actor_ids & baseline.ignored else None)
        state = baseline.apply(state, actor_ids)
        if baseline.removed_actor_ids:
            state['startup_contact_live_actor_ids'] = sorted(actor_ids) if actor_ids is not None else None
        signature = (tuple(sorted(state['raw_collision_objects'])),
                     tuple(state['startup_contact_ignored']), tuple(sorted(state['collision_objects'])))
        if signature != getattr(self, '_last_contact_signature', None):
            now = time.monotonic()
            event = dict(monotonic=now, observed_at=datetime.now().astimezone().isoformat(),
                         since_baseline_s=now - getattr(self, '_contact_started_monotonic', now),
                         source='MORAI_gRPC_vehicle_state.collision_objects', **state)
            first_reports = getattr(self, '_contact_first_reports', {})
            for item in state['raw_collision_objects']:
                first_reports.setdefault(item, event.copy())
            self._contact_first_reports = first_reports
            with (self._contact_output_dir() / 'contact_events.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(event) + '\n')
            self._last_contact_signature = signature
        return state

    def begin_startup_contact_baseline(self):
        """Call first in configure_drive_from_start, before the Ego is teleported or reset."""
        # Preserve evidence before the existing runner teleports/reset-motion.
        # This separates a pre-existing simulator contact from placement contact.
        state = self.get_ego_motion_state()
        self._contact_started_monotonic = time.monotonic()
        self._contact_first_reports = {}
        self._last_contact_signature = None
        (self._contact_output_dir() / 'before_placement.json').write_text(
            json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')
        if not self._strict_contact_mode():
            actor_ids = (self.live_actor_ids() if any(str(v).isdecimal()
                         for v in state.get('collision_objects', [])) else None)
            self._startup_contact_baseline = StartupContactBaseline(state, actor_ids)
            if self._startup_contact_baseline.active:
                print('[{}] INTERACTIVE: pre-existing contact baseline {} is recorded; '
                      'new contacts and cleared labels reappearing still stop the run. '
                      'This is not a collision-free guarantee.'.format(
                          self.contact_label, sorted(self._startup_contact_baseline.ignored)), flush=True)
        else:
            # Strict mode records the same evidence but grants no exemptions.
            self._startup_contact_baseline = StartupContactBaseline(state)
            self._startup_contact_baseline.ignored.clear()
            self._startup_contact_baseline.active = False
        return state
