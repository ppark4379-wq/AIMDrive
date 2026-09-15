"""Opt-in actor lifetime experiment; does not change the expert controller.

Distances are workload policy, not proof of invisibility. All mutations are
recorded for downstream quarantine. Only explicitly owned roundabout handles
are removed; native highway vehicles finish their existing routes.
"""
import math
import time


POLICY = dict(
    version='nearby_actors_v3',
    roundabout_lead_m=100.,
    roundabout_retire_after_m=30.,
    # A normal lap crosses the roundabout in about 35 simulation seconds.
    # If the expert advances less than this distance for this long while the
    # owned NPCs are active, retire only those verified actors.  The mutation
    # boundary is quarantined downstream.  This prevents one physics deadlock
    # from consuming the entire wall-time bound without pretending that the
    # lap completed.
    roundabout_stall_window_s=45.,
    roundabout_stall_min_progress_m=5.,
    roundabout_min_active_scenario_s=20.,
    roundabout_mutation_quarantine_s=1.,
    highway_stop_after_entry_m=220.,
    highway_spawn_distance_m=220.,
)


class NearbyActorsMixin:
    def _nearby_policy(self):
        configured = self.cfg.get('nearby_actor_policy')
        policy = dict(POLICY)
        if isinstance(configured, dict):
            policy.update(configured)
        return policy

    def _nearby_checkpoint_s(self, identity):
        checkpoint = next(c for c in self.route_definition['checkpoints'] if c['id'] == identity)
        return self.path_s_at_index(int(checkpoint['path_index']))

    def spawn_roundabout_npcs(self):
        if not self.cfg.get('nearby_actor_policy'):
            return super().spawn_roundabout_npcs()
        policy = self._nearby_policy()
        self.nearby_roundabout_state = 'armed'
        self.nearby_roundabout_spawn_elapsed = None
        self.nearby_roundabout_progress_anchor_elapsed = None
        self.nearby_roundabout_progress_anchor_s = None
        self.nearby_highway_stopped = False
        self.nearby_last_check = float('-inf')
        self.event('nearby_policy', policy=policy, visibility_verified=False,
                   mission_distribution_changed=True)

    def activate_highway_spawn_points(self, elapsed, current_s):
        if getattr(self, 'nearby_highway_stopped', False):
            return False
        return super().activate_highway_spawn_points(elapsed, current_s)

    def _nearby_mutation(self, action, current_s, operation, **fields):
        started = time.monotonic_ns()
        self.event(
            'actor_lifecycle_started', action=action, route_s=current_s,
            **fields
        )
        success = False
        try:
            result = operation()
            success = True
            return result
        finally:
            self.event('actor_lifecycle_finished', action=action, route_s=current_s,
                       started_monotonic_ns=started, success=success,
                       visibility_verified=False, **fields)

    def _retire_roundabout_npcs(self, elapsed, ego, current_s, reason,
                                **reason_fields):
        owned = {str(p['actor'].get_id()) for p in self.roundabout_npcs}
        states = self.grpc.get_all_vehicle_states()
        matches = [s for s in states if str(s.actor_info.id.value) in owned]
        # Missing/ambiguous inventory is not evidence that removal is safe.
        if (not owned or len(matches) != len(owned)
                or {str(s.actor_info.id.value) for s in matches} != owned):
            return False
        distances = [math.hypot(s.transform.location.x-ego['x'],
                                s.transform.location.y-ego['y']) for s in matches]

        def retire():
            for item in list(self.roundabout_npcs):
                actor = item['actor']
                if not self.grpc.destroy_vehicle_by_id(str(actor.get_id())):
                    raise RuntimeError(
                        'Cannot confirm owned roundabout actor retirement'
                    )
                self.scenario_actors.remove(actor)
                self.roundabout_npcs.remove(item)

        self._nearby_mutation(
            'roundabout_retire', current_s, retire,
            actor_ids=sorted(owned),
            ego_actor_distances_m=sorted(float(value) for value in distances),
            retire_reason=reason,
            active_scenario_s=(
                float(elapsed)-float(self.nearby_roundabout_spawn_elapsed)
                if self.nearby_roundabout_spawn_elapsed is not None else None
            ),
            mutation_boundary_requires_dataset_mask=True,
            mutation_boundary_quarantine_s=float(
                self._nearby_policy()['roundabout_mutation_quarantine_s']
            ),
            **reason_fields
        )
        self.nearby_roundabout_state = 'retired'
        return True

    def update_nearby_actors(self, elapsed, ego, current_s):
        if not self.cfg.get('nearby_actor_policy'):
            return
        policy = self._nearby_policy()
        if elapsed-self.nearby_last_check < .5:
            return
        self.nearby_last_check = elapsed
        zone = self.route_definition['zones']['roundabout']
        entry = self._nearby_checkpoint_s(zone['ego_entry_checkpoint'])
        exit_s = self._nearby_checkpoint_s(zone['ego_exit_checkpoint'])
        if (self.nearby_roundabout_state == 'armed'
                and current_s >= entry-policy['roundabout_lead_m']):
            self._nearby_mutation('roundabout_spawn', current_s,
                                 lambda: super(NearbyActorsMixin, self).spawn_roundabout_npcs())
            self.nearby_roundabout_state = 'spawned'
            self.nearby_roundabout_spawn_elapsed = float(elapsed)
            self.nearby_roundabout_progress_anchor_elapsed = float(elapsed)
            self.nearby_roundabout_progress_anchor_s = float(current_s)
            self.event(
                'roundabout_spawn_contract',
                route_s=float(current_s),
                elapsed_seconds=float(elapsed),
                actor_ids=sorted(
                    str(item['actor'].get_id())
                    for item in self.roundabout_npcs
                ),
                actor_motion_contracts=[{
                    'actor_id': str(item['actor'].get_id()),
                    'ai_enable_acknowledged': bool(
                        item.get('ai_enable_acknowledged', False)
                    ),
                    'speed_limit_acknowledged': bool(
                        item.get('speed_limit_acknowledged', False)
                    ),
                    'initial_velocity_acknowledged': bool(
                        item.get('initial_velocity_acknowledged', False)
                    ),
                } for item in self.roundabout_npcs],
                target_speed_mps=float(
                    self.cfg.get('roundabout_npc', {}).get('speed_mps', 0.)
                ),
                stall_window_s=float(policy['roundabout_stall_window_s']),
                stall_min_progress_m=float(
                    policy['roundabout_stall_min_progress_m']
                ),
                training_visible=False,
            )
        if self.nearby_roundabout_state == 'spawned':
            route_exit_reached = (
                current_s > exit_s+policy['roundabout_retire_after_m']
            )
            if route_exit_reached:
                self._retire_roundabout_npcs(
                    elapsed, ego, current_s, 'route_exit',
                    route_exit_contract_s=float(
                        exit_s+policy['roundabout_retire_after_m']
                    ),
                )
                # A failed-closed inventory check leaves the state spawned and
                # retries on the next observation.  No stall bookkeeping is
                # needed while the ordinary route retirement applies.
                return
            if self.nearby_roundabout_progress_anchor_s is None:
                self.nearby_roundabout_progress_anchor_s = float(current_s)
            if self.nearby_roundabout_progress_anchor_elapsed is None:
                self.nearby_roundabout_progress_anchor_elapsed = float(elapsed)
            if self.nearby_roundabout_spawn_elapsed is None:
                self.nearby_roundabout_spawn_elapsed = float(elapsed)
            progress_since_anchor = (
                float(current_s)-float(self.nearby_roundabout_progress_anchor_s)
            )
            if progress_since_anchor >= float(
                    policy['roundabout_stall_min_progress_m']):
                self.nearby_roundabout_progress_anchor_s = float(current_s)
                self.nearby_roundabout_progress_anchor_elapsed = float(elapsed)
                progress_since_anchor = 0.0

            active_scenario_s = (
                float(elapsed)-float(self.nearby_roundabout_spawn_elapsed)
            )
            stalled_scenario_s = (
                float(elapsed)
                - float(self.nearby_roundabout_progress_anchor_elapsed)
            )
            progress_stalled = bool(
                active_scenario_s
                >= float(policy['roundabout_min_active_scenario_s'])
                and stalled_scenario_s
                >= float(policy['roundabout_stall_window_s'])
            )
            if progress_stalled:
                self._retire_roundabout_npcs(
                    elapsed, ego, current_s, 'progress_stall',
                    stall_scenario_s=float(stalled_scenario_s),
                    progress_anchor_route_s=float(
                        self.nearby_roundabout_progress_anchor_s
                    ),
                    progress_since_anchor_m=float(progress_since_anchor),
                    route_exit_contract_s=float(
                        exit_s+policy['roundabout_retire_after_m']
                    ),
                )
        highway = self.route_definition['zones']['highway']
        if (not self.nearby_highway_stopped and not self.highway_traffic_cleared and
                self.highway_spawn_points_active and
                current_s > self._nearby_checkpoint_s(highway['ego_entry_checkpoint'])+policy['highway_stop_after_entry_m']):
            points = [p for p in self.highway_spawn_points if not p.get('retired', False)]
            distances = [math.hypot(p['spec']['position']['x']-ego['x'],
                                   p['spec']['position']['y']-ego['y']) for p in points]
            if distances and min(distances) > policy['highway_spawn_distance_m']:
                def stop_spawning():
                    self._set_highway_spawn_points_enabled(False)
                    if any(p['enabled'] for p in points):
                        raise RuntimeError('Cannot confirm distant highway spawner disable')
                self._nearby_mutation('highway_stop_new_spawns', current_s, stop_spawning)
                self.nearby_highway_stopped = True

    def on_expert_timeline_tick(self, elapsed, ego_state, current_s, remaining_s):
        self.update_nearby_actors(elapsed, ego_state, current_s)
        return super().on_expert_timeline_tick(elapsed, ego_state, current_s, remaining_s)
