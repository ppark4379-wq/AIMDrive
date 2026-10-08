"""Creating aliases and native global addresses remain distinct lookup scopes."""

import copy
import math
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import grpc as grpc_api

from ..scenarios.full_loop_parts.tunnel_static_parts import live
from .test_tunnel_static_live import state
from . import test_tunnel_static_live as preview_fixture


FIXTURE = 'isolated_tunnel_fixture'
REQUEST = 'tunnel_preview_owned_request'
RAW = '7'


class LookupRpcError(grpc_api.RpcError):
    def __init__(self, code=grpc_api.StatusCode.UNKNOWN,
                 description='Exception was thrown by handler'):
        self.status, self.description = code, description

    def code(self):
        return self.status

    def details(self):
        return self.description

    def __str__(self):
        return str(self.status)+': '+self.description


class NativeLookupWorld:
    """Scoped request echoes alias; only unchanged global ObjectInfo reads raw."""
    def __init__(self):
        self.raw = state(RAW, 45., 0., model='WoodBox')
        self.raw.actor_info.object_type = 3
        self.raw.actor_info.client_key = ''
        self.alias = copy.deepcopy(self.raw)
        self.alias.actor_info.id.value = REQUEST
        self.alias.actor_info.client_key = FIXTURE
        self.ego = state('Ego', -40., 0., width=2., length=4.5, model='EgoCar')
        self.ego.actor_info.object_type = 1
        self.ego.actor_info.client_key = ''
        self.foreign = state('99', 140., 0., model='ForeignBox')
        self.foreign.actor_info.object_type = 3
        self.foreign.actor_info.client_key = 'foreign_owner'
        self.requests, self.inventories, self.operations = [], [], []
        self.timeouts = []
        self.alias_error, self.raw_error = None, None
        self.extra = []
        self.stub = NS(GetActorState=self.one, GetAllActorsState=self.inventory,
                       GetOptionName=self.model, DestroyActor=Mock(), DestroyAllActors=Mock())
        self.spawn = Mock()
        self.grpc = NS(world=NS(_sim_adapter=NS(_actor_stub=self.stub),
                               spawn_vehicle=self.spawn, spawn_obstacle=self.spawn), client_key=FIXTURE)
        self.handle = NS(get_object_info=lambda: NS(id=NS(value=REQUEST),
                         client_key=FIXTURE, object_type=3))
        self.placement = {
            'actor_kind': 'obstacle', 'model': 'WoodBox', 'scale': [1., 1., 1.],
            'position': {'x': 45., 'y': 0., 'z': 28.}, 'yaw_deg': 0.,
            'planned_bbox': {'width_m': 1.2, 'length_m': 1.2,
                             'center_offset_spawn_xy': [0., 0.], 'heading_offset_deg': 0.},
        }
        self.entry = {
            'handle': self.handle, 'request_id': REQUEST,
            'source_handle_id': REQUEST, 'source_client_key': FIXTURE,
            'raw_id': RAW, 'raw_object_info': copy.deepcopy(self.raw.actor_info),
            'requested_model': 'WoodBox', 'actor_kind': 'obstacle',
            'placement': copy.deepcopy(self.placement),
        }

    def one(self, info, timeout):
        self.requests.append(copy.deepcopy(info))
        self.operations.append(('state', info.id.value, info.client_key))
        self.timeouts.append(timeout)
        if info.id.value == REQUEST and info.client_key == FIXTURE:
            if self.alias_error:
                raise self.alias_error
            return copy.deepcopy(self.alias)
        if info.id.value == RAW and info.client_key == '':
            if self.raw_error:
                raise self.raw_error
            return copy.deepcopy(self.raw)
        raise LookupRpcError()

    def inventory(self, query, timeout):
        self.inventories.append(copy.deepcopy(query))
        self.operations.append(('inventory', query.client_key))
        self.timeouts.append(timeout)
        if query.client_key != '':
            raise AssertionError('Fresh fallback must read native global IDs')
        return NS(states=[copy.deepcopy(self.ego), copy.deepcopy(self.raw),
                          copy.deepcopy(self.foreign), *copy.deepcopy(self.extra)])

    def model(self, info, timeout):
        self.timeouts.append(timeout)
        if info.id.value == RAW and info.client_key == '':
            return NS(value=self.raw.actor_info.model)
        if info.id.value == '99' and info.client_key == 'foreign_owner':
            return NS(value=self.foreign.actor_info.model)
        for item in self.extra:
            if info.id.value == item.actor_info.id.value and info.client_key == item.actor_info.client_key:
                return NS(value=item.actor_info.model)
        raise LookupRpcError()


class LookupContractTests(unittest.TestCase):
    def setUp(self):
        self.world = NativeLookupWorld()
        modules = {
            'proto.morai.actor.actor_get_pb2': NS(GetAllActorsFilter=lambda **values: NS(**values)),
            'proto.morai.actor.actor_enum_pb2': NS(GEAR_MODE_P=1, GEAR_MODE_D=4),
            'proto.morai.common.enum_pb2': NS(OBJECT_TYPE_VEHICLE=1, OBJECT_TYPE_PEDESTRIAN=2,
                                            OBJECT_TYPE_OBSTACLE=3, STATUS_CODE_SUCCESS=1),
        }
        injected = patch.dict(sys.modules, modules)
        injected.start()
        self.addCleanup(injected.stop)

    def owned(self):
        return live._owned_raw_info(self.world.grpc, self.world.entry, FIXTURE)

    def test_request_alias_echo_uses_unmodified_global_address_for_exact_raw_read(self):
        original = copy.deepcopy(self.world.entry['raw_object_info'])
        info, returned_key = self.owned()
        self.assertEqual(info.id.value, RAW)
        self.assertEqual(info.client_key, '')
        self.assertEqual(returned_key, '')
        self.assertEqual(self.world.entry['raw_object_info'], original)
        self.assertIn((RAW, ''), [(row.id.value, row.client_key) for row in self.world.requests])
        self.assertNotIn((RAW, FIXTURE), [(row.id.value, row.client_key) for row in self.world.requests])
        self.assertEqual(self.world.entry['source_client_key'], FIXTURE)
        self.assertEqual(self.world.entry['source_handle_id'], REQUEST)
        self.world.stub.DestroyActor.assert_not_called()
        self.world.stub.DestroyAllActors.assert_not_called()

    def test_creating_request_returning_raw_id_preserves_global_query_namespace(self):
        self.world.alias.actor_info.id.value = RAW
        self.world.alias.actor_info.client_key = ''
        info, returned_key = self.owned()
        self.assertEqual(info.id.value, RAW)
        self.assertEqual(info.client_key, '')
        self.assertEqual(returned_key, '')
        self.assertEqual(self.world.entry['raw_object_info'].client_key, '')
        self.assertNotIn((RAW, FIXTURE), [(row.id.value, row.client_key) for row in self.world.requests])

    def test_known_global_handler_unknown_uses_fresh_unique_global_state(self):
        self.world.raw_error = LookupRpcError()
        info, _ = self.owned()
        self.assertEqual(info.id.value, RAW)
        self.assertEqual(info.client_key, '')
        self.assertGreater(len(self.world.inventories), 0)
        self.assertTrue(all(query.client_key == '' for query in self.world.inventories))

    def test_known_global_not_found_rechecks_fresh_unique_native_identity(self):
        self.world.raw_error = LookupRpcError(grpc_api.StatusCode.NOT_FOUND, 'global address unsupported')
        info, _ = self.owned()
        self.assertEqual(info.id.value, RAW)
        self.assertEqual(info.client_key, '')
        self.assertGreater(len(self.world.inventories), 0)

    def test_verified_owning_alias_handler_failure_uses_existing_raw_identity(self):
        self.world.alias_error = LookupRpcError()
        self.world.entry['spawn_global_binding_verified'] = True
        self.world.entry['raw_observation'] = live._observation(self.world.raw)
        info, _ = self.owned()
        self.assertEqual(info.id.value, RAW)
        self.assertEqual(info.client_key, '')
        self.assertEqual(self.world.entry['source_client_key'], FIXTURE)
        self.world.spawn.assert_not_called()

    def test_current_native_model_change_invalidates_the_pinned_owned_binding(self):
        self.world.raw.actor_info.model = 'DifferentBox'
        with self.assertRaises(RuntimeError):
            self.owned()
        self.world.stub.DestroyActor.assert_not_called()
        self.world.stub.DestroyAllActors.assert_not_called()

    def test_transport_failure_cannot_be_replaced_by_fresh_global_proof(self):
        for code in (grpc_api.StatusCode.UNAVAILABLE, grpc_api.StatusCode.DEADLINE_EXCEEDED):
            with self.subTest(code=code):
                self.world = NativeLookupWorld()
                self.world.raw_error = LookupRpcError(code, 'transport unavailable')
                with self.assertRaises((grpc_api.RpcError, RuntimeError)):
                    self.owned()
                self.assertEqual(self.world.operations[-1], ('state', RAW, ''))
                self.assertNotIn('verified_global_actor_id', self.world.entry['lookup_evidence'])
                self.assertNotIn('unsupported_lookup', self.world.entry['lookup_evidence'].get('pinned_global_address', {}))

    def test_unreviewed_unknown_error_does_not_acquire_generic_retry_permission(self):
        self.world.raw_error = LookupRpcError(description='server internal database failure')
        with self.assertRaises((grpc_api.RpcError, RuntimeError)):
            self.owned()
        self.assertEqual(self.world.operations[-1], ('state', RAW, ''))
        self.assertNotIn('verified_global_actor_id', self.world.entry['lookup_evidence'])
        self.assertNotIn('unsupported_lookup', self.world.entry['lookup_evidence'].get('pinned_global_address', {}))

    def test_foreign_returned_namespace_is_rejected_without_deletion(self):
        self.world.raw.actor_info.client_key = 'foreign_owner'
        # The pinned global address stays unchanged, proving this is a response
        # identity mutation rather than an intentionally different lookup scope.
        with self.assertRaises(RuntimeError):
            self.owned()
        self.world.stub.DestroyActor.assert_not_called()
        self.world.stub.DestroyAllActors.assert_not_called()

    def test_alias_and_pinned_raw_geometry_mismatch_is_rejected(self):
        self.world.raw.transform.location.x += .011
        self.world.raw.bounding_box.location.x += .011
        with self.assertRaises(RuntimeError):
            self.owned()
        self.world.stub.DestroyActor.assert_not_called()

    def test_global_duplicate_id_prevents_unknown_fallback_proof(self):
        self.world.raw_error = LookupRpcError()
        self.world.extra.append(copy.deepcopy(self.world.raw))
        with self.assertRaises(RuntimeError):
            self.owned()
        self.world.stub.DestroyActor.assert_not_called()

    def test_pinned_raw_address_or_actor_type_change_cannot_be_rebound(self):
        for mutation in ('pinned_address', 'actor_type'):
            with self.subTest(mutation=mutation):
                self.world = NativeLookupWorld()
                if mutation == 'pinned_address':
                    self.world.entry['raw_object_info'].id.value = 'another_raw_id'
                else:
                    self.world.raw.actor_info.object_type = 1
                with self.assertRaises(RuntimeError):
                    self.owned()
                self.world.stub.DestroyActor.assert_not_called()
                self.world.stub.DestroyAllActors.assert_not_called()

    def initial_registration(self):
        self.world.alias_error = LookupRpcError()
        entry = dict(self.world.entry, raw_id=None, raw_object_info=None)
        report, diagnostics = {'inventory_checks': 0}, {}
        observed, identity = live._wait_spawn_readback(
            self.world.grpc, self.world.handle, {'Ego', '99'}, 'WoodBox',
            report, diagnostics, 'Ego', live._observation(self.world.ego),
            placement=self.world.placement, owned_entry=entry,
        )
        return observed, identity, diagnostics, entry

    def test_initial_handler_unknown_resolves_unique_new_actor_without_respawn(self):
        observed, identity, diagnostics, entry = self.initial_registration()
        self.assertEqual(identity, RAW)
        self.assertEqual(observed[RAW].actor_info.client_key, '')
        self.assertEqual(diagnostics['status'], 'verified')
        self.assertEqual(diagnostics['attempts'], 1)
        self.assertTrue(entry['spawn_global_binding_verified'])
        self.world.spawn.assert_not_called()
        self.world.stub.DestroyActor.assert_not_called()

    def test_initial_handler_unknown_ambiguous_native_matches_are_rejected(self):
        another = copy.deepcopy(self.world.raw)
        another.actor_info.id.value = '8'
        self.world.extra.append(another)
        with self.assertRaisesRegex(RuntimeError, 'unique'):
            self.initial_registration()
        self.world.spawn.assert_not_called()
        self.world.stub.DestroyActor.assert_not_called()

    def test_initial_lookup_uses_reviewed_bbox_heading_convention_not_stationary_drift(self):
        for heading, accepted in ((1.35, True), (3.01, False)):
            with self.subTest(bbox_heading=heading):
                self.world = NativeLookupWorld()
                self.world.raw.transform.rotation.z = 1.347
                self.world.raw.bounding_box.rotation.z = heading
                if accepted:
                    _, identity, _, _ = self.initial_registration()
                    self.assertEqual(identity, RAW)
                else:
                    with self.assertRaisesRegex(RuntimeError, 'unique'):
                        self.initial_registration()
                self.world.spawn.assert_not_called()
                self.world.stub.DestroyActor.assert_not_called()
        self.assertEqual(live.ROTATION_TOLERANCE_DEG, .5)

    def test_initial_handler_unknown_matching_foreign_owner_is_rejected(self):
        self.world.raw.actor_info.client_key = 'foreign_owner'
        # Model lookup deliberately succeeds so ownership rejection must still
        # happen after the same model and requested pose were established.
        self.world.model = lambda info, timeout: NS(value='WoodBox')
        self.world.stub.GetOptionName = self.world.model
        with self.assertRaisesRegex(RuntimeError, 'foreign'):
            self.initial_registration()
        self.world.spawn.assert_not_called()
        self.world.stub.DestroyActor.assert_not_called()

    def test_changed_creating_handle_namespace_cannot_rebind_lookup_ownership(self):
        self.world.handle.get_object_info = lambda: NS(id=NS(value=REQUEST),
                                                     client_key='different_owner', object_type=3)
        with self.assertRaises(RuntimeError):
            self.owned()
        self.assertEqual(self.world.requests, [])
        self.world.stub.DestroyActor.assert_not_called()



class BindingClock:
    def __init__(self):
        self.value = 40.
        self.sleeps = []

    def now(self):
        return self.value

    def sleep(self, duration):
        self.sleeps.append(duration)
        self.value += duration


def delta_values(value):
    if isinstance(value, dict):
        return [item for child in value.values() for item in delta_values(child)]
    if isinstance(value, list):
        return [item for child in value for item in delta_values(child)]
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return [value]
    return []


class InitialBodyBindingTests(unittest.TestCase):
    def setUp(self):
        LookupContractTests.setUp(self)
        self.clock = BindingClock()
        self.world.entry['binding_phase'] = 'initial_before_parking'

    def mismatch_pose(self):
        self.world.alias.transform.location.z += .04
        self.world.alias.bounding_box.location.z += .04

    def settle_alias(self):
        self.world.alias = copy.deepcopy(self.world.raw)
        self.world.alias.actor_info.id.value = REQUEST
        self.world.alias.actor_info.client_key = FIXTURE

    def validated(self):
        with patch.object(live.time, 'monotonic', side_effect=self.clock.now), \
                patch.object(live.time, 'sleep', side_effect=self.clock.sleep):
            return live.validated_global_actor_state(self.world.grpc, self.world.entry, FIXTURE)

    def test_initial_pose_snapshots_settle_on_next_inventory_without_recreating_actor(self):
        self.mismatch_pose()
        inventory = self.world.stub.GetAllActorsState

        def fresh(query, timeout):
            if self.world.inventories:
                self.settle_alias()
            return inventory(query, timeout)

        self.world.stub.GetAllActorsState = fresh
        observed = self.validated()
        self.assertEqual(live._id(observed), RAW)
        self.assertEqual(len(self.world.inventories), 2)
        self.assertGreater(len(self.clock.sleeps), 0)
        self.assertEqual(self.world.entry['lookup_evidence']['verified_global_actor_id'], RAW)
        self.assertLessEqual(self.clock.now()-40., 1.5+1e-8)
        self.world.spawn.assert_not_called()

    def test_permanent_initial_pose_difference_exhausts_bounded_budget_with_numeric_evidence(self):
        self.mismatch_pose()
        with self.assertRaises(RuntimeError):
            self.validated()
        evidence = self.world.entry['lookup_evidence']
        self.assertNotIn('verified_global_actor_id', evidence)
        self.assertGreater(len(evidence['attempts']), 1)
        self.assertLessEqual(len(evidence['attempts']), 17)
        numbers = [number for attempt in evidence['attempts']
                   for number in delta_values(attempt.get('body_deltas', {}))]
        self.assertTrue(numbers, evidence)
        self.assertTrue(all(math.isfinite(number) for number in numbers))
        self.assertTrue(any(abs(number-.04) < 1e-6 for number in numbers), evidence)
        self.assertLessEqual(self.clock.now()-40., 1.5+1e-8)
        self.world.spawn.assert_not_called()

    def test_frozen_pose_mismatch_fails_without_retry(self):
        self.world.entry['binding_phase'] = 'frozen'
        self.mismatch_pose()
        with self.assertRaises(RuntimeError):
            self.validated()
        self.assertEqual(len(self.world.inventories), 1)
        self.assertEqual(self.clock.sleeps, [])
        self.world.spawn.assert_not_called()

    def test_initial_shape_change_is_fatal_even_when_position_would_be_retryable(self):
        self.mismatch_pose()
        self.world.alias.bounding_box.extent.x += .02
        with self.assertRaises(RuntimeError):
            self.validated()
        self.assertEqual(len(self.world.inventories), 1)
        self.assertEqual(self.clock.sleeps, [])
        self.assertNotIn('verified_global_actor_id', self.world.entry['lookup_evidence'])

    def test_initial_identity_model_namespace_and_ambiguity_fail_without_retry(self):
        for mutation in ('alias_id', 'type', 'model', 'client', 'ambiguous'):
            with self.subTest(mutation=mutation):
                self.world = NativeLookupWorld()
                self.world.entry['binding_phase'] = 'initial_before_parking'
                self.clock = BindingClock()
                if mutation == 'alias_id':
                    self.world.alias.actor_info.id.value = 'different_alias'
                elif mutation == 'type':
                    self.world.alias.actor_info.object_type = 1
                elif mutation == 'model':
                    self.world.raw.actor_info.model = 'AnotherBox'
                elif mutation == 'client':
                    self.world.alias.actor_info.client_key = 'foreign_owner'
                else:
                    other = copy.deepcopy(self.world.raw)
                    other.actor_info.id.value = '8'
                    self.world.extra.append(other)
                with self.assertRaises(RuntimeError):
                    self.validated()
                self.assertEqual(len(self.world.inventories), 1)
                self.assertEqual(self.clock.sleeps, [])
                self.world.spawn.assert_not_called()

    def test_initial_transport_or_remote_deadline_failure_is_not_retried(self):
        for code in (grpc_api.StatusCode.UNAVAILABLE, grpc_api.StatusCode.DEADLINE_EXCEEDED):
            with self.subTest(code=code):
                self.world = NativeLookupWorld()
                self.world.entry['binding_phase'] = 'initial_before_parking'
                self.clock = BindingClock()
                self.world.raw_error = LookupRpcError(code, 'transport failure')
                with self.assertRaises((RuntimeError, grpc_api.RpcError)):
                    self.validated()
                self.assertEqual(len(self.world.inventories), 1)
                self.assertEqual(self.clock.sleeps, [])
                self.assertNotIn('verified_global_actor_id', self.world.entry['lookup_evidence'])

    def test_initial_known_global_not_found_still_uses_exact_global_proof_with_deadline(self):
        self.world.raw_error = LookupRpcError(grpc_api.StatusCode.NOT_FOUND, 'global lookup unavailable')
        observed = self.validated()
        self.assertEqual(live._id(observed), RAW)
        self.assertEqual(len(self.world.inventories), 1)
        evidence = self.world.entry['lookup_evidence']
        self.assertEqual(evidence['verified_global_actor_id'], RAW)
        self.assertIn('unsupported_lookup', evidence['pinned_global_address'])
        self.assertEqual(self.clock.sleeps, [])

    def test_failed_initial_numeric_history_survives_later_cleanup_binding(self):
        self.mismatch_pose()
        with self.assertRaises(RuntimeError):
            self.validated()
        initial = copy.deepcopy(self.world.entry['lookup_evidence'])
        self.world.entry['binding_phase'] = 'frozen'
        self.settle_alias()
        info, _ = live._owned_raw_info(self.world.grpc, self.world.entry, FIXTURE)
        self.assertEqual(info.id.value, RAW)
        self.assertIn(initial, self.world.entry['lookup_history'])
        self.assertEqual(self.world.entry['lookup_evidence']['verified_global_actor_id'], RAW)
        self.world.stub.DestroyActor.assert_not_called()

    def test_malformed_supported_alias_is_not_silently_replaced_by_global_state(self):
        self.world.alias.actor_info.id.value = ''
        with self.assertRaises(RuntimeError):
            self.validated()
        self.assertEqual(len(self.world.inventories), 1)
        self.assertEqual(self.clock.sleeps, [])
        self.assertNotIn('verified_global_actor_id', self.world.entry['lookup_evidence'])


class FinalNpcLookupIntegrationTest(unittest.TestCase):
    def setUp(self):
        preview_fixture.LivePreviewTest.setUp(self)

    candidate = preview_fixture.LivePreviewTest.candidate
    run_preview = preview_fixture.LivePreviewTest.run_preview
    assert_clean = preview_fixture.LivePreviewTest.assert_clean

    def test_final_npc_alias_unknown_uses_verified_global_state_and_cleans_without_respawn(self):
        original = self.sim.stub.GetActorState
        unknown_reads = []

        def individual(info, timeout):
            route_assigned = any(row[0] == info.id.value and row[1] == 'route' for row in self.sim.commands)
            if info.id.value in self.sim.aliases and route_assigned:
                unknown_reads.append(info.id.value)
                raise LookupRpcError()
            return original(info, timeout)

        self.sim.stub.GetActorState = individual
        report = self.run_preview(self.candidate('vehicle'))
        self.assertEqual(report['status'], 'valid_preview', report)
        self.assert_clean(report)
        self.assertEqual(len(self.sim.spawns), 1)
        self.assertGreater(len(unknown_reads), 0)
        observed = report['observed'][0]
        self.assertEqual(observed['owned_actor_state_at_validation']['raw_actor_id'], observed['raw_actor_id'])
        proof = observed['owned_actor_lookup_evidence']
        self.assertEqual(proof['verified_global_actor_id'], observed['raw_actor_id'])
        self.assertIn('unsupported_lookup', proof['creating_alias'])
        self.assertEqual(proof['native_request']['id'], observed['raw_actor_id'])

if __name__ == '__main__':
    unittest.main()
