import unittest
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

from scenario_runner.adapters.pedestrian_binding import bind_spawn, PedestrianSpawnNotObserved


class SpawnBindingTests(unittest.TestCase):
    def bind(self, inventories, rejected=True, model='NCAP_EBT'):
        state = NS(actor_info=NS(id=NS(value='14'), object_type=2, client_key=''),
                   transform=NS(location=NS(x=0., y=0.)))
        client = NS(client_key='runner',
                    get_global_pedestrian_states=Mock(side_effect=[([state] if x else []) for x in inventories]),
                    world=NS(_sim_adapter=NS(_actor_stub=NS(GetOptionName=Mock(return_value=NS(value=model))))))
        with patch('scenario_runner.adapters.pedestrian_binding.time.sleep'):
            return bind_spawn(client, [], 'alias', 'NCAP_EBT', state.transform, 2.5, 0.,
                              creation_rejected=rejected)

    def test_rejection_with_ten_empty_observations_is_distinguished(self):
        with self.assertRaises(PedestrianSpawnNotObserved):
            self.bind([False]*10)

    def test_acknowledged_spawn_without_binding_is_not_safe_to_skip(self):
        with self.assertRaises(RuntimeError) as caught:
            self.bind([False]*10, rejected=False)
        self.assertNotIsInstance(caught.exception, PedestrianSpawnNotObserved)

    def test_unmatched_new_actor_is_not_safe_to_skip(self):
        with self.assertRaises(RuntimeError) as caught:
            self.bind([True]+[False]*9, model='Man1')
        self.assertNotIsInstance(caught.exception, PedestrianSpawnNotObserved)

    def test_rejected_rpc_with_late_actor_recovers_native_identity(self):
        actor = self.bind([False, False, True])
        self.assertEqual(actor.get_id(), '14')
