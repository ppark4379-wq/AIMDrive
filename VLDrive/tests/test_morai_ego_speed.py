import unittest
from data.src.loaders.morai_ego_speed import CausalEgoSpeed


class EgoSpeedTests(unittest.TestCase):
    def test_no_future_and_age_bound(self):
        speeds = CausalEgoSpeed([(0, 4.0, 0), (100000000, 9.0, 1)])
        self.assertEqual(speeds.at(50000000)[0]['velocity_mps'], 4.0)
        self.assertEqual(speeds.at(100000000)[0]['velocity_mps'], 9.0)
        self.assertEqual(speeds.at(-1), (None, 'ego_no_preceding_sample'))
        self.assertEqual(speeds.at(50000001), (None, 'ego_older_than_50ms'))

    def test_invalid_and_conflict(self):
        for value in (float('nan'), -1.0):
            self.assertEqual(CausalEgoSpeed([(0, value, 0)]).at(0), (None, 'ego_invalid_speed'))
        conflict = CausalEgoSpeed([(0, 1.0, 0), (0, 2.0, 1)])
        self.assertEqual(conflict.at(0), (None, 'ego_conflicting_duplicate'))
        retransmit = CausalEgoSpeed([(0, 1.0, 0), (0, 1.0, 1)])
        self.assertEqual(retransmit.at(0)[0]['ego_record_id'], 0)

    def test_empty(self):
        with self.assertRaises(ValueError):
            CausalEgoSpeed([])


if __name__ == '__main__':
    unittest.main()
