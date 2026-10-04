import unittest
from data.src.loaders.morai_ego_speed import ego_speed_timeline, causal_ego_speed
class EgoSpeedTests(unittest.TestCase):
    def test_no_future_and_age_bound(self):
        t,v=ego_speed_timeline([{'role':'ego','source_timestamp_ns':n,'signal':{'speed':s}} for n,s in [(0,4),(100000000,9)]])
        self.assertEqual(causal_ego_speed(t,v,50000000),4)
        self.assertEqual(causal_ego_speed(t,v,100000000),9)
        for n in [-1,50000001]:
            with self.assertRaises(ValueError):causal_ego_speed(t,v,n)
    def test_invalid_and_conflict(self):
        for values in [[float('nan')],[-1],[71],[1,2]]:
            with self.assertRaises(ValueError):ego_speed_timeline([{'role':'ego','source_timestamp_ns':0,'signal':{'speed':s}} for s in values])
if __name__=='__main__':unittest.main()
