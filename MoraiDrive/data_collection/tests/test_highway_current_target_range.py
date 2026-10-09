import unittest
from MoraiDrive.data_collection.situation_labeling.highway_target_selection import select_current_highway_target


def frame():
    return {'index':10,'lidar_source_ns':1000000000,'zone_id':'highway',
        'valid':True,'dataset_build_candidate':True,'object_inventory_usable':True,
        'highway_nearby_geometry_unverified':False,'lead_actor_id':None,
        'ego_current_lane_id':'L2'}


def actor(distance=45.):
    return {'actor_id':'21','index':10,'lidar_source_ns':1000000000,
        'sensor_time_ns':1000000000,'kind':'npc','instance_key':'native21',
        'model_identity_verified':True,'semantic_class_valid':True,
        'semantic_class':'vehicle','rule_geometry_usable':True,'forward_m':distance,
        'left_m':0.,'actor_width_m':2.,'distance_to_ego_m':distance,
        'highway_lane_id':'L3'}


class RangeTests(unittest.TestCase):
    def test_45m_current_actor_survives_descriptive_boundary(self):
        a,role,reasons,receipt=select_current_highway_target(frame(),[actor()])
        self.assertEqual(a['actor_id'],'21');self.assertEqual(role,'approaching_vehicle')
        self.assertEqual(reasons,[]);self.assertEqual(receipt['description_range_m'],85.)

    def test_adjacent_vehicle_is_never_promoted_to_lead(self):
        other=actor();other['left_m']=3.5
        _,role,_,_=select_current_highway_target(frame(),[other])
        self.assertEqual(role,'approaching_vehicle')
        f=frame();f.update(lead_actor_id='21',lead_distance_m=45.)
        a,role,reasons,_=select_current_highway_target(f,[other])
        self.assertIsNone(a);self.assertIn('current_highway_lead_unverified',reasons)

    def test_beyond_verified_current_cohort_not_selected(self):
        a,role,reasons,_=select_current_highway_target(frame(),[actor(86.)])
        self.assertIsNone(a);self.assertIsNone(role);self.assertEqual(reasons,[])

    def test_unverified_or_stale_45m_actor_is_not_described(self):
        for update in ({'model_identity_verified':False},{'rule_geometry_usable':False},
                       {'lidar_source_ns':900000000},{'semantic_class_valid':False}):
            other=actor();other.update(update)
            a,_,reasons,_=select_current_highway_target(frame(),[other])
            self.assertIsNone(a);self.assertIn('current_highway_approach_unverified',reasons)


if __name__=='__main__':unittest.main()
