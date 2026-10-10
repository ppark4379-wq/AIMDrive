import math
import unittest
import json
import tempfile
from pathlib import Path

from MoraiDrive.data_collection.situation_labeling.highway_plan import (
    classify_selected_path, current_rule_features, lane_corridor,
    attach_current_highway_plans, bind_current_highway_plan_rules,
    validate_current_highway_plan,
)
from MoraiDrive.data_collection.situation_labeling.mapper import attach_rule_instructions


class Map:
    def __init__(self, curves=False):
        def points(y):
            return [[x, y+(0.002*x*x if curves else 0)] for x in range(101)]
        self.links = {}
        for name,offset,lane in (("c",0,2),("l",3.5,1),("r",-3.5,3)):
            self.links[name]={"points":points(offset),"ego_lane":lane,
                "from_node_idx":name+"0","to_node_idx":name+"1"}
        self.links['c'].update(left_lane_change_dst_link_idx='l',
                              right_lane_change_dst_link_idx='r')


class SharedEndpointMap(Map):
    """Two original lanes merge, then their map successors diverge."""
    def __init__(self, merge_x):
        super().__init__()
        self.links['c']['points'] = [[x, 0.] for x in range(merge_x + 1)]
        self.links['r']['points'] = [
            [x, -3.5 * min(1., (merge_x - x) / 20.)]
            for x in range(merge_x + 1)]
        for name, lane, offset in [('c', 2, 0.), ('r', 3, -.1)]:
            self.links[name]['to_node_idx'] = 'merge'
            self.links[name + '_next'] = {
                'points': [[x, offset * (x - merge_x)]
                           for x in range(merge_x, merge_x + 101)],
                'ego_lane': lane, 'from_node_idx': 'merge',
                'to_node_idx': name + '_end'}


def frame():
    return {"index":20,"lidar_source_ns":2000000000,"current_link_id":"c",
            "ego_world_x_m":0.,"ego_world_y_m":0.,
            "highway_lane_association_ambiguous":False}


def publication(points):
    return {"status":"RECEIVE_BOUND_CURRENT_CONTEXT","sensor_ns":2000000000,
        "ego":{"world_xyz":[0.,0.,0.],"speed_mps":10.},
        "local_path":{"header":{"frame":"map"},"world_xyz":points,
            "payload_sha256":"fixture_sha","record_id":1,"receive_age_ms":10}}


class SelectedPlanTests(unittest.TestCase):
    def test_curvature_body_offset_does_not_create_change(self):
        m=Map(curves=True)
        p=publication([[x,.002*x*x,0.] for x in range(81)])
        result=classify_selected_path(frame(),p,m)
        self.assertTrue(result['valid']);self.assertEqual(result['decision'],'keep_current_lane')
        self.assertGreater(p['local_path']['world_xyz'][60][1],3.5)

    def test_current_selected_right_before_motion_at_zero_speed(self):
        p=publication([[x,-3.5*min(1.,x/25.)**2*(3-2*min(1.,x/25.)),0.] for x in range(81)])
        p['ego']['speed_mps']=0.
        result=classify_selected_path(frame(),p,Map())
        self.assertTrue(result['valid']);self.assertEqual(result['direction'],'right')

    def test_future_executed_poses_cannot_backdate_change(self):
        f=frame();f['future_ego_y']=-3.5;f['lane_change_active']=True
        result=classify_selected_path(f,publication([[x,0.,0.] for x in range(81)]),Map())
        self.assertEqual(result['decision'],'keep_current_lane')
        self.assertFalse(result['uses_future_ego'])

    def test_stale_future_wrong_frame_publication_rejected(self):
        for change in ({'receive_age_ms':101},{'receive_age_ms':-1},
                       {'header':{'frame':'base_link'}},{'payload_sha256':None}):
            p=publication([[x,0.,0.] for x in range(81)]);p['local_path'].update(change)
            self.assertFalse(classify_selected_path(frame(),p,Map())['valid'])

    def test_frame_time_mismatch_rejected(self):
        p=publication([[x,0.,0.] for x in range(81)]);p['sensor_ns']+=1
        self.assertFalse(classify_selected_path(frame(),p,Map())['valid'])

    def test_wrong_map_lane_and_ambiguity_rejected(self):
        for key,value in [('current_link_id','missing'),('highway_lane_association_ambiguous',True)]:
            f=frame();f[key]=value
            self.assertFalse(classify_selected_path(f,publication([[x,0.,0.] for x in range(81)]),Map())['valid'])

    def test_current_control_does_not_mutate_observed_braking(self):
        f=frame();f.update(ego_hard_braking=True,ego_decelerating=True,ego_stopped=False)
        f['current_highway_selected_plan']={'valid':True,'command':{'stopped':False,
             'decelerating':False,'hard_braking':False,'kind':'commanded_acceleration'}}
        result=current_rule_features(f)
        self.assertFalse(result['ego_hard_braking']);self.assertTrue(f['ego_hard_braking'])

    def test_ambiguous_fork_cannot_merge_two_lanes_into_keep_corridor(self):
        m=Map();m.links['c']['to_node_idx']='shared'
        for name,y,lane in [('a',0.,1),('b',0.,3)]:
            m.links[name]={'points':[[100,y],[150,y]],'ego_lane':lane,
                          'from_node_idx':'shared','to_node_idx':name+'end'}
        corridor,reason=lane_corridor(m,'c',{},max_length=300)
        self.assertEqual(corridor,['c']);self.assertEqual(reason,'ambiguous_lane_continuation')

    def test_postmerge_branch_is_not_a_change_into_original_adjacent_lane(self):
        m = SharedEndpointMap(20)
        p = publication([[x, -.1 * max(0, x - 20), 0.] for x in range(81)])
        result = classify_selected_path(frame(), p, m)
        self.assertFalse(result['valid'])
        self.assertEqual(result['reason'], 'published_path_only_postmerge_branch')
        support = result['shared_endpoint_lane_change_support']
        self.assertEqual(support['maximum_consecutive_premerge_destination_samples'], 0)
        self.assertEqual(result['corridors']['keep']['links'], ['c', 'c_next'])
        self.assertEqual(result['corridors']['right']['links'], ['r', 'r_next'])

    def test_real_lane_change_before_shared_endpoint_remains_supported(self):
        m = SharedEndpointMap(100)
        p = publication([[x, -3.5 * min(1., x / 25.) ** 2 *
                          (3 - 2 * min(1., x / 25.)), 0.] for x in range(81)])
        result = classify_selected_path(frame(), p, m)
        self.assertTrue(result['valid'])
        self.assertEqual(result['decision'], 'select_adjacent_lane')
        self.assertEqual(result['direction'], 'right')
        self.assertTrue(result['shared_endpoint_lane_change_support']['valid'])

    def test_keep_lane_with_shared_endpoint_does_not_require_lane_change_support(self):
        result = classify_selected_path(
            frame(), publication([[x, 0., 0.] for x in range(81)]),
            SharedEndpointMap(100))
        self.assertTrue(result['valid'])
        self.assertEqual(result['decision'], 'keep_current_lane')

    def test_current_keep_replaces_false_future_return_and_final_validator_recomputes(self):
        with tempfile.TemporaryDirectory() as root:
            m=Map();links=[dict(idx=k,**v) for k,v in m.links.items()]
            (Path(root)/'link_set.json').write_text(json.dumps(links))
            f=frame();f.update(zone_id='highway',valid=True,dataset_build_candidate=True,
                object_inventory_usable=True,highway_nearby_geometry_unverified=False,
                ego_stopped=False,ego_decelerating=False,ego_hard_braking=False,
                ego_speed_mps=10.,lane_change_active=False,lane_change_direction=None)
            f['expert_publications']=publication([[x,0.,0.] for x in range(81)])
            r={'index':20,'zone_id':'highway','label_candidate':'CHANGE_RIGHT',
               'instruction_state':'return_to_source_lane','status':'matched',
               'evidence':{},'variables':{},'unknown_reasons':[]}
            attach_current_highway_plans([f],[],{'map_root':root})
            bind_current_highway_plan_rules([f],[],[r]);attach_rule_instructions([r])
            self.assertEqual(r['instruction'],'Continue driving.')
            self.assertEqual(validate_current_highway_plan([f],[],r),(True,[]))
            r['instruction']='Return to the original lane while moving forward.'
            self.assertFalse(validate_current_highway_plan([f],[],r)[0])
            r['instruction']='Continue driving.'
            f['expert_publications']['local_path']['world_xyz'][20][1]=-3.5
            self.assertFalse(validate_current_highway_plan([f],[],r)[0])

    def test_selected_change_not_deleted_only_because_expert_has_not_moved(self):
        from MoraiDrive.data_collection.situation_labeling.highway_maneuver import align_current_highway_maneuvers
        r={'zone_id':'highway','instruction_id':'change_right','instruction':'Move into the right lane.',
           'evidence':{'current_highway_selected_plan':{'valid':True}}}
        counts=align_current_highway_maneuvers([], [r])
        self.assertEqual(r['instruction'],'Move into the right lane.')
        self.assertEqual(counts['captured_selected_plan_uses_own_validator'],1)

    def test_missing_current_plan_cannot_leave_strong_action_as_gt(self):
        f=frame();f['current_highway_selected_plan']={'valid':False,'reason':'published_path_missing'}
        r={'index':20,'zone_id':'highway','label_candidate':'CHANGE_RIGHT',
           'instruction_state':'hard_braking','observation_valid':True,
           'evidence':{},'unknown_reasons':[]}
        bind_current_highway_plan_rules([f],[],[r])
        self.assertIs(r['action_valid'],False);self.assertIs(r['observation_valid'],True)
        self.assertIn('current_highway_selected_plan_unverified',r['unknown_reasons'])
        self.assertEqual(r['label_candidate'],'CHANGE_RIGHT')
        self.assertFalse(r['evidence']['current_highway_selected_plan']['valid'])

    def test_missing_plan_does_not_blanket_exclude_current_keep_observation(self):
        f=frame();f['current_highway_selected_plan']={'valid':False,'reason':'published_path_missing'}
        r={'index':20,'zone_id':'highway','label_candidate':'KEEP_LANE',
           'instruction_state':'approach_keep','observation_valid':True,
           'evidence':{},'unknown_reasons':[]}
        bind_current_highway_plan_rules([f],[],[r])
        self.assertNotIn('action_valid',r);self.assertIs(r['observation_valid'],True)


if __name__=='__main__':unittest.main()
