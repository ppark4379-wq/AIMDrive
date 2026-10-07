import math
import unittest

from MoraiDrive.data_collection.situation_labeling.ego_position import update_static_positions
from MoraiDrive.data_collection.situation_labeling.side_view_hold import in_side_view


class SideViewHoldTests(unittest.TestCase):
    cameras = {'front': (1.9,0,0,math.radians(45)), 'right': (1.15,-0.65,math.radians(-70),math.radians(65)),
               'left': (1.15,0.65,math.radians(70),math.radians(65))}

    def test_hold_across_avoidance_and_release_after_extent_exits(self):
        rules=[]; objects=[]
        for index, forward in enumerate([5,4,1]):
            state='approach_maintain' if index==0 else 'avoid_continue'
            rules.append(dict(index=index,zone_id='static_obstacle',instruction_state=state,
                instruction_id='follow_approach_maintain' if index==0 else 'follow_avoid_continue',
                instruction='original',label='follow',variables=dict(target_actor_id='3',target_kind='obstacle',avoid_direction='right'),
                evidence=dict(target_actor_ref=dict(actor_id='3',instance_key='barrel'), relevant_obstacle_ids=['1','3'])))
            objects += [dict(index=index,actor_id='1',instance_key='bike',forward_m=forward,left_m=-2.5,actor_width_m=.4,actor_length_m=1),
                        dict(index=index,actor_id='3',instance_key='barrel',forward_m=15,left_m=1,actor_width_m=.5,actor_length_m=.5)]
        update_static_positions(rules,objects,cameras=self.cameras)
        self.assertIn('to the right of', rules[1]['instruction'])
        self.assertIn('Maintain', rules[1]['instruction'])
        self.assertNotIn('avoid_direction',rules[1]['variables'])
        self.assertEqual(rules[2]['instruction_state'],'avoid')
        self.assertIn('to the left of', rules[2]['instruction'])
        self.assertIn('maneuver to the right', rules[2]['instruction'])
        self.assertTrue(in_side_view(objects[4],self.cameras['right']))
        self.assertFalse(in_side_view(objects[4],self.cameras['front']))

    def test_edge_intersection_and_left_symmetry(self):
        obj=dict(forward_m=-1,left_m=-2.5,actor_width_m=1,actor_length_m=2)
        self.assertTrue(in_side_view(obj,self.cameras['right']))
        obj['left_m']=2.5
        self.assertTrue(in_side_view(obj,self.cameras['left']))
        obj['forward_m']=-10
        self.assertFalse(in_side_view(obj,self.cameras['left']))

    def test_missing_extent_is_unknown(self):
        self.assertIsNone(in_side_view(dict(forward_m=1,left_m=-3),self.cameras['right']))
