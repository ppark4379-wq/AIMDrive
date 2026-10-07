import copy
import unittest

from MoraiDrive.data_collection.situation_labeling.ego_position import ego_position, update_static_positions


class EgoPositionTests(unittest.TestCase):
    def rule(self):
        return dict(index=7, zone_id='static_obstacle', instruction_id='follow_approach_maintain',
            instruction_state='approach_maintain', instruction='old', instruction_consistency='consistent',
            variables=dict(target_actor_id='1', target_kind='obstacle', target_side='right', avoid_direction='left'),
            evidence=dict(target_actor_ref=dict(actor_id='1', instance_key='instance')))

    def obj(self, **kwargs):
        return dict(index=7, actor_id='1', instance_key='instance', forward_m=10, left_m=2, **kwargs)

    def test_vehicle_crosses_route_side(self):
        rule = self.rule()
        update_static_positions([rule], [self.obj()])
        self.assertIn('to the left of', rule['instruction'])
        self.assertEqual(rule['variables']['avoid_direction'], 'left')
        self.assertEqual(rule['evidence']['route_target_side'], 'right')

    def test_missing_geometry_does_not_use_route_side(self):
        rule = self.rule()
        update_static_positions([rule], [])
        self.assertIsNone(rule['instruction'])
        self.assertEqual(rule['instruction_missing_variables'], ['position'])

    def test_held_instruction_uses_current_frame(self):
        rule = self.rule()
        rule['evidence']['instruction_held_from_index'] = 6
        old = self.obj(); old.update(index=6, left_m=-2)
        update_static_positions([rule], [old, self.obj()])
        self.assertIn('to the left of', rule['instruction'])

    def test_exclusion_remains_excluded(self):
        rule = self.rule()
        rule.update(instruction=None, instruction_dropped='old', instruction_consistency='dropped_direction')
        update_static_positions([rule], [self.obj()])
        self.assertIsNone(rule['instruction'])
        self.assertIn('to the left of', rule['instruction_dropped'])
        self.assertEqual(rule['instruction_consistency'], 'dropped_direction')

    def test_identity_collision_is_unresolved(self):
        rule = self.rule()
        obj = self.obj(); obj['instance_key'] = 'other'
        update_static_positions([rule], [obj])
        self.assertIsNone(rule['instruction'])

    def test_heading_and_center_conventions(self):
        self.assertEqual(ego_position(10, -1), 'right')
        self.assertEqual(ego_position(10, 0.2), 'front')
        self.assertEqual(ego_position(-10, 0), 'rear')
        self.assertIsNone(ego_position(10, float('nan')))

    def test_pair_uses_individual_positions(self):
        rule = self.rule(); rule.update(instruction_id='follow_between', instruction_state='pass_between')
        rule['evidence']['corridor_pair_actor_refs'] = [dict(actor_id='1'), dict(actor_id='2')]
        other = self.obj(); other.update(actor_id='2', left_m=-2)
        update_static_positions([rule], [self.obj(), other])
        self.assertIn('to the left and right of', rule['instruction'])
        self.assertEqual(rule['variables']['target_position'], 'left_right')
        self.assertEqual(len(rule['evidence']['pair_actor_positions']), 2)

    def test_pair_positions_do_not_depend_on_centroid_or_actor_order(self):
        for lefts, expected in [((1,-4),'left_right'), ((-4,1),'left_right'),
                                ((1,4),'left'), ((-1,-4),'right'), ((0,2),'front_left')]:
            with self.subTest(lefts=lefts):
                rule=self.rule(); rule.update(instruction_id='follow_between',instruction_state='pass_between')
                rule['evidence']['corridor_pair_actor_refs']=[dict(actor_id='1'),dict(actor_id='2')]
                first=self.obj(); first['left_m']=lefts[0]
                second=self.obj(); second.update(actor_id='2',left_m=lefts[1])
                update_static_positions([rule],[first,second])
                self.assertEqual(rule['variables']['target_position'],expected)
                self.assertIn('There are two obstacles',rule['instruction'])

    def test_pair_requires_two_distinct_current_actors(self):
        rule=self.rule(); rule.update(instruction_id='follow_between',instruction_state='pass_between')
        rule['evidence']['corridor_pair_actor_refs']=[dict(actor_id='1'),dict(actor_id='2')]
        update_static_positions([rule],[self.obj()])
        self.assertIsNone(rule['instruction'])

    def test_other_zones_unchanged_and_idempotent(self):
        rule = self.rule(); rule['zone_id'] = 'dynamic_obstacle'
        original = copy.deepcopy(rule)
        update_static_positions([rule], [self.obj()])
        self.assertEqual(rule, original)

    def test_near_right_obstacle_wins_during_maintain(self):
        rule = self.rule()
        rule['evidence']['relevant_obstacle_ids'] = ['1', '2']
        far = self.obj(); far.update(forward_m=17, left_m=0.58)
        near = self.obj(); near.update(actor_id='2', instance_key='bike', forward_m=8, left_m=-2.5)
        update_static_positions([rule], [far, near])
        self.assertEqual(rule['related_actors'], ['2'])
        self.assertIn('to the right of', rule['instruction'])
        self.assertEqual(rule['evidence']['rule_target_actor_ref']['actor_id'], '1')

    def test_avoidance_keeps_its_own_target(self):
        rule = self.rule()
        rule.update(instruction_state='avoid', instruction_id='follow_avoid')
        rule['evidence']['relevant_obstacle_ids'] = ['1', '2']
        far = self.obj(); far.update(forward_m=17, left_m=0.58)
        near = self.obj(); near.update(actor_id='2', instance_key='bike', forward_m=8, left_m=-2.5)
        update_static_positions([rule], [far, near])
        self.assertEqual(rule['variables']['target_actor_id'], '1')
        self.assertIn('to the left of', rule['instruction'])

    def test_hysteresis_then_switch_when_target_passes(self):
        rules = [self.rule() for _ in range(3)]
        objects = []
        for i, rule in enumerate(rules):
            rule['index'] = 7+i
            rule['evidence']['relevant_obstacle_ids'] = ['1','2']
            for actor, forward in [('1', [8,8,-1][i]), ('2', [9,7.8,7][i])]:
                obj = self.obj(); obj.update(index=7+i,actor_id=actor,instance_key=actor,forward_m=forward,left_m=0)
                objects.append(obj)
        update_static_positions(rules, objects)
        self.assertEqual([r['variables']['target_actor_id'] for r in rules], ['1','1','2'])
        rule = self.rule(); update_static_positions([rule], [self.obj()])
        original = copy.deepcopy(rule)
        update_static_positions([rule], [self.obj()])
        self.assertEqual(rule, original)
