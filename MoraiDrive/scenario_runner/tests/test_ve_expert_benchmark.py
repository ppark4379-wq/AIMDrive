"""Deterministic offline checks; no ROS master, simulator or model forwards."""
import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

from scenario_runner.tools.ve_expert_benchmark.metrics import footprint, match_centers, score, compare_pair
from scenario_runner.tools.ve_expert_benchmark.__main__ import trial_config, fingerprint, render
from scenario_runner.tools.ve_expert_benchmark.scenario import EvaluationScenario, configure_expert
from scenario_runner.tools.ve_expert_benchmark.runtime import object_rows, PerceptionSession


def box(x=12.,y=0.,identity='1',speed=3.,yaw=0.):
    return dict(center_world=[x,y],heading_deg=yaw,length_m=4.,width_m=2.,speed_mps=speed,track_id=identity)


def rows(n=10,objects=True):
    return [dict(t=i*.1,ego=dict(x=0.,y=0.,yaw_deg=0.,speed_mps=2.,collision_objects=[]),
        gt=[box()] if objects else [],ve=[box(identity=17)] if objects else [],
        gt_fresh=True,ve_fresh=True,progress_s=i,remaining_s=30.-i) for i in range(n)]


class MetricsTests(unittest.TestCase):
    def test_rotated_body_distance_is_not_center_distance(self):
        a,b=box(x=0.),box(x=7.,yaw=90.)
        self.assertAlmostEqual(4.,footprint(a).distance(footprint(b)))

    def test_match_assignment_is_one_to_one_and_gated(self):
        pairs=match_centers([box(x=0.),box(x=1.)],[box(x=0.)])
        self.assertEqual([(0,0,0.)],pairs)
        self.assertEqual([],match_centers([box(x=0.)],[box(x=2.1)]))
        self.assertEqual(2,len(match_centers([box(x=0.),box(x=-1.9)], [box(x=0.),box(x=1.9)])))

    def test_collision_blocks_pass_even_after_goal(self):
        data=rows()
        data[3]['ego']['collision_objects']=['vehicle_19']
        result=score(data,'slow_lead',True)
        self.assertTrue(result['collision_observed'])
        self.assertFalse(result['basic_trial_passed'])

    def test_missing_encounter_never_counts_as_obstacle_success(self):
        self.assertFalse(score(rows(objects=False),'slow_lead',True)['basic_trial_passed'])
        self.assertTrue(score(rows(objects=False),'empty',True)['basic_trial_passed'])

    def test_stale_gt_is_not_treated_as_a_clear_road(self):
        data=rows()
        data[2]['gt_fresh']=False
        result=score(data,'slow_lead',True)
        self.assertFalse(result['measurement_coverage_ok'])
        self.assertEqual(9,result['gt_center_observations_in_bev'])

    def test_ve_miss_and_id_change_are_counted_without_changing_gt(self):
        data=rows()
        data[3]['ve_fresh']=False
        data[4]['ve']=[box(identity=18)]
        result=score(data,'slow_lead',True)
        self.assertEqual(10,result['gt_center_observations_in_bev'])
        self.assertEqual(.9,result['center_match_2m_rate'])
        self.assertEqual(1,result['ve_stale_observations'])
        self.assertEqual(2,result['estimated_id_switches'])
        self.assertGreater(result['longest_any_object_center_miss_s'],0.)

    def test_large_observation_gap_is_not_success(self):
        data=rows()
        data[-1]['t']=2.
        self.assertFalse(score(data,'slow_lead',True)['measurement_coverage_ok'])

    def test_road_stop_candidates_do_not_claim_proven_false_stops(self):
        data=rows(50,False)
        for row in data:
            row['ego']['speed_mps']=0.
            row['remaining_s']=30.
        result=score(data,'empty',True)
        self.assertGreater(result['stop_without_near_ahead_object_candidate_s'],1.)
        self.assertFalse(result['unnecessary_stops_confirmed'])

    def test_paired_comparison_rejects_mismatched_initial_scene(self):
        metric=score(rows(),'slow_lead',True)
        a=dict(pair_contract_sha256='a',expert_config_sha256='b',metrics=metric,
               initial_scene=dict(ego=[0.,0.],objects=[box()]))
        b=copy.deepcopy(a)
        self.assertTrue(compare_pair(a,b)['comparison_ready'])
        b['initial_scene']['objects'][0]['center_world']=[50.,0.]
        self.assertFalse(compare_pair(a,b)['comparison_ready'])

    def test_failed_gt_baseline_prevents_a_ready_comparison(self):
        a=dict(pair_contract_sha256='a',expert_config_sha256='b',metrics=score(rows(),'slow_lead',False),
               initial_scene=dict(ego=[0.,0.],objects=[box()]))
        self.assertFalse(compare_pair(a,a)['comparison_ready'])


class WiringTests(unittest.TestCase):
    def test_stop_is_verified_before_actor_cleanup_and_after_brake_release(self):
        obj=object.__new__(EvaluationScenario)
        events=[]
        obj.grpc=NS(world=object())
        obj.trace=Mock()
        obj.stopped_verified=False
        obj.stop_expert_controller=lambda: events.append('stop_expert')
        obj._wait_stopped=lambda: events.append('observed_stop')
        with patch('scenario_runner.tools.ve_expert_benchmark.scenario.MissionScenario.cleanup',side_effect=lambda:events.append('actor_cleanup')), \
             patch('scenario_runner.tools.ve_expert_benchmark.scenario.stopping.hold_common_brake',side_effect=lambda _:events.append('hold')), \
             patch('scenario_runner.tools.ve_expert_benchmark.scenario.stopping.release_common_brake',side_effect=lambda _:events.append('release')), \
             patch('scenario_runner.tools.ve_expert_benchmark.scenario.stopping._require_no_controller'):
            obj.cleanup()
        self.assertEqual(['stop_expert','hold','observed_stop','actor_cleanup','release','observed_stop','release'],events)
        self.assertTrue(obj.stopped_verified)

    def test_post_teleport_readiness_rejects_old_sensor_pose(self):
        session=object.__new__(PerceptionSession)
        session.error=None
        session.sender=Mock()
        session.sender.poll.return_value=None
        data=dict(gt_fresh=True,ve_fresh=True,ego_age_s=.1,ego=dict(x=0.,y=0.),
                  packet=dict(sensor_received_ns=10_000_000_000,ego_pose=dict(x=0.,y=0.)))
        session.snapshot=lambda:data
        with self.assertRaisesRegex(RuntimeError,'post-placement'):
            session.require_ready(since=11.)
        data['packet']['ego_pose']['x']=3.
        with self.assertRaisesRegex(RuntimeError,'pose mismatch'):
            session.require_ready(since=9.)
        data['packet']['ego_pose']['x']=.2
        self.assertIs(data,session.require_ready(since=9.))

    def test_only_object_remap_differs_between_arms(self):
        from scenario_runner.main_runner import load_configuration
        _,_,base=load_configuration()
        spec=dict(case='slow_lead',seed=26101400,weather='SUNNY',hour=15,arm='GT')
        gt=trial_config(base,spec,Path('/tmp/private'))
        ve=trial_config(base,dict(spec,arm='VE'),Path('/tmp/private'))
        self.assertEqual(['/Object_topic:=/ve/Object_topic'],ve.pop('expert_ros_remaps'))
        self.assertEqual([],gt.pop('expert_ros_remaps'))
        self.assertEqual(gt,ve)
        self.assertFalse(gt['sync_runtime']['enabled'])
        self.assertEqual(1,gt['max_laps'])
        self.assertEqual(180.,gt['timeout_sec'])

    def test_scene_gt_injection_removed_and_grpc_inputs_disabled(self):
        doc=dict(common=dict(sync_grpc_inputs=dict(enabled=True)),planning=dict(jamming_scene_geometry=[box()],jamming_trial_contract={}))
        configured=configure_expert(doc)
        self.assertFalse(configured['common']['sync_grpc_inputs']['enabled'])
        self.assertEqual('ros',configured['common']['perception_source'])
        self.assertEqual({},configured['planning'])

    def scenario_with_topology(self,pubs,subs):
        obj=object.__new__(EvaluationScenario)
        ros=Mock()
        ros.get_master.return_value.getSystemState.return_value=(1,'ok',[pubs,subs,[]])
        ros.get_name.return_value='/observer'
        obj.session=NS(ros=ros)
        obj.spec=dict(arm='VE')
        obj.expert_controller=NS(process=NS(pid=123))
        obj.topology_checks=0
        return obj

    def test_expert_initialization_wait_is_allowed_but_duplicate_control_is_not(self):
        obj=self.scenario_with_topology([],[])
        self.assertFalse(obj.verify_topology(allow_wait=True))
        obj=self.scenario_with_topology([('/ctrl_cmd',['/foreign'])],[])
        with self.assertRaisesRegex(RuntimeError,'제어 노드'):
            obj.verify_topology(allow_wait=True)

    def test_ve_expert_cannot_subscribe_to_gt_even_if_ve_also_connected(self):
        node='/morai_standard_123_000'
        pubs=[('/ctrl_cmd',[node]),('/ve/Object_topic',['/observer'])]
        subs=[('/ve/Object_topic',[node]),('/Object_topic',[node])]
        obj=self.scenario_with_topology(pubs,subs)
        with self.assertRaisesRegex(RuntimeError,'GT 장애물'):
            obj.verify_topology()
        obj=self.scenario_with_topology(pubs,subs[:1])
        self.assertTrue(obj.verify_topology())

    def test_gt_units_and_bad_geometry(self):
        obj=NS(position=NS(x=10.,y=2.),size=NS(x=4.,y=2.),heading=90.,velocity=NS(x=36.),unique_id=5,name='car')
        message=NS(npc_list=[obj],obstacle_list=[],pedestrian_list=[])
        self.assertEqual(10.,object_rows(message)[0]['speed_mps'])
        obj.size.x=0.
        with self.assertRaises(ValueError):
            object_rows(message)

    def test_saved_review_escapes_errors_and_does_not_convert_failed_trial_to_pass(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)
            r=dict(spec=dict(case='empty',arm='GT',repeat=1),output=str(p),
                   metrics=score(rows(objects=False),'empty',False,'<failure>'))
            result=render(p,[r],'BLOCKED')
            self.assertEqual(0,result['passed_trials'])
            self.assertIn('&lt;failure&gt;', (p/'review.html').read_text())


if __name__=='__main__':
    unittest.main()
