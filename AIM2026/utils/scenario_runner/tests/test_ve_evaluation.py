# Deterministic offline checks; no ROS master, simulator or model forwards.
import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

from scenario_runner.tools.ve_expert_benchmark.metrics import footprint, match_centers as benchmark_match_centers, score, compare_pair
from scenario_runner.tools.ve_expert_benchmark.__main__ import trial_config, fingerprint, render
from scenario_runner.tools.ve_expert_benchmark.scenario import EvaluationScenario, configure_expert
from scenario_runner.tools.ve_expert_benchmark.runtime import object_rows, PerceptionSession


def benchmark_box(x=12.,y=0.,identity='1',speed=3.,yaw=0.):
    return dict(center_world=[x,y],heading_deg=yaw,length_m=4.,width_m=2.,speed_mps=speed,track_id=identity)


def rows(n=10,objects=True):
    return [dict(t=i*.1,ego=dict(x=0.,y=0.,yaw_deg=0.,speed_mps=2.,collision_objects=[]),
        gt=[benchmark_box()] if objects else [],ve=[benchmark_box(identity=17)] if objects else [],
        gt_fresh=True,ve_fresh=True,progress_s=i,remaining_s=30.-i) for i in range(n)]


class MetricsTests(unittest.TestCase):
    def test_rotated_body_distance_is_not_center_distance(self):
        a,b=benchmark_box(x=0.),benchmark_box(x=7.,yaw=90.)
        self.assertAlmostEqual(4.,footprint(a).distance(footprint(b)))

    def test_match_assignment_is_one_to_one_and_gated(self):
        pairs=benchmark_match_centers([benchmark_box(x=0.),benchmark_box(x=1.)],[benchmark_box(x=0.)])
        self.assertEqual([(0,0,0.)],pairs)
        self.assertEqual([],benchmark_match_centers([benchmark_box(x=0.)],[benchmark_box(x=2.1)]))
        self.assertEqual(2,len(benchmark_match_centers([benchmark_box(x=0.),benchmark_box(x=-1.9)], [benchmark_box(x=0.),benchmark_box(x=1.9)])))

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
        data[4]['ve']=[benchmark_box(identity=18)]
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
               initial_scene=dict(ego=[0.,0.],objects=[benchmark_box()]))
        b=copy.deepcopy(a)
        self.assertTrue(compare_pair(a,b)['comparison_ready'])
        b['initial_scene']['objects'][0]['center_world']=[50.,0.]
        self.assertFalse(compare_pair(a,b)['comparison_ready'])

    def test_failed_gt_baseline_prevents_a_ready_comparison(self):
        a=dict(pair_contract_sha256='a',expert_config_sha256='b',metrics=score(rows(),'slow_lead',False),
               initial_scene=dict(ego=[0.,0.],objects=[benchmark_box()]))
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
        doc=dict(common=dict(sync_grpc_inputs=dict(enabled=True)),planning=dict(jamming_scene_geometry=[benchmark_box()],jamming_trial_contract={}))
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


import json

from MoraiDrive.scenario_runner.legacy_expert.ve_gt_comparison import (
    analyze, gt_motion_check, in_roi, match_centers, nearest, residual_jitter,
    roi, score_frame, serialize_gt, summarize, timeline)


def vector(x=0., y=0., z=0.):
    return NS(x=x, y=y, z=z)


def message():
    obj = NS(unique_id=7, type=1, name='car', position=vector(10., 0.),
             velocity=vector(10.8, 14.4), size=vector(4.4, 1.8, 1.5), heading=0.)
    return NS(header=NS(stamp=NS(to_nsec=lambda: 1_000_000_000), frame_id='ObjectInfo'),
              npc_list=[obj], obstacle_list=[], pedestrian_list=[], num_of_npcs=1,
              num_of_obstacle=0, num_of_pedestrian=0)


def box(x=10., y=0.):
    return dict(track_id='npc:7', kind='npc', name='car', center_world=[x,y],
                heading_deg=0., length_m=4.4, width_m=1.8, speed_mps=5.)


def prediction(x=10., y=0.):
    return dict(box(x,y), track_id=1, planning_speed_mps=5., motion_state='moving')


class ComparisonTests(unittest.TestCase):
    def test_live_native_size_axes_and_planar_kmh_conversion(self):
        row = serialize_gt(message(), 123)
        b = row['boxes'][0]
        self.assertEqual((b['length_m'], b['width_m'], b['speed_mps']), (4.4, 1.8, 5.))
        self.assertEqual(b['velocity_raw_kmh'], [10.8, 14.4, 0.])
        self.assertEqual(b['raw_size_xyz_m'], [4.4, 1.8, 1.5])

    def test_count_invalid_and_duplicate_gt_are_flagged(self):
        msg = message()
        msg.npc_list *= 2
        self.assertEqual(len(serialize_gt(msg, 1)['errors']), 2)
        msg = message()
        msg.npc_list[0].size.x = float('nan')
        row = serialize_gt(msg, 1)
        self.assertEqual(row['boxes'], [])
        self.assertTrue(row['errors'])

    def test_exact_identical_bbox_and_two_speed_outputs(self):
        p = prediction()
        p.update(speed_mps=4., planning_speed_mps=6., heading_deg=180.)
        result = score_frame([box()], [p], dict(x=0.,y=0.,yaw_deg=0.))
        m = result['matched'][0]
        self.assertAlmostEqual(m['iou'], 1.)
        self.assertEqual(m['heading_error_deg_mod180'], 0.)
        self.assertEqual(m['raw_speed_error_mps'], 1.)
        self.assertEqual(m['raw_speed_bias_mps'], -1.)
        self.assertEqual(m['planning_speed_error_mps'], 1.)

    def test_miss_and_false_prediction_are_not_hidden_by_regression(self):
        score = score_frame([box(),box(20.)], [prediction(),prediction(28.)], dict(yaw_deg=0.))
        result = summarize([dict(score=score)])
        self.assertEqual((result['missed_gt'], result['unmatched_ve']), (1,1))
        self.assertEqual(result['center_recall_2m'], .5)
        self.assertEqual(result['metrics']['center_error_m']['count'], 1)

    def test_outside_gate_cannot_steal_valid_assignment(self):
        self.assertEqual(len(match_centers([box(0.),box(3.)],
                                          [prediction(1.),prediction(5.)])),2)

    def test_gt_motion_checks_kmh_against_displacement(self):
        rows = []
        for i in range(20):
            m = message()
            m.npc_list[0].position.x = .4*i
            m.npc_list[0].velocity = vector(14.4)
            r = serialize_gt(m,1_000_000_000+i*100_000_000)
            r['source_stamp_ns'] = r['received_ns']
            rows.append(r)
        v = gt_motion_check(rows)['npc:7']
        self.assertAlmostEqual(v['displacement_to_raw_ratio']['mean'],1/3.6)
        self.assertLess(v['kmh_conversion_error_mps']['max'],1e-10)

    def test_residual_jitter_subtracts_real_motion(self):
        frames = []
        for i in range(5):
            g,p = box(10+i),prediction(10.3+i)
            frames.append(dict(target_ns=i*100_000_000,gt=[g],ve=[p],
                score=score_frame([g],[p],dict(yaw_deg=0.))))
        result = residual_jitter(frames)
        self.assertEqual(result['center_residual_step_m']['count'],4)
        self.assertLess(result['center_residual_step_m']['max'],1e-10)
        self.assertEqual(result['ve_id_switches'],0)

    def test_roi_includes_partial_and_uses_ego_rotation(self):
        area = roi(dict(x=0.,y=0.,yaw_deg=90.))
        self.assertTrue(in_roi(box(0.,30.), area))
        self.assertFalse(in_roi(box(0.,33.), area))

    def test_pairing_rejects_gap_and_stale_repeated_header(self):
        rows = [serialize_gt(message(), 1_000_000_000 + i*60_000_000) for i in range(4)]
        times = timeline(rows)
        self.assertIs(nearest(rows,times,1_010_000_000),rows[0])
        self.assertIsNone(nearest(rows,times,1_120_000_000))
        self.assertIsNone(nearest(rows,times,1_500_000_000))

    def test_comparison_filters_to_actual_expert_callbacks_and_saves_artifacts(self):
        from AIM2026.Localization.map_geometry import validate_map_basis
        from scenario_runner.utils.perception import MAP_BASIS
        g = serialize_gt(message(),1_000_000_000)
        p = prediction()
        p['heading_deg'] = validate_map_basis(MAP_BASIS).convergence_degrees([10.,0.,0.])
        packet = dict(sequence=1,checkpoint_sha256='hash',boxes=[p],
            ego_pose=dict(x=0.,y=0.,yaw_raw_deg=0.),
            audit=dict(selected_samples=dict(lidar=dict(received_ns=1_000_000_000))),
            delivery_audit=dict(ros_stamp_ns=123))
        other = copy.deepcopy(packet)
        other['delivery_audit']['ros_stamp_ns'] = 456
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            for name, rows in [('gt_objects',[g]),('adapter',[packet,other]),
                ('expert_input',[dict(ros_stamp_ns=123,received_ns=1_030_000_000)])]:
                (output/(name+'.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in rows))
            result = analyze(output)
            self.assertEqual(result['expert_packets'],1)
            self.assertAlmostEqual(result['summaries']['input_receipt']['metrics']['iou']['mean'],1.)
            self.assertEqual(result['summaries']['expert_receipt']['matched'],1)
            self.assertFalse(result['gt_used_for_driving'])
            self.assertTrue((output/'gt_comparison_review.html').is_file())


from MoraiDrive.scenario_runner.legacy_expert.ve_gt_geometry_audit import interpolate, metric



class GeometryAuditTests(unittest.TestCase):
    def test_source_time_interpolation_wrap_and_no_extrapolation(self):
        history = ([100_000_000,200_000_000], [dict(position=[0.,0.], heading_deg=179.),
                                             dict(position=[1.,0.], heading_deg=-179.)])
        r = interpolate(history,150_000_000,['position','heading_deg'])
        self.assertEqual(r['position'], [.5,0.])
        self.assertEqual(r['heading_deg'],180.)
        self.assertIsNone(interpolate(history,50_000_000,['position']))
        self.assertIsNone(interpolate(history,250_000_000,['position']))

    def test_gap_and_teleport_not_filled(self):
        history = ([100_000_000,400_000_000], [dict(position=[0.,0.]),dict(position=[1.,0.])])
        self.assertIsNone(interpolate(history,200_000_000,['position']))
        history = ([100_000_000,200_000_000], [dict(position=[0.,0.]),dict(position=[20.,0.])])
        self.assertIsNone(interpolate(history,150_000_000,['position']))

    def test_independent_bbox_origin_correction_keeps_prediction(self):
        gt, ve = box(10.), prediction(11.3)
        before = metric(gt,ve)
        corrected = dict(gt,center_world=[gt['center_world'][0]+1.3,0.])
        after = metric(corrected,ve)
        self.assertAlmostEqual(before['center_error_m'],1.3)
        self.assertEqual(after['center_error_m'],0.)
        self.assertAlmostEqual(after['iou'],1.)
        self.assertEqual(ve['center_world'],[11.3,0.])
        self.assertEqual(gt['center_world'],[10.,0.])


if __name__ == "__main__":
    unittest.main()
