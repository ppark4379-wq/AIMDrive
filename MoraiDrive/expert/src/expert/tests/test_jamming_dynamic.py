"""Moving GT must not change static avoidance or latch an emergency stop."""
import contextlib
import io
import json
import math
from pathlib import Path
import unittest

import numpy as np

from expert.src.localization.point import Point
from expert.src.vehicle_state import VehicleState
from expert.src.obstacle.object_info import ObjectInfo
from expert.src.planning.Jamming.recorded_planner import JammingPlanner
from expert.src.planning.Jamming.dynamic_planner import DynamicObstaclePlanner
from expert.src.planning.Jamming.motion import stationary, moving
from expert.src.planning.Jamming.dynamic_recovery import make_recovery, recovery_result, clearance, replan_tracker
from expert.src.autonomous_driving import AutonomousDriving


class JammingDynamicTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root=Path(__file__).resolve().parents[3]
        cls.route=[Point(*row[:2]) for row in np.loadtxt(
            cls.root/'map_data/R_KR_PG_KATRI_2025/2026_molit_comp_global_path.txt')]

    def setUp(self):
        self.planner=JammingPlanner(background=False)
        self.addCleanup(self.planner.close)
        self.section=self.planner.section
        coords=np.asarray(self.planner._path(self.route))
        self.index=int(np.argmin(np.linalg.norm(coords-np.array([-.6,8.]),axis=1)))
        a,b=self.route[self.index:self.index+2]
        self.ego=VehicleState(*a,math.atan2(b.y-a.y,b.x-a.x),5.)
        self.reference=self.route[self.index:self.index+80]

    def obj(self,x,y,speed=2.,heading=math.pi/2,uid=1,static=False,length=4.5,width=1.8):
        return ObjectInfo(*self.section.to_world((x,y)),speed,1,is_static=static,
            size_x=length,size_y=width,heading_deg=math.degrees(heading-self.section.rotation),unique_id=uid)

    def run_scene(self,objects):
        return self.planner.run(self.ego,self.reference,[{'object_info':o} for o in objects],
                                global_path=self.route,desired_velocity=8.)

    def test_speed_classification_is_local_and_native_source_does_not_hide_motion(self):
        obj=self.obj(-1.,25.,speed=0.)
        self.assertTrue(stationary(obj))
        self.assertFalse(obj.is_static)
        self.assertEqual(len(self.planner._obstacles([{'object_info':obj}])),1)
        obj.velocity=2.
        obj.is_static=True
        self.assertTrue(moving(obj))
        self.assertEqual(self.planner._obstacles([{'object_info':obj}]),[])

    def test_front_mover_prefers_validated_avoidance_over_following(self):
        lead=self.obj(-1.,25.)
        for _ in range(12):
            result=self.run_scene([lead])
            if self.planner.dynamic.active:
                break
        self.assertTrue(self.planner.dynamic.active,result.debug_details)
        d=result.debug_details['jamming_dynamic']
        self.assertEqual(d['state'],'AVOID')
        self.assertIsNone(d['collision'])
        self.assertGreater(result.acc_target_velocity,0.)
        tracker=self.planner.dynamic.tracker
        xy=np.asarray(tracker.path)
        yaw=np.array([tracker.planner._path_heading(tracker.path,i) for i in range(len(xy))])
        self.assertTrue(tracker._envelope_clear(xy,yaw))
        maneuver=xy[:,1]<=self.planner.dynamic.return_y
        self.assertTrue(tracker._envelope_clear(xy[maneuver],yaw[maneuver],body_margin=.25))
        self.assertIsNone(DynamicObstaclePlanner.conflict(tracker.path,self.planner._ego(self.ego),
            self.planner._moving_objects([{'object_info':lead}],self.ego),result.acc_target_velocity))
        self.assertIs(result.path,result.candidates[0].points)

    def test_occupied_escape_corridor_follows_moving_lead(self):
        objects=[self.obj(-1.,25.),self.obj(-4.,24.,uid=2),self.obj(-7.,24.,uid=3)]
        for _ in range(8):
            result=self.run_scene(objects)
            self.assertFalse(self.planner.dynamic.active,result.debug_details)
            d=result.debug_details['jamming_dynamic']
            self.assertIn(d['state'],('FOLLOW','YIELD'),d)
            self.assertGreater(result.acc_target_velocity,0.)
            self.assertFalse(d['emergency'])
        self.assertLess(result.acc_target_velocity,8.)

    def test_emergency_rechecks_gt_and_releases_after_risk_moves_away(self):
        local=self.planner._ego(self.ego)
        hazard=self.obj(local.position.x,local.position.y+3.,speed=1.)
        for _ in range(3):
            result=self.run_scene([hazard])
            self.assertEqual(result.debug_details['jamming_dynamic']['state'],'EMERGENCY')
            self.assertEqual(result.acc_target_velocity,0.)
        hazard.position=Point(*self.section.to_world((15.,local.position.y+10.)))
        result=self.run_scene([hazard])
        self.assertFalse(result.debug_details['jamming_dynamic']['emergency'])
        self.assertGreater(result.acc_target_velocity,0.)
        self.assertGreater(self.run_scene([]).acc_target_velocity,0.)

    def test_crossing_prediction_can_brake_without_current_obb_overlap(self):
        local=self.planner._ego(self.ego)
        hazard=self.obj(local.position.x-4.,local.position.y+5.,speed=5.,heading=0.)
        result=self.run_scene([hazard])
        details=result.debug_details['jamming_dynamic']
        self.assertTrue(details['emergency'],details)
        self.assertGreater(details['collision']['time'],0.)
        self.assertLessEqual(details['collision']['time'],1.)
        self.assertEqual(result.acc_target_velocity,0.)

    def test_mover_stopping_uses_static_planner_then_restart_uses_dynamic(self):
        lead=self.obj(-1.,25.)
        result=self.run_scene([lead])
        self.assertIn('jamming_dynamic',result.debug_details)
        lead.velocity=0.
        result=self.run_scene([lead])
        self.assertNotIn('jamming_dynamic',result.debug_details)
        self.assertEqual(result.reason,'static_recorded_follow')
        self.assertGreater(result.acc_target_velocity,0.)
        self.assertFalse(lead.is_static)
        controller=AutonomousDriving.__new__(AutonomousDriving)
        self.assertEqual(controller._legacy_acc_entries(self.ego,result,[{'object_info':lead}]),[])
        lead.velocity=2.
        result=self.run_scene([lead])
        self.assertIn('jamming_dynamic',result.debug_details)

    def test_native_static_scene_retains_existing_geometry_and_speed(self):
        data=json.loads((Path(__file__).parent/'jamming_entry_scene_20261006.json').read_text())
        ego=VehicleState(*data['ego'])
        objects=[{'object_info':ObjectInfo(**{k:v for k,v in row.items() if k!='group'})} for row in data['objects']]
        p=self.planner
        expected=p.local.plan(p._ego(ego),p._path(self.reference),p._obstacles(objects),p._path(self.route))
        actual=p.run(ego,self.reference,objects,global_path=self.route)
        self.assertNotIn('jamming_dynamic',actual.debug_details)
        np.testing.assert_allclose([p.section.to_local(v) for v in actual.path],expected.path,atol=1e-8)
        self.assertAlmostEqual(actual.acc_target_velocity,expected.acc_target_velocity)

    def test_full_controller_releases_emergency_without_vehicle_motion(self):
        with contextlib.redirect_stdout(io.StringIO()):
            controller=AutonomousDriving()
            controller.lattice_planning_process.recorded_jamming.close()
            controller.lattice_planning_process.recorded_jamming=self.planner
            self.addCleanup(controller.lattice_planning_process.recorded_static.executor.shutdown,True)
            local=self.planner._ego(self.ego)
            hazard=self.obj(local.position.x,local.position.y+3.,speed=1.)
            controller.execute(self.ego,[hazard],None,control_dt=1./30.)
            result=controller.lattice_planning_process.last_result
            self.assertEqual(result.debug_details['static_avoidance']['final_target_velocity'],0.)
            hazard.position=Point(*self.section.to_world((15.,local.position.y+10.)))
            self.ego.velocity=0.
            for _ in range(10):
                control,_=controller.execute(self.ego,[hazard],None,control_dt=1./30.)
            result=controller.lattice_planning_process.last_result
            self.assertGreater(result.debug_details['static_avoidance']['final_target_velocity'],0.)
            self.assertGreater(control.accel,0.)
            self.assertEqual(control.brake,0.)

    def test_moving_gt_does_not_break_global_route_handoff(self):
        lead=self.obj(-1.,25.)
        for _ in range(12):
            self.run_scene([lead])
            if self.planner.dynamic.active:
                break
        self.assertTrue(self.planner.dynamic.active)
        path=list(self.planner.dynamic.tracker.path)
        # Traffic leaves the corridor after the detour is committed. Follow
        # the held geometry through its return and the static global handoff.
        lead.position=Point(*self.section.to_world((15.,60.)))
        for i in range(1,len(path)-1,60):
            a,b=path[i:i+2]
            self.ego=VehicleState(*self.section.to_world(a),
                math.atan2(b.y-a.y,b.x-a.x)-self.section.rotation,5.)
            index=min(range(len(self.route)),key=lambda j:self.route[j].distance(self.ego.position))
            self.reference=self.route[index:index+80]
            result=self.run_scene([lead])
            if self.planner.local.done:
                break
        self.assertTrue(self.planner.local.done,result.debug_details)
        self.assertFalse(self.planner.dynamic.active)
        self.assertEqual(result.reason,'static_recorded_rejoined')

    def boundary_scene(self):
        # Actual 2026-10-06 stop: front corner 3.3 cm outside the empirical
        # envelope, although center tracking error was only 1.3 cm.
        objects=[ObjectInfo(x,y,0.,2,is_static=True,size_x=.49,size_y=.49,heading_deg=-90.)
                 for x,y in ((16.384878,-496.295044),(15.848386,-495.572906))]
        entries=[dict(object_info=o) for o in objects]
        p=self.planner
        entry=VehicleState(40.17,-479.75,math.radians(-143.1),0.)
        p.local.plan(p._ego(entry),p._path(self.route),p._obstacles(entries),p._path(self.route))
        ego=p._ego(VehicleState(20.80,-497.56,math.radians(-122.59),0.))
        return ego,p._obstacles(entries)

    def test_recorded_boundary_stop_has_validated_inward_recovery(self):
        ego,statics=self.boundary_scene()
        owner=self.planner.local
        self.assertFalse(owner._envelope_clear(np.asarray([ego.position]),np.asarray([ego.yaw])))
        recovered=make_recovery(owner,ego,statics)
        self.assertIsNotNone(recovered)
        tracker,end,initial=recovered
        self.assertAlmostEqual(initial,-.033,delta=.002)
        xy=np.asarray(tracker.path)
        curve_yaw=np.array([tracker.planner._path_heading(tracker.path,i) for i in range(len(xy))])
        room=clearance(owner,xy,curve_yaw)
        self.assertGreaterEqual(float(room.min()),initial-.002)
        self.assertTrue(owner._envelope_clear(xy[xy[:,1]>end],curve_yaw[xy[:,1]>end]))
        result=recovery_result(tracker,ego,statics,initial)
        self.assertGreater(result.acc_target_velocity,0.)
        self.assertLessEqual(result.acc_target_velocity,.5)
        self.assertTrue(result.debug_details['static_avoidance']['path_validated'])
        # Shared static boundary decisions are unchanged even during recovery.
        self.assertFalse(owner._envelope_clear(np.asarray([ego.position]),np.asarray([ego.yaw])))
        obstacle=ObjectInfo(ego.position.x,ego.position.y+1.,0.,2,is_static=True,size_x=1.,size_y=1.)
        self.assertIsNone(recovery_result(tracker,ego,statics+[obstacle],initial))
        ego.position=Point(ego.position.x-.3,ego.position.y)
        self.assertIsNone(make_recovery(owner,ego,statics))
        self.assertIsNone(recovery_result(tracker,ego,statics,initial))

    def test_boundary_recovery_still_yields_to_moving_collision(self):
        ego,statics=self.boundary_scene()
        p=self.planner
        p.dynamic.tracker,p.dynamic.return_y,p.dynamic.recovery_initial=make_recovery(p.local,ego,statics)
        mover=ObjectInfo(ego.position.x,ego.position.y+1.,1.,1,
            size_x=4.5,size_y=1.8,heading_deg=math.degrees(ego.yaw),unique_id=99)
        result=p.dynamic.run(p.local,ego,p.local.path,statics,[mover],
                             p.local._result(p.local.path,True,'follow'),8.)
        self.assertEqual(result.debug_details['jamming_dynamic']['state'],'EMERGENCY')
        self.assertEqual(result.acc_target_velocity,0.)
        result=p.dynamic.run(p.local,ego,p.local.path,statics,[],
                             p.local._result(p.local.path,True,'follow'),8.)
        self.assertEqual(result.debug_details['jamming_dynamic']['state'],'RECOVER')
        self.assertGreater(result.acc_target_velocity,0.)

    def test_dynamic_tracking_failure_reconnects_instead_of_latching_zero(self):
        lead=self.obj(-1.,25.)
        for _ in range(12):
            self.run_scene([lead])
            if self.planner.dynamic.active:
                break
        self.assertTrue(self.planner.dynamic.active)
        previous=self.planner.dynamic.tracker
        local=self.planner._ego(self.ego)
        local.position=Point(local.position.x-1.2,local.position.y)
        local.velocity=0.
        self.ego=VehicleState(*self.section.to_world(local.position),
            local.yaw-self.section.rotation,0.)
        failed=previous.plan(local,self.planner._path(self.reference),[])
        self.assertEqual(failed.debug_details['static_avoidance']['error'],'tracking_error')
        result=self.run_scene([])
        self.assertIsNot(previous,self.planner.dynamic.tracker)
        self.assertTrue(result.debug_details['static_avoidance']['path_validated'])
        self.assertGreater(result.acc_target_velocity,0.)
        self.assertIsNone(result.debug_details['jamming_dynamic']['tracking_error'])

    def test_off_corridor_npc_stop_does_not_discard_the_committed_path(self):
        lead=self.obj(-1.,25.)
        remote=self.obj(70.,25.,uid=100)
        for _ in range(12):
            self.run_scene([lead,remote])
            if self.planner.dynamic.active:
                break
        owner_path=self.planner.local.path
        held=self.planner.dynamic.tracker
        remote.velocity=0.
        result=self.run_scene([lead,remote])
        self.assertIs(self.planner.local.path,owner_path)
        self.assertIs(self.planner.dynamic.tracker,held)
        self.assertGreater(result.acc_target_velocity,0.)

    def test_replanning_finds_an_alternate_route_from_fresh_geometry(self):
        self.run_scene([self.obj(15.,30.)])
        owner=self.planner.local
        original_path=owner.path
        ego=self.planner._ego(self.ego)
        blocked=self.planner._obstacles([dict(object_info=self.obj(-1.,25.,speed=0.))])
        tracker=replan_tracker(owner,ego,blocked)
        self.assertIsNotNone(tracker)
        result=tracker.plan(ego,self.planner._path(self.reference),blocked)
        self.assertTrue(result.debug_details['static_avoidance']['path_validated'])
        self.assertGreater(result.acc_target_velocity,0.)
        self.assertIs(owner.path,original_path)


if __name__=='__main__':
    unittest.main()
