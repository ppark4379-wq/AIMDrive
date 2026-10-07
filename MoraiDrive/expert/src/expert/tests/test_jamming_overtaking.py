"""Slow following must become a complete, separated tunnel overtake."""
import math
import unittest

import numpy as np

from . import test_jamming_dynamic as fixtures
from expert.src.localization.point import Point
from expert.src.vehicle_state import VehicleState
from expert.src.planning.Jamming.dynamic_planner import DynamicObstaclePlanner, RETURN_GAP
from expert.src.planning.highway_lane_planning.planner import HighwayLanePlanner


class SlowOvertakingTests(unittest.TestCase):
    setUpClass=classmethod(fixtures.JammingDynamicTests.setUpClass.__func__)
    setUp=fixtures.JammingDynamicTests.setUp
    obj=fixtures.JammingDynamicTests.obj
    run_scene=fixtures.JammingDynamicTests.run_scene

    def start_slow_scene(self):
        self.ego.velocity=1.2
        local=self.planner._ego(self.ego)
        # Approximate the user's 10.7 m center-to-center following gap.
        lead=self.obj(local.position.x,local.position.y+10.7,speed=1.2)
        for _ in range(40):
            result=self.run_scene([lead])
            if self.planner.dynamic.active:
                return lead,result
        self.fail('No overtake from slow following: '+str(result.debug_details))

    def test_slow_lead_is_passed_and_rejoined_with_actual_gap(self):
        self._complete_scene()

    def test_lead_accelerating_before_return_does_not_latch_stop(self):
        self._complete_scene(accelerate=True)

    def _complete_scene(self, accelerate=False):
        lead,result=self.start_slow_scene()
        dynamic=self.planner.dynamic
        self.assertGreater(dynamic.maneuver_speed,lead.velocity+.2)
        return_y=dynamic.return_y
        dt=.2
        passed=False
        stopped=0.
        accelerated=False
        for tick in range(180):
            local=self.planner._ego(self.ego)
            mover=self.planner._moving_objects([dict(object_info=lead)],self.ego)[0]
            self.assertFalse(HighwayLanePlanner._obb_collision(local.position,local.yaw,
                mover.position,math.radians(mover.heading_deg),mover),
                dict(tick=tick,ego=list(local.position),ego_speed=local.velocity,
                     mover=list(mover.position),mover_speed=mover.velocity,
                     dynamic=result.debug_details.get('jamming_dynamic'),return_y=return_y))
            if not dynamic.active:
                self.assertGreaterEqual(local.position.y,return_y)
                front_gap=local.position.y-mover.position.y-.5*mover.size_x
                rear_gap=mover.position.y-.5*mover.size_x-local.position.y-4.635
                self.assertGreaterEqual(max(front_gap,rear_gap) if accelerate else front_gap,RETURN_GAP-.2)
                self.assertGreater(result.acc_target_velocity,0.)
                passed=True
                break
            target=result.acc_target_velocity
            speed=self.ego.velocity+float(np.clip(target-self.ego.velocity,-3.*dt,1.2*dt))
            xy=np.asarray(self.planner._path(result.path))
            station=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(xy,axis=0),axis=1))]
            progress=dynamic._project(self.planner._path(result.path),local)[0]
            distance=progress+.5*(self.ego.velocity+speed)*dt
            next_xy=np.array([np.interp(distance,station,xy[:,i]) for i in range(2)])
            index=min(len(xy)-2,max(0,int(np.searchsorted(station,distance))-1))
            tangent=math.atan2(*(xy[index+1]-xy[index])[::-1])
            self.ego=VehicleState(*self.section.to_world(next_xy),tangent-self.section.rotation,speed)
            if accelerate and not accelerated and local.position.y>=dynamic.return_start_y-5.:
                lead.velocity=8.
                accelerated=True
            lead.position=Point(*self.section.to_world((mover.position.x,mover.position.y+dt*lead.velocity)))
            route_i=min(range(len(self.route)),key=lambda j:self.route[j].distance(self.ego.position))
            self.reference=self.route[route_i:route_i+80]
            result=self.run_scene([lead])
            stopped=stopped+dt if result.acc_target_velocity<.05 and self.ego.velocity<.05 else 0.
            self.assertLess(stopped,10.,result.debug_details)
        self.assertTrue(passed,result.debug_details)
        if accelerate:
            self.assertTrue(accelerated)

    def test_return_handoff_rejects_accelerated_lead_alongside(self):
        lead,_=self.start_slow_scene()
        dynamic=self.planner.dynamic
        owner=self.planner.local
        path=dynamic.tracker.path
        index=min(range(len(path)),key=lambda i:abs(path[i].y-dynamic.return_y))
        a,b=path[index:index+2]
        ego=VehicleState(*a,math.atan2(b.y-a.y,b.x-a.x),dynamic.maneuver_speed)
        mover=self.planner._moving_objects([dict(object_info=lead)],self.ego)[0]
        mover.position=Point(a.x,a.y+2.)
        mover.velocity=8.
        tail=path[index:]
        self.assertFalse(dynamic._handoff_clear(owner,ego,[mover],tail,ego.velocity))
        self.assertIsNotNone(dynamic.conflict(tail,ego,[mover],ego.velocity))
        mover.position=Point(a.x,a.y+30.)
        self.assertTrue(dynamic._handoff_clear(owner,ego,[mover],tail,ego.velocity))

    def test_base_tracking_error_repairs_after_moving_gt_disappears(self):
        self.run_scene([self.obj(15.,30.)])
        self.assertFalse(self.planner.dynamic.active)
        local=self.planner._ego(self.ego)
        local.position=Point(local.position.x-1.2,local.position.y)
        local.velocity=0.
        self.ego=VehicleState(*self.section.to_world(local.position),local.yaw-self.section.rotation,0.)
        failed=self.planner.local.plan(local,self.planner._path(self.reference),[])
        self.assertEqual(failed.debug_details['static_avoidance']['error'],'tracking_error')
        result=self.run_scene([])
        self.assertTrue(result.debug_details['static_avoidance']['path_validated'],result.debug_details)
        self.assertGreater(result.acc_target_velocity,0.)


if __name__=='__main__':
    unittest.main()
