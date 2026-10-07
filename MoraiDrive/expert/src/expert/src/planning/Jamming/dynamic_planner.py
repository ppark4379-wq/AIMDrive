"""Moving-object overlay on the unchanged recorded static corridor.

Reuse highway SAT OBBs, acceleration/delay prediction and following policy.
Every speed decision uses the latest GT; only a validated lateral path is held.
All geometry here is in the section's forward-Y frame.
"""
import copy
import math
import time
from concurrent.futures import Future

import numpy as np
from scipy.interpolate import CubicSpline

from ...localization.point import Point
from ..highway_lane_planning.planner import HighwayLanePlanner
from ..highway_lane_planning.lane_change_safety import LaneChangeSafety
from ..highway_lane_planning import parameters as hp
from ..정적장애물.recorded_planner import connector
from .motion import key
from .dynamic_recovery import make_recovery, recovery_result, reconnect_tracker, replan_tracker

HORIZON = 6.
MAX_MANEUVER_HORIZON = 20.
SEARCH_BUDGET = .08
TRIGGER_DISTANCE = 50.
CRUISE_LIMIT = 30./3.6
OVERTAKE_LEAD_LIMIT = 15./3.6
RETURN_GAP = 5.
PREDICT_ACCEL = 1.2


class DynamicObstaclePlanner:
    def __init__(self):
        self.reset()

    def reset(self):
        pending=getattr(self,'repair_pending',None)
        if pending is not None:
            pending.cancel()
        self.repair_pending=None
        self.last_repair_attempt=-float('inf')
        self.tracker = None
        self.return_y = None
        self.search_index = 0
        self.last_state = None
        self.recovery_initial = None
        self.passing_id = None
        self.return_start_y = None
        self.maneuver_speed = None
        self.dynamic_seen = False
        self.last_return_replan = -float('inf')

    def _repair(self, owner, ego, reference, statics, result):
        """A failed held path must be retried, never retained as a zero latch."""
        now=time.monotonic()
        if self.repair_pending is not None:
            if not self.repair_pending.done():
                return result
            try:
                tracker=self.repair_pending.result()
            except Exception:
                tracker=None
            self.repair_pending=None
            if tracker is not None:
                tested=tracker.plan(ego,reference,statics)
                if tested.debug_details.get('static_avoidance',{}).get('path_validated'):
                    self.tracker=tracker
                    self.return_y=tracker.rejoin_y
                    return tested
        if now-self.last_repair_attempt<.5:
            return result
        self.last_repair_attempt=now
        if not owner._envelope_clear(np.asarray([ego.position]),np.asarray([ego.yaw])):
            return result
        for template in (self.tracker,owner):
            tracker=reconnect_tracker(template,ego,statics)
            if tracker is None:
                continue
            tested=tracker.plan(ego,reference,statics)
            if tested.debug_details.get('static_avoidance',{}).get('path_validated'):
                self.tracker=tracker
                if template is owner:
                    self.return_y=tracker.recovery_until_y
                return tested
        # Full alternative-corridor search runs on the existing worker in
        # production. Its output is validated against fresh GT before adoption.
        snapshot=copy.deepcopy(ego)
        obstacles=copy.deepcopy(statics)
        template=copy.copy(owner)
        if owner.executor is not None:
            self.repair_pending=owner.executor.submit(replan_tracker,template,snapshot,obstacles)
        else:
            self.repair_pending=Future()
            self.repair_pending.set_result(replan_tracker(template,snapshot,obstacles))
        return result

    @property
    def active(self):
        return self.tracker is not None

    @staticmethod
    def _project(path, obj):
        xy = np.asarray(path)
        delta = np.diff(xy, axis=0)
        lengths = np.linalg.norm(delta, axis=1)
        ratio = np.clip(np.sum((np.asarray(obj.position)-xy[:-1])*delta, axis=1)
                        / np.maximum(lengths**2, 1e-9), 0., 1.)
        projected = xy[:-1]+ratio[:, None]*delta
        i = int(np.argmin(np.linalg.norm(projected-np.asarray(obj.position), axis=1)))
        yaw = math.atan2(delta[i, 1], delta[i, 0])
        station = float(np.sum(lengths[:i])+ratio[i]*lengths[i])
        lateral = float(np.linalg.norm(projected[i]-np.asarray(obj.position)))
        return station, lateral, yaw

    def front(self, path, ego, movers):
        if len(path) < 2:
            return None
        ego_s = self._project(path, ego)[0]
        nearest = None
        for obj in movers:
            station, lateral, tangent = self._project(path, obj)
            angle = math.radians(obj.heading_deg)-tangent
            half_width = .5*(max(1.,obj.size_x)*abs(math.sin(angle))
                              +max(1.,obj.size_y)*abs(math.cos(angle)))
            half_length = .5*(max(1.,obj.size_x)*abs(math.cos(angle))
                               +max(1.,obj.size_y)*abs(math.sin(angle)))
            if station <= ego_s or lateral > .5*hp.VEHICLE_WIDTH+half_width+hp.COLLISION_MARGIN:
                continue
            gap = max(0., station-ego_s-hp.VEHICLE_LENGTH-half_length)
            speed = float(obj.velocity)*math.cos(angle)
            if nearest is None or gap < nearest['gap']:
                closing = max(0., ego.velocity-speed)
                nearest = dict(object=obj, gap=gap, speed=speed,
                               ttc=gap/closing if closing>.1 else float('inf'))
        return nearest

    @staticmethod
    def conflict(path, ego, movers, target_speed, horizon=HORIZON, acceleration=None):
        """Check matched-time OBBs, including crossing and target-side traffic."""
        if not movers or len(path)<2:
            return None
        xy = np.asarray(path)
        cumulative = np.r_[0.,np.cumsum(np.linalg.norm(np.diff(xy,axis=0),axis=1))]
        last_yaw, last_travel = ego.yaw, 0.
        for t in np.arange(0.,horizon+hp.PREDICTION_TIME_STEP*.5,hp.PREDICTION_TIME_STEP):
            travel = LaneChangeSafety.travel(max(0.,ego.velocity),max(0.,target_speed),t,acceleration)
            index = min(len(path)-2,max(0,int(np.searchsorted(cumulative,travel,side='right'))-1))
            ratio = float(np.clip((travel-cumulative[index])/max(1e-9,cumulative[index+1]-cumulative[index]),0.,1.))
            position = xy[index]+ratio*(xy[index+1]-xy[index])
            tangent = math.atan2(*(xy[index+1]-xy[index])[::-1])
            yaw_delta = math.atan2(math.sin(tangent-last_yaw),math.cos(tangent-last_yaw))
            yaw = last_yaw+float(np.clip(yaw_delta,
                -(travel-last_travel)*hp.PREDICTION_MAX_YAW_PER_METER,
                (travel-last_travel)*hp.PREDICTION_MAX_YAW_PER_METER))
            point = ego.position if travel<=1e-9 else Point(*position)
            last_yaw,last_travel=yaw,travel
            for obj in movers:
                angle = math.radians(obj.heading_deg)
                center = Point(obj.position.x+obj.velocity*t*math.cos(angle),
                               obj.position.y+obj.velocity*t*math.sin(angle))
                if point.distance(center) > hp.VEHICLE_LENGTH+math.hypot(obj.size_x,obj.size_y)+2.:
                    continue
                if HighwayLanePlanner._obb_collision(point,yaw,center,angle,obj):
                    return dict(time=float(t), object_id=key(obj), kind='moving_obb')
        return None

    @staticmethod
    def _travel_time(distance, speed, target):
        if LaneChangeSafety.travel(max(0.,speed),target,MAX_MANEUVER_HORIZON,PREDICT_ACCEL)<distance:
            return None
        low,high=0.,MAX_MANEUVER_HORIZON
        for _ in range(25):
            middle=.5*(low+high)
            if LaneChangeSafety.travel(max(0.,speed),target,middle,PREDICT_ACCEL)<distance:
                low=middle
            else:
                high=middle
        return high

    def _candidate(self, owner, ego, baseline, statics, lead, offset, transition, cruise):
        xy=np.asarray(baseline)
        if len(xy)<3 or np.any(np.diff(xy[:,1])<=0.):
            return None
        curve=CubicSpline(xy[:,1],xy[:,0])
        start=float(ego.position.y)
        join=start+transition
        # The return is timed against the speed this curved maneuver can
        # actually command, including acceleration from a slow following state.
        # The old cruise-only estimate returned beside the moving lead and
        # consequently rejected otherwise usable passing corridors.
        lead_speed=max(0.,lead['speed'])
        plateau=max(join+6.,lead['object'].position.y+.5*max(1.,lead['object'].size_x)+RETURN_GAP)
        for _ in range(12):
            candidate=self._candidate_geometry(owner,ego,xy,curve,statics,offset,transition,plateau,cruise)
            if candidate is None:
                return None
            tracker,finish,target,positions,station=candidate
            if target<=lead_speed+.2:
                return None
            return_distance=float(np.interp(plateau,positions[:,1],station))
            passing_time=self._travel_time(return_distance,ego.velocity,target)
            if passing_time is None:
                return None
            needed=lead['object'].position.y+lead_speed*passing_time+.5*max(1.,lead['object'].size_x)+RETURN_GAP
            if plateau>=needed-.05:
                horizon=self._travel_time(float(np.interp(finish,positions[:,1],station)),ego.velocity,target)
                if horizon is None:
                    return None
                tracker.dynamic_return_start_y=plateau
                return tracker,finish,target,max(HORIZON,horizon+hp.LANE_CHANGE_SETTLE_TIME)
            plateau=max(plateau+.5,needed+.1)
        return None

    def _candidate_geometry(self, owner, ego, xy, curve, statics, offset, transition, plateau, cruise):
        start=float(ego.position.y)
        join=start+transition
        finish=plateau+transition
        if finish>=xy[-1,1]-1.:
            return None
        state=lambda y, shift=0.: (y,float(curve(y))+shift,float(curve(y,1)),float(curve(y,2)))
        entry=connector((start,ego.position.x,math.cos(ego.yaw)/max(.5,math.sin(ego.yaw)),0.),state(join,offset))
        middle=lambda y, derivative=0: curve(y,derivative)+(offset if derivative==0 else 0.)
        exit_curve=connector(state(plateau,offset),state(finish))
        pieces=[]
        for c,a,b in ((entry,start,join),(middle,join,plateau),
                      (exit_curve,plateau,finish),(curve,finish,xy[-1,1])):
            # A nominally valid corner at the edge can leave the corridor
            # with only one degree of tracking lag. Reserve 25 cm for the
            # additional dynamic maneuver; keep the static tail unchanged.
            edge=owner._edge(c,a,b,statics,body_margin=.25 if a<finish else 0.)
            if edge is None:
                return None
            pieces.append(edge if not pieces else edge[1:])
        points=[Point(*p) for p in np.vstack(pieces)]
        tracker=copy.copy(owner)
        tracker.executor=tracker.pending=None
        tracker.path,tracker.progress=points,0
        tracker.done=False
        # The wrapper hands back at this maneuver's return point. The shared
        # static planner's earlier global handoff must not discard this path.
        tracker.handoff_y=None
        tracker.recovery_until_y=None
        tracker._speed_profile_path=tracker._cached_speed_profile=None
        tracker.selected_names=['dynamic_offset_{:+.1f}m'.format(offset)]
        positions=np.asarray(points)
        station=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(positions,axis=0),axis=1))]
        _,_,profiles=tracker._speed_profile(points)
        maneuver=positions[1:-1,1]<=finish
        target=min(cruise,*(float(np.min(values[maneuver])) for values in profiles.values()))
        return tracker,finish,target,positions,station

    def _search(self, owner, ego, baseline, statics, movers, lead, cruise):
        offsets=(2.5,-2.5,3.5,-3.5,1.5,-1.5,4.5,-4.5)
        transitions=(max(6.,min(18.,max(ego.velocity,cruise)*1.8)),12.,8.,18.)
        specs=[(offset,distance) for distance in dict.fromkeys(transitions) for offset in offsets]
        begin=time.monotonic()
        for _ in range(len(specs)):
            offset,distance=specs[self.search_index%len(specs)]
            self.search_index=(self.search_index+1)%len(specs)
            candidate=self._candidate(owner,ego,baseline,statics,lead,offset,distance,cruise)
            if candidate is not None:
                tracker,finish,target,horizon=candidate
                if all(self.conflict(tracker.path,ego,movers,target,horizon,accel) is None
                       for accel in (PREDICT_ACCEL,hp.LANE_CHANGE_PREDICT_ACCEL,hp.LANE_CHANGE_PREDICT_ACCEL_MAX)):
                    self.tracker,self.return_y=tracker,finish
                    self.passing_id=key(lead['object'])
                    self.return_start_y=tracker.dynamic_return_start_y
                    self.maneuver_speed=target
                    return True
            if time.monotonic()-begin>=SEARCH_BUDGET:
                break
        return False

    def _handoff_clear(self, owner, ego, movers, path, speed):
        """Hand back only after a physically aligned, separated return.

        A lead that accelerates may finish ahead instead; a safe return behind
        it is preferable to requiring an overtake forever. In either ordering
        there must be a real body-to-body gap, not merely a reached Y value.
        """
        if self.passing_id is None:
            return True
        _,lateral,tangent=self._project(owner.path,ego)
        heading=abs(math.atan2(math.sin(ego.yaw-tangent),math.cos(ego.yaw-tangent)))
        if lateral>.35 or heading>math.radians(10.):
            return False
        lead=next((o for o in movers if key(o)==self.passing_id),None)
        if lead is not None:
            ego_s=self._project(owner.path,ego)[0]
            lead_s=self._project(owner.path,lead)[0]
            angle=math.radians(lead.heading_deg)-tangent
            half=.5*(max(1.,lead.size_x)*abs(math.cos(angle))+max(1.,lead.size_y)*abs(math.sin(angle)))
            passed_gap=ego_s-lead_s-half
            following_gap=lead_s-half-ego_s-hp.VEHICLE_LENGTH
            if max(passed_gap,following_gap)<RETURN_GAP:
                return False
        return self.conflict(path,ego,movers,speed) is None

    def _defer_return(self, owner, ego, statics, movers, cruise):
        """Keep the passing corridor when a changed actor blocks the old exit.

        Braking on a curve that already points into an accelerating rear car
        does not resolve that conflict. Reconnect from the measured pose to a
        parallel continuation first, then validate a later return end to end.
        """
        if time.monotonic()-self.last_return_replan<.25 or not owner.path:
            return False
        xy=np.asarray(owner.path)
        if len(xy)<3 or np.any(np.diff(xy[:,1])<=0.):
            return False
        curve=CubicSpline(xy[:,1],xy[:,0])
        offset=float(ego.position.x-curve(ego.position.y))
        if abs(offset)<.8:
            return False
        self.last_return_replan=time.monotonic()
        for transition in (6.,8.,12.):
            plateau=ego.position.y+transition+8.
            candidate=self._candidate_geometry(owner,ego,xy,curve,statics,offset,transition,plateau,cruise)
            if candidate is None:
                continue
            tracker,finish,target,positions,station=candidate
            horizon=self._travel_time(float(np.interp(finish,positions[:,1],station)),ego.velocity,target)
            if horizon is None:
                continue
            horizon=max(HORIZON,horizon+hp.LANE_CHANGE_SETTLE_TIME)
            if all(self.conflict(tracker.path,ego,movers,target,horizon,accel) is None
                   for accel in (PREDICT_ACCEL,hp.LANE_CHANGE_PREDICT_ACCEL_MAX)):
                self.tracker,self.return_y=tracker,finish
                self.return_start_y=plateau
                self.maneuver_speed=target
                return True
        return False

    def run(self, owner, ego, reference, statics, movers, base, desired_speed):
        self.dynamic_seen=self.dynamic_seen or bool(movers)
        if not movers and not self.active and not self.dynamic_seen:
            return base
        if base.reason == 'static_recorded_rejoined':
            self.reset()
            return base
        result=base
        base_failure=base.debug_details.get('static_avoidance',{}).get('error')
        if (not self.active and owner.active and self.dynamic_seen
                and base_failure in ('tracking_error','remaining_path_collision',
                                     'alignment_recovery_unavailable','recorded_envelope')):
            # A moving-traffic episode can disturb the base path before an
            # overtake is adopted. Give that path the same fresh-pose repair
            # policy as a held detour, even if the actor has since left GT.
            self.tracker=copy.copy(owner)
            self.tracker.executor=self.tracker.pending=None
            self.return_y=owner.rejoin_y
        if self.active:
            result=self.tracker.plan(ego,reference,statics)
            failure=result.debug_details.get('static_avoidance',{}).get('error')
            if failure=='recorded_envelope' and self.recovery_initial is None:
                recovered=make_recovery(owner,ego,statics)
                if recovered is not None:
                    self.tracker,self.return_y,self.recovery_initial=recovered
            if self.recovery_initial is not None:
                recovered=recovery_result(self.tracker,ego,statics,self.recovery_initial)
                if recovered is not None:
                    result=recovered
                elif self.tracker.selected_names==['dynamic_inward_recovery']:
                    result=self.tracker._result(result.path,False,'blocked','boundary_recovery_invalid')
            elif failure in ('tracking_error','remaining_path_collision','alignment_recovery_unavailable'):
                result=self._repair(owner,ego,reference,statics,result)
            if (ego.position.y>=self.return_y
                    and result.debug_details['static_avoidance']['path_validated']
                    and self._handoff_clear(owner,ego,movers,result.path,
                        min(CRUISE_LIMIT,max(0.,desired_speed),result.acc_target_velocity))):
                self.reset()
                # Keep this last continuous path for this tick; static tracking
                # resumes on its original path next tick.
        cruise=min(CRUISE_LIMIT,max(0.,desired_speed),result.acc_target_velocity)
        if self.active and self.maneuver_speed is not None:
            cruise=min(cruise,self.maneuver_speed)
        if (self.active and self.passing_id is not None and self.recovery_initial is None
                and ego.position.y<self.return_y
                and self.conflict(result.path,ego,movers,cruise) is not None
                and self._defer_return(owner,ego,statics,movers,cruise)):
            result=self.tracker.plan(ego,reference,statics)
            cruise=min(CRUISE_LIMIT,max(0.,desired_speed),result.acc_target_velocity,self.maneuver_speed)
        lead=self.front(result.path,ego,movers)
        if (not self.active and owner.active and lead is not None
                and lead['gap']<TRIGGER_DISTANCE and lead['speed']<cruise-.2
                and (lead['object'].type not in (1,2) or lead['speed']<=OVERTAKE_LEAD_LIMIT)
                and result.debug_details.get('static_avoidance',{}).get('path_validated')):
            if self._search(owner,ego,owner.path,statics,movers,lead,cruise):
                result=self.tracker.plan(ego,reference,statics)
                cruise=min(CRUISE_LIMIT,max(0.,desired_speed),result.acc_target_velocity)
                if self.maneuver_speed is not None:
                    cruise=min(cruise,self.maneuver_speed)
                lead=self.front(result.path,ego,movers)
        # A validated detour owns the commanded path. Recompute its actual
        # lead, including vehicles on the target side, on every frame.
        limit=cruise
        state='AVOID' if self.active else 'CLEAR'
        if self.recovery_initial is not None:
            state='RECOVER'
        if lead is not None:
            limit=min(limit,HighwayLanePlanner._acc_speed_limit(
                max(0.,ego.velocity),lead['gap'],lead['ttc'],max(0.,lead['speed'])))
            state='FOLLOW' if not self.active else 'AVOID_FOLLOW'
        collision=self.conflict(result.path,ego,movers,limit)
        if collision is not None:
            if collision['time']<=hp.MERGE_EMERGENCY_COLLISION_TIME:
                limit,state=0.,'EMERGENCY'
            else:
                for speed in (limit*.75,limit*.5,limit*.25,0.):
                    if self.conflict(result.path,ego,movers,speed) is None:
                        limit=speed
                        break
                else:
                    limit=0.
                state='YIELD'
        # A cleared hazard never inherits a previous zero speed ceiling.
        tracking_error=result.debug_details.get('static_avoidance',{}).get('error')
        if tracking_error and state!='EMERGENCY':
            state='BLOCKED'
        output=copy.copy(result)
        output.debug_details=copy.deepcopy(result.debug_details)
        output.acc_target_velocity=min(result.acc_target_velocity,limit)
        output.acc_override=True
        output.obstacle_detected=bool(movers)
        output.reason='jamming_dynamic_'+state.lower()
        output.debug_details['jamming_dynamic']=dict(
            state=state, speed_authority=True, moving_count=len(movers),
            lead_id=key(lead['object']) if lead else None,
            lead_gap_m=lead['gap'] if lead else None,
            collision=collision, emergency=state=='EMERGENCY', speed_limit_mps=limit)
        output.debug_details['jamming_dynamic']['tracking_error']=tracking_error
        output.debug_details['jamming_dynamic']['replanning']=self.repair_pending is not None
        output.debug_details['jamming_dynamic']['overtake_lead_limit_kph']=OVERTAKE_LEAD_LIMIT*3.6
        output.debug_details['jamming_dynamic']['passing_id']=self.passing_id
        output.debug_details['jamming_dynamic']['return_start_y']=self.return_start_y
        output.debug_details['jamming_dynamic']['return_y']=self.return_y
        output.debug_details['jamming_dynamic']['maneuver_speed_mps']=self.maneuver_speed
        output.debug_details.setdefault('static_avoidance',{})['dynamic_speed_limit_mps']=limit
        output.predicted_object_paths=[[
            Point(o.position.x+o.velocity*t*math.cos(math.radians(o.heading_deg)),
                  o.position.y+o.velocity*t*math.sin(math.radians(o.heading_deg)))
            for t in (0.,HORIZON)] for o in movers]
        self.last_state=state
        return output
