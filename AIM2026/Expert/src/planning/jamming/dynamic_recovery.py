"""Bounded inward recovery for a moving-object detour, never a wider corridor."""
import copy
import math
import numpy as np
from scipy.interpolate import CubicSpline
from ...localization.point import Point
from ..static_obstacle.recorded_planner import connector, MAX_CURVATURE

from ...config.parameters.jamming import (
    RECOVERY_SPEED,
    MAX_OVERRUN,
)


def reconnect_tracker(template, ego, statics):
    """Keep the static implementation intact; repair only a dynamic copy."""
    tracker=copy.copy(template)
    tracker.executor=tracker.pending=None
    tracker.rejections=dict(template.rejections)
    tracker._center_bounds_cache={}
    tracker.known=list(statics)
    tracker.handoff_y=None
    if not tracker._reconnect_measured_pose(ego,statics):
        return None
    tracker.selected_names=['dynamic_reconnect']
    return tracker


def replan_tracker(template, ego, statics):
    """Search again from the measured pose using the existing corridor graph."""
    tracker=copy.copy(template)
    tracker.executor=tracker.pending=None
    tracker.rejections={k:0 for k in template.rejections}
    tracker._center_bounds_cache={}
    tracker.known=list(statics)
    tracker.path=[]
    tracker.progress=0
    tracker.recovery_until_y=None
    tracker.search_details={}
    tracker.search_progress=0
    tracker.alignment_slow=False
    complete=tracker._search(ego,statics)
    if complete is None:
        return None
    complete=tracker._smooth_return(tracker._smooth_recorded_path(complete,statics),statics)
    tracker.path=[Point(*p) for p in complete]
    tracker.handoff_y=None
    tracker._speed_profile_path=tracker._cached_speed_profile=None
    return tracker


def clearance(owner, xy, yaw):
    """Signed clearance to the same center and four-corner bounds as static."""
    xy,yaw=np.asarray(xy),np.asarray(yaw)
    centers=np.array([c(xy[:,1]) for c in owner.curves+[owner._global_return_corridor]])
    result=np.minimum(xy[:,0]-centers.min(axis=0)+.15,
                      centers.max(axis=0)+.15-xy[:,0])
    for along in (0.,owner.planner.vehicle_length):
        for side in (-owner.planner.vehicle_half_width,owner.planner.vehicle_half_width):
            x=xy[:,0]+along*np.cos(yaw)-side*np.sin(yaw)
            y=xy[:,1]+along*np.sin(yaw)+side*np.cos(yaw)
            result=np.minimum(result,np.minimum(
                x-np.interp(y,owner.envelope_y,owner.envelope_low)+.15,
                np.interp(y,owner.envelope_y,owner.envelope_high)+.15-x))
    return result


def make_recovery(owner, ego, statics):
    """Rejoin the static route only if the measured overrun never increases.

    This is limited to a stopped/creeping car, <=10 cm empirical-boundary
    overrun, full static OBB clearance, and re-entry within the first metre.
    The original corridor bounds and obstacle margins are never modified.
    """
    initial=float(clearance(owner,[ego.position],[ego.yaw])[0])
    if not (-MAX_OVERRUN<=initial<0.) or abs(ego.velocity)>.6 or math.sin(ego.yaw)<.5:
        return None
    xy=np.asarray(owner.path)
    if len(xy)<3 or np.any(np.diff(xy[:,1])<=0.):
        return None
    curve=CubicSpline(xy[:,1],xy[:,0])
    start=ego.position.y
    slope=math.cos(ego.yaw)/math.sin(ego.yaw)
    for distance in (6.,8.,10.,12.,16.):
        index=int(np.searchsorted(xy[:,1],start+distance))
        if index>=len(xy)-2:
            continue
        end=xy[index,1]
        direction=np.sign(float(curve(end))-ego.position.x)
        for curvature in (.12,.16,.20,MAX_CURVATURE*.95):
            c=connector((start,ego.position.x,slope,direction*curvature*(1+slope*slope)**1.5),
                        owner._state(curve,end))
            ys=np.linspace(start,end,int(math.ceil((end-start)/.025))+1)
            dx=c(ys,1)
            if np.max(np.abs(c(ys,2))/(1+dx*dx)**1.5)>MAX_CURVATURE:
                continue
            edge=np.column_stack((c(ys),ys))
            yaw=np.arctan2(1.,dx)
            room=clearance(owner,edge,yaw)
            station=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(edge,axis=0),axis=1))]
            # Until the whole body is inside, every sample must move inward.
            outside=room<0.
            if (np.any(room<initial-1e-5) or np.any(outside & (station>1.))
                    or np.any(np.diff(np.minimum(room,0.))<-.0001)):
                continue
            points=[Point(*p) for p in np.vstack((edge,xy[index+1:]))]
            headings=np.array([owner.planner._path_heading(points,i) for i in range(len(points))])
            road_clear = getattr(owner, 'road_clear', None)
            if (road_clear is not None and (not road_clear([ego.position], [ego.yaw])
                    or not road_clear(points, headings))):
                continue
            if (not owner._envelope_clear(np.asarray(points[len(edge):]),headings[len(edge):])
                    or not owner.collision_free(np.asarray(points),headings,statics,.12)
                    or owner.planner._max_curvature(points)>MAX_CURVATURE):
                continue
            tracker=copy.copy(owner)
            tracker.path,tracker.progress=points,0
            tracker.executor=tracker.pending=None
            tracker.handoff_y=None
            tracker.recovery_until_y=None
            tracker._speed_profile_path=tracker._cached_speed_profile=None
            tracker.selected_names=['dynamic_inward_recovery']
            return tracker,float(end),initial
    return None


def recovery_result(tracker, ego, statics, initial):
    """Revalidate the bounded prefix from the actual pose on every frame."""
    room=float(clearance(tracker,[ego.position],[ego.yaw])[0])
    # Five millimetres accommodates steering response/GT jitter at launch;
    # it is a fixed allowance, never accumulated from one frame to the next.
    if room<max(-MAX_OVERRUN,initial-.005) or abs(ego.velocity)>.8:
        return None
    xy=np.asarray(tracker.path)
    index=int(np.argmin(np.linalg.norm(xy-np.asarray(ego.position),axis=1)))
    if ego.position.distance(tracker.path[index])>.15:
        return None
    travelled=float(np.linalg.norm(np.diff(xy[:index+1],axis=0),axis=1).sum())
    if travelled>1. and room<-.005:
        return None
    points=tracker.path[max(0,index-1):]
    headings=np.array([tracker.planner._path_heading(points,i) for i in range(len(points))])
    road_clear = getattr(tracker, 'road_clear', None)
    if (road_clear is not None and (not road_clear([ego.position], [ego.yaw])
            or not road_clear(points, headings))):
        return None
    if (not tracker.collision_free(np.asarray([ego.position]),np.asarray([ego.yaw]),statics,.12)
            or not tracker.collision_free(np.asarray(points),headings,statics,.12)):
        return None
    result=tracker._result(points,True,'dynamic_recover_envelope')
    result.acc_target_velocity=min(result.acc_target_velocity,RECOVERY_SPEED)
    result.debug_details['static_avoidance']['boundary_recovery']=dict(
        initial_overrun_m=-initial,current_overrun_m=max(0.,-room))
    return result
