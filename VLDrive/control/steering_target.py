"""Choose an unpassed target on the model polyline in current ego coordinates."""
import math

import numpy as np


def select_target(points, speed, cfg):
    points = np.asarray(points, dtype=np.float64)
    if cfg['mode'] == 'first_two_midpoint':
        return dict(aim_left_forward=((points[0]+points[1])/2).tolist(),
                    target_selection='first_two_midpoint',target_waypoint_indices=[0,1])
    radius = float(np.clip(max(0.,speed)*cfg['time_s'],cfg['min_distance_m'],cfg['max_distance_m']))
    lengths = np.linalg.norm(np.diff(points,axis=0),axis=1)
    cumulative = np.r_[0.,np.cumsum(lengths)]
    projections = []
    for i,length in enumerate(lengths):
        if length <= 1e-8:
            continue
        delta = points[i+1]-points[i]
        t = float(np.clip(-np.dot(points[i],delta)/(length*length),0.,1.))
        projected = points[i]+t*delta
        projections.append((float(np.dot(projected,projected)),i,t))
    if not projections:
        raise ValueError('insufficient_remaining_path')
    _,start,progress = min(projections)  # Earliest segment wins exact ties.
    progress_m = float(cumulative[start]+progress*lengths[start])
    # Search in path order AFTER the nearest projection. Clip each segment to
    # the forward half-plane, then find its first point outside the aim radius.
    # Never insert an ego origin or extrapolate beyond the predicted endpoint.
    for i in range(start,len(lengths)):
        if lengths[i] <= 1e-8:
            continue
        p,d = points[i],points[i+1]-points[i]
        lo,hi = (progress if i==start else 0.),1.
        if abs(d[1]) < 1e-10:
            if p[1] < cfg['min_forward_m']:
                continue
        elif d[1] > 0:
            lo = max(lo,(cfg['min_forward_m']-p[1])/d[1])
        else:
            hi = min(hi,(cfg['min_forward_m']-p[1])/d[1])
        if lo > hi or hi < 0 or lo > 1:
            continue
        candidate = p+lo*d
        if np.linalg.norm(candidate) < radius-1e-8:
            a,b,c = float(np.dot(d,d)),float(2*np.dot(p,d)),float(np.dot(p,p)-radius*radius)
            discriminant = b*b-4*a*c
            if discriminant < 0:
                continue
            lo = max(lo,(-b+math.sqrt(discriminant))/(2*a))
            if lo > hi+1e-9:
                continue
            lo = min(lo,hi)
            candidate = p+lo*d
        return dict(aim_left_forward=candidate.tolist(),
                    target_selection='remaining_path',target_waypoint_indices=[i,i+1],
                    target_segment_fraction=float(lo),lookahead_m=radius,
                    path_progress_m=progress_m,remaining_path_m=float(cumulative[-1]-progress_m),
                    target_path_distance_m=float(cumulative[i]+lo*lengths[i]-progress_m))
    raise ValueError('insufficient_remaining_path')
