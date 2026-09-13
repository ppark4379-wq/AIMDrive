"""Receive-clock diagnostics and conservative training-teacher validation.

Header clocks are analysed separately. Wall minus header is explicitly NOT a
latency estimate. Agreement within Ego alone never certifies sensor alignment.
"""
from __future__ import annotations

from collections import defaultdict
import math
import numpy as np


def stats(values):
    x = np.asarray(list(values), dtype=float)
    x = x[np.isfinite(x)]
    if not len(x):
        return None
    return dict(count=len(x), min=float(x.min()), median=float(np.median(x)),
                p95=float(np.percentile(x, 95)), max=float(x.max()))


def unique_receive(samples):
    by_time = {int(s["receive_monotonic_ns"]): s for s in samples}
    return [by_time[k] for k in sorted(by_time)]


def xyz(value):
    if isinstance(value, dict):
        return [float(value.get(k, 0)) for k in ("x", "y", "z")]
    return list(value)


def normalize_sample(row):
    p = row["payload"]
    s = {k: row.get(k) for k in ("receive_monotonic_ns", "receive_wall_ns", "source_stamp_ns", "seq", "frame_id")}
    s["callerid"] = row.get("connection", {}).get("callerid")
    topic = row["topic"]
    if topic == "/Ego_topic":
        s.update(position_xyz=xyz(p["position"]), velocity_xyz=xyz(p["velocity"]),
                 heading_deg=p["heading"], unique_id=p["unique_id"])
        s["speed_xy_mps"] = math.hypot(*s["velocity_xyz"][:2])
    elif topic == "/gps":
        s.update({k: p.get(k) for k in ("latitude", "longitude", "altitude", "eastOffset", "northOffset", "status")})
    elif topic == "/imu":
        q = p["orientation"]
        s["orientation_xyzw"] = [q[k] for k in ("x", "y", "z", "w")]
        s["angular_velocity_xyz"] = xyz(p["angular_velocity"])
        s["orientation_covariance"] = p.get("orientation_covariance", [])
    return s


def manifest_streams(rows):
    streams = defaultdict(list)
    for row in rows:
        for name in ("ego_teacher", "gps", "imu", "lidar"):
            if row.get(name):
                streams[name].append(row[name])
        for name, s in row.get("cameras", {}).items():
            streams[name].append(s)
    return {k: unique_receive(v) for k, v in streams.items()}


def clock_summary(samples):
    samples = unique_receive(samples)
    out = {"samples": len(samples)}
    if len(samples) < 2:
        return out
    r = np.array([s["receive_monotonic_ns"] for s in samples], dtype=np.int64)
    out.update(receive_duration_s=float((r[-1]-r[0])/1e9),
               receive_rate_hz=float((len(r)-1)*1e9/(r[-1]-r[0])),
               callback_interval_ms=stats(np.diff(r)/1e6))
    good = [s for s in samples if s.get("source_stamp_ns") is not None]
    if len(good) > 1:
        source = np.array([s["source_stamp_ns"] for s in good], dtype=np.int64)
        receive = np.array([s["receive_monotonic_ns"] for s in good], dtype=np.int64)
        x, y = (receive-receive[0])/1e9, (source-source[0])/1e9
        out.update(source_backsteps=int(np.sum(np.diff(source)<0)),
                   source_duplicate_steps=int(np.sum(np.diff(source)==0)),
                   source_to_receive_slope=float(np.polyfit(x,y,1)[0]),
                   relative_progress_deficit_s=float(x[-1]-y[-1]),
                   source_interval_ms=stats(np.diff(source)/1e6))
        if all(s.get("receive_wall_ns") is not None for s in good):
            wall = np.array([s["receive_wall_ns"] for s in good], dtype=np.int64)
            out["wall_minus_header_s_NOT_latency"] = [(int(wall[i])-int(source[i]))/1e9 for i in (0,-1)]
            out["wall_step_residual_ms"] = stats((np.diff(wall)-np.diff(receive))/1e6)
            out["wall_step_over_50ms_count"] = int(np.sum(np.abs(np.diff(wall)-np.diff(receive))>50e6))
    out["callerids"] = sorted({s["callerid"] for s in samples if s.get("callerid")})
    seq = [s["seq"] for s in samples if s.get("seq") is not None]
    if len(seq)>1:
        out["seq_backsteps"] = sum(b<a and not (a>2**32-100 and b<100) for a,b in zip(seq,seq[1:]))
        out["seq_gaps"] = sum(max(0,b-a-1) for a,b in zip(seq,seq[1:]))
    return out


def gps_xy(samples):
    lat = np.array([s["latitude"] for s in samples], dtype=float)
    lon = np.array([s["longitude"] for s in samples], dtype=float)
    return np.column_stack(((lon-lon[0])*111320*math.cos(math.radians(lat[0])), (lat-lat[0])*111320))


def imu_heading_comparison(ego, imu, max_age_ms=250):
    ego,imu=unique_receive(ego),unique_receive(imu)
    if not ego or not imu:
        return {"valid_pairs":0}
    times=np.array([s["receive_monotonic_ns"] for s in ego],dtype=np.int64)
    errors=[]; invalid=0
    for s in imu:
        q=s.get("orientation_xyzw",[])
        covariance=s.get("orientation_covariance",[])
        if len(q)!=4 or not all(math.isfinite(x) for x in q) or abs(sum(x*x for x in q)-1)>0.01 or (covariance and covariance[0]<0):
            invalid+=1
            continue
        i=int(np.searchsorted(times,s["receive_monotonic_ns"],side="right")-1)
        if i<0 or s["receive_monotonic_ns"]-times[i]>max_age_ms*1e6:
            continue
        x,y,z,w=q
        yaw=math.degrees(math.atan2(2*(w*z+x*y),1-2*(y*y+z*z)))
        errors.append(abs((yaw-ego[i]["heading_deg"]+180)%360-180))
    return dict(valid_pairs=len(errors),invalid_orientation_samples=invalid,
                yaw_minus_ego_heading_abs_deg=stats(errors),
                caveat="Assumes aligned IMU mounting/world axes; diagnostic only. IMU and Ego can share the same backlog.")


def motion_comparison(ego, gps, max_age_ms=250, window_s=1.0):
    """Pair at GPS callback anchors, always latest Ego receive <= GPS receive.

    Short-window horizontal displacement is coordinate-origin independent.
    GPS course is east/north atan2, not a field supplied by GPSMessage.
    """
    ego, gps = unique_receive(ego), unique_receive(gps)
    gps = [s for s in gps if s.get("latitude") is not None and s.get("longitude") is not None
           and math.isfinite(s["latitude"]) and math.isfinite(s["longitude"])
           and abs(s["latitude"]) <= 90 and abs(s["longitude"]) <= 180
           and (s["latitude"] != 0 or s["longitude"] != 0)]
    if len(ego)<2 or len(gps)<2:
        return {"status": "unverified", "reason": "insufficient independent GPS/Ego samples", "windows": []}
    et=np.array([s["receive_monotonic_ns"] for s in ego], dtype=np.int64)
    gt=np.array([s["receive_monotonic_ns"] for s in gps], dtype=np.int64)
    ep=np.array([s["position_xyz"][:2] for s in ego], dtype=float)
    gp=gps_xy(gps)
    windows=[]
    end=0
    for i,t in enumerate(gt):
        if t < end:
            continue
        j=int(np.searchsorted(gt,t-int(window_s*1e9), side="right")-1)
        if j<0 or t-gt[j] > 2*window_s*1e9:
            continue
        a=int(np.searchsorted(et,gt[j],side="right")-1)
        b=int(np.searchsorted(et,t,side="right")-1)
        if a<0 or b<=a or max(gt[j]-et[a],t-et[b])>max_age_ms*1e6:
            continue
        dt=(int(t)-int(gt[j]))/1e9
        dg=gp[i]-gp[j]; de=ep[b]-ep[a]
        gs=float(np.linalg.norm(dg)/dt); es=float(np.linalg.norm(de)/dt)
        course=math.degrees(math.atan2(dg[1],dg[0]))
        angle=abs((float(ego[b]["heading_deg"])-course+180)%360-180)
        windows.append(dict(receive_monotonic_ns=int(t), duration_s=dt, gps_speed_receive_mps=gs,
                            ego_speed_receive_mps=es, gps_displacement_m=float(np.linalg.norm(dg)),
                            ego_displacement_m=float(np.linalg.norm(de)),
                            gps_course_deg=course,
                            heading_axis_error_deg=min(angle,180-angle),
                            ego_heading_deg=float(ego[b]["heading_deg"]),
                            mismatch=bool(max(gs,es)>2 and min(gs,es)<0.25*max(gs,es))))
        end=t+int(window_s*1e9)
    moving=[w for w in windows if max(w["gps_speed_receive_mps"],w["ego_speed_receive_mps"])>2]
    mismatch=sum(w["mismatch"] for w in moving)
    heading_checks=[w for w in moving if not w["mismatch"] and w["gps_speed_receive_mps"]>2]
    heading_bad=sum(w["heading_axis_error_deg"]>45 for w in heading_checks)
    if len(moving)<5:
        status="unverified"
    elif mismatch/len(moving)>0.3 or (len(heading_checks)>=5 and heading_bad/len(heading_checks)>0.3):
        status="fail"
    else:
        status="pass"
    return dict(status=status, moving_windows=len(moving), mismatch_windows=mismatch,
                heading_mismatch_windows=heading_bad,
                gps_unique_latlon=len({(s["latitude"],s["longitude"]) for s in gps}),
                gps_speed_receive_mps=stats(w["gps_speed_receive_mps"] for w in windows),
                ego_speed_receive_mps=stats(w["ego_speed_receive_mps"] for w in windows), windows=windows)


def validate_manifest(rows, max_age_ms=250):
    streams=manifest_streams(rows)
    ego=streams.get("ego_teacher", [])
    clocks={k:clock_summary(v) for k,v in streams.items()}
    failures=[]
    if not rows or len(ego)<2:
        failures.append("empty_or_insufficient_teacher")
    if any(not r.get("ego_teacher") for r in rows):
        failures.append("missing_teacher")
    if any(not r.get("imu") or not r.get("lidar") or
           not {"front","left","right"} <= set(r.get("cameras",{})) for r in rows):
        failures.append("missing_required_model_sensor")
    if ego:
        if any(s.get("source_stamp_ns") is None for s in ego):
            failures.append("missing_teacher_header_clock")
        if any(not all(math.isfinite(float(x)) for x in s.get("position_xyz",[])+s.get("velocity_xyz",[])+[s.get("heading_deg",float('nan'))]) for s in ego):
            failures.append("nonfinite_teacher")
        if clocks["ego_teacher"].get("source_backsteps",0):
            failures.append("teacher_header_reset")
        if len({s.get("source_stamp_ns") for s in ego}) < 2:
            failures.append("teacher_header_not_progressing")
        if clocks["ego_teacher"].get("seq_backsteps",0):
            failures.append("teacher_publisher_sequence_reset")
        if len({s.get("unique_id") for s in ego})>1 or len(clocks["ego_teacher"].get("callerids",[]))>1:
            failures.append("teacher_identity_changed_or_multiple_publishers")
        for a,b in zip(ego,ego[1:]):
            ds=(b.get("source_stamp_ns",0) or 0)-(a.get("source_stamp_ns",0) or 0)
            if ds > 500_000_000:
                failures.append("teacher_source_gap_over_500ms")
            d=math.dist(a["position_xyz"][:2],b["position_xyz"][:2])
            if ds == 0 and d > 0.001:
                failures.append("conflicting_teacher_at_same_source_stamp")
            v=max(a.get("speed_xy_mps",0),b.get("speed_xy_mps",0))
            if d>max(5, 3*v*max(0,ds)/1e9+1):
                failures.append("teacher_teleport_or_kinematic_discontinuity")
                break
    ages=[]
    for row in rows:
        anchor=row.get("lidar",{}).get("receive_monotonic_ns")
        if anchor is None:
            continue
        selected=list(row.get("cameras",{}).values()) + [row[k] for k in ("ego_teacher","imu") if row.get(k)]
        ages.extend((anchor-s["receive_monotonic_ns"])/1e6 for s in selected)
    if any(a<0 or a>max_age_ms for a in ages):
        failures.append("future_or_stale_selected_sensor")
    motion=motion_comparison(ego,streams.get("gps",[]),max_age_ms)
    if motion["status"]=="fail":
        failures.append("gps_ego_receive_motion_disagreement")
    # Different header rates are not proof of latency, but must be resolved
    # before calling the common sensor capture epoch and teacher horizons valid.
    slopes=[c["source_to_receive_slope"] for c in clocks.values()
            if c.get("receive_duration_s",0)>=10 and "source_to_receive_slope" in c]
    if len(slopes)>1 and max(slopes)-min(slopes)>0.15:
        failures.append("unvalidated_divergent_header_progress")
    verified=motion["status"]=="pass"
    status="fail" if failures else ("pass" if verified else "unverified")
    return dict(schema_version="stage6e-teacher-integrity-v1", status=status, frames=len(rows),
                failures=sorted(set(failures)), clocks=clocks, selected_age_ms=stats(ages), motion=motion,
                imu_heading=imu_heading_comparison(ego,streams.get("imu",[]),max_age_ms),
                limitations=["Callback entry is not capture time or wire arrival.",
                             "Header progress differences require per-stream clock validation; they do not prove absolute latency.",
                             "GPS agreement is a consistency check, not an independent absolute truth certificate."])
