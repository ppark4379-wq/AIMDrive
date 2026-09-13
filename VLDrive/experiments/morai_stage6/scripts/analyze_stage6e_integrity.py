#!/usr/bin/env python3
"""Analyse saved runs or a diagnostic drive; writes only a new evidence file."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

from aimdrive.ros_training.teacher_integrity import (
    clock_summary, manifest_streams, motion_comparison, normalize_sample, stats, validate_manifest,
)

ROOT=Path(__file__).resolve().parents[1]


def read_rows(path):
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def source_alignment_evidence(streams):
    """A clock hypothesis test ONLY; never a synchronization policy change.

    EPSG:32652 minus GPS offsets is an explicit K-City candidate. Report raw
    residuals; a small result empirically supports this coordinate hypothesis.
    """
    import numpy as np
    try:
        from pyproj import Transformer
    except ImportError:
        return {"status":"unavailable", "reason":"pyproj not installed"}
    e,g=streams.get("ego_teacher",[]),streams.get("gps",[])
    if len(e)<2 or len(g)<2:
        return {"status":"unavailable"}
    tr=Transformer.from_crs("EPSG:4326","EPSG:32652",always_xy=True)
    gx,gy=tr.transform([s["longitude"] for s in g],[s["latitude"] for s in g])
    gp=np.column_stack([gx,gy])-np.array([[s["eastOffset"],s["northOffset"]] for s in g])
    ep=np.array([s["position_xyz"][:2] for s in e])
    result={"coordinate_hypothesis":"WGS84 UTM 52N minus GPS eastOffset/northOffset",
            "note":"Source-time matching tests a hypothesis; cross-sensor causal policy remains receive monotonic."}
    for key in ["receive_monotonic_ns","source_stamp_ns"]:
        if any(s.get(key) is None for s in e+g):
            continue
        et=np.array([s[key] for s in e],dtype=np.int64)
        gt=np.array([s[key] for s in g],dtype=np.int64)
        if np.any(np.diff(gt)<0):
            result[key]={"status":"not_compared_nonmonotonic_GPS"}
            continue
        valid=(et>=gt[0])&(et<=gt[-1])
        if not valid.any():
            result[key]={"status":"no_overlap"}
            continue
        # Do not interpolate across long holes or extrapolate boundaries.
        right=np.searchsorted(gt,et,side="left").clip(1,len(gt)-1)
        valid &= gt[right]-gt[right-1] < 500_000_000
        zero=min(int(et[0]),int(gt[0]))
        q=np.column_stack([np.interp((et-zero)/1e9,(gt-zero)/1e9,gp[:,i]) for i in range(2)])
        moving=valid & np.array([s.get("speed_xy_mps",0)>=2 for s in e])
        result[key]={"position_error_m":stats(np.linalg.norm(ep[valid]-q[valid],axis=1)),
                     "moving_ego_position_error_m":stats(np.linalg.norm(ep[moving]-q[moving],axis=1))}
    return result


def analyze_drive(directory):
    samples=[]
    other=[]
    for profile in ["state","sensors"]:
        path=directory/(profile+".jsonl")
        if path.exists():
            for r in read_rows(path):
                (samples if r.get("kind")=="sample" else other).append(r)
    samples.sort(key=lambda r:r["receive_monotonic_ns"])
    events=read_rows(directory/"events.jsonl")
    stages=[(e["event"],e["monotonic_ns"]) for e in events if e["event"] in
            ["baseline_start","recorder_start_requested","recorder_exit","diagnostic_stop"]]
    phases={}
    for (name,start),(_,stop) in zip(stages,stages[1:]):
        by_topic=defaultdict(list)
        for r in samples:
            if start<=r["receive_monotonic_ns"]<stop:
                by_topic[r["topic"]+"|"+r.get("frame_id","")].append(normalize_sample(r))
        phases[name]={k:clock_summary(v) for k,v in by_topic.items()}
    independent=[normalize_sample(r) for r in samples if r["topic"]=="/Ego_topic"]
    gps=[normalize_sample(r) for r in samples if r["topic"]=="/gps"]
    # Construct diagnostic-only LiDAR anchors using the same receive policy.
    mapping={"/Ego_topic":"ego_teacher","/gps":"gps","/imu":"imu","/lidar3D":"lidar"}
    cameras={"Camera-1":"front","Camera-2":"left","Camera-3":"right","Camera-4":"rear"}
    buffers=defaultdict(list)
    preflight_rows=[]
    baseline_start=next((e["monotonic_ns"] for e in events if e["event"]=="baseline_start"),0)
    for r in samples:
        name=cameras.get(r.get("frame_id")) if r["topic"]=="/image_jpeg/compressed" else mapping.get(r["topic"])
        if not name:
            continue
        s=normalize_sample(r)
        if name=="lidar":
            row={"lidar":s,"cameras":{}}
            for source,buffer in buffers.items():
                if source in cameras.values():
                    row["cameras"][source]=buffer[-1]
                else:
                    row[source]=buffer[-1]
            if r["receive_monotonic_ns"] >= baseline_start:
                preflight_rows.append(row)
        else:
            buffers[name]=[s]
    manifest=directory/"dataset/run_0001/manifest.jsonl"
    match={}
    if manifest.exists():
        rows=read_rows(manifest)
        by_source=defaultdict(list)
        for s in independent:
            by_source[s["source_stamp_ns"]].append(s)
        matched=0; delays=[]
        for r in rows:
            s=r["ego_teacher"]
            candidates=[e for e in by_source[s["source_stamp_ns"]]
                        if e["position_xyz"]==s["position_xyz"] and e["velocity_xyz"]==s["velocity_xyz"]
                        and e["heading_deg"]==s["heading_deg"] and e["unique_id"]==s["unique_id"]]
            if candidates:
                matched+=1
                delays.append(min(((s["receive_monotonic_ns"]-e["receive_monotonic_ns"])/1e6
                                   for e in candidates), key=abs))
        match=dict(frames=len(rows), exact_teacher_payload_matches=matched,
                   recorder_minus_observer_callback_ms=stats(delays), validation=validate_manifest(rows))
    metrics=[r for r in other if r.get("kind")=="metrics"]
    return dict(phases=phases, recorder=match, preflight=validate_manifest(preflight_rows),
                motion=motion_comparison(independent,gps),
                source_alignment=source_alignment_evidence({"ego_teacher":independent,"gps":gps}),
                callback_work_ms=stats(r.get("callback_work_ms",0) for r in samples),
                diagnostic_writer_wait_ms=stats(r.get("diagnostic_writer_wait_ms",0) for r in samples),
                observation_errors=sorted({r["error"] for r in other if r.get("kind")=="observation_error"}),
                diagnostic_drops=max([r.get("diagnostic_drops",0) for r in metrics]+[0]), events=events)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",required=True)
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    path=Path(args.input).resolve(); output=Path(args.output).resolve()
    if ROOT not in output.parents or output.exists():
        parser.error("Output must be a new file inside this repository")
    if path.is_dir() and (path/"events.jsonl").exists():
        report=analyze_drive(path)
    else:
        manifests=[path] if path.is_file() else sorted(path.rglob("manifest.jsonl"))
        report={}
        for manifest in manifests:
            rows=read_rows(manifest)
            result=validate_manifest(rows)
            result["source_alignment"]=source_alignment_evidence(manifest_streams(rows))
            report[str(manifest.relative_to(ROOT))]=result
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open("x") as handle:
        json.dump(report,handle,indent=2)
    print(output)


if __name__=="__main__":
    main()
