#!/usr/bin/env python3
"""Offline actor-origin and timestamp audit. Does not alter predictions or GT."""
import argparse
from bisect import bisect_left
from collections import defaultdict
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from MoraiDrive.scenario_runner.legacy_expert.ve_gt_comparison import (load_rows, stats, rect, roi, in_roi,
    score_frame, summarize, render_review, breakdown)


def interpolate(history, target, fields, max_gap_ns=200_000_000):
    """Interpolate only bracketed samples of the same actor; never extrapolate."""
    times, rows = history
    i = bisect_left(times, target)
    if i < len(times) and times[i] == target:
        return {key: rows[i][key] for key in fields}
    if i == 0 or i == len(times) or times[i]-times[i-1] > max_gap_ns:
        return None
    a, b = rows[i-1], rows[i]
    for key in ('position', 'center_world'):
        if key in a and math.dist(a[key], b[key]) > 5.:
            return None
    weight = (target-times[i-1])/(times[i]-times[i-1])
    result = {}
    for key in fields:
        aa, bb = a[key], b[key]
        if isinstance(aa, list):
            result[key] = [x+weight*(y-x) for x,y in zip(aa,bb)]
        elif 'heading' in key:
            result[key] = aa+weight*((bb-aa+180.)%360.-180.)
        else:
            result[key] = aa+weight*(bb-aa)
    return result


def histories(rows, entries, stamp_key, id_key, scale=1):
    groups = defaultdict(dict)
    for row in rows:
        if row.get('error') or row.get('errors'):
            continue
        for item in row[entries]:
            groups[item[id_key]][row[stamp_key]*scale] = item
    return {key:(sorted(v), [v[t] for t in sorted(v)]) for key,v in groups.items()}


def metric(g, v):
    from utils.evaluation.morai_object_metrics import oriented_iou
    return dict(center_error_m=math.dist(g['center_world'], v['center_world']),
                iou=oriented_iou(rect(g), rect(v)),
                speed_error_mps=abs(g['speed_mps']-v['speed_mps']) if v['speed_mps'] is not None else None)


def summarize_pairs(values):
    return {k:stats([v[k] for v in values]) for k in ('center_error_m','iou','speed_error_mps')}


def analyze(output):
    from data.src.loaders.morai_map_geometry import validate_map_basis
    from inference.realtime_ve_objects import MAP_BASIS
    geo = validate_map_basis(MAP_BASIS)
    gt_rows = load_rows(output/'gt_objects.jsonl')
    geometry = load_rows(output/'gt_geometry.jsonl')
    frames = load_rows(output/'gt_comparison_frames.jsonl')
    packets = {p['sequence']:p for p in load_rows(output/'adapter.jsonl')}
    gt = histories(gt_rows,'boxes','source_stamp_ns','track_id')
    # MORAI ActorStates timestamps are microseconds. Verify clock range below.
    geom = histories(geometry,'states','source_timestamp_raw','actor_id',1000)
    valid_geom = [r for r in geometry if not r.get('error')]
    if not gt_rows or not valid_geom:
        raise ValueError('Missing GT/geometry evidence')
    gt_min, gt_max = min(r['source_stamp_ns'] for r in gt_rows), max(r['source_stamp_ns'] for r in gt_rows)
    geom_min, geom_max = min(r['source_timestamp_raw']*1000 for r in valid_geom), max(r['source_timestamp_raw']*1000 for r in valid_geom)
    if min(gt_max,geom_max) <= max(gt_min,geom_min):
        raise ValueError('GT and gRPC source clocks do not overlap; cannot interpolate')
    by_name = defaultdict(lambda:defaultdict(list))
    # Check source-clock rate correspondence instead of trusting epoch magnitude.
    for a,b in zip(valid_geom,valid_geom[1:]):
        dt = (b['source_timestamp_raw']-a['source_timestamp_raw'])*1000
        if 0 < dt < 500_000_000:
            by_name['_clock']['grpc_source_to_receipt_rate'].append(dt/(b['received_ns']-a['received_ns']))
    for r in gt_rows:
        if r['errors']:
            continue
        for b in r['boxes']:
            actor = b['track_id'].split(':',1)[1]
            if actor not in geom:
                continue
            g = interpolate(geom[actor],r['source_stamp_ns'],['position','bbox_position','bbox_extent'])
            if g is None:
                continue
            name = by_name[b['name']]
            name['ros_to_actor_position_m'].append(math.dist(b['center_world'],g['position'][:2]))
            name['ros_to_bbox_position_m'].append(math.dist(b['center_world'],g['bbox_position'][:2]))
            dx,dy = [g['bbox_position'][k]-g['position'][k] for k in (0,1)]
            yaw = math.radians(b['heading_raw_deg']+geo.convergence_degrees([*b['center_world'],b['z_m']]))
            name['actor_to_bbox_forward_m'].append(dx*math.cos(yaw)+dy*math.sin(yaw))
            name['actor_to_bbox_left_m'].append(-dx*math.sin(yaw)+dy*math.cos(yaw))
            name['extent_length_vs_ros_m'].append(abs(2*g['bbox_extent'][1]-b['length_m']))
            name['extent_width_vs_ros_m'].append(abs(2*g['bbox_extent'][0]-b['width_m']))
    timing = defaultdict(list)
    comparisons = defaultdict(list)
    pair_rows = []
    common = []
    corrected_frames = []
    corrected_skipped = 0
    gt_by_stamp = {r['source_stamp_ns']:r for r in gt_rows if not r['errors']}
    for frame in frames:
        packet = packets[frame['sequence']]
        # Re-evaluate detection with the independent bbox-center reference,
        # including all GT boxes before applying the same BEV and 2m gate.
        corrected, valid = [], True
        area = roi(frame['ego'])
        for b in gt_by_stamp[frame['gt_source_stamp_ns']]['boxes']:
            actor = b['track_id'].split(':',1)[1]
            sample = interpolate(geom[actor],frame['gt_source_stamp_ns'],['position','bbox_position']) if actor in geom else None
            if sample is None:
                unshifted = dict(b,heading_deg=b['heading_raw_deg']+geo.convergence_degrees([*b['center_world'],b['z_m']]))
                if in_roi(unshifted,area):
                    valid = False
                continue
            bb = dict(b,center_world=[b['center_world'][i]+sample['bbox_position'][i]-sample['position'][i] for i in (0,1)])
            bb['heading_deg'] = b['heading_raw_deg']+geo.convergence_degrees([*b['center_world'],b['z_m']])
            if in_roi(bb,area):
                corrected.append(bb)
        if valid:
            corrected_frames.append(dict(frame,gt=corrected,score=score_frame(corrected,frame['ve'],frame['ego'])))
        else:
            corrected_skipped += 1
        samples = packet['audit']['selected_samples']
        camera_stamps = [samples[k].get('source_stamp_ns',0) for k in ('rgb_front','rgb_left','rgb_right','rgb_rear')]
        front = camera_stamps[0]
        status = samples['vehicle_status'].get('source_stamp_ns',0)
        if all(camera_stamps):
            timing['camera_source_spread_ms'].append((max(camera_stamps)-min(camera_stamps))/1e6)
        if front and status:
            timing['ego_minus_front_source_ms'].append((status-front)/1e6)
        timing['ego_receipt_minus_lidar_completion_ms'].append(
            (samples['vehicle_status']['received_ns']-samples['lidar']['received_ns'])/1e6)
        if front:
            timing['gt_at_lidar_receipt_minus_front_source_ms'].append((frame['gt_source_stamp_ns']-front)/1e6)
        for p in frame['score']['matched']:
            g,v = frame['gt'][p['gt_index']], frame['ve'][p['ve_index']]
            key = g['track_id']
            actor = key.split(':',1)[1]
            values = {'actor_at_lidar_receipt':metric(g,v)}
            for label,target in [('lidar_receipt',frame['gt_source_stamp_ns']),('front_source',front),('ego_source',status)]:
                if not target or key not in gt:
                    continue
                temporal = interpolate(gt[key],target,['center_world','heading_raw_deg','speed_mps'])
                if temporal is None:
                    continue
                gg = dict(g,**temporal)
                gg['heading_deg'] = gg['heading_raw_deg']+geo.convergence_degrees([*gg['center_world'],g['z_m']])
                values['actor_at_'+label] = metric(gg,v)
                geometry_sample = interpolate(geom[actor],target,['position','bbox_position']) if actor in geom else None
                if geometry_sample is not None:
                    # Offset independently supplied by MORAI, never fitted to VE.
                    gg['center_world'] = [gg['center_world'][i]+geometry_sample['bbox_position'][i]-geometry_sample['position'][i] for i in (0,1)]
                    values['bbox_at_'+label] = metric(gg,v)
            for label, m in values.items():
                comparisons[label].append(m)
            if len(values) == 6:
                common.append(values)
            pair_rows.append(dict(sequence=frame['sequence'],gt_id=key,gt_name=g['name'],metrics=values))
    report = dict(schema='ve_bbox_origin_timing_audit_v1', status='COMPLETE', run=str(output),
        coordinate_reference={k:{kk:stats(vv) for kk,vv in v.items()} for k,v in by_name.items()},
        timestamp_metrics={k:stats(v) for k,v in timing.items()},
        per_available_pair={k:summarize_pairs(v) for k,v in comparisons.items()},
        same_pairs_count=len(common), same_pairs={k:summarize_pairs([r[k] for r in common]) for k in comparisons},
        by_gt_model={name:{label:summarize_pairs([r['metrics'][label] for r in pair_rows
            if r['gt_name'] == name and label in r['metrics']]) for label in comparisons}
            for name in sorted({r['gt_name'] for r in pair_rows})},
        grpc_roundtrip_ms=stats([(r['received_ns']-r['start_ns'])/1e6 for r in valid_geom]),
        grpc_errors=sum(bool(r.get('error')) for r in geometry),
        bbox_reference_detection=summarize(corrected_frames),
        bbox_reference_by_model=breakdown(corrected_frames,'model'),
        bbox_reference_skipped_frames=corrected_skipped,
        policy='Same original 2m-gated actor matches; interpolated GT, no rematching; bbox offset from gRPC, never fitted to VE',
        limitations=['LiDAR physical source timestamp is still unknown; front camera alignment is a sensitivity check.',
                    'gRPC bbox is simulator metadata; no independent mesh/camera certification.',
                    'Source clocks checked for overlap and rate, not hardware synchronization.',
                    'Matched metrics exclude misses; use gt_comparison_report.json for recall.'],
        training_updates=0, prediction_modified=False, gt_used_for_driving=False)
    (output/'gt_geometry_audit_pairs.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in pair_rows))
    review = output/'bbox_reference_review'
    review.mkdir(exist_ok=True)
    render_review(corrected_frames,review)
    page = review/'gt_comparison_review.html'
    page.write_text(page.read_text().replace('초록: MORAI Object_topic','초록: MORAI bbox 중심').replace(
        '원본 객체 기준점·비동기 센서의 오차 포함.', 'gRPC에서 관측한 bbox 중심 차이를 반영. 비동기 센서 오차 포함.'))
    (output/'bbox_reference_frames.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in corrected_frames))
    (output/'gt_geometry_audit_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,ensure_ascii=False))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    analyze(parser.parse_args().output)
