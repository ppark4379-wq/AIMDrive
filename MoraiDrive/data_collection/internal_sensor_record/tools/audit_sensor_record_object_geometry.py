#!/usr/bin/env python3
"""Check observed gRPC boxes against pinned per-model measurement evidence.

This gate authorizes only an API-box geometry candidate. It does not turn a
transport/source-time association into a camera exposure certificate and never
silently accepts an unknown model.
"""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from async_capture.journal import digest,write_json
from async_capture.sensor_record_admission import verify_sealed_directory
from ssd_storage import require_ssd


DEFAULT_CATALOG=(ROOT/'evidence/native_mission_integration_v1/'
                 'catalog_review_v1/report.json')
EXPECTED_CATALOG_KIND={'npc':'vehicle','pedestrian':'pedestrian','obstacle':'static'}


def median_dimensions_wlh(case):
    rows=[view.get('dimensions_lwh_m') for view in case.get('views',[])]
    if len(rows)!=4 or any(not isinstance(row,list) or len(row)!=3 for row in rows):
        return None
    # The catalog names its axes L/W/H; the GT timeline explicitly stores W/L/H.
    return [statistics.median(row[index] for row in rows) for index in (1,0,2)]


def check_object_geometry(item,catalog_cases,catalog_hash,absolute_tolerance_m,
                          relative_tolerance):
    """Return a fail-closed API-box comparison for one model instance."""
    model=str(item.get('model_name','')).strip();kind=item.get('kind')
    dims=item.get('size_wlh_m');case=catalog_cases.get(model)
    check=dict(instance_key=item.get('instance_key'),actor_id=item.get('id'),
               model_name=model,kind=kind,observed_size_wlh_m=dims,
               catalog_report_sha256=catalog_hash)
    reasons=[]
    if (not isinstance(dims,list) or len(dims)!=3 or
            not all(math.isfinite(float(value)) and float(value)>0 for value in dims)):
        reasons.append('invalid_observed_dimensions')
    reference=median_dimensions_wlh(case) if case else None
    if case is None:reasons.append('model_absent_from_measured_catalog')
    elif case.get('kind')!=EXPECTED_CATALOG_KIND.get(kind):reasons.append('catalog_kind_mismatch')
    elif case.get('visible_return_contradiction') is True:
        reasons.append('catalog_visible_return_contradiction')
    elif reference is None:reasons.append('catalog_lacks_four_views')
    elif not reasons:
        errors=[abs(float(value)-reference[index]) for index,value in enumerate(dims)]
        tolerances=[max(absolute_tolerance_m,
                        relative_tolerance*reference[index]) for index in range(3)]
        check.update(reference_size_wlh_m=reference,absolute_error_m=errors,
                     tolerance_m=tolerances)
        if any(error>tolerance for error,tolerance in zip(errors,tolerances)):
            reasons.append('observed_dimensions_differ_from_catalog')
    check['reasons']=reasons;check['api_box_geometry_candidate']=not reasons
    return check


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--gt-timeline-dir',default='gt_timeline_v1')
    parser.add_argument('--actor-inventory-dir',default='actor_inventory_audit_v1')
    parser.add_argument('--catalog-report',type=Path,default=DEFAULT_CATALOG)
    parser.add_argument('--absolute-tolerance-m',type=float,default=.03)
    parser.add_argument('--relative-tolerance',type=float,default=.02)
    args=parser.parse_args()
    for path in (args.episode,args.output):require_ssd(path,reserve_bytes=0)
    if args.output.exists():raise FileExistsError(args.output)
    if not 0<args.absolute_tolerance_m<=.2 or not 0<args.relative_tolerance<=.1:
        raise ValueError('Invalid bounded geometry tolerances')
    gt_dir=args.episode/args.gt_timeline_dir
    actor_dir=args.episode/args.actor_inventory_dir
    gt_seal=verify_sealed_directory(gt_dir)
    actor_seal=verify_sealed_directory(actor_dir)
    actor=json.loads((actor_dir/'report.json').read_text())
    frames=json.loads((gt_dir/'frames.json').read_text())
    catalog=json.loads(args.catalog_report.read_text())
    catalog_cases={row['model']:row for row in catalog.get('cases',[])}
    catalog_hash=digest(args.catalog_report)
    observations=defaultdict(list);observed_models=set();observed_instances=set()
    output=[];status=Counter()
    for frame in frames:
        reasons=[];objects=frame['object_gt'].get('objects',[])
        if frame['object_gt'].get('source')!='grpc_full_inventory':
            reasons.append('not_full_grpc_inventory')
        if frame['object_gt'].get('transport_timing_candidate') is not True:
            reasons.append('object_time_association_failed')
        if frame['object_gt'].get('model_identity_candidate') is not True:
            reasons.append('actor_model_identity_failed')
        checks=[]
        for item in objects:
            check=check_object_geometry(item,catalog_cases,catalog_hash,
                args.absolute_tolerance_m,args.relative_tolerance)
            model=check['model_name'];dims=check['observed_size_wlh_m']
            key=item.get('instance_key');checks.append(check)
            if model:observed_models.add(model)
            if key:observed_instances.add(key)
            if isinstance(dims,list) and len(dims)==3:
                observations[(model,key)].append([float(value) for value in dims])
            reasons.extend('%s:%s'%(model or '<unknown>',reason) for reason in check['reasons'])
        reasons=sorted(set(reasons));candidate=not reasons
        status['candidate' if candidate else '+'.join(reasons)]+=1
        output.append(dict(index=frame['index'],lidar_source_ns=frame['lidar_source_ns'],
            object_count=len(objects),objects=checks,reasons=reasons,
            api_box_geometry_candidate=candidate,
            physical_sensor_gt_atomicity_verified=False,training_visible=False))
    per_instance=[]
    for (model,key),rows in sorted(observations.items()):
        ranges=[max(row[i] for row in rows)-min(row[i] for row in rows) for i in range(3)]
        per_instance.append(dict(model_name=model,instance_key=key,observations=len(rows),
                                 observed_size_range_wlh_m=ranges))
    report=dict(schema='morai_sensor_record_object_geometry_audit_v1',frames=len(frames),
        frame_status=dict(status),api_box_geometry_candidate_frames=sum(
            row['api_box_geometry_candidate'] for row in output),
        raw_manifest_sha256=actor.get('raw_manifest_sha256'),
        observed_models=sorted(observed_models),
        observed_actor_instances=len(observed_instances),per_instance_dimension_stability=per_instance,
        full_actor_model_identity_complete=actor.get('model_identity_complete') is True,
        catalog_report=str(args.catalog_report.resolve()),catalog_report_sha256=catalog_hash,
        catalog_measurement_sha256=catalog.get('measurement_sha256'),
        structural_valid=(len(output)==len(frames) and actor.get('structural_valid') is True),
        complete_observed_geometry_scope=bool(output and all(
            row['api_box_geometry_candidate'] for row in output)),
        physical_sensor_gt_atomicity_verified=False,object_training_label_valid=False,
        training_visible=False,
        limitation=('Per-model gRPC API-box consistency against stationary four-view measurements. '
                    'Only actors retained in a stable sensor-frame bracket are in this scope. '
                    'It does not prove camera exposure atomicity, occluded-box tightness, or unknown models.'))
    args.output.mkdir(parents=True,exist_ok=False)
    write_json(args.output/'report.json',report);write_json(args.output/'frames.json',output)
    write_json(args.output/'COMPLETE.json',dict(schema=report['schema'],files={
        name:digest(args.output/name) for name in ('report.json','frames.json')},
        gt_timeline_seal_sha256=digest(gt_dir/'COMPLETE.json'),
        actor_inventory_seal_sha256=digest(actor_dir/'COMPLETE.json'),
        catalog_report_sha256=catalog_hash,script_sha256=digest(Path(__file__)),
        training_visible=False))
    print(json.dumps(report),flush=True)
    if not report['structural_valid']:raise SystemExit(2)


if __name__=='__main__':main()
