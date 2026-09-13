"""Seal a per-frame diagnostic ledger without publishing training data."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from async_capture.journal import digest,write_json
from ssd_storage import require_ssd


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--episode',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gt-timeline-dir',default='gt_timeline_v1')
    p.add_argument('--actor-inventory-dir',default='actor_inventory_audit_v1')
    p.add_argument('--object-geometry-dir',default='object_geometry_audit_v1')
    p.add_argument('--traffic-audit-dir',default='traffic_audit_v1')
    p.add_argument('--route-signal-dir',default='route_signal_audit_v1')
    a=p.parse_args()
    for path in (a.episode,a.output):require_ssd(path,reserve_bytes=0)
    load=lambda path:json.loads(path.read_text())
    seals={}
    for folder,sealname in [('native_audit','COMPLETE.json'),('quality','QUALITY_COMPLETE.json'),
                            ('native_associations','COMPLETE.json'),('motion_time_v1','COMPLETE.json'),
                            ('exposure_v1','COMPLETE.json'),(a.gt_timeline_dir,'COMPLETE.json'),
                            (a.actor_inventory_dir,'COMPLETE.json'),
                            (a.object_geometry_dir,'COMPLETE.json'),
                            (a.traffic_audit_dir,'COMPLETE.json'),
                            (a.route_signal_dir,'COMPLETE.json')]:
        sealpath=a.episode/folder/sealname
        for name,sha in load(sealpath)['files'].items():
            if Path(name).name!=name or digest(a.episode/folder/name)!=sha:raise ValueError('Changed evidence: '+folder)
        seals[folder]=digest(sealpath)
    cat=load(a.episode/'native_audit/report.json')
    motion=load(a.episode/'motion_time_v1/report.json')
    exposure=load(a.episode/'exposure_v1/report.json')
    gt_report=load(a.episode/a.gt_timeline_dir/'report.json')
    index=load(a.episode/'quality/INDEX_COMPLETE.json')
    assocseal=load(a.episode/'native_associations/COMPLETE.json')
    if assocseal['native_audit_sha256']!=digest(a.episode/'native_audit/report.json') or assocseal['raw_manifest_sha256']!=cat['raw_manifest_sha256']:
        raise ValueError('Mixed GPS/teacher associations')
    associations={r['index']:r for r in load(a.episode/'native_associations/associations.json')}
    gt_timeline_seal=load(a.episode/a.gt_timeline_dir/'COMPLETE.json')
    if (gt_timeline_seal['native_audit_sha256']!=digest(a.episode/'native_audit/report.json') or
            gt_timeline_seal['raw_manifest_sha256']!=cat['raw_manifest_sha256']):
        raise ValueError('Mixed object/light GT timeline')
    gt_frames={r['index']:r for r in load(a.episode/a.gt_timeline_dir/'frames.json')}
    actor_frames={r['index']:r for r in load(
        a.episode/a.actor_inventory_dir/'frame_brackets.json')}
    geometry_frames={r['index']:r for r in load(
        a.episode/a.object_geometry_dir/'frames.json')}
    route_frames={r['index']:r for r in load(
        a.episode/a.route_signal_dir/'frames.json')}
    if len(gt_frames)!=len(cat['bundles']):raise ValueError('Incomplete object/light GT timeline')
    if any(len(rows)!=len(cat['bundles']) for rows in
           (actor_frames,geometry_frames,route_frames)):
        raise ValueError('Incomplete actor/geometry/route frame evidence')
    if len({r['raw_manifest_sha256'] for r in (cat,motion,exposure,index)})!=1:raise ValueError('Mixed recording evidence')
    if exposure['input_native_audit_sha256']!=digest(a.episode/'native_audit/report.json') or motion['native_catalog_sha256']!=digest(a.episode/'native_audit/report.json'):
        raise ValueError('Mixed native evidence')
    intervals=load(a.episode/'motion_time_v1/intervals.json')
    adjacent_intervals=defaultdict(list)
    for row in intervals:
        adjacent_intervals[row['before_index']].append(row)
        adjacent_intervals[row['after_index']].append(row)
    grouped=defaultdict(list)
    for row in load(a.episode/'exposure_v1/samples.json'):grouped[row['group_index']].append(row)
    exact={r['relative_path'] for r in cat['comparison'] if r['role']=='lidar' and r.get('exact_xyzi_array_equal')}
    frames=[]
    for i,bundle in enumerate(cat['bundles']):
        views={}
        for row in grouped[i]:
            if not row['valid']:status='insufficient_projected_edges'
            elif row['category']=='stationary_or_slow':status='insufficient_motion'
            elif row['scores_px'][3]<min(row['scores_px'][:3]+row['scores_px'][4:]):status='same_cycle_strictly_best'
            else:status='cycle_not_distinguished_by_edges'
            views[row['view']]=dict(status=status,projected_points=row['points'],scores_px=row['scores_px'])
        for view in ('front','left','right','rear'):
            views.setdefault(view,dict(status='boundary_not_tested'))
        adjacent=adjacent_intervals[i]
        gt=gt_frames[i]
        actor=actor_frames[i];geometry=geometry_frames[i];route=route_frames[i]
        frames.append(dict(index=i,native_filename_time_ns=bundle['filename_simulation_time_hypothesis_ns'],
            complete_native_inputs=not bundle['missing'],exact_ros_native_lidar=bundle['assets'].get('lidar') in exact,
            same_source_gps_teacher_associated=i in associations,
            adjacent_teacher_intervals_consistent=bool(adjacent) and all(r['status']=='kinematically_consistent' for r in adjacent),
            camera_cycle_diagnostics=views,all_four_views_distinguish_cycle=all(v['status']=='same_cycle_strictly_best' for v in views.values()),
            object_gt_transport_timing_candidate=gt['object_gt']['transport_timing_candidate'],
            full_actor_inventory_transport_candidate=actor['transport_candidate'],
            object_model_identity_candidate=bool(actor.get('model_identity_candidate') and
                gt['object_gt'].get('model_identity_candidate')),
            object_api_box_geometry_candidate=geometry['api_box_geometry_candidate'],
            object_gt_reasons=gt['object_gt']['reasons'],
            traffic_light_gt_transport_timing_candidate=gt['traffic_light_gt']['transport_timing_candidate'],
            route_signal_binding_candidate=route['valid'],
            traffic_light_gt_reasons=gt['traffic_light_gt']['reasons'],
            physical_exposure_bound_ms=None,dynamic_object_gt_verified=False,traffic_light_transition_gt_verified=False,
            training_visible=False))
    report=dict(schema='morai_sensor_record_review_summary_v1',frames=len(frames),
        complete_input_frames=sum(r['complete_native_inputs'] for r in frames),
        exact_lidar_frames=sum(r['exact_ros_native_lidar'] for r in frames),
        teacher_consistent_frames=sum(r['adjacent_teacher_intervals_consistent'] for r in frames),
        same_source_gps_teacher_frames=sum(r['same_source_gps_teacher_associated'] for r in frames),
        four_view_cycle_distinguished_frames=sum(r['all_four_views_distinguish_cycle'] for r in frames),
        object_gt_transport_candidate_frames=sum(r['object_gt_transport_timing_candidate'] for r in frames),
        full_actor_inventory_candidate_frames=sum(
            r['full_actor_inventory_transport_candidate'] for r in frames),
        object_model_identity_candidate_frames=sum(
            r['object_model_identity_candidate'] for r in frames),
        object_api_box_geometry_candidate_frames=sum(
            r['object_api_box_geometry_candidate'] for r in frames),
        traffic_light_gt_transport_candidate_frames=sum(r['traffic_light_gt_transport_timing_candidate'] for r in frames),
        route_signal_binding_candidate_frames=sum(
            r['route_signal_binding_candidate'] for r in frames),
        object_gt_status=gt_report['object_frame_status'],traffic_light_gt_status=gt_report['light_frame_status'],
        per_view_diagnostic_counts={v:dict(Counter(r['camera_cycle_diagnostics'][v]['status'] for r in frames)) for v in ('front','left','right','rear')},
        native_payload_bytes=sum(r['size_bytes'] for r in cat['assets']),native_files=len(cat['assets']),
        native_time_span_s=(frames[-1]['native_filename_time_ns']-frames[0]['native_filename_time_ns'])/1e9,
        reference_clock_rtf=cat['observed_rtf'],capture_mode='Simulation Time / internal sensor sync 10Hz / external sync OFF',
        input_seals=seals,raw_manifest_sha256=cat['raw_manifest_sha256'],source_timestamps_modified=False,
        physical_simultaneity_verified=False,training_visible=False,
        decision=('Input preservation, teacher kinematics, full actor inventory, model identity, '
                  'measured API-box geometry, and route signal candidates tested; per-frame '
                  'physical exposure remains unapproved'),
        limitation='Weak image edges are not missing sensor data and are not evidence of misalignment. Successful cycle controls do not establish a millisecond bound.')
    a.output.mkdir(exist_ok=False)
    write_json(a.output/'report.json',report);write_json(a.output/'frame_ledger.json',frames)
    write_json(a.output/'COMPLETE.json',dict(files={n:digest(a.output/n) for n in ('report.json','frame_ledger.json')},
        script_sha256=digest(Path(__file__)),training_visible=False))
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
