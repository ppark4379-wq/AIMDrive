#!/usr/bin/env python3
"""Audit full gRPC actor inventory and its source-time coverage of Sensor Record."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from async_capture.actor_inventory_audit import parse_snapshot,audit_snapshots
from async_capture.alignment import PayloadReader
from async_capture.journal import digest,verify_run,write_json
from ssd_storage import require_ssd


def verified(folder, seal_name):
    seal=json.loads((folder/seal_name).read_text())
    for name,sha in seal['files'].items():
        if Path(name).name!=name or digest(folder/name)!=sha:raise ValueError('Changed sealed evidence: '+str(folder/name))
    return seal


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--episode',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gt-timeline-dir',default='gt_timeline_v1')
    a=p.parse_args()
    for path in (a.episode,a.output):require_ssd(path,reserve_bytes=0)
    qseal=verified(a.episode/'quality','QUALITY_COMPLETE.json')
    gt_dir=a.episode/a.gt_timeline_dir
    tseal=verified(gt_dir,'COMPLETE.json')
    index=json.loads((a.episode/'quality/INDEX_COMPLETE.json').read_text())
    if tseal['raw_manifest_sha256']!=index['raw_manifest_sha256']:
        raise ValueError('GT timeline and actor inventory use different raw runs')
    raw=Path(index['raw_path']);manifest,_=verify_run(raw,payloads=False)
    rows=json.loads((a.episode/'quality/active_signals.json').read_text())
    reader=PayloadReader(raw);snapshots=[];payload_failures=[];failed_rpc_records=[]
    identity_records=[];identity_failures=[];model_identities={}
    try:
        for meta in rows:
            if meta['role']!='actor_identity':continue
            payload=json.loads(reader.read(meta));identity_records.append(meta['record_id'])
            key=payload.get('instance_key');model=str(payload.get('model_name','')).strip()
            if payload.get('valid_nonempty_model_name') is not True or not key or not model:
                identity_failures.append(dict(record_id=meta['record_id'],instance_key=key,
                                              actor_id=payload.get('actor_id'),error=payload.get('error')))
                continue
            previous=model_identities.get(key)
            if previous is not None and previous.get('model_name')!=model:
                raise ValueError('Conflicting model identity evidence: '+str(key))
            model_identities[key]=payload
        for meta in rows:
            if meta['role']!='actor_inventory':continue
            payload=json.loads(reader.read(meta))
            parsed,errors=parse_snapshot(meta['record_id'],payload,
                                         model_identities=model_identities)
            payload_failures.extend(dict(record_id=meta['record_id'],reason=e) for e in errors)
            if parsed is None:failed_rpc_records.append(meta['record_id'])
            else:snapshots.append(parsed)
    finally:reader.close()
    frames=json.loads((gt_dir/'frames.json').read_text())
    audit=audit_snapshots(snapshots,[row['lidar_source_ns'] for row in frames])
    status=Counter('candidate' if row['transport_candidate'] else '+'.join(row['reasons'])
                   for row in audit.pop('frame_brackets'))
    frame_brackets=audit_snapshots(snapshots,[row['lidar_source_ns'] for row in frames])['frame_brackets']
    non_ego_events=[row for row in audit['lifecycle_events']
                    if row['actor_id']!='Ego' and row['kind']!='initial_presence']
    report=dict(schema='morai_sensor_record_actor_inventory_audit_v1',
        raw_manifest_sha256=index['raw_manifest_sha256'],raw_records=manifest['statistics']['records'],
        actor_payload_failures=payload_failures,failed_actor_rpc_records=failed_rpc_records,
        sensor_frame_status=dict(status),sensor_frames=len(frames),
        sensor_frames_transport_candidate=sum(row['transport_candidate'] for row in frame_brackets),
        sensor_frames_model_identity_candidate=sum(row['model_identity_candidate'] for row in frame_brackets),
        non_ego_lifecycle_events=len(non_ego_events),lifecycle_evidence_present=bool(non_ego_events),
        full_inventory_filter_verified=not any(x['reason']=='actor_filter_is_not_full_inventory' for x in payload_failures),
        actor_identity_records=len(identity_records),valid_actor_identity_records=len(model_identities),
        actor_identity_query_failures=identity_failures,
        transient_actor_rpc_failures_preserved=len(failed_rpc_records),
        transient_actor_rpc_failure_policy=(
            'Preserve failed polls and reject only sensor frames without a stable '
            'successful source-time bracket; do not convert missing snapshots to empty scenes.'),
        transport_gaps_masked_at_frame_level=True,
        object_geometry_training_valid=False,physical_sensor_gt_atomicity_verified=False,
        training_visible=False,limitation=('Full gRPC inventory, actor incarnation, and model-identity provenance audit only. '
            'Per-model box calibration and sensor/GT atomicity remain separate gates.'),
        **audit)
    a.output.mkdir(parents=True,exist_ok=False)
    write_json(a.output/'report.json',report);write_json(a.output/'frame_brackets.json',frame_brackets)
    write_json(a.output/'lifecycle_events.json',non_ego_events)
    write_json(a.output/'COMPLETE.json',dict(schema=report['schema'],files={name:digest(a.output/name) for name in
        ('report.json','frame_brackets.json','lifecycle_events.json')},quality_seal_sha256=digest(a.episode/'quality/QUALITY_COMPLETE.json'),
        gt_timeline_seal_sha256=digest(gt_dir/'COMPLETE.json'),script_sha256=digest(Path(__file__)),training_visible=False))
    print(json.dumps(report),flush=True)
    # Transient failed polls are evidence, not an empty actor snapshot.  The
    # source-time bracket audit above excludes every affected sensor frame.
    # Abort only when the remaining successful sequence is structurally
    # invalid; otherwise later geometry/admission stages consume the explicit
    # per-frame transport_candidate mask.
    if not report['structural_valid']:raise SystemExit(2)


if __name__=='__main__':main()
