#!/usr/bin/env python3
"""Local raw -> timing index + diagnostics. No NAS transfer or training data."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.runtime/mpl_cache'))

from async_capture.alignment import build_index,timing_report
from async_capture.journal import digest,write_json
from async_capture.scene_timing import extract_scene_signals,scene_report
from ssd_storage import require_ssd
from async_capture.clock_health import probe as probe_clocks
from async_capture.receipt_groups import build_receipt_groups


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-active-records',type=int,default=1000000)
    p.add_argument('--diagnostic-failed-run',action='store_true',help='Inspect sealed failed raw evidence; never approve training')
    a=p.parse_args()
    require_ssd(a.raw_run,reserve_bytes=0)
    require_ssd(a.output)
    started=time.monotonic()
    started_raw=time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
    software_files=[Path(__file__).resolve(),ROOT/'async_capture/alignment.py',ROOT/'async_capture/scene_timing.py',
                    ROOT/'async_capture/clock_health.py',ROOT/'async_capture/journal.py',ROOT/'async_capture/live.py',
                    ROOT/'async_capture/receipt_groups.py',ROOT/'async_capture/temporal_quality.py',
                    ROOT/'async_capture/realtime_pairing.py']
    software={str(f.relative_to(ROOT)):digest(f) for f in software_files}
    info,rows,session=build_index(a.raw_run,a.output,max_active_records=a.max_active_records,diagnostic_failed_run=a.diagnostic_failed_run)
    print(json.dumps(dict(stage='indexed',seconds=info['seconds'],records=info['records']),ensure_ascii=False),flush=True)
    required=('lidar','ego','gnss','imu')
    if session.get('sensor_contract',{}).get('capture',{}).get(
            'record_ros_cameras',True):
        required=('camera:Camera-1','camera:Camera-2','camera:Camera-3',
                  'camera:Camera-4')+required
    timing,anchors=timing_report(rows,info,required=required)
    timing['required_ros_streams']=list(required)
    write_json(a.output/'timing_report.json',timing)
    write_json(a.output/'lidar_associations.json',anchors)
    if session.get('time_mode_contract',{}).get('requested_time_mode')=='simulation_free_run':
        # RealTime receipt age limits are not Simulation Time scheduling rules.
        # The mode-specific build_temporal_groups audit evaluates that contract.
        receipt_summary=dict(status='not_applicable',reason='use_build_temporal_groups_for_internal10',
                             training_visible=False,physical_simultaneity_verified=False)
        receipt_groups,stream_quality,temporal_events=[],{},[]
    else:
        receipt_summary,receipt_groups,stream_quality,temporal_events=build_receipt_groups(
            rows,info['active_start_ns'],info['active_end_ns'])
    write_json(a.output/'bounded_receipt_summary.json',receipt_summary)
    write_json(a.output/'bounded_receipt_groups.json',receipt_groups)
    write_json(a.output/'stream_temporal_quality.json',stream_quality)
    write_json(a.output/'temporal_events.json',temporal_events)
    print(json.dumps(dict(stage='timing_checked',associations=timing['associations']),ensure_ascii=False),flush=True)
    phase=time.monotonic()
    signals,inputs=extract_scene_signals(a.raw_run,rows,info)
    report=scene_report(signals,rows,timing)
    report['inputs']=inputs
    write_json(a.output/'scene_signals.json',signals)
    write_json(a.output/'scene_report.json',report)
    write_json(a.output/'sensor_contract.json',session['sensor_contract'])
    clock_health=probe_clocks()
    write_json(a.output/'analysis_clock_health.json',clock_health)
    if any(digest(ROOT/name)!=sha for name,sha in software.items()):
        raise RuntimeError('Analysis source changed while running')
    require_ssd(a.output)
    files=['INDEX_COMPLETE.json','active_signals.json','record_index.jsonl','timing_report.json',
           'lidar_associations.json','scene_signals.json','scene_report.json','sensor_contract.json','analysis_clock_health.json',
           'bounded_receipt_summary.json','bounded_receipt_groups.json','stream_temporal_quality.json','temporal_events.json']
    final=dict(schema='morai_local_alignment_review_v1',raw_run_id=info['raw_run_id'],
        elapsed_monotonic_seconds=time.monotonic()-started,
        elapsed_monotonic_raw_seconds=(time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)-started_raw)/1e9,
        index_monotonic_seconds=info['seconds'],scene_and_clock_probe_monotonic_seconds=time.monotonic()-phase,
        analysis_software=software,
        files={name:digest(a.output/name) for name in files},
        status='diagnostic_complete',raw_status=info['raw_status'],diagnostic_failed_run=a.diagnostic_failed_run,training_visible=False,physical_simultaneity_verified=False,
        original_timestamps_preserved=True,nas_used=False,model_loaded=False)
    write_json(a.output/'QUALITY_COMPLETE.json',final)
    print(json.dumps(final,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
