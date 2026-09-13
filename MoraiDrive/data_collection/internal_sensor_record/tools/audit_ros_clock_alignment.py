"""Audit real /clock observations against unchanged source/receipt clocks.

A preceding received /clock is a causal receipt witness, NOT exposure time.
No header replacement, simulated clock generation, offsets, or labels.
"""
import argparse
from bisect import bisect_right
from collections import Counter,defaultdict
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from async_capture.alignment import distribution
from async_capture.journal import digest,write_json
from ssd_storage import require_ssd


def summarize(rows,start,end):
    rows=sorted(rows,key=lambda r:(r['receive_monotonic_ns'],r['record_id']))
    clocks=[r for r in rows if r['role']=='clock' and r.get('signal') is not None]
    if len(clocks)<2:raise ValueError('Actual serialized /clock evidence missing')
    active=[r for r in clocks if start<=r['receive_monotonic_ns']<=end]
    if len(active)<2:raise ValueError('No progressing active clock evidence')
    ct=[r['receive_monotonic_ns'] for r in clocks]
    cv=[r['signal']['clock_ns'] for r in clocks]
    deltas=[b['signal']['clock_ns']-a['signal']['clock_ns'] for a,b in zip(active,active[1:])]
    # A duplicate does not prove clock progress. Measure the wall gap between
    # genuinely different values, in addition to ordinary receipt intervals.
    progress=[active[0]]
    for r in active[1:]:
        if r['signal']['clock_ns']!=progress[-1]['signal']['clock_ns']:progress.append(r)
    back=[dict(before_record=a['record_id'],after_record=b['record_id'],
               delta_ns=b['signal']['clock_ns']-a['signal']['clock_ns'])
          for a,b in zip(active,active[1:]) if b['signal']['clock_ns']<a['signal']['clock_ns']]
    groups=defaultdict(list)
    for r in rows:
        if start<=r['receive_monotonic_ns']<=end and r['role']!='clock':groups[r['stream']].append(r)
    stats={}
    for name,items in groups.items():
        source_delta=[];witness_age=[];control_clock=[];no_clock=0
        for r in items:
            i=bisect_right(ct,r['receive_monotonic_ns'])-1
            if i<0:no_clock+=1;continue
            witness_age.append((r['receive_monotonic_ns']-ct[i])/1e6)
            if r.get('source_timestamp_ns') is not None:
                source_delta.append((r['source_timestamp_ns']-cv[i])/1e6)
            control_clock.append(cv[i])
        stats[name]=dict(records=len(items),without_preceding_clock=no_clock,
            source_minus_preceding_received_clock_ms=distribution(source_delta),
            preceding_clock_receipt_age_ms=distribution(witness_age))
        if name=='expert_control' and len(control_clock)>1:
            stats[name]['clock_observed_at_control_receipt_delta_ms']=distribution(
                [(b-a)/1e6 for a,b in zip(control_clock,control_clock[1:])])
            sim=(active[-1]['signal']['clock_ns']-active[0]['signal']['clock_ns'])/1e9
            stats[name]['messages_per_observed_simulation_second']=len(items)/sim if sim>0 else None
    wall=(active[-1]['receive_monotonic_ns']-active[0]['receive_monotonic_ns'])/1e9
    sim=(active[-1]['signal']['clock_ns']-active[0]['signal']['clock_ns'])/1e9
    return dict(schema='morai_ros_clock_receipt_audit_v1',clock_records=len(active),
        clock_publishers=dict(Counter(r.get('caller_id') for r in active)),
        duplicates=sum(x==0 for x in deltas),backward_events=back,
        positive_clock_step_ms_counts=dict(Counter(str(x/1e6) for x in deltas if x>0)),
        clock_progress_wall_gap_ms=distribution([(b['receive_monotonic_ns']-a['receive_monotonic_ns'])/1e6 for a,b in zip(progress,progress[1:])]),
        observed_wall_seconds=wall,observed_simulation_seconds=sim,observed_rtf=sim/wall,
        streams=stats,source_timestamps_rewritten=False,training_visible=False,
        physical_simultaneity_verified=False,
        interpretation='Per-stream source minus preceding received /clock is a numeric domain/receipt diagnostic; NOT transport latency, exposure offset, or proof of a shared acquisition clock.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--quality-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();require_ssd(a.quality_dir,reserve_bytes=0);require_ssd(a.output)
    q=a.quality_dir;seal=json.loads((q/'QUALITY_COMPLETE.json').read_text())
    for f in ['INDEX_COMPLETE.json','active_signals.json']:
        if digest(q/f)!=seal['files'][f]:raise ValueError('Quality seal mismatch '+f)
    info=json.loads((q/'INDEX_COMPLETE.json').read_text())
    rows=json.loads((q/'active_signals.json').read_text())
    report=summarize(rows,info['active_start_ns'],info['active_end_ns'])
    report.update(raw_run_id=info['raw_run_id'],quality_seal_sha256=digest(q/'QUALITY_COMPLETE.json'))
    a.output.mkdir(parents=True,exist_ok=False)
    write_json(a.output/'report.json',report)
    write_json(a.output/'COMPLETE.json',dict(report_sha256=digest(a.output/'report.json'),
        script_sha256=digest(Path(__file__).resolve()),status='diagnostic_complete',training_visible=False))
    print(json.dumps({k:v for k,v in report.items() if k!='streams'},indent=2))
if __name__=='__main__':main()
