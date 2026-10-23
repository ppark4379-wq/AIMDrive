"""Read-only localization metrics. Evaluation references never enter the estimator."""
import hashlib
import json
import math
from pathlib import Path
import statistics


def rows(path):
    if not Path(path).is_file():
        return []
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def statistics_of(values):
    values = sorted(float(value) for value in values)
    if not values:
        return dict(count=0, max=None, median=None, p95=None, mean=None)
    q = .95*(len(values)-1)
    i = int(q)
    return dict(count=len(values), max=values[-1], median=statistics.median(values),
                p95=values[i]+(q-i)*(values[min(i+1,len(values)-1)]-values[i]), mean=statistics.mean(values))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pose_rows(root):
    root = Path(root)
    path = root/'poses.jsonl'
    return rows(path if path.is_file() else root/'control.jsonl')


def gps_metadata(row):
    return ((row.get('selected_samples') or row.get('control_pose_selected_samples') or {}).get('gps') or {})


def error(real, estimate, yaw):
    dx, dy = estimate[0]-real[0], estimate[1]-real[1]
    return dict(error_2d_m=math.hypot(dx,dy), longitudinal_m=dx*math.cos(yaw)+dy*math.sin(yaw),
                lateral_m=-dx*math.sin(yaw)+dy*math.cos(yaw))


def finite(values):
    try:
        return all(math.isfinite(float(value)) for value in values)
    except (TypeError,ValueError,OverflowError):
        return False


def audit_integrity(records):
    last,seen={},{}
    result=dict(sample_ids_available=False,missing_sample_ids=0,duplicate_sample_ids=0,stream_epochs={})
    for record in sorted(records,key=lambda row:row['received_ns']):
        sensor=record.get('sensor')
        epoch=record.get('clock_epoch',0)
        key=(sensor,epoch)
        result['stream_epochs'].setdefault(sensor,[])
        if epoch not in result['stream_epochs'][sensor]:result['stream_epochs'][sensor].append(epoch)
        sample=record.get('sample_id')
        if not isinstance(sample,int) or isinstance(sample,bool):continue
        result['sample_ids_available']=True
        ids=seen.setdefault(key,set())
        if sample in ids:
            result['duplicate_sample_ids']+=1
            continue
        ids.add(sample)
        if key in last and sample>last[key]+1:result['missing_sample_ids']+=sample-last[key]-1
        last[key]=max(sample,last.get(key,sample))
    return result


def analyze(root):
    root = Path(root)
    seen, samples, diagnostics = set(), [], dict(duplicate_gps=0, invalid_comparisons=0)
    transitions, previous, modes = [], None, {}
    evaluation, eval_seen = [], set()
    references = {}
    audit = root/'sensor_audit.jsonl'
    input_records=rows(audit)
    for record in input_records:
        ref = record.get('evaluation_only', {})
        value = record.get('value', {})
        if record.get('sensor') == 'status' and ref.get('position_provided') and value.get('sim_time') is not None:
            references[round(value['sim_time']*1e9)] = ref['packet_position_xyz']
    for row in pose_rows(root):
        meta = gps_metadata(row)
        mode = meta.get('position_mode', 'unknown')
        modes[mode] = modes.get(mode,0)+1
        if previous != mode:
            transitions.append(dict(from_mode=previous, to_mode=mode,
                                    monotonic_ns=row.get('monotonic_ns', round(row.get('monotonic',0)*1e9))))
            previous = mode
        comparisons = meta.get('simulated_comparisons') or ([meta['simulated_comparison']] if meta.get('simulated_comparison') else [])
        for comparison in comparisons:
            receipt = comparison.get('gps_received_ns')
            if (not isinstance(receipt,int) or receipt <= 0 or comparison.get('dr_received_ns') != receipt
                    or comparison.get('time_basis') != 'receipt_monotonic'
                    or comparison.get('alignment_method') != 'interpolated_dr_history'):
                diagnostics['invalid_comparisons'] += 1
                continue
            if receipt in seen:
                diagnostics['duplicate_gps'] += 1
                continue
            real, estimate = comparison.get('real_gps_xy'), comparison.get('estimated_xy')
            yaw = comparison.get('yaw_grid_rad')
            if (not isinstance(real,list) or not isinstance(estimate,list) or len(real)!=2 or len(estimate)!=2
                    or not finite(real+estimate+[yaw])):
                diagnostics['invalid_comparisons'] += 1
                continue
            seen.add(receipt)
            samples.append(dict(comparison, **error(real,estimate,yaw)))
        pose, stamp = row.get('position'), meta.get('source_stamp_ns')
        # These positions are optional scoring-only data, looked up after pose output.
        if mode == 'dead_reckoning' and pose and stamp in references and stamp not in eval_seen:
            eval_seen.add(stamp)
            reference = references[stamp]
            evaluation.append(dict(source_stamp_ns=stamp, **error(reference,[pose['x'],pose['y']],pose['yaw_grid_rad'])))
    metrics = {key: statistics_of([abs(sample[key]) for sample in samples])
               for key in ('error_2d_m','longitudinal_m','lateral_m')}
    sources = {name:sha256(root/name) for name in ('plan.json','sensor_audit.jsonl','poses.jsonl','control.jsonl')
               if (root/name).is_file()}
    samples.sort(key=lambda sample:sample['gps_received_ns'])
    plan_path=root/'plan.json'
    plan=json.loads(plan_path.read_text()) if plan_path.is_file() else {}
    integrity=json.loads((root/'recording_status.json').read_text()) if (root/'recording_status.json').is_file() else {}
    return dict(schema='gps_jamming_localization_analysis_v1', source=str(root.resolve()),
        assessment='not_assessed_no_application_tolerance', scope='recorded_localization_only',
        recording_integrity=integrity,audit_integrity=audit_integrity(input_records),
        comparison_basis='GPS host receipt time; not acquisition-time ground truth',
        metrics=metrics, unique_gps_comparisons=len(samples), position_mode_frames=modes,
        transitions=transitions, diagnostics=diagnostics, samples=samples,
        last_observed_error=samples[-1] if samples else None,
        evaluation_reference=dict(used_by_estimator=False, source='exact status source stamp; optional scoring only',
            unique_samples=len(evaluation), error_2d_m=statistics_of([r['error_2d_m'] for r in evaluation])),
        source_sha256=sources,source_code_sha256=plan.get('source_code_sha256',plan.get('source_sha256',{})),
        input_provenance={key:plan[key] for key in ('method','source','scope','input_sha256','anchor_window_s','gps_hold_method','gps_anchor_window_s','max_hold_s','max_distance_m','replay_visibility') if key in plan},
        limitations=['GPS noise, NMEA resolution, unknown acquisition delay and antenna/reference-point differences remain.',
                     'Replay preserves recorded visibility batches when present and receipt causality; exact sub-call timings are not recorded. No new live trial validates the standalone package.',
                     'No application tolerance or automatic approval is inferred.'])
