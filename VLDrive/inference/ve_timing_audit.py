"""Summarize VE timing evidence without treating latest-frame skips as UDP loss."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import statistics


def rows(path):
    if not path.exists():
        return []
    result = []
    with path.open() as stream:
        for line in stream:
            try:
                result.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # an interrupted final write is not a measured frame
    return result


def distribution(values):
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    return dict(count=len(values), min=values[0], median=statistics.median(values),
                p95=values[int(.95*(len(values)-1))], max=values[-1])


def rate(stamps):
    if len(stamps) < 2 or stamps[-1] <= stamps[0]:
        return None
    intervals = [(b-a)/1e6 for a, b in zip(stamps, stamps[1:])]
    return dict(samples=len(stamps), seconds=(stamps[-1]-stamps[0])/1e9,
                hz=(len(stamps)-1)*1e9/(stamps[-1]-stamps[0]),
                interval_ms=distribution(intervals), gaps_over_200ms=sum(v>200 for v in intervals),
                gaps_over_500ms=sum(v>500 for v in intervals))


def summarize(root):
    root = Path(root)
    packets = rows(root/'adapter.jsonl')
    perf = rows(root/'performance.jsonl')
    transport = rows(root/'transport.jsonl')
    callbacks = rows(root/'expert_input.jsonl')
    # Older records lack delivery times. Their sent-time rate is still useful,
    # but does not prove when the Expert received or used a result.
    report = dict(schema='ve_input_timing_audit_v1', run=str(root.resolve()),
        accepted_outputs=len(packets), sender_output_rate=rate([p['sent_ns'] for p in packets]),
        ros_publish_rate=rate([p['delivery_audit']['published_ns'] for p in packets if 'delivery_audit' in p]),
        expert_callback_rate=rate([p['received_ns'] for p in callbacks]),
        legacy_pipeline_ms=distribution([p.get('model_inference_ms') for p in packets]),
        legacy_pipeline_scope='build + pose + model + output decode; not pure model time',
        legacy_receipt_to_send_ms=distribution([(p['sent_ns']-p['sensor_received_ns'])/1e6 for p in packets]),
        source_synchrony_verified=False, simulator_to_socket_loss_verified=False,
        all_inputs_lossless_verified=False,
        limitations=['UDP socket completion is not simulator capture time.',
            'Sensor sample IDs count locally reconstructed samples, not all simulator emissions.',
            'LiDAR sweep completion does not prove that every azimuth packet arrived.',
            'JPEG warnings are decoder warnings, not a count of unique damaged frames.',
            'ROS publish rate alone does not prove Expert receipt; callback audit is separate.'])
    stages = defaultdict(list)
    for p in packets:
        for key, value in p.get('audit', {}).get('stage_ms', {}).items():
            stages[key].append(value)
    report['stage_ms'] = {k: distribution(v) for k, v in stages.items()}
    report['stage_timing_scope'] = ('CUDA events include stream execution/idle time between events; '
        'host forward is asynchronous; decode includes pending GPU work. No extra CUDA synchronization.')
    report['udp_transit_ms'] = distribution([(p['delivery_audit']['received_ns']-p['sent_ns'])/1e6
                                            for p in packets if 'delivery_audit' in p])
    report['adapter_track_and_publish_ms'] = distribution([
        (p['delivery_audit']['published_ns']-p['delivery_audit']['received_ns'])/1e6
        for p in packets if 'delivery_audit' in p])
    report['sensor_receipt_skew_ms'] = distribution([p.get('audit', {}).get('receipt_skew_ms') for p in packets])
    report['oldest_selected_completion_to_send_ms'] = distribution([
        (p['sent_ns']-p['audit']['oldest_selected_received_ns'])/1e6 for p in packets
        if p.get('audit', {}).get('oldest_selected_received_ns')])
    # Source rates use actual elapsed monotonic time, not a presumed 2s log interval.
    streams = [p for p in perf if p.get('event') == 'streams']
    report['sensors'] = {}
    if len(streams) >= 2:
        a, b = streams[0], streams[-1]
        elapsed = (b['monotonic_ns']-a['monotonic_ns'])/1e9
        if elapsed > 0:
            for key, end in b['streams'].items():
                start = a['streams'].get(key, {})
                report['sensors'][key] = dict(window_seconds=elapsed,
                    completed_sample_hz=(end['samples']-start.get('samples', 0))/elapsed,
                    packet_hz=(end['packets']-start.get('packets', 0))/elapsed,
                    final_counters=end)
    report['input_selection'] = {}
    keys = {k for p in packets for k in p.get('audit', {}).get('selected_samples', {})}
    for key in sorted(keys):
        ids = [p['audit']['selected_samples'][key]['sample_id'] for p in packets
               if key in p.get('audit', {}).get('selected_samples', {})]
        report['input_selection'][key] = dict(
            completed_samples_not_selected_between_outputs=sum(max(0,b-a-1) for a,b in zip(ids,ids[1:])),
            reused_selected_sample=sum(a==b for a,b in zip(ids,ids[1:])),
            meaning='latest-only selection/reuse, not proof of network loss')
    unique_lidar = []
    last_selected_lidar = None
    for p in packets:
        identity = p.get('audit', {}).get('selected_samples', {}).get('lidar', {}).get('sample_id')
        if identity is not None and identity != last_selected_lidar:
            unique_lidar.append(p['sent_ns'])
        last_selected_lidar = identity
    report['distinct_selected_lidar_output_rate'] = rate(unique_lidar)
    report['lidar_gate_counter_differs_from_selected_sample'] = sum(
        p['audit'].get('gate_lidar_id') != p['audit']['selected_samples']['lidar']['sample_id']
        for p in packets if 'lidar' in p.get('audit', {}).get('selected_samples', {}))
    duplicate = [p for p in perf if p.get('event') == 'duplicate_lidar_gate']
    report['builds_skipped_by_existing_lidar_gate'] = dict(count=len(duplicate),
        build_ms=distribution([p['build_ms'] for p in duplicate]))
    sent = {p['datagram_sequence']: p for p in perf if p.get('event') == 'sent'}
    received = {p['datagram_sequence'] for p in transport if p.get('identity_matched')
                and isinstance(p.get('datagram_sequence'), int)}
    report['transport_rejections'] = dict(Counter(p.get('reason') or p['status']
                                                 for p in transport if p['status'] != 'published'))
    report['transport_delivery'] = None
    if sent and received:
        interior = {seq for seq in sent if min(received) <= seq <= max(received)}
        report['transport_delivery'] = dict(journaled_sends=len(sent), received_sequences=len(received),
            unreceived_within_observed_window=sorted(interior-received),
            unreceived_at_start_or_end=sorted(set(sent)-interior-received),
            note='Compare real datagram_sequence, never legacy loop sequence. Boundaries can be shutdown/startup.')
    by_stamp = defaultdict(list)
    for p in packets:
        if 'delivery_audit' in p:
            by_stamp[p['delivery_audit']['ros_stamp_ns']].append(p)
    matches = [(c, by_stamp[c['ros_stamp_ns']][0]) for c in callbacks
               if len(by_stamp.get(c['ros_stamp_ns'], [])) == 1]
    report['expert_delivery'] = None
    if matches:
        matched = {p['delivery_audit']['ros_stamp_ns'] for c,p in matches}
        times = [p['delivery_audit']['published_ns'] for c,p in matches]
        expected = {p['delivery_audit']['ros_stamp_ns'] for p in packets if 'delivery_audit' in p
                    and min(times) <= p['delivery_audit']['published_ns'] <= max(times)}
        report['expert_delivery'] = dict(callbacks=len(callbacks), unique_stamp_matches=len(matches),
            unmatchable_callbacks=len(callbacks)-len(matches),
            published_stamps_not_seen_inside_callback_window=len(expected-matched),
            udp_receipt_to_callback_ms=distribution([(c['received_ns']-p['delivery_audit']['received_ns'])/1e6 for c,p in matches]),
            oldest_sensor_completion_to_callback_ms=distribution([
                (c['received_ns']-p['audit']['oldest_selected_received_ns'])/1e6 for c,p in matches
                if p.get('audit', {}).get('oldest_selected_received_ns')]))
    log = (root/'publisher.log').read_text(errors='replace') if (root/'publisher.log').exists() else ''
    stopped = [p for p in rows(root/'publisher.log') if p.get('status') == 'STOPPED']
    report['model_forwards_at_stop'] = stopped[-1].get('model_forwards') if stopped else None
    report['jpeg_warning_occurrences'] = log.count('Corrupt JPEG data')
    report['sender_stopped_normally'] = any(p.get('event') == 'stopped' for p in perf)
    report['five_second_output_windows'] = []
    if packets:
        windows = defaultdict(list)
        for p in packets:
            offset = int((p['sent_ns']-packets[0]['sent_ns']) // 5_000_000_000) * 5
            windows[offset].append(p)
        for offset in range(0, max(windows)+1, 5):
            window = windows[offset]
            report['five_second_output_windows'].append(dict(offset_s=offset,
                output_rate=rate([p['sent_ns'] for p in window]),
                pipeline_ms=distribution([p.get('model_inference_ms') for p in window])))
    return report


def write_report(root):
    root = Path(root)
    path = root/'timing_report.json'
    temp = root/'timing_report.json.tmp'
    temp.write_text(json.dumps(summarize(root), indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    temp.replace(path)
    return path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_directory', type=Path)
    args = parser.parse_args()
    print(write_report(args.run_directory))
