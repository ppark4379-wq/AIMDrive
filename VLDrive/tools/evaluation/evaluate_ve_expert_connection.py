#!/usr/bin/env python3
"""Read-only recorded-input VE to actual Expert lattice connection diagnostic.

No ROS objects, control messages, optimizers, deployment, or expert future
waypoints are used. Both arms receive the same sealed official global path and
causal recorded ego speed. This is a planning snapshot, not closed-loop driving.
"""
import argparse
from collections import Counter, defaultdict
import datetime
import gc
import hashlib
import html
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCHEMA = 'morai_ve_expert_connection_plan_v1'
LIMITS = dict(closed_loop_verified=False, dynamic_future_verified=False,
              protected_left_verified=False, full_expert_stack_verified=False,
              road_boundary_verified=False, tracking_verified=False,
              traffic_light_stop_module_connected=False)


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            checksum.update(chunk)
    return checksum.hexdigest()


def write_json(path, value):
    path = Path(path)
    pending = path.with_suffix(path.suffix + '.partial')
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                  allow_nan=False) + '\n', encoding='utf-8')
    pending.replace(path)


def verify_file(record, resolver):
    path = resolver(record['path'])
    if not path.is_file() or digest(path) != record['sha256']:
        raise ValueError('Changed or missing pinned evidence: ' + str(path))
    return path


def validate_selection(sequence, document):
    """Require exact, increasing sealed row identifiers, never guessed indices."""
    indices, frame_ids = sequence['entry_indices'], sequence['frame_ids']
    if (not isinstance(indices, list) or not indices or len(indices) != len(frame_ids)
            or any(not isinstance(i, int) or isinstance(i, bool) for i in indices)
            or any(a >= b for a, b in zip(indices, indices[1:]))
            or indices[0] < 0 or indices[-1] >= len(document['entries'])):
        raise ValueError('Invalid chronological sequence selection')
    entries = [document['entries'][index] for index in indices]
    if [entry['frame_id'] for entry in entries] != frame_ids:
        raise ValueError('Selected frame identity differs from sealed manifest')
    if any(a['sensor_time_ns'] >= b['sensor_time_ns'] for a, b in zip(entries, entries[1:])):
        raise ValueError('Selected sequence sensor time does not advance')
    if any(entry.get('object_valid') is not True for entry in entries):
        raise ValueError('Planning comparison requires verified object supervision')
    return entries


def load_route(sequence, document, resolver):
    import numpy as np
    route = sequence['route']
    if (route.get('format') != 'whitespace_xyz'
            or route.get('coordinate_frame') != 'projected_grid'
            or route.get('xy_columns') != [0, 1]):
        raise ValueError('Unsupported official global path coordinate contract')
    path = verify_file(route, resolver)
    if document.get('navigation_contract', {}).get('global_path_sha256') != route['sha256']:
        raise ValueError('Official global path differs from sealed navigation provenance')
    points = np.loadtxt(str(path), dtype=np.float64)
    if (points.ndim != 2 or points.shape[1] != 3
            or len(points) != route['points_count'] or len(points) < 2
            or not np.isfinite(points).all()):
        raise ValueError('Invalid official global path geometry')
    return points[:, :2]


def preflight(plan_path, checkpoint_label=None):
    from data.src.loaders.morai_sensor_record_dataset import (
        PathResolver, SensorRecordDataset, load_publication)
    plan = json.loads(Path(plan_path).read_text(encoding='utf-8'))
    if plan.get('schema') != SCHEMA or plan.get('velocity_source') != 'ego_topic':
        raise ValueError('Unsupported connection plan or causal velocity source')
    if not plan.get('checkpoints') or not plan.get('sequences'):
        raise ValueError('Missing selected checkpoints or sequences')
    resolver = PathResolver(plan.get('path_mappings', []))
    for record in plan.get('evidence_files', []):
        verify_file(record, resolver)
    for name, expected in plan.get('implementation_sha256', {}).items():
        path = (ROOT / name).resolve()
        if ROOT not in path.parents or digest(path) != expected:
            raise ValueError('Pinned implementation changed: ' + name)
    checkpoints, labels = [], set()
    for record in plan['checkpoints']:
        label = record['label']
        if (not isinstance(label, str) or not label or Path(label).name != label
                or label in labels):
            raise ValueError('Invalid or duplicate checkpoint label')
        labels.add(label)
        selected = dict(record, path=str(verify_file(record, resolver)))
        if checkpoint_label is None or label == checkpoint_label:
            checkpoints.append(selected)
    if not checkpoints:
        raise ValueError('Selected checkpoint label is absent from plan')
    publications, admitted, identities, ids = {}, [], set(), set()
    for sequence in plan['sequences']:
        if (sequence.get('split') != 'val' or sequence['id'] in ids
                or not isinstance(sequence['id'], str) or not sequence['id']
                or Path(sequence['id']).name != sequence['id']):
            raise ValueError('Connection probe uses unique held-out validation sequences')
        ids.add(sequence['id'])
        manifest = verify_file(dict(path=sequence['manifest_path'],
                                    sha256=sequence['manifest_sha256']), resolver)
        index = verify_file(dict(path=sequence['manifest_index_path'],
                                 sha256=sequence['manifest_index_sha256']), resolver)
        if str(index) not in publications:
            publications[str(index)] = load_publication(index, resolver)
        publication, splits = publications[str(index)]
        matching = [(path, doc) for path, doc in splits['val'] if path == manifest]
        if len(matching) != 1:
            raise ValueError('Selected manifest is not an indexed held-out lap')
        document = matching[0][1]
        entries = validate_selection(sequence, document)
        dataset = SensorRecordDataset(matching, resolver, velocity_source='ego_topic')
        if dataset.map_basis != plan['map_basis']:
            raise ValueError('Dataset and planned checkpoint coordinate basis differ')
        row_lookup = {(entry['frame_id'], entry['sensor_time_ns']): index
                      for index, (_, entry, _, _) in enumerate(dataset.rows)}
        selected_rows = []
        for entry in entries:
            key = (entry['frame_id'], entry['sensor_time_ns'])
            if key not in row_lookup:
                raise ValueError('Selected frame was rejected by official evidence admission')
            identity = (str(manifest), key)
            if identity in identities:
                raise ValueError('Frame duplicated between evaluation sequences')
            identities.add(identity)
            selected_rows.append(row_lookup[key])
        route = load_route(sequence, document, resolver)
        admitted.append(dict(sequence=sequence, dataset=dataset, entries=entries,
                             row_indices=selected_rows, route=route,
                             publication_scenario=publication.get('scenario_name')))
    frames = sum(len(row['entries']) for row in admitted)
    if plan.get('expected_frames', frames) != frames:
        raise ValueError('Prepared frame count differs from exact selection')
    if frames > 1000:
        raise ValueError('Connection probe is bounded to at most 1000 selected frames')
    summary = dict(status='PASS', schema='morai_ve_expert_connection_preflight_v1',
                   frames=frames, sequences=len(admitted),
                   checkpoints=[row['label'] for row in checkpoints],
                   map_basis=plan['map_basis'], velocity_source='ego_topic',
                   official_evidence_admission=True, model_loaded=False,
                   maximum_sensor_forwards=frames * len(checkpoints),
                   training_updates=0, original_data_modified=False,
                   sequence_admission=[dict(id=row['sequence']['id'],
                       frames=len(row['entries']), total_admitted_lap_frames=len(row['dataset']),
                       excluded=dict(row['dataset'].excluded)) for row in admitted], **LIMITS)
    return plan, checkpoints, admitted, summary, resolver


def color_diagnostic(logits, entry):
    """Only color supervision is scored; permission is a separate diagnostic."""
    import numpy as np
    values = np.asarray(logits, dtype=np.float64)
    if values.shape != (2,) or not np.isfinite(values).all():
        raise ValueError('Expected two finite pure-color logits')
    scores = np.exp(values - values.max())
    probabilities = scores / scores.sum()
    predicted = int(probabilities.argmax())
    target = int(entry['traffic_light'])
    if target not in (-1, 0, 1):
        raise ValueError('Invalid sealed pure-color target')
    signal = entry.get('traffic_signal', {})
    permission = entry.get('route_permission', {})
    eligible = permission.get('valid') is True
    permission_target = permission.get('target') if eligible else None
    if eligible and not isinstance(permission_target, bool):
        raise ValueError('Malformed route-permission evidence')
    # This describes an inadequate color-only gate, never an Expert input.
    naive_go = predicted == 1
    return dict(predicted_class=predicted, target_class=target,
                probabilities_red_not_red=probabilities.tolist(),
                color_supervised=target >= 0,
                color_correct=predicted == target if target >= 0 else None,
                bound_gt_light_id=signal.get('light_id'),
                bound_gt_raw_signal_bits=signal.get('raw_signal_bits'),
                bound_gt_signal_movement=signal.get('signal_movement'),
                route_permission_gt=permission_target,
                color_only_counterfactual_go=naive_go,
                color_only_counterfactual_false_go=(naive_go and not permission_target) if eligible else None,
                color_only_counterfactual_false_stop=(not naive_go and permission_target) if eligible else None,
                predicted_light_id=None, predicted_route_permission=None,
                expert_signal_stop_evaluated=False,
                note='GT movement/ID/permission are diagnostic labels only, not VE planner inputs')


def summarize_colors(rows):
    observed = [row for row in rows if row['color_supervised']]
    permission = [row for row in rows if row['route_permission_gt'] is not None]
    confusion = [[0, 0], [0, 0]]
    for row in observed:
        confusion[row['target_class']][row['predicted_class']] += 1
    phases = {}
    groups = defaultdict(list)
    for row in rows:
        groups[str(row['bound_gt_raw_signal_bits'])].append(row)
    for key, group in sorted(groups.items()):
        scored = [row for row in group if row['color_supervised']]
        phases[key] = dict(frames=len(group), color_observed_frames=len(scored),
                          color_accuracy=(sum(row['color_correct'] for row in scored) / len(scored)) if scored else None)
    return dict(frames=len(rows), color_observed_frames=len(observed),
                masked_color_frames=len(rows) - len(observed),
                color_accuracy=(sum(row['color_correct'] for row in observed) / len(observed)) if observed else None,
                confusion_gt_rows_prediction_columns_red_not_red=confusion,
                by_gt_raw_signal_bits=phases,
                bound_route_permission_frames=len(permission),
                counterfactual_false_go_frames=sum(row['color_only_counterfactual_false_go'] for row in permission),
                counterfactual_false_stop_frames=sum(row['color_only_counterfactual_false_stop'] for row in permission),
                predicted_route_permission_available=False,
                interpretation='Counterfactual counts expose missing permission/arrow representation; they are not actual Expert-control errors')


def flatten_numeric(value, prefix=''):
    result = {}
    if isinstance(value, dict):
        for key, child in value.items():
            result.update(flatten_numeric(child, prefix + ('.' if prefix else '') + key))
    elif isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        result[prefix] = float(value)
    return result


def summarize_planning(rows):
    """Retain exact per-frame reports; mean numeric outputs are descriptive."""
    sums, counts, flags = defaultdict(float), Counter(), Counter()
    for row in rows:
        for key, value in flatten_numeric(row).items():
            sums[key] += value
            counts[key] += 1
        def visit(value, prefix=''):
            if isinstance(value, dict):
                for key, child in value.items():
                    visit(child, prefix + ('.' if prefix else '') + key)
            elif isinstance(value, bool):
                flags[prefix + ('.true' if value else '.false')] += 1
        visit(row)
    pairs = [row['comparison'] for row in rows if 'comparison' in row]
    rate = lambda key: sum(bool(row[key]) for row in pairs) / len(pairs) if pairs else None
    return dict(frames=len(rows), paired_geometry_comparison=dict(
                    compared_frames=len(pairs), decision_agreement_rate=rate('decision_agreement'),
                    obstacle_flag_agreement_rate=rate('obstacle_flag_agreement'),
                    gt_path_collision_frames=sum(bool(row['gt_path_collision']) for row in pairs),
                    ve_path_collision_frames=sum(bool(row['ve_path_collision']) for row in pairs),
                    ve_collision_gt_clear_frames=sum(bool(row['ve_path_collision']) and not bool(row['gt_path_collision']) for row in pairs),
                    both_paths_collide_frames=sum(bool(row['ve_path_collision']) and bool(row['gt_path_collision']) for row in pairs)),
                numeric_means={key: sums[key] / counts[key] for key in sorted(sums)},
                numeric_observations=dict(counts), boolean_counts=dict(flags),
                interpretation='Recorded-input current-geometry lattice snapshots; no road boundary, object motion, controller, or closed-loop certification')


def _review_svg(row):
    """Small standalone engineering overlay, no external dependencies."""
    ego = row['ego_pose']
    c, s = math.cos(ego['yaw']), math.sin(ego['yaw'])
    def xy(point):
        dx, dy = point[0] - ego['x'], point[1] - ego['y']
        forward, left = c * dx + s * dy, -s * dx + c * dy
        return 220 - left * 8, 380 - forward * 8
    shapes = ['<svg viewBox="0 0 440 440" role="img"><rect width="440" height="440" fill="#f4f6f8"/>',
              '<path d="M220 0V440 M0 380H440" stroke="#ddd"/>']
    planning = row['planning']
    for path, color, dash in ((planning.get('reference_path_world', []), '#777', '4 3'),
                             (planning.get('generic_gt', {}).get('path', []), '#2255aa', ''),
                             (planning.get('generic_ve', {}).get('path', []), '#c52f2f', '')):
        points = ' '.join('{:.1f},{:.1f}'.format(*xy(point)) for point in path)
        shapes.append('<polyline points="' + points + '" fill="none" stroke="' + color
                      + '" stroke-width="2" stroke-dasharray="' + dash + '"/>')
    for key, color in (('gt_world_boxes', '#2255aa'), ('ve_world_boxes', '#c52f2f')):
        for box in row[key]:
            angle = math.radians(box['heading_deg'])
            ac, ass = math.cos(angle), math.sin(angle)
            center = box['center_world']
            corners = [xy([center[0] + ac * along - ass * across,
                           center[1] + ass * along + ac * across])
                       for along, across in ((box['length_m']/2, box['width_m']/2),
                           (box['length_m']/2, -box['width_m']/2),
                           (-box['length_m']/2, -box['width_m']/2),
                           (-box['length_m']/2, box['width_m']/2))]
            points = ' '.join('{:.1f},{:.1f}'.format(*p) for p in corners)
            shapes.append('<polygon points="' + points + '" fill="none" stroke="' + color + '" stroke-width="1.5"/>')
    shapes.append('<circle cx="220" cy="380" r="4" fill="#222"/><text x="8" y="20">GT blue / VE red; ego forward up, 8 px/m</text></svg>')
    return ''.join(shapes)


def write_review(output, reviews):
    sections = []
    for row in reviews:
        camera = ('<img style="width:440px;max-width:100%" src="' + html.escape(row['camera_review'])
                  + '" alt="sealed source front camera">') if row.get('camera_review') else ''
        sections.append('<article><h2>' + html.escape(row['checkpoint'] + ' / ' + row['sequence_id'] + ' / frame ' + str(row['frame_id']))
                        + '</h2>' + camera + _review_svg(row)
                        + '<pre>' + html.escape(json.dumps(dict(signal=row['signal'], planning=row['planning']),
                                                           ensure_ascii=False, indent=2)) + '</pre></article>')
    (output / 'review.html').write_text('<!doctype html><meta charset="utf-8"><title>VE Expert connection</title>'
        + '<style>body{font:15px sans-serif;margin:24px}article{border-top:1px solid #ccc;padding:20px 0}svg{width:440px;max-width:100%}pre{white-space:pre-wrap}</style>'
        + '<h1>Recorded-input VE / Expert lattice comparison</h1><p>Current geometry only. No closed-loop or protected-left certification. GT labels are never injected into VE inputs.</p>'
        + ''.join(sections), encoding='utf-8')


def run(args):
    started = time.monotonic()
    print(json.dumps(dict(phase='verify_sealed_metadata', training_updates=0, model_loaded=False)), flush=True)
    plan, checkpoints, admitted, summary, resolver = preflight(args.plan, args.checkpoint_label)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    if args.check:
        return summary
    import torch
    from utils.evaluation.morai_transfer_evaluation import _load_model
    from utils.evaluation.morai_object_metrics import ObjectMetricsAccumulator
    from utils.evaluation.morai_ve_expert_adapter import (
        decode_world_predictions, world_ground_truth, verify_adapter_roundtrip)
    from tools.evaluation.ve_expert_planning_engine import evaluate_planning_pair
    torch.set_num_threads(args.threads)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; full CPU evaluation is never selected silently')
    if args.smoke:
        checkpoints = checkpoints[:1]
        admitted = [dict(admitted[0], entries=admitted[0]['entries'][:1], row_indices=admitted[0]['row_indices'][:1])]
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output = (args.output or args.plan.parent / 'runs' / ('run_' + stamp + '_' + str(time.time_ns() % 1000000))).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'plan.json', plan)
    write_json(output / 'preflight.json', summary)
    forwards, results, reviews = 0, [], []
    deadline = started + args.max_seconds
    try:
        for selected in checkpoints:
            model, payload, checkpoint_hash = _load_model(Path(selected['path']), selected['sha256'], args.device)
            contract = payload.get('morai_encoder_contract', {})
            if contract.get('map_basis') != plan['map_basis'] or contract.get('velocity_input') != 'causal_ego_topic_speed_mps_50ms':
                raise ValueError('Loaded checkpoint coordinate/velocity contract differs from sealed inputs')
            metadata = dict(epoch=payload.get('epoch'), steps=payload.get('steps'))
            del payload
            metrics, by_category, signals, planned = ObjectMetricsAccumulator(), {}, [], []
            frame_count, sequence_reports = 0, []
            prediction_path = output / (selected['label'] + '_frames.jsonl')
            with prediction_path.open('x', encoding='utf-8') as stream, torch.inference_mode():
                for sequence in admitted:
                    case, dataset = sequence['sequence'], sequence['dataset']
                    sequence_metrics, sequence_signals, sequence_planned = ObjectMetricsAccumulator(), [], []
                    for index, (entry, dataset_index) in enumerate(zip(sequence['entries'], sequence['row_indices'])):
                        if time.monotonic() >= deadline:
                            raise TimeoutError('Bounded connection evaluation exceeded --max-seconds')
                        roundtrip = verify_adapter_roundtrip(entry)
                        inputs, targets = dataset[dataset_index]
                        if int(targets['frame_id']) != entry['frame_id'] or int(targets['sensor_time_ns']) != entry['sensor_time_ns']:
                            raise ValueError('Official sensor input frame identity mismatch')
                        batch = {key: value.unsqueeze(0).to(args.device) for key, value in inputs.items()}
                        forward_started = time.monotonic()
                        predictions = model(batch)
                        if args.device == 'cuda':
                            torch.cuda.synchronize()
                        forward_seconds = time.monotonic() - forward_started
                        forwards += 1
                        if not isinstance(predictions, (list, tuple)) or len(predictions) < 3:
                            raise ValueError('Unexpected visual encoder output contract')
                        detection = predictions[0][0].detach().cpu()
                        if detection.shape != (2500, 8) or not bool(torch.isfinite(detection).all()):
                            raise ValueError('Invalid raw VE detection shape or nonfinite outputs')
                        decoded = decode_world_predictions(detection, entry['ego_pose'])
                        truth = world_ground_truth(entry)
                        signal = color_diagnostic(predictions[2][0].detach().cpu().numpy(), entry)
                        speed = float(inputs['velocity'].reshape(-1)[0])
                        frame = dict(ego_pose={key: float(entry['ego_pose'][key]) for key in ('x','y','yaw')},
                                     speed_mps=speed, official_reference_path=sequence['route'],
                                     reference_path_source='official_global_path',
                                     official_reference_closed=case['route']['is_closed'],
                                     reference_path_start_index=entry['navigation']['route_index'],
                                     reference_path_index_source='official_route_index',
                                     gt_world_boxes=truth['boxes'], ve_world_boxes=decoded['boxes'])
                        frame['official_reference_path'] = frame['official_reference_path'].tolist()
                        planning = evaluate_planning_pair(frame, planner_parameters=plan.get('planner_parameters'))
                        object_frame = metrics.update(detection, entry)
                        sequence_metrics.update(detection, entry)
                        by_category.setdefault(case['category'], ObjectMetricsAccumulator()).update(detection, entry)
                        row = dict(checkpoint=selected['label'], sequence_id=case['id'], category=case['category'],
                                   manifest=case['manifest_path'], manifest_sha256=case['manifest_sha256'],
                                   frame_id=entry['frame_id'], sensor_time_ns=entry['sensor_time_ns'],
                                   ego_pose=frame['ego_pose'], input_velocity_mps=speed,
                                   case_tags=case.get('case_tags', {}).get(str(case['entry_indices'][index]), []),
                                   gt_world_boxes=truth['boxes'], gt_outside_bev=truth['outside_bev'],
                                   ve_world_boxes=decoded['boxes'], decode_rejections=decoded['rejections'],
                                   adapter_roundtrip=roundtrip, object_metrics=object_frame,
                                   signal=signal, planning=planning, model_forward_seconds=forward_seconds,
                                   gt_future_waypoints_used_by_planner=False,
                                   ve_gt_identity_class_speed_injected=False)
                        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
                        stream.flush()
                        signals.append(signal); planned.append(planning)
                        sequence_signals.append(signal); sequence_planned.append(planning)
                        if index in (0, len(sequence['entries']) // 2, len(sequence['entries']) - 1):
                            # Small read-only copies make the offline comparison reviewable.
                            from PIL import Image
                            import io
                            camera_dir = output / 'camera'
                            camera_dir.mkdir(exist_ok=True)
                            camera_name = case['id'] + '_frame_' + str(entry['frame_id']) + '.jpg'
                            camera_path = camera_dir / camera_name
                            if not camera_path.exists():
                                native = dataset.rows[dataset_index][0]
                                raw = dataset.read_asset(native, entry, entry['asset_roles']['camera_1'])
                                with Image.open(io.BytesIO(raw)) as camera:
                                    camera = camera.convert('RGB')
                                    camera.thumbnail((640, 360))
                                    camera.save(camera_path, format='JPEG', quality=86)
                            row['camera_review'] = 'camera/' + camera_name
                            reviews.append(row)
                        frame_count += 1
                        if frame_count % 10 == 0 or args.smoke:
                            print(json.dumps(dict(phase='recorded_planning_snapshot', checkpoint=selected['label'],
                                                  sequence=case['id'], frames=frame_count, model_forwards=forwards,
                                                  elapsed_seconds=round(time.monotonic() - started, 1)), ensure_ascii=False), flush=True)
                        del predictions, batch
                    sequence_reports.append(dict(id=case['id'], category=case['category'], frames=len(sequence['entries']),
                        object_metrics=sequence_metrics.compute(), signal_metrics=summarize_colors(sequence_signals),
                        planning_metrics=summarize_planning(sequence_planned)))
            result = dict(label=selected['label'], checkpoint=selected['path'], checkpoint_sha256=checkpoint_hash,
                          checkpoint_metadata=metadata, frames=frame_count, model_forwards=frame_count,
                          object_metrics=metrics.compute(), by_category={key: value.compute() for key,value in by_category.items()},
                          signal_metrics=summarize_colors(signals), planning_metrics=summarize_planning(planned),
                          sequences=sequence_reports, per_frame_jsonl=str(prediction_path),
                          per_frame_sha256=digest(prediction_path))
            write_json(output / (selected['label'] + '_report.json'), result)
            results.append(result)
            del model
            gc.collect()
            if args.device == 'cuda':
                torch.cuda.empty_cache()
        # Re-check pinned source metadata so mutations by concurrent jobs fail visibly.
        for record in plan.get('evidence_files', []):
            verify_file(record, resolver)
        write_review(output, reviews)
        report = dict(schema='morai_ve_expert_connection_result_v1', status='COMPLETE',
                      plan=str(args.plan), plan_sha256=digest(args.plan),
                      scope='one_frame_smoke' if args.smoke else 'selected_chronological_validation_sequences',
                      frames_per_checkpoint=results[0]['frames'], results=results, model_forwards=forwards,
                      sensor_bundles_verified=forwards, training_updates=0, weights_updated=False,
                      original_data_modified=False, ros_connected=False, ctrl_cmd_published=False,
                      gt_future_waypoints_used_by_planner=False, ve_gt_identity_class_speed_injected=False,
                      review=str(output / 'review.html'), elapsed_seconds=time.monotonic() - started,
                      execution=dict(device=args.device, main_torch_threads=args.threads, batch_size=1),
                      automatic_training_or_deployment=False, **LIMITS)
        report['geometry_coverage'] = 'Only GT and VE boxes with positive-area intersection in the encoder BEV are compared; objects outside that BEV are not treated as visible planner ground truth'
        report['implementation_sha256'] = {str(path.relative_to(ROOT)): digest(path) for path in (
            Path(__file__), ROOT / 'utils/evaluation/morai_ve_expert_adapter.py',
            ROOT / 'tools/evaluation/ve_expert_planning_engine.py',
            ROOT / 'utils/evaluation/morai_object_metrics.py',
            ROOT / 'data/src/loaders/morai_sensor_record_dataset.py')}
        write_json(output / 'report.json', report)
        print(json.dumps(dict(status='COMPLETE', report=str(output / 'report.json'), review=report['review'],
                              model_forwards=forwards, training_updates=0,
                              elapsed_seconds=report['elapsed_seconds']), ensure_ascii=False), flush=True)
        return report
    except BaseException as error:
        write_json(output / 'failure.json', dict(status='FAILED', error=repr(error),
                   completed_checkpoints=[row['label'] for row in results], model_forwards=forwards,
                   training_updates=0, original_data_modified=False, **LIMITS))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--check', action='store_true', help='Official metadata admission and source hashes; no model loaded')
    parser.add_argument('--smoke', action='store_true', help='Exactly one selected frame and the first selected checkpoint')
    parser.add_argument('--checkpoint-label', help='Use one checkpoint explicitly named in the sealed plan')
    parser.add_argument('--device', choices=('cpu','cuda'), default='cuda')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--max-seconds', type=float, default=1800.)
    args = parser.parse_args()
    if args.threads < 1 or not math.isfinite(args.max_seconds) or args.max_seconds <= 0:
        parser.error('Invalid main thread count or time limit')
    if args.check and args.smoke:
        parser.error('--check and --smoke are mutually exclusive')
    args.plan = args.plan.expanduser().resolve()
    run(args)


if __name__ == '__main__':
    main()
