"""Stage and apply ego-relative static position text from sealed geometry."""
import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import shutil

from .relabel_common import DATA, read, write, lines, digest, apply, bind_cache
from MoraiDrive.data_collection.situation_labeling.label_episode import verify_stage
from MoraiDrive.data_collection.situation_labeling.ego_position import update_static_positions, MAINTAIN_STATES
from MoraiDrive.data_collection.situation_labeling.side_view_hold import camera_views


def stage(output, baseline=None, pairs_only=False):
    output.mkdir(parents=True, exist_ok=False)
    report = dict(state='running', data_root=str(DATA), episodes=[])
    total = Counter()
    for path in sorted(DATA.glob('*/*/episode_*/situation_labels/labeled_frames.jsonl')):
        original = [json.loads(s) for s in path.read_text(encoding='utf-8-sig').splitlines()]
        if not any(r.get('zone') == 'static_obstacle' for r in original):
            continue
        verify_stage(path.parent)
        info = read(path.parent/'report.json')
        geometry_dir = path.parent.parent/info['object_geometry_dir']
        verify_stage(geometry_dir)
        geometry = read(geometry_dir/'frames.json')
        stamps = {r['frame_id']: r['sensor_time_ns'] for r in original}
        objects = []
        for frame in geometry:
            if frame['index'] not in stamps:
                continue
            if frame['lidar_source_ns'] != stamps[frame['index']]:
                raise ValueError('Geometry timestamp mismatch')
            for obj in frame['objects']:
                objects.append(dict(index=frame['index'], actor_id=obj.get('actor_id'),
                    instance_key=obj.get('instance_key'), forward_m=obj.get('center_forward_m'),
                    left_m=obj.get('center_left_m'),
                    actor_width_m=(obj.get('observed_size_wlh_m') or [None,None])[0],
                    actor_length_m=(obj.get('observed_size_wlh_m') or [None,None])[1]))
        identity = path.parent.parent.relative_to(DATA).as_posix()
        base_rows = original
        if baseline is not None:
            prior = read(baseline/'report.json')
            entry = next(e for e in prior['episodes'] if e['identity'] == identity)
            base_path = baseline/'before'/identity/'situation_labels/labeled_frames.jsonl'
            if digest(path) != entry['new_sha256'] or digest(base_path) != entry['old_sha256']:
                raise ValueError('Baseline/current revision mismatch')
            base_rows = [json.loads(s) for s in base_path.read_text(encoding='utf-8').splitlines()]
            assert len(base_rows) == len(original)
        updated = copy.deepcopy(base_rows)
        rules_to_update = [r['labeling'] for r in updated if not pairs_only or r['labeling'].get('instruction_state') == 'pass_between']
        counts = update_static_positions(rules_to_update, objects, cameras=None if pairs_only else camera_views(path.parent.parent))
        if counts.get('missing_geometry'):
            raise ValueError(f'Missing ego geometry: {path}: {counts}')
        changes = []
        for old, new, base in zip(original, updated, base_rows):
            new['instruction'] = new['labeling'].get('instruction')
            assert {k:v for k,v in old.items() if k not in ('labeling', 'instruction')} == {
                k:v for k,v in new.items() if k not in ('labeling', 'instruction')}
            for key in ('instruction_consistency',):
                assert old['labeling'].get(key) == new['labeling'].get(key)
            delayed = new['labeling'].get('evidence', {}).get('side_view_deferred_rule') or new['labeling'].get('evidence', {}).get('side_view_delayed_avoid_start')
            if pairs_only and old['labeling'].get('instruction_state') != 'pass_between':
                assert old == new
            if base['labeling'].get('instruction_state') not in MAINTAIN_STATES and not delayed and not (pairs_only and base['labeling'].get('instruction_state') == 'pass_between'):
                assert base['labeling'].get('instruction_id') == new['labeling'].get('instruction_id')
                assert base['instruction'] == new['labeling'].get('instruction')
            if not delayed:
                assert base['labeling'].get('variables', {}).get('avoid_direction') == new['labeling'].get('variables', {}).get('avoid_direction')
            assert bool(old['instruction']) == bool(new['instruction'])
            if old['instruction'] != new['instruction']:
                changes.append(dict(frame_id=old['frame_id'], before=old['instruction'], after=new['instruction']))
        identity = path.parent.parent.relative_to(DATA).as_posix()
        target = output/'new'/identity/'situation_labels'
        shutil.copytree(path.parent, output/'before'/identity/'situation_labels')
        shutil.copytree(path.parent, target)
        lines(target/'labeled_frames.jsonl', updated)
        lines(target/'features/situation_rules.jsonl', [r['labeling'] for r in updated])
        note = dict(reference='current_ego_geometry', target_policy='hold_descriptive_target_until_front_camera_exit', baseline_revision=str(baseline) if baseline else None, geometry_sha256=digest(geometry_dir/'frames.json'),
                    previous_labels_sha256=digest(path), counts=counts)
        if pairs_only:
            note['scope'] = 'pair positions individually classified; non-pair rows unchanged'
        feature = read(target/'features/report.json')
        feature['static_ego_position_revision'] = note
        rules = [r['labeling'] for r in updated]
        if 'rules' in feature:
            feature['rules']['instruction_state_counts'] = dict(Counter(r['instruction_state'] for r in rules if r.get('instruction_state')))
            feature['rules']['instruction_id_counts'] = dict(Counter(r['instruction_id'] for r in rules if r.get('instruction_id')))
        write(target/'features/report.json', feature)
        info.update(static_ego_position_revision=note, instruction_report=feature)
        write(target/'report.json', info)
        seal = read(target/'COMPLETE.json')
        seal['files'] = {n:digest(target/n) for n in seal['files']}
        write(target/'COMPLETE.json', seal)
        verify_stage(target)
        assert [json.loads(s) for s in (target/'labeled_frames.jsonl').read_text(encoding='utf-8').splitlines()] == updated
        total.update(counts)
        report['episodes'].append(dict(identity=identity, old_sha256=digest(path),
            new_sha256=digest(target/'labeled_frames.jsonl'), changes=changes, **note))
    report.update(state='staged', validation=dict(violations=[], episodes=len(report['episodes']), **total))
    write(output/'report.json', report)
    print(json.dumps(report['validation']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--bind-cache', action='store_true')
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--pairs-only', action='store_true')
    args = parser.parse_args()
    if args.apply or args.bind_cache:
        (apply if args.apply else bind_cache)(args.output.resolve())
    else:
        stage(args.output.resolve(), args.baseline.resolve() if args.baseline else None, args.pairs_only)
