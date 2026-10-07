"""Restore static obstacle-only instructions removed by longitudinal checks."""
import argparse
import copy
from collections import Counter
import json
from pathlib import Path
import shutil

from .relabel_common import DATA, read, write, lines, digest, apply, bind_cache
from MoraiDrive.data_collection.situation_labeling.label_episode import verify_stage

LONGITUDINAL = {'dropped_stationary', 'dropped_keep_going_while_stopped',
                'dropped_keep_going_while_stopping'}


def restore(row):
    rule = row['labeling']
    if row.get('zone') != 'static_obstacle' or rule.get('instruction_consistency') not in LONGITUDINAL:
        return False
    if rule.get('zone_id') != 'static_obstacle' or row.get('instruction') or rule.get('instruction'):
        raise ValueError('Unexpected dropped instruction state')
    text = rule.pop('instruction_dropped')
    if not isinstance(text, str) or not text.strip():
        raise ValueError('Missing original instruction')
    row['instruction'] = rule['instruction'] = text
    rule['instruction_consistency'] = 'consistent'
    return True


def stage(output):
    output.mkdir(parents=True, exist_ok=False)
    report = dict(state='running', data_root=str(DATA), episodes=[])
    total = Counter()
    for path in sorted((DATA/'static_obstacle').glob('*/episode_*/situation_labels/labeled_frames.jsonl')):
        verify_stage(path.parent)
        original = [json.loads(s) for s in path.read_text(encoding='utf-8-sig').splitlines()]
        updated = copy.deepcopy(original)
        counts = Counter()
        for old, new in zip(original, updated):
            if restore(new):
                counts[old['labeling']['instruction_consistency']] += 1
            else:
                assert old == new
            assert {k:v for k,v in old.items() if k not in ('labeling', 'instruction')} == {
                k:v for k,v in new.items() if k not in ('labeling', 'instruction')}
        identity = path.parent.parent.relative_to(DATA).as_posix()
        target = output/'new'/identity/'situation_labels'
        backup = output/'before'/identity/'situation_labels'
        shutil.copytree(path.parent, backup)
        shutil.copytree(path.parent, target)
        lines(target/'labeled_frames.jsonl', updated)
        lines(target/'features/situation_rules.jsonl', [r['labeling'] for r in updated])
        consistency = dict(Counter(r['labeling'].get('instruction_consistency', 'null') for r in updated))
        note = dict(scope='static obstacle handling only; independent of longitudinal constraints',
                    restored=dict(counts), previous_labels_sha256=digest(path))
        feature = read(target/'features/report.json')
        feature.update(instruction_consistency=consistency, static_revision=note)
        write(target/'features/report.json', feature)
        info = read(target/'report.json')
        info.update(static_revision=note, instruction_report=feature,
                    frame_counts=dict(labeled=sum(bool(r.get('instruction')) for r in updated),
                                      unlabeled=sum(not r.get('instruction') for r in updated)))
        write(target/'report.json', info)
        seal = read(target/'COMPLETE.json')
        seal['files'] = {n:digest(target/n) for n in seal['files']}
        write(target/'COMPLETE.json', seal)
        verify_stage(target)
        # Round-trip verification, including preservation of all other fields.
        actual = [json.loads(s) for s in (target/'labeled_frames.jsonl').read_text(encoding='utf-8').splitlines()]
        assert actual == updated
        assert all(r['instruction'] == r['labeling'].get('instruction') for r in actual)
        assert not any(r.get('zone') == 'static_obstacle' and
                       r['labeling'].get('instruction_consistency') in LONGITUDINAL for r in actual)
        total.update(counts)
        report['episodes'].append(dict(identity=identity, old_sha256=digest(path),
                                       new_sha256=digest(target/'labeled_frames.jsonl'), **note))
    report.update(state='staged', validation=dict(violations=[], restored=dict(total),
                  restored_frames=sum(total.values()), episodes=len(report['episodes'])))
    write(output/'report.json', report)
    print(json.dumps(report['validation']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--bind-cache', action='store_true')
    args = parser.parse_args()
    (apply if args.apply else bind_cache if args.bind_cache else stage)(args.output.resolve())
