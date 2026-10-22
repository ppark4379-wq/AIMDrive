"""Bounded, GT-free proposal consensus experiment on sealed cached outputs."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils.evaluation.morai_object_metrics import _ground_truth, _match, decode_object_boxes
from utils.evaluation.morai_detection_regression_audit import reliable_center_pairs
from utils.evaluation.morai_proposal_consensus import consensus_boxes, POLICY


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/'report.json').exists():
        raise FileExistsError('Refusing to overwrite prior result')
    plan = json.loads((args.source/'plan.json').read_text())
    torch.set_num_threads(2); start = time.monotonic()
    docs = {}; counts = defaultdict(Counter); values = defaultdict(lambda: defaultdict(list))
    with (args.output/'paired_objects.jsonl').open('w') as out:
        for i, row in enumerate(plan['samples']):
            path = row['manifest']
            if path not in docs:
                data = Path(path).read_bytes()
                assert hashlib.sha256(data).hexdigest() == row['manifest_sha256']
                docs[path] = {e['frame_id']:e for e in json.loads(data)['entries']}
            item = torch.load(args.source/'cache'/('%05d.pt'%row['old_index']), map_location='cpu', weights_only=True)
            assert item['manifest_sha256'] == row['manifest_sha256']
            assert item['frame_id'] == row['frame_id']
            assert item['checkpoint_sha256'] == plan['checkpoint_sha256']
            entry = docs[path][row['frame_id']]
            if not entry.get('object_valid'): continue
            gt, _ = _ground_truth(entry); baseline, _ = decode_object_boxes(item['baseline'])
            candidate, modified = consensus_boxes(item['baseline'])
            assert len(candidate) == len(baseline)
            assert all(a['cell_index']==b['cell_index'] and a['score']==b['score'] for a,b in zip(baseline,candidate))
            keys = [row['experiment_split']+'/'+c for c in ['all', row['category']]]
            for key in keys:
                counts[key]['frames'] += 1; counts[key].update(modified)
                for name, boxes in [('baseline',baseline),('candidate',candidate)]:
                    for threshold in [.3,.5]:
                        tp = len(_match(boxes,gt,threshold)); tag = name+'/'+str(threshold)+'/'
                        counts[key][tag+'tp'] += tp
                        counts[key][tag+'fp'] += len(boxes)-tp
                        counts[key][tag+'fn'] += len(gt)-tp
            # Freeze both association and coverage to the original baseline.
            for gi, pi in reliable_center_pairs(gt,baseline).items():
                g = gt[gi]; errors = {}
                for name, p in [('baseline',baseline[pi]),('candidate',candidate[pi])]:
                    errors[name] = dict(center=float(np.linalg.norm(p['center']-g['center'])),
                                        corner=float(min(np.linalg.norm(np.roll(p['polygon'],s,axis=0)-g['polygon'],axis=1).mean() for s in range(4))))
                    for key in keys:
                        for metric, value in errors[name].items():
                            values[key][name+'/'+metric].append(value)
                out.write(json.dumps(dict(native=row['native'],frame_id=row['frame_id'],
                    split=row['experiment_split'],category=row['category'],model=g['model_name'],
                    cell=baseline[pi]['cell_index'],geometry_source_cell=candidate[pi].get('geometry_source_cell'),errors=errors))+'\n')
            if (i+1)%400==0:
                print(json.dumps(dict(frames=i+1,elapsed_s=round(time.monotonic()-start,1))),flush=True)
    geometry = {k:{m:dict(n=len(v),mean=float(np.mean(v)),p95=float(np.percentile(v,95)))
                   for m,v in group.items()} for k,group in values.items()}
    failures = []
    for key, c in counts.items():
        if key.startswith('fit/'): continue
        for threshold in [.3,.5]:
            for metric in ['tp','fp']:
                before=c['baseline/'+str(threshold)+'/'+metric]; after=c['candidate/'+str(threshold)+'/'+metric]
                if (metric=='tp' and after<before) or (metric=='fp' and after>before):
                    failures.append(key+'/'+str(threshold)+'/'+metric)
        for metric in ['center','corner']:
            g=geometry[key]
            if g['candidate/'+metric]['mean'] > g['baseline/'+metric]['mean'] + .01:
                failures.append(key+'/'+metric+'_regression')
        if key.endswith('/static') and geometry[key]['candidate/corner']['mean'] > .95*geometry[key]['baseline/corner']['mean']:
            failures.append(key+'/static_corner_gain_below_5pct')
    report = dict(status='COMPLETE',decision='KEEP_EXISTING_P1' if failures else 'OFFLINE_CANDIDATE_ONLY',
        failed_gates=failures,policy=POLICY,checkpoint_sha256=plan['checkpoint_sha256'],counts=dict(counts),
        geometry=geometry,elapsed_seconds=time.monotonic()-start,training_updates=0,model_forwards=0,
        production_modified=False,notes=['Fixed baseline center associations; same pairs before and after.',
        'Counts include deferred small models. No threshold tuning or GT inference inputs.',
        'Previously inspected evaluation episodes; not independent deployment certification.'])
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status='COMPLETE',decision=report['decision'],report=str(args.output/'report.json'))),flush=True)


if __name__=='__main__': main()
