"""Separate detection and regression on sealed bbox experiment caches (CPU)."""
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
from utils.evaluation.morai_detection_regression_audit import audit_frame


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    plan_bytes = (args.source / 'plan.json').read_bytes()
    plan = json.loads(plan_bytes)
    torch.set_num_threads(2)
    docs = {}; groups = defaultdict(Counter); metrics = defaultdict(lambda: defaultdict(list))
    detection = defaultdict(Counter); start = time.monotonic()
    with (args.output / 'objects.jsonl').open('w') as out, \
            (args.output / 'fit_geometry_candidates.jsonl').open('w') as geometry_queue, \
            (args.output / 'fit_detection_separation_review.jsonl').open('w') as review_queue:
        for i, row in enumerate(plan['samples']):
            path = row['manifest']
            if path not in docs:
                data = Path(path).read_bytes()
                assert hashlib.sha256(data).hexdigest() == row['manifest_sha256']
                docs[path] = {e['frame_id']: e for e in json.loads(data)['entries']}
            item = torch.load(args.source / 'cache' / ('%05d.pt' % row['old_index']),
                              map_location='cpu', weights_only=True)
            for k, expected in [('manifest_sha256', row['manifest_sha256']),
                                ('checkpoint_sha256', plan['checkpoint_sha256']),
                                ('frame_id', row['frame_id'])]:
                assert item[k] == expected, k
            result = audit_frame(item['baseline'], docs[path][row['frame_id']])
            if not result['supervised']:
                continue
            prefix = row['experiment_split'] + '/'
            keys = [prefix + 'all', prefix + row['category']]
            for key in keys:
                groups[key]['frames'] += 1
                for policy, counts in result['detection'].items():
                    detection[key + '/' + policy].update(counts)
            for obj in result['rows']:
                obj.update(native=row['native'], frame_id=row['frame_id'],
                           split=row['experiment_split'], category=row['category'],
                           deferred=obj['model'] in plan['deferred_models'])
                out.write(json.dumps(obj, allow_nan=False) + '\n')
                # Preparation only: uncertain identities are review cases, not
                # automatically generated positive regression targets.
                if obj['split'] == 'fit':
                    if obj['geometry'] is not None and not obj['deferred']:
                        geometry_queue.write(json.dumps(obj, allow_nan=False) + '\n')
                    elif obj['geometry'] is None and obj['reason'] != 'iou05_pass':
                        review_queue.write(json.dumps(obj, allow_nan=False) + '\n')
                subkeys = keys + [prefix + 'model/' + obj['model'],
                                 prefix + row['category'] + ('/partial' if obj['partial'] else '/full'),
                                 prefix + row['category'] + ('/nearby_gt' if obj['nearest_gt_center_m'] is not None
                                                            and obj['nearest_gt_center_m'] < 5. else '/isolated_5m')]
                for key in subkeys:
                    groups[key]['objects'] += 1
                    groups[key][obj['reason']] += 1
                    groups[key]['shared_box_flag'] += int(obj['shared_box_flag'])
                    groups[key]['recovered_without_nms'] += int(obj['recovered_without_nms'])
                    groups[key]['recovered_score02'] += int(obj['recovered_score02'])
                    if obj['geometry'] is not None:
                        groups[key]['geometry_pairs'] += 1
                        for metric, value in obj['geometry'].items():
                            metrics[key][metric].append(value)
            if (i + 1) % 200 == 0:
                print(json.dumps(dict(frames=i+1, total=len(plan['samples']),
                                      elapsed_s=round(time.monotonic()-start, 1))), flush=True)
    report = dict(status='COMPLETE', checkpoint_sha256=plan['checkpoint_sha256'],
                  source_plan_sha256=hashlib.sha256(plan_bytes).hexdigest(),
                  frames=len(plan['samples']), training_updates=0, model_forwards=0,
                  live_policy_modified=False, counts=dict(groups), detection=dict(detection),
                  geometry={k: {m: dict(n=len(v), mean=float(np.mean(v)), p95=float(np.percentile(v, 95)))
                                for m, v in values.items()} for k, values in metrics.items()},
                  limitations=[
                      'Frame-object counts, not independent actors or runs.',
                      'Mutual center association (2m gate, 0.5m margin) is a proxy, not certified identity.',
                      'Geometry is conditional on association coverage; compare on identical baseline pairs.',
                      'Shared-box coverage flags do not establish occlusion or a true merged detection.',
                      'Counterfactual thresholds are offline sensitivity checks, not deployment recommendations.',
                      'Previously inspected tune/diagnostic sets are not fresh blind tests.',
                      'Cached P1 outputs only; no temporal tracking claims.'],
                  elapsed_seconds=time.monotonic()-start)
    (args.output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(status='COMPLETE', report=str(args.output/'report.json'))), flush=True)


if __name__ == '__main__':
    main()
