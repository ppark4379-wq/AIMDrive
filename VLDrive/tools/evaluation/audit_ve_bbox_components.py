"""Offline diagnostic attribution of fixed P1 matches; never a deployable oracle."""
import argparse
from collections import defaultdict
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS','1')
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
from utils.evaluation.morai_object_metrics import _ground_truth, _rectangle, decode_object_boxes, oriented_iou

COMPONENTS = ('center','size','heading')


def corner_error(a,b):
    return min(float(np.linalg.norm(np.roll(a['polygon'],k,axis=0)-b['polygon'],axis=1).mean()) for k in range(4))


def replaced(pred, truth, keys):
    return _rectangle(truth['center'] if 'center' in keys else pred['center'],
        truth['heading'] if 'heading' in keys else pred['heading'],
        truth['length'] if 'size' in keys else pred['length'],
        truth['width'] if 'size' in keys else pred['width'])


def decompose(pred, truth):
    errors={}
    for n in range(4):
        for keys in itertools.combinations(COMPONENTS,n):
            errors[frozenset(keys)]=corner_error(replaced(pred,truth,keys),truth)
    attribution=dict.fromkeys(COMPONENTS,0.)
    for order in itertools.permutations(COMPONENTS):
        previous=frozenset()
        for key in order:
            after=previous|{key}
            attribution[key]+=(errors[previous]-errors[after])/6.
            previous=after
    return dict(corner_m=errors[frozenset()],center_m=float(np.linalg.norm(pred['center']-truth['center'])),
        signed_right_m=float(pred['center'][0]-truth['center'][0]),
        signed_forward_m=float(pred['center'][1]-truth['center'][1]),
        length_bias_m=pred['length']-truth['length'],width_bias_m=pred['width']-truth['width'],
        heading_axis_error_deg=math.degrees(math.acos(float(np.clip(abs(np.dot(pred['heading'],truth['heading'])),0,1)))),
        iou=oriented_iou(pred,truth),
        excess_left_m=float(max(0,truth['bounds'][0]-pred['bounds'][0])),
        excess_right_m=float(max(0,pred['bounds'][1]-truth['bounds'][1])),
        missed_left_m=float(max(0,pred['bounds'][0]-truth['bounds'][0])),
        missed_right_m=float(max(0,truth['bounds'][1]-pred['bounds'][1])),
        corner_after_oracle={'+'.join(keys) or 'none':errors[frozenset(keys)]
            for n in range(4) for keys in itertools.combinations(COMPONENTS,n)},
        corner_shapley_m=attribution)


def decompose_footprint(pred, truth):
    """Attribute BEV shape error without mistaking equivalent axes for errors.

    Heading+90 degrees with length/width swapped describes the same rectangle.
    Align that representation to GT before replacing components. This uses GT
    only for offline attribution and never changes the predicted footprint.
    Near-square heading remains weakly meaningful; corner error is primary.
    """
    heading = np.asarray(pred['heading'])
    transverse = np.array([-heading[1], heading[0]])
    swap = abs(float(transverse @ truth['heading'])) > abs(float(heading @ truth['heading']))
    aligned = (_rectangle(pred['center'], transverse, pred['width'], pred['length'])
               if swap else pred)
    result = decompose(aligned, truth)
    if abs(result['corner_m'] - corner_error(pred, truth)) > 1e-7:
        raise ValueError('Axis alignment changed the rectangle footprint')
    result['axis_representation_swapped'] = bool(swap)
    result['original_length_bias_m'] = pred['length'] - truth['length']
    result['original_width_bias_m'] = pred['width'] - truth['width']
    result['size_only_corner_m'] = result['corner_after_oracle']['center+heading']
    result['heading_only_corner_m'] = result['corner_after_oracle']['center+size']
    return result


def summary(rows):
    def stats(v):
        return dict(mean=float(np.mean(v)),p95=float(np.percentile(v,95)))
    scalar=[k for k,v in rows[0]['metrics'].items() if isinstance(v,(float,int)) and not isinstance(v,bool)]
    return dict(pairs=len(rows),episodes=len(set(r['native'] for r in rows)),
        metrics={k:stats([r['metrics'][k] for r in rows]) for k in scalar},
        mean_corner_after_oracle={k:float(np.mean([r['metrics']['corner_after_oracle'][k] for r in rows]))
            for k in rows[0]['metrics']['corner_after_oracle']},
        mean_corner_shapley_m={k:float(np.mean([r['metrics']['corner_shapley_m'][k] for r in rows])) for k in COMPONENTS})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--splits',nargs='+',default=['train','val'])
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();torch.set_num_threads(1)
    plan=json.loads(args.plan.read_text());deferred=set(plan['deferred_models'])
    samples=[(i,r) for i,r in enumerate(plan['samples']) if r['split'] in args.splits]
    manifests={};groups=defaultdict(list);counts=defaultdict(lambda:defaultdict(int))
    with (args.output/'rows.jsonl').open('w') as out:
        for ordinal,(i,row) in enumerate(samples):
            path=row['manifest']
            if path not in manifests:
                raw=Path(path).read_bytes()
                if hashlib.sha256(raw).hexdigest()!=row['manifest_sha256']:
                    raise ValueError('Manifest changed: '+path)
                manifests[path]={e['frame_id']:e for e in json.loads(raw)['entries']}
            entry=manifests[path][row['frame_id']]
            truth,_=_ground_truth(entry)
            cache=torch.load(args.cache/('%05d.pt'%i),map_location='cpu',weights_only=True)
            pred,_=decode_object_boxes(cache['baseline'])
            c=counts[row['split']+'/'+row['category']]
            c['frames']+=1;c['gt']+=len(truth);c['detections']+=len(pred)
            pairs=[]
            if truth and pred:
                distance=np.array([[np.linalg.norm(t['center']-b['center']) for b in pred] for t in truth])
                a,b=linear_sum_assignment(np.where(distance<=2.,distance,1e5))
                pairs=[(int(j),int(k)) for j,k in zip(a,b) if distance[j,k]<=2.]
            c['matched']+=len(pairs);c['unmatched_gt']+=len(truth)-len(pairs)
            for gi,pi in pairs:
                t,b=truth[gi],pred[pi]
                r=dict(native=row['native'],frame_id=row['frame_id'],category=row['category'],split=row['split'],
                    model=t['model_name'],range_m=t['range_m'],partial=t['partial'],
                    deferred=t['model_name'] in deferred,cell_index=b['cell_index'],
                    pred={k:np.asarray(b[k]).tolist() for k in ('center','heading','length','width')},
                    truth={k:np.asarray(t[k]).tolist() for k in ('center','heading','length','width')},metrics=decompose(b,t))
                out.write(json.dumps(r,allow_nan=False)+'\n')
                prefix=row['split']+'/'
                keys=['all','deferred' if r['deferred'] else 'priority',row['category'],
                      'model/'+str(r['model']),'range/'+str(int(t['range_m']//10)*10)]
                if not r['deferred']:keys.append('priority/'+row['category'])
                for key in keys:groups[prefix+key].append(r)
            if (ordinal+1)%200==0:
                print(json.dumps(dict(frames=ordinal+1,total=len(samples),elapsed_s=round(time.monotonic()-started,1))),flush=True)
    report=dict(status='COMPLETE',frames=len(samples),checkpoint_sha256=plan['checkpoint_sha256'],
        plan_sha256=hashlib.sha256(args.plan.read_bytes()).hexdigest(),
        counts=dict(counts),groups={k:summary(v) for k,v in groups.items()},
        elapsed_seconds=time.monotonic()-started,training_updates=0,model_forwards=0,production_modified=False,
        limitations=['GT replacement is diagnostic upper bound, not an implementable correction.',
            'Fixed baseline Hungarian matches within 2m; unmatched objects are counted, not evaluated as geometry.',
            'Purpose-selected cached samples; correlated frames are not independent observations.',
            'Prior validation has already been inspected; this is not a fresh held-out certification.',
            'Component attribution can be negative when errors cancel; size and heading interact.',
            'Heading is rectangle-axis error modulo 180 degrees, not vehicle motion-heading error.'])
    (args.output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status='COMPLETE',report=str(args.output/'report.json')),ensure_ascii=False),flush=True)

if __name__=='__main__':main()
