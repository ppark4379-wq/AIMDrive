#!/usr/bin/env python3
"""Evaluate annotated frames only; does not compute MOT metrics or driving readiness."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from camera_mot.evaluation import evaluate_frame, aggregate


def key(row):
    return row['camera_id'], row['stream_epoch'], row['received_frame_id']


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--annotations',required=True)
    p.add_argument('--run-dir',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--kind',choices=('detections','tracks'),default='detections')
    p.add_argument('--score-threshold',type=float,default=.25)
    p.add_argument('--iou-threshold',type=float,default=.5)
    args=p.parse_args()
    if not 0 < args.iou_threshold <= 1 or not 0 <= args.score_threshold <= 1:
        p.error('Invalid IoU or score threshold')
    annotations=json.loads(Path(args.annotations).read_text())
    expected={key(row):row for row in annotations['frames']}
    if len(expected)!=len(annotations['frames']) or not expected:
        raise ValueError('Annotations must contain unique, nonempty frame keys')
    run=Path(args.run_dir)
    manifest=json.loads((run/'manifest.json').read_text())
    names={int(k):v for k,v in manifest['class_map'].items()}
    evaluated={}
    for line in (run/'tracks.jsonl').open():
        row=json.loads(line);frame_key=key(row)
        if frame_key not in expected:continue
        if frame_key in evaluated:raise ValueError('Duplicate prediction frame key')
        label=expected[frame_key]
        if row['timestamp'] != label['timestamp']:
            raise ValueError('Annotation timestamp differs from prediction input')
        predictions=[{**item,'class_name':names[int(item['class_id'])]}
                     for item in row[args.kind] if item['score']>=args.score_threshold]
        result=evaluate_frame(label['objects'],predictions,args.iou_threshold)
        evaluated[frame_key]={**result,'camera_id':row['camera_id'],
                             'received_frame_id':row['received_frame_id']}
    if set(evaluated)!=set(expected):
        raise ValueError('Some annotated frames have no prediction record; report input failure separately')
    report={'annotation_scope':annotations.get('scope','unspecified'),
        'annotated_frames':len(expected),'kind':args.kind,'score_threshold':args.score_threshold,
        'iou_threshold':args.iou_threshold,'run_session_id':manifest['session_id'],
        **aggregate(evaluated.values()),'frames':list(evaluated.values()),
        'limitations':'Only supplied labels are evaluated. No AP, HOTA, IDF1, ID-switch count or unseen-scene claim.'}
    with Path(args.output).open('x') as output:
        json.dump(report,output,indent=2,ensure_ascii=False,allow_nan=False)
    print(json.dumps(report['total']))


if __name__=='__main__':main()
