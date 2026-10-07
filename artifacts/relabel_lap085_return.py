import copy
from collections import Counter
import json
from pathlib import Path
import shutil
from VLDrive.training.situation.relabel_common import DATA,read,write,lines,digest
from MoraiDrive.data_collection.situation_labeling.label_episode import verify_stage
from MoraiDrive.data_collection.situation_labeling.reviewed_overrides import LAP085,apply_reviewed_overrides

out=Path('artifacts/static_lap085_return_20261008').resolve()
out.mkdir(exist_ok=False)
identity='static_obstacle/'+LAP085+'/episode_000'
folder=DATA/identity/'situation_labels';verify_stage(folder)
original=[json.loads(s) for s in (folder/'labeled_frames.jsonl').read_text(encoding='utf-8').splitlines()]
updated=copy.deepcopy(original)
assert apply_reviewed_overrides([r['labeling'] for r in updated],folder.parent)==5
for old,new in zip(original,updated):
    new['instruction']=new['labeling'].get('instruction')
    if not 158<=old['frame_id']<=162:assert old==new
    assert {k:v for k,v in old.items() if k not in ('instruction','labeling')}=={k:v for k,v in new.items() if k not in ('instruction','labeling')}
assert all(r['instruction'] for r in updated if r.get('zone')=='static_obstacle')
assert 'Maintain the current driving behavior.' in updated[157]['instruction']
assert updated[158]['instruction']==updated[163]['instruction']=='Return to the original driving path while moving forward.'
target=out/'new'/identity/'situation_labels'
shutil.copytree(folder,out/'before'/identity/'situation_labels');shutil.copytree(folder,target)
lines(target/'labeled_frames.jsonl',updated);rules=[r['labeling'] for r in updated];lines(target/'features/situation_rules.jsonl',rules)
feature=read(target/'features/report.json');feature['reviewed_override_frames']=5
feature['instruction_consistency']=dict(Counter(r.get('instruction_consistency','null') for r in rules))
for key in ('instruction_state','instruction_id'):
    feature['rules'][key+'_counts']=dict(Counter(r[key] for r in rules if r.get(key)))
write(target/'features/report.json',feature)
info=read(target/'report.json');info.update(instruction_report=feature,reviewed_override='lap085_return_from_158',frame_counts=dict(labeled=sum(bool(r['instruction']) for r in updated),unlabeled=sum(not r['instruction'] for r in updated)))
write(target/'report.json',info)
seal=read(target/'COMPLETE.json');seal['files']={n:digest(target/n) for n in seal['files']};write(target/'COMPLETE.json',seal);verify_stage(target)
write(out/'report.json',dict(state='staged',data_root=str(DATA),episodes=[dict(identity=identity,old_sha256=digest(folder/'labeled_frames.jsonl'),new_sha256=digest(target/'labeled_frames.jsonl'))],validation=dict(violations=[],changed_frames=5,last_visible_frame=157,return_start_frame=158)))
print('Validated lap085: maintain through 157; return from 158; 5 gaps filled; all other rows unchanged.')
