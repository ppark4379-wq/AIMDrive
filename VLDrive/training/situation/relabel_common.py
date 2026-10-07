"""Shared verified publication and cache binding for static GT revisions."""
import hashlib
import json
from pathlib import Path
import shutil
from MoraiDrive.data_collection.situation_labeling.label_episode import verify_stage
REPO = Path(__file__).resolve().parents[3]
DATA = Path("C:/dataset_label").resolve()

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))

def write(p, value):
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

def lines(p, rows):
    p.write_text(''.join(json.dumps(r, ensure_ascii=False, allow_nan=False)+'\n' for r in rows), encoding='utf-8')

def apply(output):
    report = read(output/'report.json')
    if report['state'] != 'staged' or Path(report['data_root']).resolve() != DATA:
        raise ValueError('Not a staged revision for this dataset')
    if 'validation' not in report or report['validation']['violations']:
        raise ValueError('Validate this revision before applying')
    for e in report['episodes']:
        target = (DATA/e['identity']/'situation_labels').resolve()
        source = (output/'new'/e['identity']/'situation_labels').resolve()
        if DATA not in target.parents or output not in source.parents:
            raise ValueError('Unsafe revision path')
        if digest(target/'labeled_frames.jsonl') != e['old_sha256'] or digest(source/'labeled_frames.jsonl') != e['new_sha256']:
            raise ValueError('Labels changed since staging')
        verify_stage(source)
        backup = output/'before'/e['identity']/'situation_labels'
        for name in ['features/frame_features.jsonl','features/object_features.jsonl','features/situation_rules.jsonl',
                     'features/report.json','labeled_frames.jsonl','report.json','COMPLETE.json']:
            if digest(target/name) != digest(backup/name):
                raise ValueError('Original file changed since staging: '+str(target/name))
    for e in report['episodes']:
        target = DATA/e['identity']/'situation_labels'
        source = output/'new'/e['identity']/'situation_labels'
        names = ['features/frame_features.jsonl','features/object_features.jsonl','features/situation_rules.jsonl',
                 'features/report.json','labeled_frames.jsonl','report.json','COMPLETE.json']
        for name in names:
            temporary = (target/name).with_suffix('.relabel_tmp')
            shutil.copyfile(source/name, temporary)
            temporary.replace(target/name)
    report['state'] = 'applied'
    write(output/'report.json', report)

def bind_cache(output):
    """Bind unchanged VE inputs to verified new GT, preserving generation provenance."""
    report = read(output/'report.json')
    if report['state'] != 'applied' or report['validation']['violations']:
        raise ValueError('Apply a validated revision first')
    cache = REPO/'VLDrive/data/processed/situation_dataset_label_c5/vision_tokens'
    source, meta = read(cache/'source.json'), read(cache/'cache_meta.json')
    if digest(cache/'source.json') != meta['source_sha256'] or digest(cache/'manifest.jsonl') != meta['manifest_sha256']:
        raise ValueError('Cache provenance changed')
    for e in report['episodes']:
        if (digest(DATA/e['identity']/'situation_labels/labeled_frames.jsonl') != e['new_sha256']
                or source['episodes'][e['identity']]['labels_sha256'] != e['old_sha256']
                or meta['labels_sha256'][e['identity']] != e['old_sha256']):
            raise ValueError('Unexpected labels/cache revision')
    backup = output/'cache_before'
    backup.mkdir(exist_ok=False)
    for name in ('source.json', 'cache_meta.json'):
        shutil.copyfile(cache/name, backup/name)
    for e in report['episodes']:
        source['episodes'][e['identity']]['labels_sha256'] = e['new_sha256']
        meta['labels_sha256'][e['identity']] = e['new_sha256']
    write(cache/'source.relabel_tmp', source)
    (cache/'source.relabel_tmp').replace(cache/'source.json')
    meta['source_sha256'] = digest(cache/'source.json')
    meta['label_revision'] = dict(report=str(output/'report.json'),
                                  original_source_sha256=digest(backup/'source.json'),
                                  visual_features_recomputed=False,
                                  verification='all non-label frame fields identical; original labels archived')
    write(cache/'cache_meta.relabel_tmp', meta)
    (cache/'cache_meta.relabel_tmp').replace(cache/'cache_meta.json')
    report['cache_bound'] = True
    report['source_code_sha256'] = {str(p.relative_to(REPO)):digest(p) for p in
        [Path(__file__).resolve(), REPO/'MoraiDrive/data_collection/situation_labeling/consistency.py',
         REPO/'MoraiDrive/data_collection/situation_labeling/rules.py', REPO/'MoraiDrive/data_collection/situation_labeling/build_labels.py']}
    write(output/'report.json', report)
