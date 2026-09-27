"""Register sealed MORAI captures and create immutable Stage2 data versions.

Sensor payloads stay at their original location. Registration stores references,
not copies; no raw data or signed MORAI metadata is modified.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from utils.config.config_files import load_config

CATEGORIES = ('normal', 'static_object', 'dynamic_object', 'roundabout', 'highway')
SCENARIOS = {'full_loop_normal': 'normal', 'full_loop_static': 'static_object',
             'full_loop_dynamic': 'dynamic_object', 'full_loop_highway': 'highway',
             'full_loop_roundabout': 'roundabout',
             'full_loop_roundabout_nonuniform': 'roundabout',
             'full_loop_roundabout_speed_random': 'roundabout',
             'full_loop_roundabout_brake': 'roundabout'}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def component(value):
    if not re.fullmatch(r'[A-Za-z0-9_-]+', value):
        raise ValueError('Use a simple name without path separators')
    return value


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def register(source, report, category, name, tags=(), root=ROOT):
    source, report = Path(source).resolve(), Path(report).resolve()
    if not source.is_dir() or '.collecting' in source.parts:
        raise ValueError('Use a finalized campaign, never .collecting')
    if source not in report.parents:
        raise ValueError('Validation report must be inside this finalized campaign')
    document = read(report)
    seal = read(report.parent/'COMPLETE.json')
    if (document.get('schema') != 'morai_sensor_record_campaign_validation_v1'
            or document.get('valid') is not True
            or seal.get('status') != 'passed'
            or seal.get('files', {}).get(report.name) != digest(report)):
        raise ValueError('Requires an intact, sealed, fully passed campaign report')
    if SCENARIOS.get(document.get('scenario_name')) != category:
        raise ValueError('Category does not match the recorded scenario')
    episodes = sorted(source.rglob('episode.json'))
    if not episodes:
        raise ValueError('No episode.json found; not a supported Stage2 capture')
    for existing in (root/'data/raw').glob('*/*/registration.json'):
        if Path(read(existing)['source']).resolve() == source:
            raise ValueError('Campaign already registered: ' + str(existing))
    record = dict(schema='vldrive-raw-registration-v1', category=category,
                  tags=sorted(set(tags) | {category}), source=str(source),
                  validation_report=str(report), validation_sha256=digest(report),
                  seal_sha256=digest(report.parent/'COMPLETE.json'),
                  frames=document.get('total_frames'),
                  episodes=[dict(path=str(p.parent.relative_to(source)),
                                 episode_sha256=digest(p)) for p in episodes])
    target=root/'data/raw'/category/component(name)/'registration.json'
    write_new(target, record)
    return target


def version(config, name, root=ROOT):
    cfg=load_config(config)
    nav=load_config(config, 'navigation')
    vision=root/cfg['feature_cache']
    meta=read(vision/'cache_meta.json')
    labels=root/cfg['instructions']/'manifest.json'
    texts=root/cfg['navigation_sampling']['token_cache']/'token_meta.json'
    if meta['checkpoint_sha256'] != cfg['model']['visual_sha256']:
        raise ValueError('Visual cache and selected VE differ')
    if meta['labels_manifest_sha256'] != digest(labels):
        raise ValueError('Visual cache and navigation labels differ')
    token_meta=read(texts)
    if token_meta['visual_manifest_sha256'] != digest(vision/'manifest.jsonl'):
        raise ValueError('Text cache and visual cache differ')
    if token_meta['labels_manifest_sha256'] != digest(labels):
        raise ValueError('Text cache and navigation labels differ')
    files=[vision/'cache_meta.json',vision/'manifest.jsonl',vision/'features.jsonl',
           vision/'visual_source.json',vision/'split.json',labels,texts,
           texts.parent/'tokens.jsonl']
    if meta['manifest_sha256'] != digest(vision/'manifest.jsonl'):
        raise ValueError('Visual manifest changed')
    if token_meta['tokens_sha256'] != digest(texts.parent/'tokens.jsonl'):
        raise ValueError('Text token payload changed')
    target=root/'data/processed'/component(name)/'manifest.json'
    traffic=target.parent/'traffic_labels'
    if traffic.is_dir():
        files.extend(sorted(traffic.glob('*.json')))
    snapshot=target.parent/'processing_config.json'
    if snapshot.is_file():
        files.append(snapshot)
    for key, filename in (('features_manifest_sha256', 'features.jsonl'),
                          ('visual_source_sha256', 'visual_source.json')):
        if key in meta and meta[key] != digest(vision/filename):
            raise ValueError('Visual cache changed: ' + filename)
    sources=[dict(s, available=(root/s['root']).is_dir()) for s in nav['datasets']]
    record=dict(schema='vldrive-data-version-v1', name=component(name),
                config=cfg, navigation=nav, sources=sources,
                assets=dict(instructions=str(labels.parent.resolve()),
                            vision_tokens=str(vision.resolve()),
                            navigation_tokens=str(texts.parent.resolve()),
                            traffic_labels=str(traffic.resolve()) if traffic.is_dir() else None),
                fingerprints={str(p.resolve().relative_to(root.resolve())).replace('\\','/'):digest(p) for p in files},
                visual_frames=meta.get('visual_frames'), training_frames=meta.get('training_frames'),
                raw_available=all(s['available'] for s in sources),
                sequence_length=cfg['sequence_length'],
                frame_sampling_interval=cfg['frame_sampling_interval'])
    write_new(target, record)
    return target


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    add=commands.add_parser('register')
    add.add_argument('--source',type=Path,required=True)
    add.add_argument('--report',type=Path,required=True)
    add.add_argument('--category',choices=CATEGORIES,required=True)
    add.add_argument('--name',required=True)
    add.add_argument('--tag',action='append',default=[])
    bundle=commands.add_parser('version')
    bundle.add_argument('--config',type=Path,default=ROOT/'configs/vldrive_train.json')
    bundle.add_argument('--name',required=True)
    args=parser.parse_args()
    if args.command=='register':
        target=register(args.source,args.report,args.category,args.name,args.tag)
    else:
        target=version(args.config,args.name)
    print(target)


if __name__=='__main__':
    main()
