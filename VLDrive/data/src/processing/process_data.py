"""Build a complete Stage2 dataset inside a new processed/<id> directory."""
import argparse
import copy
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.src.processing.manage_data import component, digest, read, version
from utils.config.config_files import load_config

KINDS = ('instructions', 'navigation_tokens', 'traffic_labels', 'vision_tokens')


def write(path, value):
    pending = path.with_suffix(path.suffix + '.pending')
    pending.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    pending.replace(path)


def prepare_config(source, name):
    cfg = copy.deepcopy(source)
    cfg.pop('data_version', None)
    base = 'data/processed/' + component(name)
    cfg['instructions'] = base + '/instructions'
    cfg['feature_cache'] = base + '/vision_tokens'
    cfg['split_file'] = base + '/vision_tokens/split.json'
    cfg['navigation_sampling']['token_cache'] = base + '/navigation_tokens'
    cfg['preprocessing']['vision']['contract_sidecar'] = base + '/vision_tokens/encoder_contract.json'
    return cfg


def process(config, name=None, activate=True, root=ROOT, run=subprocess.run):
    config = Path(config).resolve()
    original = read(config)
    original_digest = digest(config)
    name = component(name or datetime.now().strftime('stage2_%Y%m%d_%H%M%S_%f'))
    cfg = prepare_config(original, name)
    preparation = cfg['preprocessing']
    if not preparation['datasets']:
        raise ValueError('No preprocessing datasets configured')
    # Fail before allocating a version or loading the GPU if inputs are absent.
    for source in preparation['datasets']:
        if not (root / source['root']).is_dir():
            raise FileNotFoundError('Raw dataset missing: ' + source['root'])
    for key in ('visual_checkpoint', 'language_checkpoint'):
        if not (root / cfg['model'][key]).exists():
            raise FileNotFoundError('Model input missing: ' + cfg['model'][key])
    jobs = preparation.get('traffic_labels', {}).get('jobs', [])
    names = [component(job['name']) for job in jobs]
    if len(set(names)) != len(names):
        raise ValueError('Traffic-label job names must be unique')
    folder = root / 'data/processed' / name
    folder.mkdir(parents=True, exist_ok=False)
    for kind in KINDS:
        (folder / kind).mkdir()
    snapshot = folder / 'processing_config.json'
    write(snapshot, cfg)
    state = dict(schema='vldrive-processing-status-v1', status='running', completed=[])
    write(folder / 'status.json', state)
    script_root = root / 'data/src/processing'

    def execute(script, *args):
        run([sys.executable, str(script_root / script), *map(str, args)], cwd=root, check=True)
        state['completed'].append(script)
        write(folder / 'status.json', state)

    try:
        # Resolve registrations and verify seals before starting any processing.
        load_config(snapshot, 'navigation')
        execute('build_instructions.py', '--config', snapshot)
        for job in jobs:
            label = folder / 'traffic_labels' / (job['name'] + '.json')
            execute('build_traffic_labels.py', '--dataset-root', root / job['dataset_root'],
                    '--split-file', root / job['split_file'], '--output', label,
                    '--report', label.with_name(job['name'] + '_report.json'))
        write(folder / 'traffic_labels/status.json', dict(
            status='complete' if jobs else 'not_requested', jobs=names,
            reason=None if jobs else 'Stage2 navigation/waypoint training does not consume traffic auxiliary labels.'))
        execute('cache_velocity_tokens.py', '--config', snapshot)
        execute('cache_navigation_text_tokens.py', '--config', snapshot)
        manifest = version(snapshot, name, root=root)
        state['status'] = 'complete'
        write(folder / 'status.json', state)
    except BaseException as error:
        state.update(status='failed', error=str(error))
        write(folder / 'status.json', state)
        raise
    if activate:
        if digest(config) != original_digest:
            raise RuntimeError('Processing completed, but training config changed; select the new manifest manually.')
        original['data_version'] = str(manifest.relative_to(root)).replace('\\', '/')
        write(config, original)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'training/vldrive_train.json')
    parser.add_argument('--id', help='Optional unique version ID; defaults to a timestamp')
    parser.add_argument('--no-activate', action='store_true', help='Do not select the result for training')
    parser.add_argument('--prepare-only', action='store_true', help='Prepare joint GP/targets without running the VE')
    parser.add_argument('--resume', action='store_true', help='Continue the unfinished joint version selected by --id')
    parser.add_argument('--reuse-version', help='Reuse compatible prepared episodes/VE records from a completed joint version')
    args = parser.parse_args()
    source = read(args.config)
    if source.get('preprocessing', {}).get('mode') == 'ego_joint':
        if str(ROOT.parent) not in sys.path:
            sys.path.insert(0, str(ROOT.parent))
        if args.resume and not args.id:
            parser.error('--resume requires --id')
        if sys.platform.startswith('linux'):
            sys.path.append(str(ROOT / '.runtime/stage2_wsl_deps'))
            sys.path.append(str(ROOT / '.runtime/offline_viz_wsl_deps'))
        from data.src.processing.joint_preprocessing import process as joint_process
        print(joint_process(args.config, args.id, prepare_only=args.prepare_only, resume=args.resume,
                            reuse_version=args.reuse_version, activate=not args.no_activate))
    else:
        if args.prepare_only or args.resume or args.reuse_version:
            parser.error('Incremental/resume flags require preprocessing.mode=ego_joint')
        print(process(args.config, args.id, not args.no_activate))


if __name__ == '__main__':
    main()
