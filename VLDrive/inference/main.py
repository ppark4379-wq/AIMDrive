"""Prepare a selected replay, then run/cache inference separately from its viewer."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT.parent)]
if (ROOT / '.runtime/stage2_wsl_deps').is_dir() and sys.platform != 'win32':
    sys.path.insert(0, str(ROOT / '.runtime/stage2_wsl_deps'))

from inference.checkpoint import inspect_checkpoint, load_joint, local_path
from inference.dataset import ReplayDataset, read_json, sha256
from inference.results import Results, write_json


def config_digest(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, allow_nan=False).encode()).hexdigest()


def prepare(config, output):
    selection = read_json(local_path(config['selection']))
    episodes = selection['episodes']
    counts = {key: sum(e['category'] == key for e in episodes) for key in config['scenario_counts']}
    if counts != config['scenario_counts'] or len(episodes) != sum(counts.values()):
        raise ValueError('Selected scenario counts differ from the requested counts')
    ids = [e['episode_id'] for e in episodes]
    if len(ids) != len(set(ids)):
        raise ValueError('A drive was selected more than once')
    state, info = inspect_checkpoint(config['joint_checkpoint'])
    if (config['sequence_length'] != state['settings']['sequence_length'] or
            config['frame_interval'] != state['settings']['frame_sampling_interval']):
        raise ValueError('Joint replay history must match the saved training settings')
    del state
    results = Results(output, dict(config=config, config_sha256=config_digest(config),
        selection_sha256=sha256(local_path(config['selection'])), joint=info,
        situation_checkpoint_sha256=sha256(local_path(config['situation_checkpoint'])), episodes=episodes))
    dataset = ReplayDataset(local_path(config['bundle']), ids,
                            sequence_length=config['sequence_length'], frame_interval=config['frame_interval'])
    total = 0
    for i, entry in enumerate(episodes):
        document = dataset.review_document(entry['episode_id'])
        if 'traffic_event' in entry:
            document['traffic_event'] = entry['traffic_event']
            positions = {f['index']: j for j, f in enumerate(document['frames'])}
            for key in ('last_red_frame', 'first_green_frame', 'departure_frame'):
                position = positions[entry['traffic_event'][key]]
                if dataset.window(entry['episode_id'], position)[0] is None:
                    raise ValueError('Selected traffic event lacks a valid model input window: ' + entry['episode_id'])
        total += len(document['frames'])
        results.episode(i, document)
        results.update(prepared_episodes=i+1, total_frames=total)
        print(json.dumps(dict(event='prepared', episode=i+1, total=len(episodes),
                              frames=len(document['frames']))), flush=True)
    results.update(status='ready', prepared_episodes=len(episodes))


def infer(config, output, *, max_predictions=None):
    metadata = read_json(output / 'run.json')
    try:
        _infer(config, output, max_predictions=max_predictions)
    except BaseException as error:
        # Include initialization failures, before the prediction loop starts.
        metadata = read_json(output / 'run.json')
        metadata.update(status='failed', error=f'{type(error).__name__}: {error}')
        write_json(output / 'run.json', metadata)
        raise


def _infer(config, output, *, max_predictions=None):
    import torch
    from utils.common.runtime import bootstrap
    from inference.runtime import JointPredictor
    from inference.situation import SituationPredictor
    metadata = read_json(output / 'run.json')
    if metadata['config_sha256'] != config_digest(config):
        raise ValueError('Replay config changed; create a new run')
    if metadata['selection_sha256'] != sha256(local_path(config['selection'])):
        raise ValueError('Replay selection changed')
    if metadata['situation_checkpoint_sha256'] != sha256(local_path(config['situation_checkpoint'])):
        raise ValueError('Situation checkpoint changed; create a new run')
    if metadata.get('prepared_episodes') != len(metadata['episodes']):
        raise ValueError('Replay preparation is incomplete')
    metadata.update(status='loading', error=None)
    write_json(output / 'run.json', metadata)
    print(json.dumps(dict(event='loading', checkpoint=config['joint_checkpoint'])), flush=True)
    bootstrap(ROOT)
    torch.set_num_threads(config.get('cpu_threads', 2))
    dataset = ReplayDataset(local_path(config['bundle']), [e['episode_id'] for e in metadata['episodes']],
                            sequence_length=config['sequence_length'], frame_interval=config['frame_interval'])
    model, joint_info, text_contract = load_joint(config['joint_checkpoint'], dataset.meta, device=config['device'])
    if joint_info['sha256'] != metadata['joint']['sha256']:
        raise ValueError('Joint best checkpoint changed since preparation; create a new run')
    situation = SituationPredictor(local_path(config['situation_checkpoint']), dataset.meta, text_contract,
                                   local_path(config['situation_renderer']), device=config['device'],
                                   length=config['situation_sequence_length'])
    predictor = JointPredictor(model, situation, device=config['device'])
    results = object.__new__(Results)
    results.root, results.metadata = output, metadata
    results.update(status='running', joint=joint_info, situation=situation.info, error=None)
    successful, start = 0, time.perf_counter()
    try:
        for i, entry in enumerate(metadata['episodes']):
            identity = entry['episode_id']
            frames = dataset.episode(identity)['document']['frames']
            path = output / 'predictions' / f'{i:03d}.jsonl'
            completed = set()
            if path.exists():
                # Resume only complete records. A partial trailing write is an
                # explicit error; do not append after corrupt JSON.
                with path.open(encoding='utf-8') as stream:
                    for line in stream:
                        if not line.endswith('\n'):
                            raise ValueError('Incomplete prediction record: ' + str(path))
                        completed.add(json.loads(line)['position'])
            for position in range(len(frames)):
                if position in completed:
                    continue
                prediction = predictor.predict(dataset, identity, position)
                results.prediction(i, prediction)
                if prediction.get('trajectory') is not None:
                    successful += 1
                if successful % 20 == 0 or position == len(frames)-1:
                    progress = dict(current_episode=i, current_position=position,
                                    session_predictions=successful, session_elapsed_s=time.perf_counter()-start)
                    results.update(**progress)
                    print(json.dumps(dict(event='progress', **progress)), flush=True)
                if max_predictions is not None and successful >= max_predictions:
                    results.update(status='paused', reason='requested_prediction_limit')
                    return
        results.update(status='complete', session_predictions=successful,
                       session_elapsed_s=time.perf_counter()-start)
    except BaseException as error:
        results.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise


def main():
    from utils.common.runtime import configure_runtime
    configure_runtime()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'inference/config.json')
    parser.add_argument('--output', type=Path, required=True)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument('--prepare-only', action='store_true')
    actions.add_argument('--predict', action='store_true')
    parser.add_argument('--max-predictions', type=int)
    args = parser.parse_args()
    if args.max_predictions is not None and args.max_predictions < 1:
        parser.error('--max-predictions must be positive')
    config = read_json(args.config)
    output = local_path(args.output)
    if (ROOT / 'outputs/inference').resolve() not in output.parents:
        parser.error('Use an output folder inside VLDrive/outputs/inference')
    if args.prepare_only:
        prepare(config, output)
    else:
        infer(config, output, max_predictions=args.max_predictions)


if __name__ == '__main__':
    main()
