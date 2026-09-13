#!/usr/bin/env python3
"""Export every supervised frame occurrence of the original completed run."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time

os.environ['LAVIS_IMPORT_PROFILE'] = 'morai'
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from lavis.common.morai_runtime import configure_runtime, bootstrap, sha256
configure_runtime()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    assert ROOT / 'outputs' in output.parents
    output.mkdir(parents=True, exist_ok=True)
    destination = output / 'all_frame_readout.json'
    matrix_file = output / 'all_frame_readout.npz'
    head_file = output / 'head_initial.pt'
    assert not any(path.exists() for path in (destination, matrix_file, head_file))
    import fcntl
    lock = (ROOT / '.runtime/training_runs/last_encoder.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    import numpy as np
    import torch
    bootstrap(ROOT.parent)
    from lavis.models.vldrive.training_model import VLDriveTrainingConfig, VLDriveTrainingModel
    import lavis.models.vldrive.ddia_attention as attention
    from lavis.runners.vldrive_trainer import parameter_hash, source_hashes, seed_all
    from lavis.runners.morai_trainer import autocast
    from lavis.datasets.datasets.morai.expert_vldrive_dataset import MoraiExpertVLDriveDataset
    from tools.morai.compare_rope_weight_transfer import historical_cast_before_trig
    torch.set_num_threads(2)
    run = ROOT / 'outputs/morai_paper_last_encoder_20260910T055054Z'
    settings = read(run / 'resolved_config.json')
    provenance = read(run / 'provenance.json')
    checkpoint = run / 'model-last.pt'
    digest = read(checkpoint.with_suffix('.pt.sha256.json'))
    assert checkpoint.stat().st_size == digest['bytes'] and sha256(checkpoint) == digest['sha256']
    sources = source_hashes(ROOT)
    seed_all(settings['seed'])
    dataset = MoraiExpertVLDriveDataset(settings['dataset'], cache_root=settings['feature_cache'],
        split_file=ROOT/settings['split_file'], instructions=settings['instructions'], split='train', sequence_length=11)
    assert dataset.describe() == provenance['datasets']['train']
    indices = read(run / 'window_indices.json')['train']
    saved = read(run / 'train_predictions.json')
    assert len(indices) == len(saved) == 636
    print('LOADING_ORIGINAL_9540_FOR_6996_OCCURRENCE_EXPORT', flush=True)
    model = VLDriveTrainingModel(VLDriveTrainingConfig(**settings['model']), device='cuda').eval()
    state = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=False)
    assert asdict(VLDriveTrainingConfig(**state['model_config'])) == asdict(model.config)
    assert state['updates'] == 9540
    frozen = parameter_hash(model, trainable=False)
    assert frozen == state['frozen_hash']
    names = {n for n, p in model.named_parameters() if p.requires_grad}
    assert names == set(state['trainable_state'])
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name in names:
                parameter.copy_(state['trainable_state'][name])
    trainable = parameter_hash(model, trainable=True)
    assert trainable == state['trainable_hash']
    del state
    head_state = {key: value.detach().cpu().clone() for key, value in model.language_model.waypoints_predictor.state_dict().items()}
    original_rope = attention.apply_rope
    vectors = {key: [] for key in ('frame_ids', 'native_sync_frames', 'anchor_frames', 'command',
        'target', 'prediction', 'llm_readout', 'head_relu', 'selected_count')}
    trace = {}
    def head_input_hook(module, args):
        trace['llm_readout'] = args[0][0].float().cpu().numpy()
    def head_relu_hook(module, args, result):
        trace['head_relu'] = result[0].float().cpu().numpy()
    handles = [model.language_model.waypoints_predictor.register_forward_pre_hook(head_input_hook),
               model.language_model.waypoints_predictor[1].register_forward_hook(head_relu_hook)]
    started = time.monotonic()
    try:
        attention.apply_rope = historical_cast_before_trig
        with torch.inference_mode():
            for i, (index, record) in enumerate(zip(indices, saved)):
                sample = dataset[index]
                frames = sample['metadata']['frame_ids'].numpy()
                native = sample['metadata']['native_sync_frames'].numpy()
                frame = int(record['sample_id'].rsplit('/', 1)[1])
                assert frames[-1] == frame and np.array_equal(frames, np.arange(frame-10, frame+1))
                x = sample['model_inputs']['visual_feature'][None].cuda()
                sentence = sample['model_inputs']['instruction']
                trace.clear()
                with autocast(model):
                    result = model.predict(dict(visual_feature=x, instruction=[sentence]), compute_auxiliary=False)
                pred = result['trajectory'][0].float().cpu().numpy()
                assert np.array_equal(pred[-1], np.asarray(record['trajectory'])), ('Prediction mismatch', frame)
                assert trace['llm_readout'].shape == trace['head_relu'].shape == (11, 2048)
                vectors['frame_ids'].append(frames)
                vectors['native_sync_frames'].append(native)
                vectors['anchor_frames'].append(frame)
                vectors['command'].append(sample['metadata']['command'])
                vectors['target'].append(sample['training_targets']['future_trajectory'].numpy())
                vectors['prediction'].append(pred)
                vectors['selected_count'].append(result['selected_count'][0].cpu().numpy())
                for name in ('llm_readout', 'head_relu'):
                    vectors[name].append(trace[name])
                if (i+1) % 25 == 0 or i == 0:
                    elapsed = time.monotonic() - started
                    progress = dict(completed_windows=i+1, total_windows=636,
                        completed_frame_occurrences=(i+1)*11, elapsed_s=elapsed,
                        estimated_remaining_s=elapsed/(i+1)*(636-i-1), baseline_anchor_max_difference_m=0)
                    write(output/'all_frame_export_progress.json', progress)
                    print(json.dumps(progress), flush=True)
    finally:
        attention.apply_rope = original_rope
        for handle in handles:
            handle.remove()
    assert source_hashes(ROOT) == sources
    assert trainable == parameter_hash(model, trainable=True)
    assert frozen == parameter_hash(model, trainable=False)
    arrays = {name: np.asarray(value) for name, value in vectors.items()}
    np.savez(matrix_file, **arrays)
    torch.save(dict(state_dict=head_state, checkpoint_sha256=digest['sha256'], updates=9540,
                    source_hashes=sources, dtype='bfloat16'), head_file)
    write(destination, dict(complete=True, optimizer_updates=0, split='train_only',
        original_updates=9540, checkpoint_sha256=digest['sha256'], source_sha256=sources,
        original_dataset_description=dataset.describe(), windows=636, frame_occurrences=6996,
        arrays={key:list(value.shape) for key,value in arrays.items()},
        trainable_hash_unchanged=True, frozen_hash_unchanged=True, apply_rope_restored=True,
        production_sources_unchanged=True, all_saved_anchor_predictions_exact=True,
        matrix_sha256=sha256(matrix_file), head_initial_sha256=sha256(head_file),
        elapsed_s=time.monotonic()-started,
        limitations='Frozen original final model in eval mode, historical RoPE scoped to this process. '
            'Not a replay of stochastic Gumbel training gradients or a new full-model checkpoint. '
            'Every original supervised frame occurrence exported separately with its actual window context.'))
    print('ALL_FRAME_READOUT_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
