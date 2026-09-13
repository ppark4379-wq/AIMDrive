#!/usr/bin/env python3
"""Train-only speed interventions with fixed recorded RGB/LiDAR; no training."""
import argparse
from pathlib import Path
from types import SimpleNamespace

from offline_train_replay import ROOT, bootstrap, np, read, sha256, torch, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run, output = args.run.resolve(), args.output.resolve()
    if ROOT not in output.parents or output.exists():
        raise ValueError('Use a new diagnostic JSON file inside LAVIS')
    output.parent.mkdir(parents=True, exist_ok=True)
    settings, provenance = read(run/'resolved_config.json'), read(run/'provenance.json')
    backend = bootstrap(ROOT.parent)
    from lavis.datasets.datasets.morai.expert_vldrive_dataset import MoraiExpertVLDriveDataset
    from lavis.runners.vldrive_trainer import seed_all, source_hashes, restore_model_checkpoint
    from lavis.runners.morai_trainer import autocast
    from lavis.models.vldrive.training_model import VLDriveTrainingConfig, VLDriveTrainingModel
    from tools.morai.cache_expert_vision_tokens import make_dataset_and_loader, load_feature_model
    from lavis.common.morai_runtime import complete_visual_checkpoint_initialization
    from torch.utils.data._utils.collate import default_collate

    torch.set_num_threads(settings.get('cpu_threads', 4))
    seed_all(settings['seed'])
    dataset = MoraiExpertVLDriveDataset(settings['dataset'], cache_root=settings['feature_cache'],
        split_file=ROOT/settings['split_file'], instructions=settings['instructions'], split='train',
        sequence_length=settings['sequence_length'])
    if (backend != provenance['backend'] or source_hashes(ROOT) != provenance['source']
            or dataset.describe() != provenance['datasets']['train']):
        raise ValueError('Provenance mismatch')
    if sha256(Path(settings['model']['visual_checkpoint'])) != dataset.checkpoint_sha256:
        raise ValueError('Vision checkpoint mismatch')
    model = VLDriveTrainingModel(VLDriveTrainingConfig(**settings['model']), device='cuda')
    state = restore_model_checkpoint(run/'model-last.pt', model, provenance=provenance)
    assert state['updates'] == 9540
    del state
    model.eval()
    with complete_visual_checkpoint_initialization():
        encoder, strict = load_feature_model(Path(settings['model']['visual_checkpoint']), torch.device('cuda'))
    raw_args = SimpleNamespace(dataset_root=Path(settings['dataset']),
        split_file=ROOT/settings['split_file'], batch_size=4, workers=0)
    raw_dataset, _ = make_dataset_and_loader(raw_args, 'train')
    lookup = {key: i for i, key in enumerate(raw_dataset.route_frames)}
    window_by_frame = {int(dataset.entries[end-1]['frame_id']): i
                       for i, (begin, end) in enumerate(dataset.windows)}
    windows = {}
    for frame in (990, 991, 995):
        index = window_by_frame[frame]
        begin, end = dataset.windows[index]
        entries = dataset.entries[begin:end]
        sensors = [raw_dataset[lookup[(r['route_id'], int(r['frame_id']))]][0] for r in entries]
        windows[frame] = dict(sensors=default_collate(sensors), instruction=entries[-1]['instruction'],
            cached=dataset[index]['model_inputs']['visual_feature'],
            gt=dataset[index]['training_targets']['future_trajectory'][-1].numpy())
    synthetic_speed = float(windows[995]['sensors']['velocity'][-1].reshape(-1)[0])
    probes = [(990, 'recorded_velocity'), (991, 'recorded_velocity'), (995, 'recorded_velocity'),
              (995, 'all_history_velocity_zero'), (991, 'all_history_velocity_zero'),
              (991, 'current_velocity_from_995')]
    results = []
    original_predictions = {}
    with torch.inference_mode():
        for frame, intervention in probes:
            window = windows[frame]
            sensors = dict(window['sensors'])
            sensors['velocity'] = sensors['velocity'].clone()
            if intervention == 'all_history_velocity_zero':
                sensors['velocity'].zero_()
            elif intervention == 'current_velocity_from_995':
                sensors['velocity'][-1].fill_(synthetic_speed)
            features = []
            for offset in range(0, 11, 4):
                chunk = {k: v[offset:offset+4].cuda() for k, v in sensors.items()}
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    features.append(encoder(chunk).float())
            features = torch.cat(features)[None]
            with autocast(model):
                prediction = model.predict(dict(visual_feature=features,
                    instruction=[window['instruction']]), compute_auxiliary=False)
            downstream = prediction['trajectory'][0, -1].float().cpu().numpy()
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                head = encoder.waypoints_generator(features[:, -1, 101:106],
                    sensors['target_point'][-1:].cuda())[0].float().cpu().numpy()
            if intervention == 'recorded_velocity':
                original_predictions[frame] = downstream
            gt = window['gt']
            row = dict(frame_id=frame, intervention=intervention,
                recorded_velocities_mps=window['sensors']['velocity'].reshape(-1).tolist(),
                supplied_velocities_mps=sensors['velocity'].reshape(-1).tolist(),
                target_point=sensors['target_point'][-1].tolist(), instruction=window['instruction'],
                actual_future=gt.tolist(), downstream_prediction=downstream.tolist(),
                vision_head_prediction=head.tolist(),
                downstream_ade_vs_recorded_gt_m=float(np.linalg.norm(downstream-gt, axis=1).mean()),
                downstream_fde_vs_recorded_gt_m=float(np.linalg.norm(downstream[-1]-gt[-1])),
                downstream_endpoint_norm_m=float(np.linalg.norm(downstream[-1])),
                vision_head_endpoint_norm_m=float(np.linalg.norm(head[-1])),
                prediction_max_change_from_raw_baseline_m=float(np.abs(downstream-original_predictions[frame]).max()),
                feature_mae_from_recorded_cache=float((features[0].cpu()-window['cached']).abs().mean()))
            results.append(row)
            payload = dict(split='train_only', updates=9540, strict_encoder_load=strict,
                checkpoint_sha256=sha256(run/'model-last.pt') if len(results)==len(probes) else None,
                note='Only speed is intervened. RGB, LiDAR, target point and instruction fixed. '
                     'Intervened inputs are counterfactual; recorded GT is descriptive, not their valid new ground truth. '
                     'No optimization and no simulator control.',
                probes=results)
            write(output, payload)
            print(row, flush=True)
    print('DEPARTURE_SPEED_DIAGNOSIS_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
