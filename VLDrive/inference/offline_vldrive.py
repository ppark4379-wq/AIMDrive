#!/usr/bin/env python3
"""Recorded MORAI sensors -> VLDrive waypoints -> publication PNG/PDF figures.

    python inference/offline_vldrive.py --config configs/vldrive_offline_inference.json

Checkpoint selection belongs to the user. Validation/renderer preparation does
not choose or load any Stage-2 weights. Figure stride never subsamples history.
"""
from collections import deque
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
if sys.platform.startswith('linux') and (ROOT/'.runtime/stage2_wsl_deps').is_dir():
    sys.path.insert(0,str(ROOT/'.runtime/stage2_wsl_deps'))
if sys.platform.startswith('linux') and (ROOT/'.runtime/offline_viz_wsl_deps').is_dir():
    sys.path.insert(0,str(ROOT/'.runtime/offline_viz_wsl_deps'))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.runtime/matplotlib'))

from utils.runtime import configure_runtime, sha256
configure_runtime()
from inference.vldrive_offline_config import ConfigError, load_config, require_checkpoint


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_manifest(path, document):
    path.parent.mkdir(parents=True,exist_ok=True)
    pending=path.with_suffix('.pending')
    pending.write_text(json.dumps(document,indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf-8')
    pending.replace(path)


def detect_dataset_format(root, requested='auto'):
    root=Path(root)
    if not root.is_dir():
        raise ConfigError('Dataset root not found: '+str(root))
    episodes=sorted(p.parent for p in root.rglob('episode.json'))
    required=('CAMERA_1','CAMERA_2','CAMERA_3','CAMERA_4','LIDAR_5','GPS_6','IMU_7')
    valid=[p for p in episodes if (p/'native_associations/associations.json').is_file()
           and (p.parent/'sensor_contract.json').is_file()
           and all((p.parent/name).is_dir() for name in required)]
    if not valid or len(valid)!=len(episodes):
        raise ConfigError('Expected native sensor folders plus audited Sensor Record episodes; '
                          'CARLA-style or unaudited captures cannot reproduce this Stage-2 cache')
    if requested not in ('auto','morai_native_sensor_record'):
        raise ConfigError('Explicit dataset format differs from the inspected native Sensor Record')
    return 'morai_native_sensor_record',valid


def visual_contract(config):
    """Verify the actual frozen-cache recipe without choosing a Stage-2 file."""
    cache=config.visual_cache
    meta=read_json(cache/'cache_meta.json')
    if (meta.get('schema')!='navigation-token-cache-v1'
            or meta.get('trajectory_coordinate_convention')!='lmdrive_left_negative_forward'):
        raise ConfigError('Unsupported visual-cache/coordinate contract')
    for name,key in [('manifest.jsonl','manifest_sha256'),('features.jsonl','features_manifest_sha256'),
                     ('visual_source.json','visual_source_sha256')]:
        if sha256(cache/name)!=meta[key]:
            raise ConfigError('Visual-cache provenance changed: '+name)
    identity=read_json(cache/'visual_source.json')
    if (identity['encoder_contract']!=meta['encoder_contract']
            or identity['checkpoint_sha256']!=meta['checkpoint_sha256']
            or identity['inference_precision']!='float32' or identity['storage_precision']!='float16'):
        raise ConfigError('Unsupported encoder precision or cache recipe')
    for name,digest in identity['sources_sha256'].items():
        if sha256(ROOT/name)!=digest:
            raise ConfigError('Cache preprocessing/model source changed: '+name)
    if not config.vision_checkpoint.is_file() or sha256(config.vision_checkpoint)!=meta['checkpoint_sha256']:
        raise ConfigError('Vision checkpoint must match the Stage-2 cache SHA256')
    if meta['encoder_contract']['adapter']['lidar_axis_policy']!=config.lidar_axis_policy:
        raise ConfigError('Configured LiDAR axes differ from the cache recipe')
    return dict(cache=cache,meta=meta,identity=identity)


def build_dataset(config,contract):
    """Reuse existing native loaders, including the exact cache input adapter."""
    from datasets.morai_native_capture import MoraiNativeCaptureDataset
    from datasets.navigation_sensor_record import NativeNavigationSensors
    detected,episodes=detect_dataset_format(config.dataset_root,config.dataset_format)
    datasets=[]
    for episode in episodes:
        print('Checking recorded episode: '+str(episode),flush=True)
        # NativeCapture provides the original file inventory and image reader.
        # Its legacy __getitem__ has different velocity/crop/LiDAR choices, so
        # encoder inputs use the existing loader that actually built this cache.
        inventory=MoraiNativeCaptureDataset(episode.parent,lidar_axis_policy=config.lidar_axis_policy)
        native=NativeNavigationSensors(episode,contract['meta']['encoder_contract'],config.namespace,
                                       config.max_sensor_gap_ms)
        expected=contract['identity']['source'].get(native.identity)
        if expected!=native.provenance:
            raise ConfigError('Dataset source differs from the audited visual-cache source: '+native.identity)
        admitted=set(inventory.stamps)
        for scene in native.scenes:
            stamp=Path(scene['camera_files']['camera_1']).name.removesuffix('_Intensity.png')
            if stamp not in admitted:
                raise ConfigError('Audited frame is absent from the complete native capture inventory')
        native.capture_inventory=inventory
        datasets.append(native)
    return datasets,dict(root=str(config.dataset_root),format=detected,
        loader='NativeNavigationSensors.inputs + MoraiNativeCaptureDataset inventory/image reader',
        number_of_frames=sum(len(d) for d in datasets),
        episodes=[dict(id=d.identity,frames=len(d),source=d.provenance) for d in datasets])


def frame_records(datasets):
    records=[]
    for episode_index,dataset in enumerate(datasets):
        for index,stamp in enumerate(dataset.times):
            records.append(dict(global_index=len(records),episode_index=episode_index,
                                frame_id=index,timestamp_ns=stamp,route=dataset.identity))
    return records


def window_indices(records,anchor,sequence_length=11,sampling_interval=1,max_gap_ms=150.):
    """Validate every source frame in a window before any explicit subsampling."""
    span=(sequence_length-1)*sampling_interval+1
    if not 0<=anchor<len(records) or anchor<span-1:
        return None
    full=records[anchor-span+1:anchor+1]
    for previous,current in zip(full,full[1:]):
        if (current['route']!=previous['route'] or current['frame_id']!=previous['frame_id']+1
                or not 0<current['timestamp_ns']-previous['timestamp_ns']<=max_gap_ms*1e6):
            return None
    return [row['global_index'] for row in full[::sampling_interval]]


def select_indices(config,records):
    eligible=[i for i in range(len(records)) if window_indices(records,i,config.sequence_length,
              config.frame_sampling_interval,config.max_sensor_gap_ms) is not None]
    if config.frame_ids is not None:
        missing=sorted(set(config.frame_ids)-set(eligible))
        if missing:
            raise ConfigError('frames.ids have no complete causal history: '+str(missing))
        chosen=list(config.frame_ids)
    else:
        candidates=[i for i in eligible if i>=config.start]
        chosen=candidates[::config.stride][:config.limit]
    if not chosen:
        raise ConfigError('No complete windows selected; check frames start/ids/stride/limit')
    return chosen


def load_offline_stage2_contract(config,visual):
    from inference.vldrive_runtime import verify_sources
    require_checkpoint(config)
    run=config.vldrive_checkpoint.parent
    training=read_json(run/'resolved_config.json')
    provenance=read_json(run/'provenance.json')
    description=read_json(run/'model.json')
    if any(d['cache_meta']!=visual['meta'] for d in provenance['datasets'].values()):
        raise ConfigError('User-selected checkpoint was trained with a different visual cache')
    if (training['sequence_length']!=11 or description['config']['memory_size']!=10
            or description['config']['input_mode']!='cached'
            or description['config']['visual_sha256']!=visual['meta']['checkpoint_sha256']):
        raise ConfigError('Expected an 11-frame cached-input VLDrive model from this encoder')
    return dict(visual,checkpoint=config.vldrive_checkpoint,training=training,provenance=provenance,
                description=description,source_compatibility=verify_sources(provenance),
                training_complete=(run/'report.json').is_file())


def front_camera(dataset,index):
    from datasets.morai_sensor_record_dataset import child
    # inputs(index) verifies the same raw camera hash before prediction/rendering.
    path=child(dataset.native,dataset.scenes[index]['camera_files']['camera_1'])
    return dataset.capture_inventory._read_image(path)


def render_frame(config,record,prediction,dataset,folder,example=False):
    from inference.vldrive_waypoint_visualization import render_vldrive_waypoints
    from utils.vision_encoder_visualization import save_figure
    kwargs=dict(instruction=config.instruction,frame_id=record['frame_id'],route=record['route'],
                selected_tokens=prediction.get('selected_tokens'),example=example)
    actual=dataset.target(record['frame_id']) if config.show_actual_trajectory and not example else None
    kwargs['actual_trajectory']=actual
    stem=folder/('frame_{:06d}_vldrive'.format(record['global_index']))
    if example:
        stem=folder/'renderer_example_NOT_MODEL_PREDICTION'
    figure=render_vldrive_waypoints(prediction['waypoints_raw'],**kwargs)
    written=save_figure(figure,stem.with_name(stem.name+'_waypoints'),config.formats,config.dpi)
    if config.overview:
        figure=render_vldrive_waypoints(prediction['waypoints_raw'],
            camera_image=front_camera(dataset,record['frame_id']),**kwargs)
        written+=save_figure(figure,stem.with_name(stem.name+'_overview'),config.formats,config.dpi)
    return [dict(path=str(p.relative_to(folder)),sha256=sha256(p)) for p in written],actual


def infer_sequence(config,datasets,records,selected,encoder,model):
    """Infer each causal window, even when only sparse anchors are rendered."""
    import torch
    from inference.vldrive_runtime import encode_frame,predict_window
    span=(config.sequence_length-1)*config.frame_sampling_interval+1
    buffer=deque(maxlen=span)
    for anchor in range(selected[-1]+1):
        record=records[anchor]
        dataset=datasets[record['episode_index']]
        inputs=dataset.inputs(record['frame_id'])
        batch={key:value.unsqueeze(0).to(config.device) for key,value in inputs.items()}
        feature,encoder_ms=encode_frame(encoder,batch)
        buffer.append((anchor,feature))
        ids=window_indices(records,anchor,config.sequence_length,config.frame_sampling_interval,config.max_sensor_gap_ms)
        if ids is None:
            continue
        features=dict(buffer)
        window=torch.stack([features[i] for i in ids]).unsqueeze(0)
        result=predict_window(model,window,config.instruction)
        result.update(record,history_frame_ids=[records[i]['frame_id'] for i in ids],
            history_global_indices=ids,history_timestamps_ns=[records[i]['timestamp_ns'] for i in ids],
            instruction=config.instruction,vision_encoder_ms=encoder_ms,
            total_inference_ms=encoder_ms+result['vldrive_ms'])
        yield result


def base_manifest(config,contract,dataset):
    return dict(schema='morai-vldrive-offline-inference-report-v1',status='prepared',
        config_path=str(config.config_path),config_sha256=config.config_sha256,config=config.flat_dict(),
        vision_encoder=dict(checkpoint=str(config.vision_checkpoint),sha256=contract['meta']['checkpoint_sha256'],
            feature_model='memfuser_baseline_e1d3_return_feature',feature_shape=[1,106,256],
            recipe=contract['meta']['encoder_contract'],precision='float32',storage_boundary='float16 -> float32'),
        vldrive=dict(checkpoint=str(config.vldrive_checkpoint) if config.vldrive_checkpoint else None,
                     sha256=None,model_config=None,updates=None,loaded=False),
        dataset=dataset,runtime=dict(device=config.device,seed=config.seed,cpu_threads=config.cpu_threads),
        instruction=config.instruction,selection_mode='gumbel',model_mode='eval',reconstruction=False,
        history=dict(sequence_length=11,frame_sampling_interval=config.frame_sampling_interval,
                     figure_stride=config.stride,figure_stride_changes_history=False),
        coordinate_convention='lmdrive_left_negative_forward',plot_axes='x=right, y=forward (metres)',
        horizons_s=[.5,1.,1.5,2.,2.5],prediction_source='VLDrive stage-2 trajectory[0,-1]',windows=[])


def validate_only(config,contract,datasets,records,selected,manifest,*,check_vision=False,render_example=False):
    import numpy as np
    import torch
    from inference.vldrive_runtime import encode_frame,load_encoder,seed_runtime
    folder=config.output/'preparation'
    record=records[selected[0]]
    dataset=datasets[record['episode_index']]
    inputs=dataset.inputs(record['frame_id'])
    manifest.update(status='validation_only',inference_performed=False,selected_figure_indices=selected,
                    input_shapes={key:list(value.shape) for key,value in inputs.items()})
    if check_vision:
        encoder=load_encoder({'stage1_checkpoint':str(config.vision_checkpoint)},contract,config.device)
        seed_runtime(config.seed)
        feature,elapsed=encode_frame(encoder,{k:v.unsqueeze(0).to(config.device) for k,v in inputs.items()})
        refs=[json.loads(line) for line in (contract['cache']/'features.jsonl').read_text().splitlines()]
        ref=next(r['visual_feature'] for r in refs if r['episode_id']==dataset.identity
                 and r['frame_index']==record['frame_id'])
        path=(contract['cache']/ref['path']).resolve()
        if contract['cache'] not in path.parents or sha256(path)!=ref['sha256']:
            raise ConfigError('Cached regression token integrity mismatch')
        expected=torch.from_numpy(np.load(path,allow_pickle=False).astype('float32')).to(config.device)
        torch.testing.assert_close(feature,expected,atol=.002,rtol=.002)
        manifest['vision_regression']=dict(shape=[1,106,256],frame_id=record['frame_id'],
            max_abs=float((feature-expected).abs().max()),mean_abs=float((feature-expected).abs().mean()),
            encoder_ms=elapsed,atol=.002,rtol=.002,passed=True)
    if render_example:
        # Explicit fixture, NOT an arbitrary stage-1/Stage-2 prediction or GT.
        synthetic=dict(waypoints_raw=[[0.,-4.],[.2,-8.],[.6,-12.],[1.2,-16.],[2.,-20.]])
        written,_=render_frame(config,record,synthetic,dataset,folder,example=True)
        manifest['renderer_example']=dict(source='synthetic layout fixture; not model predictions',figures=written)
    return folder


def run(config,*,validation=False,check_vision=False,render_example=False):
    if not validation:
        require_checkpoint(config)  # No directory search or fallback checkpoint.
        if (config.output/'inference_manifest.json').exists() or (config.output/'waypoints.jsonl').exists():
            raise ConfigError('Output already contains inference results; set output.dir to a new directory')
    import torch
    from inference.vldrive_runtime import load_encoder,load_stage2,seed_runtime
    torch.set_num_threads(config.cpu_threads)
    if (not validation or check_vision) and config.device=='cuda' and not torch.cuda.is_available():
        raise ConfigError('runtime.device=cuda but no CUDA device is available')
    started=time.perf_counter()
    contract=visual_contract(config)
    datasets,dataset_description=build_dataset(config,contract)
    records=frame_records(datasets)
    selected=select_indices(config,records)
    manifest=base_manifest(config,contract,dataset_description)
    if validation:
        folder=validate_only(config,contract,datasets,records,selected,manifest,
                             check_vision=check_vision,render_example=render_example)
    else:
        contract=load_offline_stage2_contract(config,contract)
        model,checkpoint_info=load_stage2(contract,config.device)
        encoder=load_encoder({'stage1_checkpoint':str(config.vision_checkpoint)},contract,config.device)
        seed_runtime(config.seed)
        manifest['vldrive']=dict(checkpoint_info,checkpoint=str(config.vldrive_checkpoint),loaded=True,
            sha256=checkpoint_info['file_digest']['sha256'],model_config=contract['description']['config'])
        train_interval=contract['training']['frame_sampling_interval']
        manifest['history'].update(training_frame_sampling_interval=train_interval,
            matches_training=config.frame_sampling_interval==train_interval)
        if train_interval!=config.frame_sampling_interval:
            print(f'History interval: requested {config.frame_sampling_interval}, training {train_interval}; '
                  'recorded explicitly in manifest.',flush=True)
        folder=config.output
        folder.mkdir(parents=True,exist_ok=True)
        if (folder/'inference_manifest.json').exists() or (folder/'waypoints.jsonl').exists():
            raise ConfigError('Output already contains inference results; set output.dir to a new directory')
        manifest.update(status='running',model_windows_inferred=0,selected_figure_indices=selected)
        write_manifest(folder/'inference_manifest.json',manifest)
        chosen=set(selected)
        try:
            with (folder/'waypoints.jsonl').open('x',encoding='utf-8') as stream:
                for result in infer_sequence(config,datasets,records,selected,encoder,model):
                    manifest['model_windows_inferred']+=1
                    result['figures']=[]
                    if result['global_index'] in chosen:
                        written,actual=render_frame(config,result,result,datasets[result['episode_index']],folder)
                        result['figures']=written
                        result['actual_future_trajectory_raw']=None if actual is None else actual.tolist()
                        result['actual_overlay_status']=('available' if actual is not None else
                            'unavailable' if config.show_actual_trajectory else 'disabled')
                        manifest['windows'].append(dict(result))
                        print(f'[{len(manifest["windows"])}/{len(selected)}] frame {result["frame_id"]} '
                              f'| CCDP {result["selected_tokens"]}/106 | {len(written)} figure files',flush=True)
                        write_manifest(folder/'inference_manifest.json',manifest)
                    stream.write(json.dumps(result,allow_nan=False)+'\n')
            manifest['status']='completed'
        except BaseException as error:
            manifest.update(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',error=str(error))
            manifest['elapsed_sec']=time.perf_counter()-started
            write_manifest(folder/'inference_manifest.json',manifest)
            raise
    manifest['elapsed_sec']=time.perf_counter()-started
    write_manifest(folder/'inference_manifest.json',manifest)
    print('Wrote '+str(folder/'inference_manifest.json'),flush=True)
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--validate-only',action='store_true',help='Validate dataset/config without loading Stage-2')
    parser.add_argument('--check-vision',action='store_true',help='With validation: compare one raw encoder frame with cache')
    parser.add_argument('--render-example',action='store_true',help='With validation: clearly labeled synthetic PNG/PDF fixture')
    args=parser.parse_args()
    if (args.check_vision or args.render_example) and not args.validate_only:
        parser.error('--check-vision and --render-example require --validate-only')
    try:
        config=load_config(args.config if args.config.is_file() else ROOT/args.config)
        run(config,validation=args.validate_only,check_vision=args.check_vision,render_example=args.render_example)
    except (ConfigError,ValueError,OSError) as error:
        parser.exit(2,str(error)+'\n')


if __name__=='__main__':
    main()
