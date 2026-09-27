#!/usr/bin/env python3
"""Review sealed signal GT beside the exact visual-encoder camera inputs.

This is a read-only diagnostic. Stop-line coordinates are not lamp positions;
camera visibility, lamp identity and occlusion require independent evidence.
The report never changes a training label or claims visibility from route GT.
"""
import argparse
import base64
import hashlib
import html
import io
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image
from data.loaders.morai_sensor_record_dataset import (
    PathResolver, SensorRecordDataset, digest, read_json, verify_seal,
)
from data.loaders.morai_transfer_adapter import (
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD,
    MoraiLMDriveVisionAdapter, MoraiVisionAdapterConfig,
)

VIEWS = ('front', 'left', 'right', 'rear')
INPUTS = ('rgb_front', 'rgb_center', 'rgb_left', 'rgb_right', 'rgb_rear')


def crop_rectangle(width, height, view):
    """Raw pixel rectangle retained by deterministic lmdrive_compat transforms.

    Mirrors torchvision CenterCrop's round-half-to-even integer offsets.
    Front focus first resizes to 800x600. Returned edges are continuous raw
    pixel coordinates; they describe geometric support, not lamp visibility.
    """
    if width <= 0 or height <= 0:
        raise ValueError('Image dimensions must be positive')
    if view == 'rgb_front':
        resized_width, resized_height, size = 341, 256, 224
    elif view == 'rgb_center':
        resized_width, resized_height, size = 800, 600, 128
    elif view in ('rgb_left', 'rgb_right', 'rgb_rear'):
        resized_width, resized_height, size = 195, 146, 128
    else:
        raise ValueError('Unknown model view: ' + str(view))
    left = int(round((resized_width - size) / 2.))
    top = int(round((resized_height - size) / 2.))
    return [left * width / resized_width, top * height / resized_height,
            (left + size) * width / resized_width,
            (top + size) * height / resized_height]


def rgb_from_model_tensor(tensor):
    array = tensor.detach().cpu().numpy().transpose(1, 2, 0)
    array = array * np.asarray(IMAGENET_DEFAULT_STD) + np.asarray(IMAGENET_DEFAULT_MEAN)
    return np.rint(np.clip(array, 0, 1) * 255).astype(np.uint8)


def encoded_image(rgb):
    buffer = io.BytesIO()
    Image.fromarray(rgb).save(buffer, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode('ascii')


def load_sealed_frame_sources(manifest_path):
    manifest_path = Path(manifest_path).expanduser().resolve()
    publication = verify_seal(manifest_path.parent)
    if (publication.get('schema') != 'morai_sensor_record_training_publication_v1'
            or publication.get('status') != 'passed'
            or publication.get('training_visible') is not True
            or publication['files'].get(manifest_path.name) != digest(manifest_path)):
        raise ValueError('Manifest must belong to a sealed training publication')
    manifest = read_json(manifest_path)
    if (manifest.get('schema') != 'morai_sensor_record_visual_manifest_v2'
            or manifest.get('training_visible') is not True):
        raise ValueError('Unsupported or unpublished Sensor Record manifest')
    resolver = PathResolver()
    episode = resolver(manifest['episode'], manifest_path.parent)
    native = resolver(manifest['native_root'], manifest_path.parent)
    contract_path = episode.parent / 'sensor_contract.json'
    if (digest(contract_path) != manifest['contract_sha256']
            or read_json(contract_path) != manifest['contract']):
        raise ValueError('Sensor contract changed')
    if manifest['contract'].get('preprocessing') != 'lmdrive_compat':
        raise ValueError('This review supports only lmdrive_compat crop geometry')
    signal_dir = episode / 'route_signal_audit_v1'
    verify_seal(signal_dir)
    if digest(signal_dir / 'COMPLETE.json') != manifest['evidence_seal_sha256']['route_signal']:
        raise ValueError('Route-signal evidence changed')
    signals = read_json(signal_dir / 'frames.json')
    by_id = {int(entry['frame_id']): entry for entry in manifest['entries']}
    if len(by_id) != len(manifest['entries']):
        raise ValueError('Duplicate manifest frame IDs')
    return manifest_path, manifest, native, signals, by_id


def review(manifest_path, frame_ids, output):
    path, manifest, native, signals, entries = load_sealed_frame_sources(manifest_path)
    selected = list(dict.fromkeys(frame_ids))
    if not selected or any(frame not in entries for frame in selected):
        raise ValueError('All selected frame IDs must belong to the manifest')
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError('Review output must be new: ' + str(output))
    adapter = MoraiLMDriveVisionAdapter(MoraiVisionAdapterConfig(), rgb_augmentation_policy='none')
    rows, sections = [], []
    for frame_id in selected:
        entry = entries[frame_id]
        signal = signals[frame_id]
        if (int(signal['index']) != frame_id
                or int(signal['sensor_source_ns']) != int(entry['sensor_time_ns'])):
            raise ValueError('Signal frame/time does not match the manifest')
        raw, assets = {}, {}
        for number, view in enumerate(VIEWS, 1):
            relative = entry['asset_roles']['camera_' + str(number)]
            payload = SensorRecordDataset.read_asset(native, entry, relative)
            with Image.open(io.BytesIO(payload)) as image:
                raw[view] = np.asarray(image.convert('RGB')).copy()
            assets[view] = dict(relative_path=relative, sha256=hashlib.sha256(payload).hexdigest(),
                                width=int(raw[view].shape[1]), height=int(raw[view].shape[0]))
        tensors = adapter.camera_tensors(**{view + '_rgb': value for view, value in raw.items()})
        crops = {}
        for key in INPUTS:
            view = 'front' if key == 'rgb_center' else key[4:]
            height, width = raw[view].shape[:2]
            crops[key] = dict(raw_view=view, raw_pixel_rectangle=crop_rectangle(width, height, key),
                              tensor_shape=list(tensors[key].shape),
                              tensor_sha256=hashlib.sha256(tensors[key].numpy().tobytes()).hexdigest())
        row = dict(frame_id=frame_id, sensor_time_ns=entry['sensor_time_ns'],
                   traffic_light=entry['traffic_light'], bound_signal=signal,
                   ego_pose=entry['ego_pose'], assets=assets, model_inputs=crops,
                   lamp_visibility='unverified', lamp_identity_verified=False,
                   occlusion_verified=False, training_label_modified=False)
        rows.append(row)
        title = 'Frame {} · label {} · light {} · stop line {:.2f} m'.format(
            frame_id, entry['traffic_light'], signal.get('light_id'),
            float(signal.get('distance_to_stopline_m') or 0))
        figures = []
        for view, rgb in raw.items():
            figures.append('<figure><figcaption>RAW {}</figcaption><img src="{}"></figure>'.format(view, encoded_image(rgb)))
        for key in INPUTS:
            figures.append('<figure><figcaption>MODEL {}</figcaption><img src="{}"></figure>'.format(key, encoded_image(rgb_from_model_tensor(tensors[key]))))
        sections.append('<section><h2>{}</h2><div class="views">{}</div><details><summary>Frame evidence</summary><pre>{}</pre></details></section>'.format(
            html.escape(title), ''.join(figures), html.escape(json.dumps(row, indent=2, ensure_ascii=False))))
    result = dict(schema='morai_light_input_review_v1', source_manifest=str(path),
                  source_manifest_sha256=digest(path), source_split=manifest['split'],
                  script_sha256=digest(__file__), frames=rows,
                  automatic_visibility_inference=False, training_labels_modified=False,
                  note='Route-bound GT is not proof of image visibility. No lamp-world coordinates or occlusion certificate are assumed.')
    document = '''<!doctype html><meta charset="utf-8"><title>Signal input review</title>
<style>body{font:16px sans-serif;background:#101820;color:#eee;margin:24px}section{border-top:1px solid #678;margin-top:32px}.views{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}figure{margin:0}img{max-width:100%;height:auto;image-rendering:auto}figcaption{padding:8px;color:#ffc857}pre{white-space:pre-wrap}@media(max-width:900px){.views{grid-template-columns:1fr}}</style>
<h1>Signal GT and actual model inputs</h1><p>Read-only diagnostic. Top row group: raw cameras. MODEL panels: exact training preprocessing, normalization inverted for display. Label 0=red, 1=green, -1=masked. Visibility and lamp identity remain unverified; no labels are changed.</p>'''+''.join(sections)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'report.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    (output / 'review.html').write_text(document, encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--frame-id', type=int, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = review(args.manifest, args.frame_id, args.output)
    print(json.dumps(dict(output=str(args.output.resolve()), frames=len(result['frames']),
                          training_labels_modified=False)))


if __name__ == '__main__':
    main()
