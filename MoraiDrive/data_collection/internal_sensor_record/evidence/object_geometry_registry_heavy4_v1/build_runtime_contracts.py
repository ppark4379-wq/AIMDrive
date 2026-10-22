"""Pin heavy-vehicle raw API geometry from the selected calibration only."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, '/home/dydwns/AIMDrive/MoraiDrive/data_collection/internal_sensor_record')
from morai_pretraining.center_geometry import pivot_local_offset


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--profiles', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--additional-source', type=Path, action='append', default=[])
    a = p.parse_args()
    profiles = read(a.profiles)
    sources = [a.source] + a.additional_source
    rows = []
    skipped = []
    for source in sources:
        report = read(source / 'report.json')
        if not report['execution_complete'] or report.get('fatal_error'):
            raise ValueError('Incomplete calibration')
        for row in report['results']:
            if row['status'] == 'FAILED':
                skipped.append(dict(source=str(source), model=row['model'], reason=row.get('error')))
            else:
                rows.append((source, row))
    models = {}
    for source, row in rows:
        if row['model'] not in profiles['models']:
            continue
        if row['kind'] != 'vehicle' or row['status'] == 'FAILED':
            raise ValueError('Failed or nonvehicle calibration')
        for path, digest in row['evidence_sha256'].items():
            if sha(source / path) != digest:
                raise ValueError('Changed calibration evidence: ' + path)
        if row['model'] in models:
            raise ValueError('Multiple contract sources for a model require explicit aggregation')
        actors = [read(source / s['directory'] / 'gt.json')['before'] for s in row['samples']]
        offsets = np.asarray([pivot_local_offset(actor) for actor in actors])
        sizes = np.asarray([[actor['bounding_box']['extent'][k] * 2 for k in 'yxz'] for actor in actors])
        if len(actors) < 8 or np.max(np.ptp(offsets, axis=0)) >= .01 or np.max(np.ptp(sizes, axis=0)) >= .01:
            raise ValueError('Unstable raw API geometry: ' + row['model'])
        models[row['model']] = dict(
            kind='vehicle', raw_offset_flu_m=np.median(offsets, axis=0).tolist(),
            raw_size_lwh_m=np.median(sizes, axis=0).tolist(),
            offset_range_xyz_m=np.ptp(offsets, axis=0).tolist(),
            size_range_lwh_m=np.ptp(sizes, axis=0).tolist(), views=len(actors),
            source_report=str(source / 'report.json'), source_sha256=sha(source / 'report.json'))
    if set(models) != set(profiles['models']):
        raise ValueError('Profile/model mismatch')
    result = dict(schema='morai_dynamic_current_api_contract_v1',
                  surface_profiles_sha256=sha(a.profiles), bounds_unchanged=True,
                  source_reports=[dict(path=str(source / 'report.json'),
                                       sha256=sha(source / 'report.json')) for source in sources],
                  skipped_failed_model_captures=skipped, models=models)
    with a.output.open('x') as f:
        json.dump(result, f, indent=2)
    print('Raw API contracts:', len(models))


if __name__ == '__main__':
    main()
