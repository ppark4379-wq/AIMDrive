"""Read the audited Sensor Record v2 publication, without ROS or local checkout imports.

Keeps the old v3_lav pixels, point-cloud preprocessing, geometry targets and
masks. GPS interval speed is the explicit new input. Remapping paths never
rewrites signed manifests; all hashes still refer to their original bytes.
"""
import hashlib
import io
import json
import math
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from torch.utils.data import Dataset
from .morai_transfer_adapter import MoraiLMDriveVisionAdapter, MoraiVisionAdapterConfig
from .morai_transfer_targets import render_object_target
from .morai_gps_speed import parse_native_gps, causal_gps_speed


CAMPAIGN_VALIDATION_SCHEMAS = {
    'morai_sensor_record_campaign_validation_v1',
    'morai_sensor_record_campaign_validation_subset_v1',
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def child(root, name):
    name = Path(name)
    root = Path(root).resolve()
    path = (root / name).resolve()
    if name.is_absolute() or '..' in name.parts or root not in path.parents:
        raise ValueError('Unsafe asset path: ' + str(name))
    return path


def validate_relative_name(name):
    """Check catalog syntax without thousands of redundant Windows stat RPCs.

    Containment through symlinks is checked by child() when each asset is read.
    The previously sealed native audit covers the full source byte inventory.
    """
    p = Path(name)
    if p.is_absolute() or '..' in p.parts or not p.parts:
        raise ValueError('Unsafe asset path: ' + str(name))


def validate_navigation_target(nav, contract):
    """Validate an official route target, including a proven closed-loop wrap.

    A lower target index is valid only near the seam of a manifest whose
    builder explicitly identified the audited global path as closed. Older
    publications without this provenance remain accepted while their target
    index advances normally.
    """
    if (not isinstance(nav, dict) or nav.get('route_target_valid') is not True
            or nav.get('target_source') != 'official_route_lookahead'):
        raise ValueError('Invalid official navigation target')
    route_index = nav.get('route_index')
    target_index = nav.get('target_route_index')
    if (not isinstance(route_index, int) or isinstance(route_index, bool)
            or not isinstance(target_index, int) or isinstance(target_index, bool)
            or route_index < 0 or target_index < 0):
        raise ValueError('Invalid official navigation target')
    topology = contract.get('route_topology')
    point_count = contract.get('route_point_count')
    if point_count is not None:
        if (not isinstance(point_count, int) or isinstance(point_count, bool)
                or point_count < 2 or route_index >= point_count
                or target_index >= point_count):
            raise ValueError('Invalid official navigation target')
    if target_index < route_index:
        # A real wrap crosses the route seam, so its raw index difference is
        # larger than half the closed route. This still rejects a corrupted
        # short backward jump within an otherwise closed route.
        if (topology != 'closed_loop' or point_count is None
                or route_index - target_index <= point_count // 2):
            raise ValueError('Invalid official navigation target')
    for key in ('target_x', 'target_y', 'distance_to_route_m'):
        value = nav.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
            raise ValueError('Invalid official navigation target')


class PathResolver:
    def __init__(self, mappings=()):
        self.mappings = sorted([(str(a).rstrip('/'), str(b).rstrip('/')) for a,b in mappings],
                               key=lambda row: len(row[0]), reverse=True)

    def __call__(self, value, parent=None):
        value = str(value)
        for old, new in self.mappings:
            if value == old or value.startswith(old + '/'):
                value = new + value[len(old):]
                break
        p = Path(value).expanduser()
        return ((Path(parent) / p) if not p.is_absolute() and parent else p).resolve()


def verify_seal(folder, seal_name='COMPLETE.json'):
    seal = read_json(Path(folder) / seal_name)
    if not isinstance(seal.get('files'), dict) or not seal['files']:
        raise ValueError('Missing seal members: ' + str(folder))
    for name, expected in seal['files'].items():
        if Path(name).name != name or digest(child(folder, name)) != expected:
            raise ValueError('Changed sealed evidence: ' + name)
    return seal


def verify_admission_recovery(episode, evidence):
    """Verify the additive admission selected by the publication builder."""
    episode = Path(episode).resolve()
    run = episode.parent
    keys = ('admission_recovery', 'effective_admission')
    if not any(key in evidence for key in keys):
        return episode / 'admission_v1'
    if not all(key in evidence for key in keys):
        raise ValueError('Incomplete admission recovery evidence')
    recovery_path = run / 'ADMISSION_RECOVERY.json'
    if (not recovery_path.is_file()
            or digest(recovery_path) != evidence['admission_recovery']):
        raise ValueError('Changed admission recovery record')
    recovery = read_json(recovery_path)
    original_path = episode / 'admission_v1/report.json'
    if not (
        recovery.get('schema') == 'morai_sensor_record_admission_recovery_v1'
        and recovery.get('status') == 'completed'
        and recovery.get('recovery_is_additive') is True
        and recovery.get('source_timestamps_modified') is False
        and recovery.get('training_visible') is False
        and recovery.get('original_admission', {}).get('sha256') == digest(original_path)
    ):
        raise ValueError('Invalid admission recovery record')
    corrected_path = child(run, recovery.get('corrected_admission', {}).get('path', ''))
    if corrected_path.name != 'report.json':
        raise ValueError('Invalid corrected admission path')
    corrected_dir = corrected_path.parent
    verify_seal(corrected_dir)
    corrected = read_json(corrected_path)
    original = read_json(original_path)
    if not (
        recovery['corrected_admission'].get('sha256') == digest(corrected_path)
        and recovery['corrected_admission'].get('seal_sha256')
            == digest(corrected_dir / 'COMPLETE.json')
        and evidence['effective_admission'] == digest(corrected_dir / 'COMPLETE.json')
        and corrected.get('raw_manifest_sha256') == original.get('raw_manifest_sha256')
        and corrected.get('capture_contract_valid') is True
        and corrected.get('runtime_streams_valid') is True
        and corrected.get('sensor_capture_status') == 'admitted'
        and corrected.get('teacher_transport_status') == 'admitted'
        and corrected.get('source_timestamps_modified') is False
        and corrected.get('training_visible') is False
    ):
        raise ValueError('Changed corrected admission evidence')
    geometry_expected = evidence.get('effective_object_geometry')
    if geometry_expected is not None:
        geometry_path = child(
            run, recovery.get('corrected_object_geometry', {}).get('path', ''))
        if geometry_path.name != 'report.json':
            raise ValueError('Invalid corrected object geometry path')
        geometry_dir = geometry_path.parent
        verify_seal(geometry_dir)
        geometry = read_json(geometry_path)
        geometry_seal = digest(geometry_dir / 'COMPLETE.json')
        if not (
            recovery['corrected_object_geometry'].get('sha256') == digest(geometry_path)
            and recovery['corrected_object_geometry'].get('seal_sha256') == geometry_seal
            and geometry_expected == geometry_seal
            and geometry.get('schema') == 'morai_sensor_record_object_geometry_audit_v2'
            and geometry.get('raw_manifest_sha256') == original.get('raw_manifest_sha256')
            and geometry.get('structural_valid') is True
            and corrected.get('input_seals', {}).get('object_geometry_audit')
                == geometry_seal
        ):
            raise ValueError('Changed corrected object geometry evidence')
    return corrected_dir


def verify_publication_source(index, source_report, source_seal):
    """Accept strict or fail-closed whole-lap-subset validation evidence."""
    schema = index.get('source_schema')
    if schema not in CAMPAIGN_VALIDATION_SCHEMAS:
        raise ValueError('Unsupported Sensor Record validation schema')
    if not (
        source_report.get('schema') == schema
        and source_seal.get('schema') == schema
        and source_report.get('valid') is True
        and source_seal.get('status') == 'passed'
    ):
        raise ValueError('Source validation schema or status mismatch')
    if schema == 'morai_sensor_record_campaign_validation_subset_v1':
        if not (
            source_report.get('subset_coverage_preserved') is True
            and source_report.get('subset_policy')
                == 'whole_lap_fail_closed_exclusion_with_full_variation_preservation_v1'
            and isinstance(source_report.get('selected_laps'), int)
            and source_report['selected_laps'] >= 2
            and len(source_report.get('laps', [])) == source_report['selected_laps']
            and bool(source_report.get('excluded_laps'))
        ):
            raise ValueError('Invalid fail-closed validation subset')


def load_publication(path, resolver):
    path = resolver(path)
    index = read_json(path)
    if (index.get('schema') != 'morai_visual_manifest_index_v1'
            or index.get('training_ready') is not True
            or index.get('source_schema') not in CAMPAIGN_VALIDATION_SCHEMAS):
        raise ValueError('Requires a published Sensor Record train/val index')
    seal = verify_seal(path.parent)
    if (seal.get('status') != 'passed' or seal.get('training_visible') is not True
            or seal.get('schema') != 'morai_sensor_record_training_publication_v1'
            or seal['files'].get(path.name) != digest(path)):
        raise ValueError('Training index is not covered by publication seal')
    source = resolver(index['source_validation'])
    if digest(source) != index['source_validation_sha256']:
        raise ValueError('Source validation changed')
    source_seal = verify_seal(source.parent)
    source_report = read_json(source)
    verify_publication_source(index, source_report, source_seal)
    if source_seal['files'].get(source.name) != digest(source):
        raise ValueError('Source validation is not a sealed passed report')
    result = {}
    for split in ('train', 'val'):
        result[split] = []
        for row in index[split]:
            p = resolver(row['path'], path.parent)
            if digest(p) != row['sha256']:
                raise ValueError('Indexed manifest changed: ' + str(p))
            doc = read_json(p)
            if len(doc['entries']) != row['frames']:
                raise ValueError('Manifest frame count mismatch')
            if (doc.get('campaign_validation_sha256')
                    != index['source_validation_sha256']):
                raise ValueError('Manifest source validation mismatch')
            result[split].append((p, doc))
        if not result[split]:
            raise ValueError('Empty split: ' + split)
    train_episodes = {str(resolver(d['episode'])) for _,d in result['train']}
    val_episodes = {str(resolver(d['episode'])) for _,d in result['val']}
    if train_episodes & val_episodes:
        raise ValueError('Train/validation episodes overlap')
    def assets(split):
        return {tuple(e['assets'][e['asset_roles'][r]] for r in
                      ('camera_1','camera_2','camera_3','camera_4','lidar'))
                for _,d in result[split] for e in d['entries']}
    if assets('train') & assets('val'):
        raise ValueError('Train/validation sensor bundles overlap')
    return index, result


def load_publications(paths, resolver):
    """Preserve each publication's split and reject cross-campaign leakage.

    Identical manifests are counted once. Changed revisions of the same episode
    must be selected explicitly rather than concatenated or silently replaced.
    """
    combined = {'train': [], 'val': []}
    episodes, bundles, publications = {}, {}, []
    for path in paths:
        index, rows = load_publication(path, resolver)
        publications.append({'path': str(resolver(path)), 'sha256': digest(resolver(path))})
        for split in combined:
            for p, doc in rows[split]:
                episode = str(resolver(doc['episode']))
                identity = (split, digest(p))
                if episode in episodes:
                    if episodes[episode] != identity:
                        raise ValueError('Episode duplicated with another split/revision: ' + episode)
                    continue
                for entry in doc['entries']:
                    key = tuple(entry['assets'][entry['asset_roles'][role]] for role in
                                ('camera_1','camera_2','camera_3','camera_4','lidar'))
                    if key in bundles and bundles[key] != episode:
                        raise ValueError('Sensor bundle duplicated across episodes: ' + episode)
                    bundles[key] = episode
                episodes[episode] = identity
                combined[split].append((p, doc))
    if not all(combined.values()):
        raise ValueError('Both train and validation publications are required')
    return {'publications': publications,
            'validation_scope': 'Original held-out full laps preserved for each campaign; same-map/route evaluation'}, combined


class SensorRecordDataset(Dataset):
    def __init__(self, manifests, resolver, velocity_source='gps', verify_evidence=True):
        if velocity_source not in ('gps', 'zero'):
            raise ValueError('velocity_source must be gps or explicit legacy zero')
        self.velocity_source = velocity_source
        self.rows = []
        self.excluded = Counter()
        self.coverage = Counter()
        self.manifest_hashes = {}
        for path, doc in manifests:
            print('[DATA] checking '+str(path), flush=True)
            if (doc.get('schema') != 'morai_sensor_record_visual_manifest_v2'
                    or doc.get('training_visible') is not True):
                raise ValueError('Requires visible Sensor Record v2 manifest')
            self.manifest_hashes[str(path)] = digest(path)
            episode, native = resolver(doc['episode']), resolver(doc['native_root'])
            contract = doc['contract']
            contract_path = episode.parent/'sensor_contract.json'
            if digest(contract_path) != doc['contract_sha256'] or read_json(contract_path) != contract:
                raise ValueError('Changed sensor contract')
            if verify_evidence:
                for key, folder, seal in (
                    ('quality','quality','QUALITY_COMPLETE.json'),
                    ('native_audit','native_audit','COMPLETE.json'),
                    ('gt_timeline','gt_timeline_v1','COMPLETE.json'),
                    ('object_geometry','object_geometry_audit_v1','COMPLETE.json'),
                    ('route_signal','route_signal_audit_v1','COMPLETE.json'),
                    ('motion_time','motion_time_v1','COMPLETE.json'),
                    ('admission','admission_v1','COMPLETE.json')):
                    verify_seal(episode/folder, seal)
                    if digest(episode/folder/seal) != doc['evidence_seal_sha256'][key]:
                        raise ValueError('Changed episode evidence: ' + key)
                verify_admission_recovery(
                    episode, doc.get('evidence_seal_sha256', {}))
            if (contract.get('raw_lidar_axes') != 'FLU' or contract.get('lidar_sweeps') != 1
                    or contract.get('preprocessing') != 'lmdrive_compat'):
                raise ValueError('Unsupported LiDAR input contract')
            extrinsic = contract['lidar_to_base']
            adapter = MoraiLMDriveVisionAdapter(MoraiVisionAdapterConfig(
                use_previous_lidar_sweep=False, lidar_intensity_policy='reflectance_uint8_unit',
                lidar_translation_xyz=tuple(extrinsic['translation_xyz']),
                lidar_rotation_xyzw=tuple(extrinsic['rotation_xyzw'])))
            previous = None
            previous_stamp = None
            for entry_number, entry in enumerate(doc['entries']):
                if entry_number % 1000 == 0:
                    print('[DATA] %s: GPS/metadata %d/%d' % (path.name, entry_number, len(doc['entries'])), flush=True)
                stamp = int(entry['sensor_time_ns'])
                if previous_stamp is not None and stamp <= previous_stamp:
                    raise ValueError('Nonadvancing episode sensor time')
                previous_stamp = stamp
                roles = entry['asset_roles']
                if set(roles) != {'camera_1','camera_2','camera_3','camera_4','lidar','gps','imu'}:
                    raise ValueError('Incomplete seven-sensor bundle')
                for rel in roles.values():
                    if rel not in entry['assets'] or rel not in entry['asset_sizes']:
                        raise ValueError('Unhashed input asset')
                    validate_relative_name(rel)
                speed = 0.0
                if velocity_source == 'gps':
                    raw = self.read_asset(native, entry, roles['gps'])
                    fix = (stamp, *parse_native_gps(raw.decode('utf-8-sig')))
                    if previous is None:
                        self.excluded['gps_no_previous_fix'] += 1
                        previous = fix
                        continue
                    try:
                        speed = causal_gps_speed(previous, fix)
                    except ValueError:
                        self.excluded['gps_invalid_interval'] += 1
                        previous = fix
                        continue
                    previous = fix
                nav = entry['navigation']
                validate_navigation_target(nav, doc.get('navigation_contract', {}))
                wp = np.asarray(entry['waypoints'])
                if wp.shape != (5,2) or not np.isfinite(wp).all():
                    raise ValueError('Invalid timed future waypoint targets')
                self.rows.append((native, entry, adapter, speed))
                self.coverage['frames'] += 1
                self.coverage['object_valid'] += bool(entry['object_valid'])
                self.coverage['object_positive'] += bool(entry['object_positive'])
                self.coverage['traffic_light_%s' % entry['traffic_light']] += 1
            print('[DATA] admitted total=%d; excluded=%s' % (len(self.rows), dict(self.excluded)), flush=True)
        if not self.rows:
            raise ValueError('Empty training dataset after GPS/mask validation')

    @staticmethod
    def read_asset(native, entry, rel):
        raw = child(native, rel).read_bytes()
        if len(raw) != entry['asset_sizes'][rel] or hashlib.sha256(raw).hexdigest() != entry['assets'][rel]:
            raise ValueError('Changed native asset: ' + rel)
        return raw

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        from PIL import Image
        native, entry, adapter, speed = self.rows[index]
        # Validate and decode the same bytes; all seven roles are integrity-checked.
        payloads = {role:self.read_asset(native,entry,rel) for role,rel in entry['asset_roles'].items()}
        rgb = {}
        for role,view in zip(('camera_1','camera_2','camera_3','camera_4'),('front','left','right','rear')):
            with Image.open(io.BytesIO(payloads[role])) as im:
                rgb[view+'_rgb'] = np.asarray(im.convert('RGB')).copy()
        inputs = adapter.camera_tensors(**rgb)
        raw = payloads['lidar']
        if not raw or len(raw)%16:
            raise ValueError('Incomplete native XYZI')
        points = np.frombuffer(raw, dtype='<f4').reshape(-1,4).copy()
        if not np.isfinite(points).all():
            raise ValueError('Nonfinite native LiDAR')
        lidar,num = adapter.lidar_to_raw_features(points, rng_seed=entry['frame_id'])
        if num == 0:
            raise ValueError('Empty LiDAR after preprocessing')
        ego,nav = entry['ego_pose'],entry['navigation']
        dx,dy=nav['target_x']-ego['x'],nav['target_y']-ego['y']
        c,s=math.cos(ego['yaw']),math.sin(ego['yaw'])
        target = np.asarray([dx*s-dy*c, -dx*c-dy*s],dtype=np.float32)
        if not np.isfinite(target).all() or np.linalg.norm(target)>100:
            raise ValueError('Invalid local route target')
        inputs.update(lidar=torch.from_numpy(lidar), num_points=torch.tensor(num),
                      velocity=torch.tensor([speed],dtype=torch.float32), target_point=torch.from_numpy(target))
        det,valid = np.zeros((2500,8),np.float32), False
        if entry['object_valid']:
            det,valid,_ = render_object_target({'ego_raw':ego,'capture':{}},entry['objects'])
            if not valid:
                raise ValueError('Object geometry validity changed')
        light = int(entry['traffic_light'])
        if light not in (-1,0,1):
            raise ValueError('Invalid traffic-light target')
        targets = dict(detection=torch.from_numpy(det), object_valid=torch.tensor(valid),
                       waypoint_valid=torch.tensor(True), waypoints=torch.tensor(entry['waypoints'],dtype=torch.float32),
                       traffic_light=torch.tensor(light))
        return inputs, targets
