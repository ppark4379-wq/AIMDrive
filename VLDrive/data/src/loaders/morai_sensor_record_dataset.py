"""Read the audited Sensor Record v2 publication, without ROS or local checkout imports.

Keeps the old v3_lav pixels, point-cloud preprocessing, geometry targets and
masks. Velocity is explicitly GPS, recorded Ego-topic speed, or legacy zero.
Remapping paths never
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
from .morai_encoder_contract import world_target_to_local
from .morai_transfer_targets import render_object_target, OBJECT_TARGET_POLICIES
from .morai_gps_speed import parse_native_gps, causal_gps_speed
from .morai_ego_speed import CausalEgoSpeed


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


def verify_admission_recovery(episode, evidence, recovery_name='ADMISSION_RECOVERY.json'):
    """Verify the additive admission selected by the publication builder."""
    episode = Path(episode).resolve()
    run = episode.parent
    name = Path(str(recovery_name))
    if name.name != str(recovery_name) or name.suffix != '.json':
        raise ValueError('Admission recovery must be a run-local JSON filename')
    keys = ('admission_recovery', 'effective_admission')
    if not any(key in evidence for key in keys):
        if recovery_name != 'ADMISSION_RECOVERY.json':
            raise ValueError('Selected admission recovery lacks sealed evidence')
        return episode / 'admission_v1'
    if not all(key in evidence for key in keys):
        raise ValueError('Incomplete admission recovery evidence')
    recovery_path = child(run, str(recovery_name))
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
            and geometry.get('schema') in (
                'morai_sensor_record_object_geometry_audit_v2',
                'morai_sensor_record_object_geometry_audit_v3',
                'morai_sensor_record_object_geometry_audit_v4',
                'morai_sensor_record_object_geometry_audit_v5')
            and geometry.get('raw_manifest_sha256') == original.get('raw_manifest_sha256')
            and geometry.get('structural_valid') is True
            and corrected.get('input_seals', {}).get('object_geometry_audit')
                == geometry_seal
        ):
            raise ValueError('Changed corrected object geometry evidence')

    return corrected_dir



def verify_subset_selection(index, report):
    """Require an exact partition of admitted and excluded whole laps."""
    selected = report.get('laps', [])
    excluded = report.get('excluded_laps', [])
    indexed = [row['lap'] for split in ('train', 'val', 'test')
               for row in index.get(split, [])]
    captured_invalid = any(row.get('capture_status') == 'succeeded'
                           for row in excluded)
    selected_ids = [row.get('lap') for row in selected]
    excluded_ids = [row.get('lap') for row in excluded]
    if (report.get('subset_policy')
            != 'whole_lap_fail_closed_exclusion_with_full_variation_preservation_v1'
            or report.get('subset_coverage_preserved') is not True
            or report.get('all_captured_laps_valid') is not (not captured_invalid)
            or report.get('selected_laps') != len(indexed)
            or not excluded
            or any(row.get('valid') is not True
                   or row.get('training_usable') is not True for row in selected)
            or any(row.get('valid') is not False for row in excluded)
            or len(set(indexed)) != len(indexed)
            or len(set(selected_ids)) != len(selected_ids)
            or len(set(excluded_ids)) != len(excluded_ids)
            or set(indexed) != set(selected_ids)
            or set(indexed) & set(excluded_ids)):
        raise ValueError('Subset validation does not preserve complete admitted laps')


def verify_train_supplement(index, report, resolver, parent):
    """An explicitly planned train-only collection may supplement held-out campaigns.

    It cannot provide validation itself or reassign a held-out collection. The
    pre-capture split plan must match the sealed campaign validation evidence.
    """
    plan_path = resolver(index['split_plan'], parent)
    plan_hash = digest(plan_path)
    plan = read_json(plan_path)
    planned = plan.get('runs', [])
    admitted = report.get('laps', [])
    rows = index.get('train', [])
    ids = [r.get('run') for r in planned]
    if (index.get('val') or index.get('test') or not rows or not planned
            or plan_hash != index.get('split_plan_sha256')
            or plan_hash != report.get('static_collection_plan_sha256')
            or plan.get('schema') != 'morai_static_collection_plan_v1'
            or plan.get('ready') is not True
            or any(r.get('split') != 'train' for r in planned)
            or len(set(ids)) != len(ids)
            or set(ids) != set(range(1, len(ids) + 1))
            or len(rows) != len(ids) or len(admitted) != len(ids)
            or {r.get('lap') for r in rows} != set(ids)
            or {r.get('lap') for r in admitted} != set(ids)
            or any(r.get('valid') is not True or r.get('training_usable') is not True
                   for r in admitted)):
        raise ValueError('Train supplement must preserve its sealed all-train collection plan')


def load_publication(path, resolver):
    path = resolver(path)
    index = read_json(path)
    allowed_sources = ('morai_sensor_record_campaign_validation_v1',
                       'morai_sensor_record_campaign_validation_subset_v1')
    if (index.get('schema') != 'morai_visual_manifest_index_v1'
            or index.get('training_ready') is not True
            or index.get('source_schema') not in allowed_sources):
        raise ValueError('Requires a published Sensor Record train/val index')
    seal = verify_seal(path.parent)
    if (seal.get('status') != 'passed' or seal.get('training_visible') is not True
            or seal.get('schema') != 'morai_sensor_record_training_publication_v1'
            or seal['files'].get(path.name) != digest(path)):
        raise ValueError('Training index is not covered by publication seal')
    source = resolver(index['source_validation'], path.parent)
    if digest(source) != index['source_validation_sha256']:
        raise ValueError('Source validation changed')
    source_seal = verify_seal(source.parent)
    source_report = read_json(source)
    if (source_report.get('valid') is not True
            or source_report.get('schema') != index['source_schema']
            or source_seal.get('status') != 'passed'
            or source_seal['files'].get(source.name) != digest(source)):
        raise ValueError('Source validation is not a sealed passed report')
    if index['source_schema'] == 'morai_sensor_record_campaign_validation_subset_v1':
        verify_subset_selection(index, source_report)
    supplement = index.get('publication_role') == 'train_supplement'
    if supplement:
        verify_train_supplement(index, source_report, resolver, path.parent)
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
        if not result[split] and not (supplement and split == 'val'):
            raise ValueError('Empty split: ' + split)
    train_episodes = {str(resolver(d['episode'], p.parent)) for p,d in result['train']}
    val_episodes = {str(resolver(d['episode'], p.parent)) for p,d in result['val']}
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
    manifest_scenarios = {}
    for path in paths:
        index, rows = load_publication(path, resolver)
        publications.append({'path': str(resolver(path)), 'sha256': digest(resolver(path))})
        for split in combined:
            for p, doc in rows[split]:
                episode = str(resolver(doc['episode'], p.parent))
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
                manifest_scenarios[str(p)] = index.get('scenario_name')
    if not all(combined.values()):
        raise ValueError('Both train and validation publications are required')
    return {'publications': publications, 'manifest_scenarios': manifest_scenarios,
            'validation_scope': 'Original held-out full laps preserved for each campaign; same-map/route evaluation'}, combined


def verify_geometry_derivation(doc, resolver, parent, episode):
    """Bind additive full-pose labels to sealed audits and their source recording."""
    if doc.get('map_basis_reprocessing') is not None:
        from .morai_map_derivation import verify_map_derivation
        return verify_map_derivation(doc, resolver, parent, episode)
    if doc.get('map_basis') is not None:
        raise ValueError('Map basis lacks verified additive derivation')
    evidence = doc.get('geometry_reprocessing')
    if evidence is None:
        return
    if (evidence.get('schema') != 'morai_fullpose_geometry_derivation_v1'
            or evidence.get('policy') != 'source_admitted_frames_intersect_fullpose_geometry_pass'):
        raise ValueError('Unsupported geometry derivation')
    source = resolver(evidence['source_manifest'], parent)
    if digest(source) != evidence['source_manifest_sha256']:
        raise ValueError('Changed geometry source manifest')
    raw_hash = digest(episode / 'raw/raw_manifest.json')
    directories = {}
    for kind in ('timeline', 'geometry'):
        folder = resolver(evidence[kind + '_directory'], parent)
        if folder.parent != episode:
            raise ValueError('Geometry derivation escapes episode')
        seal = verify_seal(folder)
        if digest(folder/'COMPLETE.json') != evidence[kind + '_seal_sha256']:
            raise ValueError('Changed geometry derivation seal')
        bound_raw_hash = (seal.get('raw_manifest_sha256') if kind == 'timeline'
                          else read_json(folder/'report.json').get('raw_manifest_sha256'))
        if bound_raw_hash != raw_hash:
            raise ValueError('Geometry derivation belongs to another recording')
        directories[kind] = folder
    if read_json(directories['geometry']/'COMPLETE.json').get('gt_timeline_seal_sha256') != digest(directories['timeline']/'COMPLETE.json'):
        raise ValueError('Geometry audit references another GT timeline')
    frames = {row['index']: row for row in read_json(directories['geometry']/'frames.json')}
    original = {row['frame_id']: row for row in read_json(source)['entries']}
    admission = {row['index']: row for row in read_json(episode/'admission_v1/frames.json')}
    for entry in doc['entries']:
        frame = frames.get(entry['frame_id'], {})
        before = original.get(entry['frame_id'])
        if (frame.get('api_box_geometry_candidate') is not True
                or frame.get('lidar_source_ns') != entry['sensor_time_ns']
                or before is None or entry.get('object_valid') is not True):
            raise ValueError('Unapproved geometry frame')
        if not before.get('object_valid'):
            prior = admission.get(entry['frame_id'], {})
            if (prior.get('lidar_source_time_ns') != entry['sensor_time_ns'] or not all(prior.get(k) is True for k in ('dataset_build_candidate','object_gt_transport_timing_candidate','full_actor_inventory_transport_candidate','object_model_identity_candidate'))):
                raise ValueError('Geometry correction cannot approve missing transport or identity')
        for key in set(before) | set(entry):
            if key not in ('objects', 'object_valid', 'object_positive') and before.get(key) != entry.get(key):
                raise ValueError('Geometry derivation changed non-object data: ' + key)


class SensorRecordDataset(Dataset):
    def __init__(self, manifests, resolver, velocity_source='gps', verify_evidence=True,
                 object_target_policy='footprint_v1', include_object_instance_ids=False):
        if velocity_source not in ('gps', 'zero', 'ego_topic'):
            raise ValueError('velocity_source must be gps, ego_topic or explicit legacy zero')
        if object_target_policy not in OBJECT_TARGET_POLICIES:
            raise ValueError('Unsupported object target policy')
        self.object_target_policy = object_target_policy
        self.include_object_instance_ids = bool(include_object_instance_ids)
        from .morai_map_geometry import manifest_map_basis
        self.map_basis = manifest_map_basis(manifests)
        self.velocity_source = velocity_source
        self.rows = []
        self.row_manifest_paths = []
        self.excluded = Counter()
        self.coverage = Counter()
        self.manifest_hashes = {}
        for path, doc in manifests:
            print('[DATA] checking '+str(path), flush=True)
            if (doc.get('schema') != 'morai_sensor_record_visual_manifest_v2'
                    or doc.get('training_visible') is not True):
                raise ValueError('Requires visible Sensor Record v2 manifest')
            self.manifest_hashes[str(path)] = digest(path)
            episode, native = resolver(doc['episode'], path.parent), resolver(doc['native_root'], path.parent)
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
                    episode, doc.get('evidence_seal_sha256', {}),
                    doc.get('admission_recovery_path', 'ADMISSION_RECOVERY.json'))
            verify_geometry_derivation(doc, resolver, path.parent, episode)
            if (contract.get('raw_lidar_axes') != 'FLU' or contract.get('lidar_sweeps') != 1
                    or contract.get('preprocessing') != 'lmdrive_compat'):
                raise ValueError('Unsupported LiDAR input contract')
            extrinsic = contract['lidar_to_base']
            adapter = MoraiLMDriveVisionAdapter(MoraiVisionAdapterConfig(
                use_previous_lidar_sweep=False, lidar_intensity_policy='reflectance_uint8_unit',
                lidar_translation_xyz=tuple(extrinsic['translation_xyz']),
                lidar_rotation_xyzw=tuple(extrinsic['rotation_xyzw'])))
            if velocity_source == 'ego_topic':
                verify_seal(episode/'quality', 'QUALITY_COMPLETE.json')
                if digest(episode/'quality/QUALITY_COMPLETE.json') != doc['evidence_seal_sha256']['quality']:
                    raise ValueError('Changed Ego speed evidence')
                # active_signals holds the same decoded /Ego_topic hypot(vx, vy) as raw/, without ROS.
                ego_speed = CausalEgoSpeed([
                    (int(row['source_timestamp_ns']), float(row['signal']['speed']), row['record_id'])
                    for row in read_json(episode/'quality/active_signals.json')
                    if row.get('role') == 'ego' and row.get('source_timestamp_ns') is not None])
            previous = None
            previous_stamp = None
            for entry_number, entry in enumerate(doc['entries']):
                if entry_number % 1000 == 0:
                    print('[DATA] %s: sensor metadata %d/%d' % (path.name, entry_number, len(doc['entries'])), flush=True)
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
                if velocity_source == 'ego_topic':
                    value, reason = ego_speed.at(stamp)
                    if reason:
                        self.excluded[reason] += 1
                        continue
                    speed = value['velocity_mps']
                nav = entry['navigation']
                validate_navigation_target(nav, doc.get('navigation_contract', {}))
                wp = np.asarray(entry['waypoints'])
                if wp.shape != (5,2) or not np.isfinite(wp).all():
                    raise ValueError('Invalid timed future waypoint targets')
                self.rows.append((native, entry, adapter, speed))
                self.row_manifest_paths.append(str(path))
                self.coverage['frames'] += 1
                self.coverage['object_valid'] += bool(entry['object_valid'])
                self.coverage['object_positive'] += bool(entry['object_positive'])
                self.coverage['traffic_light_%s' % entry['traffic_light']] += 1
            print('[DATA] admitted total=%d; excluded=%s' % (len(self.rows), dict(self.excluded)), flush=True)
        if not self.rows:
            raise ValueError('Empty training dataset after speed/mask validation')

    @staticmethod
    def read_asset(native, entry, rel):
        from .morai_native_asset_io import read_native_bytes
        raw = read_native_bytes(native, rel)
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
        target = world_target_to_local((nav['target_x'], nav['target_y']),
                                      (ego['x'], ego['y']), ego['yaw'])
        if not np.isfinite(target).all() or np.linalg.norm(target)>100:
            raise ValueError('Invalid local route target')
        inputs.update(lidar=torch.from_numpy(lidar), num_points=torch.tensor(num),
                      velocity=torch.tensor([speed],dtype=torch.float32), target_point=torch.from_numpy(target))
        det,valid = np.zeros((2500,8),np.float32), False
        instance_ids = np.full(2500, -1, dtype=np.int64)
        if entry['object_valid']:
            rendered = render_object_target(
                {'ego_raw':ego,'capture':{}}, entry['objects'],
                target_policy=self.object_target_policy,
                return_instance_ids=self.include_object_instance_ids)
            det,valid,_ = rendered[:3]
            if self.include_object_instance_ids:
                instance_ids = rendered[3]
            if not valid:
                raise ValueError('Object geometry validity changed')
        light = int(entry['traffic_light'])
        if light not in (-1,0,1):
            raise ValueError('Invalid traffic-light target')
        targets = dict(frame_id=torch.tensor(entry['frame_id'], dtype=torch.int64),
                       sensor_time_ns=torch.tensor(entry['sensor_time_ns'], dtype=torch.int64),
                       detection=torch.from_numpy(det), object_valid=torch.tensor(valid),
                       waypoint_valid=torch.tensor(True), waypoints=torch.tensor(entry['waypoints'],dtype=torch.float32),
                       traffic_light=torch.tensor(light))
        if self.include_object_instance_ids:
            targets['object_instance_ids'] = torch.from_numpy(instance_ids)
        return inputs, targets
