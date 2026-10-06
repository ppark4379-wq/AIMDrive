"""Publish additive projected-grid targets from sealed full-pose publications.

Run with python -m data.src.processing.correct_morai_map_basis from VLDrive.
Original sensor bytes, source manifests, dimensions, times and splits are retained.
"""
import argparse
import copy
import json
from collections import Counter
from pathlib import Path

from data.src.loaders.morai_map_geometry import BASIS_SCHEMA, validate_map_basis
from data.src.loaders.morai_map_derivation import SCHEMA, correct_entry, verify_map_derivation
from data.src.loaders.morai_sensor_record_dataset import (
    PathResolver, digest, load_publication, read_json, verify_geometry_derivation, verify_seal)


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, separators=(',', ':'), allow_nan=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--map-metadata', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = read_json(args.config)
    resolver = PathResolver(config.get('path_mappings', []))
    # Never resume over a partly written or previously sealed publication.
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    map_path = out/'map_metadata.json'
    map_path.write_bytes(args.map_metadata.read_bytes())
    metadata = read_json(map_path)
    basis = dict(schema=BASIS_SCHEMA, position_frame='projected_grid',
                 source_orientation_frame='true_enu', orientation_frame='projected_grid',
                 crs=metadata['global_coordinate_system'], origin_xyz=metadata['local_origin_in_global'])
    geo = validate_map_basis(basis)
    totals, publications = Counter(), []
    for index_path in config['manifest_indices']:
        src = resolver(index_path)
        index, _ = load_publication(src, resolver)
        dest = out/src.parent.name
        dest.mkdir()
        new_index = copy.deepcopy(index)
        for key in ('source_validation', 'split_plan'):
            if new_index.get(key):
                new_index[key] = str(resolver(new_index[key], src.parent))
        files, counts, coverage, audit_rows = {}, Counter(), {}, []
        for split in ('train', 'val', 'test'):
            new_index[split] = []
            c = Counter()
            for row in index[split]:
                p = resolver(row['path'], src.parent)
                if digest(p) != row['sha256']:
                    raise ValueError('Changed source manifest')
                original = read_json(p)
                if original.get('map_basis') or original.get('map_basis_reprocessing'):
                    raise ValueError('Already converted manifest')
                episode = resolver(original['episode'], p.parent)
                verify_geometry_derivation(original, resolver, p.parent, episode)
                ev = original['geometry_reprocessing']
                ts = {r['index']:r for r in read_json(resolver(ev['timeline_directory'])/'frames.json')}
                gs = {r['index']:r for r in read_json(resolver(ev['geometry_directory'])/'frames.json')}
                verify_seal(episode/'route_signal_audit_v1')
                signals = {r['index']:r for r in read_json(episode/'route_signal_audit_v1/frames.json')}
                pivots = read_json(resolver(ev['geometry_directory'])/'pivot_evidence.json')['instances']
                doc = copy.deepcopy(original)
                entries, stats = [], Counter()
                for before in original['entries']:
                    i = before['frame_id']
                    after, review = correct_entry(before, ts[i], gs[i], geo, signals[i], pivots)
                    entries.append(after)
                    stats['frames'] += 1
                    stats['object_frames_masked'] += bool(review['object_mask_reasons'])
                    stats['light_frames_masked'] += review['light_masked']
                    c['waypoint_frames'] += 1
                    c['navigation_target_frames'] += 1
                    c['object_frames'] += bool(after['object_valid'])
                    c['object_positive_frames'] += after['object_positive']
                    c['traffic_light_'+str(after['traffic_light'])] += 1
                doc['entries'] = entries
                doc['map_basis'] = basis
                doc['map_basis_reprocessing'] = dict(
                    schema=SCHEMA, source_manifest=str(p), source_manifest_sha256=digest(p),
                    map_metadata=str(map_path), map_metadata_sha256=digest(map_path),
                    source_timestamps_modified=False, original_frame_order_and_splits_preserved=True,
                    physical_simultaneity_verified=False, counts=dict(stats),
                    route_policy='preserve_sealed_world_route_targets; conservative_20deg_light_heading_bound',
                    implementation_sha256={name:digest(Path(__file__).parents[1]/'loaders'/name)
                                           for name in ('morai_map_geometry.py','morai_map_derivation.py')})
                # A fresh reconstruction is the acceptance check, including map
                # identity and original audit seals; not a flag set by the writer.
                verify_map_derivation(doc, resolver, dest, episode)
                target = dest/p.name
                write(target, doc)
                files[target.name] = digest(target)
                new_row = dict(row, path=target.name, sha256=files[target.name], frames=len(entries))
                new_index[split].append(new_row)
                counts[split] += len(entries)
                audit_rows.append(dict(manifest=target.name, split=split, **dict(stats)))
                print(json.dumps(dict(publication=dest.name, manifest=target.name, **dict(stats))), flush=True)
            coverage[split] = dict(c)
        new_index['counts'] = dict(counts)
        new_index['coverage'] = coverage
        new_index['map_basis'] = basis
        new_index['map_basis_source_index'] = dict(path=str(src), sha256=digest(src))
        write(dest/'map_basis_audit.json', dict(manifests=audit_rows, map_basis=basis,
              limitation='Coordinate-basis correction; does not certify every sensor exposure or object surface.'))
        write(dest/'manifest_index.json', new_index)
        for name in ('manifest_index.json', 'map_basis_audit.json'):
            files[name] = digest(dest/name)
        write(dest/'COMPLETE.json', dict(schema='morai_sensor_record_training_publication_v1',
              status='passed', training_visible=True, files=files))
        load_publication(dest/'manifest_index.json', resolver)
        totals.update(counts)
        publications.append(dict(path=str(dest/'manifest_index.json'), counts=dict(counts),
                                 object_frames_masked=sum(r['object_frames_masked'] for r in audit_rows),
                                 light_frames_masked=sum(r['light_frames_masked'] for r in audit_rows)))
    updated = dict(config, manifest_indices=[p['path'] for p in publications])
    write(out/'morai_transfer.grid.json', updated)
    write(out/'summary.json', dict(status='completed', counts=dict(totals), total=sum(totals.values()),
          publications=publications, map_basis=basis, source_config=str(args.config.resolve()),
          source_config_sha256=digest(args.config),
          original_sensor_bytes_and_timestamps_preserved=True, checkpoint_retrained=False))
    print('COMPLETE '+str(out/'summary.json'), flush=True)


if __name__ == '__main__':
    main()
