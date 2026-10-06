"""CLI for the new instruction package; does not modify legacy navigation data."""
import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from instruction.global_path.preprocess.load import load_episodes
from instruction.global_path.preprocess.process import process_path
from instruction.global_path.encoder import PathEncoder


def build_path_features(bundle, output, *, geometry_options=None, encoder_options=None,
                        instruction_options=None, max_sensor_gap_ms=150.):
    """Project each recorded pose chronologically. Never consume GT waypoints."""
    if not math.isfinite(max_sensor_gap_ms) or max_sensor_gap_ms <= 0:
        raise ValueError('max_sensor_gap_ms must be positive')
    episodes, digest = load_episodes(bundle)
    result = dict(schema='frame-path-cache-v1', labels_manifest_sha256=digest,
                  max_sensor_gap_ms=max_sensor_gap_ms, encoder_options=encoder_options or {}, episodes=[])
    instruction = instruction_options or {}
    from instruction.navigation.selector import INSTRUCTION_OPTION_KEYS
    if set(instruction) - INSTRUCTION_OPTION_KEYS:
        raise ValueError('Unknown navigation instruction option')
    if instruction:
        result['instruction_options'] = instruction
    for episode in episodes:
        route = process_path(episode['route'], **(geometry_options or {}))
        encoder = PathEncoder(route, **(encoder_options or {}))
        if instruction:
            # Validate selection options before creating any output artifact.
            from instruction.navigation.selector import select_instruction
            select_instruction(encoder, [dict(progress_m=0.)], **instruction)
        previous_s = previous = None
        frames = []
        for frame in episode['frames']:
            index, stamp = frame['index'], frame['sensor_source_ns']
            if type(index) is not int or type(stamp) is not int or index < 0 or stamp < 0:
                raise ValueError('Invalid source frame identity')
            if previous is not None:
                if index <= previous['index'] or stamp <= previous['sensor_source_ns']:
                    raise ValueError('Frames must be strictly chronological')
                if index != previous['index'] + 1 or stamp - previous['sensor_source_ns'] > max_sensor_gap_ms * 1e6:
                    previous_s = None
            projection = encoder.project(frame, previous_s)
            previous_s = projection['progress_m'] if projection else None
            # Retain input pose for auditing, never future_trajectory/teacher data.
            pose = {key: frame.get(key) for key in ('x', 'y', 'yaw_rad')}
            frames.append(dict(index=index, sensor_source_ns=stamp, pose=pose, projection=projection))
            previous = frame
        result['episodes'].append(dict(episode_id=episode['episode_id'], route=route, frames=frames))
    if not result['episodes']:
        raise ValueError('No episodes to preprocess')
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive create: never overwrite an existing cache or a source artifact.
    payload = json.dumps(result, allow_nan=False, sort_keys=True) + '\n'
    with target.open('x', encoding='utf-8') as stream:
        stream.write(payload)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--instructions', required=True, type=Path, help='Existing verified navigation bundle')
    parser.add_argument('--output', required=True, type=Path, help='New JSON path feature cache')
    parser.add_argument('--options', type=Path, help='JSON with geometry, encoder, max_sensor_gap_ms options')
    args = parser.parse_args()
    options = json.loads(args.options.read_text(encoding='utf-8')) if args.options else {}
    if set(options) - {'geometry', 'encoder', 'instruction', 'max_sensor_gap_ms'}:
        parser.error('Unknown path preprocessing option')
    result = build_path_features(args.instructions, args.output, geometry_options=options.get('geometry'),
        encoder_options=options.get('encoder'), instruction_options=options.get('instruction'),
        max_sensor_gap_ms=options.get('max_sensor_gap_ms', 150.))
    print(json.dumps(dict(output=str(args.output), episodes=len(result['episodes']),
                         frames=sum(len(e['frames']) for e in result['episodes']))))


if __name__ == '__main__':
    main()
