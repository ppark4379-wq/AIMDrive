"""Copy a finalized local run to an archive, retaining the local source.

Retry: python3 data_collection/writer/archive.py --local-root /mnt/c/DCM/data_collector --route-dir <local-run> --archive-root /mnt/z/DCM/data_collector
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile


def _atomic_copy(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.copy-', dir=str(target.parent))
    os.close(fd)
    try:
        shutil.copyfile(str(source), temporary)
        os.replace(temporary, str(target))
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _write(target, content):
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.metadata-', dir=str(target.parent))
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            stream.write(content)
        os.replace(temporary, str(target))
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _copy_tree(source, target):
    # Walk and copy sequentially: never load sensor files or the tree into RAM.
    for directory, _, filenames in os.walk(str(source)):
        destination = target / Path(directory).relative_to(source)
        destination.mkdir(parents=True, exist_ok=True)
        for filename in filenames:
            _atomic_copy(Path(directory) / filename, destination / filename)


def archive_route(local_root, route_dir, archive_root):
    local = Path(local_root).expanduser().resolve()
    route = Path(route_dir).expanduser().resolve()
    archive = Path(archive_root).expanduser().resolve()
    relative = route.relative_to(local)
    if len(relative.parts) != 3 or relative.parts[1] != 'data':
        raise ValueError('Expected <local-root>/<job>/data/<run>')
    if local == archive or local in archive.parents or archive in local.parents:
        raise ValueError('Local and archive roots must be separate')
    job = route.parent.parent
    key = 'data/' + route.name + '/'
    summary = json.loads((job / 'results/episode_summary.json').read_text(encoding='utf-8'))
    episode = next((e for e in summary['episodes'] if e['route'] == key), None)
    if not episode or episode['status'] != 'completed':
        raise RuntimeError('Only successfully finalized runs can be archived')
    if str(archive) == '/mnt/z' or str(archive).startswith('/mnt/z/'):
        if not os.path.ismount('/mnt/z'):
            raise RuntimeError('Z: is not mounted at /mnt/z; local data retained. Mount and retry archive.py')
    destination_job = archive / relative.parts[0]
    print('[collect/archive] copying {} -> {} (local files retained)'.format(route, archive / relative), flush=True)
    try:
        _copy_tree(route, archive / relative)
        metadata = json.loads((route / 'route.json').read_text(encoding='utf-8'))
        map_source = (route / metadata['map_path']).resolve()
        map_relative = map_source.relative_to(job)
        _copy_tree(map_source, destination_job / map_relative)
        for filename in ('navigation_instruction_list.txt', 'notice_instruction_list.json'):
            target = destination_job / filename
            if not target.exists():
                _atomic_copy(job / filename, target)
        summary_target = destination_job / 'results/episode_summary.json'
        previous = json.loads(summary_target.read_text(encoding='utf-8')) if summary_target.exists() else {'episodes': []}
        previous['episodes'] = [e for e in previous['episodes'] if e['route'] != key] + [episode]
        _write(summary_target, json.dumps(previous, ensure_ascii=False, indent=2))
        index = destination_job / 'dataset_index.txt'
        lines = index.read_text(encoding='utf-8').splitlines() if index.exists() else []
        lines = [line for line in lines if not line.split() or line.split()[0] != key]
        lines.append('{} {}'.format(key, episode['frames']))
        # Publish the index only after this run's files have been copied.
        _write(index, '\n'.join(lines) + '\n')
    except Exception as exc:
        raise RuntimeError('Archive copy failed; local data retained at {}. Retry archive.py: {}'.format(route, exc)) from exc
    print('[collect/archive] Z:/archive file writes completed; NAS upload completion is managed by the drive provider', flush=True)
    return str(archive / relative)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--local-root', required=True)
    parser.add_argument('--route-dir', required=True)
    parser.add_argument('--archive-root', required=True)
    args = parser.parse_args()
    archive_route(args.local_root, args.route_dir, args.archive_root)
