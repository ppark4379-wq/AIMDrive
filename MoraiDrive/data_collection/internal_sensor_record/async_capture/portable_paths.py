"""Resolve immutable capture paths after a verified cross-PC transfer."""
import json
from pathlib import Path

from async_capture.journal import digest


def recorded_episode_directory(recorded_path, episode, leaf_name, ssd_root):
    """Return the capture path or its integrity-checked local episode mirror.

    Evidence files keep absolute paths from the acquisition PC.  Rewriting
    those sealed files would destroy provenance, so transferred runs resolve
    the local directory only when the run/episode/leaf identity and raw seal
    agree.
    """
    episode=Path(episode).resolve();root=Path(ssd_root).resolve()
    recorded=Path(recorded_path)
    if leaf_name not in ('raw',):
        raise ValueError('Unsupported transferred episode directory: '+str(leaf_name))
    try:
        recorded.resolve().relative_to(root)
        if not recorded.is_dir():
            raise ValueError('Recorded episode directory is missing: '+str(recorded))
        return recorded.resolve()
    except ValueError as outside:
        expected=(episode.parent.name,episode.name,leaf_name)
        if tuple(recorded.parts[-3:]) != expected:
            raise ValueError('Recorded path identity differs from transferred episode') from outside
        local=episode/leaf_name
        seal_path=local/'RAW_COMPLETE.json';manifest_path=local/'raw_manifest.json'
        if not local.is_dir() or not seal_path.is_file() or not manifest_path.is_file():
            raise ValueError('Transferred raw mirror is incomplete') from outside
        seal=json.loads(seal_path.read_text())
        if (seal.get('status')!='completed'
                or seal.get('manifest_sha256')!=digest(manifest_path)):
            raise ValueError('Transferred raw mirror seal does not match its manifest')
        return local.resolve()
