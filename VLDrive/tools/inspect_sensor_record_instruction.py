#!/usr/bin/env python3
"""Look up the route-derived instruction paired with a Sensor Record image.

This is an audit utility.  It never infers an instruction from image pixels;
it resolves the image to a sealed manifest frame and then finds the semantic
clip generated from that frame's official-route progress.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


def local_path(value: str) -> Path:
    """Accept either a WSL path or a Windows drive path."""
    value = value.strip().strip('"')
    match = re.match(r"^([A-Za-z]):[\\/](.*)$", value)
    if match:
        drive, tail = match.groups()
        value = "/mnt/{}/{}".format(drive.lower(), tail.replace("\\", "/"))
    return Path(value).expanduser().resolve()


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True,
                        help="dataset image path in Windows or WSL form")
    parser.add_argument("--clip-index", type=Path, required=True,
                        help="directory containing clips.jsonl")
    args = parser.parse_args()

    image = local_path(args.image)
    if not image.is_file():
        raise SystemExit("[FAIL] image does not exist: {}".format(image))
    clips_path = args.clip_index.expanduser().resolve()
    if clips_path.is_dir():
        clips_path = clips_path / "clips.jsonl"
    if not clips_path.is_file():
        raise SystemExit("[FAIL] clips.jsonl does not exist: {}".format(clips_path))

    clips = list(read_jsonl(clips_path))
    image_sha256 = sha256_file(image)
    by_manifest = {}
    for clip in clips:
        by_manifest.setdefault(str(Path(clip["source_manifest"]).resolve()), []).append(clip)

    matches = []
    checked_manifests = 0
    for manifest_key, manifest_clips in by_manifest.items():
        manifest_path = Path(manifest_key)
        if not manifest_path.is_file():
            continue
        checked_manifests += 1
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        native_root = Path(manifest["native_root"]).expanduser().resolve()
        try:
            image_relative = image.relative_to(native_root).as_posix()
        except ValueError:
            image_relative = None
        episode = str(manifest_clips[0]["episode_id"])
        for frame in manifest["entries"]:
            for role, relative in frame.get("asset_roles", {}).items():
                if not str(role).startswith("camera_"):
                    continue
                relative = Path(relative).as_posix()
                exact_path = relative == image_relative
                same_bytes = frame.get("assets", {}).get(relative) == image_sha256
                if not exact_path and not same_bytes:
                    continue
                frame_id = int(frame["frame_id"])
                containing = [clip for clip in manifest_clips
                              if str(clip["episode_id"]) == episode
                              and int(clip["start_frame_id"]) <= frame_id
                              <= int(clip["stop_frame_id"])]
                if not containing:
                    matches.append({
                        "image": str(image), "camera": role,
                        "match_method": "path" if exact_path else "sha256",
                        "episode_id": episode, "frame_id": frame_id,
                        "instruction_available": False,
                        "reason": "frame is not inside a validated semantic clip",
                    })
                    continue
                for clip in containing:
                    matches.append({
                        "image": str(image), "camera": role,
                        "match_method": "path" if exact_path else "sha256",
                        "episode_id": episode, "frame_id": frame_id,
                        "clip_id": str(clip["clip_id"]),
                        "clip_range": [int(clip["start_frame_id"]),
                                       int(clip["stop_frame_id"])],
                        "command": int(clip["command"]),
                        "instruction": str(clip["instruction"]),
                        "instruction_available": True,
                        "instruction_source": str(clip["instruction_source"]),
                    })

    if not matches:
        raise SystemExit(
            "[FAIL] image is not referenced by this sealed clip set "
            "(checked {} source manifests): {}".format(checked_manifests, image))
    print(json.dumps({"matches": matches}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
