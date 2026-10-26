#!/usr/bin/env python3
"""Compact generated static presets into byte-preserving suite catalogs.

By default, source YAML/manifest files remain in place. --remove-loose verifies
all original hashes against the catalog before deleting the source files.
"""
import argparse
import hashlib
import json
from pathlib import Path

from scenario_runner.utils import static_preset_store as store


def compact(directory, remove_loose=False):
    result = store.compact_suite(directory)
    if remove_loose:
        for value, expected in result["source_sha256"].items():
            path = Path(value)
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != expected:
                raise ValueError("Source changed during compaction: " + value)
        # Read packed bytes independently of loose-file precedence before any
        # source is removed. _load already validates every embedded checksum.
        packed = store._load(Path(result["catalog"]))
        for value, expected in result["source_sha256"].items():
            if hashlib.sha256(packed[Path(value).name]).hexdigest() != expected:
                raise ValueError("Catalog differs from source: " + value)
        for value in result["source_sha256"]:
            Path(value).unlink()
        directory = Path(directory)
        if directory.is_dir():
            directory.rmdir()
    result["loose_files_removed"] = bool(remove_loose)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", nargs="*", help="Suite name or directory; omit to compact every suite")
    parser.add_argument("--remove-loose", action="store_true", help="Remove verified loose sources after compaction")
    args = parser.parse_args()
    if args.suite:
        directories = [Path(value) if Path(value).parent != Path(".") else store.GENERATED_ROOT / value
                       for value in args.suite]
    else:
        directories = sorted(path for path in store.GENERATED_ROOT.iterdir() if path.is_dir())
    try:
        results = [compact(path, args.remove_loose) for path in directories]
    except (OSError, ValueError, UnicodeError) as error:
        parser.exit(1, str(error) + "\n")
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
