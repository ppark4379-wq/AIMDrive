#!/usr/bin/env python3
"""Write <root>/dataset_index.txt for a folder of LMDrive CARLA routes.

    python tools/preprocess/build_dataset_index.py data/raw/lmdrive_carla

Lists every route under <root>/data with at least --min-frames frames as
"<route folder>/ <frame count>", the format CarlaPerceptionDataset reads
(LMDrive tools/data_preprocessing/get_list_file.py).
"""
import argparse
import os


def count_frames(route_dir):
    for folder in ("measurements", "measurements_full", "rgb_full", "lidar"):
        path = os.path.join(route_dir, folder)
        if os.path.isdir(path):
            return len(os.listdir(path))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", help="dataset root that contains the data/ folder")
    parser.add_argument("--min-frames", type=int, default=32)
    args = parser.parse_args()

    data_dir = os.path.join(args.root, "data")
    lines = []
    for route in sorted(os.listdir(data_dir)):
        route_dir = os.path.join(data_dir, route)
        if not os.path.isdir(route_dir):
            continue
        frames = count_frames(route_dir)
        if frames < args.min_frames:
            print("skip %s: only %d frames (< %d)" % (route, frames, args.min_frames))
            continue
        lines.append("%s/ %d" % (route, frames))

    index_path = os.path.join(args.root, "dataset_index.txt")
    with open(index_path, "w") as stream:
        stream.write("\n".join(lines) + "\n")
    print("wrote %d routes to %s" % (len(lines), index_path))


if __name__ == "__main__":
    main()
