"""Shared path handling for the dataset tools.

A dataset_index.txt line is "<route folder> <frame count>".  Depending on how the
index was written the folder is relative either to the dataset root
("data/routes_town01_short_w2_.../") or to its data/ subfolder
("routes_town01_short_w2_.../"), so the tools accept both, like the training code.
"""
import os


def resolve_route_dir(dataset_root, path):
    """Absolute directory of one route."""
    candidate = os.path.join(dataset_root, path)
    if os.path.isdir(candidate):
        return candidate
    return os.path.join(dataset_root, "data", path)


def route_path_relative_to_root(dataset_root, path):
    """Route path as stored in the instruction lists; the dataset joins it onto the root."""
    relative = os.path.relpath(resolve_route_dir(dataset_root, path), dataset_root)
    return relative.replace(os.sep, "/")
