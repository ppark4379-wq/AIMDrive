"""Canonical on-disk layout for direct MORAI Sensor Record acquisition.

New captures are written once below ``dataset/.collecting``.  A validated
campaign is finalized with a same-volume directory rename, so publication
does not copy or hardlink sensor payloads.
"""
import os
from pathlib import Path
import re


SCENARIO_FOLDERS = {
    "full_loop": "full_route",
    "full_loop_normal": "basic_driving",
    "full_loop_dynamic": "dynamic_obstacle",
    "full_loop_static": "static_obstacle",
    "full_loop_tunnel_static": "tunnel_static_obstacle",
    "full_loop_tunnel_normal": "tunnel_normal",
    "full_loop_roundabout": "roundabout",
    "full_loop_roundabout_nonuniform": "roundabout_nonuniform",
    "full_loop_roundabout_speed_random": "roundabout_speed_random",
    "full_loop_roundabout_brake": "roundabout_brake",
    "full_loop_highway": "highway",
    "full_loop_tunnel_dynamic": "tunnel_dynamic",
    "full_loop_tunnel_vehicle": "tunnel_vehicle",
    "full_loop_traffic_light": "traffic_light",
}


def component(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_]{1,80}", value):
        raise ValueError("Unsafe dataset path component: %r" % (value,))
    return value


def scenario_folder(scenario_name):
    try:
        return SCENARIO_FOLDERS[scenario_name]
    except KeyError:
        raise ValueError("Unsupported dataset scenario: %r" % scenario_name) from None


def dataset_root(storage_root):
    return Path(storage_root).resolve() / "dataset"


def collecting_campaign(storage_root, scenario_name, campaign_id):
    return (dataset_root(storage_root) / ".collecting"
            / scenario_folder(scenario_name) / component(campaign_id))


def final_campaign(storage_root, scenario_name, campaign_id):
    return (dataset_root(storage_root) / scenario_folder(scenario_name)
            / component(campaign_id))


def metadata_dir(campaign_root):
    return Path(campaign_root) / "_campaign"


def campaign_root_from_metadata(path):
    path = Path(path).resolve()
    return path.parent if path.name == "_campaign" else path


def run_path(campaign_metadata, run_name):
    root = campaign_root_from_metadata(campaign_metadata)
    return root / component(run_name)


def is_direct_campaign_metadata(path, storage_root):
    path = Path(path).resolve()
    if path.name != "_campaign":
        return False
    root = campaign_root_from_metadata(path)
    try:
        relative = root.relative_to(dataset_root(storage_root))
    except ValueError:
        return False
    return ((len(relative.parts) == 3 and relative.parts[0] == ".collecting")
            or len(relative.parts) == 2)


def find_campaign(storage_root, value):
    """Resolve an explicit path or an unambiguous direct/legacy campaign id."""
    value_path = Path(value).expanduser()
    if value_path.exists():
        resolved = value_path.resolve()
        if (resolved / "_campaign" / "campaign_plan.json").is_file():
            return (resolved / "_campaign").resolve()
        return resolved

    name = component(str(value))
    root = dataset_root(storage_root)
    candidates = []
    candidates.extend(root.glob(".collecting/*/%s/_campaign" % name))
    candidates.extend(root.glob("*/%s/_campaign" % name))
    legacy = Path(storage_root).resolve() / "sensor_record_auto_v1" / "campaigns" / name
    if legacy.is_dir():
        candidates.append(legacy)
    candidates = [path.resolve() for path in candidates if path.is_dir()]
    unique = sorted(set(candidates), key=str)
    if len(unique) != 1:
        raise ValueError("Campaign id is missing or ambiguous: %s (%s)" %
                         (name, ", ".join(map(str, unique))))
    return unique[0]


def relative_path(target, parent):
    """Portable POSIX path from one publication directory to a target."""
    return Path(os.path.relpath(str(Path(target).resolve()), str(Path(parent).resolve()))).as_posix()


def finalize_collecting_campaign(campaign_metadata, storage_root, scenario_name,
                                 campaign_id, dataset_campaign_id=None):
    metadata = Path(campaign_metadata).resolve()
    source = campaign_root_from_metadata(metadata)
    expected = collecting_campaign(storage_root, scenario_name, campaign_id).resolve()
    destination = final_campaign(
        storage_root, scenario_name, dataset_campaign_id or campaign_id)
    if source != expected or metadata != source / "_campaign":
        raise ValueError("Refusing to finalize an unexpected staging campaign")
    if destination.exists():
        raise FileExistsError("Final dataset campaign already exists: " + str(destination))
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(str(source), str(destination))
    return destination.resolve()
