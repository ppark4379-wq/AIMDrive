"""Route selection shared by MORAI training and dataset indexing."""

import json
from pathlib import Path


def normalize_route(route):
    if not isinstance(route, str):
        raise ValueError("MORAI routes must be names, not town numbers")
    route = route.replace("\\", "/").strip("/")
    if route.startswith("data/"):
        route = route[5:]
    if not route or any(part in (".", "..", "") for part in route.split("/")) or ":" in route:
        raise ValueError(f"Invalid MORAI route name: {route!r}")
    return route


def validate_route_split(train_routes, val_routes):
    if not train_routes or not val_routes:
        raise ValueError("MORAI requires --train-routes and --val-routes")
    overlap = {normalize_route(r) for r in train_routes} & {
        normalize_route(r) for r in val_routes
    }
    if overlap:
        raise ValueError(f"Train/validation routes overlap: {sorted(overlap)}")


def get_route_frames(root, routes=None, index_filename="dataset_index.txt"):
    root = Path(root)
    selected = None if routes is None else {normalize_route(r) for r in routes}
    found = set()
    frames_out = []
    for line in (root / index_filename).read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        route = normalize_route(parts[0])
        if selected is not None and route not in selected:
            continue
        if route in found:
            raise ValueError(f"Duplicate route in {index_filename}: {route}")
        found.add(route)
        route_dir = root / "data" / route
        for frame_id in range(int(parts[1])):
            # Read the exact frame: never substitute a previous frame's validity.
            path = route_dir / "measurements_full" / f"{frame_id:04d}.json"
            with path.open(encoding="utf-8") as stream:
                measurement = json.load(stream)
            if measurement.get("navigation", {}).get("route_valid") is False:
                continue
            frames_out.append((str(route_dir), frame_id))
    if selected is not None and selected - found:
        raise ValueError(f"Unknown MORAI routes: {sorted(selected - found)}")
    if not frames_out:
        raise ValueError("No valid MORAI frames remain in the selected routes")
    return frames_out
