"""Captured route identity, with evidenced relocation and LF/CRLF equivalence.

Hashes are never overwritten. Existing capture content must still match its
recorded digest (or an LF/CRLF-only representation) before any fallback is used.
"""
import hashlib
import json
from pathlib import Path
from async_capture.journal import digest


def _representations(path):
    raw = Path(path).read_bytes()
    lf = raw.replace(b"\r\n", b"\n")
    return {hashlib.sha256(value).hexdigest()
            for value in (raw, lf, lf.replace(b"\n", b"\r\n"))}


def _paths(root, definition=None):
    return dict(link_set=root/"link_set.json", node_set=root/"node_set.json",
                route_definition=definition or root.parent.parent/"config/full_loop_links.yaml",
                global_path=root/"2026_molit_comp_global_path.txt")


def resolve_route_sources(episode, contract, *, map_root=None, route_definition=None,
                          route_key=None, scenario_root=None):
    episode = Path(episode)
    traffic = contract["capture"]["traffic_light_ground_truth"]
    recorded = traffic.get("map_root")
    if not recorded or not Path(recorded).is_absolute():
        raise ValueError("Capture has no absolute map_root")
    root = Path(recorded).resolve()
    if map_root is not None and Path(map_root).resolve() != root:
        raise ValueError("--map-root differs from recorded capture")
    key = traffic.get("route_key")
    if not key or (route_key is not None and route_key != key):
        raise ValueError("Route key differs from recorded capture")
    metadata = json.loads((episode/"episode.json").read_text(encoding="utf-8"))
    recorded_root = metadata.get("global_config", {}).get("paths", {}).get("mgeo_root")
    if recorded_root and Path(recorded_root).resolve() != root:
        raise ValueError("Episode map_root differs from sensor contract")
    expected = dict(traffic.get("source_sha256", {}))
    expected["global_path"] = metadata.get("global_path_sha256")
    original = _paths(root)
    # Check ALL available originals before consulting any replacement.
    for name, path in original.items():
        if not expected.get(name):
            raise ValueError("Missing capture hash: " + name)
        if path.exists() and (not path.is_file() or expected[name] not in _representations(path)):
            raise ValueError("Route input differs from capture: " + name)
    selected_root = root
    selected = dict(original)
    if route_definition is not None:
        selected["route_definition"] = Path(route_definition).resolve()
    relocated = not all(path.is_file() for path in selected.values())
    if relocated:
        scenario = (Path(scenario_root) if scenario_root is not None else
                    Path(__file__).resolve().parents[3]/"scenario_runner")
        selected_root = scenario/"data"/root.name
        selected = _paths(selected_root,
                          Path(route_definition).resolve() if route_definition is not None else None)
    hashes = {}
    equivalence = {}
    for name, path in selected.items():
        if not path.is_file():
            raise ValueError("Recorded route input unavailable: " + str(path))
        if expected[name] not in _representations(path):
            raise ValueError("Route input differs from capture: " + name)
        hashes[name] = digest(path)
        equivalence[name] = hashes[name] != expected[name]
    resolution = ("verified_relocated_copy" if relocated else "recorded_path")
    if any(equivalence.values()):
        resolution += "_lf_crlf_equivalent"
    provenance = dict(recorded_map_root=str(root), selected_map_root=str(selected_root),
                      selected_paths={k:str(v) for k,v in selected.items()},
                      recorded_sha256=expected, selected_sha256=hashes,
                      newline_equivalence=equivalence, resolution=resolution)
    return dict(map_root=selected_root, route_definition=selected["route_definition"],
                global_path=selected["global_path"], route_key=key, hashes=hashes,
                paths=selected, resolution=resolution, provenance=provenance,
                line_endings_equivalence_verified=any(equivalence.values()))


def verify_route_report(sources, report):
    # A sealed report may describe the LF or CRLF representation. Accept only
    # digests independently derivable from the already verified selected bytes.
    for name in ("route_definition", "global_path"):
        if report.get(name+"_sha256") not in _representations(sources["paths"][name]):
            raise ValueError("Route audit differs from recorded capture: " + name +
                             "; rerun 03 with --rebuild-route-review")
    if report.get("route_source_sha256") is not None:
        recorded = report["route_source_sha256"]
        if set(recorded) != set(sources["hashes"]) or any(
                recorded[k] not in _representations(sources["paths"][k]) for k in recorded):
            raise ValueError("Route audit map inputs differ from capture")
    if report.get("route_key") is not None and report["route_key"] != sources["route_key"]:
        raise ValueError("Route audit key differs from capture")


def resolve_audited_map_root(map_root, scenario_root, expected_global, source_hashes):
    """Compatibility API for pre-merge callers; validate copies before use."""
    original = Path(map_root)
    candidate = Path(scenario_root)/"data"/original.name
    old, new = _paths(original), _paths(candidate)
    # Validate original geometry even when its global-path digest matches.
    for name in ("link_set", "node_set", "route_definition"):
        path = old[name]
        if path.is_file() and source_hashes.get(name) not in _representations(path):
            raise ValueError("Capture map differs from its recorded hash: " + name)
    if old["global_path"].is_file() and digest(old["global_path"]) == expected_global:
        if not all(p.is_file() for p in old.values()):
            raise ValueError("Audited map copy is unavailable")
        return original, False, "recorded_path"
    if not new["global_path"].is_file() or digest(new["global_path"]) != expected_global:
        raise ValueError("Official global path differs from the sealed route audit")
    for name, path in new.items():
        if not path.is_file():
            raise ValueError("Audited map copy is unavailable: " + name)
        if name != "global_path" and source_hashes.get(name) not in _representations(path):
            raise ValueError("Capture/audited map content differs: " + name)
        if old[name].is_file() and (old[name].read_bytes().replace(b"\r\n",b"\n")
                                  != path.read_bytes().replace(b"\r\n",b"\n")):
            raise ValueError("Capture/audited map content differs: " + name)
    resolution = ("capture_copy_lf_crlf_equivalent" if all(p.is_file() for p in old.values())
                  else "recorded_lf_sha256_equivalent_checkout")
    return candidate, True, resolution
