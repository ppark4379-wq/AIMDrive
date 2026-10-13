"""Use the collector's exact official-route tracker at inference."""
import hashlib
from pathlib import Path
import sys


def navigation_recipe(navigation_contract):
    from data.src.loaders.morai_encoder_contract import navigation_recipe as serialize_recipe
    return serialize_recipe(navigation_contract)


def _hash_representations(path):
    raw = Path(path).read_bytes()
    lf = raw.replace(b'\r\n', b'\n')
    return {hashlib.sha256(data).hexdigest() for data in (raw, lf, lf.replace(b'\n', b'\r\n'))}


def official_route_tracker(contract, global_path, *, map_root=None, route_definition=None):
    """Same link ordering, heading tie-break, forward tracking and closure as GT.

    Path relocation permits byte-identical or LF/CRLF-identical source files,
    matching the collector's route-source policy. It does not accept a recorded
    driven trajectory as a substitute for the official path.
    """
    global_path = Path(global_path)
    map_root = Path(map_root) if map_root else global_path.parent
    route_definition = (Path(route_definition) if route_definition else
                        map_root.parent.parent / 'config/full_loop_links.yaml')
    paths = dict(global_path=global_path, route_definition=route_definition,
                 link_set=map_root/'link_set.json', node_set=map_root/'node_set.json')
    actual = {name: _hash_representations(path) for name, path in paths.items()}
    recipes = contract.get('navigation_recipes') or []
    matched = [recipe for recipe in recipes
               if all(recipe['source_sha256'].get(name) in hashes for name, hashes in actual.items())]
    if not matched:
        raise ValueError('Official route/map hashes differ from checkpoint navigation evidence; '
                         'export the contract with navigation_recipes')
    recipe = matched[0]
    from data.src.loaders import morai_encoder_contract as contract_module
    morai_root = Path(contract_module.__file__).resolve().parents[4] / 'MoraiDrive'
    if str(morai_root) not in sys.path:
        sys.path.insert(0, str(morai_root))
    from data_collection.mgeo_navigation import MGeoMap, MGeoRouteTracker, load_official_route
    world = MGeoMap(str(paths['link_set']), node_set_path=str(paths['node_set']))
    route = load_official_route(str(route_definition), str(global_path), recipe['route_key'])
    return MGeoRouteTracker(world, route, target_distance_m=recipe['target_distance_m'])
