"""Derive the signal-query set from the exact ordered scenario route."""
import json
from pathlib import Path

import yaml

from .journal import digest


def build_route_signal_contract(map_root, route_definition_path, route_key):
    map_root=Path(map_root).resolve()
    route_path=Path(route_definition_path).resolve()
    link_path=map_root/'link_set.json';node_path=map_root/'node_set.json'
    document=yaml.safe_load(route_path.read_text(encoding='utf-8-sig'))
    route=document.get(route_key,{})
    route_links=[str(value) for value in route.get('route_links',())]
    links={str(row['idx']):row for row in json.loads(link_path.read_text())}
    nodes={str(row['idx']):row for row in json.loads(node_path.read_text())}
    if len(route_links)<2 or len(route_links)!=len(set(route_links)):
        raise ValueError('Ordered route link contract is missing or duplicated')
    entries=[]
    for index,link_id in enumerate(route_links):
        if link_id not in links:raise ValueError('Route link absent from map: '+link_id)
        node_id=str(links[link_id]['to_node_idx'])
        if node_id not in nodes:raise ValueError('Route node absent from map: '+node_id)
        node=nodes[node_id];light=node.get('traffic_light_id')
        if node.get('on_stop_line') and light is not None and str(light).strip():
            entries.append(dict(route_link_index=index,route_link_id=link_id,
                                stop_node_id=node_id,light_id=str(light)))
    # A loop may reference one light more than once. Query it once while
    # preserving every route occurrence in ``route_entries``.
    light_ids=[]
    for row in entries:
        if row['light_id'] not in light_ids:light_ids.append(row['light_id'])
    if not light_ids:raise ValueError('No route-bound traffic light IDs')
    return dict(schema='morai_route_signal_query_contract_v1',map_root=str(map_root),
        route_key=str(route_key),route_links=len(route_links),route_entries=entries,
        light_ids=light_ids,source_sha256=dict(route_definition=digest(route_path),
            link_set=digest(link_path),node_set=digest(node_path)),
        label_scope='query coverage only; per-frame route binding still required',
        training_visible=False)
