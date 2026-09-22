"""Configured global path and map provenance shared by live navigation clients."""
import hashlib
import json
from pathlib import Path
from navigation.engine import NavigationEngine

ROOT = Path(__file__).resolve().parents[1]


def load_live_route(source):
    """Load an explicitly configured plan without requiring recorded driving data."""
    import yaml
    from navigation.build import read_config
    from navigation.engine import Route
    from navigation.mgeo import route_from_mgeo
    required={'global_path','map_root','route_definition','route_key','policy_config'}
    if not isinstance(source,dict) or set(source)!=required:
        raise ValueError('route must specify global_path, map_root, route_definition, route_key, policy_config')
    if any(not isinstance(v,str) or not v.strip() for v in source.values()):
        raise ValueError('Live route settings must be nonempty strings')
    paths={key:(ROOT/source[key]).resolve() for key in required-{'route_key'}}
    files=dict(global_path=paths['global_path'],route_definition=paths['route_definition'],
               link_set=paths['map_root']/'link_set.json',node_set=paths['map_root']/'node_set.json',
               policy_config=paths['policy_config'])
    # Read once so the logged hashes describe exactly the bytes used for the plan.
    raw={key:path.read_bytes() for key,path in files.items()}
    points=[]
    for line in raw['global_path'].decode('utf-8-sig').splitlines():
        line=line.split('#',1)[0].strip()
        if not line: continue
        values=line.split()
        if len(values)<2: raise ValueError('Global path must contain whitespace-separated X Y coordinates')
        points.append([float(values[0]),float(values[1])])
    Route(dict(points=points))  # Validate finite coordinates before geometry matching.
    definitions=yaml.safe_load(raw['route_definition'].decode('utf-8-sig'))
    if source['route_key'] not in definitions:
        raise ValueError('Unknown route_key: '+source['route_key'])
    definition=definitions[source['route_key']]
    links={str(row['idx']):row for row in json.loads(raw['link_set'].decode('utf-8-sig'))}
    nodes={str(row['idx']):row for row in json.loads(raw['node_set'].decode('utf-8-sig'))}
    hashes={key:hashlib.sha256(value).hexdigest() for key,value in raw.items()}
    policy_document=read_config(paths['policy_config'])
    if policy_document['config_sha256']!=hashes['policy_config']:
        raise ValueError('Navigation policy config changed while loading')
    policy=policy_document['policy']
    route=route_from_mgeo(points,definition['route_links'],links,nodes,definition,hashes['global_path'],policy)
    Route(route)
    NavigationEngine(route,policy)
    episode=dict(episode_id='configured/'+source['route_key']+'/'+hashes['global_path'][:16],
                 route=route,frames=[],sources={key:dict(path=str(path),sha256=hashes[key])
                                               for key,path in files.items()})
    return episode,dict(policy=policy),None

