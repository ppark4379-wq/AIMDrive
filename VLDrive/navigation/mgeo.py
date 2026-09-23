"""Convert MGeo topology and a selected route into portable semantic events."""
import math
from collections import defaultdict
from .engine import DEFAULT_POLICY


def boundary_heading(points, distance, at_end=False):
    """Metric tangent estimate; independent of map point spacing and rotation."""
    points=[p[:2] for p in points]
    if at_end: points=list(reversed(points))
    origin=points[0];remaining=distance;target=origin
    for a,b in zip(points,points[1:]):
        length=math.dist(a,b)
        if length<1e-9: continue
        fraction=min(1.,remaining/length)
        target=[a[0]+fraction*(b[0]-a[0]),a[1]+fraction*(b[1]-a[1])]
        remaining-=length
        if remaining<=0: break
    if math.dist(origin,target)<1e-6: return None
    dx,dy=target[0]-origin[0],target[1]-origin[1]
    return math.atan2(-dy,-dx) if at_end else math.atan2(dy,dx)


def junction_direction(link, previous, following, policy):
    sample=policy['junction_heading_sample_m']
    incoming=(previous if previous and str(previous['to_node_idx'])==str(link['from_node_idx']) else None)
    outgoing=(following if following and str(following['from_node_idx'])==str(link['to_node_idx']) else None)
    entry=boundary_heading(incoming['points'] if incoming else link['points'],sample,at_end=bool(incoming))
    exit_heading=boundary_heading(outgoing['points'] if outgoing else link['points'],sample,at_end=not bool(outgoing))
    if entry is None or exit_heading is None: return None,None
    angle=math.degrees(math.atan2(math.sin(exit_heading-entry),math.cos(exit_heading-entry)))
    if abs(angle)>=policy['junction_uturn_threshold_deg']: return None,angle
    direction='straight' if abs(angle)<policy['junction_turn_threshold_deg'] else ('left' if angle>0 else 'right')
    return direction,angle


def is_junction(link):
    return link.get('is_junction') is True or (link.get('link_type_def')=='ngii_model2'
                                              and str(link.get('link_type'))=='1')


def junction_context(links, nodes):
    """Connected intersection connectors; parallel traffic lanes form one road arm."""
    parents={}
    def find(node):
        parents.setdefault(node,node)
        if parents[node]!=node: parents[node]=find(parents[node])
        return parents[node]
    for link in links.values():
        if is_junction(link):
            a,b=str(link['from_node_idx']),str(link['to_node_idx'])
            parents[find(a)]=find(b)
    groups=defaultdict(set)
    for node in parents: groups[find(node)].add(node)
    bearings=defaultdict(list)
    for link in links.values():
        if is_junction(link): continue
        a,b=str(link['from_node_idx']),str(link['to_node_idx']);p=link['points']
        if len(p)<2: continue
        if a in parents and (b not in parents or find(a)!=find(b)):
            q=p[min(5,len(p)-1)]
            bearings[find(a)].append(math.atan2(q[1]-p[0][1],q[0]-p[0][0]))
        if b in parents and (a not in parents or find(a)!=find(b)):
            q=p[max(0,len(p)-6)]
            bearings[find(b)].append(math.atan2(q[1]-p[-1][1],q[0]-p[-1][0]))
    result={}
    for key,members in groups.items():
        angles=[]
        for angle in sorted(bearings[key]):
            if not any(abs(math.atan2(math.sin(angle-old),math.cos(angle-old)))<math.radians(25) for old in angles):
                angles.append(angle)
        # T context needs three distinct road arms including a through-road pair.
        through=any(abs(abs(math.degrees(math.atan2(math.sin(a-b),math.cos(a-b))))-180)<25
                    for a in angles for b in angles)
        context=dict(t_junction=len(angles)==3 and through,
                     traffic_light=any(nodes.get(n,{}).get('traffic_light_id') for n in members))
        for node in members: result[node]=context
    return result


def roundabout_exit(link_ids, links, ring_ids, start, end):
    ring = {key: links[key] for key in ring_ids}
    by_node = {str(link['from_node_idx']): key for key, link in ring.items()}
    if len(by_node) != len(ring) or any(str(l['to_node_idx']) not in by_node for l in ring.values()):
        return None
    outgoing = defaultdict(list)
    for key, link in links.items(): outgoing[str(link['from_node_idx'])].append(key)
    chosen = [key for key, a, b in link_ids if a <= end and b >= start]
    entry_node, desired = None, None
    for key in chosen:
        link = links[key]
        a, b = str(link['from_node_idx']), str(link['to_node_idx'])
        if entry_node is None and b in by_node and a not in by_node: entry_node = b
        if entry_node is not None and a in by_node and b not in by_node:
            desired = key
            break
    if entry_node is None or desired is None: return None
    node, count = entry_node, 0
    for _ in range(len(ring)):
        exits = [key for key in outgoing[node] if key not in ring
                 and str(links[key]['to_node_idx']) not in by_node]
        # Parallel exit lanes are one branch only when their initial bearings
        # agree; ambiguous branches at the same ring node remain unresolved.
        if exits:
            angles = []
            for key in exits:
                p = links[key]['points']; a, b = p[0], p[min(5, len(p)-1)]
                angles.append(math.atan2(b[1]-a[1], b[0]-a[0]))
            if any(abs(math.atan2(math.sin(a-angles[0]), math.cos(a-angles[0]))) > math.radians(25) for a in angles):
                return None
            count += 1
            if desired in exits: return count if count <= 3 else None
        node = str(ring[by_node[node]]['to_node_idx'])
    return None


def route_from_mgeo(points, link_ids, links, nodes, definition, route_sha256, policy=None):
    """Exact link alignment is required; never nearest-match a different branch."""
    policy=dict(DEFAULT_POLICY,**(policy or {}))
    ordered, offset, concatenated = [], 0, []
    for key in link_ids:
        p = links[key]['points']
        if len(p) < 2: raise ValueError('MGeo link has insufficient geometry: '+key)
        ordered.append((key, offset, offset+len(p)-1))
        offset += len(p)
        concatenated.extend([v[:2] for v in p])
    if len(concatenated) != len(points) or any(math.dist(a[:2], b[:2]) > .001 for a,b in zip(concatenated,points)):
        raise ValueError('Route geometry must match selected ordered links')
    checkpoints = {str(c['id']): c['path_index'] for c in definition.get('checkpoints', [])}
    events, diagnostics = [], []
    contexts=junction_context(links,nodes)
    zones = definition.get('zones', {})
    roundabout = zones.get('roundabout')
    round_bounds = None
    if roundabout:
        start = checkpoints[str(roundabout['ego_entry_checkpoint'])]
        end = checkpoints[str(roundabout['ego_exit_checkpoint'])]
        number = roundabout_exit(ordered, links, roundabout.get('internal_loop_links', []), start, end)
        round_bounds = start,end
        event = dict(id='roundabout:'+str(start), kind='roundabout' if number else 'blocked',
                     start_index=start, end_index=end, evidence='route_definition_and_directed_ring_topology')
        if number: event['exit_number'] = number
        else:
            event['reason'] = 'roundabout_exit_unresolved'
            diagnostics.append(event['reason'])
        events.append(event)
    highway = zones.get('highway')
    if highway:
        start = checkpoints[str(highway['ego_entry_checkpoint'])]
        end = checkpoints[str(highway['ego_exit_checkpoint'])]
        events.append(dict(id='highway:'+str(start),kind='highway',start_index=start,end_index=end,
                           evidence='recorded_route_highway_zone'))
    for index,(key,start,end) in enumerate(ordered):
        link = links[key]
        if round_bounds and start < round_bounds[1] and end >= round_bounds[0]: continue
        closed=math.dist(points[0][:2],points[-1][:2])<=.001
        previous = links[ordered[index-1][0]] if index or closed else None
        following = links[ordered[(index+1)%len(ordered)][0]] if index+1<len(ordered) or closed else None
        signal = str(link.get('related_signal') or '').lower()
        direction = 'left' if signal in ('left','left_unprotected') else (
                    'right' if signal in ('right','right_unprotected') else 'straight' if signal=='straight' else None)
        event = None
        # NGII link_type=1 identifies intersection connector links, unlike the
        # tunnel's straight signal. Other map schemas can specify is_junction.
        junction = is_junction(link)
        declared_direction=direction
        angle=None
        if junction:
            direction,angle=junction_direction(link,previous,following,policy)
            if direction is None:
                event=dict(id='turn:'+str(start),kind='blocked',link_id=key,
                           reason='junction_heading_unresolved' if angle is None else 'uturn_not_in_lmdrive_catalog',
                           evidence='planned_junction_entry_exit_geometry',turn_angle_deg=angle)
            if declared_direction is not None and direction!=declared_direction:
                diagnostics.append(dict(kind='junction_direction_metadata_conflict',link_id=key,
                    start_index=start,declared_direction=declared_direction,
                    geometry_direction=direction,turn_angle_deg=angle))
        if direction and (junction or direction!='straight'):
            node = nodes.get(str(link.get('from_node_idx')), {})
            prior_node = nodes.get(str(previous.get('to_node_idx')), {}) if previous else {}
            context=contexts.get(str(link.get('from_node_idx')), {})
            event = dict(id='turn:'+str(start),kind='turn',direction=direction,
                         junction=junction, traffic_light=bool(context.get('traffic_light') or node.get('traffic_light_id') or prior_node.get('traffic_light_id')),
                         t_junction=link.get('junction_type')=='T' or context.get('t_junction',False),
                         evidence='planned_junction_entry_exit_geometry' if junction else 'mgeo_related_signal_and_intersection_topology',
                         link_id=key,declared_direction=declared_direction,turn_angle_deg=angle)
        for side in ('left','right'):
            if previous and previous.get(side+'_lane_change_dst_link_idx')==key and previous.get('can_move_'+side+'_lane'):
                event = dict(id='lane:'+str(start),kind='lane_change',direction=side,
                             evidence='planned_transition_to_mgeo_adjacent_lane')
        if link.get('lane_change_dir') in ('left','right') and link.get('lane_ch_link_path'):
            event = dict(id='lane:'+str(start),kind='lane_change',direction=link['lane_change_dir'],
                         evidence='explicit_mgeo_lane_change_link')
        if (link.get('is_entrance') or link.get('is_exit')) and direction:
            event = dict(id='ramp:'+str(start),kind='highway_entry' if link.get('is_entrance') else 'highway_exit',
                         direction=direction,evidence='mgeo_ramp_flag_and_direction')
        if event:
            event.update(start_index=start,end_index=end)
            events.append(event)
    # A new map may supply reviewed semantic events in its route definition;
    # these travel with that map/route, never in Python coordinate tables.
    for event in definition.get('navigation_events', []):
        event = dict(event)
        event.setdefault('evidence','recorded_route_navigation_event')
        events.append(event)
    return dict(points=[p[:2] for p in points], closed=math.dist(points[0][:2],points[-1][:2])<=.001,
                route_sha256=route_sha256,events=events,diagnostics=diagnostics)
