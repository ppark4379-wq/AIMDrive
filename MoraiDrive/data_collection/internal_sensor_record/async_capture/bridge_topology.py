"""Explicit multi-process bridge contract; the default single bridge is unchanged."""
from pathlib import Path

REQUIRED={'Camera-1','Camera-2','Camera-3','Camera-4','/lidar3D','/gps','/imu','/Ego_topic'}
ROLE={'lidar':'/lidar3D','ego':'/Ego_topic','gnss':'/gps','imu':'/imu'}


def validate_config(rows):
    if not isinstance(rows,list) or not 1<=len(rows)<=8:raise ValueError('Explicit bridge list required')
    owners={};nodes=set();ports=set();health=set()
    for row in rows:
        if not isinstance(row.get('node'),str) or not row['node'].startswith('/'):raise ValueError('Absolute bridge node required')
        if not isinstance(row.get('health_topic'),str) or not row['health_topic'].startswith('/'):raise ValueError('Absolute health topic required')
        if type(row.get('port'))!=int or not 1024<=row['port']<=65535:raise ValueError('Invalid bridge port')
        if row['node'] in nodes or row['port'] in ports or row['health_topic'] in health:raise ValueError('Duplicate bridge identity')
        nodes.add(row['node']);ports.add(row['port']);health.add(row['health_topic'])
        if not row.get('streams'):raise ValueError('Bridge must own a required stream')
        for key in row['streams']:
            if key not in REQUIRED or key in owners:raise ValueError('Duplicate or unknown stream ownership')
            owners[key]=row['node']
    if set(owners)!=REQUIRED:raise ValueError('Bridge contract must cover every required sensor')
    return owners


def failures(config,healths,streams,now,expected_boot=None):
    owners=validate_config(config);errors=[]
    boot=expected_boot or Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    for entry in config:
        received=healths.get(entry['health_topic'])
        if received is None:errors.append('missing_bridge_health:'+entry['node']);continue
        health,receive,caller=received;prefix=entry['node']+':'
        if health.get('implementation')!='stage6e-binary-json-v1' or not health.get('binary_json_patch'):errors.append(prefix+'unpatched')
        if health.get('boot_id')!=boot:errors.append(prefix+'different_receive_clock')
        if not 0<=now-health.get('monotonic_ns',0)<=2500000000 or not 0<=now-receive<=2500000000:errors.append(prefix+'stale_health')
        if caller!=entry['node'] or health.get('node')!=entry['node'] or health.get('port')!=entry['port']:errors.append(prefix+'identity_mismatch')
        if not health.get('listening') or not health.get('instance_id'):errors.append(prefix+'listener_or_instance_missing')
        if health.get('recent_errors'):errors.append(prefix+'payload_errors')
        if any(c.get('oldest_pending_ms',0)>250 for c in health.get('clients',[])):errors.append(prefix+'queue_over_250ms')
        for key in entry['streams']:
            row=health.get('streams',{}).get(key,{})
            if row.get('samples',0)<2 or row.get('receive_age_ms',1e9)>500:errors.append(prefix+'missing_stream:'+key)
            if (row.get('queue_delay_ms') or {}).get('max',1e9)>250:errors.append(prefix+'delayed_stream:'+key)
            if len(row.get('client_ids',[]))!=1:errors.append(prefix+'producer_count:'+key)
        unexpected=set(health.get('streams',{}))-set(entry['streams'])
        if unexpected:errors.append(prefix+'unexpected_streams:'+','.join(sorted(unexpected)))
    for role,key in [(x,x.split(':',1)[1]) for x in streams if x.startswith('camera:')]+list(ROLE.items()):
        if streams.get(role,{}).get('publishers')!=[owners.get(key)]:errors.append('unexpected_publisher:'+role)
    return errors
