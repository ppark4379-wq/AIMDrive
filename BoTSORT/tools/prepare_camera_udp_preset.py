#!/usr/bin/env python3
"""Create a MORAI preset changing camera 1/2/3 networking only. Does not apply it."""
import argparse
from copy import deepcopy
import ipaddress
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sensor-file',required=True,help='Saved MORAI sensor JSON; prefer a fresh runtime save')
    parser.add_argument('--host-ip',required=True,help='Windows interface IPv4 address')
    parser.add_argument('--destination-ip',required=True,help='WSL IPv4 address')
    parser.add_argument('--output',required=True,help='New JSON file; never overwrite existing presets')
    args=parser.parse_args()
    host=str(ipaddress.IPv4Address(args.host_ip))
    destination=str(ipaddress.IPv4Address(args.destination_ip))
    if ipaddress.ip_address(destination).is_unspecified:
        parser.error('Destination must be an actual receiver address')
    original=json.loads(Path(args.sensor_file).read_text(encoding='utf-8-sig'))
    updated=deepcopy(original)
    expected={1:(9290,9291),2:(9292,9293),3:(9294,9295)}
    found=set()
    for sensor in updated.get('cameraList',[]):
        sensor_id=sensor.get('m_SensorUniqueID')
        if sensor_id not in expected:continue
        if sensor_id in found:raise ValueError('Duplicate camera sensor ID')
        found.add(sensor_id)
        camera=sensor['cc']
        if camera.get('cameraType')!=0 or camera.get('modelType')!=0:
            raise ValueError('Select Camera / Ground Truth None before preparing the preset')
        if any(camera.get(field,False) for field in ('bVizBoundingBox','vizBoundingBox2D','vizBoundingBox3D')):
            raise ValueError('Disable Viz Bounding Box before preparing the preset')
        camera['commType']=1  # Verified UDP enum in MORAI 26.R1.H3 sensor JSON.
        camera['udpConfig'].update(hostIP=host,destinationIP=destination,
            hostPort=expected[sensor_id][0],destinationPort=expected[sensor_id][1])
    if found!=set(expected):raise ValueError('Preset must contain camera sensor IDs 1, 2 and 3')
    with Path(args.output).open('x',encoding='utf-8') as output:
        json.dump(updated,output,indent=2,ensure_ascii=False)
    print('Created '+args.output)
    print('Only camera 1/2/3 communication type and UDP IP/ports changed. Calibration and other sensors preserved.')
    print('Copy to the simulator SaveFile/Sensor/version folder, load with Sensor File Manager, then verify actual packets.')


if __name__=='__main__':main()
