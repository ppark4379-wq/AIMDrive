"""Extract header prefixes from bounded classic-PCAP WebSocket samples.
Only parses message-start fragments that begin in a captured TCP segment.
Does not claim complete TCP/WebSocket reassembly or complete packet coverage.
"""
from pathlib import Path
import collections
import json
import re
import socket
import statistics
import struct

root=Path(__file__).resolve().parent
source=root/'.runtime/sensor_timestamps/morai_headers_all.pcap'
data=source.read_bytes()
if data[:4]!=bytes.fromhex('d4c3b2a1'):
    raise ValueError('Expected little-endian microsecond classic PCAP')
link=struct.unpack_from('<I',data,20)[0]
if link not in (1,113):raise ValueError('Unexpected PCAP link type')
pos=24
headers=[]
while pos+16<=len(data):
    sec,usec,n,original=struct.unpack_from('<IIII',data,pos)
    pos+=16
    packet=data[pos:pos+n]
    pos+=n
    ip=16 if link==113 else 14
    if len(packet)<ip+40 or packet[ip-2:ip]!=bytes.fromhex('0800'):continue
    if packet[ip+9]!=6:continue
    tcp=ip+(packet[ip]&15)*4
    start=tcp+(packet[tcp+12]>>4)*4
    payload=packet[start:]
    # Both FIN and non-FIN message-start fragments; continuation data is ignored.
    if len(payload)<14 or payload[0] not in (0x01,0x02,0x81,0x82):continue
    length=payload[1]&127
    offset=4 if length==126 else (10 if length==127 else 2)
    if payload[1]&128:
        key=payload[offset:offset+4]
        offset+=4
        if len(key)!=4:continue
        raw=bytes(value^key[i%4] for i,value in enumerate(payload[offset:]))
    else:raw=payload[offset:]
    if not raw.startswith(b'{'):continue
    text=raw.decode('utf-8',errors='replace')
    topic=re.search(r'"topic"\s*:\s*"([^"\n]+)"',text)
    stamp=re.search(r'"stamp"\s*:\s*\{\s*"secs"\s*:\s*(\d+)\s*,\s*"nsecs"\s*:\s*(\d+)\s*\}',text)
    if not topic or not stamp:continue
    if topic.group(1) not in ('/Ego_topic','/camera_front','/camera_left','/camera_right','/camera_rear','/lidar3D'):continue
    headers.append({'topic':topic.group(1),'stamp_ns':int(stamp.group(1))*1000000000+int(stamp.group(2)),
       'source':socket.inet_ntoa(packet[ip+12:ip+16]),'capture_ns':sec*1000000000+usec*1000})
observed=json.loads((root/'sensor_timestamp_wire_comparison.json').read_text())
summary={}
for topic in sorted(set(item['topic'] for item in headers)):
    raw=[item for item in headers if item['topic']==topic]
    received={item['stamp_ns']:item for item in observed['samples'][topic]}
    matched=[item for item in raw if item['stamp_ns'] in received]
    delays=[(received[item['stamp_ns']]['recv_wall_ns']-item['capture_ns'])/1e6 for item in matched]
    summary[topic]={'wire_headers':len(raw),'exact_stamp_matches_in_ros':len(matched),
       'capture_to_callback_ms':({'min':min(delays),'median':statistics.median(delays),'max':max(delays)} if delays else None),
       'sample_wire':matched[0] if matched else raw[0]}
output={'pcap_bytes':len(data),'summary':summary,'wire_headers':headers}
(root/'sensor_timestamp_wire_headers.json').write_bytes(json.dumps(output,indent=2).encode())
print(json.dumps(summary,indent=2))
