import json
import pytest
from inference.udp_relay import parse_ports, read_relay_config


def test_small_camera_datagrams_preserve_exact_jpeg_and_frame_identity():
    from inference.udp_relay import camera_datagrams
    from inference.morai_udp import CameraReceiver,CAMERA_PREFIX,CAMERA_JPEG_CAPACITY
    def parser():
        value=CameraReceiver.__new__(CameraReceiver)
        value._buffer=bytearray();value._expect_index=0;value._intact=False
        value._frame_stamp=None;value.torn_images=0
        return value
    jpeg=bytes(range(256))*700
    source=[]
    for i,start in enumerate(range(0,len(jpeg),CAMERA_JPEG_CAPACITY)):
        part=jpeg[start:start+CAMERA_JPEG_CAPACITY]
        tail=b'EI' if start+len(part)==len(jpeg) else b'AI'
        source.append(CAMERA_PREFIX.pack(b'MOR',10,0,i,len(part))+
                      part.ljust(CAMERA_JPEG_CAPACITY,b'\0')+tail)
    wire=[p for packet in source for p in camera_datagrams(packet,1400)]
    assert max(map(len,wire))<=1400
    assert [CAMERA_PREFIX.unpack_from(p)[3] for p in wire]==list(range(len(wire)))
    receiver=parser();output=None
    for packet in wire:
        image=receiver._consume(packet)
        if image is not None: output=image
    assert output==jpeg
    receiver=parser()
    for packet in wire[:1]+wire[2:]: assert receiver._consume(packet) is None
    # A missing end of frame A and missing start of frame B cannot form an image.
    receiver=parser();receiver._consume(source[0])
    for packet in source[1:]:
        h,s,n,i,z=CAMERA_PREFIX.unpack_from(packet)
        assert receiver._consume(CAMERA_PREFIX.pack(h,s+1,n,i,z)+packet[CAMERA_PREFIX.size:]) is None
    assert list(camera_datagrams(b'#GPSdata',1400))==[b'#GPSdata']


def test_gps_only_relay_config_does_not_enable_legacy_demultiplexer(tmp_path):
    path=tmp_path/'relay.json'
    config=dict(schema='morai-udp-relay-v1',to='172.18.254.59',ports='10506',
                demux='',listen='127.0.0.1',report=5.)
    path.write_text(json.dumps(config))
    loaded=read_relay_config(path)
    assert loaded['demux']=='' and parse_ports(loaded['ports'])==[(10506,10506)]
    assert loaded['to']=='172.18.254.59'
    config['ports']='10506:70000'
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError,match='UDP ports'):
        read_relay_config(path)
