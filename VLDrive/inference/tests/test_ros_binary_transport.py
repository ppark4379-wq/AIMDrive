import json
import socket
import struct
import threading
import unittest
from inference.sensors.ros_binary_transport import frames


class FragmentSocket:
    def __init__(self,chunks):self.chunks=iter(chunks)
    def recv(self,_):
        chunk=next(self.chunks,b'')
        if chunk is None:raise socket.timeout()
        return chunk


class TransportTests(unittest.TestCase):
    def test_partial_header_and_payload_survive_timeouts(self):
        metadata=json.dumps({'topic':'/camera','value':3}).encode()
        wire=struct.pack('!II',len(metadata),4)+metadata+b'abcd'
        connection=FragmentSocket([wire[:3],None,wire[3:10],None,wire[10:-2],None,wire[-2:]])
        stream=frames(connection,threading.Event())
        message,raw=next(stream)
        self.assertEqual(message,{'topic':'/camera','value':3})
        self.assertEqual(raw,b'abcd')
        with self.assertRaises(EOFError):next(stream)

    def test_oversize_length_is_rejected_before_payload_read(self):
        connection=FragmentSocket([struct.pack('!II',2_000_000,0)])
        with self.assertRaises(ValueError):next(frames(connection,threading.Event()))


if __name__=='__main__':unittest.main()
