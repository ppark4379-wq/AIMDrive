"""Byte-exact camera refragmentation and preservation of missing-part gaps."""
from collections import Counter
import unittest

from inference.sensors.morai_udp import CameraReceiver, CAMERA_PREFIX, CAMERA_JPEG_CAPACITY
from inference.sensors.udp_relay import camera_datagrams


def native_parts(image, stamp=10):
    for index,start in enumerate(range(0,len(image),CAMERA_JPEG_CAPACITY)):
        body=image[start:start+CAMERA_JPEG_CAPACITY]
        end=b'EI' if start+len(body)==len(image) else b'AI'
        yield CAMERA_PREFIX.pack(b'MOR',stamp,0,index,len(body))+body+b'\0'*(CAMERA_JPEG_CAPACITY-len(body))+end


def receiver():
    parser=object.__new__(CameraReceiver)
    parser._buffer=bytearray()
    parser._expect_index=0
    parser._intact=False
    parser._frame_stamp=None
    parser.torn_images=0
    parser.audit_counts=Counter()
    return parser


class CameraRelayTests(unittest.TestCase):
    def test_multiple_native_parts_reassemble_exactly_without_padding(self):
        import cv2
        import numpy as np
        ok,encoded=cv2.imencode('.jpg',np.random.RandomState(4).randint(0,256,(720,1280,3),dtype=np.uint8))
        self.assertTrue(ok)
        image=encoded.tobytes()
        for limit in (256,1200,1400):
            with self.subTest(limit=limit):
                packets=[p for old in native_parts(image) for p in camera_datagrams(old,limit)]
                self.assertTrue(all(len(p)<=limit for p in packets))
                self.assertEqual([CAMERA_PREFIX.unpack_from(p)[3] for p in packets],list(range(len(packets))))
                parser=receiver();images=[v for p in packets if (v:=parser._consume(p)) is not None]
                self.assertEqual(images,[image])
                np.testing.assert_array_equal(cv2.imdecode(np.frombuffer(images[0],np.uint8),1),cv2.imdecode(encoded,1))

    def test_missing_native_or_relay_part_never_yields_a_complete_image(self):
        image=b'\xff\xd8'+b'a'*(CAMERA_JPEG_CAPACITY*3)+b'\xff\xd9'
        old=list(native_parts(image))
        packets=[p for n in old for p in camera_datagrams(n)]
        cases=([p for n in old[1:] for p in camera_datagrams(n)],
               [p for n in old[:1]+old[2:] for p in camera_datagrams(n)],
               packets[:25]+packets[26:])
        for broken in cases:
            parser=receiver()
            self.assertFalse(any(parser._consume(p) is not None for p in broken))
            good=list(camera_datagrams(next(native_parts(b'\xff\xd8next\xff\xd9',11))))
            self.assertEqual(parser._consume(good[0]),b'\xff\xd8next\xff\xd9')

    def test_timestamp_index_and_byte_identity_for_legacy_passthrough(self):
        packet=next(native_parts(b'\xff\xd8test\xff\xd9',35))
        self.assertEqual(list(camera_datagrams(packet,0)),[packet])
        part=list(camera_datagrams(packet))[0]
        self.assertEqual(CAMERA_PREFIX.unpack_from(part),(b'MOR',35,0,0,8))
        self.assertEqual(part[-2:],b'EI')
        for other in (b'BOXdata',b'\xff\xee'+b'0'*1204,b'$GPGGA,abc',b'#IMUData$'):
            self.assertEqual(list(camera_datagrams(other)),[other])

    def test_invalid_or_partial_nonfinal_native_packet_is_not_repaired(self):
        prefix=lambda index,size:CAMERA_PREFIX.pack(b'MOR',1,0,index,size)
        for bad in (b'MOR',prefix(0,5)+b'abcEI',prefix(-1,1)+b'xEI',
                    prefix(0,1)+b'xAI',prefix(0,1)+b'x??',prefix(0,0)+b'EI'):
            with self.subTest(bad=bad),self.assertRaises(ValueError):list(camera_datagrams(bad))


if __name__=='__main__':unittest.main()
