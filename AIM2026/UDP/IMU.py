import ctypes
import math
from .Common import _Receiver, _Packet

class ImuPacket(_Packet):
    _fields_ = [
        ("header", ctypes.c_char * 9),
        ("data_length", ctypes.c_int),
        ("aux_data", ctypes.c_int * 3),
        ("sec", ctypes.c_int),
        ("nsec", ctypes.c_int),
        ("ori_w", ctypes.c_double),
        ("ori_x", ctypes.c_double),
        ("ori_y", ctypes.c_double),
        ("ori_z", ctypes.c_double),
        ("ang_vel_x", ctypes.c_double),
        ("ang_vel_y", ctypes.c_double),
        ("ang_vel_z", ctypes.c_double),
        ("lin_acc_x", ctypes.c_double),
        ("lin_acc_y", ctypes.c_double),
        ("lin_acc_z", ctypes.c_double),
        ("tail", ctypes.c_char * 2),
    ]

class ImuPacketNoStamp(_Packet):
    _fields_ = [
        ("header", ctypes.c_char * 9),
        ("data_length", ctypes.c_int),
        ("aux_data", ctypes.c_int * 3),
        ("ori_w", ctypes.c_double),
        ("ori_x", ctypes.c_double),
        ("ori_y", ctypes.c_double),
        ("ori_z", ctypes.c_double),
        ("ang_vel_x", ctypes.c_double),
        ("ang_vel_y", ctypes.c_double),
        ("ang_vel_z", ctypes.c_double),
        ("lin_acc_x", ctypes.c_double),
        ("lin_acc_y", ctypes.c_double),
        ("lin_acc_z", ctypes.c_double),
        ("tail", ctypes.c_char * 2),
    ]

IMU_BYTES = ctypes.sizeof(ImuPacket)
IMU_BYTES_NO_STAMP = ctypes.sizeof(ImuPacketNoStamp)

class ImuReceiver(_Receiver):
    def __init__(self, port, **kwargs):
        self.layout_bytes = None
        super().__init__(port, recv_bytes=512, **kwargs)

    def _consume(self, payload):
        if len(payload) >= IMU_BYTES:
            packet = ImuPacket.from_buffer_copy(payload[:IMU_BYTES])
            stamped = True
        elif len(payload) >= IMU_BYTES_NO_STAMP:
            packet = ImuPacketNoStamp.from_buffer_copy(payload[:IMU_BYTES_NO_STAMP])
            stamped = False
        else:
            return None
        if not packet.header.startswith(b"#IMU"):
            return None
        if stamped and (packet.sec < 0 or not 0 <= packet.nsec < 1_000_000_000):
            self._count('invalid_source_timestamp')
            return None
        x, y, z, w = packet.ori_x, packet.ori_y, packet.ori_z, packet.ori_w
        if not 0.9 < math.sqrt(x * x + y * y + z * z + w * w) < 1.1:
            return None  
        self.layout_bytes = IMU_BYTES if stamped else IMU_BYTES_NO_STAMP
        return {
            "yaw": math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)),
            "quaternion": (x, y, z, w),
            "angular_velocity": (packet.ang_vel_x, packet.ang_vel_y, packet.ang_vel_z),
            "linear_acceleration": (packet.lin_acc_x, packet.lin_acc_y, packet.lin_acc_z),
            "sim_time": (packet.sec + packet.nsec / 1e9) if stamped else None,
            "source_stamp_ns": (packet.sec * 1_000_000_000 + packet.nsec) if stamped else 0,
            "layout_bytes": self.layout_bytes,
        }
