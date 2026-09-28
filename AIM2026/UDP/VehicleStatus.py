"""Full and competition vehicle status packets."""
import ctypes
import math
from .Common import _Receiver, _Packet

class VehicleStatusPacket(_Packet):
    """Competition Vehicle Status; same layout as EgoVehicleStatus with the
    fields rule 2-8 withholds left at zero."""
    _fields_ = [
        ("header", ctypes.c_char * 11),
        ("data_length", ctypes.c_int),
        ("aux_data", ctypes.c_int * 3),
        ("sec", ctypes.c_int),
        ("nsec", ctypes.c_int),
        ("ctrl_mode", ctypes.c_byte),
        ("gear", ctypes.c_byte),
        ("signed_vel", ctypes.c_float),
        ("map_data_id", ctypes.c_int),
        ("accel", ctypes.c_float),
        ("brake", ctypes.c_float),
        ("size_x", ctypes.c_float),
        ("size_y", ctypes.c_float),
        ("size_z", ctypes.c_float),
        ("overhang", ctypes.c_float),
        ("wheelbase", ctypes.c_float),
        ("rear_overhang", ctypes.c_float),
        ("pos_x", ctypes.c_float),
        ("pos_y", ctypes.c_float),
        ("pos_z", ctypes.c_float),
        ("roll", ctypes.c_float),
        ("pitch", ctypes.c_float),
        ("yaw", ctypes.c_float),
        ("vel_x", ctypes.c_float),
        ("vel_y", ctypes.c_float),
        ("vel_z", ctypes.c_float),
        ("ang_vel_x", ctypes.c_float),
        ("ang_vel_y", ctypes.c_float),
        ("ang_vel_z", ctypes.c_float),
        ("accel_x", ctypes.c_float),
        ("accel_y", ctypes.c_float),
        ("accel_z", ctypes.c_float),
        ("steer", ctypes.c_float),
        ("link_id", ctypes.c_char * 38),
        ("tire_lateral_force_fl", ctypes.c_float),
        ("tire_lateral_force_fr", ctypes.c_float),
        ("tire_lateral_force_rl", ctypes.c_float),
        ("tire_lateral_force_rr", ctypes.c_float),
        ("side_slip_angle_fl", ctypes.c_float),
        ("side_slip_angle_fr", ctypes.c_float),
        ("side_slip_angle_rl", ctypes.c_float),
        ("side_slip_angle_rr", ctypes.c_float),
        ("tire_cornering_stiffness_fl", ctypes.c_float),
        ("tire_cornering_stiffness_fr", ctypes.c_float),
        ("tire_cornering_stiffness_rl", ctypes.c_float),
        ("tire_cornering_stiffness_rr", ctypes.c_float),
        ("tail", ctypes.c_char * 2),
    ]

class CompetitionVehicleStatusPacket(_Packet):
    """181 bytes: the status minus the twelve floats rule 2-8 withholds.

    Confirmed against a running simulator, which publishes this variant and
    leaves pos_x/pos_y/pos_z at 0.0 exactly as the rules describe.
    """
    _fields_ = [(name, kind) for name, kind in VehicleStatusPacket._fields_
                if not name.startswith(("tire_", "side_slip_"))]

VEHICLE_STATUS_BYTES = ctypes.sizeof(VehicleStatusPacket)
COMPETITION_STATUS_BYTES = ctypes.sizeof(CompetitionVehicleStatusPacket)

class VehicleStatusReceiver(_Receiver):
    """Competition Vehicle Status; supplies the encoder's speed and heading."""

    def __init__(self, port, **kwargs):
        self.layout_bytes = None
        super().__init__(port, recv_bytes=1024, **kwargs)

    def _consume(self, payload):
        """Accept either the full status or the shorter competition variant."""
        if len(payload) >= VEHICLE_STATUS_BYTES:
            packet = VehicleStatusPacket.from_buffer_copy(payload[:VEHICLE_STATUS_BYTES])
            competition = False
        elif len(payload) >= COMPETITION_STATUS_BYTES:
            packet = CompetitionVehicleStatusPacket.from_buffer_copy(
                payload[:COMPETITION_STATUS_BYTES])
            competition = True
        else:
            return None
        if not packet.header.startswith(b"#"):
            return None
        self.layout_bytes = COMPETITION_STATUS_BYTES if competition else VEHICLE_STATUS_BYTES
        return {
            "competition_variant": competition,
            "layout_bytes": self.layout_bytes,
            # UDP signed velocity is km/h along the vehicle's direction.
            # PID/navigation consume m/s, not one velocity-vector component.
            "speed": float(packet.signed_vel) / 3.6,
            "signed_speed": float(packet.signed_vel),
            "yaw": math.radians(float(packet.yaw)),
            "yaw_degrees": float(packet.yaw),
            "accel": float(packet.accel),
            "brake": float(packet.brake),
            "steer": float(packet.steer),
            "gear": int(packet.gear),
            "sim_time": packet.sec + packet.nsec / 1e9,
        }
