from dataclasses import dataclass
import math
import socket
import struct

# Ego Ctrl Cmd wire layouts, selected per MORAI build (Ego.cmd_layout):
#   front_rear59  MORAI 26.R1: 27-byte payload includes rear steering (zero for our controller).
#   single_steer55  competition build (25.S4 MolitComp03): 23-byte payload, front steering only.
#                   Verified on this PC: the 59-byte form left the vehicle's steer/accel unchanged.
LAYOUTS = ('front_rear59', 'single_steer55')
DEFAULT_LAYOUT = 'front_rear59'
FORMAT = struct.Struct('<14si3i3b6f2s')
FORMAT_SINGLE_STEER = struct.Struct('<14si3i3b5f2s')


@dataclass(frozen=True)
class Command:
    ctrl_mode: int = 2
    gear: int = 4
    cmd_type: int = 1
    velocity: float = 0.0
    acceleration: float = 0.0
    accel: float = 0.0
    brake: float = 0.0
    steer: float = 0.0

    def pack(self, layout=DEFAULT_LAYOUT):
        if layout not in LAYOUTS:
            raise ValueError('Unknown Ego Ctrl Cmd layout: %r; choose from %s' % (layout, LAYOUTS))
        for name, allowed in [('ctrl_mode', (2,)), ('gear', range(6)), ('cmd_type', (1,))]:
            value = getattr(self, name)
            if type(value) is not int or value not in allowed:
                raise ValueError('Invalid ' + name)
        values = (self.velocity, self.acceleration, self.accel, self.brake, self.steer)
        if not all(math.isfinite(v) for v in values):
            raise ValueError('Command values must be finite')
        if not 0 <= self.accel <= 1 or not 0 <= self.brake <= 1 or not -1 <= self.steer <= 1:
            raise ValueError('Pedals require 0..1 and normalized steer requires -1..1')
        if layout == 'single_steer55':
            return FORMAT_SINGLE_STEER.pack(b'#MoraiCtrlCmd$', 23, 0, 0, 0,
                                            self.ctrl_mode, self.gear, self.cmd_type, *values, b'\r\n')
        return FORMAT.pack(b'#MoraiCtrlCmd$', 27, 0, 0, 0,
                           self.ctrl_mode, self.gear, self.cmd_type, *values, 0.0, b'\r\n')


class CmdSender:
    """MORAI Ego Ctrl Cmd sender.

    ``source_port`` binds the sending socket so packets leave from the port
    entered as the Ctrl Cmd Destination in MORAI (7600 on the verified setup).
    Without it the OS picks an ephemeral source port.
    """
    def __init__(self, host, port, *, protocol='morai', source_port=None, bind='0.0.0.0', layout=None):
        if not isinstance(host, str) or not host.strip() or type(port) is not int or not 1 <= port <= 65535:
            raise ValueError('Invalid command target')
        self.layout = layout or DEFAULT_LAYOUT
        if self.layout not in LAYOUTS:
            raise ValueError('Ego.cmd_layout must be one of %s' % (LAYOUTS,))
        if source_port is not None and (type(source_port) is not int or not 1 <= source_port <= 65535):
            raise ValueError('Invalid command source port')
        self.target = (host, port)
        if protocol != 'morai':
            raise ValueError('Competition control requires MORAI Ego Ctrl Cmd')
        self.protocol = protocol
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        if source_port is not None:
            try:
                self.socket.bind((bind, source_port))
            except BaseException:
                self.socket.close()
                raise

    def send(self, command):
        payload = command.pack(self.layout)
        return self.socket.sendto(payload, self.target)

    def close(self):
        self.socket.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
