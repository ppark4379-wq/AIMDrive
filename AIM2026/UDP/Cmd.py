from dataclasses import dataclass
import math
import socket
import struct

FORMAT = struct.Struct('<14si3i3b5f2s')


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

    def pack(self):
        for name, allowed in [('ctrl_mode', (2,)), ('gear', range(6)), ('cmd_type', (1,))]:
            value = getattr(self, name)
            if type(value) is not int or value not in allowed:
                raise ValueError('Invalid ' + name)
        values = (self.velocity, self.acceleration, self.accel, self.brake, self.steer)
        if not all(math.isfinite(v) for v in values):
            raise ValueError('Command values must be finite')
        if not 0 <= self.accel <= 1 or not 0 <= self.brake <= 1 or not -1 <= self.steer <= 1:
            raise ValueError('Pedals require 0..1 and normalized steer requires -1..1')
        return FORMAT.pack(b'#MoraiCtrlCmd$', 23, 0, 0, 0,
                           self.ctrl_mode, self.gear, self.cmd_type, *values, b'\r\n')


class CmdSender:
    def __init__(self, host, port, *, protocol='morai'):
        if not isinstance(host, str) or not host.strip() or type(port) is not int or not 1 <= port <= 65535:
            raise ValueError('Invalid command target')
        self.target = (host, port)
        if protocol != 'morai':
            raise ValueError('Competition control requires MORAI Ego Ctrl Cmd')
        self.protocol = protocol
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, command):
        payload = command.pack()
        return self.socket.sendto(payload, self.target)

    def close(self):
        self.socket.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
