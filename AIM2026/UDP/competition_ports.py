#!/usr/bin/env python3
"""Competition UDP ports read from the run config, and a WSL-side receipt check for the Windows relay.

    python3 AIM2026/UDP/competition_ports.py ports [--config AIM2026/config_competition.yaml]
    python3 AIM2026/UDP/competition_ports.py demux [--config ...] [--shared-port 9090]
    python3 AIM2026/UDP/competition_ports.py probe [--config ...] [--seconds 5] [--control-check]

`ports` prints the relay's --ports list (every port MORAI sends to on this PC). `demux` prints the relay's --demux for
a sensor set that sends every sensor to one shared port (our local MORAI set: destination 9090, told apart by the
host port), mapping each sensor's host port to its config dest_port, so either sensor set lands on the same ports
in WSL. `probe` binds the same ports here,
counts what arrives for a few seconds and fails when a stream is missing; run it before main.py, which needs the
ports itself. `--control-check` also sends brake 0.37 then 1.0 to the simulator (only with the car standing) and
reads the change back from the vehicle status, so the control path through the Windows NAT is proven too.
Standard library only and Python 3.8, because wsl.exe runs the distro's own python3.
"""
import argparse
import importlib.util
import select
import socket
import struct
import sys
import time
from pathlib import Path

AIM = Path(__file__).resolve().parents[1]
STATUS_BYTES = 181          # Competition Vehicle Status
LIDAR_BYTES = 1206          # VLP-16 data packet


def read_config(path):
    """The fields the relay needs, without PyYAML (absent from the system python3 wsl.exe starts)."""
    streams, ego, hosts, stack = [], {}, {}, []
    for raw in Path(path).read_text(encoding='utf-8').splitlines():
        line = raw.split('#', 1)[0].rstrip()
        if not line.strip() or line.lstrip().startswith('-'):
            continue
        indent = len(line)-len(line.lstrip())
        key, _, value = line.strip().partition(':')
        value = value.strip()
        while stack and stack[-1][0] >= indent:
            stack.pop()
        if not value:
            stack.append((indent, key))
            continue
        parent = stack[-1][1] if stack else ''
        if parent == 'Ego':
            ego[key] = value
        if key == 'dest_port':
            streams.append((parent, int(value)))
        if key == 'host_port' and value.isdigit():      # null in the local config
            hosts[parent] = int(value)
    demux = [(hosts[name], port) for name, port in streams if name in hosts]
    streams.insert(0, ('VehicleStatus', int(ego['vehicle_status_port'])))
    return streams, ego, demux


def probe(streams, seconds):
    socks, stats = {}, {}
    for name, port in streams:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 22)
        try:
            s.bind(('0.0.0.0', port))
        except OSError as error:
            sys.exit('포트 %d(%s)를 열 수 없음: %s  -> main.py가 켜져 있으면 끄고 다시 실행' % (port, name, error))
        s.setblocking(False)
        socks[s] = name
        stats[name] = dict(port=port, packets=0, bytes=0, sizes={}, frames=0, last=None)
    end = time.time()+seconds
    while time.time() < end:
        for s in select.select(list(socks), [], [], .1)[0]:
            while True:
                try:
                    data = s.recv(70000)
                except BlockingIOError:
                    break
                st = stats[socks[s]]
                st['packets'] += 1
                st['bytes'] += len(data)
                st['sizes'][len(data)] = st['sizes'].get(len(data), 0)+1
                st['last'] = data
                if data.startswith(b'MOR') and data.endswith(b'EI'):
                    st['frames'] += 1          # last part of a camera frame
    for s in socks:
        s.close()
    return stats


def report(stats, seconds):
    ok, total = True, 0
    print('WSL 수신 %.0f초:' % seconds)
    for name, st in stats.items():
        total += st['bytes']
        size = max(st['sizes'], key=st['sizes'].get) if st['sizes'] else 0
        rate = st['packets']/seconds
        note = ''
        if st['packets'] == 0:
            note, ok = '  <-- 안 들어옴', False
        elif name == 'VehicleStatus' and size != STATUS_BYTES:
            note, ok = '  <-- %d B (대회형 %d B가 아님)' % (size, STATUS_BYTES), False
        elif name == 'LiDAR' and size != LIDAR_BYTES:
            note = '  <-- 패킷 %d B (VLP-16은 %d B)' % (size, LIDAR_BYTES)
        elif st['frames']:
            note = '  카메라 %.1f fps' % (st['frames']/seconds)
        print('  %-14s %5d  %7.1f pkt/s  %6.2f Mbps  주 크기 %5d B%s'
              % (name, st['port'], rate, st['bytes']*8/seconds/1e6, size, note))
    print('  합계 %.1f Mbps' % (total*8/seconds/1e6))
    return ok


def status_values(sock, wait=.4):
    last, end = None, time.time()+wait
    while time.time() < end:
        if select.select([sock], [], [], .05)[0]:
            data = sock.recv(1024)
            if len(data) == STATUS_BYTES:
                last = data
    if last is None:
        return None
    # offsets as AIM2026/UDP/VehicleStatus.py reads the 181 B layout
    return dict(speed=struct.unpack_from('<f', last, 37)[0], brake=struct.unpack_from('<f', last, 49)[0])


def control_check(ego, status_port):
    spec = importlib.util.spec_from_file_location('udp_protocol', AIM/'utils/scenario_runner/adapters/udp/protocol.py')
    protocol = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(protocol)
    target, source = (ego['ip'], int(ego['cmd_port'])), int(ego['cmd_source_port'])
    layout = ego.get('cmd_layout', 'single_steer55')
    st = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    st.bind(('0.0.0.0', status_port))
    cmd = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    cmd.bind(('0.0.0.0', source))
    try:
        before = status_values(st)
        if before is None:
            print('제어 확인: 차량 상태가 없어 건너뜀')
            return False
        if abs(before['speed']) > .2:
            print('제어 확인: 차가 움직이는 중(%.2f m/s)이라 보내지 않음' % before['speed'])
            return False
        seen = {}
        for brake in (.37, 1.):            # ends braked, as the runner leaves the car
            end = time.time()+1.
            while time.time() < end:
                cmd.sendto(protocol.pack_control(0., brake, 0., layout=layout), target)
                time.sleep(.05)
            after = status_values(st)
            seen[brake] = after and after['brake']
        ok = seen[.37] is not None and abs(seen[.37]-.37) < .02 and seen[1.] is not None and seen[1.] > .9
        shown = ['없음' if v is None else '%.2f' % v for v in (seen[.37], seen[1.])]
        print('제어 확인 (%s:%d <- 소스 %d): 처음 brake %.2f, 0.37 보냄 -> %s, 1.0 보냄 -> %s  %s'
              % (target[0], target[1], source, before['brake'], shown[0], shown[1], 'PASS' if ok else 'NG'))
        return ok
    finally:
        st.close()
        cmd.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('mode', choices=('ports', 'demux', 'probe'))
    parser.add_argument('--shared-port', type=int, default=9090)
    parser.add_argument('--config', default=str(AIM/'config_competition.yaml'))
    parser.add_argument('--seconds', type=float, default=5.)
    parser.add_argument('--control-check', action='store_true')
    args = parser.parse_args()
    streams, ego, demux = read_config(args.config)
    if args.mode == 'ports':
        print(','.join(str(port) for _, port in streams))
        return 0
    if args.mode == 'demux':
        print('%d:%s' % (args.shared_port, ','.join('%d=%d' % pair for pair in demux)))
        return 0
    ok = report(probe(streams, args.seconds), args.seconds)
    if ok and args.control_check:
        ok = control_check(ego, streams[0][1])
    print('종합: %s' % ('정상' if ok else '문제 있음'))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
