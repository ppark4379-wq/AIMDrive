"""UDP settings for runs inside WSL2: MORAI on Windows, the Expert in WSL, addresses read from the running PC.

In WSL2 the simulator (Windows host) and the Expert (WSL) have addresses that differ from PC to PC and change when WSL
restarts, so the repository's presets (127.0.0.1 for a same-PC run, 192.168.0.1 -> 192.168.0.10 for the competition) do not
fit. This module writes a per-PC copy of the UDP receiver settings from the Windows gateway and WSL address of the moment;
no repository file is edited. It acts only inside WSL with the default ``--config``: on Windows, on Linux or with an
explicit ``--config`` nothing changes.

Which sensor set MORAI uses
  * ``--morai-sensor FILE`` (or the environment variable ``MORAI_SENSOR_FILE``, or the path stored in
    ``AIM2026/outputs/network/morai_sensor_path.txt``): the sensor file MORAI loads on this PC, normally
    ``<MORAI launcher>/SaveFile/Sensor/<set>/aim_sensor.json``. The receiver settings are made to match it (each sensor's
    destination port is read from the file) and the file's addresses are checked against the current ones.
  * nothing given: the committed competition set (``aim_sensor.json``, one port per sensor) with the addresses replaced is
    written to ``AIM2026/outputs/network/sensor_set_wsl.json`` for you to load in MORAI Sensor Settings.
  * ``--competition-addresses``: the competition set exactly as committed (192.168.0.1 -> 192.168.0.10); this PC must carry
    those two addresses.

The Ego status size depends on the MORAI build (``--status-bytes``; default 229 B of the full Ego status; 181 = Competition
Vehicle Status).
"""

import copy
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess

import yaml

SENSOR_PATH_NOTE = 'AIM2026/outputs/network/morai_sensor_path.txt'


def wsl_addresses():
    """Return the current WSL address and the Windows host on its default route."""
    route = subprocess.check_output(['ip', '-4', 'route', 'show', 'default'],
                                    text=True, timeout=3)
    gateway = re.search(r'\bvia (\d+\.\d+\.\d+\.\d+)\b', route)
    if gateway is None:
        raise RuntimeError('WSL default route has no Windows gateway: ' + route.strip())
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        probe.connect((gateway.group(1), 9))
        local_ip = probe.getsockname()[0]
    if local_ip.startswith('127.'):
        raise RuntimeError('WSL UDP route resolved to loopback; check WSL networking')
    return local_ip, gateway.group(1)


def to_wsl_path(text):
    """``C:\\dir\\file`` -> ``/mnt/c/dir/file`` (a Windows path as people copy it from Explorer); other text unchanged."""
    match = re.match(r'^([A-Za-z]):[\\/](.*)$', str(text).strip().strip('"'))
    if match is None:
        return Path(str(text).strip().strip('"'))
    return Path('/mnt/' + match.group(1).lower() + '/' + match.group(2).replace('\\', '/'))


def saved_sensor_path(args, root):
    """The sensor file MORAI loads on this PC, or None when it is not configured."""
    chosen = getattr(args, 'morai_sensor', None) or os.environ.get('MORAI_SENSOR_FILE')
    if not chosen:
        note = Path(root) / SENSOR_PATH_NOTE
        if note.is_file():
            chosen = note.read_text(encoding='utf-8').strip()
    return to_wsl_path(chosen) if chosen else None


def _sensors(config):
    return (config['GPS'], config['IMU'], config['LiDAR'], *config['Camera'].values())


def _udp_by_host_port(sensor_set, path=''):
    actual = {}
    for rows in sensor_set.values():
        if not isinstance(rows, list):
            continue
        for sensor in rows:
            for settings in sensor.values():
                udp = settings.get('udpConfig') if isinstance(settings, dict) else None
                if not isinstance(udp, dict):
                    continue
                port = int(udp['hostPort'])
                if port in actual:
                    raise ValueError('Duplicate MORAI sensor host port {} in {}'.format(port, path))
                actual[port] = udp
    return actual


def match_saved_sensor(config, saved_sensor, *, wsl_ip, windows_ip):
    """Set every sensor's receiving port from the saved MORAI sensor file and check its addresses.

    Sensors are paired by their MORAI source (host) port. The file must send from the Windows host to this WSL address;
    otherwise MORAI's packets would never arrive, and the error says what to change in the file.
    """
    actual = _udp_by_host_port(json.loads(Path(saved_sensor).read_text(encoding='utf-8-sig')), saved_sensor)
    for sensor in _sensors(config):
        udp = actual.get(int(sensor['host_port']))
        if udp is None:
            raise ValueError('MORAI sensor file {} has no sensor sending from host port {}'.format(
                saved_sensor, sensor['host_port']))
        if udp['hostIP'] != windows_ip or udp['destinationIP'] != wsl_ip:
            raise ValueError(
                'MORAI sensor file {} does not match this WSL setup at host port {}: it has {} -> {}, this PC needs '
                '{} -> {}. Change the addresses in the file and reload it in MORAI Sensor Settings.'.format(
                    saved_sensor, sensor['host_port'], udp['hostIP'], udp['destinationIP'], windows_ip, wsl_ip))
        sensor['dest_port'] = int(udp['destinationPort'])


def prepare(source_config, source_sensor_set, destination, *, wsl_ip, windows_ip, competition_build=False,
            status_bytes=None, keep_addresses=False, saved_sensor=None):
    """Write per-host settings without modifying the shared competition config.

    competition_build=False describes the older local MORAI build (245-byte full Ego status, reversed UDP steering).
    competition_build=True uses the competition steering sign. status_bytes selects the Ego status layout the
    simulator really sends: 181 (Competition Vehicle Status, nothing is written), 229 or 245 (full Ego status).
    saved_sensor: the sensor file MORAI loads; the receiver ports follow it and no sensor set is written.
    """
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(Path(source_config).read_text(encoding='utf-8-sig'))
    for sensor in _sensors(config):
        sensor['bind'] = '0.0.0.0'
        if not keep_addresses:
            # Windows MORAI may choose either its loopback or vEthernet source IP; the source-port filter stays.
            sensor['ip'] = None
    config['Ego']['bind'] = '0.0.0.0'
    if not keep_addresses:
        config['Ego']['ip'] = windows_ip
    if saved_sensor is not None:
        match_saved_sensor(config, saved_sensor, wsl_ip=wsl_ip, windows_ip=windows_ip)
    if competition_build:
        # Competition steering profile; the status layout is whatever the simulator is set to send.
        status_bytes = 229 if status_bytes is None else status_bytes
        if status_bytes == 181:
            config['Ego'].pop('local_status_bytes', None)          # Competition Vehicle Status
        else:
            config['Ego']['local_status_bytes'] = status_bytes     # explicit full-status layout (229 or 245)
        config['Control']['positive_steer'] = 'left'
    else:
        # The older local MORAI build publishes the 245-byte Ego Vehicle Status.
        config['Ego']['local_status_bytes'] = 245 if status_bytes is None else status_bytes
        # Local 245-byte MORAI run: positive wire steering decreased IMU yaw while the route required a left turn.
        config['Control']['positive_steer'] = 'right'
    config_path = destination / ('config_competition_exact.yaml' if keep_addresses
                                 else 'config_wsl_competition.yaml' if competition_build else 'config_wsl.yaml')
    config_path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding='utf-8')
    if saved_sensor is not None:
        return config_path, Path(saved_sensor)

    sensor_set = copy.deepcopy(json.loads(Path(source_sensor_set).read_text(encoding='utf-8-sig')))
    for rows in sensor_set.values():
        if keep_addresses or not isinstance(rows, list):
            continue                                        # the competition set is used exactly as committed
        for sensor in rows:
            for settings in sensor.values():
                if isinstance(settings, dict) and isinstance(settings.get('udpConfig'), dict):
                    udp = settings['udpConfig']
                    udp['hostIP'] = windows_ip
                    udp['destinationIP'] = wsl_ip
    sensor_path = destination / ('sensor_set_competition_exact.json' if keep_addresses else 'sensor_set_wsl.json')
    if keep_addresses:
        shutil.copyfile(source_sensor_set, sensor_path)          # byte for byte, as committed
    else:
        sensor_path.write_text(json.dumps(sensor_set, ensure_ascii=False, indent=2), encoding='utf-8')
    return config_path, sensor_path


def keep_status_layout(base_config, derived_config):
    """Copy ``Ego.local_status_bytes`` of ``base_config`` into ``derived_config`` (file rewritten in place).

    The competition launcher's ``--competition-morai`` copy of a config drops the full-status size because the competition
    build sends the 181 B vehicle status. A config written by this module already holds the size this PC's MORAI really
    sends, so the launcher puts it back.
    """
    base = yaml.safe_load(Path(base_config).read_text(encoding='utf-8-sig'))
    size = (base.get('Ego') or {}).get('local_status_bytes')
    if size is None:
        return None
    derived = yaml.safe_load(Path(derived_config).read_text(encoding='utf-8'))
    derived.setdefault('Ego', {})['local_status_bytes'] = size
    Path(derived_config).write_text(yaml.safe_dump(derived, allow_unicode=True, sort_keys=False), encoding='utf-8')
    return size


def conda_python():
    """The Python of the active Conda environment (the VE stack lives there), or None."""
    prefix = os.environ.get('CONDA_PREFIX')
    if prefix and not os.environ.get('AIM_EXPERT_PYTHON'):
        candidate = Path(prefix) / ('python.exe' if os.name == 'nt' else 'bin/python')
        if candidate.is_file():
            return str(candidate.resolve())
    return None


def add_arguments(parser):
    parser.add_argument('--morai-build', choices=('competition', 'local'), default='competition',
                        help='MORAI build the WSL network config is written for (default: competition launcher)')
    parser.add_argument('--status-bytes', type=int, choices=(181, 229, 245), default=None,
                        help='Ego status packet size MORAI sends (competition build default 229, local 245, '
                             '181 = Competition Vehicle Status); default $MORAI_STATUS_BYTES')
    parser.add_argument('--morai-sensor', metavar='FILE',
                        help='The sensor file MORAI loads on this PC (SaveFile/Sensor/<set>/aim_sensor.json); the UDP '
                             'settings follow it. Default: $MORAI_SENSOR_FILE or ' + SENSOR_PATH_NOTE)
    parser.add_argument('--competition-addresses', action='store_true',
                        help='Use the competition sensor set and config with their own addresses (192.168.0.1 -> 192.168.0.10, '
                             'one port per sensor) unchanged; this PC must carry those two addresses (see the printed hint)')
    parser.add_argument('--prepare-network', action='store_true',
                        help='Write the WSL UDP config (and MORAI sensor set), then exit before MORAI setup')


def configure_launcher(args, root, parser):
    """WSL with the default config: write the per-host config and use it (and the sensor set, if one is generated).

    Returns True when the launcher should stop afterwards (--prepare-network).
    """
    root = Path(root)
    default_config = (root / 'AIM2026/config_local.yaml').resolve()
    wsl_child = (os.name != 'nt' and 'microsoft' in os.uname().release.lower()
                 and not str(args.expert_python).lower().endswith('.exe'))
    if args.prepare_network and not wsl_child:
        parser.error('--prepare-network requires a WSL Python interpreter')
    if wsl_child and Path(args.config).resolve() == default_config:
        wsl_ip, windows_ip = wsl_addresses()
        keep = bool(getattr(args, 'competition_addresses', False))
        status_bytes = args.status_bytes
        if status_bytes is None and os.environ.get('MORAI_STATUS_BYTES'):
            status_bytes = int(os.environ['MORAI_STATUS_BYTES'])
            if status_bytes not in (181, 229, 245):
                parser.error('MORAI_STATUS_BYTES must be 181, 229 or 245')
        if keep and args.morai_build != 'competition':
            parser.error('--competition-addresses needs --morai-build competition')
        saved = None if keep else saved_sensor_path(args, root)
        if saved is not None and not saved.is_file():
            parser.error('MORAI sensor file not found: ' + str(saved))
        if args.morai_build == 'competition':
            # The committed competition set: every sensor has its own destination port, no splitter.
            source_config, source_sensors = root / 'AIM2026/config_competition.yaml', root / 'AIM2026/aim_sensor.json'
        else:
            source_config, source_sensors = default_config, root / 'AIM2026/sensor_set_competition.json'
        try:
            args.config, sensor_set = prepare(
                source_config, source_sensors, root / 'AIM2026/outputs/network',
                wsl_ip=wsl_ip, windows_ip=windows_ip, competition_build=args.morai_build == 'competition',
                status_bytes=status_bytes, keep_addresses=keep, saved_sensor=saved)
        except (OSError, ValueError, KeyError) as error:
            parser.error(str(error))
        if keep:
            print('[Network] competition addresses kept: MORAI 192.168.0.1 -> this PC 192.168.0.10. The WSL address must exist: '
                  'sudo ip addr add 192.168.0.10/24 dev eth0 (and on Windows, as admin: netsh interface ip add address '
                  '"vEthernet (WSL (Hyper-V firewall))" 192.168.0.1 255.255.255.0)', flush=True)
        print('[Network] MORAI sensor set: {} ({})'.format(
            sensor_set, 'in use, addresses checked' if saved is not None else 'load it in MORAI Sensor Settings'), flush=True)
        print('[Network] Ego Network: Host IP {}, Destination IP {}; status 9088 -> 9099; '
              'Ctrl Cmd 9093 <- 9094'.format(*(('192.168.0.1', '192.168.0.10') if keep else (windows_ip, wsl_ip))), flush=True)
        print('[Network] AIM2026 UDP config: {}'.format(args.config), flush=True)
    return bool(args.prepare_network)
