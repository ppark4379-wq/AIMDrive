#!/usr/bin/env python3
"""Point a MORAI sensor set at the encoder host, without touching the ports.

    python inference/morai_sensorset_retarget.py aim_sensorset_udp.json \
        --dest-ip 192.168.100.2 --host-ip 192.168.100.1

The sensor set is the file MORAI loads from its Sensor Simulation dialog, and
it already carries the whole UDP layout: which port each sensor sends from and
which port it sends to.  On our own machine every sensor targets 127.0.0.1,
which is invisible to the encoder running under WSL on another host, so the
competition PC needs the same file with the destination changed.

Only the addresses move.  The ports stay exactly as they are, because they are
already the layout inference/udp_relay.py expects:

    7501 -> 9090   camera front     7506 -> 9090   GPS
    7502 -> 9090   camera left      7507 -> 9090   IMU
    7503 -> 9090   camera right     2369 -> 2368   LiDAR
    7504 -> 9090   camera rear

``hostIP`` matters as much as ``destinationIP``: it is the local address each
sensor binds its sending socket to, and a socket bound to 127.0.0.1 cannot
reach another machine, so leaving it behind silently keeps every stream on the
competition PC's loopback.  Pass --host-ip with that PC's own LAN address.

Competition Vehicle Status is not part of a sensor set; it lives in MORAI's
Network Settings dialog and still has to be pointed at the encoder host by
hand.  This prints a reminder rather than pretending otherwise.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Every list a sensor set can hold, paired with the key its entries wrap the
# comms config in.  MORAI names that key per sensor type.
SENSOR_LISTS = {
    "cameraList": "cc",
    "Lidar3DList": "lc",
    "Lidar2DList": "lc",
    "GPSList": "gc",
    "IMUList": "ic",
    "RadarList": "rc",
    "UltrasonicList": "uc",
    "GroundTruthList": "gtc",
    "LightList": "lgc",
    "DelphiRadarList": "rc",
}


def walk_udp_configs(document):
    """(list name, sensor id, udpConfig) for every sensor that speaks UDP."""
    for list_name, config_key in SENSOR_LISTS.items():
        for entry in document.get(list_name) or []:
            config = entry.get(config_key) or {}
            udp = config.get("udpConfig")
            if isinstance(udp, dict):
                yield list_name, entry.get("m_SensorUniqueID"), udp


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sensorset", help="MORAI가 저장한 sensor set JSON")
    parser.add_argument("--dest-ip", required=True,
                        help="추론 PC(이 노트북)의 랜 IP")
    parser.add_argument("--host-ip",
                        help="대회 PC 자신의 랜 IP. 생략하면 hostIP를 건드리지 않는다")
    parser.add_argument("--out", help="쓸 파일. 생략하면 <원본>_retargeted.json")
    parser.add_argument("--config", default=str(ROOT / "configs/vision_encoder_realtime.json"),
                        help="포트 배치를 대조할 실시간 config")
    args = parser.parse_args()

    source = Path(args.sensorset).expanduser()
    document = json.loads(source.read_text(encoding="utf-8-sig"))
    out = Path(args.out).expanduser() if args.out else source.with_name(
        source.stem + "_retargeted.json")

    rows = []
    for list_name, sensor_id, udp in walk_udp_configs(document):
        rows.append((list_name, sensor_id,
                     udp.get("hostIP"), udp.get("hostPort"),
                     udp.get("destinationIP"), udp.get("destinationPort")))
        udp["destinationIP"] = args.dest_ip
        if args.host_ip:
            udp["hostIP"] = args.host_ip

    if not rows:
        print("UDP로 설정된 센서가 없습니다: {}".format(source))
        return 1

    out.write_text(json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8")

    print("%-14s %-4s %-24s %s" % ("list", "id", "before", "after"))
    print("-" * 78)
    for list_name, sensor_id, host_ip, host_port, dest_ip, dest_port in rows:
        print("%-14s %-4s %-24s %s" % (
            list_name, sensor_id,
            "{}:{} -> {}:{}".format(host_ip, host_port, dest_ip, dest_port),
            "{}:{} -> {}:{}".format(args.host_ip or host_ip, host_port,
                                    args.dest_ip, dest_port)))
    print()
    print("저장: {}".format(out))

    # 포트는 바꾸지 않았으니, 그 배치가 relay/인코더와 맞는지만 확인해 준다.
    expected = set(json.loads(Path(args.config).read_text(encoding="utf-8"))["ports"].values())
    arriving = {row[5] for row in rows}
    print()
    print("목적지 포트: {}".format(", ".join(str(port) for port in sorted(arriving))))
    print("인코더가 듣는 포트: {}".format(", ".join(str(port) for port in sorted(expected))))
    print("두 줄이 달라도 정상입니다. 공유 포트를 udp_relay.py가 갈라서 넘깁니다.")
    if not args.host_ip:
        print()
        print("주의: --host-ip를 주지 않아 hostIP가 그대로입니다. 127.0.0.1로 묶여 있으면")
        print("      대회 PC 밖으로 패킷이 나가지 못합니다.")
    print()
    print("이 파일은 카메라/LiDAR/GPS/IMU까지입니다. Competition Vehicle Status는")
    print("sensor set이 아니라 MORAI Network Settings에 있으니 따로 바꿔야 합니다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
