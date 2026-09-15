#!/usr/bin/env python3
"""Render the machine-readable ROS readiness report for an operator."""

import argparse
import json
from pathlib import Path
import sys


LABELS = (
    ("ego", "차량 정보 /Ego_topic"),
    ("objects", "객체 /Object_topic"),
    ("lidar", "LiDAR"),
    ("gps", "GPS"),
    ("imu", "IMU"),
    ("clock", "Simulation /clock"),
    ("traffic_light", "신호등(주행 중 선택)"),
)


def state_for(role, row, failures):
    prefixes = tuple(str(item) for item in failures)
    bridge_key = {
        "ego": "/Ego_topic", "lidar": "/lidar3D",
        "gps": "/gps", "imu": "/imu",
    }.get(role)
    if role == "traffic_light" and not row.get("messages"):
        return "선택 토픽: 아직 미수신"
    if (any(item.endswith(":" + role) and "missing_or_stale" in item for item in prefixes)
            or (bridge_key and any(
                item.startswith("SENSOR_MISSING:") and item.endswith(bridge_key)
                for item in prefixes
            ))):
        return "대기: 미수신 또는 오래된 메시지"
    if any(item.endswith(":" + role) and "reversing" in item for item in prefixes):
        return "오류: source timestamp 역행/정지"
    messages = int(row.get("messages", 0) or 0)
    age = row.get("receive_age_s")
    if messages:
        detail = "%d개" % messages
        if isinstance(age, (int, float)):
            detail += ", 최근 %.2fs 전" % max(0.0, float(age))
        return "수신 정상: " + detail
    return "대기: 미수신"


def next_actions(report, bridge_ip=None):
    failures = [str(item) for item in report.get("failures", [])]
    actions = []
    clock_ingress_broken = "clock_ingress_not_delivered_to_ros" in failures
    clock_missing = (
        "use_sim_time_not_true" in failures
        or "morai_simulator_network_clock_not_sent" in failures
        or clock_ingress_broken
        or any(item.endswith(":clock") for item in failures)
    )
    if clock_missing and not clock_ingress_broken:
        endpoint = (bridge_ip + ":9090") if bridge_ip else "01번에 표시된 MORAI 연결 주소:9090"
        actions.append(
            "MORAI를 Simulation Time Mode로 두고 "
            "Simulator Network의 ROS 주소를 %s으로 설정한 뒤 /clock 송신을 "
            "Disconnect → Connect하세요. 설정 창을 닫고 시뮬레이션 일시정지도 "
            "해제하세요. 차량 정보·센서셋도 같은 주소를 사용해야 합니다."
            % endpoint
        )
    if clock_ingress_broken:
        actions.append(
            "MORAI의 /clock은 bridge까지 도착했지만 ROS로 전달되지 않았습니다. "
            "현재 01번을 종료한 뒤 다시 실행하여 계측 bridge를 재시작하세요."
        )
    if any(
        item.endswith(":ego") or item.endswith(":objects")
        or item.endswith("bridge /Ego_topic") for item in failures
    ):
        actions.append(
            "MORAI 차량 정보 Network Settings를 표시된 IP:9090으로 "
            "Disconnect → Connect 하세요."
        )
    missing_sensor_labels = []
    for role, label in (("lidar", "LiDAR"), ("gps", "GPS"), ("imu", "IMU")):
        topic = {"lidar": "/lidar3D", "gps": "/gps", "imu": "/imu"}[role]
        if any(item.endswith(":" + role) or item.endswith("bridge " + topic)
               for item in failures):
            missing_sensor_labels.append(label)
    if missing_sensor_labels:
        actions.append(
            "MORAI Sensor Setting에서 %s를 Disconnect → Connect 하세요."
            % "/".join(missing_sensor_labels)
        )
    if any("bridge_camera_ingress" in item or "bridge Camera-" in item
           for item in failures):
        actions.append(
            "Camera 1~4를 표시된 IP:9090으로 다시 연결하세요. "
            "ROS 카메라 토픽은 의도적으로 bridge 진입점에서 폐기됩니다."
        )
    if any("running implementation differs" in item for item in failures):
        actions.append("기존 9090 bridge를 종료하고 01번을 다시 실행하세요.")
    return actions


def render(report, bridge_ip=None):
    failures = report.get("failures", [])
    streams = report.get("streams", {})
    lines = ["", "센서 연결 상태"]
    for role, label in LABELS:
        lines.append("  %-26s %s" % (label, state_for(role, streams.get(role, {}), failures)))

    bridge_clock = streams.get("bridge_clock_ingress", {})
    clock_missing = any(str(item).endswith(":clock") for item in failures)
    if ("morai_simulator_network_clock_not_sent" in failures
            or (clock_missing and int(bridge_clock.get("samples", 0) or 0) < 2)):
        clock_detail = "MORAI→bridge 패킷 없음 (Simulator Network 재연결 필요)"
    elif "clock_ingress_not_delivered_to_ros" in failures:
        clock_detail = "MORAI→bridge 수신됨, ROS 전달 없음 (bridge 재시작 필요)"
    elif int(bridge_clock.get("samples", 0) or 0) >= 2:
        age = bridge_clock.get("receive_age_ms")
        clock_detail = "MORAI→bridge 수신 정상"
        if isinstance(age, (int, float)):
            clock_detail += ", 최근 %.0fms 전" % max(0.0, float(age))
    else:
        clock_detail = "진단 대기"
    lines.append("  %-26s %s" % ("/clock 송신 경로", clock_detail))

    ingress = streams.get("bridge_camera_ingress", {})
    cameras = ingress.get("streams", {})
    camera_rates = []
    for index in range(1, 5):
        role = "Camera-%d" % index
        row = cameras.get(role, {})
        samples = int(row.get("samples", 0) or 0)
        age = row.get("receive_age_ms")
        rate = row.get("receive_rate_hz")
        dropped = int(row.get("dropped_before_ros", 0) or 0)
        if samples >= 2 and dropped == samples:
            detail = "ROS 연결 확인"
            if isinstance(rate, (int, float)):
                camera_rates.append(float(rate))
                detail += ": %.1fHz/카메라" % float(rate)
            if isinstance(age, (int, float)):
                detail += ", 최근 %.0fms 전" % max(0.0, float(age))
            detail += " (원본 저장률 판정값 아님)"
        else:
            detail = "대기: bridge ingress 미수신"
        lines.append("  %-26s %s" % (role, detail))

    if len(camera_rates) == 4:
        lines.append(
            "  %-26s %.1fHz (Camera 4대가 공유하는 ROS 진단 토픽 합계)"
            % ("Camera ROS 합계", sum(camera_rates))
        )
    if cameras:
        lines.append(
            "  %-26s 수집 종료 후 03번에서 10Hz cadence를 판정"
            % "Sensor Record 원본"
        )

    if report.get("ready") is True:
        lines.append("\n[PASS] 수집 필수 연결 stream이 모두 준비됐습니다.")
        lines.append(
            "       위 Camera Hz는 연결 진단값입니다. "
            "Sensor Record 원본 10Hz는 수집 후 03번 검증 결과로 확정합니다."
        )
    else:
        actions = next_actions(report, bridge_ip=bridge_ip)
        lines.append("\n[WAIT] 아직 수집을 시작할 수 없습니다.")
        for action in actions or ["위 상태에서 '대기' 또는 '오류'인 항목을 확인하세요."]:
            lines.append("       - " + action)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", nargs="?", type=Path)
    parser.add_argument("--bridge-ip")
    args = parser.parse_args()
    try:
        raw = args.report.read_text(encoding="utf-8") if args.report else sys.stdin.read()
        report = json.loads(raw)
    except (OSError, json.JSONDecodeError) as error:
        print("[FAIL] 센서 상태 보고서를 읽지 못했습니다: " + str(error), file=sys.stderr)
        raise SystemExit(2)
    print(render(report, bridge_ip=args.bridge_ip))


if __name__ == "__main__":
    main()
