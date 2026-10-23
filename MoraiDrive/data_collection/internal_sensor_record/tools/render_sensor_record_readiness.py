#!/usr/bin/env python3
"""Render the machine-readable ROS readiness report for an operator."""

import argparse
import json
from pathlib import Path
import sys


LABELS = (
    ("control", "차량 제어 /ctrl_cmd"),
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
    if role == "control":
        if not row and not any(item in prefixes for item in (
                "missing_control_subscriber", "control_topology_unavailable")):
            return "이 보고서에서는 미검사"
        if "control_topology_unavailable" in prefixes:
            return "대기: ROS 제어 연결 상태 확인 실패"
        listeners = row.get("subscribers", [])
        if "missing_control_subscriber" in prefixes or not listeners:
            return "대기: MORAI 제어 명령 수신 연결 없음"
        return "연결 정상: " + ", ".join(listeners)
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
    if any(item.endswith(":" + role) and "source" in item for item in prefixes):
        if row.get("source_reversals", 0) or "reversing_source:" + role in prefixes:
            return "대기: source timestamp 역행 (모드 전환·시간 초기화 확인)"
        return "대기: 메시지 수신 중, source timestamp 정지"
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
    bridge_unavailable = (
        "clock_ingress_unavailable" in failures
        or any(item.startswith("BRIDGE_") for item in failures)
    )
    if any(item.startswith("BRIDGE_HEALTH_") for item in failures):
        actions.append(
            "bridge 상태 보고의 수신·시각 검증을 다음 검사에서 다시 확인합니다. "
            "수집은 준비 완료까지 시작하지 않습니다. 계속 반복되면 표시된 heartbeat 나이와 bridge.log를 확인하세요."
        )
    if "use_sim_time_not_true" in failures:
        actions.append("ROS master에서 rosparam set /use_sim_time true를 실행하세요.")
    if "missing_control_subscriber" in failures:
        endpoint = (bridge_ip + ":9090") if bridge_ip else "01번에 표시된 MORAI 연결 주소:9090"
        topic = report.get("streams", {}).get("control", {}).get("topic", "/ctrl_cmd")
        actions.append(
            "MORAI 차량 제어 Network Settings에서 ROS 주소 %s, "
            "제어 토픽 %s의 Disconnect → Connect를 실행하세요. "
            "/Ego_topic 차량 정보 수신과 제어 명령 수신 연결은 별도로 확인해야 합니다."
            % (endpoint, topic)
        )
    if "control_topology_unavailable" in failures:
        actions.append("ROS master의 /ctrl_cmd 연결 상태를 확인하지 못했습니다. 표시된 오류와 bridge.log를 확인하세요.")
    clock_ingress_broken = "clock_ingress_not_delivered_to_ros" in failures
    clock_missing = (
        "morai_simulator_network_clock_not_sent" in failures
        or "missing_or_stale:clock" in failures
    )
    if clock_missing and not clock_ingress_broken and not bridge_unavailable:
        endpoint = (bridge_ip + ":9090") if bridge_ip else "01번에 표시된 MORAI 연결 주소:9090"
        actions.append(
            "MORAI를 Simulation Time Mode / External Sync OFF로 두고 "
            "Simulator Network에서 ROS 주소 %s, /clock 송신 설정과 연결을 확인하세요. "
            "설정 창을 닫고 일시정지를 해제하세요. 차량 정보·센서셋도 같은 주소를 사용해야 합니다. "
            "/clock을 받기 위해 External Sync를 켤 필요는 없습니다."
            % endpoint
        )
    if clock_ingress_broken and not bridge_unavailable:
        actions.append(
            "MORAI의 /clock은 bridge까지 도착했지만 ROS로 전달되지 않았습니다. "
            "현재 01번을 종료한 뒤 다시 실행하여 계측 bridge를 재시작하세요."
        )
    if any("source:" in item for item in failures):
        actions.append(
            "메시지 시간 정지·역행은 연결 끊김과 구분해야 합니다. "
            "Simulation Time / External Sync OFF와 일시정지 해제를 확인하세요. "
            "모드를 전환했다면 시간이 초기화되므로 다음 검사 결과를 확인하세요. "
            "같은 모드에서도 반복되면 원본 timestamp와 중복 송신원을 확인해야 합니다."
        )
    if any(
        item in ("missing_or_stale:ego", "missing_or_stale:objects",
                 "SENSOR_MISSING: bridge /Ego_topic") for item in failures
    ) and not bridge_unavailable:
        actions.append(
            "MORAI 차량 정보 Network Settings의 주소와 연결 상태를 확인하세요. "
            "Simulation Time이 진행 중인데도 미수신이면 표시된 IP:9090으로 다시 연결하세요."
        )
    missing_sensor_labels = []
    for role, label in (("lidar", "LiDAR"), ("gps", "GPS"), ("imu", "IMU")):
        topic = {"lidar": "/lidar3D", "gps": "/gps", "imu": "/imu"}[role]
        if ("missing_or_stale:" + role in failures
                or "SENSOR_MISSING: bridge " + topic in failures):
            missing_sensor_labels.append(label)
    if missing_sensor_labels and not bridge_unavailable:
        actions.append(
            "Simulation Time이 진행 중인데도 미수신이면 MORAI Sensor Setting에서 "
            "%s 연결을 확인하세요." % "/".join(missing_sensor_labels)
        )
    if (any("bridge_camera_ingress" in item or "bridge Camera-" in item
            for item in failures) and not bridge_unavailable):
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

    health = streams.get("bridge_health", {})
    bridge_failures = [str(item) for item in failures if str(item).startswith("BRIDGE_")]
    if bridge_failures:
        lines.append("  bridge 진단: " + "; ".join(bridge_failures))
    heartbeat_age = health.get("heartbeat_age_s")
    if isinstance(heartbeat_age, (int, float)):
        lines.append("  bridge heartbeat 나이: %.3fs" % heartbeat_age)

    bridge_clock = streams.get("bridge_clock_ingress", {})
    clock_missing = "missing_or_stale:clock" in failures
    if "clock_ingress_unavailable" in failures or bridge_failures:
        clock_detail = "bridge 상태 보고 미확인: /clock 송신 경로 판정 대기"
    elif ("morai_simulator_network_clock_not_sent" in failures
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
