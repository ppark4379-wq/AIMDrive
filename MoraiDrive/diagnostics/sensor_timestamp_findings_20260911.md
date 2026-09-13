# MORAI 센서 header.stamp 실측 — 2026-09-11

## 확인한 결론

ROS 메시지 규약에서 CompressedImage와 PointCloud2의 header timestamp는 센서 데이터의
**측정/취득 시각**이다. 하지만 규약이 MORAI 내부 구현의 계측 위치를 입증하지는 않는다.

현재 실제 송신 패킷과 ROS 수신을 대조한 결과, 전방·우측·후방 카메라와 LiDAR에서
MORAI가 보낸 stamp가 ROS 토픽까지 그대로 유지되는 샘플을 확인했다.
따라서 확인한 샘플은 **rosbridge가 ROS로 발행하는 순간 새로 찍은 시각이 아니다**.
MORAI 내부에서 렌더링/스캔 취득, 완료, 직렬화, 송신 중 어느 지점에서 시각을 얻는지는
공개 설명과 이번 외부 관측만으로 확정하지 못했다. 그 구분에는 MORAI 내부 계측 또는
해당 버전의 공급사 확인이 필요하다. 측정 시각이라고 무조건 확정해 동기화 근거로 쓰지 않는다.

## 수행한 테스트

- 실행 중인 Simulator와 rosbridge에 접속. Expert/Scenario Runner/Sync Runtime/
  Collector를 새로 시작하지 않음. 이미지 디코딩 및 NAS 저장 없음.
- 헤더 수신 15초, Pause/Resume 포함 14초, 패킷 비교용 8초의 제한된 관측.
- ROS Subscriber는 queue_size=1, 토픽당 헤더 최대 1,000개 보관.
- 원본 WebSocket 확인에는 rosbridge 수신 TCP 포트 9090만 사용.
  패킷당 앞 768bytes만 최대 5,000패킷 캡처했다. 센서 전체 프레임을 재조립하거나
  데이터셋으로 저장하지 않았다. 패킷 앞부분에는 헤더 외 일부 데이터도 포함될 수 있다.
- UI 조작 도구는 초기화 오류로 실패했다. 화면을 직접 조작한 테스트가 아니라
  사용자가 연결해 둔 실제 MORAI의 ROS 메시지 및 gRPC API를 사용한 실험이다.

## 수신 상태

최초 15초 관측:

| 토픽 | 수신 수 | stamp 증가 간격 중앙값 |
| --- | ---: | ---: |
| /camera_front | 327 | 50ms |
| /camera_left | 327 | 51ms |
| /camera_right | 327 | 50ms |
| /camera_rear | 326 | 50ms |
| /lidar3D | 156 | 102ms |
| /Ego_topic | 817 | 17ms |
| /clock | 0 | 측정 불가 |

모든 관측 센서의 stamp는 0이 아닌 Unix epoch 형태였다.
/use_sim_time은 미설정, /morai_drive/sync_runtime_active=false,
/morai_drive/scenario_ready=false였다. 등록된 /clock 토픽이 있다는 것과 실제 Clock
메시지를 발행하고 있다는 것은 다르다. 이 실험에서는 Clock을 수신하지 못했다.

## 원본 WebSocket과 ROS 대조

| 토픽 | 추출한 원본 헤더 | 동일 stamp를 ROS에서 확인 | 패킷 앞부분 도착→ROS callback 중앙값 |
| --- | ---: | ---: | ---: |
| /camera_front | 69 | 63 | 3.32ms |
| /camera_right | 68 | 63 | 2.11ms |
| /camera_rear | 72 | 66 | 2.41ms |
| /lidar3D | 35 | 32 | 8.73ms |
| /Ego_topic | 176 | 161 | 1.70ms |

예: 전방 카메라 송신 stamp=1789069578.003초가 ROS 수신 헤더에서도 동일했다.
원본 패킷과 ROS 관측 시작이 다르므로 모든 패킷이 ROS 표본에 포함되는 것은 아니다.
파서는 TCP 세그먼트 시작에서 식별 가능한 WebSocket 메시지 첫 조각만 처리한다.
좌측 카메라는 ROS에서는 수신했지만 이번 제한된 패킷 추출에서 대응 헤더를 얻지 못했다.
따라서 좌측의 원본 stamp 일치까지 확인했다고 주장하지 않는다.

위 ms 수치는 같은 WSL에서 패킷 앞부분이 관측된 뒤 callback까지의 값이다.
전체 측정→발행 지연, 전체 네트워크 지연, 센서 스캔/렌더링 지연이 아니다.

## rosbridge 처리 코드

`expert/src/rosbridge_suite/rosbridge_library/src/rosbridge_library/internal/message_conversion.py`
280행부터 Header 초기 stamp를 rospy.get_rostime()으로 채운다. 이후 292행부터
실제 메시지 필드를 처리하므로 명시적으로 들어온 stamp.secs/nsecs는 입력 값으로
덮어쓴다. 따라서 “Header에 무조건 현재 시간을 덮어쓴다”는 해석도 틀리다.
이번 센서 패킷에는 명시적인 stamp가 들어 있고 ROS 값과 일치하는 것을 실측했다.

## 시계 차이: 수신 시각−stamp를 지연으로 해석하면 안 됨

첫 관측에서 WSL 수신 시각−센서 stamp는 약 2.5~3.9초였다.
Windows에서 WSL로 시간 조회 명령을 왕복시키고 Windows 조회 전/후 시각으로
오차 범위를 제한한 결과, WSL 시계가 Windows보다 앞서 있었다:

- 첫 측정: +2,839.5~+2,907.1ms
- 다섯 번째 측정: +2,681.8~+2,745.8ms

약 2초 사이에도 차이가 줄었다. 뒤의 패킷 비교 관측에서는 일부
수신 시각−stamp가 음수가 됐다. 이는 음수 네트워크 지연을 뜻하지 않는다.
Windows/WSL 시계 오프셋 변화가 있어 서로 다른 시계의 단순 차이는 신뢰할 수 없다.
이 실험에서는 OS 시간 설정을 변경하거나 WSL을 재시작하지 않았다.

## Pause/Resume 결과와 한계

gRPC Pause 요청과 약 3초 뒤 Resume 요청 모두 STATUS_CODE_SUCCESS였다.
Pause 응답 0.5초 뒤부터 Resume 요청 0.5초 전까지, 전방/좌측 45개씩,
우측/후방 44개씩, LiDAR 21개의 메시지를 수신했고 각 stamp도 계속 변했다.
GetTimestamp API 값도 증가했다. GetSimulatorVersion/GetSynchronousMode는 빈/default
응답으로, 이번 실험에서 버전 및 모드 구분에 유효한 정보를 얻지 못했다.

이 관측은 Pause 성공 응답만으로 센서 발행/시계가 반드시 정지한다고 볼 수 없음을
보여 준다. 물리 엔진이나 렌더링의 정지를 별도 계측하지 않았고, 측정과 송신 시각을
가르는 실험으로 충분하지 않다. timestamp가 증가했다는 이유로 발행 시각이라고
결론 내리지 않는다. Resume 성공을 확인했고 관측 프로세스는 종료했다.

## 파일 및 근거

- [관측 도구](sensor_timestamp_probe.py)
- [WebSocket 헤더 추출 도구](decode_sensor_header_capture.py)
- [15초 헤더 기록](sensor_timestamp_20260911_044005.json)
- [Pause/Resume 기록](sensor_timestamp_20260911_044223.json)
- [동시 ROS 수신 기록](sensor_timestamp_wire_comparison.json)
- [원본 헤더 대조 결과](sensor_timestamp_wire_headers.json)
- [Windows/WSL 시계 비교](sensor_timestamp_clock_offsets.json)
- PCAP는 diagnostics/.runtime/sensor_timestamps 아래 로컬 진단 산출물.

공식 자료:

- [ROS CompressedImage 정의](https://raw.githubusercontent.com/ros/common_msgs/noetic-devel/sensor_msgs/msg/CompressedImage.msg): 이미지 취득 시각으로 정의.
- [ROS PointCloud2 정의](https://raw.githubusercontent.com/ros/common_msgs/noetic-devel/sensor_msgs/msg/PointCloud2.msg): 센서 데이터 취득 시각으로 정의.
- [MORAI 26.R1 센서 프로토콜](https://help-morai-sim.scrollhelp.site/ko/morai-sim-drive/26.R1/-35): 위 메시지 타입 사용을 설명하나 취득/직렬화 시각을 찍는 내부 코드를 제시하지 않음.
- [MORAI Time Manager 설명(24.R1)](https://help-morai-sim.scrollhelp.site/ko/morai-sim-drive/24.R1.0/morai-sim-time-management-function): 현실 시간/시뮬레이션 시간 모드와 Clock 발행 설정 설명. 현재 설치 버전의 내부 구현 증명과 구분.


## 추가 실측: Sync Runtime 실행 중 카메라와 LiDAR의 시간 기준 분리

2026-09-11 15:53 KST에 기존 읽기 전용 관측 도구로 8초간 동시에 측정했다.
시뮬레이터 모드, tick 제어, 차량 제어는 변경하지 않았다. 당시
`/morai_drive/sync_runtime_active=true`, `/use_sim_time` 미설정이었으며,
이전 관측과 달리 `/clock` 메시지를 실제 수신했다.

| 토픽 | 표본 수 | 첫 stamp → 마지막 stamp (초) | stamp 증가량 | 해당 표본 사이 실제 경과 시간 (monotonic) |
| --- | ---: | --- | ---: | ---: |
| /clock | 60 | 25.840 → 27.020 | 1.180초 | 7.913초 |
| /camera_front | 24 | 1789109607.701 → 1789109615.858 | 8.157초 | 7.636초 |
| /lidar3D | 12 | 25.855 → 26.966 | 1.111초 | 7.383초 |

각 토픽의 첫/마지막 수신 시점은 조금 다르므로 표의 증가량을 완전히 같은
구간의 값으로 취급하지 않는다. /clock은 매번 20ms, LiDAR stamp는 매번
101ms 증가했다. LiDAR 수신 간격 중앙값은 실제 시간 약 668ms였으므로,
101ms를 실제 수신 간격이나 실시간 10Hz로 해석하면 안 된다.

현재 실행 모드에서는 LiDAR stamp가 시뮬레이션 시간 기준으로 진행하는 것을
확인했다. 카메라 4대는 모두 Unix epoch 형태로, 느리게 진행하는 /clock과 달리
실시간 시계 계열로 증가했다. 카메라의 정확한 내부 시계 API나 시각 기록 위치는
이 실험으로 식별하지 못했다. 다른 모드에서도 동일하다고 일반화하지 않는다.

따라서 현재 카메라와 LiDAR의 원본 stamp는 직접 빼거나 동일 숫자로 매칭할 수 없다.
고정 offset만 더해 맞출 수도 없다. 원본 센서 stamp는 별도 보존하고, 수집 프레임을
동기화할 때는 검증된 공통 tick/프레임 대응이 필요하다. 콜백에서 가장 최근 /clock을
덧붙인 값은 수신 시점의 참고 값이며, 이미지 취득 시각의 정확한 변환 값이 아니다.

시간 기준을 확인한 것과 렌더링/스캔 시작·완료·송신 중 어느 시점에 stamp를 찍는지
확인한 것은 별개다. 후자는 아직 확인되지 않았다.

실측 원본: [동시 헤더·Clock 기록](sensor_timestamp_20260911_155336.json)


## 추가 실측: 현재 발행 토픽 전체의 시간 필드 조사

2026-09-11 16:02 KST, ROS Master에 등록된 29개 발행 토픽의 메시지 구조를
조회했다. 그중 ROS `time` 필드가 있는 24개 토픽을 10초 동안 관측했으며,
17개 토픽에서 메시지를 받았다. `/use_sim_time`은 미설정이고 Sync Runtime은
실행 중이었다. 시뮬레이터 상태/모드/센서 설정은 변경하지 않았다.

| 토픽/필드 | 현재 관측된 시간 기준 | 근거 또는 제한 |
| --- | --- | --- |
| /clock | 시뮬레이션 시간 | 80개, 101.840 → 103.420초; 실제 수신 구간 9.887초 |
| /Ego_topic, /morai_sync_runtime 발행분 | 시뮬레이션 시간 | 80개, 101.840 → 103.420초; 매 수신 시 최신 /clock과 같은 값 |
| /lidar3D | 시뮬레이션 시간 | 15개, 101.908 → 103.322초; 실제 수신 구간 8.845초 |
| /camera_front, /camera_left, /camera_right, /camera_rear | 실시간 시계 계열의 절대시각 | epoch 형태, 실제 약 9~10초 구간에 stamp 약 10초 증가 |
| /tf, Expert 발행 map → gps | WSL/ROS 현재 시각 | epoch 형태; Expert 코드에서 rospy.Time.now() 사용 |
| /lattice/paths, /lattice/objects, /lattice/lanes 의 Marker 헤더 | WSL/ROS 현재 시각 | epoch 형태; rviz_visualizer.py에서 rospy.Time.now() 사용 |
| /global_path, /local_path | 미설정(0) | 상위 header.stamp와 표본 poses[0:3]의 stamp가 모두 0 |
| /odom | 미설정(0) | 80개 모두 header.stamp=0 |
| /rosout, /rosout_agg | 현재 관측된 로그는 절대시각 | 로그 생성 시간. /rosout은 발행 노드별로 조사; 과거 로그 표본도 포함 |
| /connected_clients 의 connection_time | 절대시각 형태의 연결 시각 | 저장된 연결 시각 1개 메시지의 첫 3개 client 표본; 진행하는 센서 시각이 아님 |
| /CollisionData, /dafault_topic | 판정 불가 | Header는 있으나 관측 중 메시지 0개 |
| /Object_topic, /GetTrafficLightStatus, /IntscnTL_topic, /InsnStatus, /ReplayInfo_topic | 판정 불가 | Header는 있으나 관측 중 메시지 0개 |
| /SyncModeInfo | timestamp 필드 없음 | frame 번호는 있지만 ROS time/Header 필드는 없음 |
| /ctrl_cmd, /expert_target_velocity, /SetTrafficLight, /client_count | timestamp 필드 없음 | 메시지 스키마에서 확인 |

/Ego_topic과 /tf는 rosbridge와 로컬 노드가 함께 publisher로 등록되어 있다.
이번 /Ego_topic 표본은 Sync Runtime에서, /tf 시간 표본은 Expert에서 받았다.
토픽명만 보고 미관측된 다른 발행자의 시간 기준까지 동일하다고 판단하지 않는다.
GPS/IMU 토픽은 이번 발행 목록에 없어 시간 기준을 판정하지 않았다.
/InsnStatus.intersection_status_time, /IntscnTL_topic.delta_time 같은 일반 수치 필드는
Header timestamp와 별개이며, 이름만으로 센서 취득 시각으로 취급하지 않는다.

코드 대조:
- expert/src/expert/src/network/ros_manager.py:418~444의 Path/Odometry 변환은
  frame_id와 자세만 채우고 header.stamp를 설정하지 않는다.
- 같은 파일 452~457행의 TF는 입력 Ego의 원본 stamp 대신 rospy.Time.now()를 쓴다.
- expert/src/expert/src/planning/lattice_planning/rviz_visualizer.py:39에서 현재
  ROSTIME을 얻고 Marker header.stamp에 넣는다. 현재의 /use_sim_time 미설정과
  epoch 실측이 일치한다. 입력 센서의 취득 시각이라는 뜻은 아니다.

검사 한계: 센서 payload는 디코딩/보관하지 않았다. 중첩 메시지는 512KiB 이하에서
토픽·발행자별 최대 2Hz로 디코딩하고 배열 앞 3개만 표본화했다. 따라서 /tf의 모든
변환이나 모든 Marker/Pose를 전수 확인한 것은 아니다. 헤더 샘플은 그룹당 최대
256개만 보관했고 실행은 정상 종료했으며 파싱 오류는 없었다.

- [토픽별 관측 원본](topic_timestamps_20260911_160234.json)
- [재실행 가능한 읽기 전용 검사 도구](topic_timestamp_inventory.py)
