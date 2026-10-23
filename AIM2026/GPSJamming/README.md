# GPSJamming

카메라와 LiDAR를 연결하는 rule-based 인지 모듈입니다.
LiDAR 독립 모듈을 구현했습니다. 카메라 연결·투영·association·motion은 기본 틀 상태입니다.

## 구현 범위

UDP 수신 → 원본 timestamp 보존·공통 절대시각 변환 → 프레임 선택 → LiDAR 전처리 → 투영·association

목표 출력: `{track_id, class, cluster_points, centroid, timestamp, 센서 간 시간차}`

`motion/`의 속도 계산과 검증은 association 구현 이후에 진행합니다.

## 역할

- `configs/perception.yaml`: 임계값, ROI, 센서 포트, 허용 시간차 등 튜닝 파라미터
- `configs/calibration.yaml`: intrinsic, distortion, extrinsic
- `gps_jamming/types.py`: 센서 프레임, bbox, track ID, 점군, centroid, timestamp 자료형
- `gps_jamming/state.py`: 프레임 버퍼, 트랙, association 결과 등 실시간 상태
- `gps_jamming/pipeline.py`: 처리 순서와 프레임 association 실행 흐름
- `gps_jamming/interfaces/`: 카메라·LiDAR UDP 수신
- `gps_jamming/camera/`: 기존 AIMDrive/BoTSORT 코드 연결
- `gps_jamming/lidar/`: 전처리, 지면 제거, 클러스터링, 특징 계산
- `gps_jamming/timing/`: 시간 변환, 시간 급변 감지, 프레임 동기화
- `gps_jamming/geometry/`: 보정값, 좌표 변환, 이미지 투영
- `gps_jamming/association/`: 매칭 점수와 cluster ↔ track 연결
- `gps_jamming/motion/`: 이후 구현할 이력·속도·필터·자차 운동 보상
- `gps_jamming/visualization/`: Open3D point-cloud viewer
- `scripts/`: 실시간 실행, 오프라인 실행, 보정 확인 진입점
- `tests/`: 검증 코드
- `artifacts/`: 생성 결과 저장 위치, Git 제외

## LiDAR 독립 실행

Python 3.10 이상에서 `python -m pip install -e GPSJamming`으로 의존성을 설치합니다.
AIMDrive 폴더에서 실행합니다. 이 PC에서는 아래 Python 환경으로 검증했습니다.

```powershell
& 'C:\conda-envs\botsort\python.exe' GPSJamming/scripts/run_udp.py --duration 10 --output GPSJamming/artifacts/session01/frames.jsonl --record-packets GPSJamming/artifacts/session01/packets.jsonl
```

`--duration 0`은 Ctrl+C까지 실행합니다. 기존 출력 파일을 덮어쓰지 않으므로 다시 실행할 때는 새 경로를 사용합니다.
기본 수신 주소는 `127.0.0.1:2369`입니다. MORAI LiDAR의 Host Sensor Port는 `2368`, Destination IP/Port는 `127.0.0.1:2369`로 설정합니다.
이 프로그램은 센서 입력만 읽으며 차량 제어나 시뮬레이터 설정 변경은 하지 않습니다.
첫 유효 패킷의 송신 IP·포트에 고정하여 서로 다른 센서 입력이 섞이지 않게 합니다.

기록된 원본 UDP 패킷을 동일 파이프라인으로 재생합니다.

```powershell
& 'C:\conda-envs\botsort\python.exe' GPSJamming/scripts/run_offline.py --input GPSJamming/artifacts/session01/packets.jsonl --output GPSJamming/artifacts/session01/replayed.jsonl
& 'C:\conda-envs\botsort\python.exe' -m unittest discover -s GPSJamming/tests -v
```

원본 패킷 JSONL은 `payload_hex`와 `receive_monotonic_time`(정수 ns)을 기록합니다.
프레임 JSONL은 원본 점군과 클러스터별 점군을 모두 포함하므로 장시간 기록 시 파일이 커집니다.
번호와 색상은 프레임 내부에서만 유효하며 객체 추적 결과가 아닙니다.

## 처리 계약

### ROS 없이 직접 시각화 (Open3D)

UDP 입력과 전처리는 그대로 두고 결과를 별도 Open3D 프로세스에 전달합니다.
ROS·rosbridge·RViz가 필요하지 않습니다. 현재 botsort 환경에 Open3D 0.19.0을 설치했습니다.
새 환경은 `python -m pip install -e "GPSJamming[viewer]"`로 의존성을 설치합니다.

기존 UDP 실행 프로그램을 Ctrl+C로 종료한 뒤 AIMDrive에서 실행합니다.

```powershell
& 'C:\conda-envs\botsort\python.exe' GPSJamming/scripts/run_udp.py --direct-view
```

기본 화면은 클러스터입니다. 한 창에서 같은 프레임의 전처리 단계를 전환합니다.

| 키 | 표시 또는 동작 |
| --- | --- |
| `1` | 원본 점군 |
| `2` | ROI 결과 |
| `3` | voxel 결과 |
| `4` | 제거된 지면 |
| `5` | 비지면 점군 |
| `6` | 클러스터 색상, 미분류 점은 회색 |
| `Space` | 화면 프레임 정지·재개 (수신·처리는 계속) |
| `N` | 정지 상태에서 최신 도착 프레임으로 한 번 갱신 |
| `C` | 클러스터 centroid의 흰색 십자 표시 전환 |
| `F` | 현재 점군 전체에 맞춰 시점 초기화 |
| `Q` / 창 닫기 | 뷰어 종료 (파이프라인은 계속, 전체 종료는 Ctrl+C) |

마우스로 회전·확대해 확인합니다. 선택 단계·프레임 번호·점 개수·완전성은 터미널의 `[viewer]` 줄에 표시됩니다.
좌표축은 X 전방, Y 좌측, Z 상방이며 회색 상자는 설정된 ROI 범위입니다.
초기 시점은 ROI에 맞춰지고 원본 점군에는 ROI 밖의 점도 포함됩니다.
점 수 6,000개 제한 없이 단계의 전체 유효 XYZ를 표시합니다. 클러스터 색은 프레임 내부 구분용입니다.

기록 입력을 재생하려면 다음과 같이 실행합니다. 기본 10Hz로 재생을 제한하고,
처리에 더 오래 걸리면 실제 속도는 낮아집니다. `--replay-hz 0`은 제한 없이 재생합니다.

```powershell
& 'C:\conda-envs\botsort\python.exe' GPSJamming/scripts/run_offline.py --input GPSJamming/artifacts/lidar_threaded_packets.jsonl --direct-view --replay-hz 10
```

뷰어 대기열은 한 프레임으로 제한합니다. 렌더링이 느리면 표시할 프레임을 건너뛰며,
종료 시 `viewer_skipped_frames`에 계수합니다. 화면 정지 중에도 최신 수신 데이터를 갱신하므로
`N`은 모든 프레임의 순차 재생이 아닙니다. 처리 알고리즘의 결과나 원본 timestamp는 바꾸지 않습니다.
`--output`을 생략하면 점군 JSON 저장 비용도 없습니다.
Open3D의 [비차단 시각화 방식](https://www.open3d.org/docs/release/tutorial/visualization/non_blocking_visualization.html)으로
기존 geometry를 갱신하며, Windows의 spawn 프로세스에서 창과 이벤트를 관리합니다.


1. `interfaces/lidar_udp.py`: 기존 `VLDrive/inference/morai_udp.py`의 벡터화 좌표 변환 방식을 옮겨 사용합니다. 원본 timestamp, 조립 검증, 별도 수신 스레드는 새로 구현했습니다. 기존 학습용 전처리의 무작위 샘플링·padding·회전은 사용하지 않습니다.
2. 지원 패킷은 1206바이트, 12블록×32반환, VLP-16 product ID `0x22`, single-return `0x37/0x38`입니다. dual-return·다른 모델·추가 헤더는 거부합니다. 두 firing에 동일 블록 방위각을 사용하는 기존 MORAI 방식이며 실제 회전형 장비의 firing 시간 보간은 포함하지 않습니다.
3. byte 1200의 little-endian uint32를 `raw_timestamp`로 보존합니다. 같은 timestamp가 한 측정 묶음인 MORAI 입력을 지원합니다. 패킷마다 timestamp가 달라지는 장비에는 이 조립 정책을 사용하면 안 됩니다.
4. timestamp별 버퍼에서 패킷 중복 제거, 방위각 정렬, 충돌 검출을 수행합니다. 첫 수신 후 설정한 시간 동안 재정렬을 허용한 뒤 닫습니다. 완료한 묶음의 늦은 패킷은 재출력하지 않고 계수합니다. 버퍼·패킷 수에 상한이 있습니다.
5. `complete`는 설정한 방위각 간격으로 360°를 덮었는지 검증한 결과입니다. 75패킷이라는 개수는 사용하지 않습니다. 빈 거리 반환도 방위각 커버리지에는 포함하고 점군에서는 제외합니다. 누락·충돌·강제 퇴출은 `issues`에 표시하며, 불완전 프레임도 진단용으로 출력합니다.
6. 수신과 처리를 분리했습니다. 처리 대기열이 차면 오래된 조립 프레임을 버리고 `processing_dropped_frames`로 알립니다. `frame_id`는 출력 이전에 부여하므로 중간 누락도 확인할 수 있습니다. OS에서 전체 프레임이 유실된 경우는 패킷만으로 확정할 수 없습니다.
7. 전처리는 유한값·XYZ ROI 필터링과 voxel 평균입니다. intensity도 평균하며 원본 intensity 값에 XYZ 제한을 적용하지 않습니다. 지면은 예상 높이·기울기 제약 RANSAC 및 평면 재추정으로 제거합니다. 근거가 부족하면 점을 유지합니다.
8. KD-tree로 거리 이내 점 쌍을 일괄 검색하고 SciPy 희소 그래프의 연결 성분으로 군집을 계산한 뒤 점 개수로 필터링합니다. 점 인덱스와 군집 순서는 기존 순회 방식과 동일하게 유지합니다. 점 쌍이 100만 개를 넘으면 메모리 급증을 막기 위해 기존 점별 탐색으로 전환합니다. centroid·축 정렬 크기·점 개수는 전처리 후 각 클러스터 점군에서 계산합니다. `cluster_id`는 매 프레임 새로 부여됩니다.

### 출력과 시간

`LidarFrame`은 `frame_id`, `raw_timestamp`, `timestamp`, `receive_monotonic_time`,
`receive_end_monotonic_time`, `coordinate_frame`, `complete`, `points`, `clusters`,
`issues`, `timestamp_status`, `clock_epoch`, `diagnostics`를 가집니다.
`points`와 각 cluster의 `points`는 Nx4 `[x, y, z, intensity]`이며 XYZ는 미터입니다.
수신 점군은 센서 원점 기준 +x 전방, +y 좌측, +z 상방입니다. 기본 설정에서는 ROI 전에 장착 위치 `[1.0, 0.0, 1.45]` m와 회전 `[0.0, 0.0, 0.0]` 도(roll, pitch, yaw)를 적용해 `base_link`로 변환합니다. 출력 점군·클러스터·디버그 시각화의 모든 단계는 `base_link` 기준이며, 원본 단계는 변환 후 필터링 전 점군입니다. ROI와 `ground.expected_z`도 이 좌표계 기준입니다. 지면 높이 초기값 `-0.03` m는 기존 센서 기준 추정값 `-1.48` m에 장착 높이를 더한 값입니다.

- `raw_timestamp`: 수신한 원본 uint32. 단위나 epoch를 임의로 바꾸지 않습니다.
- `timestamp`: 공통 절대시각 **Unix ns**. 기준점이 확인되지 않으면 `null`입니다.
- `receive_monotonic_time`: 첫 패킷 수신의 monotonic ns. 센서 timestamp를 대체하지 않습니다.
- `receive_end_monotonic_time`: 마지막 패킷 수신의 monotonic ns.

`timing.lidar`에 검증된 `ns_per_tick`, `source_anchor`, `unix_anchor_ns`를 넣으면
`unix_anchor_ns + (raw_timestamp - source_anchor) * ns_per_tick`으로 변환합니다.
카메라도 같은 절대시각 기준을 사용해야 합니다. 현재 실제 입력의 절대 기준점은 미확정이므로 기본값은 `null`입니다.
시간 역행·큰 순방향 급변을 감지하면 epoch를 올리고 기준점을 무효화합니다.
시간 wrap/reset과 늦게 도착한 과거 프레임을 임의로 구분하거나 보정하지 않습니다.
이후 동기화에서는 `timestamp_status=mapped`이며 `complete=true`인 프레임을 선택해야 합니다.

## 검증과 한계

2026-09-27 실제 로컬 MORAI UDP를 5초 수신한 최종 확인에서 48프레임을 출력했습니다.
48개 모두 방위각 커버리지를 통과했고, 유효 점 16,374개와 후보 31개가 프레임별로 나왔습니다.
잘못된 패킷 및 처리 대기열에서 버린 프레임은 0개였습니다.
캡처의 900개 블록은 방위각 간격 0.39/0.40/0.41°였으며 설정은 0.4°, 허용 오차 0.02°입니다.
센서 설정을 바꾸면 이 값을 다시 검증해야 합니다.

검증 산출물은 `artifacts/lidar_threaded_verified.jsonl`, `lidar_threaded_packets.jsonl`,
`lidar_threaded_bev.png`에 있습니다. Git에는 포함하지 않습니다.
합성 테스트에서는 평지·완만한 경사에서 지면 제거 후 차량 크기 후보와 보행자 크기 후보를 보존하고,
누락·중복·순서 변경·잘못된 패킷·버퍼 제한·timestamp 보존을 검증합니다.

이는 객체 **후보** 검출입니다. 도로 경계·벽·잔여 지면도 클러스터가 될 수 있고,
근접 객체는 합쳐지거나 멀리 있는 객체는 나뉠 수 있습니다. 단일 평면 지면 모델은 급경사·굴곡 도로에 한계가 있습니다.
실제 차량·보행자 정답과 비교한 검출률, 움직이는 장면의 centroid 안정성, 장시간 부하와 절대시각 정합은 아직 검증하지 않았습니다.
후속 순서는 정지 장면 보정·투영 → 카메라 시간 매칭 → association → 동적 장면 검증 → 속도 계산입니다.


## GPS·IMU·차속 위치 추정

`gps_jamming/localization/`은 인지 파이프라인과 독립적으로 위치만 추정합니다.
차량 제어 명령을 보내지 않습니다. 설치는 `python -m pip install -e "AIM2026/GPSJamming[localization]"`입니다.
기록 재생·평가·오프라인 HTML 보고서의 실행 방법과 검증 범위는 [localization.md](docs/localization.md)를 참고하세요.
