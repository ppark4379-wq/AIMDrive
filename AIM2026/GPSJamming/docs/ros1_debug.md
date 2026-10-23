# WSL Ubuntu 20.04 / ROS Noetic LiDAR 디버깅

수정 및 ROS 로그 저장 위치는 GPSJamming 내부이다. Windows Git 작업 폴더를
WSL의 `/mnt/c`로 그대로 사용하며 복사하거나 catkin 빌드할 필요가 없다.
센서 UDP 수신 → 기존 LiDAR 처리 → ROS PointCloud2 → RViz 구조이다.
지면 RANSAC과 구간별 벽 후보 경계 추정을 표시한다.

## 실행

WSL Ubuntu 터미널 3개에서 각각 아래 경로로 이동한다.

```bash
cd '/mnt/c/Users/장원정/AIMDrive/AIM2026/GPSJamming'
```

첫 번째 터미널:

```bash
bash run_ros1.sh master
```

두 번째 터미널:

```bash
bash run_ros1.sh node
```

세 번째 터미널:

```bash
bash run_ros1.sh rviz
```

메시지 직렬화 및 LiDAR 회귀 테스트는 `bash run_ros1.sh test`로 실행한다.

각 프로세스는 Ctrl+C로 종료한다. 기존 UDP 수신 프로그램은 먼저 종료한다.
GUI는 WSLg 또는 사용자가 구성한 X 서버가 필요하다.

## Windows 시뮬레이터 → WSL UDP

노드는 기본 `0.0.0.0:2369`에서 수신한다. 원래 perception.yaml은 변경하지 않는다.
WSL NAT 환경에서는 Ubuntu에서 `hostname -I`로 확인한 WSL IPv4를
MORAI LiDAR의 Destination IP로 설정하고 Destination Port는 2369로 설정한다.
`0.0.0.0`은 수신 bind 주소이며 시뮬레이터의 목적지 주소가 아니다.
WSL 재시작으로 IP가 바뀌면 목적지 설정도 갱신한다.
Windows의 127.0.0.1로 보낸 UDP가 NAT의 WSL로 자동 전달된다고 가정하지 않는다.
방화벽이나 WSL 네트워크 설정은 이 스크립트가 변경하지 않는다.

```bash
bash run_ros1.sh node _bind:=0.0.0.0 _port:=2369
```

## 토픽과 좌표

- `/gps_jamming_lidar/raw`: base_link로 변환된 전체 점군(센서 원시 좌표가 아님)
- `/gps_jamming_lidar/roi`: ROI 점군
- `/gps_jamming_lidar/voxel`: 다운샘플 점군
- `/gps_jamming_lidar/ground`: 지면 inlier, 초록
- `/gps_jamming_lidar/nonground`: 비지면 점군, 빨강(벽으로 확정된 점이 아님)
- `/gps_jamming_lidar/status`: JSON 상태, 검출 진단 및 시간 출처

PointCloud2는 XYZI float32, 미터 단위이다. Fixed Frame은 기본 설정의
`base_link`이다. 같은 좌표계로 이미 변환되어 별도 TF는 필요하지 않다.
사용자 설정에서 target_frame을 바꿨다면 RViz Fixed Frame도 일치시킨다.
모든 단계에 같은 stamp를 사용한다. 센서 시간 매핑이 없으면 시각화용 ROS
발행 시각을 쓰며 status의 stamp_source에 명시한다. 센서 획득 시각이 아니다.
이 단계에서는 `/use_sim_time`을 켜지 않는다(별도 /clock 입력을 구성하지 않음).

```bash
source /opt/ros/noetic/setup.bash
rostopic hz /gps_jamming_lidar/ground
rostopic echo /gps_jamming_lidar/status
```

점군이 없으면 UDP 목적지 IP/port와 수신 프로세스 중복을 먼저 확인한다.
본 디코더는 프로젝트의 MORAI VLP-16 형식 전용이다.

## Python 호환성

프로젝트 전체 패키지의 Python 요구 사항은 3.10+로 유지한다.
이 ROS 실행 경로는 Noetic의 `/usr/bin/python3`(3.8)에서 소스 직접 실행하며
지연 타입 주석을 사용한다. `pip install -e .`는 이 경로에 필요하지 않다.
rospy, sensor_msgs, std_msgs, numpy, scipy, yaml이 해당 인터프리터에 필요하다.
실행 스크립트는 `artifacts/ros_python`의 전용 의존성을 우선 사용한다.
재현 설치 명령(WSL에서 프로젝트 폴더를 작업 디렉터리로 사용):

```bash
python3 -m pip install --no-cache-dir --target artifacts/ros_python numpy==1.24.4 scipy==1.10.1 PyYAML==6.0.2
```

실행 스크립트 자체는 패키지 설치나 시스템 Python 변경을 하지 않는다.

참고: [Microsoft WSL 네트워크 문서](https://learn.microsoft.com/windows/wsl/networking)

## 벽 후보 경계 (2026-10-23)

현재 센서 설정은 MORAI Host 127.0.0.1:2369 → Destination 127.0.0.1:2368이다.
WSL mirrored 모드에서 localhost 수신을 확인했다. 노드는 아래처럼 실행한다.

```bash
bash run_ros1.sh node _port:=2368
```

`configs/perception.yaml`의 `walls`에서 검출 조건을 조정한다.
기본 `observed_extent: true`에서는 화면의 빨간 점과 같은 현재 프레임의
`nonground` 점을 사용한다. `x_min/x_max/max_height: null`은 이 입력의 관측
범위를 사용한다는 뜻이며 전처리 ROI를 우회하지 않는다. 기본 ROI의 전방 한계는
30m이고, 더 멀리 사용할 때는 표시 점과 검출 입력에 공통으로 적용되는 ROI를 조정한다.
좌우 각각 전체 관측으로 직선 벽을
추정하므로 먼 5m 구간에 수직 링이나 점이 적어도 같은 벽의 지지점으로 포함된다.
좌우 Y 부호와 `min_lateral`은 초기 직선 후보를 찾을 때만 사용한다. 후보의 원래
피팅 직선(`seed_line_slope/intercept`) 주변 점을 Y 부호와 관계없이 다시 모으며,
두 후보 모두 가까운 점은 더 가까운 후보 하나에만 배정한다. 벽 이름은 유지하므로
비스듬한 왼쪽 벽이 먼 곳에서 Y<0으로 넘어가도 왼쪽 벽의 관측 길이에 포함된다.
최종 경계가 Y=0을 가로지른다는 이유로 선 전체를 폐기하지 않는다. 이는 벽 검출
기하 정보이며, 차량이 해당 경계를 통과해도 된다는 의미는 아니다.
높이 분포·최소 길이·지지율은 전체 벽에서 검증한다. 이 모드는 좌우의 지배적인
직선 벽을 위한 것이며, 굽거나 꺾인 벽 전체를 하나의 직선으로 검출하는 기능은 아니다.
전방 간격이 `max_observation_gap`보다 크면서 방위각 공백도
`angular_gap_deg`(기본 1.2도)를 넘으면 별도 선분으로 분리한다.
거리 공백이 `max_observation_gap`의 두 배(기본 4m)를 넘으면 각도와 관계없이 분리한다.
센서 각 해상도를 변경하면 이 각도 기준도 확인한다. 실제 벽 지지점의 끝점까지
출력하며 센서 최대 사거리까지 임의로 외삽하지 않는다.

`observed_extent: false`로 기존 구간별 방식을 사용할 때는 유한한
`x_min/x_max/max_height`를 함께 설정한다. 기존 방식은 아래와 같다.
지면 대비 높이 0.3~2.2m, 차량 좌우 1.5m 바깥의 비지면 점을 선택하고,
전방 5m 구간마다 XY RANSAC 직선을 추정한다. 이는 수직 벽을 가정한
국소 평면 y=mx+b 모델이며 일반적인 기울어진 3D 벽면 모델은 아니다.
높이 분포 1m 이상, 관측 길이 3m 이상 및 지지율 조건을 통과해야 한다.
노면 높이는 같은 프레임의 지면 평면에서 계산한다.
출력은 0.5m 간격 점열이다. 인접 구간의 끝점 간격 2.5m 이하,
이전 벽 방향으로 예측한 횡방향 오차 0.5m 이하, 방향 차이 20도 이하이면
동일 경계 그룹으로 묶는다. `connect_max_gap/offset/angle_deg`로 조정한다.
최종 선은 RANSAC 직선 연결이 아니라, 그룹의 실제 벽 inlier에서 구한 대표점의
최소제곱 직선이다. 대표점 범위 밖으로 외삽하지 않는다.
기본 방향 추정 허용 거리는 25cm, 최소 지지점은 8개, 지지율은 30%, 높이 분포는
0.5m, 관측 길이는 2m이다. 경계 위치 계산에는 검출된 벽 직선의
80cm 이내(`boundary_support_distance`)의 실제 관측점을 추가해 돌출부와 성긴 원거리 벽을 포함한다.
선의 끝점은 이 전체 벽 지지점으로 정하며, 25cm 방향 추정 지지점의 끝으로 다시 자르지 않는다.
`line_fit_range`에는 방향 추정에 사용한 범위, `observed_range`에는 최종 벽 관측 범위를 기록한다.
그 점들이 모두 벽 쪽에 놓이도록 지지선을 옮긴 뒤 차량 쪽으로 수직 거리
30cm(`clearance_m: 0.30`) 이동한다. 80cm 밖의 점까지 벽으로 분류하는 것은 아니다.
지면 검출 실패 시 경계를 생성하지 않는다. 원본 좌표에는 시간 누적이 없다.
RViz 마커는 현재 프레임 경계를 직접 표시한다. `wall_boundaries_filtered`의
EMA 결과는 진단용으로만 발행하며, 이전 프레임의 끝점을 마커에 유지하지 않는다.
한쪽 검출이 사라지면 `wall_width_fallback`이 현재 반대쪽 선과 최근 측정한 경계 간
수직 폭으로 보완한다. 양쪽이 3프레임 연속 일치해야 폭을 사용하고 마지막 폭 측정부터
최대 2초까지만 보완한다. 두 선 모두 없거나 불완전 프레임이면 보완하지 않는다.
보완 길이는 현재 반대쪽 길이와 누락 측의 마지막 관측 길이의 교집합이다.
폭은 이미 30cm 여유가 반영된 경계 사이 거리이므로 60cm를 다시 빼지 않는다.
보완선은 반투명이며 `/wall_boundaries_display`에 `inferred: true`, `valid: false`로
발행한다. 원본 `/wall_boundaries`와 UDP 검출 결과에는 보완선을 넣지 않는다.

- `/gps_jamming_lidar/wall_left`, `wall_right`: 벽 후보 inlier PointCloud2
- `/gps_jamming_lidar/wall_representatives_left`: 곡선 피팅 전 대표점, 청록색
- `/gps_jamming_lidar/wall_representatives_right`: 곡선 피팅 전 대표점, 노란색
- `/gps_jamming_lidar/wall_markers`: MarkerArray, 왼쪽 파랑 / 오른쪽 주황
- `/gps_jamming_lidar/wall_boundaries`: std_msgs/String에 JSON 결과

검출 노드를 재시작해야 새 설정과 코드가 적용된다. RViz만 재시작해서는 적용되지 않는다.
Marker는 같은 namespace/id로 갱신하며,
이전 출력에서 사라진 ID만 개별 삭제한다. 매 프레임 DELETEALL을 보내지 않는다.

JSON 계약:

- `schema_version=1`, `frame_id=base_link`, `frame_sequence`: 프레임 식별
- `stamp_ns`, `stamp_source`, `sensor_timestamp_ns`: 발행 또는 매핑된 센서 시각
- `complete`: 원본 점군 수집 완전성
- `valid.left/right`: 해당 측 경계 존재 및 완전한 프레임 여부. 안전성 보증 아님
- `left/right`: 연결 가능한 구간을 합친 경계 목록. 각 항목에 `points` XYZ(m), `observed_range`,
  `inliers`, `support_ratio`, `rmse_m`, `confidence` 포함
- `confidence`: 지지율과 잔차를 이용한 기하 점수, 확률이 아님
- `semantics=geometric_wall_candidates_not_drivable_area`

`observed_range`는 합친 전체 범위이며, 실제 지지 구간은 `observed_ranges`,
연결한 간격은 `interpolated_ranges`, 각 보간점 여부는 `interpolated`에 기록한다.
confidence/support_ratio는 구성 구간의 최솟값, rmse_m은 최댓값이다.
판제는 별개 경계 목록 사이를 추가로 연결하지 않고, 빈 목록/valid=false/오래된 메시지를
검출 실패로 처리해야 한다. 현재 결과는 차선이나 완성된 주행가능영역이 아니다.
차량 폭·여유 거리 반영 및 벽 사이 장애물 회피는 아직 포함하지 않는다.
대형 차량 옆면, 평평한 시설물도 벽 후보가 될 수 있다. 수직에서 크게 기울어진 벽,
급곡선, 희소한 원거리 점군은 검출 누락 또는 오차가 생길 수 있다.
실제 터널에서 RViz로 검증하기 전 자동 제어 입력으로 확정하지 않는다.

```bash
rostopic echo -n 1 /gps_jamming_lidar/wall_boundaries
```

## 프레임별 직선 경계

시간 필터 및 급변 시 선을 숨기는 처리는 실행 경로에서 제거했다.
wall_boundaries는 현재 프레임의 직선 좌표를 유지한다.
RViz는 아래 EMA 표시 결과를 사용한다. 원본 검출은 ID나 속도를 사용하지 않는다.

`walls` 설정:

- `curve_bin_size: 0.5`: 전방 대표점 추출 구간(m)
- `curve_quantile: 0.25`: 차량 쪽 분위수. 왼쪽 y의 25%, 오른쪽 y의 75%

구간별 x 중앙값과 차량 쪽 y 분위수를 대표점으로 사용한다.
연결 그룹의 대표점 전체로 최소제곱 직선 방향 m을 구한 후,
그룹의 모든 벽 inlier가 벽 쪽 반평면에 놓이도록 절편 b를 이동한다.
왼쪽은 min(y-mx), 오른쪽은 max(y-mx)를 기준으로 한다.
`walls.clearance_m: 0.20`에 따라 BEV에서 차량 쪽으로 수직 거리 20cm 이동한다.
절편에 -sign*clearance_m*sqrt(1+m*m)을 적용하며 z는 이동한 XY의 지면 높이다.
JSON의 clearance_m에 설정값을 기록한다. 대표점 및 벽 inlier 토픽은 이동하지 않는다.
EMA는 이동한 원본 선을 입력받지만 과거 관측과 섞이므로 현재 점군과 정확히
20cm 간격인 것은 원본 경계 기준이다. 선분 양 끝은 벽 inlier의
최소/최대 x까지 포함한다. 검출에서 제외된 원시 점은 포함 보장 대상이 아니다.
극단값을 사용하므로 잘못 검출된 점 하나에도 경계가 안쪽으로 이동할 수 있다.
스플라인이나 꺾은선 보간을 사용하지 않는다. 실제 곡선 벽도 직선으로 근사하므로
오차를 확인해야 한다. 분위수는 최내측 점의 보장은 아니다.
출력에 `representatives`, `curve_model`, `curve_rmse_m`, `facing_quantile`을 추가했다.
기존 필드명은 유지하며 `curve_model=vehicle_facing_support_line`으로 표시한다.
`enclosing_point_count`는 포함 대상 점 수, `support_offset_m`은 평균선 대비 이동 거리다.
`line_slope`, `line_intercept`는 직선 계수이고 `curve_rmse_m`은 직선의 대표점 오차이다.
대표점 PointCloud2의 z는 지면 평면 높이이고 intensity=1은 표시용 상수이다.
원본 LiDAR 점을 그대로 고르는 것이 아니라 x 중앙값과 y 분위수로 계산한 점이다.
대표점 토픽은 벽 검출 및 연결 그룹 구성 뒤, 곡선 피팅 전에 생성한다.
곡선 생성에 실패해도 추출된 대표점은 표시하며, 벽 검출에 탈락한 점은 포함하지 않는다.
기존 rmse_m/confidence는 벽 검출 품질이며 곡선 오차는 curve_rmse_m로 구분한다.

## EMA와 짧은 경계 유지

`wall_ema` 설정으로 RViz 표시와 `/gps_jamming_lidar/wall_boundaries_filtered`
(std_msgs/String JSON)에 EMA를 적용한다. 원본 `/wall_boundaries`와 대표점은 유지한다.

- `time_constant: 0.15`: EMA 시정수(초). 실제 관측 간격 dt로 alpha=1-exp(-dt/tau) 계산
- `hold_seconds: 0.3`: 마지막 관측부터 유지 가능한 최대 시간(10Hz에서는 약 3프레임)
- `sample_step: 0.5`: 동일 전방 x 위치에서 비교하는 격자 간격(m)
- `reset_jump: 1.0`: 큰 위치 변화는 숨기지 않고 새 관측으로 초기화하는 기준(m)

현재 보이는 부분은 EMA로 갱신하고 사라진 부분만 짧게 유지한다.
격자상 연속 구간은 직선으로 다시 맞춰 굽이치는 선을 방지한다.
좌우 각각 고정 namespace/id의 LINE_LIST Marker 하나로 표시한다.
투명도는 1로 고정하고, 부분 검출 상태에 따라 표시 개체를 나누지 않는다.
LINE_LIST의 각 점 쌍은 독립된 선분이므로 실제 빈 구간을 가로질러 연결하지 않는다.
실제로 관측하지 않은 공백을
새로 채우지는 않으며, 반복된 누락이 유지 시간을 연장하지 않는다.
좌표계 변경, 시간 역행 및 긴 입력 공백에서 이력을 초기화한다.
관측 시각은 수신 프레임의 monotonic 시각을 쓰며 처리 대기 시간도 수명에 포함한다.

필터 결과는 `semantics=ema_display_only_no_ego_compensation`이다.
각 선분의 `held`, `observation_age_s`, `contains_held`, `max_age_s`를 확인할 수 있다.
유지한 부분을 포함한 측은 valid=false이며, 이것은 원본 토픽의 valid와 별개이다.
관측 유지 0.3초는 EMA가 관리하고, 만료한 부분은 다음 프레임 갱신에서 제거한다.
Marker lifetime은 관측 유지와 분리한 0.5초 통신 watchdog이다.
따라서 짧은 갱신 간격 때문에 선이 자체 만료되지 않으며, 발행이 멈추면
마지막 표시 후 0.5초 이내에 제거된다. held 여부는 JSON 필드로 확인한다.
필터에는 자차 이동·회전 보정이 없어 주행 중 지연이 생길 수 있다.
EMA와 재피팅 결과는 현재 벽 점 전체를 감싸는 성질도 보장하지 않는다.
따라서 이 토픽은 시각화 비교용이며, 판제에는 원본 경계의 품질과 시각을 확인해야 한다.

`bash run_ros1.sh audit`는 실행 중인 원본·필터·Marker 토픽을 8초간 읽어
빈 경계 프레임 수, 최대 발행 간격, 짧은 Marker 수명을 출력한다.
