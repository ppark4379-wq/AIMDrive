#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""레티스·정적 장애물·고속도로 전환·회전교차로 파라미터.

찾아보기:
  1. 레티스 경로 후보 생성 및 충돌 검사
  2. 정적 장애물 사전 계획·속도·추종·복귀
  3. 고속도로 진입 영역 및 레티스/차간거리 제어 전환
  4. 회전교차로 진입 및 차량 예측

고속도로 차선변경·첫 번째/두 번째 합류: highway.py
정적 장애물 자세 오차 감속: static_obstacle.py의 자세 정렬 항목
정적 장애물 통로 탐색: static_obstacle.py의 통로 탐색 항목
OBB는 차량 방향을 반영한 충돌 검사 사각형, TTC는 예상 충돌 시간이다.
"""


# =============================================================================
# 1. 레티스 플래닝: 경로 후보 생성·평가·충돌 검사
# =============================================================================
# 후보 개수와 탐색 거리
NUM_OFFSETS = 17                     # 장거리·단거리 각 계층에서 생성할 횡방향 후보 개수
LATERAL_OFFSET_STEP = 1.0            # 인접 후보 경로 사이의 횡방향 간격 [m]
SHORT_LOOKAHEAD_DISTANCE = 15.0      # 실제 회피 경로를 생성·선택하는 단거리 계층 길이 [m]
LONG_LOOKAHEAD_DISTANCE = 35.0       # 전방 장애물 조기 감지에 사용하는 장거리 계층 길이 [m]
OBSTACLE_DETECTION_DISTANCE = 35.0   # 레티스 활성화를 검토할 전방 정적 장애물 최대 거리 [m]
OBSTACLE_PATH_THRESHOLD = 3.0        # 중앙 경로 주변 장애물을 관련 대상으로 보는 횡거리 [m]

# 차량 크기와 충돌 검사 여유: 정적 장애물 플래너에서도 공통 사용
VEHICLE_LENGTH = 4.635               # 자차 충돌 검사에 사용하는 차량 길이 [m]
VEHICLE_WIDTH = 2.0                  # 자차 충돌 검사에 사용하는 차량 폭 [m]
COLLISION_MARGIN = 0.5               # 차량 충돌 검사 사각형에 추가하는 안전 여유 [m]
SAMPLE_SPACING = 0.5                 # 레티스 후보의 종방향 충돌 검사 샘플 간격 [m]

# 후보 선택 비용과 장거리 장애물 감속
RIGHT_PREFERENCE_WEIGHT = 5.0        # 중앙·왼쪽 후보에 추가하는 비용; 클수록 오른쪽 선호
CURVATURE_COST_WEIGHT = 0.35         # 최대 곡률 비용 가중치; 클수록 완만한 경로 선호
LONG_OBSTACLE_SPEED_SCALE = 0.4      # 장거리 중앙 후보 충돌 시 목표속도 배율; 1.0이면 감속 없음


# =============================================================================
# 2. 정적 장애물 플래닝: 사전 계획·회피 속도·추종·전역 경로 복귀
# =============================================================================
# 사전 계획과 경로 미리보기
STATIC_PREPLAN_ENTRY_Y = -164.0      # 사전 계산할 회피 경로의 고정 진입점 전역 y 좌표 [m]
STATIC_PREVIEW_POINTS = 200          # 정적 장애물 계획에 제공할 전방 경로점 개수
STATIC_RESET_DISTANCE = 15.0        # 한 번에 이 거리 이상 이동하면 순간이동으로 보고 계획 초기화 [m]

# 기록·지도 경로의 작은 굴곡 제거: 수정한 경로는 차체 경계·장애물 검사를 다시 통과해야 한다.
STATIC_PATH_DESIGN_SPEED_KPH = 20.0  # 회피·복귀 경로 선택 시 감속 없이 추종하고자 하는 기준속도; 강제 최저속도 아님 [km/h]
STATIC_PATH_SMOOTHING_RMS = 0.005    # 기록·지도 점의 작은 흔들림을 제거할 기본 평균제곱근 오차 [m]
STATIC_PATH_SMOOTHING_MAX_DEVIATION = 0.05 # 평활화 경로가 원래 경로에서 벗어날 수 있는 최대 횡거리 [m]

# 곡률·조향·통로 여유에 따른 속도 제한
# 곡률 기준속도 [m/s] = sqrt(STATIC_LATERAL_ACCEL / abs(곡률)) + 추가속도 / 3.6
# 추가속도는 곡률 기준에만 적용한다. 최종 속도에는 경로·신호·충돌·추종 제한도 적용한다.
STATIC_MAX_SPEED_KPH = 55.0          # 정적 장애물 회피 속도 상한; 다른 제한이 낮으면 그 값을 적용 [km/h]
STATIC_LATERAL_ACCEL = 1.5           # 추가속도를 더하기 전 곡률 기준속도 계산용 횡가속도 [m/s^2]
STATIC_CURVATURE_SPEED_OFFSET_KPH = 5.0 # 곡률 기준속도에 더하는 값; 다른 제한에는 가산하지 않음 [km/h]
STATIC_WHEELBASE = 3.0               # 곡률을 조향각으로 변환할 축거; 추종 제어기와 같은 값 [m]
STATIC_STEERING_RATE = 0.7853981634  # 속도 계산에 사용하는 앞바퀴 최대 조향 변화율; 45도/초 [rad/s]
# 진단 로그의 corridor_tracking은 통로 여유와 곡률을 반영한 추종 속도 제한이다.
STATIC_STEERING_RESPONSE_TIME = 0.35 # 통로 여유 기반 추종 속도 계산용 응답시간; 클수록 감속 강화 [s]
STATIC_SPEED_PREVIEW = 3.0          # 앞쪽 속도 제한에 도달하기 전에 확보할 감속 거리 여유 [m]
STATIC_TARGET_ACCEL = 1.2           # 회피 및 전역 경로 복귀 중 목표속도의 최대 상승률 [m/s^2]
STATIC_END_DECELERATION = 2.0       # 전방 속도 제한 및 안전한 복귀 경로가 없을 때 정지 계산용 감속도 [m/s^2]
STATIC_END_BUFFER = 1.0             # 유효 경로 끝에 도달하기 전에 남길 정지 여유 [m]

# 곡률과 경로 추종
STATIC_MAX_CURVATURE = 0.24          # 차량 조향 가능 범위 안에서 허용할 경로 최대 곡률 [1/m]
STATIC_TRACKING_MARGIN = 0.1         # 계획 경로 충돌 검사에 추가할 추종 오차 여유 [m]
STATIC_TRACKING_LOOKAHEAD = (1.0, 1.5) # 좁은 회피 경로 추종의 기본 전방주시거리 하한·상한 [m]
# 자세 오차별 감속 시작/최대 감속 기준은 recorded_planner.py의 ALIGNMENT_*에서 설정한다.
# 기록 경로 추종에서는 속도와 전방 곡률에 따라 전방주시거리 상한을 추가로 조정한다.

# 레티스 기반 대체 회피의 경계와 경로 검사
# 좌측은 양수, 우측은 음수다. 아래 경계는 설정된 차체 허용영역이며 실측 도로 경계가 아니다.
STATIC_MAX_RIGHT_OFFSET = 3.0        # 중앙 경로에서 차량 중심을 오른쪽으로 옮길 최대 거리 [m]
STATIC_LEFT_BOUNDARY = 1.75          # 중앙 경로 기준 차체가 넘어가면 안 되는 좌측 경계 [m]
STATIC_RIGHT_BOUNDARY = -4.5         # 중앙 경로 기준 차체가 넘어가면 안 되는 우측 경계 [m]
STATIC_BOUNDARY_MARGIN = 0.2         # 차체와 설정 경계 사이의 여유; 장애물 충돌 여유와 별개 [m]
STATIC_SAMPLE_SPACING = 0.2          # 대체 회피 경로와 복귀 연결 경로의 샘플 간격 [m]
STATIC_SIDE_TOLERANCE = 0.1          # 횡방향 회피 범위를 검사할 때 허용할 수치·추종 오차 [m]
STATIC_MAX_TRACKING_ERROR = 2.0      # 대체 회피 경로를 유효하다고 보는 자차와 최근접 경로점 간 거리 [m]

# 레티스 기반 대체 회피의 전역 경로 복귀
STATIC_RETURN_MAX_DISTANCE = 35.0   # 회피 경로 끝에서 복귀할 전역 경로점을 찾는 최대 거리 [m]
STATIC_RECOVERY_LOOKAHEAD = 2.0      # 회피 경로 끝까지 남은 거리가 이 값 이하면 복귀 연결 계산 [m]
STATIC_RECOVERY_MAX_CURVATURE = 0.17 # 복귀 연결에 허용할 최대 곡률; 조향 한계 대비 추종 여유 확보 [1/m]

# 기본 정적 장애물 구간의 회피 시작 기준선
# MGeo 노드 A1256W000807, 신호 C1256W000091의 좌표를 기존 자차 충돌 상자 앞면과 비교한다.
# 기준선 전에는 원래 경로를 유지하며, 이 좌표 자체가 강제 정지 명령을 만들지는 않는다.
STATIC_PLANNING_STOP_POINT = (-59.83293701335788, -155.39320968370885) # 기준선 전역 (x, y) [m]
STATIC_PLANNING_STOP_TOLERANCE = 0.05 # 차량 앞면의 기준선 통과 판정 허용오차 [m]


# =============================================================================
# 3. 고속도로 플래닝: 활성화 영역·레티스/차간거리 제어 전환
# =============================================================================
# 차선변경·합류 세부 설정은 highway.py에서 조정한다.
# ACC는 선행차와의 간격을 유지하는 속도 제어다. 진입/해제 기준을 달리해 잦은 전환을 막는다.
HIGHWAY_ZONE_HALF_WIDTH = 6.0        # 고속도로 중심선 주변에서 정책을 활성화할 반폭 [m]
HIGHWAY_RISK_LATERAL_THRESHOLD = 2.5 # 거리·충돌시간 위험 판단에 포함할 동일차로 횡거리 [m]
HIGHWAY_ACC_ENTER_DISTANCE = 10.0    # 이 거리 이하이면 레티스 회피를 중단하고 ACC 우선 진입 [m]
HIGHWAY_ACC_RELEASE_DISTANCE = 15.0  # ACC 우선 상태 해제에 필요한 거리 기준 [m]
HIGHWAY_ACC_ENTER_TTC = 2.0          # 예상 충돌 시간이 이 값 이하이면 ACC 우선 진입 [s]
HIGHWAY_ACC_RELEASE_TTC = 3.0        # ACC 우선 상태 해제에 필요한 예상 충돌 시간 기준 [s]
HIGHWAY_LATTICE_SPEED_SCALE = 1.0    # 안전한 고속도로 레티스 회피 상태의 목표속도 배율


# =============================================================================
# 4. 회전교차로 플래닝: 진입 영역·양보·차량 예측
# =============================================================================
# 진입 영역 및 정지 기준 좌표
# 정지선은 측정점 (-99.17, 317.79)에 가장 가까운 MGeo 노드 A1256W000709를 사용한다.
ROUNDABOUT_ENTRY_APPROACH_START = (-100.32, 290.23) # 진입 접근 영역 시작점 전역 (x, y) [m]
ROUNDABOUT_ENTRY_APPROACH_END = (-93.1035536401546, 327.685278548199) # 접근 영역 끝점 전역 (x, y) [m]
ROUNDABOUT_ENTRY_APPROACH_HALF_WIDTH = 5.0 # 접근 영역 중심선 좌우 반폭 [m]
ROUNDABOUT_ENTRY_STOP_POINT = (-99.312672408123, 317.798954707105) # 정지선 노드 전역 (x, y) [m]
ROUNDABOUT_ENTRY_HOLD_POINT = (-99.02, 319.29) # 정지 시 자차 기준점의 측정 좌표; 범퍼 좌표와 다름 [m]
ROUNDABOUT_ENTRY_MERGE_POINT = ROUNDABOUT_ENTRY_APPROACH_END # 진입 완료 판단용 합류점 전역 (x, y) [m]
ROUNDABOUT_ENTRY_CIRCULATING_LINK_ID = "A2256W000134" # 합류점의 회전 차로; 후방 차량 방향 판정 기준
ROUNDABOUT_HOLD_TOLERANCE = 0.25      # 정지 목표 위치 도달 및 통과 판정 허용오차 [m]
ROUNDABOUT_ENTRY_STOP_BACK_DISTANCE = 1.2 # 정지 목표 위치 뒤쪽으로 확보할 진입 판단 영역 [m]
ROUNDABOUT_ENTRY_STOP_FORWARD_DISTANCE = 1.2 # 정지 목표 위치 앞쪽으로 확보할 진입 판단 영역 [m]

# 주행 속도와 전방 경로
ROUNDABOUT_APPROACH_VELOCITY = 30.0 / 3.6 # 정지 목표 위치 접근 속도 상한; 15 km/h [m/s]
ROUNDABOUT_ENTRY_CRUISE_VELOCITY = 30.0 / 3.6 # 진입 주행 속도 상한; 30 km/h [m/s]
ROUNDABOUT_ENTRY_ACCEL_P_GAIN = 0.80 # 진입 허용 후 단독 가속에만 적용할 속도 PID P 게인
ROUNDABOUT_PREVIEW_POINTS = 200       # 회전교차로 예측에 제공할 전방 경로점 개수

# 충돌 예측과 선행차 추종
ROUNDABOUT_PREDICTION_HORIZON = 8.0    # 미래 차량 위치를 검사할 예측 시간 범위 [s]
ROUNDABOUT_PREDICTION_STEP = 0.1       # 미래 위치 및 충돌 검사 시간 간격 [s]
ROUNDABOUT_PREDICTION_ACCELERATION = 1.5 # 자차 미래 이동거리 예측에 사용할 가속도 [m/s^2]
ROUNDABOUT_SLOW_ACCELERATION_SCALE = 0.65 # 느린 출발 상황도 검사하기 위한 예측 가속도 배율
ROUNDABOUT_LAUNCH_DELAY = 0.4          # 느린 출발 예측에 반영할 반응 지연; 실제 강제 대기시간 아님 [s]
ROUNDABOUT_COMFORT_BRAKING = 2.5       # 이동거리 예측 및 정지 목표속도 계산에 사용할 감속도 [m/s^2]
ROUNDABOUT_EMERGENCY_TTC = 0.8         # 예측 충돌 시간이 이 값 이하면 비상 정지 판단 [s]
ROUNDABOUT_FOLLOW_TIME_GAP = 1.2       # 선행차 추종 시 자차 속도에 곱할 시간 간격 [s]
ROUNDABOUT_FOLLOW_MIN_GAP = 3.0        # 선행차 추종 시 기본으로 확보할 범퍼 간 거리 [m]
ROUNDABOUT_TRACK_DISTANCE = 60.0      # 자차 주변에서 예측 대상으로 추적할 차량 최대 거리 [m]
ROUNDABOUT_REAR_TRACK_DISTANCE = 80.0 # 회전 차로에서 합류점으로 뒤따라오는 차량의 추적 거리 [m]
ROUNDABOUT_ENTRY_REAR_FRONT_MARGIN = 4.0 # 진입 전 후방 회전 차량의 앞쪽에 추가할 충돌 예측 여유 [m]; 0이면 기존 판정
ROUNDABOUT_COMMITTED_DISTANCE = 35.0  # 진입 시작/완료 후 정지선 주변에서 정책을 유지할 거리 [m]
ROUNDABOUT_STOP_BUFFER = 0.8          # 현재 미사용: 이전 정지 여유 설정 [m]
ROUNDABOUT_DEBUG_ENABLED = True      # main.py 실행 시 회전교차로 진입 진단 JSONL 기록
ROUNDABOUT_DEBUG_INTERVAL = 0.2      # 일반 기록 간격 [s]; 판단/진입/차량 목록 변경 시 즉시 기록

# 시각화: MGeo 링크 A2256W000134를 따라 그리는 원호형 감시 영역
# 아래 중심선과 반폭은 RViz 표시용이며 실제 충돌 판단은 차량 궤적 예측으로 수행한다.
ROUNDABOUT_ENTRY_MONITOR_POLYLINE = (
    (-108.793, 326.851),
    (-108.328, 326.666),
    (-107.864, 326.481),
    (-107.391, 326.319),
    (-106.908, 326.190),
    (-106.424, 326.062),
    (-105.935, 325.958),
    (-105.444, 325.865),
    (-104.458729039151, 325.695102002511),
    (-103.962, 325.639),
    (-103.465, 325.583),
    (-102.967, 325.541),
    (-102.467, 325.524),
    (-101.967, 325.533),
    (-101.467, 325.542),
    (-100.969, 325.582),
    (-100.472, 325.642),
    (-99.9760589772946, 325.701930661996),
    (-99.482, 325.781),
    (-98.993, 325.883),
    (-98.505, 325.993),
    (-98.019, 326.109),
    (-97.535, 326.234),
    (-97.057, 326.383),
    (-96.580, 326.531),
    (-96.1097768079744, 326.701646399367),
    (-95.644, 326.884),
    (-95.181, 327.073),
)
ROUNDABOUT_ENTRY_MONITOR_HALF_WIDTH = 3.0 # 표시할 감시 영역 중심선의 좌우 반폭 [m]
