import math as _math

LANE_WIDTH = 3.5                       # 차선 번호 차이 1개를 표시할 때 사용하는 대표 차선 폭 [m]
LANE_MATCH_MAX_DISTANCE = 2.4          # 차량 중심을 링크 중앙선에 매칭할 최대 횡거리 [m]
LANE_CHANGE_DISTANCE = 55.0            # 고속 차선변경을 완만하게 완료할 종방향 거리 [m]
PATH_LOOKAHEAD_MIN = 65.0              # 제어·예측에 제공할 최소 링크 경로 길이 [m]
PATH_SAMPLE_SPACING = 0.5              # 링크 기반 후보 경로 샘플 간격 [m]
PREDICTION_HORIZON = 0.4               # 감속 후 회피 가능성을 과도하게 막지 않을 미래 OBB 예측 시간 [s]
PREDICTION_TIME_STEP = 0.1             # 미래 OBB SAT 검사 고정 시간 간격 [s]
PREDICTION_MIN_SPEED = 2.0             # 정지 시에도 후보 경로를 검사하기 위한 최소 가상 속도 [m/s]
COLLISION_MARGIN = 0.25                # 미래 OBB에 추가하는 안전 여유 [m]
MANEUVER_COMPLETE_DISTANCE = 0.45      # 목표 링크 중앙 도착 판정 횡거리 [m]
MANEUVER_TARGET_MAX_DISTANCE = 4.0     # 차선변경 중 목표 링크를 계속 유지할 최대 횡거리 [m]
FRONT_RELEVANCE_DISTANCE = 55.0        # 현재 차선 선행차를 위험 판단에 포함할 거리 [m]
ACC_FALLBACK_LATERAL_THRESHOLD = 2.2   # 링크 매칭 실패 시 로컬 좌표 선행차 판정 횡거리 [m]
LANE_CHANGE_TRIGGER_DISTANCE = 45.0    # 이 앞 범퍼 거리부터 안전한 옆 차선 회피를 미리 검토 [m]
LANE_CHANGE_TRIGGER_TTC = 6.0          # 이 TTC부터 안전한 옆 차선 회피를 미리 검토 [s]
ACC_TIME_GAP = 0.25 #원래1.5              # 강제 ACC가 유지할 시간 간격 [s]
ACC_STANDSTILL_DISTANCE = 5.0          # 강제 ACC가 정지 시 유지할 기본 앞 범퍼 거리 [m]
ACC_DISTANCE_GAIN = 0.55               # 안전거리 오차를 목표속도 감소량으로 바꾸는 비례 계수
ACC_LANE_CHANGE_MIN_DISTANCE = 8.0     # 차선변경을 검토할 최소 앞 범퍼 간격 [m]
ACC_LANE_CHANGE_MIN_TTC = 3.0          # 차선변경을 검토할 최소 TTC [s]
ACC_STOPPED_SPEED = 1.0                # 이 속도 이하면 간격 대신 정지 상태로 회피 검토 [m/s]
ACC_UNMATCHED_HAZARD_SPEED = 10.0      # 동일차로 매칭 없이 OBB 충돌만 잡힐 때 속도 상한 [m/s]
TARGET_LANE_FRONT_CLEARANCE = 20.0     # 차선변경 전 목표차선에서 확보할 최소 전방 거리 [m]
TARGET_LANE_REAR_CLEARANCE = 20.0      # 차선변경 전 목표차선에서 확보할 최소 후방 거리 [m]
TARGET_LANE_FRONT_TIME_GAP = 1.5       # 목표차선 앞차가 느릴수록 추가 확보할 상대속도 시간 간격 [s]
TARGET_LANE_REAR_TIME_GAP = 2.5        # 목표차선 뒤차가 빠를수록 추가 확보할 상대속도 시간 간격 [s]
MERGE_LOOKAHEAD_DISTANCE = 60.0        # 동일 합류점으로 접근하는 차량을 검사할 최대 거리 [m]
MERGE_MIN_REMAINING_DISTANCE = 3.0     # 합류점 직전 좌표 투영 오차로 제한속도가 0이 되는 것을 막는 거리 [m]
MERGE_ARRIVAL_TIME_WINDOW = 3.0        # 합류점 도착시간 차가 이 값 이하면 충돌 가능으로 판단 [s]
MERGE_EGO_PRIORITY_MARGIN = 0.1        # ETA 수치오차를 제외하고 Ego가 먼저일 때 MERGE_GO 확정 [s]
MERGE_YIELD_TIME_GAP = 1.5             # 합류 차량을 먼저 보낸 뒤 확보할 도착시간 여유 [s]
MERGE_TRAJECTORY_HORIZON = 3.0         # 미매칭 차량의 단기 OBB 비상검사 시간 [s]
MERGE_PRIORITY_HORIZON = 8.0           # 충분한 정지거리 전에 합류 우선권을 확정할 시간 [s]
MERGE_TRAJECTORY_TIME_STEP = 0.25      # 합류 교차궤적 SAT 검사 시간 간격 [s]
MERGE_TRAJECTORY_MAX_DISTANCE = 70.0   # 교차궤적 검사에 포함할 주변 차량 최대 중심거리 [m]
MERGE_SAFETY_HORIZON = 5.0
MERGE_SAFETY_STEP = 0.1
MERGE_CONTROL_DELAY = 0.25
MERGE_PREDICT_ACCEL = 1.5
MERGE_PREDICT_DECEL = 3.0
MERGE_SPEED_SEARCH_STEP = 0.5
MERGE_BUMPER_CLEARANCE = 3.0
MERGE_TRACK_HOLD_TIME = 0.5
MERGE_GO_CLEAR_TIME_MARGIN = 1.5       # Ego 완전 이탈 후 상대 진입까지 필요한 최소 시간 [s]
MERGE_CLOSE_GAP_YIELD_DISTANCE = 10.0  # 예상 범퍼 간격이 이 값 이하면 뒤로 양보 [m]
MERGE_MIN_YIELD_SPEED_KPH = 30.0       # 움직이는 상대차 양보 시 최저 주행속도 [km/h]
MERGE_MIN_YIELD_SPEED = MERGE_MIN_YIELD_SPEED_KPH / 3.6  # 제어용 [m/s]
MERGE_ACC_RECOVERY_RATE = 2.0          # 합류 위험 해제 후 속도상한을 복원하는 변화율 [m/s^2]
# 두 번째 합류 안전 진입: 아래 좌표를 지난 뒤부터 목표속도를 35~45km/h 밴드로
# 제한한다. 첫 번째 합류와 같은 방식(좌표 + 원)이며, 원의 앞쪽 경계가 이
# 좌표에 오도록 중심을 합류점 방향으로 반경만큼 밀어 둔다. 그래서 이 좌표
# '이전'에는 감속하지 않고, 원의 뒤쪽 경계는 합류점을 지나 여유까지 덮는다.
SECOND_MERGE_SLOWDOWN_POINT = (71.22, -121.62)
SECOND_MERGE_SLOWDOWN_POINT_Z = 29.87
SECOND_MERGE_SLOWDOWN_EXIT_MARGIN = 5.0 # 원이 합류점을 지나 더 덮을 거리 [m]
SECOND_MERGE_SLOWDOWN_MIN_SPEED_KPH = 35.0    # 밴드 하한 [km/h]
SECOND_MERGE_SLOWDOWN_MAX_SPEED_KPH = 45.0    # 밴드 상한 [km/h]
SECOND_MERGE_SLOWDOWN_MIN_SPEED = SECOND_MERGE_SLOWDOWN_MIN_SPEED_KPH / 3.6
SECOND_MERGE_SLOWDOWN_MAX_SPEED = SECOND_MERGE_SLOWDOWN_MAX_SPEED_KPH / 3.6

CONSERVATIVE_MERGE_POINT = (74.98, -178.99) # 사용자가 지정한 합류 진입구역 중심 (x, y) [m]
CONSERVATIVE_MERGE_POINT_Z = 28.75     # MORAI 월드 좌표의 합류 진입구역 높이 [m]
CONSERVATIVE_MERGE_ENTRY_RADIUS = 10.0 # 실제 두 차로가 겹치는 충돌 코어 반경 [m]
CONSERVATIVE_MERGE_SUCCESSORS = ("A2256W000153",)  # 두 진입 차선이 합쳐지는 공통 후속 링크
CONSERVATIVE_MERGE_EGO_RADIUS = 130.0  # 고속에서도 정지거리 전에 우선권을 판단할 반경 [m]
CONSERVATIVE_MERGE_OBJECT_RADIUS = CONSERVATIVE_MERGE_ENTRY_RADIUS
CONSERVATIVE_MERGE_PASS_DISTANCE = 15.0 # 상대차가 합류 링크로 진입 후 이 거리만큼 지나야 양보 해제 [m]
CONSERVATIVE_MERGE_MAX_SPEED = 14.0     # 양보 시 상한; 실제 값은 상대차 통과시간으로 계산 [m/s]
CONSERVATIVE_MERGE_CRAWL_SPEED = 3.0   # 일반 양보 시 완전정지 대신 유지할 통과속도 (약 11 km/h) [m/s]
CONSERVATIVE_MERGE_STOP_MARGIN = 2.0   # 양보 시 진입구역 경계 전에 정지할 추가 여유 [m]
CONSERVATIVE_MERGE_COMFORT_DECEL = 3.0 # 진입구역 정지 목표속도 계산에 사용할 감속도 [m/s^2]
CONSERVATIVE_MERGE_CLEAR_CYCLES = 30   # 상대차 소실 시 양보 해제 전 확인할 주기 (30 Hz에서 1초)
LANE_CHANGE_COOLDOWN_DISTANCE = 50.0  # 변경 완료 후 새 차선변경을 막는 거리 [m]
SECOND_MERGE_PREFERRED_LANE_NUMBERS = (2, 3)  # 주행 가능 3개 중 왼쪽·가운데
SECOND_MERGE_RIGHT_LANE_COST_PENALTY = 10.0    # 맨 오른쪽 선택 비용(금지는 아님)
SECOND_MERGE_LANE_PREFERENCE_RADIUS = 160.0  # 막히기 전부터 좌/중앙 차선 선호 [m]
MERGE_EMERGENCY_COLLISION_TIME = 1.0   # Ego가 양보해야 할 때 이 시간 안의 OBB 충돌은 즉시 정지 [s]
TARGET_SPEED_DROP_RESET_THRESHOLD = 2.0 # 목표속도 급강하 시 PID 적분을 초기화할 차이 [m/s]
ACCEL_COMMAND_RISE_RATE = 3.0          # 가속 페달 명령 상승률 [command/s], 재가속 충격 억제
BRAKE_RELEASE_RATE = 6.0               # 위험 해제 후 제동 명령 해제율 [command/s]
PLANNER_UPDATE_RATE = 30.0             # 합류 ACC 속도상한 필터의 예상 실행 주기 [Hz]
HIGHWAY_DEBUG_ENABLED = True           # 터미널에 고속도로 판단 근거와 후보 비용을 상세 출력
HIGHWAY_DEBUG_INTERVAL_CYCLES = 15     # 동일 판단 유지 중 상세 로그를 출력할 주기 [cycle]
LANE_CHANGE_STEERING_LIMIT_DEG = 5.0   # 차선변경 중 허용할 최대 앞바퀴 조향각 [deg]
STEERING_RATE_LIMIT_DEG_PER_SEC = 30.0 # 제어 주기당 조향각 급변을 막는 최대 변화율 [deg/s]
VEHICLE_LENGTH = 4.635                 # Ego 미래 OBB 길이 [m]
VEHICLE_WIDTH = 2.0                    # Ego 미래 OBB 폭 [m]

# 제공된 고속도로 좌표가 통과하는 레인 섹션. ego_lane=1은 모든 다차선 섹션에서 제외합니다.
SECTION_ROADS = (
    (1, "4000104"),
    (2, "4000142"),
    (3, "4000107"),
    (4, "4000112"),
    (5, "4000111"),
)
ENTRY_LINK_ID = "A2256W000846"         # 다차선 진입 전 단일 링크; 왼쪽차선 제외 정책을 적용하지 않음
PROHIBITED_LINK_IDS = ("A2256W000179",)  # (65.16, -28.69)부터 시작하는 실선 우측 경계 차선

# 첫 번째 합류 지점: 두 차량의 원 진입 예상시간을 비교해 선진입 차량을 우선합니다.
PREDICTIVE_MERGE_POINT = (64.35, 132.39)
PREDICTIVE_MERGE_POINT_Z = 29.86
PREDICTIVE_MERGE_BRAKE_START_POINT = (63.06, 179.63)
PREDICTIVE_MERGE_BRAKE_START_POINT_Z = 29.51
PREDICTIVE_MERGE_ENTRY_RADIUS = 8.0  # 첫 번째 합류의 실제 충돌 지점만 포함하는 작은 원 [m]
PREDICTIVE_MERGE_EGO_RADIUS = 130.0   # 고속에서도 정지거리 전에 우선권을 판단할 반경 [m]
PREDICTIVE_MERGE_SUCCESSORS = ("A2256W000420",)
PREDICTIVE_MERGE_EGO_CURVATURE_SPEED_FLOOR = True
PREDICTIVE_MERGE_OBJECT_LINK_IDS = ("A2256W000409",)  # ego(A2256W000411) 옆 실제 합류 상대차선
PREDICTIVE_MERGE_CLEAR_CYCLES = 3       # 예측 진입 소실 후 약 0.1초 확인하고 양보 해제
PREDICTIVE_MERGE_PRIORITY_TIME_MARGIN = -0.2 # 상대가 0.2초 이상 확실히 빠를 때만 OTHER [s]
MERGE_EGO_TAKEOVER_TIME_MARGIN = 0.2       # OTHER 잠금 후 Ego ETA가 이만큼 앞서면 우선권 재판정 [s]
PREDICTIVE_MERGE_MAX_SPEED = 16.0       # 첫 번째 합류 양보 시작 속도 상한 (약 58 km/h) [m/s]
PREDICTIVE_MERGE_CRAWL_SPEED = 8.0      # 움직이는 선행차 뒤로 합류할 때 유지할 최저 롤링 속도 [m/s]
PREDICTIVE_MERGE_STOP_MARGIN = 1.0      # 비상 정지가 필요할 때 사용할 원 경계 안전 여유 [m]
PREDICTIVE_MERGE_COMFORT_DECEL = 4.0   # 롤링 양보 목표속도의 최대 감속 프로파일 [m/s^2]
PREDICTIVE_MERGE_PASS_DISTANCE = 3.0  # 상대차가 공통 링크에서 이 거리까지 지나야 겹침 해제 [m]
PREDICTIVE_MERGE_FOLLOW_TIME_GAP = 0.8 # 상대차 뒤로 원에 진입할 목표 시간 간격 [s]
PREDICTIVE_MERGE_MOVING_SPEED = 1.0    # 이 속도 이상의 상대차만 정지 없이 롤링 추종 [m/s]

# 레인 섹션 경계의 진행 연결. 리스트가 2개인 곳은 실제 분기입니다.
CONSERVATIVE_MERGE_EGO_CURVATURE_SPEED_FLOOR = True
PREDICTIVE_MERGE_DROPOUT_MIN_HOLD_TIME = 0.5
PREDICTIVE_MERGE_DROPOUT_MAX_HOLD_TIME = 5.0
CONSERVATIVE_MERGE_PRIORITY_TIME_MARGIN = -0.2 # 두 번째 합류도 상대가 0.2초 이상 빠를 때만 OTHER [s]
CONSERVATIVE_MERGE_FOLLOW_TIME_GAP = 0.8
CONSERVATIVE_MERGE_MOVING_SPEED = 1.0

SUCCESSOR_LINKS = {
    "A2256W000846": ("A2256W000411",),
    "A2256W000429": ("A2256W000433",),
    "A2256W000418": ("A2256W000431",),
    "A2256W000410": ("A2256W000430",),
    "A2256W000409": ("A2256W000420",),
    "A2256W000411": ("A2256W000420",),
    "A2256W000433": ("A2256W000421",),
    "A2256W000431": ("A2256W000435",),
    "A2256W000430": ("A2256W000434",),
    "A2256W000420": ("A2256W000408", "A2256W000179"),
    "A2256W000421": ("A2256W000425",),
    "A2256W000435": ("A2256W000423",),
    "A2256W000434": ("A2256W000422",),
    "A2256W000408": ("A2256W000445",),
    "A2256W000179": (),                # 우측 분기 차선은 다음 주행 섹션으로 이어지지 않음
    "A2256W000425": ("A2256W000424",),
    "A2256W000423": ("A2256W000432",),
    "A2256W000422": ("A2256W000153",),
    "A2256W000445": ("A2256W000153",),
}

# 두 번째 합류 감속 원: 앞쪽 경계 = SECOND_MERGE_SLOWDOWN_POINT,
# 뒤쪽 경계 = 합류점을 SECOND_MERGE_SLOWDOWN_EXIT_MARGIN 만큼 지난 지점.
# 좌표를 옮기면 중심/반경이 함께 따라가도록 계산해 둔다.
_SECOND_MERGE_SLOWDOWN_DX = (
    CONSERVATIVE_MERGE_POINT[0] - SECOND_MERGE_SLOWDOWN_POINT[0]
)
_SECOND_MERGE_SLOWDOWN_DY = (
    CONSERVATIVE_MERGE_POINT[1] - SECOND_MERGE_SLOWDOWN_POINT[1]
)
_SECOND_MERGE_SLOWDOWN_SPAN = _math.hypot(
    _SECOND_MERGE_SLOWDOWN_DX, _SECOND_MERGE_SLOWDOWN_DY
)
SECOND_MERGE_SLOWDOWN_RADIUS = (
    0.5 * (_SECOND_MERGE_SLOWDOWN_SPAN + SECOND_MERGE_SLOWDOWN_EXIT_MARGIN)
)
SECOND_MERGE_SLOWDOWN_CENTER = (
    SECOND_MERGE_SLOWDOWN_POINT[0]
    + _SECOND_MERGE_SLOWDOWN_DX
    / _SECOND_MERGE_SLOWDOWN_SPAN
    * SECOND_MERGE_SLOWDOWN_RADIUS,
    SECOND_MERGE_SLOWDOWN_POINT[1]
    + _SECOND_MERGE_SLOWDOWN_DY
    / _SECOND_MERGE_SLOWDOWN_SPAN
    * SECOND_MERGE_SLOWDOWN_RADIUS,
)
