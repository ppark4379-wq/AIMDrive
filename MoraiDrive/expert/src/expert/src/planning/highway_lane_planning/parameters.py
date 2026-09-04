LANE_WIDTH = 3.5                       # 차선 번호 차이 1개를 표시할 때 사용하는 대표 차선 폭 [m]
LANE_MATCH_MAX_DISTANCE = 2.4          # 차량 중심을 링크 중앙선에 매칭할 최대 횡거리 [m]
LANE_CHANGE_DISTANCE = 55.0            # 고속 차선변경을 완만하게 완료할 종방향 거리 [m]
PATH_LOOKAHEAD_MIN = 65.0              # 제어·예측에 제공할 최소 링크 경로 길이 [m]
PATH_SAMPLE_SPACING = 0.5              # 링크 기반 후보 경로 샘플 간격 [m]
PREDICTION_HORIZON = 0.25               # 감속 후 회피 가능성을 과도하게 막지 않을 미래 OBB 예측 시간 [s]
PREDICTION_TIME_STEP = 0.1             # 미래 OBB SAT 검사 고정 시간 간격 [s]
PREDICTION_MIN_SPEED = 2.0             # 정지 시에도 후보 경로를 검사하기 위한 최소 가상 속도 [m/s]
COLLISION_MARGIN = 0.25                # 미래 OBB에 추가하는 안전 여유 [m]
MANEUVER_COMPLETE_DISTANCE = 0.45      # 목표 링크 중앙 도착 판정 횡거리 [m]
MANEUVER_TARGET_MAX_DISTANCE = 4.0     # 차선변경 중 목표 링크를 계속 유지할 최대 횡거리 [m]
FRONT_RELEVANCE_DISTANCE = 55.0        # 현재 차선 선행차를 위험 판단에 포함할 거리 [m]
ACC_FALLBACK_LATERAL_THRESHOLD = 2.2   # 링크 매칭 실패 시 로컬 좌표 선행차 판정 횡거리 [m]
LANE_CHANGE_TRIGGER_DISTANCE = 45.0    # 이 앞 범퍼 거리부터 안전한 옆 차선 회피를 미리 검토 [m]
LANE_CHANGE_TRIGGER_TTC = 6.0          # 이 TTC부터 안전한 옆 차선 회피를 미리 검토 [s]
ACC_TIME_GAP = 1.5                     # 강제 ACC가 유지할 시간 간격 [s]
ACC_STANDSTILL_DISTANCE = 5.0          # 강제 ACC가 정지 시 유지할 기본 앞 범퍼 거리 [m]
ACC_DISTANCE_GAIN = 0.55               # 안전거리 오차를 목표속도 감소량으로 바꾸는 비례 계수
ACC_PREPARE_TARGET_SPEED = 13.0        # 회피 준비 ACC 단계의 목표속도(약 47 km/h) [m/s]
ACC_LANE_CHANGE_MAX_SPEED = 15.0       # 이 속도 이하로 감속한 뒤 차선변경 후보를 허용 [m/s]
ACC_LANE_CHANGE_MIN_DISTANCE = 8.0     # 감속 후 차선변경을 검토할 최소 앞 범퍼 간격 [m]
ACC_LANE_CHANGE_MIN_TTC = 3.0          # 감속 후 차선변경을 검토할 최소 TTC [s]
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
MERGE_TRAJECTORY_HORIZON = 3.0         # 링크 매칭 실패 차량의 합류 교차궤적 ACC 예측시간 [s]
MERGE_TRAJECTORY_TIME_STEP = 0.25      # 합류 교차궤적 SAT 검사 시간 간격 [s]
MERGE_TRAJECTORY_MAX_DISTANCE = 70.0   # 교차궤적 검사에 포함할 주변 차량 최대 중심거리 [m]
MERGE_ACC_RECOVERY_RATE = 2.0          # 합류 위험 해제 후 속도상한을 복원하는 변화율 [m/s^2]
CONSERVATIVE_MERGE_POINT = (75.25339642604672, -219.84857834122045)  # 안전 우선 합류 중심점 (x, y) [m]
CONSERVATIVE_MERGE_SUCCESSORS = ("A2256W000153",)  # 이 링크로 합쳐지는 진입 차선에서 무조건 양보
CONSERVATIVE_MERGE_EGO_RADIUS = 65.0   # 중심점부터 Ego 안전 양보 로직을 적용하는 반경 [m]
CONSERVATIVE_MERGE_OBJECT_RADIUS = 55.0 # 중심점 주변에서 합류 차량으로 간주할 반경 [m]
CONSERVATIVE_MERGE_PASS_DISTANCE = 15.0 # 상대차가 합류 링크로 진입 후 이 거리만큼 지나야 양보 해제 [m]
CONSERVATIVE_MERGE_MAX_SPEED = 8.0     # 상대차를 먼저 보낼 때 허용하는 최대 목표속도 (약 29 km/h) [m/s]
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

# 레인 섹션 경계의 진행 연결. 리스트가 2개인 곳은 실제 분기입니다.
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
