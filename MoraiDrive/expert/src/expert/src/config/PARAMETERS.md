# GT-BEV Expert 파라미터 안내

`main.py`를 단독 실행하면 같은 폴더의 `config.json`을 읽는다. JSON 문법은
인라인 주석을 지원하지 않으므로 각 설정의 의미를 이 문서에 정리한다.

## `planning.velocity_profile`

- `max_velocity`: 기본 최고 목표속도 `[km/h]`.
- `road_friction`: 곡선구간 속도 프로파일 계산에 사용하는 노면 마찰계수. 낮을수록 곡선에서 느려진다.
- `window_size`: 곡률을 계산할 때 앞뒤로 참고하는 경로 포인트 개수.
- `speed_limit_deceleration`: 낮은 제한속도 구간 진입 전 적용할 최대 감속도 `[m/s²]`.
- `speed_limit_zones`: 특정 링크 구간에 적용할 별도 최고속도 목록.
- `terminal_deceleration`: 열린 경로의 마지막 지점에 정지하기 위한 감속도 `[m/s²]`.

## `planning.adaptive_cruise_control`

- `velocity_gain`: 선행 객체와의 상대속도 오차에 대한 ACC 반응 가중치.
- `distance_gain`: 안전거리 오차에 대한 ACC 반응 가중치.
- `time_gap`: Ego 속도에 비례하여 확보할 시간 간격 `[s]`.
- `pedestrian_lateral_threshold`: 보행자를 경로상 객체로 판단할 최대 횡거리 `[m]`.
- `vehicle_lateral_threshold`: 차량·정적장애물을 경로상 객체로 판단할 최대 횡거리 `[m]`.
- `traffic_light_lateral_threshold`: 신호등 정지 대상을 판단할 최대 횡거리 `[m]`.
- `pedestrian_longitudinal_max`: 보행자 ACC 판단에 사용할 최대 전방거리 `[m]`.

## `control.pid`

- `p_gain`: 목표속도와 현재속도 오차에 대한 비례 제어 게인.
- `i_gain`: 누적 속도 오차에 대한 적분 제어 게인.
- `d_gain`: 속도 오차 변화량에 대한 미분 제어 게인.

## `control.pure_pursuit`

- `lfd_gain`: 속도에 비례해 Pure Pursuit look-ahead 거리를 늘리는 게인.
- `min_lfd`: 최소 look-ahead 거리 `[m]`.
- `max_lfd`: 최대 look-ahead 거리 `[m]`.

레티스 관련 값은 `../planning/lattice_planning/parameters.py`에서 조정한다.
