# 고속도로 계획

`planner.py`의 `HighwayLanePlanner`가 전체 상태와 `plan()`의 분기 순서를 소유한다.
구간별 처리 순서를 보존하기 위해 `plan()`, 생성자, 초기화 메서드는 그대로 두었다.

| 메서드 파일 | 책임 |
|---|---|
| `merge_speed.py` | 합류 속도 제한, 우선권, 목표 속도 안정화 |
| `predictive_merge.py` | 예측 합류 진입, 도착 시간, 차체 영역 계산 |
| `lane_change.py` | 차선 변경 시작, 후보 여유, 복귀와 진행 결과 |
| `lane_following.py` | 기준 차로 매칭, 전방 위험과 ACC 제한 |
| `trajectory.py` | 후보 경로 생성, 다항식과 경로 기하 |
| `collision_prediction.py` | 객체 예측 경로, 시각, 차체 충돌 |

이 여섯 파일은 별도 실행 객체가 아니라 같은 `HighwayLanePlanner` 인스턴스의
메서드를 묶은 mixin이다. 생성자나 별도 상태가 없으며 기존 45개 메서드의 본문과
인자, `staticmethod`를 유지했다. 기존 `MergeSafety`, `LaneChangeSafety`,
`ObstacleMotion`, `MultiLanePlanning`, `HighwayStaticCorridor` 호출도 그대로다.

튜닝은 `src/config/parameters/highway.py`의 구간별 주석을 따른다.
이 폴더의 `parameters.py`는 같은 설정 모듈로 연결하는 호환 import다.
`config/parameters/sections.py`에는 고속도로 활성 영역과 종료 우회전 영역이 있다.
