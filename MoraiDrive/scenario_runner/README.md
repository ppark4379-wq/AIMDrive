# K-City 2025 Scenario Runner

MORAI Simulator 26.R1과 `R_KR_PR_K-city_2025` 맵을 위한 gRPC 기반 시나리오 자동화 패키지입니다.

기존 `full_loop`는 정적·동적 장애물, 회전교차로·고속도로 NPC와 터널 급제동 이벤트를 모두 유지합니다. 장애물·NPC 없는 전체 주행은 `full_loop_normal`을 사용합니다.
차량 제어에는 AIMDrive의 `MoraiDrive/expert`를 사용하고, 원격 main의 Sync Runtime 연동도 모든 모드에서 유지됩니다.

## Sync Runtime 먼저 실행

ROS 환경을 불러온 별도 터미널에서 실행합니다. Runner가 READY를 설정할 때까지 대기합니다.

```bash
cd ~/AIMDrive/MoraiDrive
source /opt/ros/noetic/setup.bash
source expert/devel/setup.bash
python3 -m sync_runtime.main --mode scenario
```

Runner는 `setup()`과 Expert 프로세스 실행 확인 후 `/morai_drive/scenario_ready=true`를 설정합니다. `/morai_drive/sync_runtime_active=true`를 관측한 뒤 새 `/Ego_topic` 메시지까지 받아야 주행 감시를 시작합니다. 준비 대기는 기본 30초이며 `sync_runtime.startup_timeout_sec`로 조정합니다.

차량 상태는 gRPC를 우선 사용하고, 조회 실패나 빈 응답이면 최근 ROS 수신 상태를 사용합니다. 기본 1초(`sync_runtime.ego_max_age_sec`)보다 오래된 ROS 상태는 사용하지 않으며, 양쪽 모두 사용할 수 없으면 오류로 종료합니다.

종료 시 READY를 해제하고 ROS 구독을 해제한 뒤 Expert를 종료합니다. Runtime은 READY 해제를 확인하여 SyncMode를 종료합니다. 매 회차 Runtime을 먼저 다시 실행하고 Runner를 `--once`로 실행하는 것을 권장합니다.

시나리오 내부 전체 제한은 0으로 비활성화했고, 진행 watchdog도 비활성화했습니다. 경로 이탈 타임아웃은 1e12초로 사실상 비활성입니다. 데이터 수집 명령의 별도 wall-time 제한은 그대로 적용됩니다.

## 전체 폐루프 실행

원래 START에서 출발하여 정적 장애물 구간을 포함한 전체 폐루프를 주행합니다.

```bash
cd ~/AIMDrive/MoraiDrive
source /opt/ros/noetic/setup.bash
source expert/devel/setup.bash

python3 -m scenario_runner.main_runner \
  --zone full_loop \
  --scenario full_loop --once
```

## 구간별 모드 선택

위 실행 명령의 `--scenario` 값만 바꿉니다. 일반 사용 모드는 6개입니다.

| 모드 | 주행 범위 | 장애물·NPC |
|---|---|---|
| `full_loop` | 기존 전체 START → END | 기존 모든 이벤트 |
| `full_loop_normal` | 기존 전체 START → END | 러너 생성 장애물·NPC 없음 |
| `full_loop_static` | 인덱스 483 → 753 (132.4m) | 정적 장애물만 랜덤 생성 |
| `full_loop_dynamic` | 인덱스 753 → 1081 (160.0m) | 동적 장애물만 기존 방식으로 횡단 |
| `full_loop_roundabout` | 인덱스 1668 → 1975 (150.0m) | NPC 3대, 진입 100m 전 생성, 출구 30m 후 제거 |
| `full_loop_highway` | 인덱스 1975 → 3638 (825.0m) | SpawnPoint 3개, 각 4대, 총 12대, 5초 주기 |

예: 장애물·NPC 없는 전체 주행을 3회 실행:

```bash
python3 -m scenario_runner.main_runner --zone full_loop --scenario full_loop_normal --max-runs 3
```

예: 정적 장애물 구간만 반복:

```bash
python3 -m scenario_runner.main_runner --zone full_loop --scenario full_loop_static
```

여섯 모드는 정상 도착 후 객체 정리 → 같은 모드 시작점으로 재배치 → 다음 회차를 반복합니다.
날씨는 SUNNY/FOGGY 교대, 시간은 11/13/15시 중 선택하며 시간과 정적·동적 장애물 모델은 직전 회차와 다르게 재선택합니다.
새 모드는 각자 날씨 이력을 저장하므로 기존 full_loop 날씨 이력을 변경하지 않습니다.
`--once`는 1회, `--max-runs 10`은 10회입니다. 복구되지 않은 실행 오류는 전체 반복 명령을 중단합니다.

새 구간 모드는 해당 이벤트만 켜며 터널 이벤트는 모두 끕니다. 신호등은 유지하고 기존 expert가 신호에 대응합니다.
전체 normal 모드는 러너 소유 객체를 정리하고 새 객체를 생성하지 않습니다. 수동 배치/다른 client 소유 객체까지 삭제하지 않으므로 깨끗한 맵에서 사용하세요.
동시에 두 러너를 실행하지 마세요.

실행에 사용하는 구간 시작·끝 좌표는 [segment_modes/modes.py](scenarios/segment_modes/modes.py)에 고정되어 있으며 테스트에서 직접 검증합니다.

오프라인 확인: 원하는 명령에 `--check`를 붙이면 접속 없이 경계 좌표·길이·이벤트 ON/OFF를 출력합니다.
이 검사는 실제 AI 주행, 신호 준수, 무충돌 또는 MORAI RPC 성공을 보장하지 않습니다.

## 기존 정적 장애물 이후 명령 (호환 유지)

정적 장애물 다음 위치에서 출발하여 원래 END까지 주행합니다. 시나리오의 장애물과 NPC 구성은 전체 실행과 동일합니다.

```bash
cd ~/AIMDrive/MoraiDrive
source /opt/ros/noetic/setup.bash
source expert/devel/setup.bash

python3 -m scenario_runner.main_runner \
  --zone full_loop \
  --scenario full_loop_after_static --once
```

위 명령은 한 회차만 실행합니다. `--once`를 생략하면 기존 자동 반복 설정이 적용되지만, 다음 회차 전에 독립 Runtime도 다시 준비해야 합니다.

## MGeo 뷰어 실행

먼저 원하는 시나리오를 실행하여 현재 런타임 경로를 생성한 뒤, 새 터미널에서 실행합니다.

```bash
cd ~/AIMDrive/MoraiDrive
source /opt/ros/noetic/setup.bash
source expert/devel/setup.bash

export SCENARIO_MGEO_ROOT=$PWD/scenario_runner/data/R_KR_PR_K-city_2025
export SCENARIO_ROUTE_PATH_TXT=$PWD/scenario_runner/.runtime/expert/path_scenario_runner_current.txt

SDL_AUDIODRIVER=dummy python3 expert/src/expert/src/path/mgeo_visualization.py
```
