# K-City 2025 Scenario Runner

MORAI Simulator 26.R1과 `R_KR_PR_K-city_2025` 맵을 위한 gRPC 기반 시나리오 자동화 패키지입니다.

정적·동적 장애물, 회전교차로 및 고속도로 NPC, 터널 급정거 이벤트, 날씨·시간 설정과 시나리오 자동 반복을 포함합니다. 차량 제어에는 AIMDrive의 `MoraiDrive/expert`를 사용합니다.

## 전체 폐루프 실행

원래 START에서 출발하여 정적 장애물 구간을 포함한 전체 폐루프를 주행합니다.

```bash
cd ~/AIMDrive/MoraiDrive
source /opt/ros/noetic/setup.bash
source expert/devel/setup.bash

python3 -m scenario_runner.main_runner \
  --zone full_loop \
  --scenario full_loop
```

## 정적 장애물 이후부터 실행

정적 장애물 다음 위치에서 출발하여 원래 END까지 주행합니다. 시나리오의 장애물과 NPC 구성은 전체 실행과 동일합니다.

```bash
cd ~/AIMDrive/MoraiDrive
source /opt/ros/noetic/setup.bash
source expert/devel/setup.bash

python3 -m scenario_runner.main_runner \
  --zone full_loop \
  --scenario full_loop_after_static
```

두 실행 명령은 END 도착 후 다음 시나리오를 자동으로 시작합니다. 한 회차만 실행하려면 마지막 줄에 `--once`를 추가합니다.

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
