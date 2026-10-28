# 시나리오 N회 반복과 진단 리포트 저장

`run_highway.sh`, `run_roundabout.sh`, `run_static.sh`, `run_tunnel.sh`에서 같은 옵션을 씁니다. 회전교차로 4종
(`--variant normal|nonuniform|speed_random|brake`)도 같습니다.

```bash
bash AIM2026/Expert/run_highway.sh --runs 10 --seed 100
bash AIM2026/Expert/run_highway.sh --highway-density competition --runs 10 --seed 100   # 대회 교통
bash AIM2026/Expert/run_roundabout.sh --variant brake --runs 10
bash AIM2026/Expert/run_static.sh --runs 5 --static-obstacle-count 2
bash AIM2026/Expert/run_tunnel.sh --variant pedestrian --runs 10 --seed 100
bash AIM2026/Expert/run_tunnel.sh --variant all                    # brake, static, pedestrian, vehicle 한 번씩
```

| 옵션 | 설명 |
|---|---|
| `--runs N` | 시나리오를 N번 실행합니다. 매번 MORAI 세션을 새로 시작하고 끝에 정리합니다. 기본 1 |
| `--seed S` | 첫 실행의 seed입니다. i번째 실행은 `S+i-1`을 씁니다. 안 주면 실행마다 시나리오가 정합니다 |
| `--same-seed` | 모든 실행에 같은 seed `S`를 씁니다(같은 조건의 편차를 볼 때) |
| `--stop-on-failure` | COMPLETE가 아닌 실행이 나오면 거기서 멈춥니다. 기본은 실패해도 계속합니다 |
| `--pause-s T` | 실행 사이에 쉬는 시간(초). 기본 10 |
| `--run-timeout-s T` | 한 실행이 T초를 넘으면 중단하고(정리·리포트 생성 후) 다음 실행으로 넘어갑니다. 기본 1200, 0이면 제한 없음 |

- 반복할 때는 `--validation-events`가 자동으로 켜집니다. 실행마다 Failure Diagnosis 리포트가 만들어집니다.
- Ctrl+C를 누르면 지금까지의 결과를 저장하고 멈춥니다.
- 고속도로 미션에는 진행 감시(watchdog)가 꺼져 있어서, 차가 멈추면 실행이 끝나지 않습니다. 반복 실행에서는
  `--run-timeout-s`가 그 실행을 끊어 줍니다(시간 제한은 Linux/WSL에서만 동작합니다).
- 터널(`run_tunnel.sh`)의 `--variant`는 normal(기본)·brake·static·pedestrian·vehicle·all입니다. 장면 로직은 러너의
  `run_tunnel_aim2026.py`를 그대로 쓰고, 배치 폴더 이름은 `tunnel_<variant>-<시각>-x<N>`입니다.
  `--variant static`은 러너의 승인 후보 파일(`.runtime/tunnel_static_trials/.../approved_candidates.json`)이 있어야 합니다.
- 모든 `run_*.sh`는 `AIM2026/utils/scenario_runner`(시나리오 러너 이식본)의 `project_env.sh`와 환경 구성을 씁니다.
  진단 기록(`validation_runs`, `incidents`)은 `AIM2026/utils/scenario_runner/.runtime`에 남습니다.
- `--check`는 한 번만 검증합니다(`--runs`는 무시).
- 같은 seed는 시나리오 조건만 같게 합니다. 조향 명령이나 주행 궤적까지 같다는 보장은 없습니다.

## 결과 위치

```
AIM2026/outputs/batches/<시나리오>-<시각>-x<N>/
  index.html           실행별 한 줄: seed, 결과, 진단 요약(미검증 후보 포함), 리포트 링크
  batch_summary.json   같은 내용(스크립트용)
  reports/             실행별 failure_report/report.html 사본 (run_01_COMPLETE_seed100.html ...)
```

각 실행의 전체 로그와 원본 리포트는 평소처럼 `AIM2026/outputs/scenarios/<run>/`와
`AIM2026/outputs/expert/<run>/`에 남습니다. `index.html`이 그 폴더의 위치도 보여줍니다.
`index.html`은 실행이 하나 끝날 때마다 갱신되므로 도중에도 볼 수 있습니다.

종료 코드는 모든 실행이 COMPLETE일 때만 0입니다.
