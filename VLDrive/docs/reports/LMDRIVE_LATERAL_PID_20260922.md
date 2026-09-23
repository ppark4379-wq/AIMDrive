# LMDrive 횡방향 PID 적용 및 기록 재생 검증

작성: 2026-09-22. 범위: VLDrive 실시간 추론의 횡방향만 교체. 모델·navigation·expert 종방향 코드와 파라미터는 변경하지 않음.

## 1. 참조와 구현

- [LMDrive agent](https://github.com/opendilab/LMDrive/blob/main/leaderboard/team_code/lmdriver_agent.py): `control_pid` 539–559행의 첫 두 waypoint 중간점·방향 오차·정차 입력·조향 clipping, `run_step` 499–500행의 최종 배율 0.8.
- [설정](https://github.com/opendilab/LMDrive/blob/main/leaderboard/team_code/lmdriver_config.py): turn_KP=1.25, turn_KI=.75, turn_KD=.3, turn_n=40.
- [PID](https://github.com/opendilab/LMDrive/blob/main/leaderboard/team_code/pid_controller.py): 창 평균 I, 연속 차분 D. 기존 vendored 소스 그대로 사용. 출처 commit `308d8da0b4ff3463b1649fe483a9b2c56b108189`, Apache-2.0 LICENSE 보존.
- 원본 agent는 20Hz CARLA tick에서 홀수 tick의 이전 제어를 반환하므로 실제 갱신은 약 10Hz. MORAI에서는 조향만 명목 10Hz로 분리하고 expert 종방향 및 UDP 30Hz는 유지.

새 파일 `control/lmdrive_lateral.py`에서 횡방향만 수행한다. `expert_controller.py`는 기존 expert 호출의 가속·브레이크를 사용하고 조향 반환값만 교체한다. 공통 expert 함수의 Pure Pursuit 중간 계산은 그대로 수행되지만 최종 조향에 반영되지 않는다. 이렇게 원본 종방향 함수와 호출 조건을 보존했다.

```text
aim = (wp0 + wp1)/2             # 현재 차량 좌표 [left, forward]
e = degrees(pi/2 - atan2(aim.forward, aim.left))/90
실제 속도 < .01 m/s: e = 0
u = 1.25*e + .75*mean(e의 최근 40개) + .3*(e - 이전 e)
좌측 양수 조향 = clip(u, -1, 1)*.8
```

현재 UDP 부호 변환은 좌측 양수. 요청 휠각 표시는 상태 패킷 부호에 맞춰 `-steer*40°`. ±0.8은 요청 휠각 기준 ±32°. 이는 원본 배율이며 새 임의 제한값이 아니다. 차량 응답이 정확히 요청각을 따른다는 보장은 아니며 상태 패킷으로 비교한다.

창은 0으로 시작하고, I에 dt를 곱하거나 D를 dt로 나누지 않는다. 정차는 오차 입력만 0으로 만들며 원본처럼 잔여 I/D를 유지한다. 원본의 뒤쪽 목표점 처리도 그대로여서 앞쪽 점 재선택은 하지 않는다.

## 2. MORAI 연결

- 기존 좌표 변환과 이동 보정 유지. 원점 삽입이나 시간 경과에 따른 waypoint 삭제 없이 원본 다섯 점을 사용.
- `runtime.py`: 명목 10Hz 조향, 사이 출력 유지. 종방향은 모든 30Hz 제어 tick에서 기존 고정 dt=1/30초로 계산.
- 조향 스케줄은 작은 jitter에 대해 위상을 유지하고, 지연 시 가짜 catch-up 샘플을 삽입하지 않음.
- 이전 조향의 예측 만료 전에 최신 입력으로 갱신. 새 예측 frame을 받았다는 이유만으로 이전 조향의 유효시간을 연장하지 않음.
- 입력 무효 시 기존 제동 및 PID reset. 센서·GPS·status·bridge ACK 유효성 검사를 유지.
- 유지 중인 목표점을 월드 좌표로 보관해 현재 차량 좌표로 표시. 조향/종방향 frame을 구분.
- `configs/vldrive_control.json`의 `lateral`이 횡방향 설정. 종방향 설정은 계속 expert JSON 단일 원본에서 읽음.
- 원본 expert 프로그램, 네트워크 checkpoint, navigation 설정, UDP 포트는 변경하지 않음.

## 3. 자동 검증

```text
tests/test_lmdrive_lateral.py
tests/test_expert_control.py
tests/test_waypoint_control.py
tests/test_live_navigation_dashboard.py
91 passed
```

검증 내용:

- 독립적으로 전사한 LMDrive 수식과 160회 연속 입력 비교. 창 길이를 넘는 이력, 정차, 양방향 조향, 포화 포함.
- 90회 30Hz tick에서 조향 30회 갱신. 그 사이에도 expert 가속이 계속 갱신됨.
- jitter·긴 실행 공백에서 동일 관측의 중복 적분 없음.
- 변경 전 expert와 80회 시퀀스에서 가속·브레이크 정확히 동일. 정지 및 목표 속도 하락 포함.
- 5개 waypoint 보존, UDP 부호, 센서 오류 제동, 오래된 조향 만료, 유지 목표점 좌표 변환, 재시작 reset.
- Windows 표준 라이브러리 환경에서 설정 로딩 확인. JS 문법 검사.
- Headless Edge에서 원본 5점·주황색 중간점·조향/종방향 frame 분리·유지 시간·뒤쪽 목표점 표시를 확인. JavaScript 오류 0.

## 4. 최근 실주행 입력 재생

입력: `outputs/inference/vldrive_20260922_140644_996813/control.jsonl`, `waypoints.jsonl`.
실제 차량에는 송신하지 않고, 기록된 경로·현재 속도·시간·reset 경계를 사용해 새 제어기 출력을 계산했다. 기존 로그의 expert 명령과 비교했다. 가속·브레이크에는 변경된 조향으로 인한 미래 차량 운동을 반영하지 않았으므로, 이것은 같은 입력에서의 로직 비교다.

- tracking tick 481회, 조향 갱신 161회, 연속 tracking 구간 2개.
- 첫 tracking부터 마지막까지 16.333초. 구간 사이 무효 입력은 controller reset.
- 기존 원본 waypoint 시간 간격을 해당 frame의 waypoints.jsonl에서 읽음.
- 가속 최대 절대 차이: **0**.
- 브레이크 최대 절대 차이: **0**.
- 목표 속도 최대 절대 차이: **0 m/s**.

### 송신 tick 기준 조향 변화

요청 휠각 degree 기준, reset 경계를 연결하지 않음.

| 지표 | 기존 expert PP | LMDrive 횡방향 PID |
|---|---:|---:|
| 평균 절대 요청 휠각 | 9.032° | 13.228° |
| 최대 절대 요청 휠각 | 40.000° | 32.000° |
| 연속 tick 절대 변화 평균 | 1.981° | 1.274° |
| 연속 tick 절대 변화 p95 | 8.413° | 3.951° |
| 연속 tick 최대 절대 변화 | 66.800° | 48.943° |

새 제어기가 10Hz 사이에 출력을 유지하므로 30Hz tick 통계에는 변화 0이 많이 포함된다. 이 표만으로 채터링 감소를 확정하면 안 된다.

### 동일한 조향 갱신 시점끼리 비교

새 제어기가 실제 조향을 갱신한 시점에 기존 명령도 같이 샘플링. 각 연속 구간 안의 샘플 사이 변화 비교.

| 지표 | 기존 expert PP | LMDrive 횡방향 PID |
|---|---:|---:|
| 절대 변화 평균 | 4.626° | 3.838° |
| 절대 변화 p95 | 19.229° | 25.615° |

평균 변화는 약 17.0% 감소했지만 p95는 약 33.2% 증가했다. 따라서 큰 방향 전환까지 안정화됐다고 할 수 없다. 실제 차량의 횡오차·yaw 진동이 개선되는지도 이 재생으로 판단할 수 없다.

## 5. 남은 핵심 문제: 늦게 도착한 첫 두 waypoint

조향 갱신 161회를 기준으로:

- 사용한 예측 나이 평균: **0.8201초**. 모델 계산 시간만이 아니라 anchor 수신부터 제어까지의 전체 나이.
- 새 중간점 거리 평균: **2.9658m**, 범위 **0.0411–8.8622m**.
- 거리 2m 미만: **97/161회 (60.2%)**.
- 중간점이 차량 뒤 또는 같은 전후 위치: **39/161회 (24.2%)**.

모델의 첫 두 점은 anchor로부터 0.5초·1.0초 위치다. 그 공간 중간점은 일정 속도라면 대략 0.75초 앞에 해당한다. 그런데 제어에 도달한 예측이 평균 0.82초 지난 상태라, 이동 보정 후 중간점이 매우 가까워지거나 뒤로 넘어갈 수 있다. 이는 기하학적 해석이며, 39회 뒤쪽 목표점 자체는 실제 기록으로 확인했다.

원본 LMDrive의 첫 두 점 고정 선택은 최소 lookahead 거리를 보장하지 않는다. 따라서 PID 교체만으로 짧은 목표점·지연 문제를 해결했다고 결론내리지 않는다. 뒤쪽 점을 향하는 큰 오차나 0 부근의 방향 변동이 큰 조향 변화와 연결될 수 있다. 화면에는 뒤쪽 목표점 여부를 표시한다.

요청대로 원본 횡방향 로직을 우선 적용했으며, 멀리 있는 점 선택·동적 lookahead·새 smoothing은 임의로 추가하지 않았다. 후속 개선은 입력 지연을 먼저 줄이고, 별도 실험으로 경과 시간을 고려한 앞쪽 목표점 선택을 검토하는 것이 타당하다. 후자는 LMDrive 원본의 첫 두 점 고정 규칙을 변경하는 작업이다.

## 6. 실행과 결론

WSL lmdrive, VLDrive 폴더:

```bash
python inference/realtime_vldrive.py
```

횡방향 PID 교체 및 종방향 출력 보존 검증 완료. 원본 0.8 배율을 포함해 적용. 재생에서는 평균 변화가 줄었지만 큰 변화가 남고, 실제 MORAI 폐루프 개선은 미검증. 이 작업에서 차량에 UDP 제어를 송신하지 않았다.

일회성 재생·브라우저 검사 스크립트는 확인 후 삭제하며, 유지보수용 회귀 테스트와 이 보고서를 남긴다.
