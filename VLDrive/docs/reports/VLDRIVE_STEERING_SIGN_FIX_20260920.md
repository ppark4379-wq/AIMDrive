# 좌측 예측 → 실제 우회전: 조향 부호 수정

2026-09-20. 원인은 `configs/vldrive_control.json`의 `pid.morai_steering_sign=-1`이었다. 이 설치 환경의 EgoCtrlCmd는 음수 명령에서 실제 yaw가 감소하며 우회전했다. 따라서 **+1로 수정**했다. 모델 waypoint의 좌우 좌표와 PID angle 계산에서 방향이 뒤집힌 문제는 아니었다.

## 실제 로그 근거

| 실행 | 비교 가능한 연속 구간 | 음수 steer + 음수 yaw 변화 |
|---|---:|---:|
| `vldrive_20260920_022732_541827` | 113 | 113 |
| `vldrive_20260920_022940_895418` | 82 | 82 |

tracking이 연속되는 5개 샘플에서 진행 속도 ≥1m/s, 이전 4개 명령 평균의 절댓값 ≥0.05, 현재→다음 status 시각 차이 0..0.2초, yaw 변화율 절댓값 ≥0.2°/s인 구간을 비교했다. yaw wrap을 보정하고 한 샘플에서 0.15rad 넘게 튀는 구간을 제외했다. 두 실행의 비교 가능한 195개 구간 모두 음수 steer에서 yaw 감소, 즉 우회전이었다. 양수 명령의 실제 주행 표본은 이 로그의 선택 구간에 없으며, 양수 좌회전 설정은 관측한 반대 방향 응답에 맞춰 적용했다.

두 번째 실행의 frame 674 예시:

- instruction: 현재 교차로 좌회전 후 다음 교차로 직진.
- raw waypoint의 left 성분: `[0.1572, 0.3242, 0.6758, 0.9688, 1.2188]`m로 모두 좌측.
- 지연/현재 pose 보정 후 PID 평균 목표: `[left=1.9355, forward=0.8575]`m.
- PID angle error: `+0.73451`로 좌측.
- 기존 전송값: `steer=-0.8`; 실제 yaw 감소 구간에서 약 -39°/s까지 관측.
- 수정 후 같은 PID 상태의 전송값: `steer=+0.8`.

모델의 좌측 목표를 반대로 움직일수록 좌측 오차가 커지고 기존 음수 명령이 더욱 커지는 잘못된 피드백이었다. Windows 중계기와 packet serializer는 받은 steer 값을 그대로 전달하며 추가 부호 반전은 없었다.

## 왜 최초 검사에서 놓쳤는가

처음에는 다른 MORAI 예제의 부호를 이 실행 환경에도 적용했다. [UDP 문서](https://help-morai-sim.scrollhelp.site/ko/morai-sim-drive/24.R2/udp-1)에서 현재 사용하는 55바이트 EgoCtrlCmd와 오른쪽 양수 표기가 있는 Ground Vehicle Direct Ctrl Cmd는 서로 다른 메시지다. 예제/다른 controller의 부호를 현재 환경 전체에 일반화한 것이 잘못이었다.

최초 자동 테스트는 가정한 음수 좌회전 규칙대로 코드가 동작하는지만 확인했다. 테스트 수신기는 숫자가 전달되는지 확인했으며 실제 차량의 회전 방향을 확인하지 않았다. 당시 정지 차량으로 받은 실제 예측 테스트에서는 원본 LMDrive의 정지 시 조향 오차 0 처리 때문에 비영 조향이 나오지 않았다. 이번에는 사용자의 실제 주행 로그에서 명령과 yaw 응답을 비교했다.

## 수정 내용

- `configs/vldrive_control.json`: `morai_steering_sign`을 -1→+1. 작업 중 현재 설정이 3m/s로 조정된 것을 확인했고 그대로 보존했다.
- `control/waypoint_pid.py`: MORAI 전체가 오른쪽 양수라는 잘못된 주석 제거. 로그에 `steer_pid_left_positive`, `steering_sign` 추가.
- `control/runtime.py`: 활성 설정에 따른 `steering_positive_direction`을 화면 상태에 포함.
- `utils/inference_dashboard/app.js`: 오른쪽 + 고정 문구 제거. 현재 설정은 왼쪽 + 표시. 구버전 상태처럼 방향 metadata가 없으면 임의의 방향을 표시하지 않는다. BEV 그림은 기존처럼 화면 오른쪽이 +x다.
- 기존 안내/보고서에 초기 부호 가정의 오류와 후속 수정을 명시.

모델, 원본 waypoint, 좌표 변환, PID 계수, 제어 주기, 패킷 구조는 변경하지 않았다. 현재 속도 상한 3m/s(10.8km/h)도 유지했다.

## 검증

두 실행의 실제 PID 갱신 **245+85=330회**를 기록 당시 계수·속도 제한으로 재생했다. 기록된 aim/예측 속도/현재 속도와 PID reset 순서를 사용했다. 기존 steer·accel·brake를 허용 오차 1e-10 내 재현했고, +1 설정에서 steer만 정확히 반전되고 accel/brake는 동일했다. 이는 기록 입력을 재생한 결과이며 수정된 모델 주행의 미래 입력까지 같다는 뜻은 아니다.

좌/우 목표의 부호, 서로 다른 polarity의 가속·제동 동일성, 기록된 좌측 aim이 실제 직렬화 패킷에서 +0.8인지 검사하는 회귀 테스트를 추가했다. 속도 상한을 조정해도 원본 수식 검사가 깨지지 않도록 해당 단위 테스트의 5m/s fixture는 파일을 수정하지 않고 메모리에서 고정했다.

측정 결과: `outputs/inference/steering_sign_audit_20260920.json`.

최종 전체 회귀 결과: **358 passed, 1 skipped, 30 subtests passed (33.46초)**. 임시 분석 스크립트는 삭제했고 측정 JSON과 회귀 테스트를 남겼다.

수정 후 새로운 실제 차량 주행은 이 분석에서 시작하지 않았다. 다음 실행부터 +1이 적용된다. 기존 프로세스가 실행 중이면 재시작해야 한다.

```bash
# VLDrive 폴더, WSL lmdrive 환경
python inference/realtime_vldrive.py --control
```
