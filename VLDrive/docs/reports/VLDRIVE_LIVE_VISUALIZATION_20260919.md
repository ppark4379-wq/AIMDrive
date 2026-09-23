# MORAI UDP → navigation → best VLDrive → waypoint 화면 검증

2026-09-19 최종 상태: **실제 MORAI UDP를 수신해 best checkpoint로 추론하고 브라우저에 표시했다. 차량 제어기는 연결하지 않았다.** 최종 40초 추론 구간에서 유효한 출력 17개를 저장했다. 이 기록은 연결과 화면 동작 검증이며, 수동 주행 한 바퀴의 경로 정확도 평가가 아니다.

## 실행

WSL `(lmdrive)` 환경의 `AIMDrive/VLDrive` 폴더에서:

```bash
python inference/realtime_vldrive.py --config configs/vldrive_realtime.json
```

`http://127.0.0.1:8502` 화면이 자동으로 열린다. 처음에는 모델 로딩과 실제 입력 이력 `n/11`을 표시한다. 실행별 로그는 `outputs/inference/vldrive_<시각>/`에 저장한다. 종료는 터미널 Ctrl+C다. 브라우저 창만 닫으면 추론은 계속된다.

추론 검증 프로세스는 종료했다. 원래 실행 중이던 Windows 센서 중계는 아래 작은 카메라 packet 설정으로 교체해 유지했다. 중복 중계나 별도 navigation observer를 같은 수신 포트에 띄우지 않는다.

## 기능별 파일 위치

| 위치 | 역할 |
|---|---|
| `inference/realtime_vldrive.py` | 설정 검증, checkpoint 계약 검사, 화면/추론 실행 |
| `inference/live_pipeline.py` | 센서 수집, 제한 큐, encoder 이력, Stage-2 추론, 로그 |
| `inference/live_frame.py` | 학습 캐시와 동일한 센서 전처리, anchor 카메라 및 GPS/status snapshot |
| `inference/vldrive_runtime.py` | 기존 checkpoint loader/forward, 실제 11프레임 이력 |
| `inference/morai_udp.py`, `inference/udp_relay.py` | 센서 수신 및 Windows→WSL 센서 중계 |
| `navigation/route_source.py` | 기존 observer의 global path/MGeo 경로 로딩 공용화 |
| `navigation/live.py` | 같은 수신기의 GPS/status로 현재 경로 진행과 instruction 생성 |
| `utils/inference_dashboard/` | 로컬 HTTP 서버, HTML/CSS/JS 시각화 |
| `configs/vldrive_realtime.json` | 모델, 포트, 실행 빈도, 화면 설정 |
| `configs/navigation_observer.json` | 기존 global path, route definition, 지도, 좌표 변환 |
| `configs/vldrive_udp_relay.json` | 현재 Windows→WSL 중계 IP/port와 카메라 packet 크기 |

## 입력과 표시의 연결

4개 카메라와 LiDAR를 기존 adapter로 전처리하고 Stage-1 encoder의 `[106,256]` 특징을 얻는다. 학습 설정의 0.2초 간격을 목표로 실제 11프레임을 모아 `[1,11,106,256]`으로 Stage-2에 전달한다. 허용 간격은 0.2±0.06초이며, 누락을 복제 프레임이나 zero padding으로 메우지 않는다. 실제 검사 기준 시각은 monotonic **수신 시각**이다. 센서 공통 하드웨어 동기화를 입증한 것은 아니다.

Navigation은 기록된 데이터셋 문장을 재생하지 않는다. `navigation_observer.json`에 지정된 실제 global path/MGeo와 현재 GPS/status의 위치·방향으로 기존 NavigationEngine이 문장을 생성한다. 새 route source 모듈을 observer도 공유한다. 기존 문장 template 및 경로 판단 규칙은 유지했다. live instruction 연속성의 최대 센서 간격만 observer의 pose timeout 0.5초에 맞췄으며, 실제 적용 policy는 `planned_route.json`에 남긴다.

Encoder 입력에 사용한 front JPEG와 해당 anchor의 instruction/pose를 이력에 함께 보관한다. Stage-2에는 그 anchor instruction을 전달하고 화면에도 같은 문장을 표시한다. 출력 이후 새 instruction으로 과거 예측의 문구를 바꾸지 않는다. `--instruction "..."`를 명시한 경우에만 고정 문장 모드가 된다. 기본은 `navigation.mode=route`다.

화면은 front 이미지, instruction, 5개 미래 waypoint(0.5/1/1.5/2/2.5초), global path/차량 위치, token 수, 계산 시간, anchor 수신 이후 경과 시간, 센서별 packet/sample/age를 표시한다. Waypoint는 기존 좌표 변환을 재사용해 오른쪽 +x, 전방 +y, m 단위로 동일 축척 표시한다. 원본 출력 좌표는 `[left, -forward]`다.

## 모델과 성능 제한

- Stage-1: `outputs/checkpoints/morai_visual_encoder.pth.tar`, SHA256 `d0cd815c5ca5303b4844532b18c1420d508bd408f244bad08b9d9d48eee25be4`.
- Stage-2: `outputs/logs/vldrive_20260919_182054_301289/model-best.pt`, epoch 10 / update 3730, SHA256 `7b369b0349ede0c2f8511e677ea53b4dba73a498fde6dfebf2aacea580c022c8`. 전체 학습은 15 epoch 완료했고 이 best를 strict load했다.
- 기존 추론 설정 그대로 eval + hard Gumbel, reconstruction off. 이 작업에서 학습 모델, loss, checkpoint를 바꾸지 않았다. Gumbel 때문에 정지 장면에서도 출력이 조금 변할 수 있다.
- Encoder는 기존 cache 계약의 FP32 계산과 FP16 저장 경계를 재현한다. Stage-2는 기존 autocast를 재사용한다. 이 encoder의 velocity 입력은 학습 때처럼 0 고정이다.
- `runtime.target_hz=5`: 입력 이력 목표 빈도. `runtime.prediction_hz=0.5`: 최대 2초에 한 번 waypoint 계산. `visualization.refresh_hz=5`: 브라우저 상태 조회 빈도. 세 값의 의미는 다르다.
- MORAI 실행 중 GPU 경합이 있어 encoder만 측정한 중앙값도 약 158ms였다. 수집과 encoder를 분리하고 최대 3개 전처리 입력 큐를 사용한다. CPU 연산 스레드 설정은 기존 2개다. 프레임마다 PNG/PDF나 별도 모델 프로세스를 만들지 않는다.
- 수집 센서의 최대 age는 0.3초, 센서 간 수신 시차는 0.15초다. 화면 예측 만료 3초는 이 입력 조건과 별개다. anchor 시각을 현재 시각으로 바꾸지 않으며 화면에 경과 시간을 표시한다. 이 설정은 **관찰용**이고 제어 지연 요구를 만족한다는 뜻이 아니다.
- 경로 이탈/GPS 불량/이력 단절/예측 만료 시 과거 경로를 숨기고 대기 상태를 표시한다. 이력이 끊기면 11프레임을 다시 모으므로 출력 간격이 길어질 수 있다.

## 실제 UDP에서 발견하고 수정한 문제

Windows→WSL의 기존 약 65KB 카메라 datagram에서 조각 누락이 발생했다. 초기 8초 점검에서는 관측한 timestamp 146개 중 완성 영상 63개, 조립 중 불연속 36회, 완성 영상 사이 최대 간격 약 561ms였다. 이 상태에서는 11프레임 연속 이력이 만들어지지 않았다.

중계에 선택 설정 `camera_datagram_bytes=1400`을 추가했다. JPEG 내용과 timestamp를 보존하면서 MORAI application chunk를 연속 index의 작은 datagram으로 재분할한다. 기존 중계 기본 동작은 0으로 유지한다. receiver에는 서로 다른 timestamp의 조각이 섞이지 않도록 검사를 추가했다. 변경 후 별도 8초 측정은 **159/159개 완성, 조립 불연속 0회, 최대 간격 74.4ms**였다. 이는 각각 별도 시간 구간의 관측값이다. 특정 시스템 네트워크 설정이나 전역 socket buffer 한도는 바꾸지 않았다.

현재 중계 설정은 `configs/vldrive_udp_relay.json`: WSL IP `172.18.254.59`, Windows localhost의 `909→10909`, `2368→2368`, `9090` source port demux를 재사용한다. 기존 일반 중계만 교체했고 기존 GPS 전용 중계는 유지했다. 새 일반 중계 PID는 검증 시점 49556이며 숨김 프로세스로 실행 중이다. 이전 명령은 `outputs/inference/udp_relay_previous_command.txt`에 보관했다. WSL 재시작 후 IP가 바뀌면 config의 `to`를 갱신해야 한다.

## 최종 실측과 증거

실행: `outputs/inference/vldrive_20260919_230547_947660/`. 모델 로딩 이후 40초의 추론 구간이다.

| 항목 | 결과 |
|---|---:|
| 유효 waypoint 출력 | 17개 |
| Encoder 시간 중앙값 | 142.6ms |
| Stage-2 시간 중앙값 | 470.0ms |
| Encoder + Stage-2 중앙값 | 613.8ms |
| Anchor 수신 → 출력 중앙값 | 770.4ms |
| 출력 시점의 지연 범위 | 539.2–969.0ms |
| 마지막 anchor 유지 token 평균 | 36.65 / 106 |
| 센서 시차로 거부한 프레임 | 0 |
| 유효성 검사에서 거부한 프레임 | 0 |
| 입력 큐에서 버린 프레임 | 2 |
| 이력 초기화 | 4회 |
| 계산 중 이력 변경 등으로 버린 예측 | 1개 |

최종 실측에서는 `Left, then straight.`가 실제 입력 문장이었다. 차량은 거의 정지 상태였다. 마지막 예측의 2.5초 지점은 왼쪽 약 1.11m, 전방 약 10.88m였다. **정지 입력에서도 전진 경로가 나온다는 사실을 확인했으며, 이를 올바른 정지 행동이라고 판단하지 않는다.** 손 주행 중 좌회전/직진/정지 등에 맞는 예측인지 별도 확인해야 한다. 현재 결과로 주행 품질이나 전체 route의 instruction 정확도를 확정할 수 없다.

실제 브라우저 검증은 현재 HTTP 서버에 접속해 anchor 이미지, waypoint 5개와 ego 원점, global path가 렌더링되는 것을 확인했다. JavaScript page error는 0개였다.

- [실제 화면](../../outputs/inference/vldrive_20260919_230547_947660/live_dashboard.png)
- [브라우저 검사](../../outputs/inference/vldrive_20260919_230547_947660/browser_check.json)
- [최종 실행 보고서](../../outputs/inference/vldrive_20260919_230547_947660/inference_report.json)
- [출력 waypoint와 anchor 이력](../../outputs/inference/vldrive_20260919_230547_947660/waypoints.jsonl)
- [카메라 수신 점검](../../outputs/inference/front_packet_check.json)

## 회귀 검증과 제어 분리

최종 `pytest tests -q`: **307 passed, 1 skipped, 30 subtests passed**. 프로젝트의 격리 의존성 `.runtime/stage2_wsl_deps`와 `.runtime/offline_viz_wsl_deps`를 PYTHONPATH에 지정한 WSL lmdrive 환경에서 실행했다.

추가 검증은 current pose에 따른 문장 변경, stale/skew/offroute/rewind, 실제 이력과 anchor 문장/이미지 연결, 화면 만료 및 좌표, JPEG 내용 보존과 timestamp 혼합 방지, local HTTP GET 전용, 제어 송신 API 부재를 포함한다. 임시 packet/encoder/browser 실험 스크립트는 삭제했고 결과 파일과 필요한 회귀 테스트만 남긴다.

추론과 navigation에는 차량 제어 UDP 송신이 없다. HTTP 서버는 localhost 읽기 전용이다. 중계의 UDP 송신 대상은 WSL 센서 수신기이며 MORAI 제어 포트가 아니다. steering/throttle/brake를 생성하거나 제어기에 전송하지 않는다.
