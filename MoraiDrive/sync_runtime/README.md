# Sync Runtime

MORAI의 ROS SyncMode를 시작·종료하고 시뮬레이션을 한 프레임씩 진행하는 독립 실행 모듈입니다. Expert의 제어 명령을 적용하고, 센서 캡처 요청과 해당 프레임의 차량 상태를 기록합니다.

## 역할과 동작

1. `/ctrl_cmd`에서 최신 제어 명령을 받습니다.
2. `/SyncModeCtrlCmd`로 현재 프레임의 제어 명령과 센서 캡처 여부를 전달합니다.
3. `/SyncModeWaitForTick`으로 한 스텝을 진행하고 응답 프레임이 정확히 1 증가했는지 확인합니다.
4. 응답의 차량 상태를 `/Ego_topic`으로 발행하여 Expert가 다음 제어를 계산할 수 있게 합니다.
5. 캡처 대상 프레임이면 차량 상태·제어 명령·예상 센서 파일 경로를 JSONL로 기록합니다.

첫 스텝은 차량 상태를 확보하는 초기화 단계라 캡처하지 않습니다. 이후 초기 응답 프레임을 기준으로 `capture_stride`마다 캡처합니다.

새 명령을 기다리며 프레임 진행을 멈추지는 않습니다. 새 명령이 있으면 `fresh`, 없으면 이전 명령을 사용하는 `reused`, 일정 프레임 이상 새 명령이 없으면 제동하는 `failsafe`로 동작합니다. 최초 명령 수신 전에도 `failsafe`를 사용합니다.

## 파일 구성

| 파일 | 역할 |
|---|---|
| `main.py` | 실행 진입점. 설정 로딩, ROS 연결, 준비 신호 대기, 프레임 루프, 기록 및 종료 처리 |
| `sync_client.py` | SyncMode 서비스 호출, 프레임 증가 검증, 차량 상태 발행, 기어 설정 |
| `control_buffer.py` | 최신 제어 명령 보관과 `fresh` / `reused` / `failsafe` 선택 |
| `frame_record.py` | 캡처 프레임 선정과 차량 상태·제어 명령의 기록 형식 정의 |
| `capture_journal.py` | 별도 스레드에서 `capture_manifest.jsonl` 저장 |
| `config.yaml` | 시간 간격, 캡처 주기, 토픽·서비스 이름, 제동 및 실행 모드 설정 |
| `__init__.py` | Python 패키지 정의 |

## 다른 모듈과의 관계

```text
Expert ── /ctrl_cmd ──> Sync Runtime ── 제어·프레임 진행·캡처 요청 ──> MORAI
Expert <── /Ego_topic ── Sync Runtime <── 다음 프레임의 차량 상태 ── MORAI

Sync Runtime ── capture_manifest.jsonl ──> Data Collector
MORAI ── 카메라·LiDAR 원본 파일 ──────────> Data Collector ──> 데이터셋
```

- **`expert/`**: 경로 추종과 제어 명령 계산을 담당합니다. Runtime은 받은 명령을 시뮬레이터에 적용하며, 주행 경로를 직접 계획하지 않습니다.
- **`scenario_runner/`**: 시나리오 환경 구성과 Expert 실행을 담당합니다. Runtime의 `scenario` 모드는 `/morai_drive/scenario_ready`가 `false → true`로 바뀌기를 기다린 뒤 시작하고, `/morai_drive/sync_runtime_active`로 실행 상태를 알립니다. Runner는 Expert 실행 확인 후 준비 신호를 설정하고, Runtime 활성화 후 새 Ego 메시지 수신까지 기다립니다. Runner가 종료하면서 READY를 해제하면 Runtime도 SyncMode를 종료합니다. Runtime은 Runner보다 먼저 시작하여 새 준비 신호를 관측해야 합니다.
- **`data_collection/`**: `--mode sync_native_files`에서 `--capture-manifest`로 받은 기록과 MORAI가 저장한 센서 파일을 연결해 데이터셋으로 변환합니다. Runtime은 이미지·LiDAR 파일을 직접 저장하거나 최종 데이터셋으로 변환하지 않습니다. 수집기의 별도 `sync` 모드와는 SyncMode 소유권이 겹치므로 동시에 실행하지 않습니다.
- **`diagnostics/`**: 프레임·센서 동작을 조사하는 도구와 분석 문서를 보관합니다. `sync_frame_probe.py`도 SyncMode를 직접 사용하므로 Runtime과 동시에 실행하지 않습니다.

## 실행과 주요 설정

ROS 환경과 `morai_msgs`를 로딩하고 MORAI의 SyncMode 서비스가 준비된 상태에서, `MoraiDrive/`를 작업 디렉터리로 실행합니다.

```bash
python3 -m sync_runtime.main --mode standalone
```

`standalone`은 준비 신호 대기 없이 시작하며 기본 설정에서 기어를 D로 변경합니다. `scenario`는 위 준비 신호 연동 후 사용하며 Runtime에서 기어를 변경하지 않습니다. 설정 파일은 기본적으로 이 폴더의 `config.yaml`을 읽고, `--config`로 변경할 수 있습니다.

| 기본 설정 | 의미 |
|---|---|
| `time_step_ms: 20` | 한 프레임당 시뮬레이션 시간 20ms. 양의 20ms 배수만 허용 |
| `capture_enabled: true`, `capture_stride: 5` | 5프레임마다 캡처 요청. 기본 시간 간격에서는 시뮬레이션 시간 기준 0.1초 간격 |
| `warn_after_frames: 5` | 새 제어 명령 없이 5프레임 진행하면 경고 |
| `fail_safe_after_frames: 25` | 새 제어 명령 없이 25프레임 진행하면 제동 명령 적용 |

시뮬레이션 시간 간격과 실제 처리 속도는 다릅니다. 제어가 끊겨 `failsafe`가 되어도 Runtime의 프레임 진행과 캡처는 계속됩니다.

## 생성되는 기록

기본 저장 위치는 **실행 작업 디렉터리 기준** `.runtime/sync_runtime/<session_id>/capture_manifest.jsonl`입니다. 위 실행 기준으로는 `MoraiDrive/.runtime/sync_runtime/` 아래에 생성됩니다.

각 줄에는 캡처 기준 프레임 `N`의 상태(`ego_state_for_frame`), 해당 스텝에 적용한 명령(`applied_control`), 응답 프레임 `N+1`의 상태(`next_ego_state`), 제어 수신 상태와 예상 센서 파일 경로가 들어갑니다.

`capture_requested`와 `expected_sensor_files`는 캡처 요청과 예상 경로를 뜻하며 실제 파일 저장 성공을 보장하지 않습니다. 기록 큐가 가득 차거나 쓰기에 실패하면 메타데이터가 누락될 수 있고, Runtime은 이를 경고하면서 프레임 진행을 계속합니다.

## Scenario Runner 연동 실행

별도 터미널에서 `python3 -m sync_runtime.main --mode scenario`를 먼저 실행하고, Runner를 `python3 -m scenario_runner.main_runner --scenario full_loop --once`로 실행합니다. 두 터미널 모두 `MoraiDrive` 디렉터리에서 ROS 및 `expert/devel/setup.bash` 환경을 불러와야 합니다. 매 회차 Runtime을 다시 실행합니다. Runner의 준비 대기는 기본 30초입니다.
