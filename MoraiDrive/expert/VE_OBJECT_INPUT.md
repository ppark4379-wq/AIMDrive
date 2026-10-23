# VE 장애물 입력으로 기존 Expert 실행

카메라 4개·LiDAR UDP → VE bbox·스칼라 속도 → 로컬 객체 추적 → UDP 12050
→ `/ve/Object_topic` → 기존 Expert의 경로 계획·제어로 연결한다.
Expert 실행 시 `/Object_topic:=/ve/Object_topic`으로 장애물 구독만 바꾼다.
자차 `/Ego_topic`과 신호등 `/GetTrafficLightStatus`는 기존 ROS 입력이다.
VE waypoint와 LLM 출력은 주행에 사용하지 않는다.

객체 클래스는 `type=-1`로 유지한다. `ve_tracked_unknown:*` 이름을 가진
객체만 기존 Expert의 ACC·회전교차로·고주로·정적 회피 분기에 추가로 허용한다.
이동/정지/불확실 상태는 위치 이력과 VE 속도로 추정하며, 시뮬레이터 객체
ID나 정답 분류를 가져오지 않는다. UDP 실행은 `perception_source=ros`와
토픽 remap을 사용한다. 별도 ROS 전용 실험 모드인 `perception_source=ve`는
다른 실행 경로이므로 UDP 실행 설정에 혼용하지 않는다.

현재 추적 정책은 `world_position_supported_speed_v2`다. 최소 4회·0.3초의
일관된 위치 변화로 이동을 확인하고, 그 속도와 맞을 때만 VE 스칼라 속도를
Expert에 전달한다. 두 값이 다르면 확인된 위치 이력 속도를 사용한다.
높은 VE 속도만으로 이동을 선언하거나 다음 bbox 위치를 예측하지 않는다.
정지는 0.5초 이상의 안정된 위치 이력으로 판단한다.

관측 초기나 bbox 흔들림으로 움직임을 확인하지 못하면 객체는 `uncertain`으로
계속 전달하고 계획용 속도는 0으로 둔다. 이것은 정지 GT나 정확한 속도 추정이
아니며, 새로 보이는 이동 차량에도 일시적인 감속·양보가 늘 수 있다.
원본 `speed_mps`, bbox 위치·크기는 보존하며 `planning_speed_mps`와
`planning_speed_source`에 Expert용 속도와 근거를 별도로 기록한다.
센서 시점 오차나 연속적인 위치 편향까지 이 추적기가 해결하지는 않는다.

LiDAR UDP 입력은 MORAI의 프레임별 microsecond-of-hour 타임스탬프와
연속된 방위각을 확인한다. 한 프레임의 마지막 패킷을 받으면 바로 출력하며,
다음 프레임을 기다리지 않는다. 패킷이 빠지거나 섞인 프레임은 제외한다.
촬영 시각은 Ego 또는 카메라의 source header를 기준으로 복원해 계측 로그의
`selected_samples.lidar.source_stamp_ns`에 기록한다. UDP VE의 ROS 자차 입력은
이 LiDAR 시각 이전 50ms 이내의 이력에서 선택하며 미래 자차 위치를 적용하지
않는다. 카메라는 여전히 각 최신값이므로 `source_spread_ms`를 확인해야 하며, 이 수정만으로
모든 센서의 동시 촬영이나 충돌 회피 성공을 보장하지 않는다.

## 실행 환경

ROS Noetic 및 프로젝트 `morai_msgs`, 기존 Expert 환경, GPU 추론용 Python이
필요하다. 기존 `internal_sensor_record_env.sh`와 PC별 환경 설정을 사용한다.
체크포인트 파일은 Git에 포함하지 않는다.

- 기본 E3: `mixed_grid_fast_20260930_152712_134039/checkpoint-best-object.pth.tar`
- 선택 E4: `continue_e4e5_20261004_050644/checkpoint-epoch-001.pth.tar`
- 체크포인트 경로·SHA256: `VLDrive/inference/realtime_ve_objects.py`의 `CHECKPOINTS`
- 센서 포트·LiDAR 변환: `VLDrive/configs/vision_encoder_realtime_best_20261003.json`

현재 기본 경로는 `/mnt/d/MoraiDrive/dataset/_training_runs/morai_transfer`다.
다른 PC에서 경로가 다르면 위 체크포인트 경로를 해당 위치로 바꾸되 SHA256은
유지한다. 센서 설정의 경로 항목에도 로컬 경로가 있으므로 사용 전에 확인한다.
UDP 장애물 실행은 `perception_only=True`여서 지도/route 파일을 읽지 않지만,
Expert의 주행 경로와 지도는 별도로 필요하다. 임의 센서 위치 변경 시 기존
LiDAR 변환을 그대로 적용하지 않는다.

MORAI는 Play 상태로 유지한다. 차량 ROS 제어 `/ctrl_cmd`, 자차·신호등 연결과
카메라·LiDAR UDP 전송이 필요하다. 다른 VE 실행이나 데이터 수집과 동시에
같은 센서 포트를 사용하지 않는다. 기존 연결 점검 명령은 다음과 같다.

```bash
cd ~/AIMDrive
bash MoraiDrive/data_collection/tools/01_internal_sensor_record_preflight.sh --scenario-only
```

## 주행 및 화면

기존 시나리오 러너가 경로·NPC·Expert를 함께 관리하는 실행:

```bash
cd ~/AIMDrive
bash MoraiDrive/scenario_runner/tools/run_ve_scenario.sh roundabout
bash MoraiDrive/scenario_runner/tools/run_ve_scenario.sh highway
```

하나를 선택해서 실행한다. `--repeat`는 시나리오 반복, `--ve-checkpoint E4`는
E4 선택이다. `--check`는 오프라인 설정 확인이며 실제 주행 성공 판정이 아니다.
시작 후 `[PASS] Expert → VE 장애물 입력 연결 확인. GT 장애물 구독 없음.`을
확인한다. `[VE Hz]`에 입력·추론·ROS 발행 속도가 표시된다.

실제 장애물 정보와 비교하려면 위 명령에 `--ve-compare-gt`를 추가한다.
별도 기록기가 `/Object_topic`, `/Ego_topic`, 읽기 전용 gRPC bbox를 저장한다.
Expert의 장애물 입력은 계속 `/ve/Object_topic`만 사용한다. 종료 후 실행 폴더의
`gt_comparison_report.json`은 원래 토픽 기준, `gt_geometry_audit_report.json`은
차량 기준점과 bbox 중심의 차이를 반영한 비교 결과다. `bbox_reference_review/`에는
GT와 VE 박스를 겹쳐 보는 HTML이 있다. LiDAR의 실제 촬영 시각은 확인되지 않으므로
수신 시각으로 맞춘 수치는 비동기 오차를 포함한다. 분석 `COMPLETE`와 주행 성공은
별개이며, 시나리오 충돌·종료 원인은 콘솔 로그로 함께 확인한다.

이전에 생성한 기존 Expert 경로를 그대로 쓰는 단독 실행:

```bash
bash MoraiDrive/expert/tools/run_ve_udp_expert.sh --drive --seconds 0
```

`MoraiDrive/expert/.runtime/path.csv` 또는 `.runtime/scenario_runner/path.csv`가
필요하다. `--seconds 0`은 Ctrl+C까지 실행한다. `--drive`를 생략하면
`/ve/shadow_ctrl_cmd`로 계산 결과만 출력한다.
시나리오 러너와 단독 Expert를 동시에 실행하지 않는다.

별도 터미널에서 VE bbox·추적 상태와 Expert 경로를 BEV로 본다:

```bash
bash MoraiDrive/expert/tools/run_ve_bev.sh
```

## 수신·추론 계측

Windows UDP 중계(`VLDrive/inference/udp_relay.py`)는 카메라 영상의 바이트와
timestamp를 유지하면서 WSL로 보내는 UDP를 최대 1,200바이트로 분할한다.
기존 65,000바이트 전송에서 확인된 Windows→WSL 카메라 누락을 줄이기 위한
처리다. 중계 코드를 갱신했으면 실행 중인 Windows 중계도 다시 시작해야 한다.
시작 로그의 `camera UDP maximum=1200 bytes`를 확인한다.
`--camera-datagram-bytes 0`은 비교 진단용 기존 전송 방식이다.
패킷 수는 증가하므로 영상 수신 속도는 `packet_hz`가 아닌
`completed_sample_hz`로 확인한다. 센서 포트와 LiDAR 전송 내용은 동일하다.

VE UDP 실행은 카메라 수신·조립을 별도 프로세스에서 처리한다. 추론 프로세스는
공유 메모리에서 최신 완성 JPEG만 가져온다. 이미지 바이트, 수신 시각과 sample ID를
유지하며, 카메라 Hz를 낮추거나 이미지를 다시 압축하지 않는다. 수신 프로세스가
종료되면 오래된 영상을 새 영상처럼 사용하지 않고 오류로 종료한다.

Expert를 종료한 상태에서 센서만 전송하고 실행한다. 아래 명령은 차량을
제어하지 않으며 카메라·LiDAR 수신, 전처리, 추론, UDP·ROS 전달을 계측한다.

```bash
bash MoraiDrive/expert/tools/run_ve_hz_check.sh
bash MoraiDrive/expert/tools/run_ve_hz_check.sh --overnight --background
```

기본 카메라 수신 모드는 `process`다. 동일한 센서·중계 조건에서 이전 방식과
비교할 때만 `--camera-receiver-mode thread`를 지정한다. 일반 VE UDP 실행에서는
설정 파일의 `runtime.camera_receiver_mode`로 같은 옵션을 지정할 수 있다.
추론은 새 LiDAR sweep마다 한 번 수행하므로 카메라 Hz가 추론 Hz를 결정하지 않는다.

기본은 60초, overnight는 30분씩 8회다. 구간 사이에 모델 프로세스를 다시
시작한다. 시작 메시지의 경로에 `status.json`, `console.log`, 구간별
`timing_report.json`, 최종 `campaign_report.json`이 생성된다.
전체 결과는 `MoraiDrive/scenario_runner/.runtime/ve_objects/` 아래에 저장한다.
백그라운드 종료는 해당 `launch.json`의 PID에 SIGINT를 보낸다.

카메라 패킷은 같은 timestamp·연속 fragment index·정상 JPEG 경계가 확인되어야
사용한다. 추론한 LiDAR sample ID를 고정해 동일 sweep의 중복 추론을 방지한다.
이 검사는 네트워크 누락을 복구하거나 센서의 물리적 동시 촬영을 보장하지 않는다.
`COMPLETE`는 계측 완료다. 누락·지연·JPEG 오류·선택된 LiDAR 중복 항목을
별도로 확인해야 하며, 무충돌 주행이나 검출 정확도의 통과 판정이 아니다.

이번 변경에는 일반 lattice 후보의 충돌 검사 범위를 주변 객체 전체 및 경로
샘플 사이까지 확대하는 수정도 포함된다. 해당 변경은 VE 이외의 일반 lattice
사용 경로에도 적용된다. 실제 MORAI 주행 성능과 장시간 안정성은 별도 확인한다.
