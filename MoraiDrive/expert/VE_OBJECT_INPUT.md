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

Expert를 종료한 상태에서 센서만 전송하고 실행한다. 아래 명령은 차량을
제어하지 않으며 카메라·LiDAR 수신, 전처리, 추론, UDP·ROS 전달을 계측한다.

```bash
bash MoraiDrive/expert/tools/run_ve_hz_check.sh
bash MoraiDrive/expert/tools/run_ve_hz_check.sh --overnight --background
```

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
