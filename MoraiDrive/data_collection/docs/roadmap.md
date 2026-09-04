# 앞으로 할 작업

## 1. Navigation instruction 생성

- command 연속 구간을 LEFT/RIGHT/STRAIGHT/FOLLOW instruction clip으로 변환
- LMDrive `navigation_instruction_list.txt` JSON-lines 형식 준수
- `route_path`, `route_frames`, `start_frame`, `end_frame`, `instruction`,
  `instruction_id`, `instruction_args` 기록
- `route_valid=true`가 연속인 구간만 사용하고 이탈 구간을 가로지르지 않음
- MORAI map/scenario metadata의 LMDrive 호환 규칙 정의

## 2. Notice instruction 생성

- 신호등 정지·출발, 주변 차량·보행자, 충돌 위험 notice 정의
- 검증된 신호 bit `1=red`, `16=green`을 instruction event로 변환
- `notice_instruction_list.json`을 `{route_path: [events...]}` 형식으로 생성
- CARLA ScenarioRunner 전용 notice는 MORAI 시나리오 의미로 대체

## 3. LiDAR·카메라 기하 정합

- 차량 전방의 고정 물체를 카메라와 LiDAR에 함께 표시해 overlay 확인
- 좌우 반전, yaw 90° 오프셋, 높이 부호 확인
- 필요하면 MORAI calibration/rotation 값을 config로 분리

## 4. Expert 출력 연결

- tick마다 `[steer,throttle,brake]`와 목표 속도를 반환하는 인터페이스 정의
- 값을 `measurements`, `measurements_full`, 별도 expert 배열에 같은 frame ID로 저장
- 손 주행 조향각을 Expert 정답으로 사용하지 않도록 테스트 데이터 분리

## 5. ROS SyncMode 최종 수집

- [Expert SyncMode 명세](expert-sync.md)의 서비스 3종 구현
- `tick → 상태 → expert → control → wait → 센서 저장` 1:1 보장
- timeout, 중복, 센서 누락, route 이탈을 episode summary에 기록
- 최소 10분 연속 수집에서 프레임 누락·중복 0 검증

## 6. 데이터셋 및 학습 smoke test

- 테스트 수집과 최종 학습용 데이터 분리
- instruction/notice와 `dataset_index.txt` 생성 후 LMDrive Dataset 전체 순회
- RGB·LiDAR·measurement tensor의 NaN/Inf/shape 검사
- 작은 batch로 vision encoder와 LLM forward/loss 확인
- 통과 후 전체 route·시나리오 batch 수집

## 7. 시나리오 에디터 연동 및 다중 episode 자동 수집

현재 `collect.py`는 실행할 때 `DatasetWriter`를 한 번만 생성하므로
`collect.py` 1회 실행이 `<route>_<timestamp>` 폴더 1개, 즉 episode 1개에
대응한다. 실행 중 MORAI에 이미 설정되어 있는 NPC·날씨·Ego 시작 위치를 그대로
수집하며, 시나리오를 로드하거나 조건을 바꾸고 Ego를 respawn하는 기능은 없다.

시나리오 에디터 기능이 저장소에 추가되면 다음 기능을 `collect.py` 또는 별도의
상위 batch runner에 연결한다.

- 시나리오 파일/ID를 인자로 받아 MORAI에 명시적으로 로드
- episode마다 NPC 구성, 날씨·시간대, Ego 시작 위치와 route를 적용
- 시나리오 로드 완료와 센서 준비 상태를 확인한 뒤 수집 시작
- route 완주, timeout, 충돌, 센서 오류를 episode 종료 조건으로 처리
- episode 종료 후 `DatasetWriter.finalize()`를 호출하고 다음 시나리오 전에 Ego/NPC를 reset
- 매 episode마다 새로운 `<route>_<timestamp>` 폴더와 `dataset_index.txt` 항목 생성
- scenario ID와 NPC·날씨·spawn 설정을 `route.json` 및 `episode_summary.json`에 기록
- reset/respawn 경계를 하나의 episode 데이터에 섞지 않음
- 여러 scenario와 반복 횟수를 선언하는 batch 설정 파일 지원

시나리오 에디터 코드가 들어오기 전에는 인터페이스를 추측해 구현하지 않고,
추가된 API의 load/reset/status 계약을 확인한 뒤 연동한다.

권장 순서:

```text
navigation/notice 생성기
→ LiDAR·카메라 기하 정합
→ Expert 출력 인터페이스
→ ROS SyncMode 1:1 수집
→ LMDrive 학습 smoke test
→ 시나리오 에디터 연동 및 다중 episode runner
→ 본 수집
```
