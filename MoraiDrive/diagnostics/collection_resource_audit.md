# 데이터 수집 자원 사용 감사 — 2026-09-10

## 결론

현재 호스트 RAM 압박의 가장 큰 점유자는 RaiDrive.Mount.Service.x64(PID 23400)다. 조사 시점 Working Set 약 21,536 MiB, Private Bytes 약 24,742 MiB였다. Windows 가용 RAM은 1,704 MiB, 전체 Commit은 58,207,363,072 bytes였다. 이는 수집 종료 후에도 남아 있는 점유량이다. vmmemWSL Working Set은 약 215 MiB였다.

Z:의 DisplayRoot는 RaiDrive의 Synology 연결이며 Collector 기본 출력이 /mnt/z/DCM/data_collector이다. 따라서 NAS 가상 드라이브의 메모리 점유를 조사하지 않고 WSL 내부 free 결과만으로 RAM 문제를 배제한 이전 판단은 부적절했다.

확인된 사실: 현재 메모리 압박의 최대 기여 프로세스는 RaiDrive다. 아직 미확정: 그 내부에서 파일 메타데이터 캐시, 업로드 대기, 메모리 누수 중 무엇이 24GB를 보유하는지; 앞서 관측한 모든 Python 예외가 이 메모리 압박 때문에 발생했는지; 다른 PC도 같은 원인인지. 수집 도중 프로세스별 시계열이 없어 이 인과관계까지 확정할 수 없다.

## 검사 범위

LMDrive, MoraiDrive, VLDrive의 git 추적 파일을 전체 목록화하고 모든 추적 Python 파일을 AST 파싱했다. 파일별 문법/읽기 오류는 없었다. 메모리 관련 호출 패턴과 MoraiDrive import 목록은 resource_audit_inventory.json에 기록했다.

이는 전체 코드의 모든 동작을 줄 단위로 증명한 것은 아니다. 실제 수집 진입점과 연결된 모듈은 수동 검토했고, 학습/외부 라이브러리는 정적 패턴과 호출 여부를 구분했다. 생성 데이터와 바이너리는 규모 및 용도를 확인했으며 전체를 메모리에 적재하지 않았다.

## 우선순위별 발견 사항

### 1. [현재 점유 확인 / 최우선] NAS 가상 드라이브에 다수 파일 직접 생성

- data_collection/config.py:105: 기본 dataset_root가 /mnt/z/DCM/data_collector.
- writer/dataset_writer.py:174–208: 프레임당 measurements, measurements_full, actors_data JSON 및 affordances, 3d_bbs NPY 생성.
- writer/image_writer.py:21–46: 카메라 4개 JPEG 생성.
- writer/lidar_writer.py:24–26: LiDAR NPY 생성.
- sources/native_file_collector.py:444–445, 582: provenance/incomplete JSONL 추가 기록.

기본 native 모드에서 프레임당 기본 파일은 10개다. 실제 처리량이 10프레임/초라면 1시간에 36만 개 생성된다. 시뮬레이션 시간의 10Hz와 실제 처리량은 구분해야 한다. C:의 원본 5개 파일은 별도다.

RaiDrive는 파일 읽기·쓰기·목록 조회 시 파일 정보를 메모리에 보관한다고 개발사가 설명했다. 현재 버전에서 동일한 세부 구현이라는 보장은 없지만 실제 21GB Working Set과 함께 가장 우선적으로 조사할 경로다.
https://with.raidrive.com/t/memory-consideration-when-a-3s-bucket-contains-100s-of-millions-of-objects/5317
https://with.raidrive.com/t/raidrive-is-using-ram-memory/2837

권장: 결과를 로컬 SSD에 먼저 확정 저장하고 수집 후 제한된 병렬도로 전송. 가능하면 NAS의 직접 SMB 또는 별도 전송 도구를 비교. 최종 사용 형식이 허용하면 아카이브 단위 전송으로 파일별 연산을 줄일 수 있다. RaiDrive를 사용한 채 단순히 로컬 파일 수십만 개를 한꺼번에 복사하는 것만으로 문제가 보장되게 해결되지는 않는다. 데이터 전송 완료를 확인하지 않고 RaiDrive 프로세스를 강제 종료하지 않는다.

### 2. [코드 확인 / 높은 위험] 실제 rosbridge 수신 큐가 무제한

- expert/src/rosbridge_suite/rosbridge_server/src/rosbridge_server/autobahn_websocket.py:79: deque()에 maxlen이 없다.
- 동일 파일:95: 모든 수신 메시지를 append.
- 동일 파일:194–195, 242–245: WebSocket 연결에서 실제로 이 큐를 생성하고 사용.
- devel/lib/rosbridge_server/rosbridge_websocket.py는 이 소스를 실행하는 relay다.
- rosbridge_websocket.launch:10,12: fragment_timeout=600, max_message_size=None.

들어오는 속도가 파싱/ROS 전달보다 빠르면 메시지 바이트가 계속 쌓일 수 있다. 대형 카메라/LiDAR 메시지가 ROS로도 전송되는 경우 위험이 커지지만 실제 센서 전송 설정·큐 길이는 아직 측정하지 않았다. native Collector 자체는 ROS 카메라 구독을 시작하지 않는다.

권장: 총 대기 바이트/개수 상한, 큐 지연 계측, 전송 측 제어를 추가. SyncMode 서비스 요청/응답을 무작위로 버리면 교착될 수 있으므로 단순 deque(maxlen=...) 변경으로 끝내면 안 된다. native 파일 수집에서 불필요한 영상/포인트클라우드 ROS 전송이 있는지도 확인한다.

### 3. [코드 확인] 모든 프레임 메타데이터를 메모리에 영구 누적

- data_collection/writer/dataset_writer.py:65: 전체 프레임 리스트 생성/재개 로딩.
- 동일 파일:210: 매 프레임 merged_measurement를 _all_frames에 append.
- 동일 파일:216: 마지막에 measurements_all.json으로 다시 기록.

이미지 전체를 이 리스트에 보관하는 것은 아니다. 메타데이터만 누적하지만 객체 수·수집 시간에 따라 제한 없이 증가한다. 재개 시 기존 measurements_full JSON을 모두 다시 읽는다. 단일 120초 수집으로 곧바로 20GB가 된다는 증거는 없다.

권장: JSONL 스트리밍 또는 기존 프레임 파일에서 종료 후 스트리밍 집계. 재개 시 인덱스와 마지막 완료 프레임만 보관.

### 4. [코드 확인] manifest를 한꺼번에 읽고 모든 JSON 객체로 변환

- data_collection/sources/sync_capture_manifest.py:47: read()에 크기 제한 없음.
- 동일 파일:51–73: 전체 문자열 split 및 전체 records 리스트 생성.

Collector가 늦게 시작하거나 NAS 대기로 뒤처지면 파일에 밀린 메타데이터를 한 번에 Python 객체로 변환한다. 문자열, 줄 목록, dict가 한동안 함께 살아 메모리 순간 사용량이 증가한다. 즉시 센서 전체를 읽는 구조는 아니지만 오래된 큰 manifest에서 위험하다.

권장: 바이트/레코드 수를 제한한 배치 읽기, 미완성 한 줄만 보관, 재개 offset 저장.

### 5. [코드 확인] 생산자에 소비 속도·자원 압박이 전달되지 않음

- sync_runtime/main.py:351–410: 다음 tick 및 캡처를 RPC 완료 즉시 반복.
- sync_runtime/config.yaml:6: capture_enabled=true.
- data_collection/sources/native_file_collector.py:384–388: block_when_full=True는 Collector만 멈춘다.
- Collector 종료·오류가 Runtime 캡처를 중단하지 않음.

NAS가 느려져도 C: 원본과 manifest가 계속 쌓인다. 이는 먼저 디스크 증가와 처리 지연 문제이며, 그 자체가 모든 이미지를 Runtime RAM에 보관한다는 뜻은 아니다. Windows의 MORAI 저장 대기와 RaiDrive 캐시는 외부 프로세스라 별도 계측이 필요하다.

권장: RAM/가용 디스크/작성 지연/소비 지연을 관측하는 보호 장치 및 생산자-소비자 동기 제어. 원본 디스크 누적량과 보관 정책도 필요하다.

### 6. [코드 확인] 이미지 중복 디코딩과 파일 쓰기 성공 검증 누락

- sources/morai_sync_file_source.py:183–186: PNG 디코딩 후 결과를 버리고 encoded bytes만 전달.
- writer/image_writer.py:36–41: 동일 바이트를 다시 디코딩해 JPEG로 저장.
- 동일 파일:41,46: cv2.imwrite 반환값을 검사하지 않음.

불필요한 CPU와 순간 메모리 할당이 있다. NAS 압박 시 저장 실패가 False로 반환돼도 일부 실패는 검출하지 못한다. 개선 시 decoded 이미지를 큐에 넣으면 압축 bytes보다 큐 RAM이 커질 수 있으므로 큐의 바이트 상한도 함께 설계해야 한다.

### 7. [코드 확인] 부분 저장과 완료 수 오표기, 종료 대기 제한 없음

- writer/async_frame_writer.py:74–85: 메타데이터 → 이미지 → LiDAR 순으로 기록.
- sources/native_file_collector.py:582–589: submit 후 provenance와 sample_index 증가.
- 동일 파일:629–634: 작성 실패가 있어도 먼저 done saved=... 출력.
- writer/async_frame_writer.py:57–60: queue.join/thread.join에 종료 시간 상한 없음.

한 센서 저장 실패가 Collector 전체 오류로 이어지고 부분 프레임이 남는다. NAS I/O가 멈추면 close가 장시간 대기할 수 있다. 이는 메모리 누수 자체와 구분해야 한다.

## 상한이 확인된 부분 / 우선순위 낮은 부분

- AsyncFrameWriter: 기본 32개 프레임 큐. 무제한 큐는 아니다. 기존 원본 표본 약 3.2MB/프레임이면 payload만 대략 103MB에 메타데이터/처리 중 프레임이 더해진다. 해상도·LiDAR 크기에는 별도 바이트 상한이 없다.
- CaptureJournal: 기본 1,024개 메타데이터 큐, 가득 차면 드롭. 영상 큐가 아니다.
- GrpcStateHistory: 최대 2,000 tick, 오래된 항목 제거. 객체 수별 RAM은 달라진다. 5ms polling은 최대 이론상 200Hz이며 RPC 시간 때문에 실제 속도는 그 이하.
- Expert 객체/신호 히스토리: deque(maxlen=200). 고속도로 debug 리스트는 주기적으로 초기화.
- Runtime 통계: 누적 숫자와 구간 합계만 보관.
- Scenario Runner: 하나의 controller 객체 안에서는 중복 Expert 시작 방지, 종료 시 프로세스 그룹 정리. 별도 터미널의 중복 인스턴스까지 전역적으로 막지는 않는다.
- 지도: 시작 시 전체 JSON 및 공간 인덱스 로딩, 프로세스별 중복. 큰/손상된 좌표의 공간 인덱스 팽창은 잠재 위험이나 현재 맵 때문에 RAM 100%가 됐다는 증거 없음.
- LMDrive/VLDrive/LAVIS/vision_encoder 학습 코드: 모델 로딩·DataLoader 등 자원 집약 경로가 존재하지만 현재 세 실행 명령에서 학습을 시작하는 경로는 발견되지 않음. 저장소에 있다는 이유로 모델이 RAM에 올라가지 않는다.
- data_collection/tools의 일부 검증/후처리는 모든 JSON을 리스트로 읽음. 실시간 Collector와 별개이며 동시에 실행할 경우 추가 메모리 부하.
- .runtime 로그/데이터, build/devel, 외부 패키지는 디스크 용량과 인덱싱 비용 대상. VS Code가 실제로 어떤 폴더를 인덱싱했는지는 측정하지 않았음.

## 조치 우선순위

1. 전송 대기 여부 확인 후 RaiDrive 점유를 정상화하고 Windows 전체 메모리 시계열 확보. 강제 종료/캐시 삭제는 수행하지 않았다.
2. native 출력 로컬화 및 NAS 전송 분리. 동일 작업의 RaiDrive private/working set 변화 비교.
3. rosbridge 수신 대기 바이트 제한/계측 및 불필요한 대형 ROS 전송 확인.
4. manifest 제한 배치 읽기와 _all_frames 제거.
5. 프레임 완료 후 commit, 실패 기록, 정확한 saved 집계, 쓰기 실패 검사.
6. 수집 재개는 과부하 보호·모니터링과 함께 짧은 비교 시험부터 수행.

이번 감사에서는 실행 로직을 수정하지 않았고 대량 수집/부하 시험도 시작하지 않았다. 현재 RaiDrive 점유를 확인한 뒤 NAS 전체 디렉터리 순회도 하지 않았다. 이전의 순수 Python 충돌은 관측 사실로 유지하지만, 그 원인을 CPU 또는 WSL로 확정한 것으로 해석하면 안 된다.