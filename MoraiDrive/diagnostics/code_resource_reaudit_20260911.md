# 공통 코드 실행 후 정지 현상: 코드 자원·실패 처리 재감사

검사일: 2026-09-11. 기준: HEAD `af0b355`와 현재 미커밋 작업 트리.

## 결론과 판정 한계

현재 코드를 무결함으로 판단할 수 없다. 저장 실패 누락, 프레임 완료 집계 순서,
제한 없는 수신 큐·메타데이터 보관, 생산/소비 속도 분리, 전송·디스크 보호 부족을
확인했다. 공통 코드가 두 PC에서 I/O 및 메모리 압력을 유발하는 경로는 실제로 있다.

그러나 아래 작은 재현 검사는 Windows BSOD를 재현한 것이 아니다. 두 PC의
BSOD와 현재 PC Expert의 SIGFPE/SIGSEGV/SIGILL을 하나의 원인으로 묶을 증거는 없다.
시나리오 단독 실행에서도 Expert 충돌이 있었으므로 Collector/NAS 코드만 수정해
전체 장애가 해결된다고 보장할 수 없다. 네이티브 라이브러리 결함도 아직 배제하지 않았다.

이번에는 시뮬레이터·Collector·NAS 전송을 시작하지 않았다. 기존 코드의 작은
오프라인 검사만 수행했다. 아래 결함은 아직 운영 코드에서 수정하지 않았다.

## 1. 높은 우선순위: C: 저장 후 복사에도 전송·디스크 보호가 부족함

- `data_collection/writer/archive.py:39`: 디렉터리 순회 후 파일을 차례로 복사.
- `archive.py:19`: `shutil.copyfile` → 임시 파일 → `os.replace`.
- `archive.py:46`: 수집 결과가 completed인 경우만 실행하고 원본은 보존.
- `data_collection/collect.py:329`, `:970`: 수집/Writer 종료 후 archive 호출.

파일 전체를 메모리에 모아서 병렬 복사하는 구현은 아니다. 다만 순차 복사에도
초당 바이트·파일 개수 제한, RaiDrive 대기량/메모리 감시, 최소 여유 디스크 기준,
NAS의 실제 업로드 완료 확인이 없다. Z: 쓰기 호출이 반환되는 속도만큼 다음 파일을
계속 넘긴다. 이 구조는 수집과 전송의 동시 부하는 줄여도, 전송 단계의 과부하를
제한하지 못한다. 이 구현은 현재 미커밋 변경이다.

C:에는 MORAI 원본과 Collector 결과가 함께 남고, 복사 성공 후에도 결과를 보존한다.
보존 자체는 복구에 필요하지만 사용 공간 상한·보존 정책 없이 반복하면 디스크가
차오른다. 표준 실행 경로에 `disk_usage/statvfs` 기반 저용량 중단 조건은 찾지 못했다.
다른 PC에서 C: 정리 후 증상이 줄었다는 보고와 양립하는 경로이며 인과 확정은 아니다.

기존 원본 측정 `resource_audit_host_snapshot.json`의
2026-09-10 20:37:17+09:00 스냅샷에서는 RaiDrive Working Set 약 **10,771 MiB**,
Private Bytes 약 **24,751 MiB**였다. 당시 값이며 현재 값이나 다른 PC의 값이 아니다.
기존 보고서의 21,536 MiB Working Set은 다른 관측 시점의 값으로, 이 스냅샷과
혼동하지 않는다. Private Bytes 전부가 물리 RAM에 상주한다는 뜻도 아니다.
이 측정만으로 RaiDrive 내부 캐시와 누수를 구분할 수 없다.

## 2. 높은 우선순위: Collector 정지/지연이 원본 생산을 멈추지 않음

- `sync_runtime/config.yaml:6`: capture_enabled=true, 20ms tick/stride 5.
- `sync_runtime/main.py:351`: READY/종료 조건과 tick에 따라 캡처 지속.
- `data_collection/sources/native_file_collector.py:384`: 큐 포화 시
  `block_when_full=True`로 소비자만 대기.
- 같은 파일 `:629`: Collector 중단이 Sync Runtime에 영향을 주지 않는다고 명시.

Collector가 느려지거나 120초 후 끝나도 Runtime이 계속 실행 중이면 원본 캡처는
계속된다. 캡처 주기는 시뮬레이션 시간 기준 10Hz이며 실제 초당 처리량은 RPC 속도에
따라 더 낮을 수 있다. 파일·manifest 증가를 Collector 큐 상한으로 막을 수 없다.

`AsyncFrameWriter` 자체는 기본 **32프레임으로 제한**된다. 무제한 큐로 잘못 분류하지
않는다. 다만 바이트 제한은 없고 실제 카메라 압축 바이트와 LiDAR 배열을 보관한다.
큰 프레임일수록 같은 32개라도 메모리 부담이 커진다.

## 3. 높은 우선순위: ROS bridge 수신 큐의 메모리 상한 없음

- `expert/src/rosbridge_suite/rosbridge_server/src/rosbridge_server/autobahn_websocket.py:79`
  `deque()`에 maxlen 없음. `:95`에서 모든 메시지를 append.
- 파일/센서 저장과 별개로 시나리오 단독 실행에도 사용하는 ROS bridge 경로다.
- 메시지 도착이 파싱/처리보다 빠르면 대기 메시지를 계속 보관할 수 있다.
- 작은 재현: 소비 스레드를 시작하지 않고 1KiB 메시지 1,000개를 넣자 1,000개 전부
  보관했다. 한도/거절/송신측 속도 제어가 없음을 검증한 것이며 실제 실행 중 큐가
  이만큼 밀렸다는 측정은 아니다.
- 부가 결함: finish() 후에도 push()가 메시지를 받아들이는 것을 재현했다.

현재 실행 중이던 PID 3804는 AIMDrive가 아닌
`/home/jun/morai_scenario_runner/third_party/GT_BEV/devel/lib/rosbridge_server/rosbridge_websocket`
에서 기동됐다. `/proc/3804/environ`의 ROS_PACKAGE_PATH와 PYTHONPATH도 외부 GT_BEV를
우선한다. 해당 외부 `autobahn_websocket.py`와 이 저장소 파일의 SHA-256은 일치했다:
`fc83b73f0641edd8887e569e6232e225dedc70aed34ae921af3147d316e215de`.
따라서 현재 큐 구현은 동일하지만 AIMDrive 파일만 수정해도 이미 실행 중인 외부
bridge가 바뀌는 것은 아니다. 관측 당시 bridge RSS는 약 93,564 KiB로, 그 순간의
대규모 메모리 누적은 확인되지 않았다.

수정 시 서비스 응답을 무작정 버리는 deque(maxlen) 방식은 피해야 한다.
메시지 종류별 처리 정책과 총 바이트 상한, 대기량 계측이 함께 필요하다.

## 4. 확인된 메모리 증가 경로: manifest 전체 읽기와 프레임 메타데이터 보관

- `data_collection/sources/sync_capture_manifest.py:47`: 남은 전체를 read()한 뒤
  split하고 모든 레코드를 dict/list로 변환. 한 번에 읽을 바이트/레코드 상한 없음.
- 합성 JSONL **1,337,563 bytes / 2,000레코드**가 한 번의 poll()에 모두 반환됐다.
  tracemalloc 최고치는 **5,639,983 bytes**, 반환 후 보유량은 **2,849,882 bytes**였다.
  이는 이 합성 입력에서의 Python 할당량이며 실제 모든 데이터의 고정 배율은 아니다.
- `data_collection/writer/dataset_writer.py:210`: 프레임마다 `_all_frames.append`.
  디스크 I/O를 모의 처리해 1,000프레임을 쓰자 1,000개 메타데이터가 남았다.
  `:65` 및 `:78`의 resume도 기존 JSON들을 모두 다시 메모리에 읽는다.
- `data_collection/grpc_ctrl/label_spool.py:137`, `:166`: 누락 tick을 보관하는
  `_unavailable` 집합에 제거 정책 없음. SQLite DB도 완료 프레임을 삭제하지 않는다.
  정수 집합이 곧바로 수십 GB를 소비한다고 주장하지 않는다.
- 같은 파일 `:170`의 별도 manifest 독자는 Collector와 독립적으로 읽으므로,
  두 스레드가 큰 밀린 구간을 읽으면 할당이 겹칠 수 있다.

이미지 전체를 `_all_frames`에 저장하는 구조는 아니다. 이 항목만으로 짧은 수집에서
관측한 100% 메모리 점유를 설명했다고 보지 않는다. GrpcStateHistory는 2,000 tick,
Expert의 객체/신호 이력은 200개로 제한됨도 확인했다.

## 5. 실제 재현한 저장 정확성 결함

### 이미지 쓰기 실패 누락

`data_collection/writer/image_writer.py:41`, `:46`은 cv2.imwrite의 반환값을 확인하지
않는다. 모의 imwrite가 False를 반환하도록 하자 이미지 파일은 없는데 write()는
예외 없이 정상 반환했다. 디스크 부족/쓰기 실패를 놓칠 수 있는 코드 결함이다.
이번 검사는 실제 디스크를 채운 것이 아니다.

### 프레임 완료 순서와 로그

`data_collection/writer/async_frame_writer.py:73`은 메타데이터를 먼저 저장하고
그 후 `:83` 이미지, `:85` LiDAR를 저장한다. 메타데이터 작성 후 이미지 오류를
주입하자 frame_count는 1이고 LiDAR는 호출조차 되지 않았다. close()는 오류를
제대로 전달하므로 모든 실패를 성공 처리하는 것은 아니지만 부분 프레임이 남는다.

`native_file_collector.py:616`의 sample_index는 enqueue 후 증가하고 `:657`에서는
이 값을 done saved로 출력한 뒤 `:662`에서 Writer 오류를 다시 던진다.
사용자가 본 `done saved=74` 다음 비동기 쓰기 오류가 나오는 현상과 일치하는 경로다.

`async_frame_writer.py:57`~`:59`의 stop enqueue/queue.join/thread.join에는 종료 시간
제한이 없다. 실제 파일 쓰기가 반환되지 않으면 종료도 계속 기다릴 수 있다.
이는 프로그램 정지 경로이며 곧바로 Windows BSOD의 원인이라는 뜻은 아니다.

원본 PNG는 `sources/morai_sync_file_source.py:183`에서 검증용 디코드 후 버리고,
Writer에서 다시 디코드하여 JPEG로 저장한다. 불필요한 CPU/임시 버퍼 비용은 있지만
이 중복만으로 메모리 손상의 원인이 입증된 것은 아니다.

## 6. 시나리오 단독: Expert 계산 비용과 네이티브 충돌을 별도로 추적해야 함

- `expert/src/expert/src/planning/section_planning.py:239`:
  고속도로 밖인지 알아보기 위해 lane_map.match() 호출.
- `planning/highway/lane_map.py:77`, `:96`:
  거리 컷오프 전에 모든 링크/선분을 검사한다.
- 현재 지도에서 22링크, 5,599선분. 시작 위치 (-131.486,-427.961)에서 5회 실행해
  모두 None(고속도로 밖), CPU 시간 합계 0.071889109초, 평균 **14.38ms/회**였다.
  30회/초라면 이 계산만 약 0.43 CPU초/초에 해당하는 단순 환산이다.
  짧은 단독 측정이며 전체 시스템의 CPU 43%라는 뜻은 아니다.
- static 구간은 별도 `_obb_collision`을 실행하며 사용자의 SIGFPE Python 스택은
  `planning/lattice/lattice_planner.py:339`의 cos/sin 호출을 가리켰다.

이전 gdb 기록은 서로 다른 위치의 SIGSEGV(`_PyObject_Malloc`, 잘못된 주소)와
SIGILL(`PyDict_GetItem`)도 포착했다. 기록은
[sigfpe_static_obstacle_20260911.md](sigfpe_static_obstacle_20260911.md)에 있다.
오류가 보인 Python 줄이 곧 메모리를 처음 손상시킨 줄이라는 증거는 없다.
NumPy가 네이티브 스택에 포함되어 있으므로 Python 코드·확장 라이브러리 문제를
배제하지 않는다. 동시에 CPU/RAM/WSL 중 어느 하나로도 아직 확정하지 않는다.

Point는 NumPy ndarray 하위 클래스이며 입력 형상·유한성 검증이 부족하다.
그러나 지도 입력은 float로 변환되고 ROS Ego 입력은 position.x/y로 들어온다.
잘못된 배열 입력이 실제 오류 프레임에 들어왔다는 증거 없이 이전의 이상한
AttributeError들을 전부 일반 입력 엣지케이스로 설명할 수 없다.

## 커밋과 관계

| 항목 | 확인한 이력 |
| --- | --- |
| bridge 무제한 수신 큐 | `26c3937`에서 저장소에 들어온 코드 |
| manifest 전체 읽기 / PNG Writer 분기 | `cfbf49d` |
| 프레임 메타데이터 append | 해당 줄 blame `cad3e74`; 누적 구조 최초 도입과 구분 |
| 로컬 gRPC 라벨 SQLite / 추가 manifest 독자 | `ef7c6db` |
| 원본 C: 폴더를 Z:에 연결하는 선택 도구 | `007544b` |
| 수집 결과 C: 저장 후 Z: 복사 | 현재 미커밋 `writer/archive.py` 및 호출부 |

`ef7c6db`는 7파일, 420줄 추가/2줄 삭제이며 Expert/ROS bridge/Sync Runtime 계산이나
시스템 설정을 바꾸지 않았다. 라벨마다 SQLite 트랜잭션, 추가 스레드/manifest 읽기,
누락 tick 보관이라는 부하는 추가한다. label_spool_dir 기본값 누락은 Collector를
시작 시 실패시킬 수 있는 별도 결함이었고 현재 작업 트리에서 수정된 상태다.

`git diff ef7c6db..HEAD`에서 네 런타임 영역의 커밋 차이는 data_collection/config.py의
MORAI 경로와 sync_runtime/config.yaml의 센서 매핑뿐이었다. 현재 미커밋 변경은
별도로 읽었다. “그 커밋 이후 실행했다”는 시간 순서만으로 모든 충돌 원인을 해당
커밋에 귀속할 수는 없지만, 공통 코드의 위 결함들은 수정 대상이다.

## 현재 PC에서 제외/미확인한 경로

`data_collection/tools/redirect_native_storage.ps1`은 수동 실행 시 원본 폴더를
Z:\DCM\morai_raw로 연결하고 config를 Z: 경로로 덮어쓴다. 표준 Runner/Collector
진입점에서 이 스크립트를 호출하는 경로는 찾지 못했다.
현재 C:\MoraiLauncher_Win과 SensorData, 카메라·LiDAR SynchronousMode 폴더는
일반 Directory이며 ReparsePoint/SymbolicLink가 아니었다. 현재 PC에는 그 리다이렉트가
적용된 증거가 없다. 다른 PC에 적용됐는지는 검사하지 못했다.

NPC 설정은 회전교차로 6대, 고속도로 SpawnPoint 3개이고 설정상 포인트당
maximum_spawn_vehicle=50, 생성주기 10초이다. 고속도로 진입 전 비활성화 및 생성
재시도/교체 상한이 있어 무제한 생성 루프로 분류하지 않았다. 실제 MORAI 내부의
상한 해석과 차량 회수는 실측하지 않았다. 시작 수초 내 충돌을 이 설정만으로
설명하지 않는다. Expert 전역 경로 발행도 매 제어 tick이 아니라 약 1초 간격이다.

표준 진입점과 연결된 소스의 ctypes/CDLL/memmove, OS 재부팅, 무제한 프로세스 생성,
저용량 보호 등을 검색하고 호출 경로를 읽었다. 사용자 메모리를 직접 덮어쓰거나
Windows 종료를 지시하는 코드는 찾지 못했다. 이 결과는 C/C++ 바이너리 또는
드라이버 내부까지 무결함으로 검증했다는 뜻이 아니다.

## 검증과 수정 순서

재현 도구:

- [audit_resource_paths_20260911.py](audit_resource_paths_20260911.py): 실제 클래스와
  AST로 추출한 IncomingQueue 사용. OpenCV/NumPy 파일 쓰기만 모의 처리.
  작은 manifest·큐·메타데이터·이미지 실패·부분 저장 검사 모두 예상 결함 재현.
- [probe_lane_cost_20260911.py](probe_lane_cost_20260911.py): 실제 지도/NumPy 사용,
  ROS·MORAI 연결 없이 5회만 계산. 15초 외부 timeout 안에서 정상 완료.

우선 (1) 저용량/메모리/전송 압력 계측과 생산 정지 연동,
(2) Z: 복사 속도·파일 작업량 제한, (3) bridge 수신량/바이트 상한,
(4) manifest 배치 제한·메타데이터 스트리밍,
(5) 파일 쓰기 확인 후 완료 집계 순서와 종료 처리 개선이 필요하다.
Expert의 네이티브 충돌은 이와 별개로 동일 입력을 고정한 재현과 실행 라이브러리
격리를 이어가야 한다. 계산 예외를 무시하거나 무한 재시작하는 처리는 해결이 아니다.

Windows BSOD의 정확한 연결은 실제 stop code/매개변수와 커널 덤프로 확인해야 한다.
Microsoft의 [bug check 분석 안내](https://learn.microsoft.com/en-us/windows-hardware/drivers/debugger/blue-screen-data)
역시 이벤트 로그 또는 덤프의 bug check 정보와 WinDbg !analyze를 사용한다.
이번 감사에서는 Driver Verifier나 부하 검사를 켜지 않았다.
