# 검증 후 NAS 복사

## 로컬과 NAS의 공통 구조

02 수집 중에는 기존의 검증된 로컬 spool을 사용한다. 03 검증과 학습 manifest 준비가
성공하면 같은 로컬 Windows 드라이브의 `dataset` 아래에 센서 원본과 토픽/GT를 합친
정규 구조를 만든다. 대용량 원본은 NTFS 하드링크로 연결하므로 바이트를 중복 저장하지
않는다. 원래 spool과 source timestamp는 변경하지 않는다.

    <local storage>\dataset\
      static_obstacle\
        static_obstacle_lap100_20261001_123456\
          static_obstacle_lap100_20261001_123456_lap_001\
            CAMERA_1 ... CAMERA_4, LIDAR_5, GPS_6, IMU_7, SESSION.json
            episode_000, ros_logs, software_snapshot, native_setup.json, ...
            static_obstacle_lap100_..._native_session_apply.json
            static_obstacle_lap100_..._native_session_plan.json
            .LOCAL_DATASET_COMPLETE.json
          static_obstacle_lap100_20261001_123456_lap_002\
            ...
          _campaign\
            metadata\campaign_plan.json, runs.tsv, COLLECTION_COMPLETE.json
            validations\validation_<시각>\report.json, training_dataset_v1, ...


`campaign_id`를 생략하면 02번이 `<시나리오 폴더>_lap<요청 회차>_<YYYYMMDD_HHMMSS>`
형식으로 자동 생성한다. 예를 들어 정적 장애물 100회는
`static_obstacle_lap100_20261001_123456`이 된다. 로컬 dataset과 NAS가 같은 이름을
사용한다. 필요할 때만 두 번째 위치 인자로 명시적 campaign ID를 줄 수 있다.
원본 수집 ID를 변경하면 검증 경로와 seal이 깨지는 과거 campaign은
`dataset_campaign_id`를 사용해 원본 ID를 보존하면서 local dataset과 NAS에만 새 규칙의
공개 이름을 적용한다.


시나리오 폴더는 다음처럼 고정한다.

- `full_loop` → `full_route`
- `full_loop_normal` → `basic_driving`
- `full_loop_dynamic` → `dynamic_obstacle`
- `full_loop_static` → `static_obstacle`
- `full_loop_roundabout` → `roundabout`
- `full_loop_highway` → `highway`
- 전용 신호등 모드가 추가되면 `traffic_light`

NAS 자동 전송 PC에서는 위 정규 구조를 그대로
`<NAS root>\<scenario>\<campaign>\<run>`에 복사한다. 로컬과 NAS의 상대경로가 같으며,
NAS에서 파일을 다시 합치지 않는다. 각 run 및 campaign evidence를 별도 재개 단위로
복사하고 크기·SHA-256·전체 파일 수를 검증한다.

## 실행과 설정

MoraiDrive 디렉터리에서 기존 명령을 실행한다.

    ./data_collection/tools/03_validate_internal_sensor_record.py campaign_01

NAS 루트 우선순위:
1. --nas-root 'Z:\DCM\dataset'
2. MORAI_NAS_ROOT 환경 변수
3. .runtime/data_collection/local_machine.json의 nas_root

`nas_auto_publish=true`인 PC에서만 03 성공 후 자동 전송한다. 다른 PC에서는 NAS 설정이
없어도 로컬 정규 dataset을 만든 뒤 정상 종료한다. 팀원 수집 PC의 일회성 설정 예시는:

    ./data_collection/tools/00_prepare_internal_sensor_record.sh \
      --nas-root 'Z:\DCM\dataset' --nas-auto-publish

내 PC처럼 로컬에만 보존할 PC의 설정은:

    ./data_collection/tools/00_prepare_internal_sensor_record.sh \
      --no-nas-auto-publish

다른 PC에서는 실제 연결된 Windows 드라이브 또는 UNC 하위 폴더를 지정한다.
Z 드라이브는 Windows 작업자를 실행하는 사용자에게 연결되어 있어야 한다.
드라이브 연결·자격 증명·드라이버 설정은 코드가 변경하지 않는다.

    # 로컬 검증과 정규 dataset 생성까지만
    ./data_collection/tools/03_validate_internal_sensor_record.py campaign_01 --no-nas-copy

    # 로컬 검증 후 파일 목록과 목적지만 준비; NAS 접근 없음
    ./data_collection/tools/03_validate_internal_sensor_record.py campaign_01 --nas-plan-only

    # 출력됐던 성공 검증 보고서로 전송 재개
    ./data_collection/tools/03_validate_internal_sensor_record.py campaign_01 \
      --nas-only /mnt/c/MoraiDriveData/sensor_record_auto_v1/campaigns/<campaign>/validation_<시각>/report.json

검증 보고서, campaign plan, runs.tsv, COLLECTION_COMPLETE.json의 해시를 확인한다.
전송 실패 시 종료 코드는 4이며 로컬 검증 결과와 원본은 유지된다.
재검증으로 파일이 바뀌면 기존 NAS 임시 폴더와 충돌할 수 있으므로 전송 재시도에는 --nas-only를 사용한다.
기존 목적지가 다른 snapshot이면 덮어쓰지 않고 중단한다.

## 전송 방식: 제한 없는 일반 복사

로컬 저장과 03 검증이 끝난 데이터를 Windows 작업자가 Z 드라이브로 복사한다.
전송 속도는 운영체제·연결 방식·NAS의 실제 처리 속도에 따른다.

- 바이트 처리량, 파일 처리 빈도, 목록 조회 빈도의 인위적인 제한이 없다.
- CPU, RAM, Windows commit, RaiDrive 메모리를 관측하거나 이에 따라 감속·중지하지 않는다.
- 저장 공간 여유 비율을 예약하거나 주기적으로 검사하지 않는다. 공간 부족은 실제 I/O 오류로 처리한다.
- 프로세스 우선순위를 낮추지 않는다. 초기 저속 운전과 단계적 증속도 없다.
- 일반 파일 스트림으로 복사하고, 파일 쓰기가 끝나면 flush 후 이름을 확정한다.
- 파일별 I/O 실패는 별도 대기나 감속 없이 최대 3회 시도한다. 계속 실패하면 중단한다.
- 로그와 status.json에 transfer_mode=unthrottled, throttling_enabled=false를 기록한다.

파일은 한 개씩 순서대로 읽고 쓰며 1 MiB 버퍼를 사용한다.
이는 파일을 처리하는 방식이며 초당 속도나 전체 RAM 사용률을 제한하지 않는다.
목록은 SQLite로 정렬·경로 충돌을 검사한다.
중복 전송을 막는 Windows 전역 잠금과 같은 storage_root의 수집·검증·학습 잠금은 유지한다.
부모 생존 신호가 20초 끊기거나 작업 진행 신호가 120초 없으면 작업자를 종료한다.
이 감시는 전송 취소·고아 프로세스 정리를 위한 것이며 자원 상태에 따른 제어는 하지 않는다.

## 현재 전송에 코드 변경 적용

실행 중인 Windows 작업자는 시작 시 복제한 스크립트를 사용한다.
코드 파일을 수정해도 해당 작업자에 즉시 반영되지 않는다.
기존 전송 부모를 정상 중단하고 작업자 종료를 확인한 뒤,
같은 성공 보고서로 --nas-only를 실행하면 완료 기록을 재사용해 새 코드로 재개한다.
중단 전 복사 완료 기록과 로컬 원본은 보존하며, 미완료 파일은 다시 복사한다.

## 복사·검증·실패 규칙

1. 로컬 검증이 성공한 campaign만 대상이다. 원본을 이동하거나 삭제하지 않는다.
2. 저장 공간 예약 제한 없이 복사를 시작한다. 공간 부족·연결 끊김 등 I/O 오류가 계속되면 중단한다.
3. .nas-pending-<lap>에 파일 하나씩 임시 이름으로 쓴다. 복사하면서 SHA-256을 계산한다.
4. 중간 실패 시 임시 폴더와 로컬 파일별 기록을 보존한다.
   재개할 때 원본 해시를 다시 확인하고, NAS 파일은 최종 전체 재읽기로 검증한다.
5. 모든 파일의 크기·SHA-256, 전체 파일 수·바이트 수를 확인한다. 여분 파일도 오류다.
6. 일치하면 최종 lap 이름으로 바꾸고 .NAS_TRANSFER_COMPLETE.json에 완료 기록을 쓴다.
   소비자는 폴더 존재만 보지 말고 완료 기록을 확인해야 한다.
7. 최종 폴더에 이미 있는 데이터가 변경된 경우 자동 덮어쓰지 않는다.
   완료 기록은 해당 기록 시각의 검증 결과이며 이후 손상을 계속 감시하는 서비스는 아니다.

상태·telemetry·파일별 재개 기록은 로컬 storage_root/sensor_record_auto_v1/.nas_transfers 아래에 남는다.
심볼릭 링크·Windows junction·경로 이탈·대소문자 충돌은 거부한다.
미완료 임시 폴더의 자동 삭제나 원본 자동 삭제 기능은 없다.

## 안전성의 범위

현재 전송에는 부하 제한이 없으며, 검증은 파일 내용의 일치 여부를 확인한다.
이 변경은 기존 커널 손상 원인을 해결하거나 블루스크린 방지를 보장하지 않는다.
또한 Windows/RaiDrive 캐시를 통해 다시 읽은 데이터가 일치해도 NAS 물리 디스크의 영구 저장까지 입증하지 않는다.
따라서 완료 기록은 verification_scope=client_readback, remote_durability_verified=false이다.
NAS 서버 측 해시 검증·영구 저장 확인은 별도 서버 기능이 필요하다.

검증은 Python 목록/경로 테스트와 Windows 임시 로컬 파일 전송 테스트로 수행한다.
실제 Z 드라이브 대용량 전송이나 부하 시험은 자동 테스트에 포함하지 않는다.

## 진행 및 대기 원인 로그

콘솔에는 checked=처리한 파일/전체, copied=이번 실행에서 복사한 파일,
resumed=기존 기록을 확인해 건너뛴 파일, verified=NAS 재읽기 검증 파일을 표시한다.
재시작 시 checked는 0부터 기존 기록을 확인하며 증가하므로, 기존 receipt 총수와 구분한다.
최종 완료는 copy_and_readback_verified로 판단한다.

파일 I/O가 10초 이상 대기하면 별도 감시 스레드가 작업 종류, 정확한 경로와
대기 시간을 출력한다. 아래 기록은 각 작업의 로컬 상태 디렉터리에 저장한다.

- operation.json: 약 5초마다 현재 작업·파일·처리 카운터·대기 시간.
- io_events.jsonl: 1초 이상 걸린 작업과 발생한 예외.
- watchdog_failure.json: 부모 신호 소실/20초 지연, 120초 무진행 등 실제 감시 종료 사유.
- worker_exit.json: 부모가 확인한 종료 코드, 마지막 상태 및 해당 실행의 감시 실패 내용.
- telemetry.jsonl / status.json: 단계와 처리·복사·재개·검증 카운터.

기존 watchdog 실패 기록은 재시작할 때 별도 이름으로 보존한다.
이 기록은 전송 진단용이며 CPU/RAM 상태에 따른 감속이나 전송 속도 제한을 추가하지 않는다.

마지막 I/O 반환 직후 감시기가 종료하면 operation=idle로 기록될 수 있다.
이때 last_completed_operation/path/elapsed_ms/error에 마지막 완료 호출을 보존하고,
부모 오류 메시지도 이를 "last_completed_operation"으로 표시한다.
완료 호출의 duration은 종료 시점까지 계속 대기했다는 뜻이 아니다.

NAS 정체 진단에는 ETW/WPR 이벤트 및 자원 관측을 사용한다.
실행 중인 전송 프로세스의 자동 MiniDumpWriteDump 수집은 사용하지 않는다.
해당 방식이 대상 스레드를 일시 정지시켜 진행 신호와 종료 시점 관측에 간섭한 사례가 확인됐다.

## 상위 경로 조회 감소

Assert-PlainPath는 각 경로 구성 요소에 File.GetAttributes를 한 번만 호출한다.
파일/디렉터리 존재 여부를 먼저 두 번 묻지 않으며, ReparsePoint 플래그로 링크와 junction을 거부한다.
FileNotFoundException과 DirectoryNotFoundException만 생성 가능한 없는 경로로 처리한다.
권한 거부·네트워크·그 밖의 I/O 오류는 그대로 중단 사유로 전달한다.
없는 경로는 예상된 결과이므로 예외 로그를 파일마다 추가하지 않는다.
상위 경로 검사는 매번 유지하며, 오래된 검사 결과를 재사용하는 캐시는 도입하지 않는다.

현재 첫 lap의 18,927개 파일에서는 검증 단계의 명시적 경로 조회가
758,166회에서 265,340회로 약 65% 줄어든다.
이는 코드의 API 호출 수 계산이며, NAS 네트워크 요청 수나 소요 시간이 65% 감소한다는 보장은 아니다.
실행 중인 작업자는 시작 시 스크립트를 복제하므로 재개 실행부터 적용된다.

### 최종 폴더 이름 변경의 잠금 오류

모든 파일의 해시·개수·크기 검증 후 상태를 publishing으로 전환합니다.
폴더 이름 변경에서 Windows 공유/잠금 오류(32/33)가 발생하면 2·4·8·16초 후 재시도하며,
최대 5번 시도합니다. 이는 오류 복구 대기이며 정상 복사·검증의 처리량 제한은 아닙니다.
권한 거부, 대상 경로 충돌 등 다른 오류는 즉시 실패합니다. 끝내 실패하면 임시 폴더와
로컬 원본을 보존하고 완료 기록을 만들지 않습니다. 드라이브 재연결은 코드가 자동 수행하지 않습니다.

2026-09-16 lap 001의 최종 이름 변경 실패는 RaiDrive FSENGINE/MoveFile에서
File Still Opened After Default Close, OpenCount : 2를 기록했습니다.
기존 전송 프로세스가 종료된 뒤에도 같은 오류가 재현됐고, 관리자 핸들 조회에서는 해당
폴더의 핸들이 발견되지 않았습니다. RaiDrive에서 정상 연결 해제·재연결 후 같은 폴더
변경이 성공했습니다. 로그의 PID는 요청자이며 잠금 소유자를 의미하지 않습니다.
지속되는 공급자 내부 참조에는 제한된 재시도만으로 해결되지 않을 수 있습니다.
