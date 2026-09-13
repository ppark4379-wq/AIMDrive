# ef7c6db 검토 및 실행 환경 격리 — 2026-09-10

대상: ef7c6dbf785a1b62f1f2c79831bc7bc83c869cb9의 7개 변경 파일 전체.
두 번째 PC의 사용자 설명은 아래 추가했다. 정확한 Stop code/덤프는 미제공이므로 동일 원인이라고 단정하지 않는다.

## 커밋 검토

- Expert, Runtime, Scenario Runner, 의존성/설치 스크립트/바이너리/hook 변경 없음.
  문서 추가 명령은 Collector 실행뿐이며 시스템 변경/다운로드 명령은 없음.
  새 코드에서 다른 프로세스 메모리나 Expert의 Point를 직접 수정하는 경로는 발견되지 않음.
- 확정 결함: collect.py의 --label-spool-dir 기본값은 빈 문자열이고 native_file_collector.py는 필수로
  요구하지만 config.py에 필드를 추가하지 않음. README는 존재하지 않는 기본 경로를 안내함.
  사용자의 초기 스풀 경로 오류와 직접 일치. 현재 로컬 기본값은 이전 수정으로 /mnt/c/DCM/label_spool.
- README 예시는 3600초/Z: 직접 쓰기/다른 PC의 사용자 경로를 사용함. 현재 로컬 후 복사 흐름과
  충돌하는 안내는 이번에 정정함. 문서를 받은 것 자체가 명령을 실행하는 것은 아님.
- label_spool.py:137,164의 _unavailable set은 누락 tick을 계속 add하며 상한/삭제 없음.
  장기 누락 시 RAM 누적 가능. 숫자 집합이므로 이것만으로 수초 내 25GB를 설명할 근거는 없음.
- label_spool.py:169의 추가 manifest reader가 기존 sync_capture_manifest.py:47의 무제한 read()를 사용.
  큰 과거 manifest에 대해 두 소비자가 각각 전체 읽기/파싱할 수 있음. read() 구현 자체는 기존 코드.
- label_spool.py:83은 라벨마다 압축/SQLite commit하며 DB 데이터는 계속 보관됨.
  디스크 용량/정리 정책 없음. 기존 gRPC history를 감싼 것이며 reader를 두 개 시작하는 구조는 아님.
  메모리 history의 2000개 상한은 기존대로 유지됨.
- close()의 제한 시간 join 후 thread 종료 확인이 없는 점은 종료 경합 검토 대상.
  이번 실행에서 이 경합을 재현하거나 Expert 오류와 연결하지 않았음.

## 실제 격리 검사

순수 Python probe를 프로젝트 밖 /tmp/morai-runtime-isolation/probe.py로 복사.
-I -S 및 PYTHONPATH/PYTHONHOME/LD_PRELOAD/LD_LIBRARY_PATH 제거.
ROS/NumPy/gRPC/SQLite/repo 코드 import 없음. 센서/지도/NAS 읽기나 쓰기 없음.
작은 65개 좌표를 20,000회 투영하는 동일 파일을 각각 3회 실행.

| 실행 환경 | 결과 |
|---|---|
| WSL 시스템 Python 3.8.10 GCC | 3/3 실패 |
| WSL 독립 Python 3.8.20 Clang | 3/3 통과 |
| Windows 공식 Python 3.8.10 | 3/3 통과 |
| Windows Python 3.11.9 | 3/3 통과 |

실패 예: 숫자 계산에서 enumerate + float, float * LaneMap, tuple 해체에서 non-iterable float.
독립 3.8.20도 이전 긴 지도 검사에서는 실패한 적이 있어 이번 짧은 통과는 해결 보장이 아님.
Python 버전/컴파일러/코드 배치/OS/라이브러리/CPU 스케줄링 차이가 남아 있음.

가상 CPU 0,8,16,24 검사와 0,8,24 교차 반복에서 통과/실패가 섞임. SIGSEGV도 기록됨.
같은 가상 CPU에서도 결과가 달랐으며 특정 물리 코어 고장 판정이 아님.

## 실행 파일·자원 상태

- dpkg -V python3.8-minimal libpython3.8-minimal libpython3.8-stdlib libc6: 출력 없음, exit 0.
  패키지 등록 체크섬 기준 변조는 발견되지 않음. 실행 중 메모리/하드웨어 검증과는 다름.
- ldd 직접 의존성은 시스템 libc/pthread/dl/util/m/expat/z 및 로더.
- 확인 당시 Collector/Expert/Runtime/Runner 프로세스 없음. rosbridge와 Simulator는 남아 있음.
- WSL RAM 약 861MiB/15.5GiB, swap 0. Windows 가용 RAM 약 12GiB.
- RaiDrive private 약 25,002MiB, working set 약 444MiB. 과거 WS 21GiB와 구분해야 함.
- i9-14900KF, BIOS FG(2023-09-12), WSL kernel 6.18.33.1. 하드웨어 원인 확정 근거는 아님.

현재 타입 오류는 수집 큐/NAS 쓰기/새 스풀 코드가 실행 중일 필요 없이 재현됨.
다만 과거 수집이 지속되는 호스트/게스트 문제를 유발했는지는 재부팅 전후 기준선 없이 배제하지 않음.

## 다음 비교

WSL 재시작 직후 ROS/시뮬 연결/수집 전에 같은 -I -S 검사를 수행해야 현재 세션 문제를 더 분리할 수 있음.
재시작은 rosbridge와 WSL 터미널 작업을 중단하므로 아직 하지 않았음.
깨끗한 기준선이 통과하면 ROS → Runtime/Expert → 로컬 Collector → NAS 복사를 단계별 비교.
첫 실패 단계에서 ef7c6db 부모와 해당 커밋을 별도 디렉터리로 비교하며 기존 작업은 보존.
두 번째 PC도 같은 probe와 정확한 오류/CPU/WSL 정보를 확인해야 함.

원시 결과: expert/.runtime/isolation_ef7c6db_results.json,
isolation_cpu_affinity.json, isolation_cpu_repeat.json.
이번 조사에서 시스템/주행/Collector 실행 코드는 변경하지 않음. README를 정정하고 진단 파일만 추가.

## 두 번째 PC 추가 정보 — 사용자 진술

- CPU: Intel Core Ultra 7 265K. 현재 조사 PC의 i9-14900KF와 다르다.
- 수집 중 Windows가 종료되다가 C: 공간을 정리한 후 그 증상은 사라졌다.
- 이후 수집 완료 후 NAS 전송 중 VS Code 정지와 Windows 오류 화면/진행률/재부팅 발생.
- NTFS/실행 파일 등 표시 내용이 달랐으나 정확한 Stop code 및 모듈명은 기억하지 못함.
- 이는 설명상 Windows bugcheck/블루스크린에 부합하지만 dump로 확인해야 한다.
  현재 PC의 Python 사용자 프로세스 예외와 동일 계층의 오류라고 취급하지 않는다.
- 저장 공간/전송 부하/가상 드라이브 경로가 유력한 조사 단서이며, 특정 드라이버나
  물리 메모리 손상으로 확정하지 않는다. C: 후 Z: 구조도 NAS 전송 단계 자체는 남긴다.
- 두 번째 PC의 C:\Windows\Minidump\*.dmp 및 Stop code가 다음 증거다.
  Microsoft 문서상 덤프에는 Stop code, 드라이버 목록과 커널 스택이 포함된다:
  https://learn.microsoft.com/en-us/troubleshoot/windows-client/performance/read-small-memory-dump-file
- 현재 조사 PC는 C: 여유 약 476.6GB(10진 단위). 최근 3일 System 로그에서 조회한
  BugCheck/Kernel-Power/NTFS/disk/WHEA 대상 ID에 맞는 항목은 반환되지 않았다.
  이는 두 번째 PC의 기록이 아니며, 현재 PC의 모든 과거 오류 부재를 뜻하지 않는다.

## 사용자 승인 WSL 재시작 후 검사

wsl --shutdown 실행 후 Ubuntu-20.04를 다시 시작했다. 초기 접속 1회는
Wsl/Service/E_UNEXPECTED를 반환했고, 이후 /tmp를 시작 디렉터리로 지정한 접속은 정상 동작했다.
boot ID가 5cfd8d3e-2608-4218-8012-67c1b1f4f358에서 73477c03-ac51-4d7a-b330-d0f3c7b5129a로 변경되어 새 커널 부팅을 확인했다.
검사 전후 ROS master/rosbridge/Expert/Runner/Collector/Sync Runtime 프로세스가 없음을 확인했다.

같은 SHA-256(6f15b1e8f4bde68f10b376c0768759a23c47548ff62df5bb4a48c1aad0dbaaf4)의 probe를 프로젝트 밖에 복사하고, 동일 -I -S 및 환경 변수 제거 조건으로
각 실행 파일당 20,000회 계산 시도를 3번 반복했다. 아래 숫자는 프로세스 종료 코드다.

| 환경 | 재시작 전 | 재시작 후 |
|---|---|---|
| wsl_system_3.8.10 | [1, 1, 1] | [139, 0, 139] |
| wsl_independent_3.8.20 | [0, 0, 0] | [0, 0, 0] |
| windows_3.8.10 | [0, 0, 0] | [0, 0, 0] |
| windows_3.11.9 | [0, 0, 0] | [0, 0, 0] |

시스템 Python의 재시작 후 139는 faulthandler가 기록한 SIGSEGV이며 project()의
zip/enumerate 루프(27줄)에서 발생했다. 두 번의 실패와 한 번의 통과가 섞였다.
WSL RAM 약 588MiB/15.5GiB, swap 사용 0이었다. 패키지 체크섬 검사에서도 차이는 출력되지 않았다.

따라서 새 WSL 부팅 직후에도 이 시스템 Python 실행의 문제가 재현된다.
실행 중인 ROS/Collector/새 스풀/NAS 쓰기는 이 재현에 필요한 조건이 아니다.
다만 Windows 호스트와 RaiDrive/Simulator는 재시작하지 않았으며,
디스크에 설치된 실행 파일/라이브러리/WSL 버전/하드웨어 상태는 이전과 같다.
이 결과만으로 특정 Python 버전 또는 하드웨어를 원인으로 확정하지 않는다.
독립 Python의 짧은 통과도 이전 긴 검사 실패를 무효화하지 않는다.

현재 WSL은 실행 중이고 ROS/Collector는 다시 시작하지 않았다. 시스템 Python·의존성·주행 코드는 변경하지 않았다.
원시 자료: expert/.runtime/before_wsl_restart.json, after_wsl_restart.json,
isolation_after_wsl_restart_results.json. 재시작 전 자료는 덮어쓰지 않고 보존했다.

## 2026-09-11 현재 PC 실행 대응

사용자 확인: realtime Expert의 같은 배열 비교 오류는 현재 i9 PC에서 발생한 것.
추가 python3 main.py 실행은 expert/.runtime/path.csv 누락으로 초기화 전에 실패했다.

- 누락된 기본 runtime CSV를 bundled 공식 경로에서 생성하도록 config/path_file.py 추가.
  기존 파일과 명시적 GT_BEV_PATH_CSV는 유지. 명시한 파일이 없으면 오류를 숨기지 않음.
  현재 실제 파일은 4430개 좌표로 생성했다.
- expert/run_isolated_python.sh 추가: 기존 독립 Python 3.8.20 사용, ROS/catkin 환경 로드,
  user site 제외 및 자식 Expert에도 PYTHONNOUSERSITE 전달. 시스템 Python 교체 없음.
- --no-sync-runtime 추가: 기본 동기 handshake는 유지하며, 명시한 실행에서 READY를
  올리지 않고 실시간 주행 단계로 진입. ROS/rosbridge 및 MORAI realtime 상태는 필요.
- 독립 Python에서 고속도로 테스트 60개, 경로 테스트 4개, runtime lifecycle 테스트 12개 통과.
  실제 AutonomousDriving.execute 시작 구간을 realtime 150회 + sync 150회 오프라인 재생 통과.
  ROS import와 runner --check 통과. 시뮬레이터에 제어를 보내는 실제 주행은 아직 시험하지 않음.

이 대응은 FileNotFoundError 수정과 검증된 단기 실행 대안이며, 하드웨어/WSL의 근본 문제 해결 선언이 아님.
