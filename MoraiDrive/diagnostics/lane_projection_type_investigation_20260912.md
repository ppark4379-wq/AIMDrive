# lane_map 타입 이상 격리 조사 — 2026-09-12

## 확인한 결론

Expert/ROS/MORAI/컬렉터를 실행하지 않는 독립 코드에서 타입 이상이 재현됐고, NumPy를 제거한 순수 Python에서도 시스템 Python 3.8.10이 SIGSEGV로 종료됐다. 따라서 애플리케이션의 지도 좌표나 콜백 경쟁만으로 설명되는 문제가 아니다. 하위 실행 환경 이상이 확인됐지만 CPython 빌드/OS/CPU·메모리 중 최종 원인은 아직 미확정이다.

## 실험 결과

1. 실제 Point, 고정된 좌표 두 개의 반복 계산: 반복 index 546365에서 length_sq가 Point 배열로 변함.
2. Expert를 import하지 않는 최소 ndarray subclass: index 8671492에서 dx가 [[0.2961746903376934, 278.2791832251842]]로 변함. 실차 진단의 배열과 일치.
3. float/NumPy scalar/일반 ndarray/Point 각각 100만 회 재검사: 모두 통과. 간헐성 확인; 단기 통과는 해결 증거가 아님.
4. -I -S 격리 실행, NumPy 없는 SimpleNamespace + float 연산: 시스템 Python 3.8.10 최초 실행 SIGSEGV. 추가 2회 중 1회 SIGSEGV, 1회 1천만 반복 통과.
5. 동일 최소 스크립트와 별도 NumPy 1.24.4: 1천만 반복 1회 통과. 이 한 번의 통과로 NumPy 버전 변경을 해결책이라고 판단하지 않는다.
6. 독립 Clang 빌드 Python 3.8.20, NumPy 없는 동일 계산: 1천만 반복 2회 통과. 기존 다른 실험에서 독립 Python의 네이티브 오류도 기록돼 있어 현재 두 번의 통과는 원인 배제 근거가 부족하다.

SIGSEGV 스택 위치는 probe_projection_types.py의 반복 계산이다. Python 예외만 수집하는 Expert 진단은 SIGSEGV 자체의 메모리 상태를 확보하지 못한다.

## 정적 확인

Point.x는 self[0], Point.y는 self[1]이다. 지도는 float 변환 후 새 Point를 생성한다. 조사한 Expert 소스에서 HighwayLaneLink.points 원소의 shape를 바꾸는 경로는 발견하지 못했다. 최소 재현은 한 스레드에서 고정 좌표를 사용하므로 ROS 콜백이나 지도 객체 공유를 필요로 하지 않는다.

## 환경

- 실제 실패 Python: /usr/bin/python3, 3.8.10, 시스템 NumPy 1.17.4.
- CPU: Intel Core i9-14900KF.
- BIOS: FG, 2023-09-12 (Windows CIM 재확인).
- sitecustomize는 Ubuntu apport 예외 훅이며 -I -S 재현은 이를 로드하지 않는다.
- Intel은 해당 세대 데스크톱 안정성 문제에 대해 최신 BIOS(0x12F 이상)와 Intel Default Settings를 권고한다. 현재 증상이 그 결함 때문이라고 확정한 것은 아니다.
- https://www.intel.com/content/www/us/en/support/articles/000102331/processors.html

## 다음 판별

이 최소 스크립트를 다른 정상 PC/호스트에서 같은 Python/NumPy로 비교하고, 현재 호스트는 적합한 BIOS/기본 설정 적용 전후로 비교해야 호스트 문제와 소프트웨어 문제를 더 구분할 수 있다. 이번에는 BIOS/패키지/Expert/러너/Runtime을 변경하지 않았다.

재현 스크립트: diagnostics/probe_projection_types.py
결과: diagnostics/.runtime/projection_types_isolated_20260912.json 및 projection_pure_python_matrix_20260912.json
원본 오류: expert/.runtime/diagnostics/lane-project-350276-wxmoxyp6.json
## 추가 OS·인터프리터 비교

Expert/ROS/NumPy를 import하지 않는 동일한 고정 좌표 산술 스크립트를 -I -S -X faulthandler로 비교했다. 각 정상 실행은 1천만 회 반복을 완료했다.

| 환경 | 이번 반복 결과 |
|---|---|
| WSL2 시스템 Python 3.8.10 / GCC 9.4 | 5회 중 3회 통과, 1회 SIGSEGV, 1회 TypeError |
| WSL2 독립 Python 3.8.20 / Clang 18.1.8 | 5회 모두 통과 |
| Windows Python 3.8.10 / MSVC | 5회 모두 통과 |
| Windows Python 3.11.9 / MSVC | 3회 모두 통과 |

시스템 Python의 TypeError는 반복 5,124,914회째 ratio 계산에서 `range_iterator + float`였다. dx, dy, length_sq는 기록 시점 모두 float였다. 따라서 배열 입력이나 NumPy 없이도 계산 중 비정상 타입이 나타난다는 점이 재확인됐다. SIGSEGV는 반복문 26행으로 보고됐다.

현재 재현이 집중되는 조합은 WSL의 시스템 Python이다. 이것은 해당 Python 빌드 및 실행 환경을 우선 조사할 근거이며, CPython 자체 버그 또는 WSL 커널 결함을 확정한 증거는 아니다. 같은 하드웨어에서 다른 바이너리의 짧은 시험이 통과한 것만으로 CPU/RAM 불안정을 배제할 수도 없다. 이전 독립 Python의 다른 실험에서는 네이티브 오류가 있었으므로 3.8.20 교체를 검증된 해결책으로 제시하지 않는다.

WSL2 커널: 6.18.33.1-microsoft-standard-WSL2. 최근 30일 Windows System의 WHEA-Logger 이벤트 조회 결과 없음. 하드웨어 정상 판정과는 다르다.

최종 원인 판별에는 동일 바이너리의 다른 호스트 비교 또는 별도 Linux 환경/커널 비교 등 대조군이 더 필요하다. 이번에는 Expert/runner/runtime 및 BIOS/OS 설정을 수정하지 않았다.

추가 결과 파일:
- diagnostics/.runtime/projection_wsl_repeat_20260912.json
- diagnostics/.runtime/projection_windows_matrix_20260912.json
- diagnostics/.runtime/projection_windows311_matrix_20260912.json

## 동일 바이너리 이동 묶음 검증

python-repro-20260912.tar.gz에 현재 Python/stdlib/동적 라이브러리/로더를 복사하고 SHA256 목록을 포함했다. 설치 패키지 검사(dpkg -V: Python 및 libc6)는 변경 보고 없이 종료됐다. 이동 묶음을 현재 PC에서 실행한 결과 10회 중 8번째가 SIGSEGV(exit 139), 나머지 9회는 통과했다. 따라서 다른 PC에 전달할 바로 그 묶음에서도 현상이 재현됐다.

로그: diagnostics/.runtime/python-repro-20260912/results-20260912-160239-358435
안내: diagnostics/python_repro_other_pc_20260912.md

커널 로그의 Python 종료 및 레지스터 출력은 비정상 종료 증거이며 특정 CPU/RAM 고장 판정은 아니다. 다른 PC나 네이티브 Linux에서의 비교는 아직 수행하지 않았다. 이 대조 결과 없이 Python 빌드/WSL/CPU/RAM 중 하나로 확정하지 않는다.

## 다른 PC 결과 대조 (2026-09-12)

사용자가 가져온 E:\python-repro-20260912\python-repro-20260912\results-20260912-161450-37419의 summary 및 20개 실제 JSON 로그를 확인했다. 20회 모두 status=pass, 각 iterations=10000000으로 총 2억 회 계산을 완료했다.

| 항목 | 현재 PC | 다른 PC |
|---|---|---|
| CPU | Core i9-14900KF | Core Ultra 7 265K |
| WSL 커널 | 6.18.33.1-microsoft-standard-WSL2 | 6.18.33.2-microsoft-standard-WSL2 |
| 비교 Python | 동일 묶음의 Linux Python 3.8.10 | 동일 묶음의 Linux Python 3.8.10 |
| 이동 묶음 시험 | 10회 중 1회 SIGSEGV | 20회 모두 통과 |

Python 바이너리 SHA256: 298a9e830ed52f36c299427565485d717d1ce0179c0597cc16560513eb780b06 (양쪽 동일).
SHA256SUMS 파일 자체의 해시도 동일했다.

이 결과는 모든 PC에서 발생하는 Python 계산 버그라는 설명을 약화시키고, 현재 호스트 또는 환경에 의존하는 이상이라는 판단을 강화한다. 그러나 CPU와 커널이 동시에 달라 CPU/RAM 고장이나 WSL 커널 결함 중 하나로 확정할 수 없다. 20회 통과는 간헐 오류의 완전 배제를 의미하지 않는다. 다음 비교는 현재 PC에서 WSL 커널을 대조 PC와 맞춘 뒤 같은 묶음을 재실행하는 것이며, 재부팅/실행 중 작업 중단을 수반하는 환경 변경은 아직 수행하지 않았다. 커널 변경 후에도 실패하면 현재 PC의 네이티브 Linux 비교와 BIOS 기본 설정 및 메모리 검사를 통해 범위를 좁힌다.

다른 PC 원본 로그 사본: diagnostics/.runtime/other_pc_20260912/results-20260912-161450-37419
Expert/runner/runtime 코드는 수정하지 않았다.

## 17:31:43 시나리오 + Runtime 종료 분석

사용자 정정: 17:17:21 Errno 38 사례는 Expert + Runtime + collector. 17:31:43 사례는 scenario + Runtime이며 collector 실행으로 해석하지 않는다.

해당 시각의 원본 진단: expert/.runtime/diagnostics/lane-project-368475-3ju8u7uv.json (UTC 08:31:43.858975).
Expert /usr/bin/python3 3.8.10, 시스템 NumPy 1.17.4. project() index=247, 검사 중인 map link=A2256W000153, Ego=(-99.77883911132812,-370.97119140625).
project 프레임 x=np.float64(70.35800471005938), y=np.float64(-341.6158758359766), dx/dy/length_sq도 scalar, ratio=1.0. 그러나 LaneMatch.__init__의 self.x=float(x)에서 LaneMatch를 float로 변환할 수 없다는 TypeError가 발생했다. 저장된 project 프레임과 생성자 오류의 타입 불일치가 핵심이다. 현재 helper는 생성자 프레임의 self/x/object identity를 기록하지 않아 x가 self인지, 어떤 객체인지 또는 어느 단계에서 잘못됐는지는 확정 불가하다. 읽은 소스의 호출 인자 순서와 생성자 시그니처는 일치한다. 검색한 Expert/runner Python 코드에서 LaneMatch 또는 builtins.float 재정의 경로는 발견하지 못했다.

직전 다른 진단(17:28:46, pid 367721)에는 dx=end.x-start.x에서 Point에 enumerate 속성이 없다는 AttributeError가 기록돼 있다. 이 진단의 실행 조합은 확인하지 않았으며 이번 조합으로 단정하지 않는다.

이번 종료: Expert TypeError -> runner unexpected stop -> READY 해제 -> Runtime 정상 정리. 제공된 이번 rosbridge 로그에는 Errno38이 없고, /dafault_topic 오류를 이번 직접 종료 원인으로 볼 근거도 없다. Event.wait 우회가 해결하는 서비스 대기 오류와 분리한다.

15:46:07 진단의 Ego=(-100.7843,-372.7514)와 이번 좌표가 약2.05m로 가깝지만, 검사하던 map link는 각각 A2256W000429/A2256W000153로 다르다. 동일한 지점에서 발생했다는 관찰을 뒷받침하지만 특정 map segment 손상으로 확정할 수 없다. match는 여러 링크를 순회하며, 검사 링크는 Ego가 실제 주행 중인 링크와 같다는 보장이 없다.

주행 코드는 변경하지 않았다. 다음 관측에 필요한 정보는 생성자 traceback 프레임의 self/x 및 호출 프레임 x의 타입과 id이며, 이를 확보하기 전 인자 오염 경로 또는 CPU/WSL 결함을 단정하지 않는다.

## 18:09:51 getattr 오류와 독립 재현

lane-project-376363-bsgmd7ud.json: _projection_xy의 coordinates는 두 float의 list, coordinate=74.73550037834971 (builtins.float). getattr(coordinate,"ndim",0)에서 TypeError: int object is not callable. 배열 입력이라는 근거가 없다.

ROS/NumPy/Expert를 import하지 않는 -I -S 독립 스크립트에서 동일 호출을 재현했다. 시스템 Python3.8.10 3회 중 1회 통과, 1회 float 변환 후 값 비교 실패, 1회 2,392,518번째 반복에 동일한 getattr TypeError. 예외 처리 시 builtins.getattr는 built-in function, builtins.float는 class float로 기록됐다. 이는 예외 당시 평가 스택을 보존한 자료는 아니므로 일시적인 바인딩 변화를 완전히 배제하지는 않는다. 독립 Python3.8.20은 300만 반복 3회 통과; 장시간 안정성 보증 아님.

스크립트: diagnostics/.runtime/probe_builtin_calls_20260912.py
결과: diagnostics/.runtime/probe_builtin_calls_results_20260912.json

이번 결과로 해당 오류는 시나리오/ROS/NumPy를 필수 조건으로 하지 않음이 확인됐다. 특정 lane_map 연산을 계속 우회하는 대신 시스템 Python 실행 환경을 우선 분리 조사한다. CPython 빌드/OS/하드웨어 중 최종 원인은 여전히 구분하지 못했다. 주행 코드 추가 변경 없음.

## 추가 능동 조사: 실제 지도 반복 및 호스트 펌웨어

실제 A2256W000179 지도 링크와 오류 Ego 좌표를 현재 project 코드로 15초씩 교대 3회 실행했다. 두 인터프리터 모두 같은 NumPy1.17.4 사용.
- /usr/bin/python3 3.8.10: 첫 회 _projection_xy line106 SIGSEGV; 나머지2회는 각15초 통과(38338/38523 투영).
- 독립 Python3.8.20: 3회 통과(35412/35139/36403, 총106954 투영). 45초의 제한된 검사이며 장시간 안정성을 확정하지 않는다.
- Python allocator pymalloc/malloc/debug 각각6회의 기본 함수 재현: 이번18회는 통과. 이전 실패를 무효화하거나 특정 allocator를 배제하는 결과가 아니다.
- strace로 sleep(0)을10초 추적:157363회, 실패 syscall 기록 없음. 추적이 타이밍을 바꿀 수 있어 Errno38 해결 증거는 아니다.

호스트 읽기 전용 확인:
- Gigabyte Z790 AORUS ELITE, 보드 리비전 SMBIOS x.x (정확한 다운로드 대상은 아직 특정하지 않음).
- BIOS FG, 2023-09-12.
- i9-14900KF, Windows HKLM HARDWARE DESCRIPTION System CentralProcessor 0의 Update Revision 바이트=1D-01-00-00 (little endian 0x11D), Previous 동일.
- Samsung16GB 두 개, ConfiguredClockSpeed5600. 이 조회만으로 RAM 결함 또는 XMP 사용 여부를 판단하지 않는다.

Intel 공식 안정성 안내는 해당13/14세대 데스크톱에 Intel Default Settings 및 0x12F 이상 최신 BIOS를 권고한다. 현재 OS가 보고한 리비전은 이 권고와 다르다. 펌웨어 상태는 실제 점검 대상으로 격상하지만 이번 Python 오류가 그 결함에서 비롯됐다고 확정하지 않는다. BIOS 플래시/재부팅/WSL 업데이트/주행 코드 변경은 수행하지 않았다.
https://www.intel.com/content/www/us/en/support/articles/000102331/processors.html

원본 실험: diagnostics/.runtime/projection_long_matrix_20260912.json, builtin_allocator_matrix_20260912.json, sleep_zero_syscalls_20260912.log.
