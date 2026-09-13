# Expert 정적 장애물 구간 SIGFPE 조사 — 2026-09-11

## 확인된 종료

- 사용자 실행: 격리 Python 3.8.20 venv, full_loop, --no-sync-runtime --once.
- 첫 첨부: 약 84.3초, 경로 307.1/2184.6m, 정적 장애물 ElectricScooter2.
- 추가 로그: 약 50.2초, 경로 308.8/2184.6m, TrafficBarrel.
- 추가 로그의 직접 원인: `Fatal Python error: Floating point exception`.
- WSL 커널: 독립 Python 실행 파일에 signal 8 두 건, 대상 pid 11158 및 11723.
- 두 Python 스택 모두 `lattice_planner.py:339`의 `_obb_collision()`에서
  `vehicle_forward = (math.cos(vehicle_yaw), math.sin(vehicle_yaw))` 실행 중.
- 호출 경로: 정적 장애물 구간 → 후보 경로 평가 → OBB 충돌 검사 → cos/sin.
- Runner의 `AIMDrive expert process stopped unexpectedly`는 Expert 종료 감지 결과.
- 이전 no_route_progress 120초 감시 종료와 구분해야 한다.
- 이 실행에는 Collector 및 Sync Runtime을 사용하지 않았다.

## 판단의 한계

SIGFPE는 네이티브 산술 오류 신호이며, 이 로그만으로 0 나누기, 잘못된 좌표,
CPU 고장, NumPy 문제 중 하나를 확정할 수 없다. 표시된 Python 줄은 네이티브
실패 위치 자체가 아니다. 독립 Python으로 바꾼 것만으로 문제가 해결되지 않았다.

참고: https://sourceware.org/glibc/manual/2.39/html_node/Program-Error-Signals.html

## 오프라인 확인

- 실제 runtime path.csv와 정지 지점 주변 y=-169.49/-165.41/-164.21/-162.50,
  장애물 위치 (-60.61,-142.178), 두 크기 조합으로 planner.plan 200회 통과.
- ROS/Expert 모듈을 로드하고 실제 runtime config로 AutonomousDriving을 생성한 뒤,
  경로 인덱스 600~660의 정적 장애물 제어 계산 305회 통과.
- 검사 시 fegetexcept()는 import 전후/초기화 후/샘플 구간에서 0.
- 좌표 및 크기는 로그를 바탕으로 구성한 근사 입력이며 실제 실패 프레임의 전체
  입력과 동일하다고 보장하지 않는다. 이 통과로 간헐적 환경 문제를 배제할 수 없다.
- 재현 스크립트: expert/.runtime/probe_static_collision.py,
  expert/.runtime/replay_expert_static.py (로컬 진단 산출물).

## 네이티브 진단 실행

현재 PC에는 apt-get download + dpkg-deb -x로 gdb 9.2와 필요한 라이브러리만
expert/.runtime/native_debug/root 아래에 준비했다. 시스템 패키지를 설치하거나
LD_LIBRARY_PATH를 시스템 또는 Expert 환경에 추가하지 않았다.
진단 도구와 venv 바이너리는 Git 배포 대상이 아니다.

```bash
cd ~/AIMDrive/MoraiDrive
bash expert/run_isolated_python.sh diagnostics/run_scenario_native_debug.py \
  --zone full_loop --scenario full_loop --no-sync-runtime --once
```

실제 시뮬레이션을 시작한다. 기존 Runner와 Expert를 종료한 상태에서 사용한다.
이 실행기만 Expert 자식 프로세스를 gdb로 실행한다. 일반 Runner 실행 명령과
주행/충돌 판정 코드는 변경하지 않는다.
출력: expert/.runtime/native_debug/logs/expert-날짜-시간-PID.log.
신호 발생 시 siginfo, 명령어, 부동소수점 레지스터, 스레드 스택, 라이브러리를
기록한 후 신호를 전달해 원래 종료 처리가 이루어지게 한다.

## 실제 시뮬레이션에서 포착한 네이티브 오류

두 번 모두 최대 110초 제한으로 시작했으며, 충돌 후 Runner cleanup/MORAI finalized를
확인했다. 주행 계산 코드는 변경하지 않았다.

### 기본 allocator + gdb

- 원본 로그: expert/.runtime/native_debug/logs/expert-20260911-003659-14803.log
- 약 56.7초, x=-98.88/y=-264.62, 경로 198.3m 근처.
- 최초 신호: SIGSEGV, si_code=128.
- 네이티브 위치: 독립 libpython3.8.so.1.0의 `_PyObject_Malloc+67`.
- 실패 명령: `mov (%rax),%rsi`, rax=`0x78d7ede2ca3000`.
- 스택: `_PyObject_Malloc` → `PyLong_FromVoidPtr` → 시스템 NumPy
  `_multiarray_umath` → Python evaluation.
- Python 위치: lane_map.py:105의 project().
- 유효하지 않은 메모리 주소 접근은 확인됐다. 그 주소가 잘못된 원인은 아직 미확정이다.

### PYTHONMALLOC=debug + PYTHONTRACEMALLOC=1 + gdb

- 통합 로그: expert/.runtime/native_debug/scenario-20260911-003950.log
- 최초 신호: SIGILL, si_code=2.
- 네이티브 위치: 같은 독립 libpython의 `PyDict_GetItem+238`.
- 디버거에서 해당 위치를 읽었을 때 명령은 `mov %rbx,%rsi`로 표시됐다.
- 스택에 시스템 NumPy `_multiarray_umath` 포함.
- Python 위치: lane_map.py:101의 project().
- 메모리 경계 검사에서 명시적인 pad-byte 손상 보고는 발견되지 않았다.
- 최초 SIGILL 이후 faulthandler가 같은 신호를 다시 발생시켜 gdb가 한 번 더 멈췄다.
  진단 명령 파일에 SIGILL/SIGBUS 재전달 처리를 추가했다.

이는 정적 장애물 cos/sin 한 줄에만 국한된 실패가 아니다. SIGFPE, SIGSEGV,
SIGILL이 서로 다른 지점에서 발생했다. NumPy가 스택에 있다는 사실만으로 NumPy가
메모리를 손상시켰다고 결론 내릴 수 없으며, CPU 또는 RAM 물리 고장도 확정하지 않는다.
실행 라이브러리 목록에는 독립 libpython 하나가 보였고, 시스템 NumPy 1.17.4를 사용했다.
`dpkg -V python3-numpy`는 출력 없이 통과했다.

## 현재 PC의 펌웨어와 다음 확인

Windows CIM 재확인:
- CPU: Intel Core i9-14900KF.
- 메인보드: Gigabyte Z790 AORUS ELITE, SMBIOS의 보드 revision 값은 x.x.
- BIOS: FG, 2023-09-12.

Intel은 해당 13/14세대 데스크톱 안정성 문제에 대해 최신 BIOS(0x12F 이상)와
Intel Default Settings를 권고한다. BIOS 버전만으로 현재 OS에 적용된 마이크로코드
버전을 확정하지는 않는다.

- [Intel 공식 권고](https://www.intel.com/content/www/us/en/support/articles/000102331/processors.html)
- [GIGABYTE rev 1.1 지원 페이지](https://www.gigabyte.com/kr/Motherboard/Z790-AORUS-ELITE-rev-11/support)

rev 1.1 페이지의 FG 날짜가 현재 BIOS와 일치한다. 그러나 보드 PCB의 REV 표기를
직접 확인하기 전에는 rev 1.1용 펌웨어가 맞다고 단정하거나 플래시하면 안 된다.
다음 단계는 정확한 보드 revision에 맞는 BIOS 업데이트와 Intel 기본 설정을 적용한
후 동일 검사로 비교하는 것이다. 이것은 원인을 좁히고 알려진 안정성 위험을 줄이기
위한 조치이며, 이미 발생한 문제의 원인이 해당 CPU 이슈라고 확정한 것은 아니다.
BIOS 변경, Windows 재부팅, 시스템 패키지 교체는 수행하지 않았다.
