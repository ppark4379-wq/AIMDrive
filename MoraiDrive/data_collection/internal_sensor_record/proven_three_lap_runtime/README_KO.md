# 수집 실행 스택의 컴포넌트 경계

현재 대규모 수집 실행은 세 역할을 분리해서 사용한다. 실행 중인 학습과 기존 campaign의
software hash를 보존하기 위해 소스 경로를 물리적으로 이동하지 않는다.

| 역할 | 현재 소스 위치 | 책임 |
|---|---|---|
| 데이터 수집기 | `../async_capture/`, `../native_capture/`, `../tools/run_sensor_record_scenario.py` | Sensor Record 세션, ROS/GT 관측, 원본 admission, 후처리 연결 |
| expert | `expert/` | `/ctrl_cmd`를 만드는 로컬 expert fork |
| scenario runner | `scenario_runner/` | 공식 경로, 날씨·시간·객체·신호 프로필, lap lifecycle |

운영 진입점은 저장소 루트의 `02_collect_scenario_laps.sh`다. 호출 관계는 다음과 같다.

```text
02_collect_scenario_laps.sh
  -> tools/run_sensor_record_scenario.py       수집 세션과 D SSD 원본 관리
     -> tools/run_async_scenario_review.py     세 컴포넌트 orchestration
        -> mission_runtime/scenario_runner/    lap과 환경 변화
        -> mission_runtime/expert/             차량 제어
        -> async_capture/, native_capture/      센서·GT 기록과 검사
```

## 원본과 로컬 fork의 관계

`UPSTREAM_PROVENANCE.json`은 `/home/dydwns/AIMDrive/MoraiDrive`에서 최초 복사할 때의
124개 파일 SHA256을 보존한다. 팀 저장소는 이 작업에서 읽기 전용이며 수정하지 않는다.

현재 로컬 사본은 원본 그대로가 아니다. expert 4개 파일과 scenario runner 7개 파일이
복사 기준과 다르다. expert 변경에는 ROS freshness fail-safe뿐 아니라 pure pursuit,
신호 정지, 정적 장애물 재합류의 실제 주행 판단 변경도 포함된다. scenario runner 변경에는
수집 clock/lifecycle 연결과 대회 수집용 객체·신호 프로필 변경이 포함된다. 따라서 현재
실행을 `unchanged expert`라고 기록하면 안 된다.

허용된 변경의 파일·분류·현재 SHA256은 `LOCAL_FORK_CHANGES.json`에 있다. 다음 명령은
새로운 미기록 변경, 파일 누락, 선언된 SHA256 불일치를 모두 실패로 처리한다.

```bash
python3 local_integration/morai_visual_pretraining/mission_runtime/audit_local_forks.py
```

과거 validation 문서의 “expert를 수정하지 않았다”는 문장은 해당 실험에서 새로 수정하지
않았다는 당시 범위를 기록한 것이다. 현재 로컬 fork 전체가 팀 원본과 같다는 뜻으로
해석하지 않는다.

## 정리 원칙

- 수집기 수정은 `async_capture/`, `native_capture/`, 수집용 `tools/`에서 한다.
- 주행 판단 변경은 `mission_runtime/expert/`에서만 하고 변경 manifest를 갱신한다.
- 환경·lap·actor·traffic 변경은 `mission_runtime/scenario_runner/`에서만 한다.
- `mission_runtime/UPSTREAM_PROVENANCE.json`의 복사 기준 hash는 덮어쓰지 않는다.
- `.runtime/`, `data/`, `validation/`은 실행 산출물이므로 소스 폴더 정리 과정에서 이동하거나
  삭제하지 않는다.
