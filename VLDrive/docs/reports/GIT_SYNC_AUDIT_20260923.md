# 원격 동기화 및 충돌 사전 검사

검사·보고: 2026-09-22~23 (KST). 원격 `https://github.com/INHA-AIM/AIMDrive.git`를 `git fetch --no-tags origin`으로 갱신한 뒤 검사했다. 실제 merge/rebase/commit/push는 수행하지 않았다. 현재 브랜치, 사용자 index와 소스 파일은 변경하지 않았다. 이 보고서만 추가했다.

## 브랜치 상태

| 항목 | 확인 결과 |
|---|---|
| 현재 브랜치 | main |
| 로컬 HEAD | f81aeab9c059ba8a3eec2983145f827dbb02cf96 |
| origin/main | 69acda739a1235adcb30e0bbc74657ae488d9bed |
| main에만 있는 커밋 | 0 |
| origin/main에만 있는 커밋 | 18 |
| 변경된 추적 파일 | 18개 |
| staged 파일 | 0개 |
| 미추적 파일 | 검사 당시 469개, 합계 299.61MiB — 본 보고서 추가 전 |
| 원격에서 변경된 경로 | 로컬 HEAD 대비 415개 |

현재 쌓인 작업은 main의 미전송 커밋이 아니라 미커밋 파일이다. 그대로 push해도 이 파일들이 업로드되지 않는다. main은 원격보다 뒤처져 있어 일반 push는 non-fast-forward 거절 대상이다.

`refactor/lmdrive-vldrive-structure`는 해당 원격 브랜치보다 4커밋 앞서지만, 현재 origin/main과 비교하면 로컬 전용 커밋 0 / 원격 전용 52다. 네 커밋은 이미 main의 조상 이력에 포함되어 있다:

- b2b61db: Merge origin/main: 동적 CCDP 수정을 새 구조로 반영
- cd1b37b: Fix: stabilize dynamic CCDP training and retention validation
- 1dde95f: Merge origin/main: VLDrive와 .gitignore 원격 변경 반영
- 8182680: 수집 데이터 저장·재개 안정성 개선 및 주행 오류 진단 추가

이번 작업을 올리기 위해 오래된 refactor/backup 브랜치까지 다시 합치거나 `push --all` 할 필요는 없다.

## 재현 방법과 범위

무시되는 `VLDrive/.runtime/git_conflict_audit_p3tsy_ea/`에 별도 Git index/object directory를 만들었다. 원본 object store는 읽기용 alternate로 참조했다. 현재 HEAD에 추적 파일의 실제 수정 내용과 원격 신규 파일에 겹치는 로컬 파일을 반영한 임시 snapshot을 만들고, `git merge-tree --write-tree --name-only --messages`로 origin/main과 병합을 계산했다. 원본 refs/index/작업 파일은 건드리지 않았다.

서로 겹치지 않는 미추적 파일은 텍스트 충돌 대상이 아니므로 임시 snapshot에 대용량 결과물까지 넣지 않았다. 양쪽에서 수정한 파일 6개, 새 파일 경로 중복 1개를 확인했고 최종 5개 파일에서 Git 충돌을 재현했다. 보고서는 이 시점의 HEAD와 작업 내용에 대한 결과이며, 향후 원격 변경에 따라 달라질 수 있다.

`merge.txt`, `report.json`, `conflicts/`는 위 임시 검사 폴더에 보존했다. 충돌 표시가 들어 있는 파일은 검사 폴더 안의 복사본뿐이다. 일회성 검사 스크립트는 삭제했다.

## 충돌 파일 및 해결 방향

### 1. MoraiDrive/expert/src/expert/src/autonomous_driving.py — content

로컬은 기존 제어 로직을 새 `control/path_controller.py`로 분리하고 `AutonomousDriving(PathController)`로 공유한다. VLDrive 실시간 제어도 이 공유 모듈에 의존한다.

원격은 분리 이전 파일 안의 동일 제어 구간을 고쳤다. 보행자 통과 후 정지 해제, 회전교차로 진입 가속 gain, 정적 회피 lookahead, 디버그 기록, planning의 global_path 전달 등이 포함된다.

로컬만 선택하면 원격 주행 개선을 잃고, 원격만 선택하면 로컬 공통화 구조와 동작의 일관성이 깨진다. 공통화 구조를 유지한 채 원격 변경을 적절한 위치로 이식해야 한다. 특히 원격의 `self.adaptive_cruise_control.pedestrian_released` 의존성을 VLDrive용 `PathController`에 그대로 복사하면 해당 객체가 없는 추론 경로에서 문제가 생길 수 있다. planner 문맥 의존 로직과 공통 제어 로직을 구분해서 연결해야 한다.

현재 미추적인 `MoraiDrive/expert/src/expert/src/control/path_controller.py`도 함께 커밋해야 한다. 원격 `control/pid.py`의 새 `acceleration_p_gain` 인수는 기본값 None으로 추가되어 있지만, expert의 새 호출·조건과 함께 검증해야 한다. 로컬 VLDrive의 LMDrive 조향 및 목표속도 5km/h 설정을 회귀 검사한다.

### 2. VLDrive/train.py — content

로컬: navigation token cache, batch size, validation 비활성화, 전체 경로 학습, frame interval, warm-start, 모니터·평가 연결.

원격: `--notices`, `--clip-index`, Notice 입력, `morai_sensor_record_clip_v1` dataset 지원.

설정 검사와 dataset 생성 분기에서 직접 충돌한다. 세 형식 `morai_expert_cache_v1`, `navigation_token_cache_v1`, `morai_sensor_record_clip_v1`을 명시적으로 나누고 각 형식에 필요한 입력을 검증해야 한다. 로컬의 full-route/validation 정책은 현재 지원하는 형식에 한정하고, 원격 기능 때문에 기존 이어학습·batch 설정이 빠지지 않도록 통합한다. 원격 분기의 `('train','val')` 고정값과 로컬 `splits` 변수도 함께 정리해야 한다.

### 3. VLDrive/datasets/morai_expert_dataset.py — content

로컬 생성자에는 `frame_sampling_interval`, 원격에는 `notices`가 추가되어 같은 위치에서 충돌한다. 둘 다 유지해야 한다.

학습 mask 계산도 충돌한다. 로컬은 연속 프레임에서 instruction 경계를 찾고 sampling interval을 반영한다. 원격은 navigation과 notice가 모두 같은 구간을 찾는다. 연속 원본 이력에서 `(instruction, notice_text)`의 조건 경계를 계산한 뒤 선택한 프레임 인덱스에 맞춰 mask를 만드는 방향이 타당하다. Notice 변경이 샘플링에서 빠진 중간 프레임에서 일어나는 경우도 테스트해야 한다.

### 4. VLDrive/utils/runtime.py — content

같은 상대경로 변환 목록에 로컬은 `warm_start_checkpoint`, 원격은 `notices`를 추가했다. 두 키 모두 유지하면 된다. 로컬 navigation token cache 경로 처리도 보존한다. 비교적 단순한 충돌이다.

### 5. VLDrive/tools/cache_sensor_record_tokens.py — add/add

원격과 로컬이 같은 파일명을 서로 다른 신규 기능에 사용했다. 로컬에서는 아직 미추적 파일이다.

- 로컬: native episode.json 및 navigation manifest 기반. `native-navigation-token-cache-config-v1`, `morai_encoder_contract`, 재개 가능한 캐시 생성.
- 원격: sealed Sensor Record publication 기반. manifest index/campaign, feature_model_name/token_shape 계약, semantic clip dataset 연결.

전체 파일 중 한쪽을 선택하면 다른 데이터 파이프라인을 잃는다. 권장 방식은 원격 이름을 유지하고 로컬 도구를 `cache_native_navigation_tokens.py` 같은 명확한 이름으로 분리한 뒤 문서·호출부를 수정하는 것이다. 필요하면 추후 공통 CLI에서 형식별로 dispatch하도록 통합한다. 이번 검사에서는 이름이나 구현을 바꾸지 않았다.

## Git이 자동 병합하는 파일도 확인 필요

- `VLDrive/models/vldrive.py`: 로컬의 CCDP/reconstruction 옵션 및 `token_selection_mode`와 원격의 Notice 문자열 결합이 자동 병합됐다. Notice 유무와 Gumbel 평가/추론 경로를 함께 회귀 검사해야 한다.
- `VLDrive/tests/test_morai_expert_dataset.py`: 테스트 추가가 자동 병합됐지만 실제 통과를 의미하지 않는다. 본 검사는 병합 충돌 검사이며 통합 코드의 테스트 실행은 하지 않았다.

## 커밋 대상 정리

검사 당시 미추적 파일의 큰 부분:

| 경로 | 개수 | 크기 |
|---|---:|---:|
| VLDrive/outputs/inference | 297 | 277.71MiB |
| VLDrive/data/processed | 17 | 19.49MiB |
| VLDrive/outputs/analysis | 15 | 0.65MiB |

현재 ignore에는 checkpoints/logs 등이 있지만 `outputs/inference` 전체는 제외되지 않는다. `git add .`를 바로 쓰면 실행 결과 JSONL이 함께 들어간다. 필요한 결과 요약은 docs에 보존하고 재생 로그/생성 캐시는 별도 보관하는 것이 적절하다. instruction label 산출물은 재현용으로 버전 관리할 범위를 명시적으로 선택한다. 원본 데이터나 로그를 삭제할 필요는 없다.

## 권장 진행 순서

1. 현재 작업을 보존할 새 통합 브랜치를 만든다. 코드·설정·테스트·문서와 필수 신규 모듈을 선별해 복구 가능한 커밋으로 보존한다. 원본 로그/데이터는 그대로 두고 커밋에서 제외한다.
2. 별도 worktree에서 그 브랜치와 최신 origin/main을 merge한다. 현재 사용 중인 추론 작업 폴더를 충돌 상태로 만들지 않는다.
3. 위 다섯 충돌을 기능별로 통합한다. ours/theirs 전체 선택으로 덮지 않는다.
4. expert 회피·회전교차로·종방향, VLDrive 횡방향/5km/h 제한, navigation/notice dataset, 기존 checkpoint의 평가·추론, warm-start/학습 preflight를 검증한다. 장시간 재학습은 필요하지 않다.
5. 합쳐진 변경과 검증 결과를 검토한 다음 일반 push로 통합 브랜치를 올리고 PR로 main에 반영한다. main 직접 반영이 필요하면 최신 원격 변경을 포함한 상태에서 일반 push한다. force push는 필요 없다.

실제 충돌 해결과 push는 아직 수행하지 않았다.
