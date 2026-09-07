# 실행·학습 코드 범위

Git에는 현재 **MORAI 수집 → teacher·route·feature 후처리 → 전체 Goal stack 학습·평가**에
필요한 코드와 설정·실행 문서만 포함한다. 단위 테스트와 탐색 실험 코드는 로컬에 보존한다.

| 포함 | 역할 |
|---|---|
| `src/aimdrive` 37개 Python 파일 | 현재 진입점에서 사용하는 모델·데이터·수집 계약과 package 초기화 |
| `scripts` 17개 Python 파일 | bridge·수집·후처리·calibration·split·학습·checkpoint 검증 |
| `configs/goal_stack_{small,frozen_visual,all_features}.json` | 같은 전체 모델의 visual encoder 학습 정책 |
| `data/competition/global_path.txt` | local route 생성 입력 |
| 환경 명세·README·수집 및 학습 안내 | 팀 checkout에서 실행하는 방법 |

`GoalSequenceDataset`은 Stage6F trainer의 dataset content hash 함수를 재사용한다.
이 때문에 `stage6f_baseline.py`와 그 import인 `trajectory_baseline.py`는 포함한다.
함께 들어 있는 이전 함수를 제거하려고 이번 파일 선별 작업에서 학습 코드를 재작성하지 않는다.
Q-Former, TinyLlama/DDIA, custom timm은 상위 팀 VLDrive 코드에 의존한다.

다음은 업로드하지 않으며 로컬 파일은 삭제하지 않는다:

- `tests/`, `scripts/test_*.py`: 단위·회귀·통합 테스트.
- visual probe v1/v2, ablation·smoke·baseline 실험 실행기와 전용 설정.
- 현재 pipeline이 참조하지 않는 이전 모델·동기화·runtime 계약 모듈.
- 깊은 병목 계측·성능 비교·그림 생성·이관 도구와 과거 실험 보고서.
- 원본 dataset·feature cache·checkpoint·가상환경·로그·환경 snapshot.

실제 수집의 teacher/bridge 품질 gate, 학습의 실제 gradient·causality preflight,
checkpoint 무결성·복원 검사는 정상 데이터와 학습 결과를 확인하는 기능이므로 유지한다.

기존 checkpoint는 전체 `src/*.py` 목록·내용도 검증한다. 로컬에는 제외한 source도 남아 있어
기존 환경에서 재현할 수 있지만, 파일 구성이 줄어든 새 clone은 기존 checkpoint와 source 목록이
달라 strict resume이 거부된다. 새 clone에서는 새 학습을 시작한다. 기존 실험의 정확한 재개가
필요하면 원래 source·data·backend 환경을 함께 보존한 로컬 실험을 사용한다.
이번 선별을 위해 무결성 검사를 완화하거나 기존 checkpoint 내용을 고치지 않았다.

새 실행 결과는 `reports/`, 수집 데이터는 `data/` 또는 `diagnostics/`에 저장한다.
`git diff --cached --stat`으로 커밋 범위를 확인하고 local-only 파일에 `git add -f`를 사용하지 않는다.
