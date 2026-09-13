# docs/

설계 기록과 실험 보고서입니다.

> **주의**: 아래 문서들은 폴더 구조를 재편하기 전에 쓰였습니다. 본문에 나오는
> `LAVIS/lavis/models/vldrive/...`, `vision_encoder/timm/...` 같은 경로는 지금 구조에서
> 어디로 갔는지 최상위 [`README.md`](../README.md) 의 "재구성 전 코드와의 대응" 표를 보세요.
> 내용(설계 판단, 수식, 실험 결과)은 그대로 유효합니다.

## 설계·구현 기록

| 문서 | 내용 |
|---|---|
| `vldrive_training.md` | 2단계 학습 파이프라인 설명. 얼린 인코더 토큰 → CCDP/MEFA → TinyLlama/DDIA → 웨이포인트 L1 + cycle loss 연결 방식과 설정 항목. |
| `vldrive_language_model.md` | TinyLlama + DDIA 구현 문서. 어떤 원본 파일을 기준으로 검증했는지, 모달리티 어텐션이 무엇을 바꾸는지. |
| `vldrive_handoff_ko.md` | CCDP·MEFA·전체 파이프라인을 완성하기 위한 작업 인수인계서. 무엇이 남았는지 항목별로. |
| `vision_encoder_morai_notes.md` | 1단계(비전 인코더) MORAI 학습 노트. 환경 구성, LiDAR 축 정책, 학습 절차. |
| `lavis_morai_notes.md` | LAVIS 위에서 MORAI 경로를 돌리던 시절의 실행 노트. |
| `traffic_labels.md` | 신호등 라벨 사이드카의 정책과 생성 절차. |
| `stage6_collection.md` | ROS/rosbridge 로 MORAI 에서 직접 수집하던 초기 파이프라인의 수집 규격. |
| `stage6_goal_stack.md` | 같은 시기의 goal-stack 모델 설계. |
| `stage6_git_scope.md` | 그 실험 저장소의 git 추적 범위. |
| `upstream_lmdrive_readme.md` | 원본 LMDrive/VLDrive 저장소 README (CARLA 기준). |
| `pipeline.png` | 논문 파이프라인 그림. |

`stage6_*` 문서가 설명하는 코드는 삭제했습니다. 지금 학습은 MoraiDrive 가 수집한
`morai_expert_cache_v1` 형식만 씁니다. 설계 기록으로만 남겨둡니다.

## 실험 보고서 — `reports/`

2026-09-10 ~ 09-12 에 2단계 학습이 기대만큼 나오지 않은 원인을 추적한 기록입니다.
읽는 순서는 대략 이렇습니다.

1. `MORAI_PIPELINE_IMPLEMENTATION_REPORT.md` — 파이프라인을 어떻게 구성했는지
2. `MORAI_TRAIN_RESULT_20260911.md` / `MORAI_TRAIN_DIAGNOSIS_20260911.md` — 첫 학습 결과와 1차 진단
3. `MORAI_ROOT_CAUSE_AUDIT_20260911.md` / `MORAI_ROOT_CAUSE_IMPLEMENTATION_20260911.md` — 근본 원인 감사와 대응
4. `MORAI_LABEL_ALIGNMENT_DIAGNOSIS_20260911.md` — 신호등 라벨 정렬 문제
5. `MORAI_DATA_SUFFICIENCY_ANALYSIS_20260911.md` — 데이터 양이 충분한지
6. `MORAI_PILOT_ANALYSIS_20260911.md`, `MORAI_CONTROLLED_EXPERIMENT_RESULTS_20260911.md`,
   `MORAI_MATCHED_EXPOSURE_RESULTS_20260911.md` — 통제 실험들
7. `MORAI_REPAIR_TRAINING_20260911.md` — 복구 학습 시도

이 보고서들을 만들어낸 진단 스크립트는 일회성이라 삭제했습니다.
