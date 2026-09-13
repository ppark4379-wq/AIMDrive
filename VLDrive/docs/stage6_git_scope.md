# 실행·학습 코드 범위

학습의 기준 구현은 `VLDrive/LAVIS`에 있다. 모델은 `lavis/models/vldrive`,
데이터 계약은 `lavis/datasets/datasets/morai`, 학습·평가는 `lavis/runners`에서 관리한다.
전체 학습과 visual 단독 학습, preflight, checkpoint 검증 도구는 실행 코드로 포함한다.
실제 unit/regression tests는 기존과 같이 로컬에 보존한다.

`experiments/morai_stage6`에는 bridge·수집·후처리, 호환 import 및 과거 실행 기록을 남긴다.
이전 Stage6F baseline은 과거 실험용이며 새로운 LAVIS 학습의 의존성이 아니다.
데이터 fingerprint 함수는 dataset의 `provenance.py`로 분리해 별도 trainer를 import하지 않는다.

원본 데이터·feature cache·pretrained/학습 weight·가상환경·로그·보고서는 Git에 포함하지 않는다.
현재 학습 output은 `LAVIS/outputs`, 수집 output은 기존 `data`/`diagnostics`에 둔다.
이전 checkpoint의 code/data/backend 검사와 파일 hash는 완화하거나 덮어쓰지 않았다.
경로가 바뀐 새 source에서는 새 실행을 시작한다.

[새 실행 안내](../../../LAVIS/README_MORAI.md)를 참고한다. 이번 정리는 파일을 stage하거나
commit/push하지 않으며, 커밋 전 `git diff`와 새 LAVIS 파일을 함께 확인한다.
