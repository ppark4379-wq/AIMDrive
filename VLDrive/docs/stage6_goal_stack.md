# MORAI 전체 목표 모델

학습 구현과 현재 실행 안내는 [LAVIS/README_MORAI.md](../../../LAVIS/README_MORAI.md)로 이동했다.

`GoalStack` → `MoraiDrivingModel`, `GoalSequenceDataset` → `MoraiTrainingDataset`이며,
CCDP/MEFA·metric decoder·학습/preflight·checkpoint 구현은 LAVIS에서 관리한다.
`train_goal_stack.py` 대신 `LAVIS/train_morai.py`를 사용한다.
기존 `reports/`, `configs/`, raw dataset과 pretrained assets는 그대로 보존한다.
코드 이동에 따른 strict-resume 조건도 새 안내에 명시했다.
