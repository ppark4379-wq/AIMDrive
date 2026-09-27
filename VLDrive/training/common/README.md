# 학습 공통 기능

| 파일 | 기존 코드에서 옮긴 기능 |
|---|---|
| `batching.py` | 배치 결합·장치 이동, subset/stride/balanced 샘플 선택 |
| `precision.py` | 기존 autocast |
| `reproducibility.py` | 시드 고정, RNG 저장·복원 context manager |
| `optimizers.py` | CPUAdamW, optimizer 파라미터 그룹, cosine LR |
| `checkpoints.py` | 모델·학습 상태 저장·복원, warm start, 파라미터·소스 해시 |
| `logging.py` | gradient 진단, 기존 TrainingTelemetry 연결 |
| `training_helper.py` | 이전 helper import를 유지하는 호환 연결 |

기존 `training/common/training_helper.py`와 `training/joint/training.py`의 함수·클래스
본문을 옮겼습니다. 계산과 기본값은 유지하고 소스 검증 목록에 새 경로를 추가했습니다.
