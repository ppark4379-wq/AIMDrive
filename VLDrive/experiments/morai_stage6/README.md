# MORAI 수집과 이전 실험 보관

**학습 코드는 [LAVIS](../../LAVIS/README_MORAI.md)로 이동했다.**
전체 모델은 `LAVIS/train_morai.py`, visual encoder 단독 학습은
`LAVIS/train_morai_visual.py`에서 실행한다.

이 폴더에는 ROS bridge·수집·teacher/route/feature 후처리와 기존 데이터·가중치·결과를
보존한다. `src/aimdrive`의 이동된 모델·데이터 모듈은 LAVIS 구현을 가리키는 호환 import다.
새 학습 코드는 이 폴더의 source를 import하지 않는다.

- [수집 절차](docs/COLLECTION.md)
- [현재 학습 구조와 실행 방법](../../LAVIS/README_MORAI.md)
- [파일 범위와 보존 정책](docs/GIT_SCOPE.md)

기존 Python 환경 `.runtime/stage6h_env/bin/python`은 계속 사용할 수 있다.
기존 configs/reports는 과거 실행 기록이며 새 학습 설정은 LAVIS의
`lavis/projects/vldrive/morai_*.json`을 사용한다.
수집 시에도 동일 checkout의 `VLDrive/LAVIS`가 필요하다. 수집용 import는
MORAI 데이터 검증 모듈만 읽으며 Torch나 CARLA를 자동으로 불러오지 않는다.
`AIMDrive/MoraiDrive`와의 병합은 하지 않았다.
