# 커넥터 학습 구성

`setup.py`에서 기존 `QFormerMEFAConfig`와 `QFormerMEFAConnector`를 함께 제공합니다.
`utils/`는 기존 시각 토큰·마스크 보조 함수와 공통 gradient 진단을 제공합니다.
모델 구현을 복제하거나 새로운 학습 옵션을 추가하지 않습니다.

학습은 `../joint/training.py`에서 LLM과 함께 수행합니다. 실제 생성과 동결 정책은
기존 `models/vldrive.py`의 생성자가 적용합니다. learnable query 및 사용하지 않는
Q-Former embedding/pooler 파라미터의 기존 동결 정책을 유지합니다.

별도 optimizer나 독립 커넥터 학습 루프는 추가하지 않았습니다.
