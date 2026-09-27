# LLM 학습 구성

`setup.py`에서 기존 공동 설정 `VLDriveTrainingConfig`, 언어 모델 생성자
`VLDriveModel`, TinyLlama 로더와 정밀도 변환 함수를 제공합니다.
`utils/`는 기존 파라미터 통계·멀티모달 시퀀스 구성·gradient 진단을 제공합니다.

학습은 `../joint/training.py`에서 커넥터와 함께 수행합니다. 실제 생성과 동결 정책은
기존 `models/vldrive.py`의 생성자가 적용합니다. lm_head/end_predictor 동결과
설정에 따른 gradient checkpointing을 그대로 사용합니다.

LLM 전용 새 설정 클래스나 독립 학습 루프는 추가하지 않았습니다.
