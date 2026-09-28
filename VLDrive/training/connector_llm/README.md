# 커넥터 + LLM 공동 학습

`setup.py`는 기존 공동 설정·모델 클래스, JSON 설정 로더와 optimizer 구성을 모은
학습용 import 입구입니다. 모델 생성자와 설정 클래스는 기존 객체를 그대로 노출합니다.
커넥터의 `QFormerMEFAConfig`·`QFormerMEFAConnector`와 LLM의 `VLDriveModel`·
TinyLlama 로더도 이 파일에서 함께 제공합니다.

`utils/`는 기존 시각 토큰·마스크, 멀티모달 시퀀스, 파라미터·gradient 진단 API를
한곳에서 제공합니다. 실제 모델 구현은 `models/connector/`와 `models/llm/`에 있으며
이 폴더에서 모델 구현을 복제하지 않습니다.

`training.py`는 학습·평가·preflight 루프를 담당합니다. 시드·배치·optimizer·체크포인트·
gradient 진단은 `../common/`에서 가져옵니다. 커넥터와 LLM의 공동 backward,
optimizer step 순서, 손실 가중치와 동결 정책은 유지합니다.
별도 커넥터 또는 LLM 단독 학습 루프는 두지 않습니다.

```bash
bash training/connector_llm/scripts/train.sh --config training/vldrive_train.json
```

셸은 기존 `train.py vldrive`를 실행합니다. 새로운 Python 실행기를 만들지 않습니다.
