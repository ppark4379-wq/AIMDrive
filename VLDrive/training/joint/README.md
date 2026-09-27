# 커넥터 + LLM 공동 학습

`setup.py`는 기존 공동 설정·모델 클래스, JSON 설정 로더와 optimizer 구성을 모은
학습용 import 입구입니다. 모델 생성자와 설정 클래스는 기존 객체를 그대로 노출합니다.

`training.py`는 학습·평가·preflight 루프를 담당합니다. 시드·배치·optimizer·체크포인트·
gradient 진단은 `../common/`에서 가져옵니다. 커넥터와 LLM의 공동 backward,
optimizer step 순서, 손실 가중치와 동결 정책은 유지합니다.

```bash
bash training/joint/scripts/train.sh --config training/vldrive_train.json
```

셸은 기존 `train.py vldrive`를 실행합니다. 새로운 Python 실행기를 만들지 않습니다.
