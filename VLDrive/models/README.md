# 모델 구성

```text
models/
├─ ve/                       # 카메라·LiDAR → 시각 토큰
│  ├─ vision_encoder.py
│  ├─ frozen_vision_encoder.py
│  ├─ resnet.py
│  ├─ pointpillars.py
│  └─ transformer.py
├─ connector/                # 토큰 선택·시간 정보 결합·LLM 입력 연결
│  ├─ ccdp.py
│  ├─ qformer.py
│  ├─ qformer_mefa.py
│  └─ visual_token_interface.py
├─ llm/                      # 시각·지시문 입력 → waypoint 출력
│  ├─ language_model.py
│  ├─ tinyllama_backbone.py
│  ├─ ddia_attention.py
│  └─ multimodal_sequence.py
└─ vldrive.py                # 전체 모델 조립·Stage2 학습 연결
```

- 폴더 이름은 Python 패키지 관례에 맞춰 소문자를 사용합니다.
- 폴더 이동은 모델의 연산·파라미터 이름·가중치 형식을 변경하지 않습니다.
- 학습·추론 명령어와 config 경로는 기존대로 사용합니다.
