# MoraiDrive

MORAI 시뮬레이터 데이터로 [LMDrive](../LMDrive/)의 vision encoder(Memfuser)를
학습시켜, 언어 지시 없이 waypoint를 직접 예측하는 end-to-end 자율주행 모델을
만들고 대회 환경에 배포하는 프로젝트.

LLM/Q-Former(LAVIS) 스택은 쓰지 않습니다 — 대회가 자연어 지시 없이 경로만 잘
따라가면 되는 것으로 확인되어, `LMDrive/vision_encoder`(Memfuser)를 그 자체로
최종 주행 모델로 사용합니다.

## 구조

```
MoraiDrive/
├── data_collection/   # MORAI UDP 센서 데이터 -> LMDrive 데이터셋 포맷 변환 파이프라인
├── vision_encoder/    # (예정) MORAI 데이터로 Memfuser 학습 — LMDrive/vision_encoder 참고
└── agent/             # (예정) 대회 실시간 배포 에이전트 — leaderboard/team_code/lmdriver_agent.py 대응
```

- **data_collection**: 완성된 상태. 유일하게 막힌 지점은 `udp/parsers.py`의
  ego/object 패킷 파서 — MORAI 실제 UDP 스펙이 필요. 자세한 내용은
  [data_collection/README.md](data_collection/README.md) 참고.
- **vision_encoder**: 아직 미착수. `data_collection`이 쌓은 데이터를 읽는 축소판
  Dataset 클래스(`LMDrive/vision_encoder/timm/data/carla_dataset.py` 참고)와
  학습 스크립트가 필요.
- **agent**: 아직 미착수. `LMDrive/leaderboard/team_code/lmdriver_agent.py`에서
  LLM 관련 부분을 걷어내고, MORAI UDP로 센서를 받아 제어 신호(steer/throttle/brake)를
  돌려주는 실시간 에이전트.

## 지금 상태 요약

| 단계 | 상태 | 막힌 지점 |
|---|---|---|
| 센서 배치 확정 | 진행 중 | 카메라 개수/위치, MORAI ego pose 원점 기준(차체중심 vs 후륜축) 미확인 |
| 데이터 수집 파이프라인 | 코드 완성, 실행 전 | MORAI UDP 패킷 스펙 (ego/object 파서) |
| vision encoder 학습 | 미착수 | 위 두 개 선행 필요 |
| 대회 배포 에이전트 | 미착수 | vision encoder 학습 선행 필요 |
