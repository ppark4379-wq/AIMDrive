# MoraiDrive

MORAI 시뮬레이터 데이터로 LMDrive의 vision encoder(Memfuser) 구조를 독립적으로
복사·적용해, 언어 지시 없이 waypoint를 직접 예측하는 end-to-end 자율주행 모델을
만들고 대회 환경에 배포하는 프로젝트. `LMDrive/`는 참조 원본으로 유지하고 MORAI
관련 구현은 모두 `MoraiDrive/` 내부에서 관리합니다.

LLM/Q-Former(LAVIS) 스택은 쓰지 않습니다 — 대회가 자연어 지시 없이 경로만 잘
따라가면 되는 것으로 확인되어, `MoraiDrive/vision_encoder`의 Memfuser를 그 자체로
최종 주행 모델로 사용합니다.

## 구조

```
MoraiDrive/
├── data_collection/   # MORAI UDP 센서 데이터 -> LMDrive 데이터셋 포맷 변환 파이프라인
├── vision_encoder/    # LMDrive에서 독립 복사한 MORAI용 Dataset·Loader·Memfuser·학습 코드
└── agent/             # (예정) 대회 실시간 배포 에이전트 — leaderboard/team_code/lmdriver_agent.py 대응
```

- **data_collection**: 완성된 상태. 유일하게 막힌 지점은 `udp/parsers.py`의
  ego/object 패킷 파서 — MORAI 실제 UDP 스펙이 필요. 자세한 내용은
  [data_collection/README.md](data_collection/README.md) 참고.
- **vision_encoder**: `MoraiMVDetDataset`, `create_morai_loader`, Memfuser 및 학습
  스크립트를 `MoraiDrive/vision_encoder` 안에 독립 구성했습니다. LMDrive와 동일한
  `/5` sampling 및 batch/target 계약을 유지하며, 현재 독립 프레임 batch 검증까지
  통과했습니다. MORAI용 train/validation split과 실제 forward/loss smoke test는
  남아 있습니다.
- **agent**: 아직 미착수. `LMDrive/leaderboard/team_code/lmdriver_agent.py`에서
  LLM 관련 부분을 걷어내고, MORAI UDP로 센서를 받아 제어 신호(steer/throttle/brake)를
  돌려주는 실시간 에이전트.

## 지금 상태 요약

| 단계                   | 상태               | 막힌 지점                                                             |
| ---------------------- | ------------------ | --------------------------------------------------------------------- |
| 센서 배치 확정         | 진행 중            | 카메라 개수/위치, MORAI ego pose 원점 기준(차체중심 vs 후륜축) 미확인 |
| 데이터 수집 파이프라인 | 코드 완성, 실행 전 | MORAI UDP 패킷 스펙 (ego/object 파서)                                 |
| vision encoder 학습    | 미착수             | 위 두 개 선행 필요                                                    |
| 대회 배포 에이전트     | 미착수             | vision encoder 학습 선행 필요                                         |
