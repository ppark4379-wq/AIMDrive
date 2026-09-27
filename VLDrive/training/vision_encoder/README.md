# 비전 인코더 학습

| 위치 | 역할 |
|---|---|
| `setup.py` | 기존 설정 클래스·검증 함수, RGB 동결·optimizer·LR 설정, 전이학습 모델 복원·동결 범위·학습 모드 |
| `recipes/pretrain.py` | ImageNet 백본에서 시작하는 MORAI 학습 루프 |
| `recipes/transfer.py` | LMDrive/MORAI 인코더 체크포인트에서 시작하는 MORAI 추가 학습 루프 |
| `losses/morai_transfer.py` | 객체·waypoint·신호등 손실 |
| `utils/morai_transfer_metrics.py` | 전이학습 평가 지표와 배치 이동 |
| `scripts/` | `pretrain.sh`, `transfer.sh`, `object_head.sh` 실행 입구 |

`setup.py`의 함수 본문은 기존 두 recipe에서 옮겼습니다. 설정 클래스는
`utils/config/vision_encoder_config.py`의 동일한 클래스를 사용합니다.
데이터 로딩·체크포인트·학습 루프의 호출 순서와 기본값은 유지합니다.

VLDrive 루트에서 실행합니다.

```bash
bash training/vision_encoder/scripts/pretrain.sh --config configs/vldrive_train.json
bash training/vision_encoder/scripts/transfer.sh --help
bash training/vision_encoder/scripts/object_head.sh
```

`pretrain`도 학습이며 제로샷 추론이 아닙니다. `transfer`는 가중치를 이어받는다는
뜻이며 CARLA와 MORAI 데이터를 자동으로 혼합하지 않습니다.
