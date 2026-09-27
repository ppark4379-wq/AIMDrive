# 데이터 전처리 및 학습

- 1. 학습 환경에 진입하고 VLDrive 폴더로 이동합니다.

```bash
conda activate lmdrive
cd /mnt/c/Users/autonav009/Desktop/AIMDrive/VLDrive
```

- 2. 실제 원본 데이터 경로와 VE 가중치 설정을 확인합니다.

```bash
ls data/raw
sha256sum outputs/checkpoints/vision_encoder/checkpoint-best-object.pth.tar
nano configs/vldrive_train.json
```

- 3. 새 ID로 데이터를 전처리하고 완료된 버전을 학습 설정에 반영합니다.

```bash
python data/src/processing/process_data.py
```

- 4. `<ID>`를 생성된 ID로 바꾸고 처리 완료 상태와 학습에 선택된 버전을 확인합니다.

```bash
cat data/processed/<ID>/status.json
ls data/processed/<ID>/manifest.json
python -c "import json; print(json.load(open('configs/vldrive_train.json'))['data_version'])"
```

- 5. 완료 상태가 `complete`이면 Stage2 학습을 시작합니다.

```bash
python train.py --config configs/vldrive_train.json
```
