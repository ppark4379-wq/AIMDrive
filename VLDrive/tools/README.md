# 도구 위치

- 데이터 전처리·토큰 생성: `data/src/processing/`
- 데이터 검증: `data/src/eval/`
- 외부 모델 준비: `tools/assets/prepare_assets.py`
- 모델 평가: `tools/evaluation/`
- 모델 구조 검증: `tools/validation/audit_vldrive_paper.py`
- CARLA 참조 도구: `tools/carla/`
- 학습: `training/`
- 모델 시각화: `utils/visualization/`

VLDrive 폴더에서 실행합니다.

```bash
python data/src/processing/process_data.py
python train.py --config training/vldrive_train.json
```

처리 결과는 `data/processed/<ID>/`에 저장하고 학습 config의 `data_version`으로 선택합니다.
