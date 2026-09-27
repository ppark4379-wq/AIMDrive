# 시각화

오프라인 결과 확인과 실시간 대시보드를 이 폴더에서 관리합니다.

| 파일/폴더 | 역할 |
|---|---|
| `inspect_vision_tokens.py` | 저장된 센서 데이터를 비전 인코더로 처리해 토큰·예측 결과를 추출 |
| `render_vision_tokens.py` | 추출 결과를 읽어 카메라 이미지와 BEV 토큰 크기·PCA 색상 지도를 그림으로 저장 |
| `vision_encoder_visualization.py` | BEV·신호등·waypoint 그림을 만드는 공통 함수 |
| `inference_dashboard/` | 실시간 추론 상태·카메라·예측 경로를 보여주는 웹 화면 |
| `training_dashboard/` | 학습 로그·지표를 보여주는 웹 화면과 관측 보조 코드 |

오프라인 확인 흐름:

```text
저장된 이미지·LiDAR + 인코더 가중치
  → inspect_vision_tokens.py
  → inspection.json, tokens.npz 등 추출 결과
  → render_vision_tokens.py
  → 확인용 이미지
```

이 두 도구는 커넥터·LLM을 실행하거나 가중치를 학습하지 않습니다.
추출 도구는 인코더 forward를 실행하고, 렌더링 도구는 저장된 결과를 그립니다.
VLDrive 루트에서 옵션을 확인합니다.

```bash
python utils/visualization/inspect_vision_tokens.py --help
python utils/visualization/render_vision_tokens.py --help
```
