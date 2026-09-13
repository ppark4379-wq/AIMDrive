# Sync Runtime 실행 기록

[Sync Runtime](../../sync_runtime/README.md)이 MORAI 동기식 주행 중 생성하는 캡처 메타데이터를 보관하는 폴더입니다. 센서 파일과 차량 상태를 프레임 기준으로 연결하거나 주행 문제를 분석할 때 사용합니다.

## 구성

```text
sync_runtime/
├── README.md
└── sync-<날짜>-<시간>-<ID>/
    └── capture_manifest.jsonl
```

실행 세션마다 별도 폴더가 생성되며, `capture_manifest.jsonl`에는 캡처 대상 프레임마다 한 줄의 JSON 기록이 추가됩니다.

| 주요 필드 | 내용 |
|---|---|
| `session_id`, `scenario_id` | 실행 세션과 시나리오 식별자 |
| `native_sync_frame`, `request_frame`, `response_frame` | 캡처 기준 프레임과 스텝 진행 전후 프레임 |
| `ego_state_for_frame`, `next_ego_state` | 현재 프레임과 다음 프레임의 차량 위치·속도·조향 등 상태 |
| `applied_control`, `control_mode` | 적용한 제어 명령과 새 명령·재사용·제동 상태 |
| `capture_requested`, `capture_stride` | 센서 캡처 요청 여부와 주기 |
| `expected_sensor_files` | 해당 프레임의 카메라·LiDAR 예상 파일 경로 |

## 사용 관계

- `MoraiDrive/sync_runtime/`의 실행 코드가 이 기록을 생성합니다.
- `data_collection`의 `sync_native_files` 모드가 이 기록과 MORAI가 별도로 저장한 센서 원본 파일을 읽어 데이터셋을 만듭니다.
- `diagnostics/`의 주행 조사에서 차량 궤적과 제어 상태를 분석하는 근거로 사용합니다.

이미지·LiDAR 데이터 자체는 이 JSONL에 포함되지 않으며, 캡처 요청이나 예상 경로가 기록되었다고 실제 파일 저장까지 보장되는 것은 아닙니다.

자동 생성되는 실행 결과이지만 캐시는 아닙니다. 삭제하면 해당 세션의 프레임 대응 정보와 주행 분석 근거가 사라지므로, 실행·수집이 끝나고 기록이 더 필요 없을 때 정리합니다.
