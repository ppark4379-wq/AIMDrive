# Notice 라벨과 MORAI 수집 연동

작성 기준: 2026-09-17, 현재 로컬 코드. 아래 연동 계약과 확장 Notice는 **설계 제안**이며
구현 완료를 뜻하지 않는다. 이번 변경은 기존 CARLA 파서/규칙의 주석과 이 문서뿐이다.
수집기, 데이터셋, 학습 입력, 기존 라벨 파일은 변경하거나 실행하지 않았다.

## 1. 사건 좌표는 로그인가?

아니다. CARLA 라벨러에는 서로 다른 두 입력이 있다.

| 입력 | 의미 | 읽는 위치 |
|---|---|---|
| 시나리오 정의 JSON | 어떤 종류의 사건을 어느 트리거 위치에 설정했는가 | 파서의 `scenarios_path` |
| 프레임별 측정 JSON | 실제 주행에서 ego가 어디에 있었는가 | `measurements_full/0000.json` 등 |

`Scenario3`는 CARLA 시나리오 타입 이름이며, 이 코드에서는 보행자 Notice에 대응한다.
MORAI 실험의 세 번째 시나리오, 폴더 번호, 로그의 세 번째 사건이라는 뜻이 아니다.
트리거 좌표는 보행자 현재 위치, 장애물 중심, 충돌 지점과도 구분해야 한다.

아래는 **라벨러가 읽는 키만 보여주는 예시**이며 실행 가능한 전체 시나리오 설정은 아니다.

```json
{
  "available_scenarios": [{
    "Town01": [{
      "scenario_type": "Scenario3",
      "available_event_configurations": [{
        "transform": {"x": 10.0, "y": 20.0}
      }]
    }]
  }]
}
```

예를 들어 전체 주행에서 이 좌표에 가장 가까운 것이 230번 프레임이고 거리가 1m라면,
코드가 자동으로 `frame_id=230, instruction=Notice-01, instruction_id=50`을 붙인다.
최근접 거리가 2m 이상이면 붙이지 않는다. 사람이 230번 이미지를 찾아서 타이핑하지 않는다.
사람은 사건 설정/종류를 마련하고 결과를 검수한다. 원본 CARLA에서는 기존 시나리오 정의를
사용할 수 있지만, 이 로컬 파서가 해당 정의 파일을 자동 생성하거나 다운로드하지는 않는다.

이것은 **위치 기반 대리 라벨**이다. 좌표에 가까이 갔다고 보행자가 실제 등장했거나,
앞차가 실제 급정거했다고 입증하지 않는다. 실제 사건 검증에는 GT/상태 이력이 추가로 필요하다.
전체 주행에서 최근접 프레임을 찾는 것은 오프라인 정렬이므로 그대로 실시간 검출에 쓰지 않는다.

## 2. 기존 Notice의 뜻과 한계

현재 ID는 파서의 등록 순서 및 `LMDrive/datasets/instruction_templates.json` 기준이다.

| 종류 / ID | 문장 의미 | 현재 판정 | MORAI에서 확보할 근거 |
|---|---|---|---|
| Notice-01 / 50 | 전방 보행자 주의 | Scenario3 트리거와 ego 거리 < 2m | pedestrian 분류, 위치, ego 경로 관련성 |
| Notice-02 / 51 | 전방 자전거 주의 | Scenario4 트리거와 ego 거리 < 2m | 자전거를 보행자/차량과 구분하는 실제 class/model |
| Notice-03 / 52 | 전방 차량 급정거 주의 | Scenario2/5 트리거와 ego 거리 < 2m | 동일 차량의 과거 속도/가속도, 시간, 선행 관계 |
| Notice-04 / 53 | 좌측 신호위반 차량 주의 | Scenario8/9 트리거와 ego 거리 < 2m | 해당 차량에 적용되는 신호, 정지선 통과, 진입 방향 |
| Notice-05 / 54 | 전방 신호위반 차량 주의 | Scenario7 트리거와 ego 거리 < 2m | 상대 차량 신호와 통과 이력, 경로 관련성 |
| Notice-06 / 55 | 불량 노면 주의 | Scenario1 트리거와 ego 거리 < 2m | 지도/시나리오 노면 구역과 실제 활성 상태 |
| Notice-07 / 56 | 터널 주의 | Town05의 고정 좌표 6곳과 거리 < 2m | MORAI 지도 터널 구역/입구, 진행 방향 |
| Notice-08-R / 57 | 전방 빨간 신호 주의 | 관련 신호 actor의 CARLA state=0 | ego 경로에 적용되는 신호 ID 및 정규화 상태 |
| Notice-08-G / 58 | 전방 초록 신호 주의 | 관련 신호 actor의 CARLA state=2 | 위와 같음 |
| Notice-08-Y / 59 | 전방 노란 신호 주의 | 관련 신호 actor의 CARLA state=1 | 위와 같음 |

01~06의 규칙은 영상/객체 움직임을 확인하지 않는다. 07/08은 프레임 15부터 검사하고
검출 후 통상 129프레임 뒤로 이동한다. 이 간격은 중복 억제이지 클립 길이가 아니다.
08은 actor의 위치도 필요하고 Town별 25 또는 45m에 난수 [0,10)m를 더한 문턱을 쓴다.
MORAI raw 신호값은 CARLA 상태 배열 인덱스로 바로 사용할 수 없다. 복합 신호와 불명 상태도
정의해야 한다. 특히 수집 스키마의 일반적인 `sta` 설명만 믿지 말고 실제 변환기를 확인해야 한다.

주의: 기존 `notice_processing_rules`에는 Notice-05가 두 번 있다. 이번에는 동작을 보존했지만,
새 MORAI 라벨러에서 이 중복과 비결정적인 거리 난수까지 그대로 복제할 필요는 없다.

## 3. 클립과 문장은 언제 만들어지는가?

파서는 `notice_instruction_list.json`에 route별 사건 프레임을 저장한다. 영상을 자르지 않는다.
LMDrive의 `CarlaInstructionDataset.__getitem__`는 navigation의 start/end 구간을 샘플링한 뒤:

1. 같은 route에서 `start < event.frame_id < exclusive_end - 16`인 사건만 후보로 삼는다.
2. 후보가 있어도 75% 확률로 Notice를 생략하고, 나머지 경우 하나를 고른다.
3. instruction ID에 대응하는 문구를 템플릿에서 선택한다.
4. `(event.frame_id - start) // sample_interval + 1`개의 visual frame 뒤에 삽입한다.

Notice는 모델에 주는 추가 입력이다. 라벨러가 영상을 읽고 자연어 정답을 생성하는 것은 아니다.
현재 VLDrive MORAI loader의 `instruction`은 navigation이다. 같은 이름의 Notice JSON을
만들었다고 별도 시점의 Notice 입력이 자동으로 연결되지는 않는다.

## 4. 현재 수집 코드와의 연결 지점

- `MoraiDrive/data_collection/collect.py`: 일반 수집에서 raw GT/신호 및 navigation 정보를 기록한다.
- `MoraiDrive/data_collection/sources/native_file_collector.py`: native 동기 수집에서
  `native_sync_frame`, `grpc_label_tick_raw`, 가용성 플래그, `gt_objects_raw`, `traffic_lights`,
  `navigation`을 저장한다. label snapshot이 없으면 객체 목록이 비어 있을 수 있다.
- `MoraiDrive/data_collection/schema.py`: ego 위치/속도, `actors_data`, 객체 종류별 hazard 필드의
  출력 스키마를 정의한다. 키가 존재한다고 데이터가 실제로 채워졌다는 뜻은 아니다.
- `MoraiDrive/data_collection/internal_sensor_record/`: 별도 Sensor Record/GT 수집 경로다.
  시나리오 생성/활성화 기록, raw actor stream, 센서 시각을 실제 발행 데이터와 연결해야 한다.
- `VLDrive/datasets/morai_sensor_record_clip_dataset.py`: 현재 navigation instruction과
  cached visual feature를 모델 입력으로 내보낸다. 별도 Notice 시점/문장 경로는 추가 구현 대상이다.

즉 **수집과 원천 데이터로는 연결되지만, Notice 생성과 VLDrive 학습 투입까지는 미완성**이다.
원래 CARLA 파서가 MORAI collector를 호출하거나, collector가 이 Notice 규칙을 자동 실행하지 않는다.

## 5. 수집팀과 합의할 계약

| 담당/항목 | 저장하거나 합의할 내용 | 이유 |
|---|---|---|
| 수집: 식별자 | session/episode/route, 연속 dataset frame_id, 원본 tick·timestamp 및 대응표 | 서로 다른 주행/프레임을 섞지 않기 |
| 수집: 좌표 | map ID/버전, ENU 원점, XY 단위, yaw 단위/방향, ego 기준점 | 트리거·GT·ego가 같은 좌표계여야 함 |
| 수집: 객체 | 안정적인 actor ID, 실제 종류/model, 위치·방향·크기·속도, source time | 같은 객체의 상태 이력과 전방 관련성 계산 |
| 수집: 신호 | 원본 signal ID/raw state, 정규화 state와 버전, ego 관련 링크, 정지선/구역 | 다른 방향 신호나 복합 상태를 오인하지 않기 |
| 수집: 사건 | event ID/type, 관련 actor/zone, 설정 트리거와 실제 start/end/활성 상태를 분리 | 시나리오 설정만 있고 사건이 실패한 경우 제외 |
| 수집: 품질 | GT 가용성, 수신 지연, 시간 정렬 결과, 누락/불명 구분 | 빈 배열을 '장애물 없음'으로 단정하지 않기 |
| 라벨링 | 위 정보로 종류·발화 시점·중복 억제 결정, evidence와 rule version 보존 | 임계값 변경 후 재생성/추적 가능 |
| 학습 | route/frame join, clip 내 Notice 삽입 위치, tokenizer/model 입력, 설정·검증 | 파일 생성과 실제 학습 사용은 별개의 단계 |

동기 수집은 해당 경로의 tick 일치 계약을 확인한다. 비동기 Sensor Record는 timestamp 기반
매칭과 허용 age/오차, 매칭 출처를 저장한다. 모든 데이터에 무조건 tick 숫자 같음을 요구하거나,
파일 이름 정렬만으로 다른 stream을 붙이지 않는다. 초 단위 중복 억제 정책을 저장 cadence로
변환하고, 128이라는 CARLA 상수를 다른 FPS에 그대로 적용하지 않는다.

GT로 객체가 존재한다는 사실과 카메라에서 보인다는 사실도 다르다. 가시성을 주장하려면
별도의 관측/가림 검증이 필요하다. 없으면 라벨을 경로 관련 GT 주의 정보로 명시한다.
급정거 등 사건 판정은 발화 시점까지의 과거/현재 GT로 계산한다. 미래 ego 궤적이나
미래 제동 명령을 보고 현재 '위험' Notice를 만들지 않는다.

## 6. 시나리오를 CARLA JSON으로 다시 만들어야 하나?

**필수는 아니다. MORAI 의미 중심 스키마와 변환기를 권장한다.**

- CARLA 형식으로 이름만 바꾸면 입력을 흉내 낼 수 있지만, 트리거 < 2m라는 대리 라벨의 한계는 남는다.
- 보행자/신호/정적장애물은 정렬된 현재 GT와 ego 경로에서 직접 판정할 수 있다.
- 터널/노면/지정 구간은 MORAI 지도 구역 정의가 필요하다.
- 스크립트로 발생시키는 급정거/횡단 사건은 실제 activation 이력을 함께 저장하면 검수가 쉽다.

새 event 기록의 **제안 필드**는 `schema_version`, `episode_id`, `map_id`, `event_id`,
`notice_type`, `start_frame`, `anchor_frame`, `end_frame`, `source_tick/time`,
`actor_ids`, `zone_id`, `evidence`, `rule_version`, `validity`다. 설정 trigger 좌표와
실제 발생/관측 기록은 별도 필드로 둔다. 값이 없으면 unknown으로 남기고 라벨을 추측하지 않는다.

라벨러 출력은 학습 경로에 맞춰 변환한다. LMDrive 호환 경로라면 route별 `frame_id`와
`instruction_id`로 변환할 수 있지만, 현재 VLDrive의 Notice 입력 연결은 별도 개발/테스트가 필요하다.
기존 navigation 구간을 Notice 문장으로 덮어쓰지 않는다. 양쪽 정보와 시점을 분리한다.

## 7. 추가 Notice 제안과 우선순위

| 우선순위 | 유형 | 기존 재사용/신규 | 검증 근거 |
|---|---|---|---|
| 1 | 전방 관련 보행자 | Notice-01 재사용 | actor class, 현재 경로와의 위치 관계 |
| 1 | 관련 차량 신호 | Notice-08 재사용 | 경로 관련 신호 ID, 정규화 상태, 적용 구역 |
| 1 | 전방 정적장애물 | `morai.static_obstacle_ahead` 신규 제안 | 객체 종류/정적 여부, 차체 폭 포함 경로 겹침 |
| 2 | 횡단 중인 보행자 | `morai.pedestrian_crossing` 신규 제안 | 보행자 이동 이력 + 횡단보도/ego 경로 관계 |
| 2 | 회전교차로 접근 차량 | `morai.roundabout_approaching_vehicle` 신규 제안 | 회전 차로·진행 방향·현재/과거 actor 이동 |
| 2 | 합류 차량 | `morai.merging_vehicle` 신규 제안 | 합류 링크/구역과 상대 차량 이동 이력 |
| 3 | 앞차 급정거 | Notice-03 재사용 | actor 이력, 유효 dt, 선행 관계, 감속 조건 |
| 보류 | 상대 차량 신호위반 | Notice-04/05 | 상대에게 적용되는 신호/정지선이 없으면 판정 불가 |
| 해당 시 | 자전거/노면/터널 | Notice-02/06/07 | 실제 해당 객체/지도 근거가 있을 때만 |

'횡단보도 앞이다'와 '보행자가 횡단 중이다', '회전교차로 진입이다'와 '접근 차량이 있다'는
다른 문장이다. 위치만으로 동적 사건을 확정하지 않는다. '우측으로 피해라' 같은 행동 지시는
단순 주의 Notice와 구분해, 경로의 안전성을 검증한 별도 instruction 정책에서 결정해야 한다.

신규 숫자 ID를 바로 60부터 쓰면 안 된다. 현재 60~64는 Other 지시문이 사용한다.
먼저 의미 키/템플릿 레지스트리 버전을 합의하고 parser, loader, template, model을 함께 맞춘다.

권장 진행 순서: 데이터 계약 합의 -> 한 주행의 정렬/가용성 검사 -> 신호·보행자·정적장애물
후보 생성 -> 장면 대조 검수 -> label export -> loader/model Notice 입력 및 단위 테스트 연결.
시나리오 정의 누락, 객체 누락, 신호 불명, 시간 오정렬 사례는 반드시 음성/unknown 테스트에 넣는다.
