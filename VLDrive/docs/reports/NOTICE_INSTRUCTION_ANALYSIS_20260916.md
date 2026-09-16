**LMDrive notice instruction과 현재 VLDrive 구현 검증 — 2026-09-16**

LMDrive의 notice는 학습용 데이터에서 자동으로 라벨링되며, instruction finetuning의 입력과 선택적인 실시간 추론 입력으로 사용된다. 모든 종류가 measurements와 actors_data만으로 생성되는 것은 아니다. CARLA 돌발상황 유형은 시나리오 정의도 필요하다. 현재 VLDrive 구현에는 navigation 입력 경로가 있고, 시간에 따라 들어오는 notice의 별도 입력 경로는 연결되어 있지 않다.

근거는 첨부 LMDrive.pdf 14쪽, VLDrive.pdf 11쪽, 공식 LMDrive GitHub, 현재 작업 공간의 코드와 실제 수집 파일이다. 논문은 연구 내용의 근거로 읽었다. 공식 저장소는 `308d8da0b4ff3463b1649fe483a9b2c56b108189` 커밋을 내려받아 확인했다. 작업 공간의 `LMDrive/`는 재구성된 구현이므로 공식 원본과 구분했다.

**1. 어떤 도구가 notice를 만드는가**

공식 저장소에는 GUI가 아닌 Python 규칙 기반 라벨러가 있다.

| 역할 | 공식 코드 |
|---|---|
| 경로 목록·수집 프레임·시나리오를 읽어 notice 라벨 저장 | [tools/data_parsing/parse_notice.py](https://github.com/opendilab/LMDrive/blob/308d8da0b4ff3463b1649fe483a9b2c56b108189/tools/data_parsing/parse_notice.py#L91) |
| notice 종류별 판정 | [notice_rules.py](https://github.com/opendilab/LMDrive/blob/308d8da0b4ff3463b1649fe483a9b2c56b108189/tools/data_parsing/notice_rules.py) |
| instruction ID → 문장 후보 | [instruction_dict.json](https://github.com/opendilab/LMDrive/blob/308d8da0b4ff3463b1649fe483a9b2c56b108189/leaderboard/leaderboard/envs/instruction_dict.json) |
| navigation 클립 생성 | [parse_instruction.py](https://github.com/opendilab/LMDrive/blob/308d8da0b4ff3463b1649fe483a9b2c56b108189/tools/data_parsing/parse_instruction.py) |

LMDrive.pdf §3, PDF 3–4쪽은 전문가 주행을 먼저 수집하고 이후 클립을 파싱해 지시문을 붙이는 순서를 설명한다. 따라서 passenger의 실제 발화를 함께 녹음해 저장한 데이터라는 뜻이 아니다. 논문에서 ChatGPT API는 같은 의미의 문장 표현을 다양화하는 데 사용한다. 공개 파서는 매 장면을 LLM에 보내 설명문을 생성하지 않는다.

파서는 `dataset_index.txt`를 읽고 경로 이름에서 town/weather를 추출한다. 각 경로의 `measurements_full/0000.json`, `0001.json`, …을 읽은 다음 `leaderboard/data/scenarios/townXX_all_scenarios.json`을 읽는다. 출력은 route별 event 목록인 `notice_instruction_list.json`이다. README 일부에 `.txt`라고 적혀 있지만 실제 writer와 loader는 `.json`을 사용한다.

| 종류 / ID | 의미 | 실제 판정 근거 |
|---|---|---|
| Notice-01 / 50 | 전방 보행자 | Scenario3 trigger 위치와 ego x,y |
| Notice-02 / 51 | 전방 자전거 | Scenario4 trigger 위치와 ego x,y |
| Notice-03 / 52 | 앞 차량 급정지 | Scenario2 또는 Scenario5 trigger 위치와 ego x,y |
| Notice-04 / 53 | 좌측 차량 신호위반 | Scenario8 또는 Scenario9 trigger 위치와 ego x,y |
| Notice-05 / 54 | 전방 차량 신호위반 | Scenario7 trigger 위치와 ego x,y |
| Notice-06 / 55 | 불량 노면 | Scenario1 trigger 위치와 ego x,y |
| Notice-07 / 56 | 터널 | Town05의 코드에 적힌 위치 목록과 ego x,y |
| Notice-08-R/G/Y / 57/58/59 | 빨강/초록/노랑 신호 | affected_light_id, actors_data[id].loc, actors_data[id].sta, ego x,y, town |

Notice01–06은 해당 trigger와 가장 가까운 기록 프레임을 찾고 거리가 2m 미만이면 붙인다. 실제 actor 궤적을 분석해 급정지나 신호위반을 입증하는 검출기가 아니다. 위치 기반 시나리오 라벨이므로 실제 이벤트 발생과 정확히 일치하는지는 별도 확인 대상이다.

Notice08은 `affected_light_id`가 유효하고 해당 actor가 있어야 한다. CARLA state의 0/1/2를 red/yellow/green으로 읽으며, town별 25m 또는 45m에 무작위 0–10m를 더한 거리 조건을 적용한다. 프레임 15부터 검사하고 검출 후 인덱스를 128만큼 건너뛰므로 모든 프레임에 알림을 반복하지 않는다. 원본 스크립트에는 `Notice-05`가 processing list에 두 번 들어 있는 중복도 있어 이식할 때 그대로 복제하지 않는 편이 좋다.

**2. 실제 수집 데이터로 만들 수 있는가**

현재 기본 학습 설정의 입력은 `MoraiDrive/dataset_local/competition_train`이다. `data/full_loop_basic_drive_2026_09_09_01_44_15`의 4,240개 `measurements_full` 파일을 전수 검사했다.

| 검사 | 결과 |
|---|---:|
| actors_data가 비어 있지 않은 프레임 | 0 / 4,240 |
| gt_objects_raw가 비어 있지 않은 프레임 | 0 / 4,240 |
| traffic_lights가 비어 있지 않은 프레임 | 2,727 / 4,240 |
| affected_light_id가 -1이 아닌 프레임 | 2,740 / 4,240 |
| 기존 notice_instruction_list.json | `[]` |
| 공식 Notice08을 수정하지 않은 프레임에 직접 실행 | R/G/Y 모두 0개 |

원본 Notice08의 직접 실행에서는 actors_data 누락으로 종료되는지 확인했다. town 인수 `01`은 그 분기를 실행하는 검사 인수이며 MORAI를 Town01로 간주한 것이 아니다. 파서 전체를 그대로 실행하려면 추가로 CARLA 경로 이름과 시나리오 파일 조건도 충족해야 하므로 MORAI에 바로 적용할 수 없다.

하지만 MORAI의 `navigation.affected_light_id_raw`, `affected_light_state_raw`, `affected_light_state_lmdrive`와 `traffic_lights`에는 신호 정보가 있다. 실제 frame 243의 요약은 다음과 같다.

```json
{
  "frame_id": 243,
  "native_sync_frame": 1376,
  "grpc_label_tick_raw": 1376,
  "actors_data": {},
  "navigation": {
    "route_valid": true,
    "affected_light_id_raw": "C1256W000077",
    "affected_light_state_raw": 1,
    "affected_light_state_lmdrive": 0
  },
  "traffic_lights": [{"tl_id": "C1256W000077", "state": 1}]
}
```

ego에 연결된 signal ID와 실제 관측 signal ID가 일치하고, signal tick도 native frame과 같다. 따라서 빨간 신호에 대한 notice 후보를 만들 수 있다. 같은 신호가 frame 1000에서는 raw state 16, 변환 state 2로 기록되어 초록 신호 후보도 가능하다.

검증 스크립트는 route 유효성, signal ID/state 일치, gRPC tick 정렬을 확인하고 단순 red/yellow/green 상태만 채택했다. 그 결과 2,558프레임에서 문장 생성 근거를 확인했고, 동일 신호·색의 연속 구간 시작점 기준으로 **12개 후보: 빨강 7개, 초록 5개**를 생성했다. 순수 노랑 사례는 없었다. 미확인 상태 70프레임과 복합 상태 33인 112프레임은 후보에서 제외했다. 관측 공백에서 구간이 나뉘므로 12개는 서로 다른 신호등이나 12번의 독립 주행 상황이라는 뜻이 아니다.

후보에는 원본 measurement 위치와 native tick을 저장했다. CARLA의 신호등 거리 기준을 재현한 것은 아니며, 정지선·전방 관련성·발화 타이밍을 완성한 학습 라벨도 아니다. 원본 dataset의 notice 파일과 학습 설정은 바꾸지 않았다.

더 최근의 `VLDrive/data/scenario_campaign_20260914_155854_lap_001/episode_000`도 확인했다. 이 수집본은 actors_data 폴더 대신 raw journal에 다음 정보가 있다.

| 기록 stream | 개수 |
|---|---:|
| actor_inventory | 7,637 |
| objects | 21,963 |
| traffic_light_inventory | 3,171 |
| traffic_light | 6,178 |

기록 인덱스의 byte offset을 따라 실제 raw JSON payload를 읽었다. actor_inventory에는 ego와 주변 actor의 종류·좌표·회전·속도·가속도·bounding box가 있고, actor_identity에는 모델명도 있다. 같은 snapshot에서 ego heading으로 객체 좌표를 회전해 전방 0–25m, 횡방향 ±3m라는 간단한 조사 조건을 적용했을 때 다음 사례를 확인했다.

| 실제 record_id / tick | 대상 | ego 전방 거리 | 횡방향 거리 | 가능한 문장 의미 |
|---|---|---:|---:|---|
| 40799 / 4481 | CargoBox, actor 1 | 24.79m | +1.92m | 전방 장애물 주의 |
| 56292 / 5384 | Man5, pedestrian 5 | 22.08m | −2.92m | 전방 보행자 주의 |

이는 실제 수집 데이터로 주변 상황에 대한 notice 후보를 생성할 수 있다는 근거다. 속성만으로 횡단 의도·급정지·신호위반까지 확정할 수는 없다. 그런 문장은 actor history, 차로 관계, 이벤트 조건을 추가로 확인해야 한다. 장애물 주의는 LMDrive의 기존 10개 notice ID와 정확히 대응하지 않는 확장 유형이다.

이 새 수집본은 manifest에 `training_visible: false`로 기록되어 있다. 위 actor snapshot을 학습용 camera/LiDAR frame과 정렬해 notice를 붙이는 단계까지 완료했다는 뜻은 아니다. 특히 `record_id`, native tick, 가공 후 연속 `frame_id`는 서로 다른 식별자다.

**3. notice와 scene은 어떻게 mapping되는가**

공식 LMDrive의 기본 join key는 **route_path + 원본 frame_id**다. navigation은 시작·종료 프레임을 갖는 클립이고, notice는 그 경로의 특정 시점 event다. 같은 notice event가 겹치는 여러 navigation 클립의 후보가 될 수 있다.

```json
{
  "data/full_loop_basic_drive_2026_09_09_01_44_15": [
    {"frame_id": 243, "instruction_id": 57, "instruction_args": []}
  ]
}
```

이 예시는 실제 frame 243으로 만든 후보를 공식 join 구조로 줄여 표현한 것이다. CARLA 원본 parser가 이 MORAI 경로에서 산출한 라벨은 아니다.

학습의 [CarlaVoiceDataset.__getitem__](https://github.com/opendilab/LMDrive/blob/308d8da0b4ff3463b1649fe483a9b2c56b108189/LAVIS/lavis/datasets/datasets/carla_dataset_llm.py#L199)은 augmentation 후 확정된 start/end로 notice를 선택한다.

```text
같은 route의 notice 목록
  → start_frame < event.frame_id < end_frame - 16 인 것만 선택
  → 후보가 있더라도 75% 확률로 notice 제거
  → 남는 경우 하나를 무작위 선택
  → instruction_id로 문장 표현 하나 선택
  → notice_frame_id = (event.frame_id - start_frame) // sample_interval + 1
```

`notice_frame_id`는 절대 frame 번호가 아니라 **notice 앞에 배치할 샘플링된 visual frame의 개수**다. 예를 들어 학습 클립의 최종 start=220, exclusive end=300, sample_interval=2이고 notice event=243이면 값은 12다. visual frame 220, 222, …, 242의 뒤, 244의 앞에 notice text가 들어간다. 센서 파일도 같은 route와 frame 번호로 읽기 때문에 장면과 지시가 함께 연결된다. 끝에서 16프레임을 남기는 조건의 단위는 샘플링 후 token 개수가 아니라 원본 frame 번호다.

현재 MORAI VLDrive의 navigation은 `route_id + [start_frame, stop_frame]`으로 text를 매핑한다. 이 구현은 stop_frame을 포함한다. LMDrive loader의 exclusive end와 혼동하지 않아야 한다. 새 raw journal을 변환할 때는 `episode/session + source timestamp/tick → sensor frame → dataset frame`의 대응도 보존해야 한다. 생성한 파일 이름의 정렬 순서만으로 서로 다른 stream을 연결해서는 안 된다.

**4. 학습과 추론의 정확한 입력 위치**

| LMDrive 단계 | notice 사용 | 위치 |
|---|---|---|
| Vision encoder 사전학습 | 사용하지 않음 | 단일 센서 프레임과 perception/waypoint supervision |
| Instruction finetuning | 선택적으로 사용 | loader의 notice_text/notice_frame_id → LLM 시퀀스 |
| LangAuto 기본 추론 | notice 없이 가능 | agent_use_notice=False 기본값 |
| LangAuto-Notice 추론 | 사용 | 실시간 planner의 notice → 동일한 LLM 시퀀스 구성 |

공식 [학습 YAML](https://github.com/opendilab/LMDrive/blob/308d8da0b4ff3463b1649fe483a9b2c56b108189/LAVIS/lavis/projects/lmdrive/notice_llava15_visual_encoder_r50_seq40.yaml)은 dataset의 `enable_notice: True`와 model의 `use_notice_prompt: True`를 모두 설정한다. 하나는 데이터를 준비하는 설정이고 다른 하나는 모델에 그 입력을 넣는 설정이다.

공식 모델의 [concat_text_image_input_with_notice](https://github.com/opendilab/LMDrive/blob/308d8da0b4ff3463b1649fe483a9b2c56b108189/LAVIS/lavis/models/drive_models/drive.py#L226)는 notice를 tokenizer와 LLM embedding에 통과시킨 후 지정한 시점에 삽입한다.

```text
센서 → Vision encoder → Q-Former → projection → visual tokens
navigation text ───────────────────────────────┐
notice text → tokenizer → embedding ──────────┤
                                             ↓
LLM 입력: [navigation] [V0 … V(k−1)] [notice] [Vk … V(T−1)]
                                             ↓
                           waypoint head + completion head
                                             ↓
                 학습: L1 waypoint loss + 0.2 × completion CE
                 추론: waypoint → PID → steer/throttle/brake
```

코드에서 Q-Former에 직접 들어가는 텍스트는 `samples['text_input']`, 즉 navigation이다. notice는 Q-Former 뒤의 LLM 입력 결합 단계에 들어간다. `instruction_id`는 문장 lookup용이며 숫자 class ID 자체를 LLM에 넣는 것이 아니다. raw measurements/actors_data 전체를 LLM prompt에 붙이는 구조도 아니다.

notice는 문장 생성 정답이 아니라 주행 예측의 조건 입력이다. 학습 대상은 그 지시와 장면을 보고 실제 주행 궤적을 맞추는 행동이다. 기본 LMDrive는 LLM 본체와 vision encoder를 고정하고 Q-Former, projection/출력 adapter 등을 학습한다. 따라서 notice를 사용해 학습한다는 말이 반드시 LLM 본체의 모든 weight를 업데이트한다는 뜻은 아니다. 논문 PDF 6쪽, 7쪽과 부록 PDF 12쪽은 학습 및 notice 사용률을 설명한다.

추론 시 [lmdriver_agent.py](https://github.com/opendilab/LMDrive/blob/308d8da0b4ff3463b1649fe483a9b2c56b108189/leaderboard/team_code/lmdriver_agent.py#L467)는 `InstructionPlanner.pos2notice(sampled_scenarios, tick_data)`와 `traffic_notice(tick_data)`를 호출한다. [planner.py](https://github.com/opendilab/LMDrive/blob/308d8da0b4ff3463b1649fe483a9b2c56b108189/leaderboard/team_code/planner.py)는 현재 ego 위치와 scenario trigger, CARLA의 현재 신호등 상태로 템플릿을 선택한다. offline dataset의 notice JSON을 현재 운전에 다시 검색하는 방식은 아니다.

`agent_use_notice=True`이면 agent는 `notice_text`, `notice_frame_id`를 모델에 넘긴다. 새 notice의 위치는 현재 visual buffer의 `T−1`로 설정되어 현재 마지막 visual frame 앞에 놓인다. navigation 전환 또는 completion 판정 때 buffer와 notice가 초기화된다. 이 benchmark 입력은 시뮬레이터 정보를 이용해 passenger/보조 시스템의 알림을 모사한 것이다. 실차 적용에서는 사람 발화나 별도 온라인 인지 시스템이 해당 입력을 제공해야 한다.

**5. VLDrive 논문과 현재 구현에 적용할 결론**

VLDrive.pdf §3.1/Fig.2(PDF 4쪽), §3.5/Fig.4 및 §4.1(PDF 5–6쪽)은 navigation instruction, 시각 토큰, DDIA를 명시한다. 첨부본 전체에서 notice라는 용어와 LMDrive식 notice 삽입 인터페이스는 확인되지 않았다. LMDrive dataset을 사용했다는 사실만으로 VLDrive 저자가 notice를 동일하게 사용했다고 확정할 수 없다. 저자의 실제 학습 설정은 이 논문만으로 확인 불가다.

현재 코드의 실제 경로는 다음과 같다.

| 코드 | 확인 내용 |
|---|---|
| `configs/vldrive_train.json` | full_loop.jsonl navigation을 지정 |
| `tools/build_instructions.py` | global path command와 선택적인 route lookahead로 navigation 생성 |
| `datasets/morai_expert_dataset.py:202` | route와 frame 구간으로 navigation text 조회 |
| `datasets/morai_expert_dataset.py:301` | model_inputs에 instruction 전달 |
| `models/vldrive.py:159` | visual input과 instruction 두 key만 허용 |
| `models/language_model.py:84` | instruction tokenization/embedding |
| `models/multimodal_sequence.py:20` | [instruction prefix, visual sequence] 구성 |
| `models/ddia_attention.py:9` | text끼리 self-attention, visual에서 모든 instruction을 볼 수 있는 mask |

`tools/labeling/notice_rules.py`, `parse_notice_instructions.py`는 있지만 현재 MORAI 학습 loader/model에 연결되어 있지 않다. 따라서 라벨 파일 생성만으로 notice 학습이 켜지지는 않는다.

VLDrive에 notice를 추가하려면 다음 작업이 필요하다.

1. **MORAI event 라벨러:** 수집 형식에 맞는 상태/geometry 규칙, source timestamp, episode/route/frame, evidence, type/text, 발화·종료 시점을 저장한다. 시나리오 설정만 있으면 실제 발동 여부도 확인한다.
2. **Loader 연결:** 실제 학습 window에 들어온 notice만 선택한다. LMDrive의 75% 제거는 재현을 위한 시작점이며 새 데이터에서 검증할 설정이다.
3. **모델 입력과 DDIA 시간 처리:** notice 발생 시점과 가시성을 표현한다. 현재 DDIA는 모든 instruction 간 상호 attention을 허용한다. LMDrive처럼 중간에 text를 삽입하는 것만으로는 이전 visual frame에 미래 notice가 보이는 문제와 navigation token을 통한 간접 유출을 막을 수 없다. event 시각 기반 mask를 설계하거나, 현재 유효한 instruction prefix를 사용하는 window에서 그 시점 이후/현재 예측만 supervision하는 정책이 필요하다.
4. **추론 공급 경로:** 실제 사용자/온라인 모듈이 notice와 도착 시각을 제공한다. notice가 없어도 주행하는 경우와 함께 검증한다.

위 항목은 이번 조사의 구현 제안이다. 논문 재현에 notice가 필수라고 가정해 연결하지는 않았다. notice의 근거 데이터, 라벨링 가능성, 현 구현의 연결 유무를 검증했으며 모델 학습이나 폐루프 주행 실험은 수행하지 않았다.

**검증 산출물**

- [audit_data.py](notice_instruction_audit_20260916/audit_data.py): 전체 구형 measurement 검사, 공식 Notice08 실행, MORAI 후보 생성, 최신 actor journal 검사.
- [audit_results.json](notice_instruction_audit_20260916/audit_results.json): 프레임 집계, 실제 measurement 예시, 최신 raw payload 근거.
- [notice_candidates.json](notice_instruction_audit_20260916/notice_candidates.json): 신호등 후보 12개와 원본 데이터 연결.
- [newer_actor_examples.json](notice_instruction_audit_20260916/newer_actor_examples.json): 실제 전방 객체 사례와 raw byte offset.

작업 공간 루트에서 `python VLDrive/docs/reports/notice_instruction_audit_20260916/audit_data.py`로 재현한다. 공식 코드 사본은 `VLDrive/.runtime/notice_audit/upstream`에 있고 커밋/해시는 `.runtime/notice_audit/sources.json`에 있다. 스크립트는 보고서 폴더의 조사 결과만 기록한다.
