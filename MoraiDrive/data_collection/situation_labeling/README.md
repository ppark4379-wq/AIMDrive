# Situation instruction features v0.3

회전교차로에서 매칭된 Expert 관측이 있으면 Expert phase와 진입·합류 플래그에
따라 Expert 매핑에서 선택한 instruction을 우선 사용합니다. 이 매핑의 기존
`0.2m/s` 정지 기준과 `0.3m/s` 목표 속도 차이 기준은 유지합니다.
이후 규칙 기반 판정이 해당 instruction을 다른 문장으로 덮어쓰지 않습니다.
Expert 관측이 없는 경우의 기존 규칙 기반 라벨링은 유지합니다.
객체 위치 등 필수 변수가 부족하면 Expert 라벨은 보존하되 문장은 생성하지 않습니다.

## 수집 episode에서 한 번에 라벨링

수집 완료된 **Internal Sensor Record episode**에 아래 명령 하나를 실행합니다.
저장소 루트(AIMDrive)에서, 기존 collector 전처리에 사용하는 **WSL/ROS 분석
Python 환경**으로 실행합니다. Windows 기본 Python만으로 ROS 원본 메시지를
역직렬화할 수는 없습니다.

```bash
python -m MoraiDrive.data_collection.situation_labeling.label_episode --episode /mnt/g/path/to/run/episode_000
```

`--episode`에는 `episode_000` 또는 episode가 하나인 부모 run 폴더를 지정합니다.
run의 `capture_contract.json`, `sensor_contract.json`, `plan.json`과 episode의
`episode.json`, 수집 완료된 `raw/`, `quality/`가 필요합니다. 센서 이미지 파일만
모은 폴더는 이 collector의 입력 계약을 충족하지 않습니다. raw·quality는 수집
중 저장되는 기록이며 이 명령을 위해 사용자가 미리 GT·geometry를 생성할 필요는 없습니다.

명령은 다음 순서로 실행합니다.

1. 수집 완료 기록을 확인하고 수집 당시 route/map과 원본 센서 경로를 검증합니다.
2. native audit → association → GT timeline → traffic/actor audit → geometry V2
   → route/motion/exposure audit → review → admission을 순서대로 실행합니다.
3. `build_labels`를 실행하여 feature 생성 → 행동 판정 → Template 선택 → 문장
   parameter 삽입 → 프레임별 instruction 저장을 완료합니다.
4. 원본 센서 파일 경로와 instruction을 같은 행으로 묶은 최종 결과를 생성합니다.

정적 장애물은 해당 run의 `supervisor.log`가 있으면 자동으로 연결합니다.
로그 위치가 다르면 `--static-supervisor-log /path/to/supervisor.log`를 지정합니다.
로그가 없으면 rule-only로 실행됨을 출력하며, Expert phase를 사용한 viewer 결과와
동일한 결과를 보장하지 않습니다. 기존 라벨/문장 규칙의 지원 범위는 그대로 적용되므로
이 명령이 미구현 Zone 규칙을 추가하지는 않습니다.

전처리는 episode 아래 기존 결과를 해시 검증 후 재사용합니다. V1 결과를 덮어쓰지
않도록 geometry는 `object_geometry_audit_v2`, review는 `situation_review_v2`,
admission은 `situation_admission_v2`를 사용합니다. geometry catalog가 별도 위치에
있으면 최초 실행에 `--catalog-report PATH`를 지정합니다. 기존 geometry가 있는
경우 그 생성 당시 catalog를 사용하며 새 catalog 옵션을 조용히 무시하지 않습니다.

교정된 V5 기하가 이미 있는 episode는 `--geometry-tag TAG`로 그 묶음을 그대로
사용합니다. `object_geometry_audit_v5_TAG`, `review_closeout_v5_TAG`,
`admission_v5_TAG` 세 폴더를 함께 검증해 사용하고, V2 geometry/review/admission은
생성하지 않습니다. 비전 인코더 학습 데이터와 같은 박스를 쓰려면 이 옵션을 사용합니다.
수집 당시 저장소에서 옮겨진 run은 `--native-root RUN_DIR`로 센서 파일 위치를
지정합니다(참조하는 모든 파일의 존재는 그대로 검사합니다). Expert가 tick별
상태를 남기기 전에 수집된 회전교차로 run은 `--roundabout-supervisor-log RUN/supervisor.log`를
함께 지정합니다. 분석 Python은 ROS 경로와 함께 cv2·pyproj·scipy를 불러올 수
있어야 하며, 이 PC에서는 `--analysis-python ~/miniconda3/envs/lmdrive/bin/python3.8`을 사용했습니다.

```bash
python3 -m MoraiDrive.data_collection.situation_labeling.label_episode \
  --episode RUN/episode_000 --geometry-tag d4a53be1fe67 --native-root RUN \
  --analysis-python ~/miniconda3/envs/lmdrive/bin/python3.8 --output NEW_DIRECTORY
```

기본 결과는 `EPISODE/situation_labels/`에 생성됩니다. `--output NEW_DIRECTORY`로
바꿀 수 있으며, 기존 결과 디렉터리는 덮어쓰지 않습니다.

```text
situation_labels/
  labeled_frames.jsonl       # 최종 결과: 프레임별 원본 센서 파일 + instruction
  report.json               # 성공/실패, 라벨 유무 개수, 경로·입력 근거
  COMPLETE.json             # 모든 단계가 성공했을 때만 생성
  pipeline.log              # 전처리·라벨링 하위 도구 실행 로그
  features/
    frame_features.jsonl
    object_features.jsonl
    situation_rules.jsonl   # 기존 viewer가 읽는 instruction 결과
    report.json
```

`labeled_frames.jsonl`의 각 행은 `episode_path`, `frame_id`, `sensor_time_ns`,
`assets`, `zone`, `instruction`, `labeling`, 프레임 유효성 필드를 포함합니다.
`assets`는 원본 센서의 경로·해시·크기 등 admission 메타데이터를 연결합니다.
센서 파일을 복사하거나 원본 manifest를 수정하지 않는 참조형 결과이므로 원본 센서
저장소도 유지해야 합니다. `labeling`에는 내부 상태·선택한 Template·판정 근거가
들어갑니다. 미판정 프레임도 `instruction: null`로 남으며 학습 가능성을 임의로
올리지 않습니다. 완료는 실행 성공을 뜻하며 모든 프레임에 문장이 있다는 뜻은 아닙니다.

실패하면 `report.json`에 오류를 기록하고 완료 표시는 생성하지 않습니다.
원인을 해결한 뒤 새 `--output`으로 실행하면 완료된 전처리를 재사용합니다.
전처리 디렉터리가 미완료이거나 해시가 달라졌으면 자동으로 지우거나 덮어쓰지 않고
해당 경로에서 중단합니다. 마지막 stdout JSON에도 최종 결과 파일 경로를 반환합니다.

기존 viewer에는 `--features-dir EPISODE/situation_labels/features`를 전달하면 됩니다.

## 전처리 완료 데이터에서 feature/문장만 재생성

Run 'python -m MoraiDrive.data_collection.situation_labeling.build_labels' with
'--episode', '--route-definition', '--global-path', and a new '--output'
directory. The command writes frame_features.jsonl, object_features.jsonl,
situation_rules.jsonl, and report.json. Existing output directories are not
overwritten.

The output uses instruction_state values rather than S/R numbers. A matched
state describes current evidence and supplies sentence variables. It is not
an Expert action label or an entry/trajectory safety certificate. The original
frame valid and invalid_reasons fields remain visible; object geometry
quality is checked internally and rule_geometry_usable is persisted. Rules check
nearby actors after spatial filtering. A full, usable object inventory is still required.

Internal feature calculation provenance:
- Ego direction is derived from adjacent admitted GT positions; a held
  direction is marked ego_heading_source=held_prior_motion. Acceleration is a
  backward speed difference.
- MGeo link IDs come from the local official map's link_set.json when present.
  A match farther than 3 m has no link ID.
- The roundabout merge point is explicit in the official route YAML and
  matches Expert ROUNDABOUT_ENTRY_MERGE_POINT.
- A lead bumper gap is a current-frame longitudinal projection of oriented GT
  boxes. It requires a fresh Ego motion heading, a matched circulating link,
  compatible actor heading, and lateral proximity.
- Pedestrian path clearance uses an orientation-independent enclosing circle.
  It describes present path overlap, not a future collision prediction.
- Static passage gap is a conservative geometric candidate; connected
  drivable corridor and road-boundary safety are not inferred.

Current sealed GT has no LiDAR-frame-aligned Expert future OBB collision result.
Roundabout collision_risk is a current merge ETA overlap heuristic; it is
null when key traffic evidence cannot be resolved. future_collision_predicted
remains null internally and is not persisted. The same fail-closed behavior applies to avoidance and
between-obstacle passage when a safe trajectory is not verified.
The internal resume_candidate describes a possible transition, not an Expert resume event.

Highway Rule rows also emit `label_candidate` (`KEEP_LANE`, `ACC_FOLLOW`,
`CHANGE_LEFT`, `CHANGE_RIGHT`) and a separate `behavior_state`. A lane change
is inferred from consecutive Ego GT positions relative to adjacent MGeo lane
centers; aborted changes use the actual return direction. Same-lane lead
bumper gaps use oriented GT boxes. Target-lane front/rear `gap_m` uses MGeo
center distance, as in Expert; separate bumper gap fields use GT boxes. The
initial target-lane front/rear thresholds match Expert (20 m / 12 m plus
1.5 s of closing speed). `target_lane_collision_safe` remains null because
frame-aligned Expert trajectory validation is unavailable. Unresolved nearby
vehicle geometry or lane motion produces an unknown label.

## Feature definitions

config/feature_definitions.yaml lists only fields persisted for rule decisions,
sentence rendering, and alignment to existing Expert observations. Shared
inputs live under common.frame and common.object; zone-specific inputs live
under zones.<zone>.frame and zones.<zone>.object. Dynamic obstacle uses a
feature Rule without an Expert label. Other zones can attach aligned Expert
labels when available.

build_labels computes its intermediate geometry and motion features in memory,
evaluates the Rule, then writes only the fields in this YAML to
frame_features.jsonl and object_features.jsonl. Removing an intermediate from
the saved rows does not change the current Rule calculation. Zone fields are
written only for that zone. The YAML states type, unit, source, meaning, and
evidence level; editing it changes output selection, not the formulas.

The build validates the saved rows for missing common fields, unexpected zone
fields, unknown fields, and wrong types. Existing full-feature outputs are not
rewritten. To inspect one with the reduced contract, use:

    python -m MoraiDrive.data_collection.situation_labeling.feature_definitions --features-dir situation_features_dynamic_lap060 --allow-legacy-extra

Use --allow-legacy-missing as well for older outputs that predate current
common fields. Extra fields are reported, not silently treated as Rule inputs.

## Instruction mapping and optional Expert postprocessing

`build_labels` now maps a matched current-frame `instruction_state` through
`config/label_instruction_map.yaml` and renders a sentence with
`config/instruction_catalog.yaml`. `mapper.py` selects the ID and `renderer.py`
checks and fills its variables. The rule state, `instruction_source`,
`instruction_id`, and rendered `instruction` remain separate fields. A
highway `label_candidate` is used only as a candidate and never becomes an
Expert `label`. Dynamic obstacle has no Expert-sourced label (no Expert state
contract exists for it): `dynamic_obstacle_rule`'s `instruction_state` values
(`approach`, `yield_start`, `yielding`, `waiting`, `resume`, `normal`,
`behavior_unverified`) are the only label dimension and map through
`label: null` rows like static obstacle's rule-only states.

With `--expert-log PATH`, existing Expert observations are attached first.
Repeat the option to combine Roundabout JSONL and Expert text logs. An
observed Expert label plus a matched rule state selects an exact mapping; a
conflicting pair leaves the sentence null. Collection and Expert runtime are
unchanged.

Existing `*_roundabout.jsonl` plan records and `[HighwayDecision]` lines in
`driving_debug_*.log` have Ego XY but are sparse. Postprocessing assigns only
a unique Ego position match within 0.5 m; it never carries a state forward
between log entries. An independently supplied journal containing actual ROS
`ego_header_seq` plus `ego_source_ns` can also be joined to
`quality/active_signals.json` within a 25 ms source-time gap. The older
`ego_seq` timing field is an internal callback count and is never a join key.
Both methods report `physical_simultaneity_verified: false`.
The parser currently reads Roundabout plan JSONL and `[HighwayDecision]` text
lines. `[STATIC]` lines in `supervisor.log` are not frame aligned because they
have no sensor source timestamp or header sequence; they are not converted
to Expert labels.

As of this change, Expert's own `roundabout_entry` planner emits one
`[EXPERT_STATE] {...}` JSON line per tick (not deduped) directly into
`supervisor.log` during real collection, carrying true `ego_header_seq`/
`ego_source_ns` plus `roundabout_phase` (`cruise`/`follow`/`yield`/
`collision_slowdown`/`emergency`/`rear_pressure`) and the other `RAW_FIELDS`
`expert_state_mapping.yaml` matches on. This is the phase Expert itself
decided internally (`roundabout_entry.py`'s `phase` variable) -- a finer label
than the printed `[ROUNDABOUT] {reason}` text, which collapses `yield`/
`collision_slowdown`/`emergency` into just two reason strings. For episodes
collected after this change, pass that episode's `supervisor.log` as
`--expert-log` (with `--episode`) and the exact-match path above resolves
every roundabout frame with no separately supplied journal needed. Episodes
collected before this change have no `phase` in their log and cannot be
backfilled.

Static, tunnel, and much of Highway may remain without Expert labels with
current sparse logs. Missing Expert matches do not prevent a supported Rule
instruction from rendering; it is marked `instruction_source: rule_state` or
`rule_label_candidate`. Missing sentence variables leave the text null. No
GT rule candidate is promoted to an Expert label.

To attach instructions to an existing feature output without rebuilding any
features or touching the collection, run:

```text
python -m MoraiDrive.data_collection.situation_labeling.attach_expert_instructions --features-dir SITUATION_FEATURES --output NEW_OUTPUT_DIR
python -m MoraiDrive.data_collection.situation_labeling.attach_expert_instructions --features-dir SITUATION_FEATURES --expert-log EXISTING_DRIVING_DEBUG.log --expert-log EXISTING_ROUNDABOUT_DEBUG.jsonl --output NEW_OUTPUT_DIR_WITH_EXPERT
```

The new directory contains `situation_instructions.jsonl` and `report.json`;
the input feature directory is read only. `--episode EPISODE_DIR` is needed
only for a separately supplied journal with true ROS header seq and source
time. Existing sparse logs do not require it.

## 학습 라벨용 움직임 정렬 (`consistency.py`)

Stage 2는 2.5초 미래 궤적을 학습하므로, 문장이 그 궤적과 어긋나면 모델이
문장을 무시하도록 배웁니다. `build_labels`는 규칙 판정 뒤에 녹화된 자차
움직임으로 다음을 적용합니다. 미래를 쓰므로 학습 라벨 전용이며 실시간
규칙이 아닙니다.

- 정적 회피 방향은 한 회피 구간 안에서 처음 방향을 유지합니다. 자차가 경로
  반대편으로 0.3 m 이상 넘어간 경우에만 바뀝니다 (`rules.py`).
- `align_static_avoidance`: 2.5초 안에 실제 옆 이동이 시작되기 전은 "유지",
  시작부터 "회피" 5프레임 뒤 "회피 계속", 2.5초 안에 경로 쪽으로 0.3 m 이상
  돌아오면 "원래 경로로 복귀".
- `hold_dynamic_wait_until_moving`: 양보로 이어지는 접근 구간은 크게 감속하기
  시작한 프레임부터 "양보". 양보 뒤 자차가 서 있으면 "Remain stopped.",
  출발 중(3 m/s 미만)이면 "Resume driving.", 그 뒤 "Continue driving.".
- `align_highway_lane_changes`: 고주로 차선 변경은 지도 링크 상태 대신 실제 좌우
  이동으로 정합니다. 경로 기준 2 m 이상 옆으로 움직인 구간이 차선 변경이며, 2.5초
  안에 0.3 m 이상 이동이 시작되는 프레임부터 이동이 끝날 때까지입니다. 경로 차선에서
  멀어지면 "Move into the left/right lane.", 경로 차선으로 돌아오면 "Return to the
  original lane while moving forward.". 구간 밖에 남은 차선 변경 판정은 앞차 정보로
  따라가기 / 앞차 있음 유지 / 계속 주행 중 하나로 다시 정합니다.
- `fill_short_instruction_gaps`: 같은 문장 사이 3프레임 이하 판정 공백을 메웁니다.
- `smooth_short_instruction_islands`: 같은 문장 사이 1~2프레임의 다른 문장은 앞뒤
  문장으로 맞춥니다. 고주로의 계속 주행 / 앞차 있음 유지 / 따라가기는 1초 미만까지 맞춥니다.
- `use_stopped_roundabout_sentences`: 회전교차로에서 0.3초 이상 서 있으면 양보·추종·감속
  문장을 카탈로그의 "Remain stopped." 변형으로 바꿉니다.
- `drop_contradicted_instructions`: 2.5초 미래와 모순되는 문장을 `null`로 비우고
  원문은 `instruction_dropped`, 이유는 `instruction_consistency`에 남깁니다.
  회피·차선 변경 방향 반대, 서 있는데 기동 문장, 감속해 멈추는데 "계속/유지",
  서 있는데 "계속/유지"를 모순으로 봅니다. 0.3초 미만 모순은 유지하고,
  비운 구간 사이 0.5초 미만의 같은 문장은 같이 비웁니다.

`report.json`의 `motion_alignment`, `instruction_gap_frames_filled`,
`instruction_consistency`에 단계별 변경 수가 남습니다.

## 라벨 검토 뷰어 (`review_viewer.py`)

카메라 4대, 프레임별 instruction, 문장을 뺀 이유(`instruction_consistency`),
움직임 정렬 전 상태, zone_instruction, 상태 타임라인과 좌우 이동·속도 그래프를
run별 HTML로 만듭니다. 여러 run을 만들면 출력 폴더의 `index.html`이 목록이 됩니다.

```bash
python3 -m MoraiDrive.data_collection.situation_labeling.review_viewer --run RUN_DIR [RUN_DIR ...]
```

- 라벨은 기본으로 `RUN/episode_000/situation_labels/features`에서 읽습니다.
  다른 곳이면 `--labels`에 features 폴더, label_episode 출력 폴더,
  또는 `<root>/<캠페인>/<run>/features` 구조의 root를 줍니다.
- 출력 기본값 `MoraiDrive/data_collection/output/situation_label_viewer`는 git이 무시합니다.
  페이지는 상대경로만 쓰므로 출력 폴더를 압축해 다른 PC에서 열 수 있습니다.
- 화면은 `PAGE` 템플릿 하나이고, 데이터는 `frame_rows()`가 만든 프레임별 dict 목록(`ROWS`)이
  JSON으로 들어갑니다. 다른 파이프라인에서도 같은 키로 `ROWS`를 만들면 화면을 그대로 쓸 수 있습니다.

## Camera frame viewer

After `build_labels` has produced a feature directory, create a standalone
HTML viewer with front camera images, current-frame rule states, evidence,
unknown reasons, and a frame slider:

```text
python -m MoraiDrive.data_collection.situation_labeling.visualize_labels --features-dir situation_features_static_lap060 --episode-root G:\static_shapes_100_retry_20260926_112627_lap_060 --output situation_viewer_static_lap060
```

Open `index.html` from the output directory. The viewer checks that the
camera and LiDAR native filenames match index by index, then copies compressed
images beside the page so it works without the source drive. The displayed
`instruction_state` is a current-frame GT rule state. It is not an Expert
behavior label. Rule instructions are marked by their `instruction_source`;
an Expert label appears only with an aligned Expert observation. Pass
`--instructions PATH` to
view a separate `situation_instructions.jsonl` produced from existing features.
