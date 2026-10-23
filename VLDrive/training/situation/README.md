# Situation Head 학습

- 준비된 DataLoader를 받아 학습한다. 데이터 생성·변환·캐시 생성 코드는 포함하지 않는다.
- `models/llm/situation_head.py`: Connector 특징 `Fv`와 Zone으로 Transformer가 상황 embedding `Fs`를 생성한다.
- `labels.py`: 이미 제공된 구조화 GT를 분류 정답으로 매핑한다. 자연어를 임의로 해석해서 라벨을 추정하지 않는다.
- `loss.py`: 세부 상황·상대 위치·객체·회피 방향의 Cross Entropy loss만 계산한다.
- `trainer.py`: 독립 학습, 지표 기록, 체크포인트 저장·로딩을 담당한다.

## Zone

`0 일반 / 1 정적 장애물 / 2 동적 장애물 / 3 회전교차로 / 4 고속도로 / 5 터널`.
신호등 속성은 없다. 스키마는 `zone_v2_no_traffic_light`이며 이전 Zone 가중치와 호환되지 않는다.

## DataLoader가 반환할 배치

```python
batch = {
    "inputs": {
        "visual_tokens": Fv,       # float [B,T,K,Dv], VE가 아닌 Connector 출력
        "visual_mask": valid,     # bool [B,T,K], 유효한 토큰만 True
        "timestamps": times,      # float [B,T], 초 단위, 각 window에서 시간순
        "zone_ids": zones,        # int64 [B,T], 0..5
        # zone_ids 대신 zone_embeddings: float [B,T,hidden_dim]도 가능
    },
    "targets": {
        "labels": {
            "instruction": ids,   # int64 [B], zone:instruction_id의 클래스 번호
            "position": pos,      # int64 [B]
            "object": obj,        # int64 [B]
            "direction": side,    # int64 [B]
        },
        "action_valid": action_valid,  # bool [B]
    },
}
```

- GT는 window **마지막 프레임**의 상황 정답이다. 궤적 예측 head와 trajectory loss는 없다.
- 데이터에 trajectory/action 수치가 함께 있어도 이 학습기는 targets의 `labels`, `action_valid`만 읽는다. 미래 궤적을 모델 입력에 포함하지 않는다.
- 시간 정렬·episode 경계·마지막 프레임 GT 연결은 데이터를 제공하는 쪽에서 보장한다. 이미 만들어진 window를 이 코드가 다시 자르지 않는다.
- `timestamps`는 큰 절대시각을 float32로 바꾸기 전에 window 기준 상대 초로 변환하거나 float64로 전달한다.
- 각 프레임에 적어도 하나의 유효 토큰이 필요하다. 토큰 padding은 mask로 제외한다.
- `zone_embeddings`를 직접 주면 학습·추론에서 같은 Zone encoder와 가중치를 사용해야 한다. ID를 주면 내부 ZoneEmbedding을 함께 학습한다.
- 이 독립 학습에서는 외부 Fv/Zone embedding을 detach하여 VE·Connector·외부 Zone encoder를 업데이트하지 않는다.

## 정답 매핑

기존 `situation_instructions.jsonl`과 같은 구조의 **이미 읽힌 마지막 프레임 GT dict**를 사용할 수 있다.

```python
from VLDrive.training.situation.labels import vocabulary, encode_targets

# last_frame_gt_rows는 데이터 제공 측에서 이미 준비한 GT 목록
vocab = vocabulary(last_frame_gt_rows)
label_ids, action_valid = encode_targets(last_frame_gt_rows[0], vocab)
```

- `zone_id`, `instruction_id`, `instruction`, `instruction_variables`를 사용한다. 원본 데이터는 수정하지 않는다.
- 같은 `follow`라도 회피·감속·정지 등 서로 다른 `instruction_id`를 학습한다. `expert_label` 하나로 문장을 추정하지 않는다.
- `instruction_variables`의 `position`, `object`, `direction`은 제공된 문자열별 클래스로 매핑한다. 한 번 만든 vocab을 고정해 사용한다.
- 현재 버전은 템플릿 ID와 해당 슬롯을 예측한다. 자유로운 자연어 생성이나 모든 주변 객체를 개별 검출하는 모델은 아니다.
- 누락된 항목은 `-100`으로 제외한다. 실제 위치·객체 라벨을 누락하면 해당 속성은 감독되지 않는다.
- `action_valid=False`이면 상황/행동 템플릿·회피 방향 감독을 제외한다. 위치·객체는 유효한 것만 감독한다.
- 회피·양보·정지 등의 템플릿/방향은 상황 라벨로 분류하며, waypoint나 조향·가감속 제어값은 예측하지 않는다.
- 관측만 확실한 `unknown`은 `observation_valid=True, action_valid=False`와 관측 슬롯을 제공한다. 둘 다 불확실하면 두 flag 모두 False로 준다.
- 기존 `dynamic_behavior_unverified`는 관측만 감독한다. `unknown`을 정상 주행으로 치환하지 않는다.
- 자연어만 제공되고 구조화 정답이 없다면 이 매퍼로 학습할 수 없다. 문장과 함께 template ID/슬롯 또는 이미 인코딩된 정답을 전달해야 한다.

## 학습 호출

저장소 루트가 Python import 경로인 실행 코드에서 호출한다. 새로운 Dataset 구현은 필요하지 않다.

```python
import torch
from VLDrive.models.llm.situation_head import SituationHead
from VLDrive.training.situation.trainer import SituationTrainingConfig, fit

torch.manual_seed(42)
model = SituationHead(
    class_counts={name: len(values) for name, values in vocab.items()},
    visual_dim=2048, hidden_dim=256, output_dim=2048,
    layers=2, heads=8,
)
fit(
    model, train_loader, vocab,
    output_dir="VLDrive/outputs/checkpoints/situation/run_001",  # 새로운 경로
    config=SituationTrainingConfig(epochs=20, learning_rate=1e-4, device="cuda"),
    feature_source={"connector_checkpoint": "실제 사용한 checkpoint 경로/해시"},
)
```

- batch size는 제공하는 DataLoader에서 설정한다. `visual_dim`은 실제 Connector 출력 차원에 맞춘다.
- 총 loss는 유효한 상황·위치·객체·방향 각각의 평균 CE를 더한 값이다.
- `last.pt`, `best_train.pt`, `metrics.jsonl`이 생성된다. `best_train`은 **학습 중 집계한 loss** 기준이며 주행 성능 best나 validation best가 아니다.
- 체크포인트에는 Zone 스키마, 클래스 사전, 모델 설정/가중치, optimizer 상태를 저장한다. 이번 버전에는 resume 실행기는 포함하지 않는다.
- 체크포인트 형식은 `situation_head_v2_situation_only`다. 궤적 head가 있던 이전 형식은 자동으로 불러오지 않는다.

## 추론 호출

```python
from VLDrive.training.situation.trainer import load_situation_head

head, vocab = load_situation_head(".../best_train.pt", device="cuda")
with torch.inference_mode():
    result = head(**sensor_zone_inputs)  # 학습과 같은 input 키/장치, GT 없음
    fs = result["situation_embedding"]  # [B,1,output_dim]
```

상황 분류 head들이 `Fs`를 읽으므로 `Fs` 생성부에도 상황 감독의 gradient가 전달된다.
`Fs`는 연속 embedding이며 현재 LLM 입력에 자동으로 연결되지는 않는다. LLM이 활용하도록 하는 연결과 공동 학습은 다음 단계다.
실제 데이터 파일 로딩/연결 및 실제 주행 성능 검증은 데이터 제공 후 진행한다.
