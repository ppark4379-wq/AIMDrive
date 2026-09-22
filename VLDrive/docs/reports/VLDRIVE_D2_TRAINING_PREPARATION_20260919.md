# D2 본 학습 준비 — 2026-09-19

후속 데이터셋 변경: 사용자가 `basic_drive`를 본 학습 대상으로 지정하여 현재
`configs/vldrive_train.json`의 데이터·캐시·encoder provenance·navigation token 경로를
해당 데이터셋으로 변경했다. D2 loss와 optimizer 설정은 유지했다. 아래의
`competition_train/full_loop_best` 검증 수치와 preflight는 **변경 전 데이터셋의 기록**이며,
현재 basic_drive의 라벨 품질이나 전체 시나리오 학습을 보증하지 않는다.
현재 데이터 준비 상태는 `BASIC_DRIVE_DATA_READINESS_20260919.md`를 참고한다.

1. **Config:** 최초 검증에서는 별도 D2 config를 만들고 기존 config를 보존했다. 이후 사용자의 기본 설정 통합 요청에 따라 검증된 D2 내용을 `configs/vldrive_train.json`에 적용하고 중복 D2 파일을 삭제했다. 아래 명령은 통합된 기본 config를 사용한다. 모델/학습 코드는 수정하지 않았다.
2. **D2 옵션:** `training_variant=morai_repair_v1`, `reconstruction_reduction=mean`, `detach_reconstruction_loss_mask=false`, `retention_logit_weight=0.0`, `selector_task_gradient_mode=legacy`, `inference_selection_mode=argmax`를 명시했다. 분모는 프레임별 `C * dropped_mask.sum()`이며 mask gradient가 연결된다. 기존 코드의 `clamp_min(1)` 영분모 보호도 유지한다.
3. **유지:** R=0.3, pruning/reconstruction 가중치=10/1, memory=10, reconstruction layers=4, sequence=11, frame interval=2, train/val stride=5, epochs=15, seed=607, AdamW lr=1e-5/weight_decay=0.06/state_device=cpu, BF16 및 모든 pretrained asset 설정. 기존 기본 config 대비 변경은 위 model 옵션 6개뿐이다. 최초 준비 단계에서는 config 및 기존 소스 46개 SHA-256 보존을 확인했으며, 후속 통합에서는 기본 config만 검증된 D2 내용으로 변경했다.
4. **데이터/asset:** 기존 config의 `../MoraiDrive/dataset_local/competition_train`, `data/processed/vision_tokens/full_loop_best`, `configs/splits/morai_full_loop_80_20.json`, `data/processed/instructions/full_loop.jsonl`을 사용한다. 앞선 `basic_drive` 진단 설정으로 대체하지 않았다. Train 3,352 / validation 838프레임, 총 8,380개 feature/trajectory 배열의 SHA·shape·finite 검증. split/index/instruction route hash 및 cache provenance 일치. visual checkpoint SHA-256은 `56efd0b3353c7ea33c3526465ddb3f54ac307fa091ef500bb77c4d4d4e2a0573`으로 cache와 일치한다. TinyLlama/Q-Former asset manifest의 파일 해시도 검증했다. 실제 sample `[11,106,256]`, batch `[1,11,106,256]`, target `[11,5,2]`, non-empty instruction 확인. 이 dataset 정책은 instruction 경계를 넘는 window를 제외하므로 frame_loss_mask 없이 모든 11프레임을 감독한다. stride 적용 후 train 602 / val 145 windows, 학습 예정 9,030 updates이다.
5. **Graph:** frozen cache → visual_input_norm → hard Gumbel CCDP → retained query / full enhanced 106-token K/V MEFA → Q-Former → qformer_to_language → TinyLlama+DDIA → `[1,11,5,2]` waypoint. Eq.8 reconstruction 및 `L_way + 10 L_prun + L_rec` 연결 확인. 실제 training forward에서 `F.gumbel_softmax(tau=1.0, hard=True)` 호출을 감시해 확인했으며 noise scale=1, 추가 ST bypass 없음. cached 모드에서 vision encoder는 생성되지 않으며 forward/optimizer parameter에도 없다. eval에는 기존 argmax를 유지했다.
6. **Preflight:** 요청한 원래 CLI의 `--mode preflight` 성공. selector/MEFA/Q-Former/projection/reconstruction/TinyLlama/waypoint gradient 모두 nonzero·finite. future→past delta=0, history dependency delta=0.0546875, real history=10. 추가 검증에서 같은 native preflight 전후 frozen parameter/buffer와 trainable parameter의 해시가 동일했다. GPU peak 약 5.77 GiB는 preflight 측정값이며 optimizer를 포함한 본 학습 peak 검증값은 아니다.
7. **Tests:** 121개 통과. 첫 suite 99 passed/1 skipped 후, 실제 로컬 checkpoint 환경변수를 지정하여 skip된 TinyLlama 통합 테스트 1개도 통과했다. 나머지 관련 회귀 테스트 21개 통과. 새 테스트는 config 변경 범위, hard Gumbel train 경로, mean reconstruction의 quotient-rule mask gradient 3개뿐이다.
8. **실행 범위:** 본 학습 미실행. 두 native preflight 호출 모두 optimizer 미생성, stage-2 optimizer updates=0. 단위 테스트의 작은 합성 모델 optimizer 검사는 실제 본 학습과 별개다. 캐시/asset/기존 checkpoint 재생성 또는 변경 없음.
9. **본 학습 명령:** WSL Ubuntu-20.04에서 아래 한 줄을 실행한다. 환경과 의존성 경로는 이번 preflight와 동일하다. output은 실행 시 존재하지 않아야 한다.

```bash
cd /mnt/c/Users/autonav009/Desktop/AIMDrive/VLDrive && PYTHONPATH=.runtime/stage2_wsl_deps:. /opt/miniforge3/envs/lmdrive/bin/python train.py vldrive --config configs/vldrive_train.json --output outputs/logs/vldrive-d2-full-20260919
```

10. **재개 명령:** 아래는 `checkpoint.pt`와 `checkpoint.sha256.json`이 실제 존재할 때만 유효하다. 모델/config/data/code/backend provenance와 설정이 동일해야 하며, optimizer·RNG·update cursor를 복원한다. resume output도 새 경로여야 한다.

```bash
cd /mnt/c/Users/autonav009/Desktop/AIMDrive/VLDrive && PYTHONPATH=.runtime/stage2_wsl_deps:. /opt/miniforge3/envs/lmdrive/bin/python train.py vldrive --config configs/vldrive_train.json --output outputs/logs/vldrive-d2-full-resume-20260919 --checkpoint outputs/logs/vldrive-d2-full-20260919/checkpoint.pt
```

11. **남은 제한:** 확인된 데이터/asset/forward/backward blocker는 없다. 다만 보존한 `checkpoint_every=0` 때문에 **정상 종료 전에는 재개용 full checkpoint가 저장되지 않는다**. epoch의 `model-last.pt`/`model-best.pt`는 optimizer/RNG가 없는 model-only 파일이므로 exact resume에 사용할 수 없다. 갑작스러운 중단 후 정확히 재개할 수 있는 설정은 아직 아니다. 이를 보장하려면 본 학습 시작 전에 저장 주기를 별도로 결정해야 한다. 이번 작업에서는 D2 이외 설정을 바꾸지 않았다. 완료된 `checkpoint.pt`를 같은 15-epoch 설정으로 재개하면 이미 9,030 updates를 마쳤으므로 추가 학습은 발생하지 않는다. 또한 preflight 통과는 학습 후 수렴이나 argmax pruning 품질의 보증이 아니다.

검증 파일:

- `outputs/logs/vldrive-d2-preflight-20260919/preflight.json`: 원래 CLI 결과.
- `outputs/logs/vldrive-d2-preparation-20260919/assets.json`: 실제 전체 캐시/asset 검증.
- `outputs/logs/vldrive-d2-preparation-20260919/preflight_contract.json`: 실제 train sampler/파라미터 해시/optimizer 부재 검증.
- 같은 폴더의 `tests.log`, `tinyllama_integration.log`, `additional_tests.log`, `preservation.json`.

주요 코드: `models/vldrive.py:118` (cached encoder), `models/ccdp.py:150,255` (Gumbel), `models/qformer_mefa.py:1156,1186,1455,1582` (attached mean, MEFA/Q-Former, reconstruction), `utils/trainer.py:411` (native preflight), `train.py:210,226,253` (periodic/model-only/final checkpoint), `utils/training.py:105` (exact restore).
