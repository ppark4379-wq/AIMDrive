# 로컬 VLDrive 보존 및 원격 expert 통합

## 승인된 범위

- VLDrive 폴더 전체는 현재 로컬 구현을 기준으로 유지한다. 원격의 Notice/clip 학습 추가와 자동 병합 가능한 변경도 이 폴더에는 적용하지 않는다.
- MoraiDrive expert는 로컬 `PathController` 공통화 구조를 유지하면서 원격의 주행 개선을 이식한다.
- 생성 데이터·실행 로그·가중치는 커밋하지 않고 원본 위치에 보존한다.

## 브랜치와 보존

- 원래 작업 폴더의 브랜치: `main`, HEAD `f81aeab9c059ba8a3eec2983145f827dbb02cf96`.
- 통합 브랜치: `integrate/local-vldrive-expert-20260923`.
- 로컬 코드 보존 커밋: `d43677dc2e9f8ca4d47f7e57791abd66635e20a1`.
- 병합한 원격: `69acda739a1235adcb30e0bbc74657ae488d9bed`.
- 별도 worktree: `VLDrive/.runtime/integration_20260923`.

로컬 변경/신규 소스·설정·문서 159경로를 보존하고, 미추적 데이터/출력 329경로는 커밋에서 제외했다. 사용자의 원래 index/작업 파일은 변경하지 않았다. 선택된 원본 파일 159개는 작업 전 SHA256과 일치함을 확인했다. 원래 폴더에서 계속 학습/추론할 때는 기존 코드가 사용되며, 통합본은 별도 worktree/브랜치에 있다.

VLDrive의 통합 Git tree는 로컬 보존 커밋의 VLDrive tree와 완전히 같다. 이는 충돌 4개만 로컬 선택한 것이 아니라 원격 신규 VLDrive 파일도 채택하지 않은 결과다. Notice·sealed Sensor Record clip 관련 원격 VLDrive 기능은 이번 통합에 포함되지 않는다. LMDrive/MoraiDrive 등 다른 폴더의 원격 변경은 병합한다.

## Expert 구현

원격 `AutonomousDriving`을 기준으로 제어 블록을 공통 모듈에 옮겼다.

- `AutonomousDriving(PathController)` 및 `_init_control` 구조 유지.
- 원격의 보행자 통과 후 제동 해제·재출발, 회전교차로 진입 가속 gain, 정적 회피 곡률별 lookahead, 회전교차로 제어 로그 보존.
- 원격의 planning `global_path` 전달 및 정적 회피 오류 진단도 보존.
- `_calculate_control(..., pedestrian_released=False)`를 추가하고 expert에서 ACC의 해제 상태를 명시적으로 전달한다.
- VLDrive는 ACC 객체 없이 기본값 False로 공통 제어기를 호출한다. 따라서 expert 내부 객체를 잘못 참조하지 않는다.
- 다음 5개 메서드는 원격 AST와 동일: `_configure_runtime`, `_recorded_lookahead_limits`, `_roundabout_entry_acceleration_gain`, `_roundabout_entry_fast_release`, `_smooth_longitudinal_command`.
- 원격의 그 밖의 AutonomousDriving 메서드도 동일하며 `__init__`와 `execute`만 공통 모듈 호출로 바꿨다.

VLDrive의 모델·학습·Gumbel·navigation·남은 경로 목표 선택·5km/h 상한 설정은 로컬 그대로다. 공통 제어기는 원격과 통합되지만 VLDrive의 `model_waypoints` 문맥에서는 expert 전용 보행자/회전교차로 조건이 활성화되지 않는다.

## 결과물 관리

루트 `.gitignore`에 추론 기록, 분석 출력, 생성 instruction 경로를 추가했다. VLDrive 내부 `.gitignore`는 로컬 파일을 그대로 유지했다. 데이터/로그를 삭제하지 않았다. 추가된 테스트는 `MoraiDrive/expert/src/expert/tests/test_shared_path_controller.py`다.

Worktree에서 `.gitattributes`의 `*.sh text eol=lf` 규칙과 기존 CRLF blob이 어긋난 `MoraiDrive/vision_encoder/scripts/prepare_dataset.sh`, `train.sh` 두 파일은 LF로 정규화했다. 명령 내용은 원격과 동일하며 줄바꿈만 달라졌다.

## 검증

실제 MORAI 제어 송신이나 장시간 학습은 실행하지 않았다.

1. Expert 전체 테스트와 VLDrive 제어/목표점/dataset/dashboard 회귀 테스트: 최초 **410 passed, 13 failed, 5 errors**, subtests 102 passed.
2. 분리된 worktree에 없던 기존 로컬 `path.csv`를 무시되는 runtime 폴더에 복사하고, 기존 MORAI 네트워크 모듈을 테스트 PYTHONPATH에 연결했다. 해당 **6개 테스트 모두 통과**. 이 자료들은 커밋하지 않는다.
3. 나머지 expert 실패 **12개**를 수정하지 않은 원격 `AutonomousDriving`으로 교체한 별도 프로세스에서 다시 실행했다. **12개 모두 동일한 실패로 재현**됐다. 나머지 expert 소스는 원격과 같으며 공통 제어기 파일은 그 baseline 클래스가 사용하지 않는다.
4. 따라서 검사한 428개 테스트에서 환경 자료 보완 후 **416개 통과**, **12개 원격에서도 재현되는 실패**를 확인했다. 전체 suite를 재실행해 green을 확인했다는 의미는 아니다.
5. VLDrive tree 동일성, 원본 작업 파일 SHA256 보존, 공통 메서드 AST 동일성 확인.

### 기존 실패 목록

| 파일 / 테스트 | 재현된 결과 |
|---|---|
| test_highway_lane_planning / local_forward_fallback_starts_acc_when_link_match_fails | 17.800375 > 기대 상한 10 |
| test_highway_lane_planning / safe_zone_keeps_yielding_while_vehicle_enters_successor | 17.5 > 기대 상한 14 |
| test_highway_lane_planning / second_merge_middle_uses_acc_when_left_is_unavailable | KEEP_LANE, 테스트 기대 ACC_FOLLOW |
| test_highway_lane_planning / slower_lead_triggers_lane_change_before_prediction_collision | second_merge_prefer_left, 테스트 기대 safe_change |
| test_highway_lane_planning / time_based_cycle_threshold_scales_with_control_dt | 17, 테스트 기대 50 |
| test_longitudinal_control / crossing_pedestrian_is_detected_before_entering_lane | 기대 감속 없음, 16.67 유지 |
| test_project_diagnostics / array_failure_records_inputs_and_reraises | 진단 파일 0개 |
| test_project_diagnostics / constructor_identity_is_captured_without_swallowing_error | 진단 파일 없음 |
| test_project_diagnostics / projection_does_not_use_point_scalar_properties | Point.x 접근 |
| test_project_diagnostics / projection_rejects_malformed_or_nonfinite_points | ValueError 기대, IndexError 발생 |
| test_roundabout_entry / braking_begins_before_stop_line | 8.333333, 테스트 기대 7 미만 |
| test_roundabout_entry_gain / entry_uses_point_three_and_reverts_after_leaving_roundabout | 원격 설정 P=.80, 테스트 기대 .30; 출력 1.0 vs .666674 |

이 실패들은 이번 병합으로 생긴 회귀로 확인되지 않았으며, 원격 주행 정책/테스트의 정합성 문제로 따로 관리해야 한다. 이번 범위에서 원격 정책을 임의로 고치지 않았다.

## 배포 상태

통합은 로컬 브랜치에 커밋한다. 원본 main 작업 폴더는 변경하지 않는다. 원격 push와 main 반영은 아직 수행하지 않는다. push할 때는 통합 브랜치를 명시해야 한다:

```bash
git push origin integrate/local-vldrive-expert-20260923
```

원격에 올라간 통합 브랜치를 검토한 뒤 main에 반영한다. 강제 push는 필요 없다.
