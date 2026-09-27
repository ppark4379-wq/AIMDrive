# Open3D LiDAR debugging

## 클러스터링 확인

```powershell
& 'C:\conda-envs\botsort\python.exe' GPSJamming/scripts/run_udp.py --direct-view
```

전체 파이프라인을 실행하고 클러스터 결과를 기본 표시합니다.
**5**는 지면 제거 후 점군, **6**은 클러스터별 색상, **C**는 중심 십자 표시 전환입니다.
Space로 멈춘 뒤 5/6을 전환하면 같은 프레임을 비교할 수 있습니다.
회색은 점 개수 조건을 만족하지 않아 채택되지 않은 점이며, 색상은 프레임 내부 구분용입니다.
`clustering.tolerance`를 키우면 가까운 물체가 합쳐질 수 있고 줄이면 하나의 물체가 나뉠 수 있습니다.
`min_points`와 `max_points`는 군집 채택 범위입니다. YAML 변경 후 재시작합니다.
VS Code 실행 구성: `LiDAR: live Open3D clustering`.

## RANSAC 지면 제거 확인

```powershell
& 'C:\conda-envs\botsort\python.exe' GPSJamming/scripts/run_udp.py --ground-only
```

ROI → voxel → RANSAC까지만 실행합니다. 기본 화면은 지면 제거 후 점군입니다.
**1**: 제거 전 voxel 점군, **2**: 제거된 지면(갈색), **3**: 남은 비지면 점군.
Space로 멈춘 다음 1/2/3을 전환하면 같은 프레임을 비교합니다.
터미널의 `ground_status=plane_found`는 평면 검출 성공입니다.
`insufficient_seeds`는 높이 범위 내 후보 부족, `no_supported_plane`은 조건을 만족하는 지지점 충분한 평면을 찾지 못한 상태입니다.
실패 시 점을 제거하지 않습니다. 제거된 지면에 장애물이 섞이는지, 비지면에 도로가 남는지 확인하세요.
`configs/perception.yaml`의 `ground`에 각 파라미터 설명이 있습니다. 수정 후 재시작하면 반영됩니다.
VS Code 실행 구성: `LiDAR: live Open3D RANSAC preview`.

## Open3D voxel 다운샘플링 확인

```powershell
& 'C:\conda-envs\botsort\python.exe' GPSJamming/scripts/run_udp.py --voxel-only
```

ROI 다음 voxel 다운샘플링까지만 처리하며 RANSAC과 클러스터링은 실행하지 않습니다.
기본 화면은 voxel 결과입니다. **1**은 ROI(다운샘플링 전), **2**는 voxel(후)로 전환합니다.
Space로 화면을 멈추고 1/2를 누르면 같은 프레임을 비교할 수 있습니다.
`lidar.preprocessing.voxel_size`는 미터 단위이며, YAML 수정 후 재시작하면 반영됩니다.
창 제목에 적용된 voxel 크기, 터미널에 표시 단계의 점 개수가 나옵니다.
VS Code 실행 구성은 `LiDAR: live Open3D voxel preview`입니다.

## Open3D로 ROI만 빠르게 확인

```powershell
& 'C:\conda-envs\botsort\python.exe' GPSJamming/scripts/run_udp.py --roi-only
```

VS Code에서는 `LiDAR: live Open3D ROI only` 실행 구성을 선택합니다.
base_link 변환과 XYZ ROI 필터만 실행하고 voxel, RANSAC, 클러스터링은 건너뜁니다.
Open3D 창에는 ROI 내부의 전체 점을 높이별 색상으로 표시합니다.
마우스로 회전/확대, Space로 화면 일시정지/재개, N으로 최신 프레임 확인,
F로 시점 맞춤, Q 또는 창 닫기로 종료합니다.

