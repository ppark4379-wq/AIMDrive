# 동일 Python 바이너리의 다른 PC 비교

파일: python-repro-20260912.tar.gz (약 23 MB)
SHA256: 86048e27c7d71810160c0175a0e81fa655a38593aaad610f0c75b4fe2b912039

현재 PC에서 사용하던 Linux x86_64 Python 3.8.10, 표준 라이브러리, 공유 라이브러리, glibc 로더와 고정 좌표 산술 재현을 포함한다. Expert/ROS/NumPy/수집 데이터는 포함하지 않는다. 시스템 Python을 설치하거나 교체하지 않는다. 타 PC의 커널, CPU와 호스트 OS는 동일하게 묶이지 않는다.

1. 다른 x86_64 PC의 C:\Temp에 압축 파일을 복사한다. WSL Ubuntu가 준비돼 있어야 한다. ARM PC는 비교 대상에서 제외한다.
2. 다른 PC의 Ubuntu 터미널에서 아래 명령을 실행한다. 폴더가 이미 있으면 새 작업 폴더를 사용한다.

```bash
mkdir -p ~/python-repro-check
cd ~/python-repro-check
sha256sum /mnt/c/Temp/python-repro-20260912.tar.gz
# 위 SHA256과 일치하는지 확인 후 진행
 tar -xzf /mnt/c/Temp/python-repro-20260912.tar.gz
cd python-repro-20260912
bash run.sh 20
```

각 실행은 최대 35초 제한이며 정상적으로 약 5~7초, 20회는 대략 2~3분이다. run.sh는 SHA256SUMS를 먼저 검증한다. failures=0/20은 이번 반복에서 재현되지 않았다는 뜻이며 무결함 인증이 아니다. 오류는 종료 코드뿐 아니라 JSON status도 검사한다. results-날짜-시간-PID 폴더의 environment.txt, summary.txt, run-*.log를 함께 비교한다.

현재 PC 기준 시험:
```bash
cd ~/AIMDrive/MoraiDrive/diagnostics/.runtime/python-repro-20260912
bash run.sh 20
```

판별:
- 다른 PC WSL에서도 같은 오류: 이 PC만의 CPU/RAM 문제 가설이 약해진다. 공통 Linux Python/라이브러리 또는 WSL 환경을 더 분리해야 한다.
- 다른 PC WSL에서는 반복 통과, 현재 PC에서 반복 실패: 현재 호스트에 종속된 문제가 유력하다. WSL 상태와 CPU/RAM 둘 중 하나로 아직 단정할 수 없다.
- 현재 PC의 네이티브 Linux에서도 같은 묶음 실패: WSL이 필수 원인은 아니다. 소프트웨어 공통부와 하드웨어를 추가 비교해야 한다.
- 현재 PC의 네이티브 Linux에서 충분히 반복 통과하고 WSL에서만 실패: WSL/Windows 경로를 우선 조사한다. 간헐성과 부하 차이를 고려한다.

CPU와 RAM의 최종 구분은 BIOS 기본 설정에서의 재시험, 별도 메모리 검사 및 부품 교차시험이 필요하다. Intel은 13/14세대 데스크톱 안정성 문제에 최신 BIOS의 0x12F 이상 마이크로코드와 Intel Default Settings를 권고한다. 이 권고가 이 PC의 고장 확정은 아니다.
https://www.intel.com/content/www/us/en/support/articles/000102331/processors.html

BIOS 변경·재부팅·WSL 종료는 이번 조사에서 수행하지 않았다. WSL 전체 export는 계정·프로젝트까지 포함할 수 있어 이 비교에는 사용하지 않았다.
