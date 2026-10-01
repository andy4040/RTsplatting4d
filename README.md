# RTsplatting4d

N3DV **coffee_martini의 0번 프레임**을 RT-Splatting으로 학습하는 정적 장면 실험입니다.
학습 카메라 17개, 확인용 `cam00` 1개, 해상도 1352 × 1014를 사용합니다.
목표는 **30,000 iterations**, 1,000회마다 렌더링을 저장합니다. 별도 모델과의 비교 실험은 수행하지 않습니다.

## 학습 결과

**학습 진행 중입니다. 아래 이미지는 공개된 최신 중간 결과이며 최종 테스트가 아닙니다.**

[횟수별 이미지와 PSNR·SSIM·LPIPS 보기](results/coffee_martini/README.md)

![최신 RT-Splatting 렌더링과 창문 확대](results/coffee_martini/latest.png)

원본과 렌더링을 슬라이더로 보려면 저장소를 내려받아
`results/coffee_martini/index.html`을 브라우저로 여세요. GitHub 파일 화면에서는 HTML이 실행되지 않습니다.
창문 마스크는 근사 주석이며, 현재 결과만으로 다른 방법 대비 품질 개선을 주장하지 않습니다.

## 원본 사용 범위

[공식 RT-Splatting](https://github.com/sjj118/RT-Splatting)의 GaussianModel, 방향 인코딩,
반사 MLP, 반사·투과 렌더러와 CUDA rasterizer를 직접 사용합니다.
고정 커밋은 `3f45b3cac4be04db9f3092234666b695991b268a`이며 원본 checkout은 수정하지 않습니다.

**학습 파이프라인 전체가 원 논문과 동일한 것은 아닙니다.** N3DV 데이터 준비, 초기 점 생성,
근사 창문 마스크, 학습 루프를 추가했고 consistency loss의 합을 평균으로 바꿨습니다.
원본 코드의 위치와 변경점은 [실행 기록](RUN_COFFEE.md)과 [상세 사용법](docs/USAGE.md)에 있습니다.
현재 범위에는 4D 변형장이나 동적 장면 학습이 포함되지 않습니다.

## 코드 위치

| 경로 | 내용 |
|---|---|
| `rtstatic/engine.py` | 원본 모델·렌더러를 호출하는 학습, 재개, 평가 |
| `rtstatic/losses.py` | 적용한 손실 계산 및 원본 대비 변경 |
| `rtstatic/data.py` | 단일 시점 이미지와 카메라, 초기 점 준비 |
| `deploy/train_vast.sh` | 현재 실행 설정: 30,000회 종료, 1,000회 결과 저장 |
| `deploy/export_results.py` | Git에 올릴 이미지·지표 묶음 생성 |
| `upstream.json` | 원본 저장소와 고정 커밋 |

원본 소스는 `python bootstrap.py`로 `third_party/RT-Splatting`에 받습니다.
GPU 환경 구성과 실제 재개 명령은 [RUN_COFFEE.md](RUN_COFFEE.md)를 참고하세요.
처음부터 실행할 때는 기존 실행 폴더를 요구하는 `deploy/train_vast.sh` 대신 상세 사용법의 prepare/train 절차를 따르세요.

## 결과 갱신

실행 중인 작업 폴더에서 다음 명령으로 이 저장소에 최신 다운로드 결과를 내보냅니다.

```bash
python deploy/export_results.py --source /path/to/training-workspace --out /path/to/RTsplatting4d/results/coffee_martini
git add results/coffee_martini
git commit -m "Update coffee_martini training results"
git push origin main
```

커밋에는 코드와 시각화·지표·설정만 포함합니다. 원본 데이터셋, CUDA 빌드 파일,
수백 MB의 학습 체크포인트는 제외합니다. 현재 모델 파일은 학습 서버의
`/workspace/RTsplatting4d/runs/coffee_rt_100k/checkpoint.pt`에 있으며 최종 결과 수집 시 로컬에도 백업됩니다.
서버 결과 수집 도구는 `VAST_HOST`와 `VAST_PORT` 환경변수를 사용합니다.

## 출처와 검증

- 원본 RT-Splatting 저작권·라이선스: [MIT license](docs/RT-Splatting-LICENSE.txt). CUDA 하위 모듈은 각 원본 라이선스를 따릅니다.
- 데이터: N3DV coffee_martini. 전체 데이터셋을 이 저장소에서 재배포하지 않습니다.
- 검증 기록: [validation.json](validation.json). CPU 테스트 18개 통과, 실제 RTX 3090 학습·재로딩·중간 렌더링 검증.
