# coffee_martini 단독 실행 — 2026-10-01

사용자 바탕화면의 `coffee_martini.zip`에서 **0번 프레임**만 사용합니다.
동적 영상 전체의 4D 학습은 이번 실행 범위에 포함하지 않습니다.
별도 모델과의 비교 실험 없이 RT-Splatting만 학습합니다.

- 원본 ZIP: `<Desktop>/coffee_martini.zip`
- ZIP SHA256: `cbc31291ce143e6f31a00f23a049663a61cc8793af8dab259dbc6e338a676816`
- 18개 카메라, 학습 17개 / 테스트 `cam00` 1개.
- 학습 해상도: 1352 × 1014 (원본 2704 × 2028).
- 초기 점: 학습 카메라만 사용한 SIFT 삼각측량 14,658개.
- 사용자 요청으로 총 **30,000회에서 종료**. warmup 1,000, densification은 15,000 미만에서 실행.
- 재개 시 optimizer가 급변하지 않도록 기존 100,000회 학습률 스케줄을 유지하고 `--stop-at 30000`으로 종료.
- 원본 VGG LPIPS loss는 15,000부터 사용. 기본 L1 + SSIM 및 RT loss 사용.
- 15,000회 체크포인트에서 재개한 뒤 **1,000회마다** 체크포인트와 cam00 중간 렌더링을 저장.
- 중간 영상: `runs/coffee_rt_100k/previews/step_015000/`부터 `step_029000/`까지.
- 최초 실행의 폴더명 `coffee_rt_100k`는 이어 쓰지만 실제 종료 횟수는 30,000회.
- 창문 마스크: 이미지에서 확인한 근사 창문 다각형과 pretrained DeepLab 사람 분할을 조합.
  공식 데이터셋의 정답 마스크가 아니며 가는 내부 창살이 일부 포함될 수 있습니다.
  주석과 겹쳐보기는 `data/coffee_f000/mask_provenance.json`, `mask_previews/`에 기록합니다.

GPU 서버는 RTX 3090 24GB, Python 3.12.14 / Torch 2.7.1+cu128 / CUDA toolkit 12.8입니다.
원본 checkout은 고정 commit을 그대로 사용하고, 새 CUDA 컴파일러에서 생략된 표준 헤더는
`NVCC_PREPEND_FLAGS=--pre-include=cstdint,cfloat` 빌드 옵션으로 보완했습니다.
전체 설치 버전은 서버의 `deploy/environment-lock.txt`에 저장됩니다.

서버 작업 경로는 `/workspace/RTsplatting4d`입니다.

```bash
supervisorctl status rt-coffee-train
/venv/main/bin/python deploy/status.py
tail -n 5 /workspace/rt-coffee-train.log
```

중단 이후 재개할 때는 기존 프로세스가 종료되었는지 먼저 확인한 뒤 같은 설정으로 실행합니다.

```bash
/venv/main/bin/python -m rtstatic train \
  --scene data/coffee_f000 --mode rt --iterations 100000 --stop-at 30000 --warmup 1000 \
  --seed 0 --save-every 1000 --preview-every 1000 --preview-on-resume \
  --lpips --out runs/coffee_rt_100k \
  --resume runs/coffee_rt_100k/checkpoint.pt
```

학습이 끝나면 자동으로 `cam00` 테스트와 영상 저장을 수행합니다.
그 이후 다음 명령으로 시각화 보고서를 만듭니다.

```bash
/venv/main/bin/python deploy/report_results.py --run runs/coffee_rt_100k
```

결과: `runs/coffee_rt_100k/report/index.html`, `visual_summary.png`, `training_curve.png`.
지표: `runs/coffee_rt_100k/evaluation/metrics.json`.
체크포인트: `runs/coffee_rt_100k/checkpoint.pt`.

`deploy/finish_vast.py`는 중간/최종 보고서를 자동 생성합니다.
로컬 `deploy/collect_vast.py`는 중간 결과를 `runs/coffee_previews/`에 내려받고,
최종 백업을 SHA256 검증 후 `runs/coffee_completed/`에 압축 해제합니다.
체크포인트 이후 재계산된 구간의 기존 로그는 `discarded_after_*.jsonl`,
재개 기록은 `resume_events.json`에 남기며 학습 곡선에는 중복 구간을 포함하지 않습니다.

GPU 사전 점검에서는 실제 coffee_martini 입력 10회 학습, 테스트, checkpoint 재로딩을 수행했습니다.
재로딩 전후 `prediction.npy`의 SHA256이 완전히 같았고, LPIPS의 실제 해상도 forward/backward와
서버에서의 15개 테스트가 통과했습니다. 이 사전 점검 수치는 최종 학습 성능이 아닙니다.
