# RTsplatting4d — N3DV 한 시점 검증

N3DV의 **동일한 프레임을 여러 카메라에서 추출**해 RT-Splatting으로 학습하고,
학습에서 제외한 `cam00`의 창문 영역을 비교합니다. 현재 범위는 정적 한 시점이며 4D 확장은 포함하지 않습니다.

공식 [RT-Splatting](https://github.com/sjj118/RT-Splatting)의 GaussianModel,
CUDA rasterizer, 반사·투과 renderer를 직접 불러옵니다. 원본 checkout은 수정하지 않습니다.
고정 commit은 `upstream.json`에 기록했습니다. 원본 라이선스는
`third_party/RT-Splatting/LICENSE` 및 `LICENSE.md`와 각 submodule에 있습니다.

**현재 검증 범위:** CPU 전처리·평가·gradient gating 테스트를 제공합니다.
coffee_martini에서 native CUDA 학습 및 중간 렌더링이 검증됐습니다. 현재 실행은 RT 단독이며, 아래 비교 실험 명령은 선택적인 별도 기능입니다.
기존 `new4dgs`의 학습 코드나 checkpoint에는 의존하지 않는 독립 폴더입니다.

## 1. GPU 환경 준비

Linux/WSL2 + NVIDIA GPU + CUDA toolkit (`nvcc`) 환경을 사용합니다.
원본에 맞춘 별도 Python 3.10 환경을 권장합니다. Python 3.13 / CPU PyTorch 환경에서는 학습할 수 없습니다.
GPU 메모리는 해상도·Gaussian 수에 따라 달라지므로 고정 최소 용량을 보장하지 않습니다.

```bash
conda create -n rtsplat-static python=3.10 -y
conda activate rtsplat-static
cd RTsplatting4d
python bootstrap.py --install
python doctor.py
```

`bootstrap.py --install`은 현재 환경에 원본의 Torch 2.0.1 / torchvision 0.15.2 및
의존성을 설치하고 CUDA 확장 모듈을 빌드합니다. 별도 환경에서 실행하세요.
PyTorch wheel의 CUDA와 설치한 toolkit이 호환되어야 합니다. 최신 GPU에서 Torch 2.0.1이
GPU architecture를 지원하지 않으면 별도의 최신 Torch 환경으로 포팅·빌드 검증이 필요합니다.
`nvdiffrast`는 v0.3.3으로 고정합니다. 네트워크는 패키지와 공식 코드 다운로드에 필요합니다.

이미 GPU 환경을 구성했다면 `python bootstrap.py`로 원본만 받고, 원본 README에 따라
의존성을 설치한 후 `python doctor.py`로 확인할 수 있습니다.
`doctor`는 import와 GPU 인식 확인이고 실제 렌더링 검증은 다음 명령입니다.

```bash
python smoke.py --out runs/gpu_smoke
```

합성 평면으로 GS / RT의 forward, backward, 저장, 재로딩, 평가를 실행합니다.
**N3DV 품질 검증 결과가 아닙니다.** 기존 출력 경로는 덮어쓰지 않습니다.

CPU에서 이미지 준비만 하려면:

```bash
python -m pip install -r requirements-cpu.txt
```

## 2. 한 시점 이미지와 초기 점 준비

```text
/datasets/coffee_martini/
  poses_bounds.npy
  cam00.mp4
  cam01.mp4
  ...
```

이미 추출한 이미지라면 `camNN/images/0000.png` 형식도 지원합니다.
프레임 번호는 0부터 시작해야 합니다. 빠진 카메라 번호는 허용하며,
calibration row는 **존재하는 카메라 이름을 정렬한 순서**에 대응합니다.

```bash
python -m rtstatic prepare --data /datasets/coffee_martini --frame 0 --width 1352 --out data/coffee_f000
```

- `cam00`은 평가 전용입니다. SIFT 특징 매칭과 초기 점 삼각측량에도 사용하지 않습니다.
- 이미지 크기가 변하면 카메라 focal length도 원본 calibration 기준으로 조정합니다.
- 원본 좌표계를 유지합니다. 유효한 초기 점이 부족하면 random 점으로 대체하지 않고 실패합니다.
- 준비 중 실패하면 출력 디렉터리가 일부 남을 수 있습니다. 원인을 해결한 뒤 새 출력 경로로 실행하세요.
- 기본 폭 1352는 초기 실험값입니다. 작은 창문 디테일은 `--width 2704`에서도 확인해야 합니다.
- 여기서 파악한 입력 카메라와 3D 점은 `scene.json`, `points.npz`에 저장됩니다.

## 3. 창문 마스크 표시

```bash
python -m rtstatic annotate --scene data/coffee_f000 --out data/coffee_f000/annotate.html
```

생성한 HTML을 브라우저로 엽니다. 각 이미지에서 **실제로 보이는 유리 영역**을 다각형으로
표시하고 확인 완료를 누릅니다. 창틀, 앞을 가린 사람, 불투명 물체는 제외합니다.
여러 다각형을 사용할 수 있습니다. 유리가 보이지 않는 카메라도 빈 마스크로 확인해야 합니다.
모든 카메라를 확인한 뒤 `polygons.json`을 내려받습니다. 이미지가 외부 서버로 전송되지 않습니다.

```bash
python -m rtstatic masks --scene data/coffee_f000 --polygons /path/to/polygons.json
python -m rtstatic check --scene data/coffee_f000
```

`mask_previews/`의 겹쳐보기로 위치를 확인하세요. 수정이 필요하면 polygon JSON을 다시 가져올 수 있습니다.
외부 도구로 만든 마스크도 `masks/camNN.png`에 놓을 수 있습니다.
0은 불투명, 1 또는 255는 유리입니다. 회색/soft mask는 거부하고 크기 조정은 nearest-neighbor를 사용합니다.
학습 시작 후에는 동일한 비교 실험의 이미지·마스크를 바꾸지 마세요.
마스크가 없거나 학습/평가 창문 영역이 모두 비어 있으면 학습을 중단합니다.

## 4. 먼저 짧게 실행한 뒤 본 실험

데이터셋에서 몇 step의 native 실행 여부를 확인합니다. 다음 실행의 품질 지표는 의미가 없습니다.

```bash
python -m rtstatic train --scene data/coffee_f000 --mode rt --iterations 10 --warmup 2 --out runs/coffee_smoke
```

본 실험은 동일한 데이터, 초기 점, seed, 해상도로 두 모델을 순서대로 학습합니다.

```bash
python -m rtstatic compare --scene data/coffee_f000 --iterations 30000 --warmup 1000 --seed 0 --out runs/coffee_f000_s0
```

`--ablation`을 추가하면 gradient gating을 끈 RT도 학습합니다.
`--lpips`를 추가하면 원본 VGG LPIPS loss와 전체 영상 LPIPS 평가도 사용합니다.
이 옵션은 pretrained weight 다운로드가 필요합니다. 두 모델 모두 동일하게 적용됩니다.
LPIPS를 켜지 않은 기본 실험의 주 지표는 PSNR / SSIM / MAE입니다.

학습 중에는 평가 카메라의 loss로 모델을 선택하거나 early stopping하지 않습니다.
같은 프레임에서 seed 0, 1, 2를 새 출력 경로로 반복한 뒤 다른 프레임에서도 검증하세요.

원본은 장면별 foreground sphere를 설정합니다. 기본값은 모든 Gaussian을 반사 모델의 대상으로
포함합니다. 필요하면 `--env-center X Y Z --env-radius R`로 관심 영역을 제한할 수 있습니다.
창문이 sphere 밖으로 나가면 반사 분기를 사용할 수 없으므로 원본 좌표계에서 확인해야 합니다.

## 5. 결과 확인 및 재평가

```text
runs/coffee_f000_s0/
  comparison.json                 # RT - GS 차이, 양수 PSNR/SSIM은 개선
  cam00_prediction.png            # GS | RT (ablation을 켜면 세 번째 열 추가)
  cam00_window_crop.png            # 각 모델의 [GT | 예측] 창문 확대
  gs/
  rt/
    checkpoint.pt                 # 모든 Gaussian 속성 + reflection MLP + direction encoding
    config.json                   # commit, 환경, 설정, scene fingerprint
    split_audit.json
    losses.jsonl
    training_summary.json
    evaluation/
      metrics.json
      cam00/
        prediction.npy            # PNG 양자화 전 렌더링
        target.png
        prediction.png
        window_crop.png
        comparison.png            # GT | 예측 | 절대오차 x5, 빨간색
        final_tran.png
        final_scat.png
        final_spec.png
```

```bash
python -m rtstatic evaluate --checkpoint runs/coffee_f000_s0/rt/checkpoint.pt --out runs/coffee_rt_reloaded
```

데이터를 옮겼다면 `--scene /new/path`를 지정합니다. 내용의 fingerprint가 달라지면 거부합니다.
새 checkpoint는 optimizer와 RNG 상태를 포함하며, 같은 학습 설정과 출력 경로에서
`train --resume <checkpoint.pt>`로 재개할 수 있습니다. 이전 inference-only checkpoint는
재평가만 가능합니다. 실제 coffee_martini 단독 실행 설정은 [RUN_COFFEE.md](../RUN_COFFEE.md)를 참고하세요.

창문 PSNR은 창문 안 픽셀만 평균합니다. 마스크 밖을 검게 만든 전체 이미지 PSNR은 사용하지 않습니다.
SSIM은 11x11 주변 영역이 전부 해당 영역 안에 들어오는 위치만 평균합니다. 매우 좁은 마스크에서는
유효 위치가 없어 SSIM이 null일 수 있습니다. LPIPS는 전체 영상만 계산합니다.
GT에는 실제 반사도 포함되므로, 창문 영역 지표의 상승만으로 반사를 제거한 순수 투과 영상이나
뒤쪽 물체의 정확한 3D 복원을 입증할 수 없습니다. 확대 이미지와 분리 레이어도 함께 확인하세요.

## 비교 모델과 원본 대비 변경점

- `rt`: 공식 RT renderer, occupancy/opacity 분리, 반사 shading, specular gradient gating.
- `rt_no_gating`: 위 구성에서 gradient gating만 제거.
- `gs`: 동일 surfel rasterizer의 **단일 opacity + SH 색상 제어 모델**. 공식 3DGS/2DGS 논문의
  재현 결과라고 부르지 않습니다. RT와 초기 effective opacity가 같게 설정합니다.
- 원본 학습 loop 대신 한 시점용 loop를 작성했습니다. 기본 LPIPS는 꺼져 있고, mask consistency는
  해상도에 덜 민감하도록 합계 대신 평균을 사용합니다. 빈 마스크 loss를 건너뛰어 NaN을 막습니다.
- Normal loss는 두 모델 모두 warmup 후 사용합니다. Gaussian 증식·제거는 공식 모델 메서드를 사용합니다.
- 원본 capture에서 빠지는 material/opacity/MLP 파라미터까지 포함하는 tensor checkpoint를 사용합니다.
- 이 실험은 원 논문의 공식 N3DV 결과가 아니며, 4D 변형장이나 동적 반사는 구현하지 않습니다.

```bash
python -m pytest -q
```
