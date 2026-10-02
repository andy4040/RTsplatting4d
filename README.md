## 이 저장소의 N3DV / Ex4DGS 실험

공식 RT-Splatting 소스는 루트에 보존하고, N3DV `coffee_martini` 실험은 [`Ex4DGS/`](Ex4DGS)에 별도로 추가했습니다. Ex4DGS 원본은 [juno181/Ex4DGS](https://github.com/juno181/Ex4DGS)의 `7fac64997164fde64732c80261e56adf66ca14df`이며, CUDA 의존성 소스와 원본 라이선스도 포함합니다.

**현재 확장은 RT-Splatting 논문 재현이 아닙니다.** 현재 코드는 지속 고오차 영역을 임시 마스크로 만들고, 별도의 얇은 평면 가우시안 표면을 초기화하여 검증하는 실험입니다. RT-Splatting은 하나의 가우시안 집합에서 점유도와 광학적 불투명도를 분리하고 기하·광학 성질을 함께 최적화합니다. 유리의 정답 깊이를 사전에 입력하거나 별도 평면을 추가하는 것이 논문의 필수 조건은 아닙니다. 현재 평면 가정은 이 프로젝트에서 추가한 것이며, 논문의 통합 구조로 바꾸는 작업은 아직 구현하지 않았습니다. [논문 §4.1–4.4](https://arxiv.org/html/2605.18263v1)

- 데이터 준비: [`tools/extract_n3dv_frame.py`](tools/extract_n3dv_frame.py), [`tools/prepare_n3dv_colmap.py`](tools/prepare_n3dv_colmap.py).
- 기본 학습: [`Ex4DGS/run_residual_rt.py`](Ex4DGS/run_residual_rt.py). 현재 실험은 동영상 전체가 아닌 첫 프레임입니다.
- 후속 실험: [`Ex4DGS/run_surface_hypotheses.py`](Ex4DGS/run_surface_hypotheses.py). 같은 기본 체크포인트에서 기본 모델과 표면 추가 모델을 각각 5,000회 추가 학습합니다.
- 모델: [`Ex4DGS/rt_pipeline/surfaces.py`](Ex4DGS/rt_pipeline/surfaces.py). 후보 밖은 유리가 없다는 정답으로 감독하지 않습니다.
- 설정·제약: [`Ex4DGS/EXPERIMENT.json`](Ex4DGS/EXPERIMENT.json), [`후속 실험 설정`](Ex4DGS/configs/coffee_surface_hypotheses.json).
- 검증 기록: [`records/surface_validation_20261002.json`](records/surface_validation_20261002.json). CUDA 및 짧은 통합 테스트의 기록이며, 화질 개선의 증거는 아닙니다.

학습용 15개 카메라를 유지하고, cam00·cam09는 후보 채택 검증에, cam19는 선택 후 최종 평가에 사용합니다. 이전 기본 학습 프리뷰에는 세 미학습 카메라가 모두 포함되어 있었습니다. PSNR 개선은 반사·투과 표현의 유용성을 평가하며, 실제 유리 존재를 확정하지 않습니다.

`deployment/`는 현재 서버의 `/workspace/RTsplatting4d`, `/workspace/Ex4DGS`, CUDA 11.8 및 별도 Python 환경 경로를 사용하는 실행 기록입니다. 새 환경에서는 경로를 맞춰야 하며, 복구·후속 실행 활성화 스크립트는 실행 중인 작업을 제어하므로 일반 설치 스크립트처럼 일괄 실행하지 마세요. Ex4DGS 작업 디렉터리는 저장소 안의 `Ex4DGS/`입니다. 기존 체크포인트를 재사용하는 실행 예시는 다음과 같습니다.

```bash
cd Ex4DGS
python run_surface_hypotheses.py \
  --baseline-run /path/to/completed_baseline \
  --source /path/to/coffee_martini_f000 \
  --config configs/coffee_surface_hypotheses.json \
  --output /path/to/new_surface_trial
```

원본 데이터, 추출 이미지, 학습 체크포인트 및 대용량 결과물은 Git에 포함하지 않습니다. 아래는 보존된 공식 RT-Splatting 설명입니다.

---

<div align="center">

# RT-Splatting: Joint Reflection-Transmission Modeling with Gaussian Splatting

[**Ji Shi**](https://github.com/sjj118) · [**Xianghua Ying**](https://scholar.google.com/citations?hl=zh-CN&user=27o9L1wAAAAJ) · [**Bowei Xing**](https://dblp.org/pid/320/5822.html)
<br>
[**Ruohao Guo**](https://ruohaoguo.github.io/) · [**Wenzhen Yue**](https://scholar.google.com/citations?hl=zh-CN&user=UPxl-gMAAAAJ)
<br>

CVPR 2026 (Highlight)
<br>

[![arXiv](https://img.shields.io/badge/arXiv-2605.18263-b31b1b)](https://arxiv.org/pdf/2605.18263)
[![Project Page](https://img.shields.io/badge/Project-Page-green)](https://sjj118.github.io/RT-Splatting)
[![Dataset](https://img.shields.io/badge/Drive-Dataset-4285F4)](https://drive.google.com/drive/folders/1mmKcm1Fb5djX3B_PDKfC7XyfQ38_p5nl)
</div>

![Teaser image](assets/teaser.png) 
RT-Splatting is a hybrid surface-volume rendering framework that jointly models high-fidelity reflections and clear transmissions for semi-transparent scenes. It overcomes the blurry reflections and occluded backgrounds of existing methods, delivering state-of-the-art, real-time view synthesis. Beyond rendering, it perfectly decomposes the scene into independent reflection and transmission layers, unlocking powerful and intuitive material editing capabilities.

## Installation

```shell
conda create -n rtsplat python=3.10 -y
conda activate rtsplat
pip install -r requirements.txt

pip install --no-build-isolation submodules/simple-knn
pip install --no-build-isolation submodules/diff-surfel-anych

pip install --no-build-isolation git+https://github.com/NVlabs/nvdiffrast
```

## Datasets

We evaluate our method primarily on [Ref-Real](https://storage.googleapis.com/gresearch/refraw360/ref_real.zip), [NeRF-Casting](https://dorverbin.github.io/nerf-casting/), [EnvGS](https://drive.google.com/file/d/1FMtj2YvdbaQe8vxwZlULcSBuTWb4pI7I), [Tanks&Temples](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip) and our self-captured scenes. Transparent masks and our self-captured scenes are available on [Google Drive](https://drive.google.com/drive/folders/1mmKcm1Fb5djX3B_PDKfC7XyfQ38_p5nl).

Put them under the `data` folder:

```
data/
├── rt-splatting/
│   ├── van/
│   │   ├── images/
│   │   ├── sparse/
│   │   └── transparent_masks/
│   └── swab/
├── nerf-casting/
├── ...
```

## Training & Evaluation

```shell
sh eval.sh
```

## Acknowledgements

This work is built on a number of inspiring research works:

- [2DGS: 2D Gaussian Splatting for Geometrically Accurate Radiance Fields](https://surfsplatting.github.io/)
- [Ref-GS : Directional Factorization for 2D Gaussian Splatting](https://ref-gs.github.io/)
- [EnvGS: Modeling View-Dependent Appearance with Environment Gaussian](https://zju3dv.github.io/envgs/)
- [NeRF-Casting: Improved View-Dependent Appearance with Consistent Reflections](https://dorverbin.github.io/nerf-casting/)
- [Ref-NeRF: Structured View-Dependent Appearance for Neural Radiance Fields](https://dorverbin.github.io/refnerf/)
- [SAM 2: Segment Anything in Images and Videos](https://ai.meta.com/sam2)

## Citation

If you find our work useful in your research, please cite:

```bibtex
@inproceedings{RT-Splatting,
  title={{RT-Splatting}: Joint Reflection-Transmission Modeling with Gaussian Splatting},
  author={Shi, Ji and Ying, Xianghua and Xing, Bowei and Guo, Ruohao and Yue, Wenzhen},
  booktitle={CVPR},
  year={2026},
}
```
