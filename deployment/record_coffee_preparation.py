"""Record environment and data provenance for the approved preparation only."""
import hashlib
import json
from pathlib import Path
import subprocess

repo = Path('/workspace/RTsplatting4d')
data = repo / 'data/coffee_martini_f000'
report = data / 'initialization/report'
receipt = {
    'github': 'https://github.com/andy4040/RTsplatting4d',
    'github_commit': subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),
    'tracked_source_changes': subprocess.check_output(['git','-C',str(repo),'diff','--name-only','HEAD'],text=True).splitlines(),
    'runtime': json.loads(Path('/workspace/runtime-check.json').read_text()),
    'colmap_version': '3.9.1',
    'dataset': str(data),
    'training_started': False,
    'transparent_masks_created': False,
    'remaining_before_training': 'Official loader requires transparent_masks; mask-free behavior needs user approval. Scene options also remain undecided.',
    'inputs_sha256': {str(p.relative_to(data)):hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in [data/'poses_bounds.npy',data/'approved_split.json',*sorted((data/'images').glob('*.png'))]},
    'camera_conversion_reference': 'https://github.com/hustvl/4DGaussians/blob/master/scripts/llff2colmap.py',
    'triangulation_reference': 'https://colmap.github.io/faq.html#reconstruct-sparse-dense-model-from-known-camera-poses',
    'inspection': 'Sparse projections visible on exterior building edges through central window. Trees and bright window regions have few points. No glass mask or ground-truth depth available to certify behind-glass geometry.'
}
assert receipt['tracked_source_changes'] == []
(report/'preparation.json').write_text(json.dumps(receipt, indent=2))
(report/'installed-rtsplat.txt').write_text(Path('/workspace/installed-rtsplat.txt').read_text())
page = report/'index.html'
html = page.read_text()
note = '''<section lang="ko"><h2>준비 완료 — 학습 전 점검 결과</h2>
<p>공식 모델·렌더러·손실 코드는 변경하지 않았습니다. Python 3.10 / Torch 2.0.1+cu118 환경에서 CUDA 연산과 미분 검증을 통과했습니다.</p>
<p>동일한 첫 프레임의 학습 사진 15장으로 점 5,620개를 생성했습니다. 카메라 위치·방향·초점거리를 고정했습니다. 테스트 cam00·cam09·cam19는 점군 생성에 사용하지 않았습니다.</p>
<p>재투영 오차 중앙값 0.956픽셀, 95백분위 2.700픽셀(원본 2704×2028 기준). 아래 사진은 학습 결과가 아니라 초기 점군을 원본 사진에 겹친 것입니다.</p>
<p>창문 너머 건물의 창틀·외벽에도 점이 보이지만, 나뭇가지와 밝은 창 영역에는 점이 드뭅니다. 투영 위치만으로 유리 뒤 깊이의 정확도를 보장할 수는 없습니다.</p>
<p>학습은 시작하지 않았고 SAM2 마스크도 만들지 않았습니다. 공식 학습 로더는 마스크 파일을 요구하므로, 마스크 없이 실행하는 방식은 별도 승인 후 처리해야 합니다.</p>
<p><a href="preparation.json">환경·입력 파일 검증 기록</a> · <a href="metrics.json">점군 측정값</a> · <a href="installed-rtsplat.txt">설치 버전</a></p></section>'''
if '준비 완료 — 학습 전 점검 결과' not in html:
    html = html.replace('<h1>', note + '<h1>', 1)
page.write_text(html, encoding='utf-8')
print(json.dumps({k:v for k,v in receipt.items() if k!='inputs_sha256'},indent=2))
