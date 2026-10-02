import html
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image


def load_image(camera):
    with Image.open(camera.image_path) as im:
        im = im.convert('RGB').resize(camera.resolution,Image.Resampling.BILINEAR)
        return torch.from_numpy(np.asarray(im).copy()).permute(2,0,1).float().cuda()/255


def save_image(tensor, path):
    values = tensor.detach().clamp(0,1).cpu().numpy()
    if values.shape[0] == 1: values = np.repeat(values,3,axis=0)
    Image.fromarray(np.uint8(values.transpose(1,2,0)*255+.5)).save(path)


def selected_cameras(cameras, frames):
    return [c for c in cameras if int(c.timestamp) in frames]


def write_preview(root, label, cameras, render_function, extra=None):
    root = Path(root); folder = root/label; folder.mkdir(parents=True,exist_ok=True)
    metrics = []
    with torch.no_grad():
        for camera in cameras:
            target = load_image(camera)
            package = render_function(camera)
            prediction = package['render'].clamp(0,1)
            mse = (prediction-target).square().mean().item()
            name = camera.image_name
            metrics.append({'image':name,'psnr':float(-10*np.log10(max(mse,1e-12)))})
            save_image(target,folder/(name+'_gt.jpg'))
            for key in ['render','transmission','reflection','reflection_weight','surface_coverage']:
                if key in package: save_image(package[key],folder/(name+'_'+key+'.jpg'))
    result = {'step':label,'views':metrics,'mean_psnr':float(np.mean([m['psnr'] for m in metrics])) if metrics else None,**(extra or {})}
    (folder/'metrics.json').write_text(json.dumps(result,indent=2))
    rows=[]
    for item in metrics:
        name=item['image']
        images=''.join(f'<figure><figcaption>{key}</figcaption><a href="{name}_{key}.jpg"><img src="{name}_{key}.jpg"></a></figure>' for key in ['gt','render','transmission','reflection','reflection_weight','surface_coverage'] if (folder/(name+'_'+key+'.jpg')).exists())
        rows.append(f'<h2>{html.escape(name)} — {item["psnr"]:.2f} dB</h2><div class="row">{images}</div>')
    (folder/'index.html').write_text('<!doctype html><meta charset="utf-8"><style>body{font:16px system-ui;margin:25px}.row{display:flex;flex-wrap:wrap}figure{margin:5px;width:46%}img{width:100%}pre{white-space:pre-wrap}</style><h1>Ex4DGS · 반사·투과 표면 가설 실험</h1><p>후보 마스크는 임시 가정이며 유리 정답이 아닙니다. 검증 카메라는 후보 채택에, 최종 test 카메라는 선택 후 평가에 사용합니다. 기본 학습의 과거 3대 카메라 프리뷰와 구분해 아래 split을 확인하세요.</p><pre>'+html.escape(json.dumps(result,indent=2))+'</pre>'+''.join(rows),encoding='utf-8')
    steps=sorted(p.parent.name for p in root.glob('*/metrics.json'))
    (root/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>Ex4DGS 학습 결과</h1><p>기본 학습 → 지속 고오차 임시 마스크 → 별도 표면 가설 → 검증 후 채택. 사람이 표시한 유리 정답 마스크는 사용하지 않습니다.</p><p><a href="surface_candidates/index.html">후보 마스크와 표면 초기화</a></p>'+''.join(f'<p><a href="{s}/index.html">{s}</a></p>' for s in steps),encoding='utf-8')
    (root/'latest.json').write_text(json.dumps({'step':label,**result},indent=2))
    return result
