"""Read-only regional analysis of the 30k Ex4DGS baseline, before optical adaptation."""
import argparse
import csv
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from rt_port.runtime import load_baseline
from gaussian_renderer import render
from rt_pipeline.report import load_image, save_image


def db(mse):
    return float(-10*np.log10(max(float(mse), 1e-12)))


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--source', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    scene, options, _, pipe, _, iteration, _ = load_baseline(args.run, args.source, args.output)
    bg = torch.tensor([1.,1.,1.] if options.white_background else [0.,0.,0.], device='cuda')
    train = scene.train_cameras[1.0]
    validation = [c for c in scene.test_cameras[1.0] if c.image_name.startswith(('cam00_', 'cam09_'))]
    views = []; csv_rows = []
    for camera in sorted(train+validation, key=lambda c:c.image_name):
        gt = load_image(camera)
        pred = render(camera, scene.gaussians, pipe, bg, near=options.near, far=options.far)['render'].clamp(0,1)
        error = (pred.double()-gt.double()).square().mean(0).cpu().numpy()
        h,w = error.shape
        name = camera.image_name
        save_image(gt, args.output/f'{name}_gt.jpg')
        save_image(pred, args.output/f'{name}_render.jpg')
        view = {'name':name, 'split':'train' if camera in train else 'validation',
                'width':w, 'height':h, 'mse':float(error.mean()), 'psnr':db(error.mean()), 'grids':{}}
        for rows,cols in [(3,4),(6,8),(12,16)]:
            ys = np.linspace(0,h,rows+1,dtype=int); xs = np.linspace(0,w,cols+1,dtype=int)
            cells=[]
            for r in range(rows):
                for c in range(cols):
                    mse = float(error[ys[r]:ys[r+1],xs[c]:xs[c+1]].mean())
                    cell = dict(row=r+1,col=c+1,x=int(xs[c]),y=int(ys[r]),w=int(xs[c+1]-xs[c]),h=int(ys[r+1]-ys[r]),mse=mse,psnr=db(mse))
                    cells.append(cell)
                    csv_rows.append(dict(camera=name,split=view['split'],grid=f'{cols}x{rows}',**cell))
            view['grids'][f'{cols}x{rows}'] = cells
        views.append(view)
        print(name,view['split'],round(view['psnr'],3),flush=True)
    summary={}
    for split in ['train','validation']:
        group=[v for v in views if v['split']==split]
        summary[split]={'cameras':len(group), 'mean_camera_psnr':float(np.mean([v['psnr'] for v in group])),
            'pooled_pixel_psnr':db(sum(v['mse']*v['width']*v['height'] for v in group)/sum(v['width']*v['height'] for v in group))}
    data={'checkpoint':'baseline_final.pth','iteration':iteration,'views':views,'summary':summary,
          'reserved_test_camera':'cam19','metric':'RGB [0,1], predicted RGB clamped, float tensors before JPEG encoding; region PSNR from mean RGB squared error. Grid locations are not cross-view 3D correspondences.'}
    (args.output/'metrics.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    with (args.output/'regions.csv').open('w',newline='',encoding='utf-8-sig') as file:
        writer=csv.DictWriter(file,fieldnames=list(csv_rows[0]));writer.writeheader();writer.writerows(csv_rows)
    template=(Path(__file__).with_name('region_psnr_template.html')).read_text(encoding='utf-8')
    (args.output/'index.html').write_text(template.replace('__REPORT_DATA__',json.dumps(data,ensure_ascii=False)),encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
