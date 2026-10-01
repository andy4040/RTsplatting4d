"""Export a visual report for the single RT run, without a competing method."""
import argparse
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',default='/workspace/RTsplatting4d/runs/coffee_rt_100k')
    args=p.parse_args()
    run=Path(args.run)
    output=run/'report'
    output.mkdir(exist_ok=True)
    metrics=json.loads((run/'evaluation/metrics.json').read_text())
    summary=json.loads((run/'training_summary.json').read_text())
    config=json.loads((run/'config.json').read_text())
    scores=metrics['per_view']['cam00']
    target=Image.open(run/'evaluation/cam00/target.png').convert('RGB')
    prediction=Image.open(run/'evaluation/cam00/prediction.png').convert('RGB')
    try:
        font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',25)
        small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',19)
    except OSError:
        font=small=ImageFont.load_default()
    board=Image.new('RGB',(1640,1170),'#101923')
    draw=ImageDraw.Draw(board)
    draw.text((25,15),'coffee_martini | frame 0 | held-out cam00 | RT-Splatting',font=font,fill='white')
    draw.text((25,51),f"{summary['steps']:,} iterations  |  {summary['gaussians']:,} Gaussians  |  full PSNR {scores['full']['psnr']:.2f} dB  |  window PSNR {scores['window']['psnr']:.2f} dB",font=small,fill='#b8d8ef')
    for x,title,image in [(20,'Ground truth',target),(830,'RT-Splatting render',prediction)]:
        draw.text((x,87),title,font=font,fill='white')
        board.paste(image.resize((790,593),Image.Resampling.LANCZOS),(x,124))
    crops=[('Back window / house',(180,452,488,656)),('Right window / exterior',(1220,430,1352,666))]
    for col,(title,box) in enumerate(crops):
        x=20+810*col
        draw.text((x,745),title+'  (GT | render)',font=small,fill='white')
        for j,image in enumerate([target,prediction]):
            crop=image.crop(box)
            crop.thumbnail((386,330),Image.Resampling.LANCZOS)
            scale=min(386/image.crop(box).width,330/image.crop(box).height)
            original=image.crop(box)
            crop=original.resize((round(original.width*scale),round(original.height*scale)),Image.Resampling.LANCZOS)
            board.paste(crop,(x+j*402,785))
            original.save(output/f"{'gt' if j==0 else 'render'}_crop_{col}.png")
    draw.text((20,1130),'No baseline experiment. Window metrics use task-created approximate glass masks.',font=small,fill='#b8d8ef')
    board.save(output/'visual_summary.png')
    target.save(output/'ground_truth.png')
    prediction.save(output/'render.png')
    history=[json.loads(line) for line in (run/'losses.jsonl').read_text().splitlines()]
    steps=np.array([x['step'] for x in history])
    loss=np.array([x['loss'] for x in history])
    fig,ax=plt.subplots(figsize=(10,4))
    ax.plot(steps,loss,alpha=.25,color='#1679ba',label='Sampled training loss')
    if len(loss)>=20:
        ax.plot(steps[19:],np.convolve(loss,np.ones(20)/20,'valid'),color='#1679ba',label='20-sample moving average')
    ax.axvline(config['warmup'],ls='--',color='gray',label='RT warmup ends')
    if config['lpips']:
        ax.axvline(15000,ls=':',color='orange',label='LPIPS starts')
    ax.set(xlabel='Iteration',ylabel='Training loss',title='RT-Splatting training on coffee_martini frame 0')
    ax.grid(alpha=.2); ax.legend(); fig.tight_layout(); fig.savefig(output/'training_curve.png',dpi=160); plt.close(fig)
    rows=''.join(f'<tr><td>{name}</td><td>{scores[name]["psnr"]:.3f}</td><td>{scores[name]["ssim"]:.4f}</td><td>{scores[name]["mae"]:.5f}</td></tr>' for name in ['full','window','opaque'])
    document='''<!doctype html><meta charset="utf-8"><title>coffee_martini RT-Splatting test</title>
<style>body{max-width:1400px;margin:30px auto;padding:0 20px;background:#101923;color:#e8f2fb;font:17px system-ui}img{max-width:100%}table{border-collapse:collapse;margin:20px 0}td,th{padding:10px 25px;border-bottom:1px solid #486078}input{width:100%}.compare{position:relative;line-height:0}.compare>img{width:100%}.overlay{position:absolute;inset:0;overflow:hidden;clip-path:inset(0 50% 0 0)}.overlay img{width:100%;height:100%;object-fit:fill}a{color:#8fd2ff}</style>
<h1>coffee_martini — RT-Splatting 테스트</h1><p>0번 프레임 · 학습 카메라 17개 · 테스트 전용 cam00 · ITERATIONS iterations</p>
<p>왼쪽 원본 / 오른쪽 렌더링. 슬라이더를 움직여 경계를 조절하세요.</p>
<div class="compare"><img src="render.png"><div id="overlay" class="overlay"><img src="ground_truth.png"></div></div>
<input type="range" min="0" max="100" value="50" oninput="document.getElementById('overlay').style.clipPath='inset(0 '+(100-this.value)+'% 0 0)'">
<table><tr><th>영역</th><th>PSNR (dB)</th><th>SSIM</th><th>MAE</th></tr>ROWS</table>
<p>창문 마스크는 이번 실행용으로 만든 근사 주석입니다. GT에는 실제 반사가 포함되며, 별도 모델과의 비교 실험은 하지 않았습니다.</p>
<h2>원본·렌더링과 창문 확대</h2><img src="visual_summary.png">
<h2>학습 곡선</h2><img src="training_curve.png">
<p><a href="../evaluation/metrics.json">원본 테스트 지표</a> · <a href="../config.json">학습 설정</a> · <a href="../training_summary.json">실행 기록</a></p>'''
    (output/'index.html').write_text(document.replace('ITERATIONS',f"{summary['steps']:,}").replace('ROWS',rows),encoding='utf-8')
    print(json.dumps(dict(report=str(output),scores=scores,training=summary),indent=2))


if __name__=='__main__':main()
