"""Build a local iteration selector from downloaded previews and the final test."""
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
gallery = root / 'runs/coffee_previews'
gallery.mkdir(exist_ok=True)
entries = []
for folder in sorted(gallery.glob('step_*')):
    if (folder / 'report/index.html').is_file():
        score = json.loads((folder / 'evaluation/metrics.json').read_text())['per_view']['cam00']
        entries.append(dict(step=int(folder.name[5:]), path=folder.name, scores=score))
final = root / 'runs/coffee_completed/runs/coffee_rt_100k'
if (final / 'report/index.html').is_file():
    summary = json.loads((final / 'training_summary.json').read_text())
    entries.append(dict(step=summary['steps'],path='../coffee_completed/runs/coffee_rt_100k',
        scores=json.loads((final / 'evaluation/metrics.json').read_text())['per_view']['cam00']))
if not entries:
    raise RuntimeError('No downloaded preview yet')
document = '''<!doctype html><html lang="ko"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>coffee_martini 학습 중간 결과</title>
<style>body{max-width:1500px;margin:30px auto;padding:0 20px;background:#101923;color:#e8f2fb;font:18px system-ui}select{font:inherit;padding:8px}img{width:100%}a{color:#8fd2ff}.compare{position:relative;line-height:0}.overlay{position:absolute;inset:0;clip-path:inset(0 50% 0 0)}input{width:100%;margin:20px 0}p{line-height:1.6}</style>
<h1>coffee_martini · RT-Splatting</h1>
<p>0번 프레임 · 학습 17개 카메라 · 확인용 cam00 · 최종 목표 30,000회</p>
<label for="step">학습 횟수 </label><select id="step"></select>
<p id="scores"></p><p>왼쪽: 원본 / 오른쪽: 렌더링. 아래 슬라이더로 경계를 움직일 수 있습니다.</p>
<div class="compare"><img id="render" alt="선택한 학습 횟수의 RT 렌더링"><div id="overlay" class="overlay"><img id="target" alt="원본 사진"></div></div>
<input aria-label="원본과 렌더링 경계" type="range" min="0" max="100" value="50" oninput="document.getElementById('overlay').style.clipPath='inset(0 '+(100-this.value)+'% 0 0)'">
<h2>전체 화면과 창문 뒤 확대</h2><img id="summary" alt="원본과 렌더링, 창문 영역 확대">
<p><a id="report">선택한 횟수의 상세 보고서</a></p>
<p>창문 영역 지표는 이번 실행에서 만든 근사 마스크 기준입니다. 별도 모델과의 비교 실험은 하지 않았습니다.</p>
<script>const data=ENTRIES, select=document.getElementById('step');
data.forEach((r,i)=>{let o=document.createElement('option');o.value=i;o.textContent=r.step.toLocaleString()+'회';select.appendChild(o)});
function show(){let r=data[Number(select.value)],p=r.path+'/report/';
document.getElementById('render').src=p+'render.png';document.getElementById('target').src=p+'ground_truth.png';document.getElementById('summary').src=p+'visual_summary.png';document.getElementById('report').href=p+'index.html';
document.getElementById('scores').textContent='전체 PSNR '+r.scores.full.psnr.toFixed(2)+' dB · 창문 PSNR '+r.scores.window.psnr.toFixed(2)+' dB · 전체 SSIM '+r.scores.full.ssim.toFixed(4)}
select.value=data.length-1;select.onchange=show;show();</script></html>'''
(gallery/'index.html').write_text(document.replace('ENTRIES',json.dumps(entries)),encoding='utf-8')
print(json.dumps(dict(gallery=str(gallery/'index.html'),steps=[item['step'] for item in entries])))
