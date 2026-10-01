"""Export downloaded visual results to a Git-friendly, self-contained directory."""
import argparse
import json
from pathlib import Path
import shutil


def export(source, output):
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    candidates = sorted((source / 'runs/coffee_previews').glob('step_*'))
    final = source / 'runs/coffee_completed/runs/coffee_rt_100k'
    if (final / 'report/index.html').exists():
        candidates.append(final)
    rows = {}
    for folder in candidates:
        if not (folder / 'report/index.html').is_file():
            continue
        summary = json.loads((folder / 'training_summary.json').read_text(encoding='utf-8'))
        metrics = json.loads((folder / 'evaluation/metrics.json').read_text(encoding='utf-8'))
        config = json.loads((folder / 'config.json').read_text(encoding='utf-8'))
        step = summary['steps']
        destination = output / f'step_{step:06d}'
        shutil.copytree(folder / 'report', destination / 'report', dirs_exist_ok=True)
        (destination / 'evaluation').mkdir(exist_ok=True)
        shutil.copyfile(folder / 'evaluation/metrics.json', destination / 'evaluation/metrics.json')
        shutil.copyfile(folder / 'training_summary.json', destination / 'training_summary.json')
        config['scene'] = 'data/coffee_f000'
        config.pop('baseline', None)
        (destination / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
        rows[step] = dict(step=step, path=destination.name, final=folder == final,
                          metrics=metrics['per_view']['cam00'])
    entries = [rows[step] for step in sorted(rows)]
    if not entries:
        raise RuntimeError('No complete downloaded reports to export')
    latest = entries[-1]
    shutil.copyfile(output/latest['path']/'report/visual_summary.png', output/'latest.png')
    lines = ['# coffee_martini · frame 0', '',
        'RT-Splatting 단독 실행 · 학습 카메라 17개 · 확인용 cam00 · 해상도 1352 × 1014.', '',
        f"최신 공개 결과: **{latest['step']:,}회**. " + ('최종 테스트 완료.' if latest['final'] else '**학습 중간 결과이며, 최종 목표는 30,000회입니다.**'), '',
        f"![최신 원본·렌더링과 창문 확대]({latest['path']}/report/visual_summary.png)", '',
        '| Iteration | 전체 PSNR ↑ | 창문 PSNR ↑ | 전체 SSIM ↑ | LPIPS ↓ | 상세 결과 |',
        '|---:|---:|---:|---:|---:|---|']
    for entry in entries:
        score = entry['metrics']
        lpips = score['lpips_full']
        lpips_text = f'{lpips:.4f}' if lpips is not None else '—'
        lines.append(f"| {entry['step']:,} | {score['full']['psnr']:.3f} | {score['window']['psnr']:.3f} | {score['full']['ssim']:.4f} | {lpips_text} | [이미지]({entry['path']}/report/visual_summary.png) · [지표]({entry['path']}/evaluation/metrics.json) |")
    lines += ['', '브라우저에서 슬라이더를 쓰려면 저장소를 내려받고 `index.html` 또는 각 단계의 `report/index.html`을 여세요.', '',
        '창문 마스크는 이번 실행에서 만든 근사 주석입니다. GT에는 실제 반사가 포함됩니다. 별도 모델과의 비교 실험은 수행하지 않았으며, 현재 수치만으로 원본 방법 대비 개선을 주장하지 않습니다.', '',
        '모델과 렌더러는 원본을 사용하지만, 데이터 준비·학습 루프·일부 손실 계산을 변경했습니다. 자세한 설정은 루트의 `RUN_COFFEE.md`와 각 단계의 `config.json`을 참고하세요.', '',
        '대용량 모델 체크포인트, 원본 영상, float 렌더링 배열은 이 Git 결과 묶음에 포함하지 않습니다.', '']
    (output/'README.md').write_text('\n'.join(lines), encoding='utf-8')
    (output/'manifest.json').write_text(json.dumps(dict(target_iterations=30000, entries=entries),indent=2),encoding='utf-8')
    html = '''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>coffee_martini RT-Splatting 결과</title><style>body{max-width:1500px;margin:30px auto;padding:0 20px;background:#101923;color:#e8f2fb;font:18px system-ui}img{width:100%}select{font:inherit;padding:8px}a{color:#8fd2ff}</style>
<h1>coffee_martini · RT-Splatting</h1><p>frame 0 · cam00 · 목표 30,000회</p><label for="step">학습 횟수 </label><select id="step"></select><p id="scores"></p><img id="image" alt="원본과 RT 렌더링, 창문 확대"><p><a id="detail">슬라이더와 상세 보고서 열기</a></p>
<script>const entries=ENTRIES,select=document.getElementById('step');entries.forEach((r,i)=>{let o=document.createElement('option');o.value=i;o.textContent=r.step.toLocaleString()+'회'+(r.final?' (최종)':' (중간)');select.appendChild(o)});function show(){let r=entries[+select.value];document.getElementById('image').src=r.path+'/report/visual_summary.png';document.getElementById('detail').href=r.path+'/report/index.html';document.getElementById('scores').textContent='전체 PSNR '+r.metrics.full.psnr.toFixed(2)+' dB · 창문 PSNR '+r.metrics.window.psnr.toFixed(2)+' dB'}select.value=entries.length-1;select.onchange=show;show();</script></html>'''
    (output/'index.html').write_text(html.replace('ENTRIES',json.dumps(entries)),encoding='utf-8')
    print(json.dumps(dict(output=str(output),steps=list(rows),latest=latest['step'])))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--out',type=Path,default=Path(__file__).resolve().parents[1]/'results/coffee_martini')
    args=parser.parse_args()
    export(args.source,args.out)
