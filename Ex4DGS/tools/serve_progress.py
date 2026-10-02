"""Private live report server: bind only loopback and reach via SSH forwarding."""
import argparse
import json
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--port',type=int,default=8082)
args=p.parse_args();root=args.run.resolve()

class Handler(SimpleHTTPRequestHandler):
    def __init__(self,*a,**kw):super().__init__(*a,directory=str(root/'previews'),**kw)
    def do_GET(self):
        if self.path.split('?')[0] == '/status.json':
            status={}
            for name,path in [('progress',root/'progress.json'),('latest',root/'previews/latest.json')]:
                try:status[name]=json.loads(path.read_text())
                except (OSError,ValueError):status[name]={}
            body=json.dumps(status).encode()
            self.send_response(200);self.send_header('Content-Type','application/json');self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(body);return
        if self.path.split('?')[0] in ['/','/live']:
            body='''<!doctype html><meta charset="utf-8"><title>Ex4DGS 학습 진행</title>
<style>body{font:17px system-ui;margin:24px;background:#f5f6f8}iframe{width:100%;height:1400px;border:0;background:white}#state{padding:15px;background:white}</style>
<h1>coffee_martini · Ex4DGS + 반사·투과 확장</h1>
<p>첫 프레임 / 학습 15대 / 검증 cam00·cam09 / 최종 test cam19. 기본 학습 30,000회 → 고오차 후보 → Ex4DGS와 RT 이식 모델을 각각 5,000회 추가 학습 → 검증으로 전체 모델 선택.</p>
<p>1,000회마다 결과를 저장합니다. PSNR 개선은 유리 존재의 확정이 아닙니다. 이전 기본 학습 프리뷰에는 세 미학습 카메라가 모두 포함되어 있습니다.</p>
<div id="state">상태 확인 중…</div><p><a id="current" href="index.html">현재 실험 결과</a> · <a href="region_psnr_30000/index.html">기본 모델 영역별 PSNR</a></p><iframe id="preview"></iframe>
<script>let last='';async function update(){try{let s=await(await fetch('status.json',{cache:'no-store'})).json();let p=s.progress||{};document.getElementById('state').textContent=(p.stage||'준비')+' · '+(p.iteration||p.iterations_per_branch||0)+' / '+(p.iterations_per_branch||p.total||'')+' · '+(p.status||'');let v=s.latest||{};let route=p.preview||(v.step?v.step+'/index.html':'index.html');let token=route+':'+Math.floor(Date.now()/60000);if(token!==last){last=token;document.getElementById('preview').src=route;document.getElementById('current').href=route;}}catch(e){document.getElementById('state').textContent='상태를 불러오지 못했습니다. 연결을 확인해주세요.'}}update();setInterval(update,5000);</script>'''.encode()
            self.send_response(200);self.send_header('Content-Type','text/html; charset=utf-8');self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(body);return
        super().do_GET()

ThreadingHTTPServer(('127.0.0.1',args.port),Handler).serve_forever()
