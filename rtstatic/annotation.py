"""Offline polygon annotation: no web service, model download, or image upload."""
import base64
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .data import read_json


def export_editor(scene, output):
    root = Path(scene)
    manifest = read_json(root / "scene.json")
    images = []
    for c in manifest["cameras"]:
        content = base64.b64encode((root / "images" / f"{c['name']}.png").read_bytes()).decode()
        images.append(dict(name=c["name"], src="data:image/png;base64," + content))
    html = EDITOR.replace("__IMAGES__", json.dumps(images)).replace("__FRAME__", str(manifest["frame"]))
    Path(output).write_text(html, encoding="utf-8")


def import_polygons(scene, source):
    root = Path(scene)
    manifest = read_json(root / "scene.json")
    data = read_json(source)
    expected = {c["name"] for c in manifest["cameras"]}
    if data.get("frame") != manifest["frame"] or set(data.get("polygons", {})) != expected:
        raise ValueError("Annotation frame/camera names do not match this scene")
    if set(data.get("reviewed", [])) != expected:
        raise ValueError("Review every camera (including cameras with no visible glass) before exporting")
    masks = {}
    for c in manifest["cameras"]:
        mask = np.zeros((c["height"], c["width"]), dtype=np.uint8)
        for polygon in data["polygons"][c["name"]]:
            points = np.asarray(polygon, dtype=float)
            if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3 or not np.isfinite(points).all():
                raise ValueError(f"Invalid polygon for {c['name']}")
            if (points < 0).any() or (points > 1).any():
                raise ValueError("Polygon coordinates must be normalized to [0,1]")
            pixels = np.rint(points * [c["width"] - 1, c["height"] - 1]).astype(np.int32)
            cv2.fillPoly(mask, [pixels], 255)
        masks[c["name"]] = mask
    for folder in ["masks", "mask_previews"]:
        (root / folder).mkdir(exist_ok=True)
    for name, mask in masks.items():
        Image.fromarray(mask).save(root / "masks" / f"{name}.png")
        image = np.array(Image.open(root / "images" / f"{name}.png").convert("RGB"))
        image[mask > 0] = (image[mask > 0] * 0.55 + np.array([0, 255, 150]) * 0.45).astype(np.uint8)
        Image.fromarray(image).save(root / "mask_previews" / f"{name}.png")


EDITOR = r'''<!doctype html><meta charset="utf-8"><title>N3DV window masks</title>
<style>body{background:#15202b;color:white;font:16px system-ui;margin:24px}canvas{max-width:100%;cursor:crosshair;border:1px solid #aaa}button{padding:9px;margin:4px}#info{margin:12px}</style>
<h1>N3DV 창문 영역 표시 — frame __FRAME__</h1>
<p>창문 유리의 보이는 영역을 클릭해 다각형을 그립니다. 창틀·앞쪽 사람 등 불투명 가림 물체는 제외하세요.
여러 다각형을 사용할 수 있습니다. 각 카메라를 확인한 후 ‘확인 완료’를 누르세요. 창문이 없으면 빈 상태로 확인합니다.</p>
<button onclick="prev()">이전</button><button onclick="next()">다음</button>
<button onclick="finish()">다각형 완료 (Enter)</button><button onclick="undo()">점 취소</button>
<button onclick="clearMask()">이 카메라 초기화</button><button onclick="review()">확인 완료</button>
<button onclick="download()">polygons.json 저장</button><div id="info"></div><canvas id="canvas"></canvas>
<script>
const images=__IMAGES__, polygons=Object.fromEntries(images.map(x=>[x.name,[]]));
const reviewed=new Set(), canvas=document.getElementById('canvas'), ctx=canvas.getContext('2d');
let index=0, current=[], img=new Image();
function draw(){ctx.clearRect(0,0,canvas.width,canvas.height);ctx.drawImage(img,0,0);
 for(const poly of [...polygons[images[index].name],current]){if(!poly.length)continue;
 ctx.beginPath();poly.forEach((p,i)=>{let x=p[0]*(canvas.width-1),y=p[1]*(canvas.height-1);i?ctx.lineTo(x,y):ctx.moveTo(x,y)});
 ctx.closePath();ctx.fillStyle='#00ee8870';ctx.fill();ctx.strokeStyle='#00ff99';ctx.lineWidth=3;ctx.stroke()}
 document.getElementById('info').textContent=`${index+1}/${images.length}: ${images[index].name} — ${reviewed.has(images[index].name)?'확인 완료':'미확인'} (${reviewed.size}/${images.length})`}
function load(){current=[];img=new Image();img.onload=()=>{canvas.width=img.width;canvas.height=img.height;draw()};img.src=images[index].src}
canvas.onclick=e=>{let r=canvas.getBoundingClientRect();current.push([Math.min(1,Math.max(0,(e.clientX-r.left)/r.width)),Math.min(1,Math.max(0,(e.clientY-r.top)/r.height))]);reviewed.delete(images[index].name);draw()};
function finish(){if(current.length<3){alert('3개 이상의 점이 필요합니다.');return}polygons[images[index].name].push(current);current=[];reviewed.delete(images[index].name);draw()}
function undo(){current.pop();draw()}
function clearMask(){current=[];polygons[images[index].name]=[];reviewed.delete(images[index].name);draw()}
function move(delta){if(current.length){alert('현재 다각형을 완료하거나 점을 취소하세요.');return}index=(index+delta+images.length)%images.length;load()}
function prev(){move(-1)}function next(){move(1)}
function review(){if(current.length){alert('다각형을 먼저 완료하세요.');return}reviewed.add(images[index].name);draw()}
function download(){if(current.length||reviewed.size!==images.length){alert('모든 카메라를 확인하세요.');return}
 let data={frame:__FRAME__,polygons,reviewed:[...reviewed]}, a=document.createElement('a');
 a.href=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));a.download='polygons.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)}
document.addEventListener('keydown',e=>{if(e.key==='Enter')finish()});load();
</script>'''
