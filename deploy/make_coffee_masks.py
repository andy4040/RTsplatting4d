"""Frame-0 approximate glass masks: inspected pane outlines minus person segmentation.

Coordinates are on 676x507 previews of the user's coffee_martini images. These
are task-created annotations, not dataset ground truth. Keep overlays for review.
"""
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights

ROOT = Path('/workspace/RTsplatting4d/data/coffee_f000')
MAIN = {
'cam00': [[87,224],[400,224],[401,328],[87,329]],
'cam01': [[271,232],[550,216],[551,343],[271,315]],
'cam02': [[241,232],[516,221],[516,342],[241,320]],
'cam04': [[169,227],[446,225],[446,333],[168,315]],
'cam05': [[136,228],[445,229],[445,334],[136,324]],
'cam06': [[0,225],[366,234],[366,339],[0,341]],
'cam07': [[0,212],[353,223],[353,328],[0,335]],
'cam08': [[0,212],[328,224],[329,329],[0,339]],
'cam09': [[0,205],[306,216],[307,324],[0,345]],
'cam10': [[0,201],[282,218],[281,342],[0,353]],
'cam11': [[114,193],[384,189],[384,312],[114,294]],
'cam12': [[68,179],[359,182],[359,285],[67,273]],
'cam13': [[194,175],[468,177],[468,288],[193,271]],
'cam14': [[155,195],[443,199],[443,306],[154,293]],
'cam16': [[72,175],[391,186],[391,297],[73,288]],
'cam18': [[0,145],[352,185],[352,284],[0,279]],
'cam19': [[23,201],[393,237],[394,346],[19,339]],
'cam20': [[0,158],[366,216],[366,325],[0,315]],
}
CENTERS = {
'cam00': [249,224,249,330], 'cam01': [387,226,387,328],
'cam02': [362,226,362,333], 'cam04': [302,224,302,328],
'cam05': [292,228,292,329], 'cam06': [208,230,208,337],
'cam07': [185,218,184,333], 'cam08': [161,219,161,335],
'cam09': [120,210,119,340], 'cam10': [84,206,82,350],
'cam11': [243,191,243,304], 'cam12': [217,180,217,280],
'cam13': [320,176,320,280], 'cam14': [294,197,293,300],
'cam16': [235,181,234,293], 'cam18': [188,166,179,283],
'cam19': [230,221,225,342], 'cam20': [195,188,186,322],
}
RIGHT = {
'cam00': [[[614,32],[663,0],[663,348],[613,334]]],
'cam01': [], 'cam02': [], 'cam04': [],
'cam05': [[[665,3],[675,0],[675,376],[663,367]]],
'cam06': [[[572,58],[653,5],[653,359],[568,329]], [[669,0],[675,0],[675,370],[669,368]]],
'cam07': [[[528,64],[621,10],[617,348],[524,329]], [[656,0],[675,0],[675,368],[655,360]]],
'cam08': [[[492,63],[580,10],[575,345],[489,320]], [[611,0],[675,0],[675,368],[605,346]]],
'cam09': [[[464,69],[549,22],[548,342],[461,314]], [[583,5],[675,0],[675,374],[579,349]]],
'cam10': [[[440,90],[521,54],[520,362],[439,333]], [[553,45],[675,1],[675,388],[550,368]]],
'cam11': [[[606,0],[675,0],[675,344],[604,315]]],
'cam12': [[[578,3],[663,0],[662,324],[575,299]]],
'cam13': [], 'cam14': [],
'cam16': [[[600,0],[675,0],[675,303],[598,289]]],
'cam18': [[[527,54],[611,1],[608,306],[523,282]], [[647,0],[675,0],[675,324],[643,308]]],
'cam19': [[[554,103],[642,37],[655,378],[561,348]]],
'cam20': [[[525,117],[609,71],[617,355],[527,332]], [[645,51],[675,32],[675,368],[651,358]]],
}

weights = DeepLabV3_ResNet50_Weights.DEFAULT
net = deeplabv3_resnet50(weights=weights).eval().cuda()
transform = weights.transforms()
(ROOT / 'mask_previews').mkdir(exist_ok=True)
coverage = {}
for name in MAIN:
    image = Image.open(ROOT / 'images' / f'{name}.png').convert('RGB')
    array = np.array(image)
    w, h = image.size
    scale = np.array([w/676, h/507])
    mask = np.zeros((h,w), np.uint8)
    for poly in [MAIN[name], *RIGHT[name]]:
        cv2.fillPoly(mask, [np.rint(np.array(poly)*scale).astype(np.int32)], 255)
    # Main central sash: opaque frame, not transmitted radiance.
    line = np.rint(np.array(CENTERS[name]).reshape(2,2)*scale).astype(int)
    cv2.line(mask, tuple(line[0]), tuple(line[1]), 0, max(2, round(13*w/676)))
    # Person prediction is derived independently from each camera, not cross-view RGB.
    with torch.inference_mode():
        logits = net(transform(image).unsqueeze(0).cuda())['out']
        person = logits.softmax(1)[0,15].cpu().numpy()
    person = cv2.resize(person, (w,h)) > .20
    person = cv2.dilate(person.astype(np.uint8), np.ones((7,7), np.uint8)).astype(bool)
    mask[person] = 0
    mask = cv2.erode(mask, np.ones((5,5), np.uint8))
    Image.fromarray(mask).save(ROOT / 'masks' / f'{name}.png')
    overlay = array.copy()
    overlay[mask>0] = (array[mask>0]*.60 + np.array([0,255,120])*.40).astype(np.uint8)
    preview = Image.fromarray(overlay).resize((676,507))
    ImageDraw.Draw(preview).text((8,8),name,fill='yellow')
    preview.save(ROOT / 'mask_previews' / f'{name}.jpg')
    coverage[name] = float((mask>0).mean())
    print(name, coverage[name], flush=True)
record = dict(frame=0, method='Inspected coarse pane polygons, central-frame exclusion, pretrained DeepLab person exclusion and 2px erosion',
              annotation_status='task-created approximate masks, not official glass ground truth; thin internal grille bars may remain',
              main_polygons_676x507=MAIN, right_polygons_676x507=RIGHT, coverage=coverage)
(ROOT/'mask_provenance.json').write_text(json.dumps(record,indent=2))
board = Image.new('RGB',(2028,507*6))
for index,name in enumerate(MAIN):
    board.paste(Image.open(ROOT/'mask_previews'/f'{name}.jpg'),((index%3)*676,(index//3)*507))
board.save(ROOT/'mask_previews'/'all.jpg')
