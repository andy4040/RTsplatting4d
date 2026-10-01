from pathlib import Path
import sys
import numpy as np
import torch
from PIL import Image

sys.path.insert(0,str(Path('/workspace/RTsplatting4d/third_party/RT-Splatting')))
from utils.loss_utils import lpips

image=np.array(Image.open('/workspace/RTsplatting4d/data/coffee_f000/images/cam01.png').convert('RGB'))
target=torch.from_numpy(image).permute(2,0,1).float().cuda()/255
pred=target.clone().requires_grad_(True)
loss=lpips(pred,target)
loss.backward()
assert torch.isfinite(loss) and torch.isfinite(pred.grad).all()
print({'lpips_forward_backward':'passed','image_shape':list(target.shape),'peak_vram_bytes':torch.cuda.max_memory_allocated()},flush=True)
