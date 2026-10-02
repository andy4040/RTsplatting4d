import json
from pathlib import Path
import random
import numpy as np
import torch

from gaussian_renderer import render
from utils.loss_utils import l1_loss, ssim
from rt_pipeline.hard_regions import patch_mse, persistent_candidates, expand_patches, select_points
from rt_pipeline.optics import CandidateOptics
from rt_pipeline.report import load_image, save_image, write_preview, selected_cameras


class ResidualMonitor:
    def __init__(self, config, output, near, far):
        self.cfg = config
        self.output = Path(output)
        self.near,self.far = near,far
        self.history = {}
        self.last_iteration = -1
        self.frames = config['probe_frames']
        (self.output/'residuals').mkdir(parents=True,exist_ok=True)

    @torch.no_grad()
    def __call__(self, iteration, scene, pipe, background):
        cfg=self.cfg
        if iteration % 100 == 0 or iteration == 1:
            (self.output/'progress.json').write_text(json.dumps({'stage':'baseline','iteration':iteration,
                'total':cfg['baseline_iterations'],'points':len(scene.gaussians.get_xyz_at_t(0)),
                'status':'running'},indent=2))
        if iteration % cfg['preview_every'] == 0 or iteration == cfg['baseline_iterations']:
            checkpoint_path=self.output/'baseline_latest.tmp'
            torch.save((scene.gaussians.capture(),iteration),checkpoint_path)
            checkpoint_path.replace(self.output/'baseline_latest.pth')
            preview = selected_cameras(scene.test_cameras[1.0],self.frames)
            preview = [c for c in preview if c.timestamp <= scene.gaussians.duration]
            if preview:
                result=write_preview(self.output/'previews',f'baseline_{iteration:06d}',preview,
                    lambda c:render(c,scene.gaussians,pipe,background,near=self.near,far=self.far),
                    {'stage':'baseline','candidate_selection_uses_test':False})
                print(f'PREVIEW baseline {iteration}: test mean PSNR={result["mean_psnr"]:.3f}',flush=True)
        if iteration not in cfg['observe_iterations']:
            return
        train = selected_cameras(scene.train_cameras[1.0],self.frames)
        for camera in train:
            assert camera not in scene.test_cameras[1.0]
            prediction=render(camera,scene.gaussians,pipe,background,near=self.near,far=self.far)['render'].clamp(0,1)
            target=load_image(camera)
            patches, global_mse=patch_mse(prediction.cpu().numpy(),target.cpu().numpy(),cfg['patch_size'])
            self.history.setdefault(camera.image_name,[]).append({'iteration':iteration,'mse':patches,'global_mse':global_mse})
            np.savez_compressed(self.output/'residuals'/f'{camera.image_name}_{iteration:06d}.npz',mse=patches,global_mse=global_mse,iteration=iteration)
        self.last_iteration=iteration
        print(f'RESIDUAL observation {iteration}: {len(train)} train images, no held-out input',flush=True)

    @torch.no_grad()
    def select(self,scene,pipe,background):
        cfg=self.cfg; pc=scene.gaussians
        if self.last_iteration != cfg['baseline_iterations']:
            raise ValueError('Final baseline must be an observation iteration')
        n=len(pc.get_xyz_at_t(0))
        votes=np.zeros(n,np.int32); scores=np.zeros(n,np.float64)
        camera_votes={}; counts={}
        output=self.output/'candidates'; output.mkdir(exist_ok=True)
        train=selected_cameras(scene.train_cameras[1.0],self.frames)
        for camera in train:
            history=self.history[camera.image_name]
            patches,stats=persistent_candidates(history,ratio=cfg['error_ratio'],floor=cfg['mse_floor'],
                persistence=cfg['persistence'],max_improvement=cfg['max_improvement'],min_observations=cfg['min_observations'])
            package=render(camera,pc,pipe,background,near=self.near,far=self.far)
            ids=package['dominent_idxs'].squeeze().cpu().numpy().astype(np.int64)
            coverage=package['acc'].squeeze().cpu().numpy()
            h,w=ids.shape; mask=expand_patches(patches,h,w,cfg['patch_size'])
            valid=(ids>=0)&(ids<n)&(coverage>.1)
            visible=np.bincount(ids[valid],minlength=n)
            hard=np.bincount(ids[valid&mask],minlength=n)
            supported=(hard>=cfg['min_pixels_per_view']) & (hard/np.maximum(visible,1)>=cfg['min_hard_fraction'])
            # Multiple timestamps from one camera do not count as multiple views.
            camera_votes.setdefault(camera.colmap_id,np.zeros(n,bool))[:] |= supported
            scores+=hard/np.maximum(visible,1)
            counts[camera.image_name]={'selected_patches':int(patches.sum()),'total_patches':int(patches.size),
                'supported_points':int(supported.sum()),'observations':len(history)}
            np.savez_compressed(output/(camera.image_name+'.npz'),candidate_patches=patches,**stats)
            target=load_image(camera)
            overlay=target*.65+torch.from_numpy(mask).cuda()[None]*torch.tensor([.35,0.,.35],device='cuda')[:,None,None]
            save_image(overlay,output/(camera.image_name+'_candidate.jpg'))
        for support in camera_votes.values(): votes+=support
        candidates=select_points(votes,scores,cfg['min_camera_views'],cfg['max_candidate_fraction'])
        np.savez_compressed(output/'points.npz',candidates=candidates,votes=votes,scores=scores)
        summary={'total_points':n,'candidate_points':int(candidates.sum()),'views':counts,
            'semantic_glass_labels':False,'uses_test_images':False,
            'limitation':'Dominant residual-bearing Gaussians are candidates, not localized glass surfaces; reflection vs geometry errors remain ambiguous.'}
        (output/'summary.json').write_text(json.dumps(summary,indent=2))
        (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>지속적인 고오차 후보</h1><p>보라색은 추가 표현을 검토할 영역이며 유리 정답이 아닙니다. 학습 사진만 사용했습니다.</p><pre>'+json.dumps(summary,indent=2)+'</pre>'+''.join(f'<h2>{name}</h2><img style="width:80%" src="{name}_candidate.jpg">' for name in counts),encoding='utf-8')
        print(json.dumps({k:v for k,v in summary.items() if k!='views'}),flush=True)
        return candidates


def train_optics(scene,pipe,background,candidates,cfg,output,near,far):
    output=Path(output)
    pc=scene.gaussians
    # Freeze geometry, motion, features, occupancy and topology for this first adapter.
    # This guarantees stable IDs and separates added optical capacity from densification.
    for value in vars(pc).values():
        if isinstance(value,torch.nn.Parameter):
            value.requires_grad_(False)
            value.grad=None
    adapter=CandidateOptics(candidates,duration=int(max(c.timestamp for c in scene.train_cameras[1.0]))+1)
    if not candidates.any():
        (output/'rt_status.json').write_text(json.dumps({'status':'skipped_no_candidates','reason':'Thresholds were not weakened to force glass candidates.'},indent=2))
        (output/'progress.json').write_text(json.dumps({'stage':'candidate_rt','status':'skipped_no_candidates'},indent=2))
        return
    optimizer=torch.optim.Adam(adapter.parameters(),lr=cfg['rt_lr'])
    train=scene.train_cameras[1.0]
    preview=selected_cameras(scene.test_cameras[1.0],cfg['probe_frames'])
    schedule=[]
    log=(output/'rt_loss.jsonl').open('w')
    for step in range(1,cfg['rt_iterations']+1):
        if not schedule:
            schedule=list(train); random.shuffle(schedule)
        camera=schedule.pop()
        target=load_image(camera)
        result=adapter(camera,pc,pipe,background,near,far)
        prediction=result['render']
        photo=.8*l1_loss(prediction,target)+.2*(1-ssim(prediction,target))
        penalty=adapter.regularization()
        loss=photo+cfg['optical_regularization']*penalty
        if not torch.isfinite(loss): raise FloatingPointError('Non-finite RT loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in adapter.parameters()):
            raise FloatingPointError('Non-finite RT gradients')
        optimizer.step()
        if step % 10==0 or step==1:
            entry={'iteration':step,'loss':loss.item(),'photo':photo.item(),'regularization':penalty.item()}
            log.write(json.dumps(entry)+'\n');log.flush()
        if step % 100==0 or step==1:
            print(f'RT {step}/{cfg["rt_iterations"]} loss={loss.item():.6f}',flush=True)
            (output/'progress.json').write_text(json.dumps({'stage':'candidate_rt','iteration':step,
                'total':cfg['rt_iterations'],'loss':loss.item(),'status':'running'},indent=2))
        if step % cfg['preview_every']==0 or step==cfg['rt_iterations']:
            write_preview(output/'previews',f'rt_{step:06d}',preview,
                lambda c:adapter(c,pc,pipe,background,near,far),
                {'stage':'candidate_rt','candidate_points':int(candidates.sum()),'uses_test_for_training':False})
            checkpoint={'iteration':step,'adapter':adapter.state_dict(),'optimizer':optimizer.state_dict(),
                'duration':adapter.duration+1,'baseline_checkpoint':'baseline_final.pth','config':cfg,
                'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state(),'python_rng':random.getstate()}
            temp=output/'rt_latest.tmp';torch.save(checkpoint,temp);temp.replace(output/'rt_latest.pth')
    log.close()
    (output/'rt_status.json').write_text(json.dumps({'status':'complete','iterations':cfg['rt_iterations'],'candidate_points':int(candidates.sum())},indent=2))
    (output/'progress.json').write_text(json.dumps({'stage':'candidate_rt','iteration':cfg['rt_iterations'],'status':'complete'},indent=2))
