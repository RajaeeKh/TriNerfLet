import os
import glob
import tqdm
import math
import imageio
import random
import warnings
import tensorboardX

import numpy as np
import pandas as pd

import time
from datetime import datetime

import cv2
import matplotlib.pyplot as plt
import json

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader

import trimesh
import mcubes
from rich.console import Console
from torch_ema import ExponentialMovingAverage

from packaging import version as pver
import lpips
from torchmetrics.functional import structural_similarity_index_measure
import torchvision

def custom_meshgrid(*args):
    # ref: https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid
    if pver.parse(torch.__version__) < pver.parse('1.10'):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing='ij')


@torch.jit.script
def linear_to_srgb(x):
    return torch.where(x < 0.0031308, 12.92 * x, 1.055 * x ** 0.41666 - 0.055)


@torch.jit.script
def srgb_to_linear(x):
    return torch.where(x < 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def decay_function(iter,opt):
    warmup_steps = max(opt.warmup_steps,0)/opt.accumelate_steps
    if iter < warmup_steps:
        assert opt.warmup_factor < 1
        res = opt.sched_base*opt.warmup_factor + iter*(1-opt.warmup_factor)/(warmup_steps-1)
    else:
        res = opt.sched_base ** (min((iter-warmup_steps) / (opt.iters/opt.accumelate_steps), 1)**opt.sched_exp)
    return res

@torch.cuda.amp.autocast(enabled=False)
def get_rays(poses, intrinsics, H, W, N=-1, error_map=None, patch_size=1):
    ''' get rays
    Args:
        poses: [B, 4, 4], cam2world
        intrinsics: [4]
        H, W, N: int
        error_map: [B, 128 * 128], sample probability based on training error
    Returns:
        rays_o, rays_d: [B, N, 3]
        inds: [B, N]
    '''

    device = poses.device
    B = poses.shape[0]
    fx, fy, cx, cy = intrinsics

    i, j = custom_meshgrid(torch.linspace(0, W-1, W, device=device), torch.linspace(0, H-1, H, device=device)) # float
    i = i.t().reshape([1, H*W]).expand([B, H*W]) + 0.5
    j = j.t().reshape([1, H*W]).expand([B, H*W]) + 0.5

    results = {}

    if N > 0:
        N = min(N, H*W)

        # if use patch-based sampling, ignore error_map
        if patch_size > 1:

            # random sample left-top cores.
            # NOTE: this impl will lead to less sampling on the image corner pixels... but I don't have other ideas.
            num_patch = N // (patch_size ** 2)
            inds_x = torch.randint(0, H - patch_size, size=[num_patch], device=device)
            inds_y = torch.randint(0, W - patch_size, size=[num_patch], device=device)
            inds = torch.stack([inds_x, inds_y], dim=-1) # [np, 2]

            # create meshgrid for each patch
            pi, pj = custom_meshgrid(torch.arange(patch_size, device=device), torch.arange(patch_size, device=device))
            offsets = torch.stack([pi.reshape(-1), pj.reshape(-1)], dim=-1) # [p^2, 2]

            inds = inds.unsqueeze(1) + offsets.unsqueeze(0) # [np, p^2, 2]
            inds = inds.view(-1, 2) # [N, 2]
            inds = inds[:, 0] * W + inds[:, 1] # [N], flatten

            inds = inds.expand([B, N])

        elif error_map is None:
            inds = torch.randint(0, H*W, size=[N], device=device) # may duplicate
            inds = inds.expand([B, N])
        else:

            # weighted sample on a low-reso grid
            inds_coarse = torch.multinomial(error_map.to(device), N, replacement=False) # [B, N], but in [0, 128*128)

            # map to the original resolution with random perturb.
            inds_x, inds_y = inds_coarse // 128, inds_coarse % 128 # `//` will throw a warning in torch 1.10... anyway.
            sx, sy = H / 128, W / 128
            inds_x = (inds_x * sx + torch.rand(B, N, device=device) * sx).long().clamp(max=H - 1)
            inds_y = (inds_y * sy + torch.rand(B, N, device=device) * sy).long().clamp(max=W - 1)
            inds = inds_x * W + inds_y

            results['inds_coarse'] = inds_coarse # need this when updating error_map

        i = torch.gather(i, -1, inds)
        j = torch.gather(j, -1, inds)

        results['inds'] = inds

    else:
        inds = torch.arange(H*W, device=device).expand([B, H*W])
        results['inds'] = inds

    zs = torch.ones_like(i)
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    directions = torch.stack((xs, ys, zs), dim=-1)
    directions = directions / torch.norm(directions, dim=-1, keepdim=True)
    rays_d = directions @ poses[:, :3, :3].transpose(-1, -2) # (B, N, 3)

    rays_o = poses[..., :3, 3] # [B, 3]
    rays_o = rays_o[..., None, :].expand_as(rays_d) # [B, N, 3]

    results['rays_o'] = rays_o
    results['rays_d'] = rays_d

    return results


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = True


def torch_vis_2d(x, renormalize=False):
    # x: [3, H, W] or [1, H, W] or [H, W]
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    
    if isinstance(x, torch.Tensor):
        if len(x.shape) == 3:
            x = x.permute(1,2,0).squeeze()
        x = x.detach().cpu().numpy()
        
    print(f'[torch_vis_2d] {x.shape}, {x.dtype}, {x.min()} ~ {x.max()}')
    
    x = x.astype(np.float32)
    
    # renormalize
    if renormalize:
        x = (x - x.min(axis=0, keepdims=True)) / (x.max(axis=0, keepdims=True) - x.min(axis=0, keepdims=True) + 1e-8)

    plt.imshow(x)
    plt.show()


def extract_fields(bound_min, bound_max, resolution, query_func, S=128):

    X = torch.linspace(bound_min[0], bound_max[0], resolution).split(S)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(S)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(S)

    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    with torch.no_grad():
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = custom_meshgrid(xs, ys, zs)
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1) # [S, 3]
                    val = query_func(pts).reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy() # [S, 1] --> [x, y, z]
                    u[xi * S: xi * S + len(xs), yi * S: yi * S + len(ys), zi * S: zi * S + len(zs)] = val
    return u


def extract_geometry(bound_min, bound_max, resolution, threshold, query_func):
    #print('threshold: {}'.format(threshold))
    u = extract_fields(bound_min, bound_max, resolution, query_func)

    #print(u.shape, u.max(), u.min(), np.percentile(u, 50))
    
    vertices, triangles = mcubes.marching_cubes(u, threshold)

    b_max_np = bound_max.detach().cpu().numpy()
    b_min_np = bound_min.detach().cpu().numpy()

    vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
    return vertices, triangles


def concat_data(all_data):
    keys = all_data[0].keys()
    res = {}
    for key in keys:
        if isinstance(all_data[0][key], torch.Tensor):
            res[key] = torch.cat([all_data[i][key] for i in range(len(all_data))],dim=0)
        else:
            res[key] = [all_data[i][key] for i in range(len(all_data))]
    return res

def shuffle_data(all_data):
    B,N,C = all_data['rays_o'].shape
    perm = torch.randperm(B*N)
    res = {}
    for key,val in all_data.items():
        if isinstance(val, torch.Tensor):
            val = val.view(-1,*val.shape[2:])
            res[key] = val[perm]
    return res

def select_batch(all_data,batch_idx,batch_size,device):
    res = {}
    for key, val in all_data.items():
        res[key] = val[batch_idx*batch_size : (batch_idx + 1)*batch_size].unsqueeze(0).to(device)

    return res

class PSNRMeter:
    def __init__(self):
        self.V = 0
        self.N = 0

    def clear(self):
        self.V = 0
        self.N = 0

    def prepare_inputs(self, *inputs):
        outputs = []
        for i, inp in enumerate(inputs):
            if torch.is_tensor(inp):
                inp = inp.detach().cpu().numpy()
            outputs.append(inp)

        return outputs

    def update(self, preds, truths):
        preds, truths = self.prepare_inputs(preds, truths) # [B, N, 3] or [B, H, W, 3], range[0, 1]
          
        # simplified since max_pixel_value is 1 here.
        psnr = -10 * np.log10(np.mean((preds - truths) ** 2))
        
        self.V += psnr
        self.N += 1

    def measure(self):
        return self.V / self.N

    def write(self, writer, global_step, prefix=""):
        writer.add_scalar(os.path.join(prefix, "PSNR"), self.measure(), global_step)

    def report(self):
        return f'PSNR = {self.measure():.6f}'
    def report2(self):
        res = {'PSNR' : self.measure()}
        return res


class SSIMMeter:
    def __init__(self, device=None):
        self.V = 0
        self.N = 0

        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def clear(self):
        self.V = 0
        self.N = 0

    def prepare_inputs(self, *inputs):
        outputs = []
        for i, inp in enumerate(inputs):
            inp = inp.permute(0, 3, 1, 2).contiguous() # [B, 3, H, W]
            inp = inp.to(self.device)
            outputs.append(inp)
        return outputs

    def update(self, preds, truths):
        preds, truths = self.prepare_inputs(preds, truths) # [B, H, W, 3] --> [B, 3, H, W], range in [0, 1]

        ssim = structural_similarity_index_measure(preds, truths)

        self.V += ssim
        self.N += 1

    def measure(self):
        return self.V / self.N

    def write(self, writer, global_step, prefix=""):
        writer.add_scalar(os.path.join(prefix, "SSIM"), self.measure(), global_step)

    def report(self):
        return f'SSIM = {self.measure():.6f}'

    def report2(self):
        return {'SSIM' : self.measure()}


class LPIPSMeter:
    def __init__(self, net='alex', device=None):
        self.V = 0
        self.N = 0
        self.net = net

        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.fn = lpips.LPIPS(net=net).eval().to(self.device)

    def clear(self):
        self.V = 0
        self.N = 0

    def prepare_inputs(self, *inputs):
        outputs = []
        for i, inp in enumerate(inputs):
            inp = inp.permute(0, 3, 1, 2).contiguous() # [B, 3, H, W]
            inp = inp.to(self.device)
            outputs.append(inp)
        return outputs
    
    def update(self, preds, truths):
        preds, truths = self.prepare_inputs(preds, truths) # [B, H, W, 3] --> [B, 3, H, W], range in [0, 1]
        v = self.fn(truths, preds, normalize=True).item() # normalize=True: [0, 1] to [-1, 1]
        self.V += v
        self.N += 1
    
    def measure(self):
        return self.V / self.N

    def write(self, writer, global_step, prefix=""):
        writer.add_scalar(os.path.join(prefix, f"LPIPS ({self.net})"), self.measure(), global_step)

    def report(self):
        return f'LPIPS ({self.net}) = {self.measure():.6f}'

    def report2(self):
        return {'LPIPS_{}'.format(self.net): self.measure()}

class Trainer(object):
    def __init__(self, 
                 name, # name of this experiment
                 opt, # extra conf
                 model, # network 
                 criterion=None, # loss function, if None, assume inline implementation in train_step
                 optimizer=None, # optimizer
                 ema_decay=None, # if use EMA, set the decay
                 lr_scheduler=None, # scheduler
                 metrics=[], # metrics for evaluation, if None, use val_loss to measure performance, else use the first metric.
                 local_rank=0, # which GPU am I
                 world_size=1, # total num of GPUs
                 device=None, # device to use, usually setting to None is OK. (auto choose device)
                 mute=False, # whether to mute all print
                 fp16=False, # amp optimize level
                 eval_interval=1, # eval once every $ epoch
                 max_keep_ckpt=2, # max num of saved ckpts in disk
                 workspace='workspace', # workspace to save logs & ckpts
                 best_mode='min', # the smaller/larger result, the better
                 use_loss_as_metric=True, # use loss as the first metric
                 report_metric_at_train=False, # also report metrics at training
                 use_checkpoint="latest", # which ckpt to use at init time
                 use_tensorboardX=True, # whether to use tensorboard for logging
                 scheduler_update_every_step=False, # whether to call scheduler.step() after every train step
                 ):
        
        self.name = name
        self.opt = opt
        self.mute = mute
        self.metrics = metrics
        self.local_rank = local_rank
        self.world_size = world_size
        self.workspace = workspace
        self.ema_decay = ema_decay
        self.fp16 = fp16
        self.best_mode = best_mode
        self.use_loss_as_metric = use_loss_as_metric
        self.report_metric_at_train = report_metric_at_train
        self.max_keep_ckpt = max_keep_ckpt
        self.eval_interval = eval_interval
        self.use_checkpoint = use_checkpoint
        self.use_tensorboardX = use_tensorboardX
        self.time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.scheduler_update_every_step = scheduler_update_every_step
        self.device = device if device is not None else torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
        self.console = Console()

        model.to(self.device)
        if self.world_size > 1:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
        self.model = model

        if isinstance(criterion, nn.Module):
            criterion.to(self.device)
        self.criterion = criterion

        # optionally use LPIPS loss for patch-based training
        if self.opt.patch_size > 1:
            import lpips
            self.criterion_lpips = lpips.LPIPS(net='alex').to(self.device)

        if optimizer is None:
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=5e-4) # naive adam
        else:
            self.optimizer = optimizer(self.model)

        if lr_scheduler is None:
            self.lr_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: 1) # fake scheduler
        else:
            self.lr_scheduler = lr_scheduler(self.optimizer)

        self.init_ema()

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.fp16)

        # variable init
        self.epoch = 0
        self.global_step = 0
        self.local_step = 0
        self.stats = {
            "loss": [],
            "valid_loss": [],
            "results": [], # metrics[0], or valid_loss
            "checkpoints": [], # record path of saved ckpt, to automatically remove old ckpt
            "best_result": None,
            }

        # auto fix
        if len(metrics) == 0 or self.use_loss_as_metric:
            self.best_mode = 'min'

        # workspace prepare
        self.log_ptr = None
        if self.workspace is not None:
            os.makedirs(self.workspace, exist_ok=True)        
            self.log_path = os.path.join(workspace, f"log_{self.name}.txt")
            self.log_ptr = open(self.log_path, "a+")

            self.ckpt_path = os.path.join(self.workspace, 'checkpoints')
            self.best_path = f"{self.ckpt_path}/{self.name}.pth"
            os.makedirs(self.ckpt_path, exist_ok=True)
            
        self.log(f'[INFO] Trainer: {self.name} | {self.time_stamp} | {self.device} | {"fp16" if self.fp16 else "fp32"} | {self.workspace}')
        self.log(f'[INFO] #parameters: {sum([p.numel() for p in model.parameters() if p.requires_grad])}')

        if self.workspace is not None:
            if self.use_checkpoint == "scratch":
                self.log("[INFO] Training from scratch ...")
            elif self.use_checkpoint == "latest":
                self.log("[INFO] Loading latest checkpoint ...")
                self.load_checkpoint()
            elif self.use_checkpoint == "latest_model":
                self.log("[INFO] Loading latest checkpoint (model only)...")
                self.load_checkpoint(model_only=True)
            elif self.use_checkpoint == "best":
                if os.path.exists(self.best_path):
                    self.log("[INFO] Loading best checkpoint ...")
                    self.load_checkpoint(self.best_path)
                else:
                    self.log(f"[INFO] {self.best_path} not found, loading latest ...")
                    self.load_checkpoint()
            else: # path to ckpt
                self.log(f"[INFO] Loading {self.use_checkpoint} ...")
                self.load_checkpoint(self.use_checkpoint)
            self.timestamp = str(datetime.now()).replace(' ','_')
            config_f = os.path.join(self.ckpt_path,'cfg_{}_{}.json'.format(self.epoch,self.timestamp))
            with open(config_f,'w') as file:
                json.dump(vars(self.opt),file)

        if self.ema is not None:
            self.log('Trainer: Ema enabled')
            self.ema.to('cpu')

        
        # clip loss prepare
        if opt.rand_pose >= 0: # =0 means only using CLIP loss, >0 means a hybrid mode.
            from nerf.clip_utils import CLIPLoss
            self.clip_loss = CLIPLoss(self.device)
            self.clip_loss.prepare_text([self.opt.clip_text]) # only support one text prompt now...

        self.nerfacc_renderer = None
        if opt.nerfacc_renderer:
            from .nerfacc_utils import NerfAccRenderer
            self.nerfacc_renderer = NerfAccRenderer(opt,device,self.model.nerfacc_estimator)


    def __del__(self):
        if self.log_ptr: 
            self.log_ptr.close()


    def init_ema(self):
        if self.ema_decay is not None:
            self.ema = ExponentialMovingAverage(self.model.parameters(), decay=self.ema_decay)
        else:
            self.ema = None
    def log(self, *args, **kwargs):
        if self.local_rank == 0:
            if not self.mute: 
                #print(*args)
                self.console.print(*args, **kwargs)
            if self.log_ptr: 
                print(*args, file=self.log_ptr)
                self.log_ptr.flush() # write immediately to file

    ### ------------------------------	

    def train_step(self, data):

        rays_o = data['rays_o'] # [B, N, 3]
        rays_d = data['rays_d'] # [B, N, 3]

        aux_dict = {}

        # if there is no gt image, we train with CLIP loss.
        if 'images' not in data:

            B, N = rays_o.shape[:2]
            H, W = data['H'], data['W']

            # currently fix white bg, MUST force all rays!
            outputs = self.model.render(rays_o, rays_d, staged=False, bg_color=None, perturb=True, force_all_rays=True, **vars(self.opt))
            pred_rgb = outputs['image'].reshape(B, H, W, 3).permute(0, 3, 1, 2).contiguous()

            # [debug] uncomment to plot the images used in train_step
            #torch_vis_2d(pred_rgb[0])

            loss = self.clip_loss(pred_rgb)
            
            return pred_rgb, None, loss

        images = data['images'] # [B, N, 3/4]

        B, N, C = images.shape

        if self.opt.color_space == 'linear':
            images[..., :3] = srgb_to_linear(images[..., :3])

        # if C == 3 or self.model.bg_radius > 0:
        if self.model.bg_radius > 0:
            bg_color = torch.zeros_like(images[..., :3]) + self.opt.background_color
        # train with random background color if not using a bg model and has alpha channel.
        else:
            # bg_color = torch.ones(3, device=self.device) # [3], fixed white background
            #bg_color = torch.rand(3, device=self.device) # [3], frame-wise random.
            if self.opt.train_rand_bg:
                # print('hhhh')
                bg_color = torch.rand_like(images[..., :3]) # [N, 3], pixel-wise random.
            else:
                bg_color = torch.zeros_like(images[..., :3]) + self.opt.background_color  # [N, 3], ixed zero background.

        if C == 4:
            gt_rgb = images[..., :3] * images[..., 3:] + bg_color * (1 - images[..., 3:])
        else:
            gt_rgb = images

        if self.nerfacc_renderer is not None:
            # print('bg_color.shape = {}'.format(bg_color.shape))
            outputs = self.nerfacc_renderer.render_train(self.model,rays_o, rays_d,bg_color)
            if outputs['n_rendering_samples'] == 0:
                raise ValueError('empty ray set')
        else:
            outputs = self.model.render(rays_o, rays_d, staged=False, bg_color=bg_color, perturb=True, force_all_rays=False if self.opt.patch_size == 1 else True, **vars(self.opt))
            # outputs = self.model.render(rays_o, rays_d, staged=False, bg_color=bg_color, perturb=True, force_all_rays=True, **vars(self.opt))
    
        pred_rgb = outputs['image']
        pred_depth = outputs['depth']
        pred_alpha = outputs['weights_sum']

        # MSE loss
        loss_per_pixel = self.criterion(pred_rgb, gt_rgb).mean(-1) # [B, N, 3] --> [B, N]
        loss = loss_per_pixel

        aux_dict['mse'] = loss_per_pixel.mean().item()

        # patch-based rendering
        if self.opt.patch_size > 1:
            gt_rgb = gt_rgb.view(-1, self.opt.patch_size, self.opt.patch_size, 3).permute(0, 3, 1, 2).contiguous()
            pred_rgb = pred_rgb.view(-1, self.opt.patch_size, self.opt.patch_size, 3).permute(0, 3, 1, 2).contiguous()

            # torch_vis_2d(gt_rgb[0])
            # torch_vis_2d(pred_rgb[0])

            # LPIPS loss [not useful...]
            loss = loss + 1e-3 * self.criterion_lpips(pred_rgb, gt_rgb)

        # special case for CCNeRF's rank-residual training
        if len(loss.shape) == 3: # [K, B, N]
            loss = loss.mean(0)

        # update error_map
        if self.error_map is not None:
            index = data['index'] # [B]
            inds = data['inds_coarse'] # [B, N]

            # take out, this is an advanced indexing and the copy is unavoidable.
            error_map = self.error_map[index] # [B, H * W]

            # [debug] uncomment to save and visualize error map
            # if self.global_step % 1001 == 0:
            #     tmp = error_map[0].view(128, 128).cpu().numpy()
            #     print(f'[write error map] {tmp.shape} {tmp.min()} ~ {tmp.max()}')
            #     tmp = (tmp - tmp.min()) / (tmp.max() - tmp.min())
            #     cv2.imwrite(os.path.join(self.workspace, f'{self.global_step}.jpg'), (tmp * 255).astype(np.uint8))

            error = loss.detach().to(error_map.device) # [B, N], already in [0, 1]
            
            # ema update
            ema_error = 0.1 * error_map.gather(1, inds) + 0.9 * error
            error_map.scatter_(1, inds, ema_error)

            # put back
            self.error_map[index] = error_map

        loss = loss.mean()
        if self.opt.triplane_wavelet and (self.opt.wavelet_regularization > 0):
            wavelet_features = self.model.encoder.get_wavelet_features()
            if len(wavelet_features) > 0:
                all_elements = sum([val.numel() for val in wavelet_features])
                if self.opt.weighted_regularization:
                    wavelet_reg = []
                    for i,val in enumerate(reversed(wavelet_features)):
                        weight = 1/(4**i)
                        # print('weight = {}, shape = {}'.format(weight,val.shape))
                        wavelet_reg.append(weight*val.abs().mean()*(val.numel()/all_elements))
                    wavelet_reg = sum(wavelet_reg)
                else:
                    wavelet_reg = sum([val.abs().mean()*(val.numel()/all_elements) for val in (wavelet_features)])/len(wavelet_features) #TODO: remove division (is not removed in best_results)
                wavelet_reg_loss = self.opt.wavelet_regularization*wavelet_reg
                loss = loss + wavelet_reg_loss
                aux_dict['wavelet_reg'] = wavelet_reg_loss.item()
                wavelet_features_upscaled = self.model.encoder.get_wavelet_features_upscaled()
                if len(wavelet_features_upscaled) > 0:
                    # all_elements = sum([val.numel() for val in wavelet_features_upscaled])
                    #TODO weighting * (1 / 4**(i+1))
                    wavelet_reg_upscaled = sum([val.abs().mean() * (1 / 4**(i+1))  * (val.numel() / all_elements) for i,val in enumerate(wavelet_features_upscaled)])
                    wavelet_reg_loss_upscaled = self.opt.wavelet_regularization * wavelet_reg_upscaled
                    loss = loss + wavelet_reg_loss_upscaled
                    aux_dict['wavelet_reg_upscaled'] = wavelet_reg_loss_upscaled.item()

        if self.opt.alpha_bce > 0:
            alpha_reg = -1*self.opt.alpha_bce*torch.log(pred_alpha.clamp(0.01,0.99)).mean()
            loss = loss + alpha_reg
            aux_dict['alpha_reg'] = alpha_reg.item()
        if (self.opt.z_variance_reg > 0) and ('z_variance' in outputs):
            z_reg = self.opt.z_variance_reg * outputs['z_variance'].mean()
            loss = loss + z_reg
            aux_dict['z_variance_reg'] = z_reg.item()

        # extra loss
        # pred_weights_sum = outputs['weights_sum'] + 1e-8
        # loss_ws = - 1e-1 * pred_weights_sum * torch.log(pred_weights_sum) # entropy to encourage weights_sum to be 0 or 1.
        # loss = loss + loss_ws.mean()

        return pred_rgb, gt_rgb, loss, aux_dict

    def eval_step(self, data):

        rays_o = data['rays_o'] # [B, N, 3]
        rays_d = data['rays_d'] # [B, N, 3]
        images = data['images'] # [B, H, W, 3/4]
        B, H, W, C = images.shape

        if self.opt.color_space == 'linear':
            images[..., :3] = srgb_to_linear(images[..., :3])

        # eval with fixed background color
        bg_color = self.opt.background_color
        if C == 4:
            gt_rgb = images[..., :3] * images[..., 3:] + bg_color * (1 - images[..., 3:])
        else:
            gt_rgb = images
        
        if self.nerfacc_renderer is not None:
            # bg_color = torch.ones(3,device=gt_rgb.device,dtype=gt_rgb.dtype) * bg_color
            bg_color = torch.zeros_like(rays_o) + bg_color
            # print('bg_color.shape = {}'.format(bg_color.shape))
            outputs = self.nerfacc_renderer.render_eval(self.model,rays_o, rays_d,bg_color)
        else:
            outputs = self.model.render(rays_o, rays_d, staged=True, bg_color=bg_color, perturb=False, **vars(self.opt))

        pred_rgb = outputs['image'].reshape(B, H, W, 3)
        pred_depth = outputs['depth'].reshape(B, H, W)
        pred_alpha = outputs['weights_sum'].reshape(B, H, W)

        loss = self.criterion(pred_rgb, gt_rgb).mean()

        return pred_rgb, pred_depth, pred_alpha, gt_rgb, loss

    # moved out bg_color and perturb for more flexible control...
    def test_step(self, data, perturb=False):

        rays_o = data['rays_o'] # [B, N, 3]
        rays_d = data['rays_d'] # [B, N, 3]
        H, W = data['H'], data['W']

        # if bg_color is not None:
        #     bg_color = bg_color.to(self.device)

        bg_color = self.opt.background_color

        if self.nerfacc_renderer is not None:
            bg_color = torch.ones(3,device=rays_o.device,dtype=rays_o.dtype) * bg_color
            outputs = self.nerfacc_renderer.render_eval(self.model,rays_o, rays_d,bg_color)
        else:
            outputs = self.model.render(rays_o, rays_d, staged=True, bg_color=bg_color, perturb=perturb, **vars(self.opt))

        pred_rgb = outputs['image'].reshape(-1, H, W, 3)
        pred_depth = outputs['depth'].reshape(-1, H, W)

        return pred_rgb, pred_depth


    def save_mesh(self, save_path=None, resolution=256, threshold=10):

        if save_path is None:
            save_path = os.path.join(self.workspace, 'meshes', f'{self.name}_{self.epoch}.ply')

        self.log(f"==> Saving mesh to {save_path}")

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        def query_func(pts):
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    sigma = self.model.density(pts.to(self.device))['sigma']
            return sigma

        vertices, triangles = extract_geometry(self.model.aabb_infer[:3], self.model.aabb_infer[3:], resolution=resolution, threshold=threshold, query_func=query_func)

        mesh = trimesh.Trimesh(vertices, triangles, process=False) # important, process=True leads to seg fault...
        mesh.export(save_path)

        self.log(f"==> Finished saving mesh.")

    ### ------------------------------

    def train(self, train_loader, valid_loader, max_epochs,test_loader = None):
        if self.use_tensorboardX and self.local_rank == 0:
            self.writer = tensorboardX.SummaryWriter(os.path.join(self.workspace, "run", self.name))

        # mark untrained region (i.e., not covered by any camera from the training dataset)
        #TODO: make sure
        if not (self.opt.llff_dataset or self.opt.topia_dataset):
            if self.model.cuda_ray:
                self.model.mark_untrained_grid(train_loader._data.poses, train_loader._data.intrinsics)

        if self.opt.training_evaluate_test and (test_loader is not None):
            valid_loader = test_loader
        if not self.opt.fast_training:
            self.evaluate_one_epoch(valid_loader)


        # get a ref to error_map
        self.error_map = train_loader._data.error_map

        # distributedSampler: must call set_epoch() to shuffle indices across multiple epochs
        # ref: https://pytorch.org/docs/stable/data.html
        if self.world_size > 1:
            train_loader.sampler.set_epoch(self.epoch)

        all_data = []
        for data in train_loader:
            all_data.append(data)
        all_data_train = concat_data(all_data)
        # all_data['inds'] = all_data['inds'].unsqueeze(1).expand(-1,all_data['rays_o'].shape[1])

        training_time = 0

        
        for epoch in range(self.epoch + 1, max_epochs + 1):
            start_time = time.time()
            self.epoch = epoch

            # self.train_one_epoch(train_loader)
            self.train_one_epoch2(all_data_train)
            epoch_time = (time.time() - start_time)
            training_time += epoch_time
            self.log('epoch {} time: {}[s]'.format(epoch,epoch_time))

            if not self.opt.fast_training:
                if (self.workspace is not None) and (self.local_rank == 0) and (self.epoch % self.eval_interval == 0):
                    self.evaluate_one_epoch(valid_loader)
                    self.save_checkpoint(full=True, best=False)



        if self.use_tensorboardX and self.local_rank == 0:
            self.writer.close()

        self.log('training time: {}[s]'.format(training_time))
        self.save_checkpoint(full=True, best=False,remove_old = False, save_backup = True)

    def evaluate(self, loader, name=None):
        self.use_tensorboardX, use_tensorboardX = False, self.use_tensorboardX
        data_res_dict = self.evaluate_one_epoch(loader, name)
        save_f = os.path.join(self.workspace,'results.json')
        with open(save_f,'w') as file:
            json.dump(data_res_dict,file)
        self.use_tensorboardX = use_tensorboardX

    def test(self, loader, save_path=None, name=None, write_video=True):

        if save_path is None:
            save_path = os.path.join(self.workspace, 'results')

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        os.makedirs(save_path, exist_ok=True)
        
        self.log(f"==> Start Test, save results to {save_path}")

        if self.ema is not None:
            self.ema.to(self.device)
            self.ema.store()
            self.ema.copy_to()

        pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        self.model.eval()

        if write_video:
            all_preds = []
            all_preds_depth = []
        render_time = 0
        with torch.no_grad():

            for i, data in enumerate(loader):
                
                tt = time.time()
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    preds, preds_depth = self.test_step(data)

                if self.opt.color_space == 'linear':
                    preds = linear_to_srgb(preds)

                pred = preds[0].detach().cpu().numpy()
                pred = (pred * 255).astype(np.uint8)

                pred_depth = preds_depth[0].detach().cpu().numpy()
                pred_depth = (pred_depth * 255).astype(np.uint8)

                render_time += (time.time() - tt)

                if write_video:
                    all_preds.append(pred)
                    all_preds_depth.append(pred_depth)
                else:
                    cv2.imwrite(os.path.join(save_path, f'{name}_{i:04d}_rgb.png'), cv2.cvtColor(pred, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(os.path.join(save_path, f'{name}_{i:04d}_depth.png'), pred_depth)

                pbar.update(loader.batch_size)
        
        if write_video:
            try:
                all_preds = np.stack(all_preds, axis=0)
                all_preds_depth = np.stack(all_preds_depth, axis=0)
                imageio.mimwrite(os.path.join(save_path, f'{name}_rgb.mp4'), all_preds, fps=25, quality=8, macro_block_size=1)
                imageio.mimwrite(os.path.join(save_path, f'{name}_depth.mp4'), all_preds_depth, fps=25, quality=8, macro_block_size=1)
            except:
                self.log('write video failed')

        self.log('total render time: {}[s]'.format(render_time))
        self.log('render fps: {}'.format(len(loader)/render_time))
        self.save_triplane(all=True,save_wavelet=self.opt.save_wavelet)
        if self.ema is not None:
            self.ema.restore()
            self.ema.to('cpu')
        self.log(f"==> Finished Test.")
    
    # [GUI] just train for 16 steps, without any other overhead that may slow down rendering.
    def train_gui(self, train_loader, step=16):

        self.model.train()

        total_loss = torch.tensor([0], dtype=torch.float32, device=self.device)
        
        loader = iter(train_loader)

        # mark untrained grid
        if self.global_step == 0:
            self.model.mark_untrained_grid(train_loader._data.poses, train_loader._data.intrinsics)

        for _ in range(step):
            
            # mimic an infinite loop dataloader (in case the total dataset is smaller than step)
            try:
                data = next(loader)
            except StopIteration:
                loader = iter(train_loader)
                data = next(loader)

            # update grid every 16 steps
            if self.model.cuda_ray and self.global_step % self.opt.update_extra_interval == 0:
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    self.model.update_extra_state()
            
            self.global_step += 1

            self.optimizer.zero_grad()
            if self.opt.triplane_wavelet:
                self.model.encoder.reset_cahce()
                self.model.encoder.get_planes()

            with torch.cuda.amp.autocast(enabled=self.fp16):
                preds, truths, loss = self.train_step(data)

                if self.opt.triplane_wavelet:
                    self.model.encoder.reset_cahce()
         
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            if self.scheduler_update_every_step:
                self.lr_scheduler.step()

            total_loss += loss.detach()

        if self.ema is not None:
            self.ema.update()

        average_loss = total_loss.item() / step

        if not self.scheduler_update_every_step:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(average_loss)
            else:
                self.lr_scheduler.step()

        outputs = {
            'loss': average_loss,
            'lr': self.optimizer.param_groups[0]['lr'],
        }
        
        return outputs

    
    # [GUI] test on a single image
    def test_gui(self, pose, intrinsics, W, H, bg_color=None, spp=1, downscale=1):
        
        # render resolution (may need downscale to for better frame rate)
        rH = int(H * downscale)
        rW = int(W * downscale)
        intrinsics = intrinsics * downscale

        pose = torch.from_numpy(pose).unsqueeze(0).to(self.device)

        rays = get_rays(pose, intrinsics, rH, rW, -1)

        data = {
            'rays_o': rays['rays_o'],
            'rays_d': rays['rays_d'],
            'H': rH,
            'W': rW,
        }
        
        self.model.eval()

        if self.ema is not None:
            self.ema.store()
            self.ema.copy_to()

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=self.fp16):
                # here spp is used as perturb random seed! (but not perturb the first sample)
                preds, preds_depth = self.test_step(data, bg_color=bg_color, perturb=False if spp == 1 else spp)

        if self.ema is not None:
            self.ema.restore()

        # interpolation to the original resolution
        if downscale != 1:
            # TODO: have to permute twice with torch...
            preds = F.interpolate(preds.permute(0, 3, 1, 2), size=(H, W), mode='nearest').permute(0, 2, 3, 1).contiguous()
            preds_depth = F.interpolate(preds_depth.unsqueeze(1), size=(H, W), mode='nearest').squeeze(1)

        if self.opt.color_space == 'linear':
            preds = linear_to_srgb(preds)

        pred = preds[0].detach().cpu().numpy()
        pred_depth = preds_depth[0].detach().cpu().numpy()

        outputs = {
            'image': pred,
            'depth': pred_depth,
        }

        return outputs

    def train_one_epoch(self, loader):
        raise ValueError('should not get here')
        self.log(f"==> Start Training Epoch {self.epoch}, lr={self.optimizer.param_groups[0]['lr']:.6f} ...")

        total_loss = 0
        if self.local_rank == 0 and self.report_metric_at_train:
            for metric in self.metrics:
                metric.clear()

        self.model.train()

        # distributedSampler: must call set_epoch() to shuffle indices across multiple epochs
        # ref: https://pytorch.org/docs/stable/data.html
        if self.world_size > 1:
            loader.sampler.set_epoch(self.epoch)
        
        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        self.local_step = 0

        for data in loader:
            
            # update grid every 16 steps
            if self.model.cuda_ray and self.global_step % self.opt.update_extra_interval == 0:
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    self.model.update_extra_state()
                    
            self.local_step += 1
            self.global_step += 1
            if self.opt.triplane_wavelet:
                self.model.encoder.reset_cahce()
            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.fp16,dtype=torch.float16):
                preds, truths, loss, reg_dict = self.train_step(data)
                if self.opt.triplane_wavelet:
                    self.model.encoder.reset_cahce()
                # TODO: moved
                self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.scheduler_update_every_step:
                self.lr_scheduler.step()

            loss_val = loss.item()
            total_loss += loss_val

            if self.local_rank == 0:
                if self.report_metric_at_train:
                    for metric in self.metrics:
                        metric.update(preds, truths)
                        
                if self.use_tensorboardX:
                    self.writer.add_scalar("train/loss", loss_val, self.global_step)
                    self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]['lr'], self.global_step)
                    if len(reg_dict) > 0:
                        self.writer.add_scalars('regularization',reg_dict,self.global_step)

                if self.scheduler_update_every_step:
                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f}), lr={self.optimizer.param_groups[0]['lr']:.6f}")
                else:
                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f})")
                pbar.update(loader.batch_size)

        if self.ema is not None:
            self.ema.update()

        average_loss = total_loss / self.local_step
        self.stats["loss"].append(average_loss)

        if self.local_rank == 0:
            pbar.close()
            if self.report_metric_at_train:
                for metric in self.metrics:
                    self.log(metric.report(), style="red")
                    if self.use_tensorboardX:
                        metric.write(self.writer, self.epoch, prefix="train")
                    metric.clear()

        if not self.scheduler_update_every_step:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(average_loss)
            else:
                self.lr_scheduler.step()

        self.log(f"==> Finished Epoch {self.epoch}.")


    def clear_grad(self):
        if self.opt.min_wavelet_resolution_to_learn > 0:
            with torch.no_grad():
                wavelet_grad = [val.grad for val in self.model.encoder.parameters()]
                for val in self.model.parameters():
                    val.grad = None
                for idx,val in enumerate(self.model.encoder.parameters()):
                    if val.shape[-1] > self.opt.min_wavelet_resolution_to_learn:
                        # print(val.shape)
                        val.grad = wavelet_grad[idx]

    def train_one_epoch2(self, all_data):
        self.log(f"==> Start Training Epoch {self.epoch}, lr={self.optimizer.param_groups[0]['lr']:.6f} ...")

        total_loss = 0
        if self.local_rank == 0 and self.report_metric_at_train:
            for metric in self.metrics:
                metric.clear()

        self.model.train()
        self.local_step = 0
        all_data = shuffle_data(all_data)
        batch_size = self.opt.num_rays
        steps_per_batch = math.ceil(all_data['rays_o'].shape[0]/batch_size)

        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=all_data['rays_o'].shape[0],
                             bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        for batch_idx in range(steps_per_batch):
            data = select_batch(all_data,batch_idx,batch_size,self.device)
            # update grid every 16 steps

            if self.opt.triplane_wavelet:
                self.model.encoder.reset_cahce()
                self.model.encoder.get_planes()

            if self.nerfacc_renderer is not None:
                self.nerfacc_renderer.update(self.global_step,self.model)
            elif self.model.cuda_ray and self.global_step % self.opt.update_extra_interval == 0:
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    self.model.update_extra_state()

            self.local_step += 1
            self.global_step += 1

            if (max(0,batch_idx - 1)%self.opt.accumelate_steps) == 0:
                self.optimizer.zero_grad()





            with torch.cuda.amp.autocast(enabled=self.fp16, dtype=torch.float16):
                # print('here************')
                preds, truths, loss, aux_dict = self.train_step(data)
                if self.opt.triplane_wavelet:
                    self.model.encoder.reset_cahce()
                # TODO: original (and recommended) implementation with was outside autocas, but triplane wavelet won't work
                loss = loss / self.opt.accumelate_steps
                if (batch_idx % self.opt.accumelate_steps) == 0:
                    self.scaler.scale(loss).backward()

            self.clear_grad()
            if (batch_idx % self.opt.accumelate_steps) == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                if self.scheduler_update_every_step:
                    self.lr_scheduler.step()

            loss_val = loss.item()
            total_loss += loss_val

            if self.local_rank == 0:
                if self.report_metric_at_train:
                    for metric in self.metrics:
                        metric.update(preds, truths)

                if self.use_tensorboardX:
                    self.writer.add_scalar("train/loss", loss_val, self.global_step)
                    self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]['lr'], self.global_step)
                    for key,val in aux_dict.items():
                        self.writer.add_scalar("loss_components/{}".format(key), val, self.global_step)
                    # if len(aux_dict) > 0:
                    #     self.writer.add_scalars('regularization', reg_dict, self.global_step)
                    lbound_scale = self.model.encoder.get_lbound_scale()
                    if lbound_scale is not None:
                        lbound_scale_lst = lbound_scale.view(-1).tolist()
                        lbound_scale_dict = dict([(str(i),lbound_scale_lst[i]) for i in range(len(lbound_scale_lst))])
                        self.writer.add_scalars('lbound_scale', lbound_scale_dict, self.global_step)

                if self.scheduler_update_every_step:
                    pbar.set_description(
                        f"loss={loss_val:.4f} ({total_loss / self.local_step:.4f}), lr={self.optimizer.param_groups[0]['lr']:.6f}")
                else:
                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss / self.local_step:.4f})")
                pbar.update(batch_size)


        if self.ema is not None:
            self.ema.to(self.device)
            self.ema.update()
            self.ema.to('cpu')


        average_loss = total_loss / self.local_step
        self.stats["loss"].append(average_loss)

        if self.local_rank == 0:
            pbar.close()
            if self.report_metric_at_train:
                for metric in self.metrics:
                    self.log(metric.report(), style="red")
                    if self.use_tensorboardX:
                        metric.write(self.writer, self.epoch, prefix="train")
                    metric.clear()

        if not self.scheduler_update_every_step:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(average_loss)
            else:
                self.lr_scheduler.step()

        self.log(f"==> Finished Epoch {self.epoch}.")
    def evaluate_one_epoch(self, loader, name=None):
        self.log(f"++> Evaluate at epoch {self.epoch} ...")

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        total_loss = 0
        if self.local_rank == 0:
            for metric in self.metrics:
                metric.clear()

        self.model.eval()

        if self.ema is not None:
            self.ema.to(self.device)
            self.ema.store()
            self.ema.copy_to()

        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        mse_lst = []
        with torch.no_grad():
            self.local_step = 0

            for data in loader:    
                self.local_step += 1

                with torch.cuda.amp.autocast(enabled=self.fp16):
                    preds, preds_depth, preds_alpha, truths, loss = self.eval_step(data)
                    # print(preds.shape)
                    preds = preds.clamp(0,1)
                    truths = truths.clamp(0, 1)

                    if self.opt.color_space == 'linear':
                        preds = linear_to_srgb(preds)
                        truths = linear_to_srgb(truths)

                    loss = F.mse_loss(preds,truths)

                # all_gather/reduce the statistics (NCCL only support all_*)
                if self.world_size > 1:
                    dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                    loss = loss / self.world_size
                    
                    preds_list = [torch.zeros_like(preds).to(self.device) for _ in range(self.world_size)] # [[B, ...], [B, ...], ...]
                    dist.all_gather(preds_list, preds)
                    preds = torch.cat(preds_list, dim=0)

                    preds_depth_list = [torch.zeros_like(preds_depth).to(self.device) for _ in range(self.world_size)] # [[B, ...], [B, ...], ...]
                    dist.all_gather(preds_depth_list, preds_depth)
                    preds_depth = torch.cat(preds_depth_list, dim=0)

                    preds_alpha_list = [torch.zeros_like(preds_alpha).to(self.device) for _ in
                                        range(self.world_size)]  # [[B, ...], [B, ...], ...]
                    dist.all_gather(preds_alpha_list, preds_alpha)
                    preds_alpha = torch.cat(preds_alpha_list, dim=0)

                    truths_list = [torch.zeros_like(truths).to(self.device) for _ in range(self.world_size)] # [[B, ...], [B, ...], ...]
                    dist.all_gather(truths_list, truths)
                    truths = torch.cat(truths_list, dim=0)
                
                loss_val = loss.item()
                total_loss += loss_val

                mse_lst.append(loss_val)

                # only rank = 0 will perform evaluation.
                if self.local_rank == 0:

                    for metric in self.metrics:
                        metric.update(preds, truths)

                    # save image
                    save_path = os.path.join(self.workspace, 'validation', f'{name}_{self.local_step:04d}_rgb.png')
                    save_path_depth = os.path.join(self.workspace, 'validation', f'{name}_{self.local_step:04d}_depth.png')
                    save_path_alpha = os.path.join(self.workspace, 'validation',
                                                   f'{name}_{self.local_step:04d}_alpha.png')

                    #self.log(f"==> Saving validation image to {save_path}")
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)

                    # if self.opt.color_space == 'linear':
                    #     preds = linear_to_srgb(preds)

                    pred = preds[0].detach().cpu().numpy()
                    pred = (pred * 255).astype(np.uint8)

                    pred_depth = preds_depth[0].detach().cpu().numpy()
                    pred_depth = (pred_depth * 255).astype(np.uint8)

                    pred_alpha = preds_alpha[0].detach().cpu().numpy()
                    pred_alpha = (pred_alpha * 255).astype(np.uint8)
                    
                    cv2.imwrite(save_path, cv2.cvtColor(pred, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(save_path_depth, pred_depth)
                    cv2.imwrite(save_path_alpha, pred_alpha)

                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f})")
                    pbar.update(loader.batch_size)


        average_loss = total_loss / self.local_step
        self.stats["valid_loss"].append(average_loss)
        self.log('Average MSE: {}'.format(average_loss))
        self.log('MSE lst: {}'.format(mse_lst))
        psnr_lst = [-10.*np.log10(val) for val in mse_lst]
        self.log('psnr lst: {}'.format(psnr_lst))
        self.log('average psnr = {}'.format(sum(psnr_lst)/len(psnr_lst)))
        # psnr = -10.*np.log10(average_loss)
        # self.log('Average PSNR-1: {}'.format(psnr))


        if self.opt.triplane_wavelet:
            wavelet_features = self.model.encoder.get_wavelet_features()
            for wavelet in wavelet_features:
                self.log('shape: {}, abs mean: {}'.format(wavelet.shape, wavelet.abs().mean()))
            wavelet_features_upscaled = self.model.encoder.get_wavelet_features_upscaled()
            for wavelet in wavelet_features_upscaled:
                self.log('shape: {}, abs mean: {}'.format(wavelet.shape, wavelet.abs().mean()))
            lbound_scale = self.model.encoder.get_lbound_scale()
            if lbound_scale is not None:
                print('************* lbound_scale = {} **************'.format(lbound_scale))

        save_path = os.path.join(self.workspace, 'tmp_results')
        os.makedirs(save_path, exist_ok=True)
        save_f = os.path.join(save_path, 'results_{}.json'.format(self.epoch))

        data_res_dict = None
        if self.local_rank == 0:
            pbar.close()
            if not self.use_loss_as_metric and len(self.metrics) > 0:
                result = self.metrics[0].measure()
                self.stats["results"].append(result if self.best_mode == 'min' else - result) # if max mode, use -result
            else:
                self.stats["results"].append(average_loss) # if no metric, choose best by min loss

            with open(save_f, 'w') as file:
                data_res_dict = {'MSE': average_loss}
                # data_dcit['official metrics: '] = [metric.report() for metric in self.metrics]
                for metric in self.metrics:
                    metric_data = metric.report2()
                    for tmp_key,tmp_val in metric_data.items():
                        data_res_dict[tmp_key] = tmp_val
                json.dump(data_res_dict, file)

            for metric in self.metrics:
                self.log(metric.report(), style="blue")
                if self.use_tensorboardX:
                    metric.write(self.writer, self.epoch, prefix="evaluate")
                metric.clear()


        if self.ema is not None:
            self.ema.restore()
            self.ema.to('cpu')

        self.save_triplane()
        self.log(f"++> Evaluate epoch {self.epoch} Finished.")
        return data_res_dict

    def save_checkpoint(self, name=None, full=False, best=False, remove_old=True, save_backup = False):

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        state = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'stats': self.stats,
        }

        if self.model.cuda_ray:
            state['mean_count'] = self.model.mean_count
            state['mean_density'] = self.model.mean_density

        if full:
            state['optimizer'] = self.optimizer.state_dict()
            state['lr_scheduler'] = self.lr_scheduler.state_dict()
            state['scaler'] = self.scaler.state_dict()
            if self.ema is not None:
                state['ema'] = self.ema.state_dict()
        

        if not best:

            state['model'] = self.model.state_dict()

            file_path = f"{self.ckpt_path}/{name}.pth"

            if remove_old:
                self.stats["checkpoints"].append(file_path)

                if len(self.stats["checkpoints"]) > self.max_keep_ckpt:
                    old_ckpt = self.stats["checkpoints"].pop(0)
                    if os.path.exists(old_ckpt):
                        os.remove(old_ckpt)

            self.log('checkpoint1 keys:')
            self.log(state.keys())
            torch.save(state, file_path)
            if save_backup:
                file_path2 = f"{self.ckpt_path}/backup_{self.time_stamp}_{name}.pth"
                torch.save(state, file_path2)

        else:    
            if len(self.stats["results"]) > 0:
                if self.stats["best_result"] is None or self.stats["results"][-1] < self.stats["best_result"]:
                    self.log(f"[INFO] New best result: {self.stats['best_result']} --> {self.stats['results'][-1]}")
                    self.stats["best_result"] = self.stats["results"][-1]

                    # save ema results 
                    if self.ema is not None:
                        self.ema.to(self.device)
                        self.ema.store()
                        self.ema.copy_to()

                    state['model'] = self.model.state_dict()

                    # we don't consider continued training from the best ckpt, so we discard the unneeded density_grid to save some storage (especially important for dnerf)
                    if 'density_grid' in state['model']:
                        del state['model']['density_grid']

                    if self.ema is not None:
                        self.ema.restore()
                        self.ema.to('cpu')

                    self.log('checkpoint2 keys:')
                    self.log(state.keys())
                    torch.save(state, self.best_path)
                    if save_backup:
                        file_path2 = f"{self.ckpt_path}/backup_{name}.pth"
                        torch.save(state, file_path2)
            else:
                self.log(f"[WARN] no evaluated results found, skip saving best checkpoint.")
            
    def load_checkpoint(self, checkpoint=None, model_only=False):
        if checkpoint is None:
            checkpoint_list = sorted(glob.glob(f'{self.ckpt_path}/{self.name}_ep*.pth'))
            if checkpoint_list:
                checkpoint = checkpoint_list[-1]
                self.log(f"[INFO] Latest checkpoint is {checkpoint}")
            else:
                self.log("[WARN] No checkpoint found, model randomly initialized.")
                return

        checkpoint_dict = torch.load(checkpoint, map_location=self.device)
        
        if 'model' not in checkpoint_dict:
            self.model.load_state_dict(checkpoint_dict)
            self.log("[INFO] loaded model.")
            return

        missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint_dict['model'], strict=False)
        self.log("[INFO] loaded model.")
        if len(missing_keys) > 0:
            self.log(f"[WARN] missing keys: {missing_keys}")
        if len(unexpected_keys) > 0:
            self.log(f"[WARN] unexpected keys: {unexpected_keys}")   



        if self.model.cuda_ray:
            if 'mean_count' in checkpoint_dict:
                self.model.mean_count = checkpoint_dict['mean_count']
            if 'mean_density' in checkpoint_dict:
                self.model.mean_density = checkpoint_dict['mean_density']

        self.init_ema()
        if model_only:
            return

        if self.ema is not None and 'ema' in checkpoint_dict:
            self.ema.load_state_dict(checkpoint_dict['ema'])
            self.log("[INFO] loaded ema.")

        self.stats = checkpoint_dict['stats']
        self.epoch = checkpoint_dict['epoch']
        self.global_step = checkpoint_dict['global_step']
        self.log(f"[INFO] load at epoch {self.epoch}, global step {self.global_step}")
        
        if self.optimizer and 'optimizer' in checkpoint_dict:
            try:
                self.optimizer.load_state_dict(checkpoint_dict['optimizer'])
                self.log("[INFO] loaded optimizer.")
            except Exception as e:
                self.log("[WARN] Failed to load optimizer.")
                self.log(str(e))
        
        if self.lr_scheduler and 'lr_scheduler' in checkpoint_dict:
            try:
                self.lr_scheduler.load_state_dict(checkpoint_dict['lr_scheduler'])
                self.log("[INFO] loaded scheduler.")
            except Exception as e:
                self.log("[WARN] Failed to load scheduler.")
                self.log(str(e))
        
        if self.scaler and 'scaler' in checkpoint_dict:
            try:
                self.scaler.load_state_dict(checkpoint_dict['scaler'])
                self.log("[INFO] loaded scaler.")
            except Exception as e:
                self.log("[WARN] Failed to load scaler.")
                self.log(str(e))


    def save_tensor(self,planes,save_path,all,prefix = 'plane',dataset = None):
        # f_save = os.path.join(self.workspace, 'planes.pt')
        # torch.save({'planes':planes},f_save)

        cameras_loc = []
        if dataset is not None:
            for batch in dataset:
                current_loc = batch['rays_o'][0,0].clone().numpy()
                cameras_loc.append(current_loc)


        for axis in range(planes.shape[0]):
            ch_list = list(range(planes.shape[1]))
            if not all:
                ch_list = [random.choice(ch_list)]
            for ch_idx in ch_list:
                image_f = os.path.join(save_path, '{}_{}_{}_{}.png'.format(prefix,self.epoch, axis, ch_idx))
                current_plane = (planes[axis, ch_idx] * 255).round().numpy().astype(np.uint8)
                if dataset is not None:
                    current_plane = np.stack([current_plane]*3)
                    for img_loc in cameras_loc:
                       if axis==0:
                           pass
                       elif axis==1:
                           pass
                       elif axis==2:
                           pass
                cv2.imwrite(image_f, current_plane)
                # print('displaying axis={}, {}/{} plane'.format(axis,ch_idx,planes.shape[1]))
                # current_plane = planes[axis,ch_idx]
                # plt.imshow(current_plane)
                # plt.show()


    @staticmethod
    def get_wavelet_img(planes_features,planes_features_wavelet_coefs):
        planes_features = planes_features.cpu().clone().detach().float()
        tmp = planes_features.view(planes_features.shape[0], planes_features.shape[1], -1)
        a = tmp.min(dim=-1, keepdim=True).values.unsqueeze(-1)
        b = tmp.max(dim=-1, keepdim=True).values.unsqueeze(-1)
        planes_features = (planes_features - a) / (b - a)
        planes_features = torchvision.transforms.functional.adjust_contrast(planes_features.unsqueeze(2), 2).squeeze(2)
        ll = planes_features


        for i,wavelet_h in enumerate(planes_features_wavelet_coefs):
            wavelet_h = wavelet_h.cpu().clone().detach().float()
            wavelet_h = wavelet_h.abs()

            a = wavelet_h.view(wavelet_h.shape[0], wavelet_h.shape[1],wavelet_h.shape[2], -1).max(dim=-1,
                                                                                  keepdim=True).values.unsqueeze(-1)
            wavelet_h = wavelet_h / a
            # wavelet_h = wavelet_h.clamp(0,1)

            lh,hl,hh = wavelet_h[:,:,0],wavelet_h[:,:,1],wavelet_h[:,:,2]
            l_ch = torch.cat([ll,lh],dim=3)
            h_ch = torch.cat([hl,hh],dim=3)
            ll = torch.cat([l_ch,h_ch],dim=2)


        return ll




    def save_triplane(self,all=False,save_wavelet = False,dataset = None):
        # import matplotlib.pyplot as plt
        save_path = os.path.join(self.workspace, 'planes')
        os.makedirs(save_path, exist_ok=True)

        if save_wavelet:
            save_path_wavelet_features = os.path.join(save_path, 'wavelet_features')
            os.makedirs(save_path_wavelet_features, exist_ok=True)

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=self.fp16):
                planes = self.model.encoder.get_planes()
                planes_features = self.model.encoder.planes_features
                # planes_features_wavelet_coefs = self.model.encoder.planes_features_wavelet_coefs
                planes_features_wavelet_coefs = self.model.encoder.get_wavelet_features()

            if self.model.encoder.upscale_enabled:
                planes,planes_upscaled_lst = planes[0],planes[1:]
            planes = planes.cpu().clone().detach().float()
            self.log('planes.shape: {}'.format(planes.shape))

            tmp = planes.view(planes.shape[0],planes.shape[1],-1)
            a = tmp.min(dim=-1,keepdim=True).values.unsqueeze(-1)
            b = tmp.max(dim=-1, keepdim=True).values.unsqueeze(-1)
            planes = (planes - a) / (b - a)

            planes = torchvision.transforms.functional.adjust_contrast(planes.unsqueeze(2),2).squeeze(2)

            self.save_tensor(planes, save_path,all)

            if self.model.encoder.upscale_enabled:
                for idx,planes_upscaled in enumerate(planes_upscaled_lst):
                    planes_upscaled = planes_upscaled.cpu().clone().detach().float()
                    self.log('planes_upscaled_{}.shape: {}'.format(idx,planes_upscaled.shape))

                    tmp = planes_upscaled.view(planes_upscaled.shape[0], planes_upscaled.shape[1], -1)
                    a = tmp.min(dim=-1, keepdim=True).values.unsqueeze(-1)
                    b = tmp.max(dim=-1, keepdim=True).values.unsqueeze(-1)
                    planes_upscaled = (planes_upscaled - a) / (b - a)
                    planes_upscaled = torchvision.transforms.functional.adjust_contrast(planes_upscaled.unsqueeze(2), 2).squeeze(2)
                    self.save_tensor(planes_upscaled, save_path, all,prefix='plane_upscaled_{}'.format(idx))

            if save_wavelet:
                wavelet_planes = self.get_wavelet_img(planes_features,planes_features_wavelet_coefs)
                self.save_tensor(wavelet_planes, save_path_wavelet_features, all,
                                 'wavelet_features')

                with torch.cuda.amp.autocast(enabled=self.fp16):
                    self.model.encoder.reset_cahce()
                    planes_lst = self.model.encoder.get_planes(get_all_resolutions = True)
                    for idx,planes in enumerate(planes_lst):
                        planes = planes.cpu().clone().detach().float()
                        self.log('planes.shape: {}'.format(planes.shape))

                        tmp = planes.view(planes.shape[0], planes.shape[1], -1)
                        a = tmp.min(dim=-1, keepdim=True).values.unsqueeze(-1)
                        b = tmp.max(dim=-1, keepdim=True).values.unsqueeze(-1)
                        planes = (planes - a) / (b - a)
                        planes = torchvision.transforms.functional.adjust_contrast(planes.unsqueeze(2), 2).squeeze(2)
                        save_path_level = os.path.join(save_path, 'levels_{}'.format(idx))
                        os.makedirs(save_path_level, exist_ok=True)
                        self.save_tensor(planes, save_path_level, all)
