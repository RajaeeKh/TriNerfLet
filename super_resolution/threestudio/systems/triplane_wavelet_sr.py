import math
from dataclasses import dataclass, field
import os

import imageio
import numpy as np
import torch
import torchvision.transforms as TF
import torch.nn.functional as F
import threestudio
from threestudio.systems.base import BaseLift3DSystem
from threestudio.utils.ops import ShapeLoss, binary_cross_entropy, dot
from threestudio.utils.typing import *

from taming.modules.losses.vqperceptual import LPIPS
from pathlib import Path
import random
import cv2


class SSIM():
    def __init__(self, data_range=(0, 1), kernel_size=(11, 11), sigma=(1.5, 1.5), k1=0.01, k2=0.03, gaussian=True):
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.gaussian = gaussian

        if any(x % 2 == 0 or x <= 0 for x in self.kernel_size):
            raise ValueError(f"Expected kernel_size to have odd positive number. Got {kernel_size}.")
        if any(y <= 0 for y in self.sigma):
            raise ValueError(f"Expected sigma to have positive number. Got {sigma}.")

        data_scale = data_range[1] - data_range[0]
        self.c1 = (k1 * data_scale) ** 2
        self.c2 = (k2 * data_scale) ** 2
        self.pad_h = (self.kernel_size[0] - 1) // 2
        self.pad_w = (self.kernel_size[1] - 1) // 2
        self._kernel = self._gaussian_or_uniform_kernel(kernel_size=self.kernel_size, sigma=self.sigma)

    def _uniform(self, kernel_size):
        max, min = 2.5, -2.5
        ksize_half = (kernel_size - 1) * 0.5
        kernel = torch.linspace(-ksize_half, ksize_half, steps=kernel_size)
        for i, j in enumerate(kernel):
            if min <= j <= max:
                kernel[i] = 1 / (max - min)
            else:
                kernel[i] = 0

        return kernel.unsqueeze(dim=0)  # (1, kernel_size)

    def _gaussian(self, kernel_size, sigma):
        ksize_half = (kernel_size - 1) * 0.5
        kernel = torch.linspace(-ksize_half, ksize_half, steps=kernel_size)
        gauss = torch.exp(-0.5 * (kernel / sigma).pow(2))
        return (gauss / gauss.sum()).unsqueeze(dim=0)  # (1, kernel_size)

    def _gaussian_or_uniform_kernel(self, kernel_size, sigma):
        if self.gaussian:
            kernel_x = self._gaussian(kernel_size[0], sigma[0])
            kernel_y = self._gaussian(kernel_size[1], sigma[1])
        else:
            kernel_x = self._uniform(kernel_size[0])
            kernel_y = self._uniform(kernel_size[1])

        return torch.matmul(kernel_x.t(), kernel_y)  # (kernel_size, 1) * (1, kernel_size)

    def __call__(self, output, target, reduction='mean'):
        if output.dtype != target.dtype:
            raise TypeError(
                f"Expected output and target to have the same data type. Got output: {output.dtype} and y: {target.dtype}."
            )

        if output.shape != target.shape:
            raise ValueError(
                f"Expected output and target to have the same shape. Got output: {output.shape} and y: {target.shape}."
            )

        if len(output.shape) != 4 or len(target.shape) != 4:
            raise ValueError(
                f"Expected output and target to have BxCxHxW shape. Got output: {output.shape} and y: {target.shape}."
            )

        assert reduction in ['mean', 'sum', 'none']

        channel = output.size(1)
        if len(self._kernel.shape) < 4:
            self._kernel = self._kernel.expand(channel, 1, -1, -1)

        output = F.pad(output, [self.pad_w, self.pad_w, self.pad_h, self.pad_h], mode="reflect")
        target = F.pad(target, [self.pad_w, self.pad_w, self.pad_h, self.pad_h], mode="reflect")

        input_list = torch.cat([output, target, output * output, target * target, output * target])
        outputs = F.conv2d(input_list, self._kernel, groups=channel)

        output_list = [outputs[x * output.size(0): (x + 1) * output.size(0)] for x in range(len(outputs))]

        mu_pred_sq = output_list[0].pow(2)
        mu_target_sq = output_list[1].pow(2)
        mu_pred_target = output_list[0] * output_list[1]

        sigma_pred_sq = output_list[2] - mu_pred_sq
        sigma_target_sq = output_list[3] - mu_target_sq
        sigma_pred_target = output_list[4] - mu_pred_target

        a1 = 2 * mu_pred_target + self.c1
        a2 = 2 * sigma_pred_target + self.c2
        b1 = mu_pred_sq + mu_target_sq + self.c1
        b2 = sigma_pred_sq + sigma_target_sq + self.c2

        ssim_idx = (a1 * a2) / (b1 * b2)
        _ssim = torch.mean(ssim_idx, (1, 2, 3))

        if reduction == 'none':
            return _ssim
        return _ssim.mean() if reduction == 'mean' else _ssim.sum()

def to8b(img):
    img = (img.clamp(0,1)*255).round().to(torch.uint8)
    return img


def decay_function(iter,warmup_steps,warmup_factor,sched_base,sched_exp,iters):
    warmup_steps = max(warmup_steps,0)
    if iter < warmup_steps:
        assert warmup_factor < 1
        res = sched_base*warmup_factor + iter*(1-warmup_factor)/(warmup_steps-1)
    else:
        res = sched_base ** (min((iter-warmup_steps) / (iters), 1)**sched_exp)
    return res


@threestudio.register("triplane-wavelet-sr-system")
class TriplaneWaveletSR(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        guide_shape: Optional[str] = None
        refinement: bool = False
        sr_start_step: int = 1000
        sr_planes_only :bool = False
        sr_min_res : int = -1
        low_res_begining_only: bool = False
        learn_in_latent_space: bool = False

        hr_fit_mode_enabled: bool = False
        hr_fit_mode_refresh_every: int = 1000
        hr_fit_use_est_first_time: bool = False
        hr_fit_not_use_est_steps: int = 500
        hr_fit_interpolation_steps: int = 0
        hr_fit_use_interpolation: bool = False

        calculate_hr_metrics: bool = True
        hr_crop: int = -1
        hr_crop_fit: int = -1
        hr_crop_align_wth_lr:bool = False
        hr_crop_render: bool = False

        save_full_sr_steps: bool = False
        save_add_text:bool = False

        low_res_max_rays_before: int = -1
        low_res_max_rays: int = -1
        use_test_lpips: bool = False

        render_hr_max_rays: int = -1
        test_save_planes: bool = True
        automatic_optimization: bool = True
        fp16: bool = False

        low_res_loss: bool = True
        hr_crop_low_res: int = -1
        trinerflet_mode: bool = True



    cfg: Config

    def configure(self):
        # create geometry, material, background, renderer
        super().configure()

        if self.training or not self.cfg.refinement:
            self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)

        if self.cfg.guide_shape is not None:
            self.shape_loss = ShapeLoss(self.cfg.guide_shape)

        if self.cfg.calculate_hr_metrics or (self.C(self.cfg.loss.lambda_lr_sr_consistency_perceptual) > 0):
            self.lpips = LPIPS().eval()
            # self.lpips_test = self.lpips

        if self.cfg.automatic_optimization == False:
            self.automatic_optimization = False
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.cfg.fp16)

        if (self.cfg.trinerflet_mode == False):
            assert self.cfg.low_res_loss == False



    def configure_optimizers(self):
        from threestudio.systems.utils import parse_optimizer, parse_scheduler
        optimizer = parse_optimizer(self.cfg.optimizer, self)
        ret = {
            "optimizer": optimizer,
        }
        if self.cfg.scheduler is not None:
            if self.cfg.scheduler.name == 'exp_decay':
                print('********* exp decay scheduler ************')
                lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda iter:
                decay_function(iter, self.cfg.scheduler.warmup_steps, self.cfg.scheduler.warmup_factor,
                               self.cfg.scheduler.sched_base, self.cfg.scheduler.sched_exp,
                               self.cfg.scheduler.max_steps))
                interval = self.cfg.scheduler.get("interval", "step")
                assert interval in ["epoch", "step"]
                ret['lr_scheduler'] = {'scheduler': lr_scheduler, 'interval': interval}
            else:
                ret['lr_scheduler'] = parse_scheduler(self.cfg.scheduler, optimizer)
        return ret
    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        render_out = self.renderer(**batch)
        out = {
            **render_out,
        }
        return out

    def on_fit_start(self) -> None:
        super().on_fit_start()
        # only used in training
        self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(
            self.cfg.prompt_processor
        )
        self.hr_imgs_cache = {}


    def save_sr_data_grid(self,lr_gt,hr_before,hr_after,idx):
        self.save_image_grid(
            f"sr_net{self.true_global_step}_{idx}.png",
            [
                {
                    "type": "rgb",
                    "img": hr_before[0],
                    "kwargs": {"data_format": "HWC"},
                },
            ]
            + [
                {
                    "type": "rgb",
                    "img": hr_after[0],
                    "kwargs": {"data_format": "HWC"},
                },
            ] + [
                    {
                        "type": "rgb",
                        "img": F.interpolate(
                            lr_gt.permute(0,3,1,2), (hr_before.shape[1], hr_before.shape[2]),
                            mode="bilinear", align_corners=False
                        )[0],
                        "kwargs": {"data_format": "CHW"},
                    },
            ] + [
                {
                    "type": "rgb",
                    "img": torch.randn_like(hr_before[0]),
                    "kwargs": {"data_format": "HWC"},
                },
            ]
            ,
            name="sr_step",
            step=self.true_global_step,
            texts = ['high resolution before','high resolution after','low resolution gt' , 'noise'
                     ] if self.cfg.save_add_text else None
        )

    def crop_batch(self,hr_batch,gt_lr,additional_info = None,crop_sz_hr = None):
        if additional_info is None:
            if crop_sz_hr is None:
                crop_sz_hr = self.cfg.hr_crop
            imgs_hr_shape = hr_batch['rays_o'].shape

            lr_hr_ratio = round(imgs_hr_shape[1] / gt_lr.shape[1])

            row_begin = 0
            row_end = max(imgs_hr_shape[1] - crop_sz_hr, 0)
            row_interval = (row_begin, row_end + 1)
            row_idx = torch.randint(row_interval[0], row_interval[1], (imgs_hr_shape[0],))
            col_begin = 0
            col_end = max(imgs_hr_shape[2] - crop_sz_hr, 0)
            col_interval = (col_begin, col_end + 1)
            col_idx = torch.randint(col_interval[0], col_interval[1], (imgs_hr_shape[0],))

            if self.cfg.hr_crop_align_wth_lr:
                row_idx = (row_idx // lr_hr_ratio) * lr_hr_ratio
                col_idx = (col_idx // lr_hr_ratio) * lr_hr_ratio

            row_idx_lr = (row_idx // lr_hr_ratio)
            crop_sz_lr = (crop_sz_hr // lr_hr_ratio)
            col_idx_lr = (col_idx // lr_hr_ratio)
        else:
            row_idx = additional_info['row_idx_hr']
            col_idx = additional_info['col_idx_hr']
            crop_sz_hr = additional_info['crop_sz_hr']
            # assert crop_sz_hr == self.cfg.hr_crop
            row_idx_lr = additional_info['row_idx_lr']
            col_idx_lr = additional_info['col_idx_lr']
            crop_sz_lr = additional_info['crop_sz_lr']


        if (((self.C(self.cfg.loss.lambda_lr_sr_consistency) > 0) or
             (self.C(self.cfg.loss.lambda_lr_sr_consistency_perceptual) > 0)) and
                not self.cfg.hr_crop_align_wth_lr):
            raise ValueError('hr_crop_align_wth_lr must be activated')

        keys_to_cops = ['rays_o', 'rays_d']
        hr_batch_cropped = {}
        for key, val in hr_batch.items():
            if key in keys_to_cops:
                hr_batch_cropped[key] = val[:, row_idx:(row_idx + crop_sz_hr), col_idx:(col_idx + crop_sz_hr)]
                # print(hr_batch_cropped[key].shape)
            else:
                hr_batch_cropped[key] = val

        hr_batch = hr_batch_cropped

        res = {
            'row_idx_hr' : row_idx,
            'col_idx_hr' : col_idx,
            'crop_sz_hr' : crop_sz_hr,

            'row_idx_lr' : row_idx_lr,
            'col_idx_lr' : col_idx_lr,
            'crop_sz_lr' : crop_sz_lr
        }
        return hr_batch,res

    def training_step(self, batch, batch_idx):
        # print('training:')
        # print(self.geometry.training)
        if self.automatic_optimization:
            return self.training_step_aux(batch, batch_idx)
        # print('manual optimization on')
        optimizer = self.optimizers()
        optimizer.zero_grad()
        lr_scheduler = self.lr_schedulers()
        with torch.cuda.amp.autocast(enabled=self.cfg.fp16, dtype=torch.float16):
            res = self.training_step_aux(batch, batch_idx)
            loss = res['loss']
            # print('*********')
            # print('loss = {}'.format(loss))
            # print('loss = {}'.format(loss.shape))
        self.scaler.scale(loss).backward()
        self.scaler.step(optimizer)
        self.scaler.update()
        lr_scheduler.step()


    def training_step_aux(self, batch, batch_idx):
        lr_batch = batch['low_res']
        # print(lr_batch['index'])
        gt_lr = lr_batch['gt_rgb']

        sr_start_step = self.cfg.sr_start_step - self.cfg.hr_fit_interpolation_steps

        if (self.cfg.trinerflet_mode):
            self.geometry.encoding.encoding.enable_cache = True
            self.geometry.encoding.encoding.reset_cahce()

            if (not self.cfg.low_res_loss) or (self.global_step >= sr_start_step):
                self.geometry.encoding.encoding.set_double_mode(True)
            else:
                self.geometry.encoding.encoding.set_double_mode(False)
            self.geometry.encoding.encoding.set_resolution_mode('low_res')
            # with torch.cuda.amp.autocast(enabled=False):
            self.geometry.encoding.encoding.get_planes()


        # print('enter 1')
        if self.cfg.hr_fit_mode_enabled and (self.global_step % self.cfg.hr_fit_mode_refresh_every == 0):
            self.hr_imgs_cache = {}
        if (((self.global_step >= self.cfg.sr_start_step) and self.cfg.sr_planes_only and (self.cfg.sr_min_res > 0))
            or ((self.global_step >= self.cfg.sr_start_step) and self.cfg.low_res_begining_only)
            or (self.cfg.loss.lambda_l2_low_res <= 1e-8)
        ):
            guidance_out = {}
        else:
            if (not self.cfg.low_res_loss):
                guidance_out = {}
                if (self.global_step < self.cfg.sr_start_step):
                    # print('enter 2')
                    if (self.cfg.trinerflet_mode):
                        self.geometry.encoding.encoding.set_resolution_mode('high_res')
                    hr_batch = batch['high_res']

                    crop_enabled = ((self.cfg.hr_crop_low_res > 0) and (
                            self.cfg.hr_crop_low_res < min(hr_batch['rays_o'].shape[1], hr_batch['rays_o'].shape[2])))
                    if crop_enabled:
                        if self.cfg.hr_crop_render:
                            hr_batch, res = self.crop_batch(hr_batch, gt_lr,crop_sz_hr=self.cfg.hr_crop_low_res)
                            row_idx_lr = res['row_idx_lr']
                            col_idx_lr = res['col_idx_lr']
                            crop_sz_lr = res['crop_sz_lr']

                            gt_lr = gt_lr[:, row_idx_lr:(row_idx_lr + crop_sz_lr),
                                    col_idx_lr: (col_idx_lr + crop_sz_lr)]


                    out_hr_pre = self(hr_batch)
                    gt_pre = gt_lr.permute(0, 3, 1, 2)
                    est_pre = out_hr_pre["comp_rgb"]

                    resize_func = TF.Resize((gt_lr.shape[1], gt_lr.shape[2]), antialias=True)
                    est_pre = resize_func(est_pre.permute(0, 3, 1, 2))
                    # print('******************************')
                    # print(est_pre.shape)
                    guidance_out = {
                        'loss_l2_low_res': torch.nn.functional.mse_loss(est_pre, gt_pre)
                    }
            else:
                if (self.cfg.trinerflet_mode):
                    # self.geometry.encoding.encoding.set_double_mode(False)
                    self.geometry.encoding.encoding.set_resolution_mode('low_res')
                gt_img = lr_batch['gt_rgb']
                if self.cfg.learn_in_latent_space:
                    with torch.no_grad():
                        self.guidance.vae.to(dtype=torch.float32)
                        gt_img = self.guidance.encode_images(gt_img.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
                        if torch.isnan(gt_img).any():
                            print('nan values')
                gt_img2 = gt_img
                max_low_inds = -1
                lr_batch_to_render = lr_batch
                shuffled = False
                if (self.cfg.low_res_max_rays > 0) and (self.global_step >= sr_start_step):
                    max_low_inds = self.cfg.low_res_max_rays
                elif (self.cfg.low_res_max_rays_before > 0):
                    max_low_inds = self.cfg.low_res_max_rays_before
                    if 'low_res_shuffled' in batch:
                        lr_batch_to_render = batch['low_res_shuffled']
                        shuffled = True
                        # print('arrived were we should')



                if max_low_inds > 0:
                    if shuffled:
                        keys_to_perm = ['rays_o', 'rays_d','gt_rgb']
                        for key in keys_to_perm:
                            lr_batch_to_render[key] = lr_batch_to_render[key][:max_low_inds].unsqueeze(0).unsqueeze(0)
                            # print('key: {}'.format(key))
                            # print(lr_batch_to_render[key].shape)
                        gt_img2 = lr_batch_to_render['gt_rgb']
                    else:
                        B,H,W,ch = gt_img2.shape
                        assert B==1
                        idxs = torch.randperm(H*W)[:max_low_inds]
                        gt_img2 = gt_img2.view(H*W,-1)[idxs].unsqueeze(0).unsqueeze(0)
                        keys_to_perm = ['rays_o', 'rays_d']
                        for key in keys_to_perm:
                            lr_batch_to_render[key] = lr_batch_to_render[key].view(H*W,-1)[idxs].unsqueeze(0).unsqueeze(0)
                # print('enter 3')
                # print('rays_o.shape = {}'.format(lr_batch_to_render['rays_o'].shape))
                # print('rays_d.shape = {}'.format(lr_batch_to_render['rays_d'].shape))
                # print(lr_batch_to_render['rays_o'][0,0,:10])
                # print(lr_batch_to_render['rays_d'][0, 0, :10])
                out_lr = self(lr_batch_to_render)
                guidance_out = {
                    'loss_l2_low_res' : torch.nn.functional.mse_loss(out_lr["comp_rgb"], gt_img2)
                }
                # guidance_out['loss_l2_low_res_after'] = guidance_out['loss_l2_low_res']
                out = out_lr
        if self.global_step == self.cfg.sr_start_step:
            print('**********starting sd super res **************')
        if self.global_step >= sr_start_step:
            if (self.cfg.trinerflet_mode):
                # self.geometry.encoding.encoding.reset_cahce()
                # self.geometry.encoding.encoding.set_double_mode(True)
                self.geometry.encoding.encoding.set_resolution_mode('high_res')
            hr_batch = batch['high_res']
            prompt_utils = self.prompt_processor()
            if self.cfg.hr_fit_mode_enabled:
                idx = hr_batch['index']

                crop_enabled = ((self.cfg.hr_crop > 0) and (
                        self.cfg.hr_crop < min(hr_batch['rays_o'].shape[1], hr_batch['rays_o'].shape[2])))
                if not crop_enabled:
                    out_hr = self(hr_batch)
                if idx in self.hr_imgs_cache:
                    gt_hr = self.hr_imgs_cache[idx]['gt_hr'].to(device=self.device)
                    additional_info = self.hr_imgs_cache[idx]['additional_info']
                    if self.cfg.hr_crop_render:
                        hr_batch, res = self.crop_batch(hr_batch, gt_lr, additional_info)
                        row_idx_lr = res['row_idx_lr']
                        col_idx_lr = res['col_idx_lr']
                        crop_sz_lr = res['crop_sz_lr']
                        gt_lr = gt_lr[:, row_idx_lr:(row_idx_lr + crop_sz_lr),
                                col_idx_lr: (col_idx_lr + crop_sz_lr)]
                    # print('1')
                    # print('gt_lr.shape = {}'.format(gt_lr.shape))
                    # print('gt_hr.shape = {}'.format(gt_hr.shape))
                else:
                    if (self.global_step < self.cfg.sr_start_step) and ((self.cfg.sr_start_step - self.global_step) <= self.cfg.hr_fit_interpolation_steps):
                        additional_info = {}
                        if crop_enabled:
                            if self.cfg.hr_crop_render:
                                hr_batch, res = self.crop_batch(hr_batch, gt_lr)
                                row_idx_lr = res['row_idx_lr']
                                col_idx_lr = res['col_idx_lr']
                                crop_sz_lr = res['crop_sz_lr']

                                gt_lr = gt_lr[:, row_idx_lr:(row_idx_lr + crop_sz_lr),
                                        col_idx_lr: (col_idx_lr + crop_sz_lr)]
                                additional_info = res
                        gt_hr = F.interpolate(gt_lr.permute(0,3,1,2),(hr_batch['rays_o'].shape[1], hr_batch['rays_o'].shape[2])
                                              ,mode="bilinear", align_corners=False
                                              ).permute(0,2,3,1)
                        self.hr_imgs_cache[idx] = {
                            'gt_hr': gt_hr.clone().detach().cpu(),
                            'additional_info': additional_info
                        }
                    else:
                        use_hr = True
                        use_interpolation = False
                        if ((self.global_step - self.cfg.sr_start_step) < self.cfg.hr_fit_not_use_est_steps) and not self.cfg.hr_fit_use_est_first_time:
                            use_hr = False
                            if self.cfg.hr_fit_use_interpolation:
                                use_hr = True
                                use_interpolation = True
                        additional_info = {}
                        if crop_enabled:
                            if self.cfg.hr_crop_render:
                                hr_batch,res = self.crop_batch(hr_batch, gt_lr)
                                row_idx_lr = res['row_idx_lr']
                                col_idx_lr = res['col_idx_lr']
                                crop_sz_lr = res['crop_sz_lr']

                                gt_lr = gt_lr[:, row_idx_lr:(row_idx_lr + crop_sz_lr),
                                        col_idx_lr: (col_idx_lr + crop_sz_lr)]
                                additional_info = res
                            if use_interpolation:
                                intr = F.interpolate(gt_lr.permute(0,3,1,2),(hr_batch['rays_o'].shape[1], hr_batch['rays_o'].shape[2])
                                                     ,mode="bilinear", align_corners=False
                                                     ).permute(0,2,3,1).contiguous()
                                # print('intr.shape = ', intr.shape)
                                out_hr = {'comp_rgb':intr}
                            elif use_hr:
                                with torch.no_grad():
                                    self.renderer.eval()
                                    out_hr_old = self.render_high_res(hr_batch)
                                    out_hr = {'comp_rgb': out_hr_old['comp_rgb'].clone().detach()}
                                    del out_hr_old
                                    out_hr_old = None
                                    self.renderer.train()
                            else:
                                out_hr = {'comp_rgb':torch.zeros_like(hr_batch['rays_o'])}

                        tmp_guidance_out = self.guidance(
                        gt_lr,
                        out_hr["comp_rgb"],
                        prompt_utils,
                        generate_hr = True,
                        use_hr = use_hr,
                        **hr_batch,
                        )
                        gt_hr = tmp_guidance_out['comp_rgb']
                        gt_hr = gt_hr.permute(0,2,3,1)
                        self.hr_imgs_cache[idx] = {
                            'gt_hr' : gt_hr.clone().detach().cpu(),
                            'additional_info' : additional_info
                        }
                        # print('gt_lr.shape = {}'.format(gt_lr.shape))
                        # print('gt_hr.shape = {}'.format(gt_hr.shape))
                        if self.cfg.save_full_sr_steps:
                            self.save_sr_data_grid(gt_lr,out_hr["comp_rgb"],gt_hr,idx)
                        self.log('guidance_params/t', tmp_guidance_out['t'])
                        tmp_guidance_out = {}

                if crop_enabled:
                    if (not self.cfg.hr_crop_render) or (self.cfg.hr_crop_fit > 0):
                        if (not self.cfg.hr_crop_render):
                            hr_batch,res = self.crop_batch(hr_batch,gt_lr)
                        else:
                            hr_batch, res = self.crop_batch(hr_batch, gt_lr,crop_sz_hr=self.cfg.hr_crop_fit )
                        row_idx = res['row_idx_hr']
                        col_idx = res['col_idx_hr']
                        crop_sz_hr = res['crop_sz_hr']
                        row_idx_lr = res['row_idx_lr']
                        col_idx_lr = res['col_idx_lr']
                        crop_sz_lr = res['crop_sz_lr']

                        gt_hr = gt_hr[:, row_idx:(row_idx + crop_sz_hr), col_idx:(col_idx + crop_sz_hr)]
                        gt_lr = gt_lr[:, row_idx_lr:(row_idx_lr + crop_sz_lr),col_idx_lr : (col_idx_lr + crop_sz_lr)]

                    if (self.cfg.trinerflet_mode):
                        # self.geometry.encoding.encoding.reset_cahce()
                        # self.geometry.encoding.encoding.set_double_mode(True)
                        self.geometry.encoding.encoding.set_resolution_mode('high_res')
                    out_hr = self(hr_batch)


                guidance_out['loss_l2_high_res'] = torch.nn.functional.mse_loss(out_hr["comp_rgb"], gt_hr)
                guidance_out['loss_l1_high_res'] = torch.nn.functional.l1_loss(out_hr["comp_rgb"], gt_hr)
            else:
                out_hr = self(hr_batch)
                tmp_guidance_out = self.guidance(
                    lr_batch['gt_rgb'],
                    out_hr["comp_rgb"],
                    prompt_utils,
                    **hr_batch,
                )
                # print(tmp_guidance_out)
                for key,val in tmp_guidance_out.items():
                    assert not (key in guidance_out)
                    guidance_out[key] = val
            out = out_hr
            if self.C(self.cfg.loss.lambda_lr_sr_consistency) > 0:
                # print('here')
                resize_func = TF.Resize((gt_lr.shape[1],gt_lr.shape[2]),antialias=True)
                resized_hr_out = resize_func(out_hr["comp_rgb"].permute(0, 3, 1, 2))
                # print(resized_hr_out.shape)
                # print(lr_batch['gt_rgb'].permute(0, 3, 1, 2).shape)
                guidance_out['loss_lr_sr_consistency'] = torch.nn.functional.mse_loss(resized_hr_out,
                                                                            gt_lr.permute(0, 3, 1, 2))
            if self.C(self.cfg.loss.lambda_lr_sr_consistency_perceptual) > 0:
                resize_func = TF.Resize((gt_lr.shape[1], gt_lr.shape[2]),antialias=True)
                resized_hr_out = resize_func(out_hr["comp_rgb"].permute(0, 3, 1, 2))
                # print(resized_hr_out.shape)
                # print(lr_batch['gt_rgb'].permute(0, 3, 1, 2).shape)
                guidance_out['loss_lr_sr_consistency_perceptual'] =self.lpips(resized_hr_out,
                                                        gt_lr.permute(0, 3, 1, 2).contiguous()).view(-1).mean()

        loss = 0.0

        for name, value in guidance_out.items():
            if name.startswith("loss_"):
                weight = self.C(self.cfg.loss[name.replace("loss_", "lambda_")])
                if weight > 1e-6:
                    self.log(f"train/{name}", value)
                loss += value * weight

        if self.C(self.cfg.loss.lambda_orient) > 0:
            if "normal" not in out:
                raise ValueError(
                    "Normal is required for orientation loss, no normal is found in the output."
                )
            loss_orient = (
                out["weights"].detach()
                * dot(out["normal"], out["t_dirs"]).clamp_min(0.0) ** 2
            ).sum() / (out["opacity"] > 0).sum()
            self.log("train/loss_orient", loss_orient)
            loss += loss_orient * self.C(self.cfg.loss.lambda_orient)

        if (self.cfg.trinerflet_mode):
            wavelet_features = self.geometry.encoding.encoding.get_wavelet_features()
            if len(wavelet_features) > 0:
                all_elements = sum([val.numel() for val in wavelet_features])
                wavelet_reg = sum([val.abs().mean() * (val.numel() / all_elements) for val in (wavelet_features)])
                # wavelet_reg = sum([val.abs().mean() * (val.numel() / all_elements) for val in (wavelet_features)]) / len(
                #     wavelet_features)
                # print(wavelet_reg)
                self.log("train/loss_wavelet_reg", wavelet_reg)
                loss += wavelet_reg * self.C(self.cfg.loss.lambda_wavelet)

        if self.C(self.cfg.loss.lambda_sparsity) > 0:
            loss_sparsity = (out["opacity"] ** 2 + 0.01).sqrt().mean()
            self.log("train/loss_sparsity", loss_sparsity)
            loss += loss_sparsity * self.C(self.cfg.loss.lambda_sparsity)

        if self.C(self.cfg.loss.lambda_opaque):
            opacity_clamped = out["opacity"].clamp(1.0e-3, 1.0 - 1.0e-3)
            loss_opaque = binary_cross_entropy(opacity_clamped, opacity_clamped)
            self.log("train/loss_opaque", loss_opaque)
            loss += loss_opaque * self.C(self.cfg.loss.lambda_opaque)

        self.log("train/loss", loss.item())
        # if (
        #     self.cfg.guide_shape is not None
        #     and self.C(self.cfg.loss.lambda_shape) > 0
        #     and out["points"].shape[0] > 0
        # ):
        #     loss_shape = self.shape_loss(out["points"], out["density"])
        #     self.log("train/loss_shape", loss_shape)
        #     loss += loss_shape * self.C(self.cfg.loss.lambda_shape)

        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))


        self.log('guidance_params/min_step',self.guidance.min_step)
        self.log('guidance_params/max_step', self.guidance.max_step)

        if (self.cfg.trinerflet_mode):
            self.geometry.encoding.encoding.reset_cahce()
        # loss.backward()
        # print(wavelet_features[0].grad.abs().max())
        return {"loss": loss}

    def on_before_optimizer_step(self, optimizer):

        # wavelet_features = self.geometry.encoding.encoding.get_wavelet_features()
        # print('wavelet_grad = {}'.format([wavelet_features[i].grad.abs().max().item() for i in range(len(wavelet_features))]))
        # base_plane = self.geometry.encoding.encoding.planes_features
        # print('plane grad = {}'.format(base_plane.grad.abs().max().item()))
        # mlp = self.geometry.feature_network
        # print('mlp grad = {}'.format([param.grad.abs().max().item() for param in mlp.parameters()]))

        if self.cfg.sr_planes_only and (self.global_step >= self.cfg.sr_start_step):
            with torch.no_grad():
                # print('here')
                planes_grad = [val.grad for val in self.geometry.encoding.encoding.parameters()]
                for param in self.parameters():
                    param.grad = None
                for idx,param in enumerate(self.geometry.encoding.encoding.parameters()):
                    assert param.grad is None
                    if self.cfg.sr_min_res > 0:
                        if param.shape[-1] >= self.cfg.sr_min_res:
                            param.grad = planes_grad[idx]
                    else:
                        param.grad = planes_grad[idx]
                    # print('param.shape = {}'.format(param.shape))
                    # print('param grad is None = {}'.format(param.grad is None))
        # for param in self.parameters():
        # # for param in self.geometry.parameters():
        #     print(param.shape)
        #     print(param.grad.abs().mean())

    def on_validation_epoch_start(self):
        super().on_validation_epoch_start()
        self.validation_results = {
            'low_res_mse' : []
        }
        if self.cfg.calculate_hr_metrics:
            self.validation_results['high_res_mse'] = []
            self.validation_results['high_res_lpips'] = []
            self.validation_results['high_res_reference_mse'] = []
        if (self.cfg.trinerflet_mode):
            self.geometry.encoding.encoding.enable_cache = True
            self.geometry.encoding.encoding.reset_cahce()
            self.geometry.encoding.encoding.set_double_mode(True)

    def on_validation_epoch_end(self):
        psnr_func = lambda mse: -10*math.log10(mse)
        mse_lst = self.validation_results['low_res_mse']
        mse = sum(mse_lst)/len(mse_lst)
        psnr_lst = [psnr_func(val) for val in mse_lst]
        psnr = sum(psnr_lst)/len(psnr_lst)
        self.log("validation/low_res_mse",mse)
        self.log("validation/low_res_psnr", psnr)

        threestudio.info("validation/low_res_mse = {}".format(mse) )
        threestudio.info("validation/low_res_psnr = {}".format(psnr))

        res = {
            'low_res_mse': mse,
            'low_res_psnr': psnr
        }
        if self.cfg.calculate_hr_metrics:
            mse_lst = None
            mse = None
            psnr_lst = None
            psnr = None

            mse_lst_hr = self.validation_results['high_res_mse']
            mse_hr = sum(mse_lst_hr) / max(len(mse_lst_hr),1)
            psnr_lst_hr = [psnr_func(val) for val in mse_lst_hr]
            print(psnr_lst_hr)
            psnr_hr = sum(psnr_lst_hr) / max(len(psnr_lst_hr),1)
            lpips_lst = self.validation_results['high_res_lpips']
            lpips = sum(lpips_lst) / max(len(lpips_lst),1)

            high_res_reference_mse_lst = self.validation_results['high_res_reference_mse']
            high_res_reference_psnr_lst = [psnr_func(val) for val in high_res_reference_mse_lst]
            high_res_reference_psnr = sum(high_res_reference_psnr_lst) / max(len(high_res_reference_psnr_lst),1)

            self.log("validation/high_res_mse", mse_hr)
            self.log("validation/high_res_psnr", psnr_hr)
            self.log("validation/high_res_lpips", lpips)
            self.log("validation/high_res_reference_psnr", high_res_reference_psnr)

            threestudio.info("validation/high_res_mse = {}".format(mse_hr))
            threestudio.info("validation/high_res_psnr = {}".format(psnr_hr))
            threestudio.info("validation/high_res_lpips = {}".format(lpips))
            threestudio.info("validation/high_res_reference_psnr = {}".format(high_res_reference_psnr))

            res['high_res_mse'] = mse_hr
            res['high_res_psnr'] = psnr_hr
            res['high_res_lpips'] = lpips
            res['high_res_reference_psnr'] = high_res_reference_psnr

        self.save_json('val_results_{}.json'.format(self.global_step), res)
        self.validation_results = {}
        if (self.cfg.trinerflet_mode):
            wavelet_features = self.geometry.encoding.encoding.get_wavelet_features()
            for wavelet in wavelet_features:
                threestudio.info('shape: {}, abs mean: {}'.format(wavelet.shape,wavelet.abs().mean()))
            self.geometry.encoding.encoding.enable_cache = False
            self.geometry.encoding.encoding.reset_cahce()

    def render_high_res(self,hr_batch):
        if self.cfg.render_hr_max_rays > 0:
            keys_to_perm = ['rays_o', 'rays_d']
            keys2 = ['camera_positions','light_positions']
            B,H,W,C = hr_batch['rays_o'].shape
            # print('rays_o.shape: {}'.format(hr_batch['rays_o'].shape))
            assert B==1
            res = {}
            for key in keys_to_perm:
                hr_batch[key] = hr_batch[key].reshape(B,H*W,1,C)
            for idx in range(math.ceil(H*W/self.cfg.render_hr_max_rays)):
                tmp_batch = {}
                for key in keys_to_perm:
                    tmp_batch[key] = hr_batch[key][:,idx*self.cfg.render_hr_max_rays:(idx+1)*self.cfg.render_hr_max_rays]
                for key in keys2:
                    tmp_batch[key] = hr_batch[key]
                tmp_res = self(tmp_batch)
                for key in tmp_res.keys():
                    if not (key in res):
                        res[key] = []
                    res[key].append(tmp_res[key])
            for key,val in res.items():
                # print('key = {}'.format(key))
                # print('val1.shape = {}'.format([tt.shape for tt in val]))
                val = torch.cat(val,dim=1)
                # print('val2.shape = {}'.format(val.shape))
                res[key] = val.view(B,H,W,-1)

            for key in keys_to_perm:
                hr_batch[key] = hr_batch[key].view(B,H,W,C)
            return res

        return self(hr_batch)

    def validation_step(self, batch, batch_idx):
        return self.validation_step_aux(batch, batch_idx)
        # if self.automatic_optimization:
        #     return self.validation_step_aux(batch, batch_idx)
        # with torch.cuda.amp.autocast(enabled=self.cfg.fp16, dtype=torch.float16):
        #     return self.validation_step_aux(batch, batch_idx)

    def validation_step_aux(self, batch, batch_idx):
        # print('validation_step')
        if (self.cfg.trinerflet_mode):
            self.geometry.encoding.encoding.set_resolution_mode('low_res')
        lr_batch = batch['low_res']
        with torch.no_grad():
            out_lr = self(lr_batch)
        res = out_lr["comp_rgb"]
        if self.cfg.learn_in_latent_space:
            res = self.guidance.decode_latents(res.permute(0,3,1,2).float(),128,128).permute(0,2,3,1)
        self.validation_results['low_res_mse'].append(
            torch.nn.functional.mse_loss(res, lr_batch['gt_rgb']).item()
        )
        self.save_image_grid(
            f"it{self.true_global_step}-{lr_batch['index']}.png",
            [
                {
                    "type": "rgb",
                    "img": res[0],
                    "kwargs": {"data_format": "HWC"},
                },
            ]
            + (
                [
                    {
                        "type": "rgb",
                        "img": out_lr["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out_lr
                else []
            )
            + [
                {
                    "type": "grayscale",
                    "img": out_lr["opacity"][0, :, :, 0],
                    "kwargs": {"cmap": None, "data_range": (0, 1)},
                },
            ]
            +  [
                {
                    "type": "rgb",
                    "img": lr_batch['gt_rgb'][0],
                    "kwargs": {"data_format": "HWC"},
                },
            ]
            ,
            name="validation_step",
            step=self.true_global_step,
        )

        # sr_start_step = self.cfg.sr_start_step - self.cfg.hr_fit_interpolation_steps
        # if self.global_step >= sr_start_step:

        out_lr = None
        if (self.cfg.trinerflet_mode):
            self.geometry.encoding.encoding.set_resolution_mode('high_res')
        hr_batch = batch['high_res']



        with torch.no_grad():
            # out_hr = self(hr_batch)
            out_hr = self.render_high_res(hr_batch)

            interpolated_hr = F.interpolate(
                lr_batch['gt_rgb'].permute(0, 3, 1, 2), (out_hr["comp_rgb"].shape[1], out_hr["comp_rgb"].shape[2]),
                mode="bilinear", align_corners=False
            )

            if ('gt_rgb' in hr_batch) and self.cfg.calculate_hr_metrics:
                gt_image = hr_batch['gt_rgb'].permute(0,3,1,2).contiguous()
                est_image = out_hr['comp_rgb'].permute(0,3,1,2)
                self.validation_results['high_res_mse'].append(
                    torch.nn.functional.mse_loss(est_image, gt_image).item()
                )
                self.validation_results['high_res_lpips'].append(
                    self.lpips(est_image,gt_image).item()
                )

                self.validation_results['high_res_reference_mse'].append(
                    torch.nn.functional.mse_loss(interpolated_hr, gt_image).item()
                )

        self.save_image_grid(
            f"it{self.true_global_step}-{hr_batch['index']}_hr.png",
            [
                {
                    "type": "rgb",
                    "img": out_hr["comp_rgb"][0],
                    "kwargs": {"data_format": "HWC"},
                },
            ]
            + (
                [
                    {
                        "type": "rgb",
                        "img": out_hr["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out_hr
                else []
            )
            + [
                {
                    "type": "grayscale",
                    "img": out_hr["opacity"][0, :, :, 0],
                    "kwargs": {"cmap": None, "data_range": (0, 1)},
                },
            ] + [
                {
                    "type": "rgb",
                    "img": interpolated_hr[0],
                    "kwargs": {"data_format": "CHW"},
                },
            ] + ([
                {
                    "type": "rgb",
                    "img": hr_batch['gt_rgb'][0],
                    "kwargs": {"data_format": "HWC"},
                },
                {
                    "type": "rgb",
                    "img": (hr_batch['gt_rgb'][0]-out_hr["comp_rgb"][0]).abs(),
                    "kwargs": {"data_format": "HWC"},
                }
            ] if ('gt_rgb' in hr_batch) else [])
            ,
            name="validation_step_hr",
            step=self.true_global_step,
        )


    def test_step(self, batch, batch_idx):
        return self.test_step_aux(batch, batch_idx)
        # if self.automatic_optimization:
        #     return self.test_step_aux(batch, batch_idx)
        # with torch.cuda.amp.autocast(enabled=self.cfg.fp16, dtype=torch.float16):
        #     return self.test_step_aux(batch, batch_idx)

    def test_step_aux(self, batch, batch_idx):
        if (self.cfg.trinerflet_mode):
            # self.geometry.encoding.encoding.reset_cahce()
            # self.geometry.encoding.encoding.set_double_mode(False)
            self.geometry.encoding.encoding.set_resolution_mode('low_res')
        assert batch['low_res']['index'] == batch['high_res']['index']
        lr_batch = batch['low_res']
        with torch.no_grad():
            out_lr = self(lr_batch)
        if 'gt_rgb' in lr_batch:
            self.test_results['low_res_mse'].append(
                torch.nn.functional.mse_loss(out_lr["comp_rgb"], lr_batch['gt_rgb']).item()
            )
        self.save_image_grid(
            f"it{self.true_global_step}-test/{lr_batch['index']}.png",
            [
                {
                    "type": "rgb",
                    "img": out_lr["comp_rgb"][0],
                    "kwargs": {"data_format": "HWC"},
                },
            ]
            + (
                [
                    {
                        "type": "rgb",
                        "img": out_lr["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out_lr
                else []
            )
            + [
                {
                    "type": "grayscale",
                    "img": out_lr["opacity"][0, :, :, 0],
                    "kwargs": {"cmap": None, "data_range": (0, 1)},
                },
            ]
            + ([
                {
                    "type": "rgb",
                    "img": lr_batch['gt_rgb'][0],
                    "kwargs": {"data_format": "HWC"},
                },
            ] if ('gt_rgb' in lr_batch) else [])
            ,
            name="test_step",
            step=self.true_global_step,
        )
        out_lr = None
        if (self.cfg.trinerflet_mode):
            # self.geometry.encoding.encoding.reset_cahce()
            # self.geometry.encoding.encoding.set_double_mode(True)
            self.geometry.encoding.encoding.set_resolution_mode('high_res')
        hr_batch = batch['high_res']
        with torch.no_grad():
            out_hr = self.render_high_res(hr_batch)
            interpolated_hr = F.interpolate(
                lr_batch['gt_rgb'].permute(0, 3, 1, 2), (out_hr["comp_rgb"].shape[1], out_hr["comp_rgb"].shape[2]),
                mode="bilinear", align_corners=False
            )
            if ('gt_rgb' in hr_batch) and self.cfg.calculate_hr_metrics:
                gt_image = hr_batch['gt_rgb'].permute(0, 3, 1, 2).contiguous()
                est_image = out_hr['comp_rgb'].permute(0, 3, 1, 2)

                self.test_results['high_res_mse'].append(
                    torch.nn.functional.mse_loss(est_image, gt_image).item()
                )
                self.test_results['high_res_mse2'].append(
                    torch.nn.functional.mse_loss(to8b(est_image).float(),to8b(gt_image).float()).item()
                )
                self.test_results['high_res_lpips'].append(
                    self.lpips_test(est_image,gt_image).item()
                )
                self.test_results['high_res_ssim'].append(
                    self.ssim(est_image.clone().detach().cpu(), gt_image.clone().detach().cpu()).item()
                )
                self.test_results['high_res_reference_mse'].append(
                    torch.nn.functional.mse_loss(interpolated_hr, gt_image).item()
                )
                # print('index = {}, mse = {}'.format(hr_batch['index'],torch.nn.functional.mse_loss(est_image, gt_image).item()))
        # self.save_image(f"it{self.true_global_step}-test_hr_sr_only/{hr_batch['index']}.png",to8b(out_hr["comp_rgb"][0]))

        # imageio.imwrite(self.get_save_path(f"it{self.true_global_step}-test_hr_sr_only/{hr_batch['index']}.png"),to8b(out_hr["comp_rgb"][0]).clone().detach().cpu().numpy())
        # torch.save({'image' : out_hr["comp_rgb"][0].clone().detach().cpu().numpy() },self.get_save_path(f"it{self.true_global_step}-test_hr_sr_only/{hr_batch['index']}.pt"))
        self.save_image_grid(
            f"it{self.true_global_step}-test_hr/{hr_batch['index']}.png",
            [
                {
                    "type": "rgb",
                    "img": to8b(out_hr["comp_rgb"][0]),
                    "kwargs": {"data_format": "HWC"},
                },
            ]
            # + (
            #     [
            #         {
            #             "type": "rgb",
            #             "img": out_hr["comp_normal"][0],
            #             "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
            #         }
            #     ]
            #     if "comp_normal" in out_hr
            #     else []
            # )
            # + [
            #     {
            #         "type": "grayscale",
            #         "img": out_hr["opacity"][0, :, :, 0],
            #         "kwargs": {"cmap": None, "data_range": (0, 1)},
            #     },
            # ]
            + ([
                    {
                        "type": "rgb",
                        "img": to8b(interpolated_hr)[0],
                        "kwargs": {"data_format": "CHW"},
                    },
                ] if ('gt_rgb' in lr_batch) else []  )
                    +  ([
                    {
                        "type": "rgb",
                        "img": to8b(hr_batch['gt_rgb'])[0],
                        "kwargs": {"data_format": "HWC"},
                    },
                    # {
                    #     "type": "rgb",
                    #     "img": (hr_batch['gt_rgb'][0]-out_hr["comp_rgb"][0]).abs(),
                    #     "kwargs": {"data_format": "HWC"},
                    # }
                ] if ('gt_rgb' in hr_batch) else [])
            ,
            name="test_step_hr",
            step=self.true_global_step,
            texts=(['high resolution estimate', 'low resolution ground-truth'] +
                   ['high resolution ground-truth'] if 'gt_rgb' in hr_batch else []
                   ) if self.cfg.save_add_text else None
        )

    def on_test_epoch_start(self):
        super().on_test_epoch_start()
        self.test_results = {
            'low_res_mse' : []
        }
        if self.cfg.calculate_hr_metrics:
            self.test_results['high_res_mse'] = []
            self.test_results['high_res_mse2'] = []
            self.test_results['high_res_lpips'] = []
            self.test_results['high_res_ssim'] = []
            self.test_results['high_res_reference_mse'] = []

        if (self.cfg.trinerflet_mode):
            self.geometry.encoding.encoding.enable_cache = True
            self.geometry.encoding.encoding.reset_cahce()
            self.geometry.encoding.encoding.set_double_mode(True)

        self.ssim = None
        if self.cfg.calculate_hr_metrics:
            self.lpips_test = self.lpips
            self.ssim = SSIM((0,1))
            if self.cfg.use_test_lpips: #not perfect but this is the best location for not braking the code
                import lpips
                print('******** using different lpips ************')
                self.lpips_test = lpips.LPIPS(net='alex').eval().to(self.device)

    def save_tensor(self,planes,save_path,all,prefix = 'plane'):
        # f_save = os.path.join(self.workspace, 'planes.pt')
        # torch.save({'planes':planes},f_save)
        for axis in range(planes.shape[0]):
            ch_list = list(range(planes.shape[1]))
            if not all:
                ch_list = [random.choice(ch_list)]
            for ch_idx in ch_list:
                image_f = os.path.join(save_path, '{}_{}_{}_{}.png'.format(prefix,self.true_global_step, axis, ch_idx))
                current_plane = (planes[axis, ch_idx] * 255).round().numpy().astype(np.uint8)
                cv2.imwrite(image_f, current_plane)

    def on_test_epoch_end(self):
        self.save_img_sequence(
            f"it{self.true_global_step}-test",
            f"it{self.true_global_step}-test",
            "(\d+)\.png",
            save_format="mp4",
            fps=30,
            name="test",
            step=self.true_global_step,
        )
        self.save_img_sequence(
            f"it{self.true_global_step}-test_hr",
            f"it{self.true_global_step}-test_hr",
            "(\d+)\.png",
            save_format="mp4",
            fps=30,
            name="test",
            step=self.true_global_step,
        )
        if self.cfg.test_save_planes and self.cfg.trinerflet_mode:
            wavelet_dir = Path(self.get_save_dir(),'wavelet')
            wavelet_dir.mkdir(exist_ok=True)
            # with torch.cuda.amp.autocast(enabled=False):
            self.geometry.encoding.encoding.enable_cache = False
            self.geometry.encoding.encoding.reset_cahce()
            planes = self.geometry.encoding.encoding.get_planes()

            planes = planes.cpu().clone().detach().float()
            threestudio.info('planes.shape: {}'.format(planes.shape))

            tmp = planes.view(planes.shape[0], planes.shape[1], -1)
            a = tmp.min(dim=-1, keepdim=True).values.unsqueeze(-1)
            b = tmp.max(dim=-1, keepdim=True).values.unsqueeze(-1)
            planes = (planes - a) / (b - a)
            self.save_tensor(planes, str(wavelet_dir), True)

        psnr_func = lambda mse: -10 * math.log10(mse)
        psnr_func2 = lambda mse: 10 * math.log10((255**2)/mse)
        mse_lst = self.test_results['low_res_mse']
        if len(mse_lst) == 0:
            if (self.cfg.trinerflet_mode):
                self.geometry.encoding.encoding.enable_cache = True
                self.geometry.encoding.encoding.reset_cahce()
            return

        mse = sum(mse_lst) / len(mse_lst)
        psnr_lst = [psnr_func(val) for val in mse_lst]
        psnr = sum(psnr_lst) / len(psnr_lst)
        self.log("test/low_res_mse", mse)
        self.log("test/low_res_psnr", psnr)

        threestudio.info("test/low_res_mse = {}".format(mse))
        threestudio.info("test/low_res_psnr = {}".format(psnr))

        res = {
            'low_res_mse' : mse,
            'low_res_psnr' : psnr
        }
        res2 = {
            'mse_lst' : mse_lst,
            'psnr_lst' : psnr_lst,

        }

        if self.cfg.calculate_hr_metrics:
            mse_lst = None
            mse = None
            psnr_lst = None
            psnr = None

            mse_lst_hr = self.test_results['high_res_mse']
            # print(mse_lst_hr)
            mse_hr = sum(mse_lst_hr) / max(len(mse_lst_hr),1)
            psnr_lst_hr = [psnr_func(val) for val in mse_lst_hr]
            print(psnr_lst_hr)
            psnr_hr = sum(psnr_lst_hr) / max(len(psnr_lst_hr),1)
            lpips_lst = self.test_results['high_res_lpips']
            lpips = sum(lpips_lst) / max(len(lpips_lst),1)

            ssim_lst = self.test_results['high_res_ssim']
            ssim = sum(ssim_lst) / max(len(ssim_lst),1)

            mse_lst_hr2 = self.test_results['high_res_mse2']
            mse_hr2 = sum(mse_lst_hr2) / max(len(mse_lst_hr2),1)
            psnr_lst_hr2 = [psnr_func2(val) for val in mse_lst_hr2]
            psnr_hr2 = sum(psnr_lst_hr2) / max(len(psnr_lst_hr2),1)

            high_res_reference_mse_lst = self.test_results['high_res_reference_mse']
            high_res_reference_psnr_lst = [psnr_func(val) for val in high_res_reference_mse_lst]
            high_res_reference_psnr = sum(high_res_reference_psnr_lst) / max(len(high_res_reference_psnr_lst),1)

            self.log("test/high_res_mse", mse_hr)
            self.log("test/high_res_psnr", psnr_hr)
            self.log("test/high_res_lpips", lpips)
            self.log("test/high_res_ssim", ssim)

            self.log("test/high_res_mse2", mse_hr2)
            self.log("test/high_res_psnr2", psnr_hr2)

            threestudio.info("test/high_res_mse = {}".format(mse_hr))
            threestudio.info("test/high_res_psnr = {}".format(psnr_hr))
            threestudio.info("test/high_res_lpips = {}".format(lpips))
            threestudio.info("test/high_res_ssim = {}".format(ssim))

            threestudio.info("test/high_res_mse2 = {}".format(mse_hr2))
            threestudio.info("test/high_res_psnr2 = {}".format(psnr_hr2))

            threestudio.info("test/high_res_reference_psnr = {}".format(high_res_reference_psnr))

            res['high_res_mse'] = mse_hr
            res['high_res_psnr'] = psnr_hr
            res['high_res_lpips'] = lpips
            res['high_res_ssim'] = ssim

            res['high_res_mse2'] = mse_hr2
            res['high_res_psnr2'] = psnr_hr2
            res['high_res_reference_psnr'] = high_res_reference_psnr

            res2['mse_lst_hr'] = mse_lst_hr
            res2['psnr_lst_hr'] = psnr_lst_hr
            res2['lpips_lst_hr'] = lpips_lst
            res2['ssim_lst_hr'] = ssim_lst
            res2['high_res_reference_psnr_lst'] = high_res_reference_psnr_lst

        self.save_json('final_results_{}.json'.format(self.global_step),res)
        self.save_json('final_results_{}_per_frame.json'.format(self.global_step), res2)
        if (self.cfg.trinerflet_mode):
            self.geometry.encoding.encoding.enable_cache = True
            self.geometry.encoding.encoding.reset_cahce()
