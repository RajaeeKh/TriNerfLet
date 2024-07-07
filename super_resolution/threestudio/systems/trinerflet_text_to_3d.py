import math
import random
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


def batch_operation(batch,func):
    new_batch = {}
    for key,val in batch.items():
        if isinstance(val,torch.Tensor):
            new_batch[key] = func(val)
        else:
            new_batch[key] = val
    return new_batch

@threestudio.register("trinerflet-generation")
class TriplaneWaveletSR(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        guide_shape: Optional[str] = None
        refinement: bool = False

        learn_in_latent_space: bool = False



        save_full_steps: bool = False
        save_add_text: bool = False

        render_eval_max_rays: int = -1
        refresh_every: int = 500
        views_per_refresh: List[int] = field(default_factory=lambda: [200])

        automatic_optimization: bool = True
        fp16: bool = False



    cfg: Config

    def configure(self):
        # create geometry, material, background, renderer
        super().configure()

        if self.training or not self.cfg.refinement:
            self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)

        if self.cfg.guide_shape is not None:
            self.shape_loss = ShapeLoss(self.cfg.guide_shape)

        if self.cfg.automatic_optimization == False:
            self.automatic_optimization = False
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.cfg.fp16)



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
        self.imgs_cache = []


    def save_sr_data_grid(self,img_before,img_after,idx):
        self.save_image_grid(
            f"trinerflet_{self.true_global_step}_{idx}.png",
            [
                {
                    "type": "rgb",
                    "img": img_before[0],
                    "kwargs": {"data_format": "HWC"},
                },
            ]
            + [
                {
                    "type": "rgb",
                    "img": img_after[0],
                    "kwargs": {"data_format": "HWC"},
                },
            ] + [
                {
                    "type": "rgb",
                    "img": torch.randn_like(img_before[0]),
                    "kwargs": {"data_format": "HWC"},
                },
            ]
            ,
            name="step",
            step=self.true_global_step,
            texts = ['img before','img after' , 'noise'
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
        self.scaler.scale(loss).backward()
        self.scaler.step(optimizer)
        self.scaler.update()
        lr_scheduler.step()

    def training_step_aux(self, batch, batch_idx):
        # lr_batch = batch
        # print(lr_batch['index'])
        
        if (self.global_step % self.cfg.refresh_every == 0):
            self.imgs_cache = []
        
        cache_results = True
        number_of_new_imgs = self.cfg.views_per_refresh[min(len(self.cfg.views_per_refresh)-1,self.global_step // self.cfg.refresh_every)]
        self.log("train_params/imgs_refresh", self.C(number_of_new_imgs))
        # print('***********************************')
        # print(number_of_new_imgs)
        # print('***********************************')
        if ((self.global_step % self.cfg.refresh_every) // number_of_new_imgs) > 0:
            cache_results = False
            cached_data = random.choice(self.imgs_cache)
            cached_batch = batch_operation(cached_data['batch'], lambda x:x.to(batch['rays_o'].device))
            gt_img = cached_data['gt_img'].to(batch['rays_o'].device)
            batch = cached_batch

        self.geometry.encoding.encoding.enable_cache = True
        self.geometry.encoding.encoding.reset_cahce()
        self.geometry.encoding.encoding.set_double_mode(False)
        self.geometry.encoding.encoding.set_resolution_mode('low_res')
        with torch.cuda.amp.autocast(enabled=False):
            self.geometry.encoding.encoding.get_planes()


        

        if self.cfg.learn_in_latent_space:
            raise ValueError('Unimplemented')
            with torch.no_grad():
                self.guidance.vae.to(dtype=torch.float32)
                gt_img = self.guidance.encode_images(gt_img.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
                if torch.isnan(gt_img).any():
                    print('nan values')

        out = self(batch)
        prompt_utils = self.prompt_processor()

        
        if cache_results:
            tmp_guidance_out = self.guidance(
                rgb=out["comp_rgb"],
                prompt_utils=prompt_utils,
                use_img=True,
                **batch
            )
            gt_img = tmp_guidance_out['comp_rgb']
            gt_img = gt_img.permute(0, 2, 3, 1)
            self.imgs_cache.append( {
                'gt_img': gt_img.clone().detach().cpu(),
                'batch': batch_operation(batch, lambda x: x.clone().detach().cpu())
            } )
            # print('gt_lr.shape = {}'.format(gt_lr.shape))
            # print('gt_hr.shape = {}'.format(gt_hr.shape))
            if self.cfg.save_full_steps:
                self.save_sr_data_grid(out["comp_rgb"], gt_img, batch_idx)
            ref_img = tmp_guidance_out['ref_img']
            if ref_img is not None:
                self.save_image('reference_image.png',to8b(ref_img[0].permute(1,2,0).clone().detach().cpu()))
            self.log('guidance_params/t', tmp_guidance_out['t'])
        guidance_out = {}
        guidance_out['loss_l2'] = torch.nn.functional.mse_loss(out["comp_rgb"], gt_img)
        guidance_out['loss_l1'] = torch.nn.functional.l1_loss(out["comp_rgb"], gt_img)


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

        self.geometry.encoding.encoding.reset_cahce()
        # loss.backward()
        # print(wavelet_features[0].grad.abs().max())
        return {"loss": loss}

    def on_before_optimizer_step(self, optimizer):
        pass


    def on_validation_epoch_start(self):
        super().on_validation_epoch_start()
        pass

    def on_validation_epoch_end(self):
        pass

    def render_eval(self,hr_batch):
        if self.cfg.render_eval_max_rays > 0:
            keys_to_perm = ['rays_o', 'rays_d']
            keys2 = ['camera_positions','light_positions']
            B,H,W,C = hr_batch['rays_o'].shape
            # print('rays_o.shape: {}'.format(hr_batch['rays_o'].shape))
            assert B==1
            res = {}
            for key in keys_to_perm:
                hr_batch[key] = hr_batch[key].reshape(B,H*W,1,C)
            for idx in range(math.ceil(H*W/self.cfg.render_eval_max_rays)):
                tmp_batch = {}
                for key in keys_to_perm:
                    tmp_batch[key] = hr_batch[key][:,idx*self.cfg.render_eval_max_rays:(idx+1)*self.cfg.render_eval_max_rays]
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
        lr_batch = batch
        with torch.no_grad():
            out_lr = self(lr_batch)
        res = out_lr["comp_rgb"]
        if self.cfg.learn_in_latent_space:
            res = self.guidance.decode_latents(res.permute(0,3,1,2).float(),128,128).permute(0,2,3,1)

        self.save_image_grid(
            f"it{self.true_global_step}-{lr_batch['index'].item()}.png",
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
            ,
            name="validation_step",
            step=self.true_global_step,
        )


    def test_step(self, batch, batch_idx):
        lr_batch = batch
        with torch.no_grad():
            out_lr = self(lr_batch)
        self.save_image_grid(
            f"it{self.true_global_step}-test/{lr_batch['index'].item()}.png",
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
            ,
            name="test_step",
            step=self.true_global_step,
        )

    def on_test_epoch_start(self):
        super().on_test_epoch_start()
        pass

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
