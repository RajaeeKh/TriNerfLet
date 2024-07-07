import json
from dataclasses import dataclass, field

import torch

import threestudio
from threestudio.systems.base import BaseLift3DSystem
from threestudio.utils.ops import binary_cross_entropy, dot
from threestudio.utils.typing import *

import random


def batch_operation(batch,func):
    new_batch = {}
    for key,val in batch.items():
        if isinstance(val,torch.Tensor):
            new_batch[key] = func(val)
        else:
            new_batch[key] = val
    return new_batch

@threestudio.register("dreamfusion-system2")
class DreamFusion(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        save_full_steps: bool = False

        cache_steps: int = -1
        view_per_cache: int = 20
        cache_start_step: int = 0

        trinerflet_rep : bool = False

        export_views: bool = False

    cfg: Config

    def configure(self):
        # create geometry, material, background, renderer
        super().configure()
        self.cache = []

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        render_out = self.renderer(**batch)
        return {
            **render_out,
        }

    def on_fit_start(self) -> None:
        super().on_fit_start()
        # only used in training
        self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(
            self.cfg.prompt_processor
        )
        self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)



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
        )
    def training_step(self, batch, batch_idx):
        cache_results = False
        apply_guidance = True
        gt_img = None
        if (self.cfg.cache_steps > 0) and (self.global_step >= self.cfg.cache_start_step):
            assert (self.cfg.view_per_cache > 0) and (self.cfg.view_per_cache < self.cfg.cache_steps)
            if (self.global_step % self.cfg.cache_steps == 0):
                self.cache = []
            if ((self.global_step % self.cfg.cache_steps) // self.cfg.view_per_cache) == 0:
                cache_results = True
            else:
                cached_data = random.choice(self.cache)
                cached_batch = batch_operation(cached_data['batch'], lambda x: x.to(batch['rays_o'].device))
                gt_img = cached_data['gt_img'].to(batch['rays_o'].device)
                t = cached_data['t']
                batch = cached_batch
                batch['bg_color'] = cached_data['bg_color'].to(batch['rays_o'].device)
                apply_guidance = False

        if self.cfg.trinerflet_rep:
            self.geometry.encoding.encoding.enable_cache = True
            self.geometry.encoding.encoding.reset_cahce()
            self.geometry.encoding.encoding.set_double_mode(False)
            self.geometry.encoding.encoding.set_resolution_mode('low_res')
            with torch.cuda.amp.autocast(enabled=False):
                self.geometry.encoding.encoding.get_planes()


        out = self(batch)
        prompt_utils = self.prompt_processor()
        if apply_guidance:
            guidance_out = self.guidance(
                out["comp_rgb"], prompt_utils, **batch, rgb_as_latents=False
            )
        else:
            guidance_out = self.guidance(
                out["comp_rgb"], prompt_utils, **batch, rgb_as_latents=False,
                cache_data={
                    'res_img': gt_img,
                    't': t
                }
            )

        if cache_results:
            self.cache.append({
                'gt_img': guidance_out['res_img'].clone().detach().cpu(),
                'bg_color' : out['comp_rgb_bg'].clone().detach().cpu(),
                't' : guidance_out['t'],
                'batch': batch_operation(batch, lambda x: x.clone().detach().cpu())
            })


        if self.cfg.save_full_steps and ('res_img' in guidance_out):
            self.save_sr_data_grid(out["comp_rgb"], guidance_out['res_img'].permute(0,2,3,1), batch_idx)

        del guidance_out['res_img']
        self.log('guidance_params/t', guidance_out['t'])



        loss = 0.0

        for name, value in guidance_out.items():
            self.log(f"train/{name}", value)
            if name.startswith("loss_"):
                loss += value * self.C(self.cfg.loss[name.replace("loss_", "lambda_")])

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

        loss_sparsity = (out["opacity"] ** 2 + 0.01).sqrt().mean()
        self.log("train/loss_sparsity", loss_sparsity)
        loss += loss_sparsity * self.C(self.cfg.loss.lambda_sparsity)

        opacity_clamped = out["opacity"].clamp(1.0e-3, 1.0 - 1.0e-3)
        loss_opaque = binary_cross_entropy(opacity_clamped, opacity_clamped)
        self.log("train/loss_opaque", loss_opaque)
        loss += loss_opaque * self.C(self.cfg.loss.lambda_opaque)

        if self.cfg.trinerflet_rep:
            wavelet_features = self.geometry.encoding.encoding.get_wavelet_features()
            if len(wavelet_features) > 0:
                all_elements = sum([val.numel() for val in wavelet_features])
                wavelet_reg = sum([val.abs().mean() * (val.numel() / all_elements) for val in (wavelet_features)])
                # wavelet_reg = sum([val.abs().mean() * (val.numel() / all_elements) for val in (wavelet_features)]) / len(
                #     wavelet_features)
                # print(wavelet_reg)
                self.log("train/loss_wavelet_reg", wavelet_reg)
                loss += wavelet_reg * self.C(self.cfg.loss.lambda_wavelet)

        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))

        return {"loss": loss}

    def on_validation_epoch_start(self):
        super().on_validation_epoch_start()
        if self.cfg.trinerflet_rep:
            self.geometry.encoding.encoding.enable_cache = True
            self.geometry.encoding.encoding.reset_cahce()
            self.geometry.encoding.encoding.set_double_mode(False)
            self.geometry.encoding.encoding.set_resolution_mode('low_res')
            with torch.cuda.amp.autocast(enabled=False):
                self.geometry.encoding.encoding.get_planes()
    def validation_step(self, batch, batch_idx):
        out = self(batch)
        self.save_image_grid(
            f"it{self.true_global_step}-{batch['index'][0]}.png",
            [
                {
                    "type": "rgb",
                    "img": out["comp_rgb"][0],
                    "kwargs": {"data_format": "HWC"},
                },
            ]
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out
                else []
            )
            + [
                {
                    "type": "grayscale",
                    "img": out["opacity"][0, :, :, 0],
                    "kwargs": {"cmap": None, "data_range": (0, 1)},
                },
            ],
            name="validation_step",
            step=self.true_global_step,
        )

    def on_validation_epoch_end(self):
        pass

    def on_test_epoch_start(self):
        super().on_test_epoch_start()
        if self.cfg.trinerflet_rep:
            self.geometry.encoding.encoding.enable_cache = True
            self.geometry.encoding.encoding.reset_cahce()
            self.geometry.encoding.encoding.set_double_mode(False)
            self.geometry.encoding.encoding.set_resolution_mode('low_res')
            with torch.cuda.amp.autocast(enabled=False):
                self.geometry.encoding.encoding.get_planes()

        if self.cfg.export_views:
            self.view_idx = 0

    def test_step(self, batch, batch_idx):
        out = self(batch)
        if self.cfg.export_views:
            idx = self.view_idx
        else:
            idx = batch['index'][0]
        fname = f"it{self.true_global_step}-test/{idx}.png"
        save_img_only = False
        if self.cfg.export_views:
            idx = self.view_idx
            self.view_idx = self.view_idx + 1

            #TODO keep only relevant pairs in batch to save
            batch_fname = f"it{self.true_global_step}-test/cameras/{idx}.pt"
            fname = f"it{self.true_global_step}-test/images/{idx}.png"
            torch.save(batch,self.get_save_path(batch_fname))


        self.save_image_grid(
            fname,
            [
                {
                    "type": "rgb",
                    "img": out["comp_rgb"][0],
                    "kwargs": {"data_format": "HWC"},
                },
            ]
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out
                else []
            )
            + [
                {
                    "type": "grayscale",
                    "img": out["opacity"][0, :, :, 0],
                    "kwargs": {"cmap": None, "data_range": (0, 1)},
                },
            ],
            name="test_step",
            step=self.true_global_step,
        )

    def on_test_epoch_end(self):
        if self.cfg.export_views:
            mdata_fname = f"it{self.true_global_step}-test/metadata.json"
            mdata = {'num_views' : self.view_idx}
            with open(self.get_save_path(mdata_fname),'w') as f:
                json.dump(mdata,f)
            return

        self.save_img_sequence(
            f"it{self.true_global_step}-test",
            f"it{self.true_global_step}-test",
            "(\d+)\.png",
            save_format="mp4",
            fps=30,
            name="test",
            step=self.true_global_step,
        )

