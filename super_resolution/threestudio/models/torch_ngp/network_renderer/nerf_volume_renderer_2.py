from dataclasses import dataclass, field
from functools import partial

import nerfacc
import torch
import torch.nn.functional as F

import threestudio
from threestudio.models.background.base import BaseBackground
from threestudio.models.estimators import ImportanceEstimator
from threestudio.models.geometry.base import BaseImplicitGeometry
from threestudio.models.materials.base import BaseMaterial
from threestudio.models.networks import create_network_with_input_encoding
from threestudio.models.renderers.base import VolumeRenderer
from threestudio.systems.utils import parse_optimizer, parse_scheduler_to_instance
from threestudio.utils.ops import chunk_batch, get_activation, validate_empty_rays
from threestudio.utils.typing import *

from threestudio.models.torch_ngp.network_renderer.network import NeRFNetwork

from typing import  Union

@threestudio.register("nerf-volume-renderer2")
class NeRFVolumeRenderer(VolumeRenderer):
    @dataclass
    class Config(VolumeRenderer.Config):
        #mlp
        encoding: str = 'triplane_wavelet'
        encoding_dir: str = 'sphere_harmonics'
        num_layers: int = 2
        hidden_dim: int = 64
        geo_feat_dim: int = 15
        num_layers_color: int = 3
        hidden_dim_color: int = 64


        #render
        bound: float = 1,
        cuda_ray: bool = True
        density_scale: float = 1
        min_near: float = 0.2
        density_thresh:float = 10
        bg_radius: float = -1
        grid_size: int = 128

        max_ray_batch: int = 4096
        dt_gamma: float = 0
        max_steps: int = 512

        update_extra_interval: int = 16
        bg_color: Tuple = (0.0, 0.0, 0.0)

        #trinerflet
        dir_degree: int = 4
        input_dim: int = 3

        triplane_channels: int = 16
        triplane_resolution:int = 2048
        triplane_wavelet_levels: int = 32
        triplane_low_res_scale: int = 4
        triplane_high_res_scale: int = 1

        init_ckpt: Union[str,None] = None



    cfg: Config

    def configure(
        self,
        # geometry: BaseImplicitGeometry,
        # material: BaseMaterial,
        # background: BaseBackground,
    ) -> None:
        # super().configure(geometry, material, background)
        self.model = NeRFNetwork(self.cfg)

        # if self.cfg.init_ckpt is not None:
        #     data = torch.load(self.cfg.init_ckpt,map_location='cpu')
        #     self.load_state_dict(data)
        # # elif self.cfg.cuda_ray:
        # #     with torch.cuda.amp.autocast(enabled=False):
        # #         self.model.update_extra_state()


    def forward(
        self,
        rays_o: Float[Tensor, "B H W 3"],
        rays_d: Float[Tensor, "B H W 3"],
        perturb=False,
        force_all_rays = False,
        staged = False,
        **kwargs
    ) -> Dict[str, Float[Tensor, "..."]]:
        batch_size, height, width = rays_o.shape[:3]
        rays_o_flatten: Float[Tensor, "batch_size Nr 3"] = rays_o.reshape(batch_size,-1, 3)
        rays_d_flatten: Float[Tensor, "batch_size Nr 3"] = rays_d.reshape(batch_size,-1, 3)
        n_rays = rays_o_flatten.shape[0]*rays_o_flatten.shape[1]

        bg_color = torch.tensor(self.cfg.bg_color).view(1,3).to(device=rays_o_flatten.device,
                                                                dtype=rays_o_flatten.dtype)
        outputs = self.model.render(rays_o_flatten, rays_d_flatten, staged=staged, bg_color=bg_color, perturb=perturb,
                                    force_all_rays=force_all_rays, max_ray_batch = self.cfg.max_ray_batch,
                                    dt_gamma = self.cfg.dt_gamma, max_steps = self.cfg.max_steps
                                    )
        res = {
            'comp_rgb':outputs['image'].view(batch_size, height, width, -1),
            'opacity' : outputs['weights_sum'].view(batch_size, height, width, -1),
            'depth' : outputs['depth'].view(batch_size, height, width, -1)
        }
        return res


    # def state_dict(self):
    #     model_st = self.model.state_dict()
    #     state = {}
    #     state['model'] = model_st
    #     if self.model.cuda_ray:
    #         state['mean_count'] = self.model.mean_count
    #         state['mean_density'] = self.model.mean_density
    #     return state
    #
    # def load_state_dict(self,state_dict, strict: bool = True):
    #     if self.model.cuda_ray:
    #         if 'mean_count' in state_dict:
    #             self.model.mean_count = state_dict['mean_count']
    #         if 'mean_density' in state_dict:
    #             self.model.mean_density = state_dict['mean_density']
    #
    #     missing_keys, unexpected_keys = self.model.load_state_dict(state_dict['model'], strict=False)
    #     threestudio.info("[INFO] loaded model.")
    #     if len(missing_keys) > 0:
    #         threestudio.info(f"[WARN] missing keys: {missing_keys}")
    #     if len(unexpected_keys) > 0:
    #         threestudio.info(f"[WARN] unexpected keys: {unexpected_keys}")



    def update_step(
        self, epoch: int, global_step: int, on_load_weights: bool = False
    ) -> None:
        #TODO: this should be at the begining of the training step
        if self.cfg.cuda_ray and global_step % self.cfg.update_extra_interval == 0:
            with torch.cuda.amp.autocast(enabled=False):
                self.model.update_extra_state()

    def update_step_end(self, epoch: int, global_step: int) -> None:
        pass

