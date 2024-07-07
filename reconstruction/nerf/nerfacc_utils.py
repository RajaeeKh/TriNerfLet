import torch
from torch import Tensor
import copy
import torch.nn.functional as F
import nerfacc
from typing import Callable, Dict, Optional, Tuple

from nerfacc.volrend import (
    # accumulate_along_rays_,
    # render_weight_from_density,
    # rendering,
    render_weight_from_density,
    accumulate_along_rays
)

NERFACC_SETTINGS = {
    'grid_resolution' : 128,
    'grid_nlvl' : 4,

    'far_plane' : 1e10,

    # 'render_step_size' : 1e-3,
    # 'eval_chunk_size' : 100000,
    # 'alpha_thre' : 1e-2,
    # 'cone_angle' : 0.004

    'render_step_size': 1e-3,
    'eval_chunk_size': 10000,
    'alpha_thre': 0.0,
    'cone_angle': 0.004
    # 'cone_angle': 0.002

    # 'render_step_size' : 5e-3,
    # 'eval_chunk_size' : 20000,
    # 'alpha_thre' : 0.0,
    # 'cone_angle' : 0.0
}

#clonned from: https://github.com/nerfstudio-project/nerfacc/blob/master/nerfacc/volrend.py
def rendering(
    # ray marching results
    t_starts: Tensor,
    t_ends: Tensor,
    ray_indices: Optional[Tensor] = None,
    n_rays: Optional[int] = None,
    # radiance field
    rgb_sigma_fn: Optional[Callable] = None,
    rgb_alpha_fn: Optional[Callable] = None,
    # rendering options
    render_bkgd: Optional[Tensor] = None,
    expected_depths: bool = True,
):
    """Render the rays through the radience field defined by `rgb_sigma_fn`.

    This function is differentiable to the outputs of `rgb_sigma_fn` so it can
    be used for gradient-based optimization. It supports both batched and flattened input tensor.
    For flattened input tensor, both `ray_indices` and `n_rays` should be provided.


    Note:
        Either `rgb_sigma_fn` or `rgb_alpha_fn` should be provided.

    Warning:
        This function is not differentiable to `t_starts`, `t_ends` and `ray_indices`.

    Args:
        t_starts: Per-sample start distance. Tensor with shape (n_rays, n_samples) or (all_samples,).
        t_ends: Per-sample end distance. Tensor with shape (n_rays, n_samples) or (all_samples,).
        ray_indices: Ray indices of the flattened samples. LongTensor with shape (all_samples).
        n_rays: Number of rays. Only useful when `ray_indices` is provided.
        rgb_sigma_fn: A function that takes in samples {t_starts, t_ends,
            ray indices} and returns the post-activation rgb (..., 3) and density
            values (...,). The shape `...` is the same as the shape of `t_starts`.
        rgb_alpha_fn: A function that takes in samples {t_starts, t_ends,
            ray indices} and returns the post-activation rgb (..., 3) and opacity
            values (...,). The shape `...` is the same as the shape of `t_starts`.
        render_bkgd: Background color. Tensor with shape (3,).
        expected_depths: If True, return the expected depths. Else, the accumulated depth is returned.

    Returns:
        Ray colors (n_rays, 3), opacities (n_rays, 1), depths (n_rays, 1) and a dict
        containing extra intermediate results (e.g., "weights", "trans", "alphas")

    Examples:


    """
    if ray_indices is not None:
        assert (
            t_starts.shape == t_ends.shape == ray_indices.shape
        ), "Since nerfacc 0.5.0, t_starts, t_ends and ray_indices must have the same shape (N,). "

    if rgb_sigma_fn is None and rgb_alpha_fn is None:
        raise ValueError(
            "At least one of `rgb_sigma_fn` and `rgb_alpha_fn` should be specified."
        )

    # Query sigma/alpha and color with gradients
    if rgb_sigma_fn is not None:
        rgbs, sigmas = rgb_sigma_fn(t_starts, t_ends, ray_indices)
        # if t_starts.shape[0] != 0:
        #     rgbs, sigmas = rgb_sigma_fn(t_starts, t_ends, ray_indices)
        # else:
        #     rgbs = torch.empty((0, 3), device=t_starts.device)
        #     sigmas = torch.empty((0,), device=t_starts.device)
        assert rgbs.shape[-1] == 3, "rgbs must have 3 channels, got {}".format(
            rgbs.shape
        )
        assert (
            sigmas.shape == t_starts.shape
        ), "sigmas must have shape of (N,)! Got {}".format(sigmas.shape)
        # Rendering: compute weights.
        weights, trans, alphas = render_weight_from_density(
            t_starts,
            t_ends,
            sigmas,
            ray_indices=ray_indices,
            n_rays=n_rays,
        )
        extras = {
            "weights": weights,
            "alphas": alphas,
            "trans": trans,
            "sigmas": sigmas,
            "rgbs": rgbs,
        }
    elif rgb_alpha_fn is not None:
        rgbs, alphas = rgb_alpha_fn(t_starts, t_ends, ray_indices)
        # if t_starts.shape[0] != 0:
        #     rgbs, alphas = rgb_alpha_fn(t_starts, t_ends, ray_indices)
        # else:
        #     rgbs = torch.empty((0, 3), device=t_starts.device)
        #     alphas = torch.empty((0,), device=t_starts.device)
        assert rgbs.shape[-1] == 3, "rgbs must have 3 channels, got {}".format(
            rgbs.shape
        )
        assert (
            alphas.shape == t_starts.shape
        ), "alphas must have shape of (N,)! Got {}".format(alphas.shape)
        # Rendering: compute weights.
        weights, trans = render_weight_from_alpha(
            alphas,
            ray_indices=ray_indices,
            n_rays=n_rays,
        )
        extras = {
            "weights": weights,
            "trans": trans,
            "rgbs": rgbs,
            "alphas": alphas,
        }

    # Rendering: accumulate rgbs, opacities, and depths along the rays.
    colors = accumulate_along_rays(
        weights, values=rgbs, ray_indices=ray_indices, n_rays=n_rays
    )
    opacities = accumulate_along_rays(
        weights, values=None, ray_indices=ray_indices, n_rays=n_rays
    )
    depths = accumulate_along_rays(
        weights,
        values=(t_starts + t_ends)[..., None] / 2.0,
        ray_indices=ray_indices,
        n_rays=n_rays,
    )
    # print(depths.shape)
    # print(t_starts.shape)
    # print(ray_indices.shape)
    # print(depths[ray_indices].shape)
    z_variance = accumulate_along_rays(
        weights,
        values=((t_starts + t_ends)[..., None] / 2.0 - depths[ray_indices]).pow(2),
        ray_indices=ray_indices,
        n_rays=n_rays,
    )
    if expected_depths:
        depths = depths / opacities.clamp_min(torch.finfo(rgbs.dtype).eps)

    # Background composition.
    if render_bkgd is not None:
        colors = colors + render_bkgd * (1.0 - opacities)

    return colors, opacities, depths, z_variance, extras




def get_estimator(bound):
    aabb = torch.tensor([-1.0 * bound, -1.0 * bound, -1.0 * bound, 1.0 * bound, 1.0 * bound, 1.0 * bound])
    estimator = nerfacc.OccGridEstimator(
        roi_aabb=aabb, resolution=NERFACC_SETTINGS['grid_resolution'], levels=NERFACC_SETTINGS['grid_nlvl']
    )
    return estimator


class NerfAccRenderer:
    def __init__(self,opt,device,estimator):
        self.opt = copy.copy(opt)
        self.device = device
        bound = opt.bound
        self.bound = bound
        # aabb = torch.tensor([-1.0*bound, -1.0*bound, -1.0*bound, 1.0*bound, 1.0*bound, 1.0*bound], device=device)
        # self.estimator = nerfacc.OccGridEstimator(
        #     roi_aabb=aabb, resolution=NERFACC_SETTINGS['grid_resolution'], levels=NERFACC_SETTINGS['grid_nlvl']
        # ).to(device)
        # self.estimator.train()

        self.estimator = estimator
        self.render_step_size = NERFACC_SETTINGS['render_step_size']

    def _render(self,radiance_field,rays_o,rays_d,bg_color,perturb,chunk_size = -1):
        assert rays_o.shape == rays_d.shape
        rays_o_original_shape = rays_o.shape

        rays_o_all = rays_o.view(-1,rays_o_original_shape[-1])
        rays_d_all = rays_d.view(-1,rays_d.shape[-1])
        rays_d_all = F.normalize(rays_d_all,dim=-1)
        bg_color_all = bg_color.view(-1,bg_color.shape[-1])
        num_rays = rays_o_all.shape[0]
        if chunk_size < 1:
            chunk_size = num_rays
        results = []
        # print('rays_o_all.shape = {}'.format(rays_o_all.shape))
        for i in range(0, num_rays, chunk_size):
            # print('i = {}'.format(i))
            # print('chunk_size = {}'.format(chunk_size))
            # print('num_rays = {}'.format(num_rays))
            rays_o = rays_o_all[i:i + chunk_size]
            rays_d = rays_d_all[i:i + chunk_size]
            bg_color = bg_color_all
            if bg_color_all.shape[0] > 1:
                bg_color = bg_color_all[i:i + chunk_size]
            def sigma_fn(t_starts, t_ends, ray_indices):
                if t_starts.shape[0] == 0:
                    sigmas = torch.empty((0, 1), device=t_starts.device)
                else:
                    t_origins = rays_o[ray_indices]
                    t_dirs = rays_d[ray_indices]
                    positions = (
                            t_origins + t_dirs * (t_starts + t_ends)[:, None] / 2.0
                    )
                    assert len(positions.shape) == 2
                    # print('t_starts.shape: {}'.format(t_starts.shape))
                    # print('positions.shape: {}'.format(positions.shape))
                    sigmas = radiance_field.density(positions)['sigma']
                    # print('sigmas.shape: {}'.format(sigmas.shape))
                return sigmas.squeeze(-1)

            def rgb_sigma_fn(t_starts, t_ends, ray_indices):
                if t_starts.shape[0] == 0:
                    rgbs = torch.empty((0, 3), device=t_starts.device)
                    sigmas = torch.empty((0, 1), device=t_starts.device)
                else:
                    t_origins = rays_o[ray_indices]
                    t_dirs = rays_d[ray_indices]
                    positions = (
                            t_origins + t_dirs * (t_starts + t_ends)[:, None] / 2.0
                    )
                    assert len(positions.shape) == 2
                    assert len(t_dirs.shape) == 2
                    sigmas,rgbs = radiance_field(positions, t_dirs)
                return rgbs, sigmas.squeeze(-1)

            # print('rays_o.shape: {}'.format(rays_o.shape))
            ray_indices, t_starts, t_ends = self.estimator.sampling(
                rays_o,
                rays_d,
                sigma_fn=sigma_fn,
                near_plane=self.opt.min_near,
                far_plane=NERFACC_SETTINGS['far_plane'],
                render_step_size=NERFACC_SETTINGS['render_step_size'],
                stratified=perturb,
                cone_angle=NERFACC_SETTINGS['cone_angle'],
                alpha_thre=NERFACC_SETTINGS['alpha_thre'],
            )
            # print('ray_indices.shape = {}'.format(ray_indices.shape))
            # print('bg_color.shape = {}'.format(bg_color.shape))
            rgb, opacity, depth, z_variance, extras = rendering(
                t_starts,
                t_ends,
                ray_indices,
                n_rays=rays_o.shape[0],
                rgb_sigma_fn=rgb_sigma_fn,
                render_bkgd=bg_color,
            )
            # print('rgb.shape: {}'.format(rgb.shape))
            chunk_results = [rgb, opacity, depth,z_variance, len(t_starts)]
            results.append(chunk_results)
        colors, opacities, depths, z_variance, n_rendering_samples = [
            torch.cat(r, dim=0) if isinstance(r[0], torch.Tensor) else r
            for r in zip(*results)
        ]
        res = {}
        res['image'] = colors.view(*rays_o_original_shape[:-1],-1)
        res['weights_sum'] = opacities.view(*rays_o_original_shape[:-1],-1)
        res['depth'] = depths.view(*rays_o_original_shape[:-1],-1)
        res['n_rendering_samples'] = sum(n_rendering_samples)
        res['z_variance'] = z_variance

        # print("res['image'].shape = {}".format(res['image'].shape))

        return res

    def render_train(self,radiance_field,rays_o,rays_d,bg_color):
        assert self.estimator.training
        assert radiance_field.training
        return self._render(radiance_field,rays_o,rays_d,bg_color,perturb=True,chunk_size=-1)

    def render_eval(self,radiance_field,rays_o,rays_d,bg_color):
        # self.estimator.eval()
        assert not self.estimator.training
        assert not radiance_field.training
        res = self._render(radiance_field,rays_o,rays_d,bg_color,perturb=False,chunk_size=NERFACC_SETTINGS['eval_chunk_size'])
        # self.estimator.train()
        return res

    def update(self,step,radiance_field):
        def occ_eval_fn(x):
            # print('x.shape: {}'.format(x.shape))
            density = radiance_field.density(x)['sigma'].view(-1,1)
            # print('density.shape: {}'.format(density.shape))
            return density * self.render_step_size

        # update occupancy grid
        self.estimator.update_every_n_steps(
            step=step,
            occ_eval_fn=occ_eval_fn,
            occ_thre=1e-2,
        )
           