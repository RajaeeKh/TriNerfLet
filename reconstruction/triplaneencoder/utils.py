import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections.abc import Iterable,Mapping
import math

# apple func recursively on all tensor elements
def batch_func(batch,func):
    if isinstance(batch, torch.Tensor):
        return func(batch)
    if isinstance(batch,Mapping):
        res = {}
        for key,val in batch.items():
            res[key] = batch_func(val,func)
        return res
    if isinstance(batch, Iterable):
        res = []
        for val in batch:
            res.append(batch_func(val, func))
        return res
    return batch


def merge_dicts(dict1,dict2):
    new_dict = copy.copy(dict1)
    for key,val in dict2.items():
        assert not (key in new_dict)
        new_dict[key] = val
    return new_dict

import contextlib

@contextlib.contextmanager
def dummy_context(**kwargs):
    yield


def dict_pad(x):
    max_len = max([len(val) for val in x.values()])
    for key,val in x.items():
        x[key] = val + [-1] * (max_len - len(val))
    return x

def get_triplane_sample_func(triplane_lst,hr_planes_lst = None,enable_caching=False):
    cache = None
    if enable_caching:
        cache = {'planes_lst' : {},'viewdir_lst' : {}}
    def sample_func(pts,viewdirs):
        N_rays, N_samples, ch = pts.shape
        assert ch == 3
        pts = pts.view(len(triplane_lst),-1,N_samples, ch)
        if viewdirs is not None:
            viewdirs = viewdirs.view(len(triplane_lst),-1,N_samples, ch)
        batch_sz_ray = pts.shape[1]
        sampled_features = []
        for idx,triplane in enumerate(triplane_lst):
            pts_item = pts[idx].view(-1,ch)
            viewdirs_item = None
            if viewdirs is not None:
                viewdirs_item = viewdirs[idx].view(-1,ch)
            hr_plane = None
            if hr_planes_lst is not None:
                hr_plane = hr_planes_lst[idx]

            viewdir_plane = None
            if enable_caching:
                if hr_plane is None:
                    if idx in cache['planes_lst']:
                        hr_plane = cache['planes_lst'][idx]
                    else:
                        hr_plane = triplane.get_planes()
                        cache['planes_lst'][idx] = hr_plane
                if idx in cache['viewdir_lst']:
                    viewdir_plane = cache['viewdir_lst'][idx]
                else:
                    viewdir_plane = triplane.get_planes_viewdir()
                    cache['viewdir_lst'][idx] = viewdir_plane


            coded_pts = triplane.sample_from_planes(pts_item,hr_plane).view(pts_item.shape[0],-1)
            coded_viewdirs = triplane.sample_view_radiance(viewdirs_item,viewdir_plane)
            res = coded_pts
            if coded_viewdirs is not None:
                coded_viewdirs = coded_viewdirs.view(pts_item.shape[0],-1)
                res = torch.cat([coded_pts,coded_viewdirs],dim=1)
            sampled_features.append(res.view(batch_sz_ray,N_samples,-1))
        return torch.cat(sampled_features,dim=0)
    return sample_func



def get_triplane_density_gric_func(triplane_lst,grid_resolution,hr_planes_lst = None,enable_caching=False):
    if grid_resolution < 1:
        return None

    cache = None
    if enable_caching:
        cache = {}
    if triplane_lst[0].enable_grid_acc:
        enable_caching = True
        cache = {}
        grid_lst = []
        lbound_lst = []
        for idx, triplane in enumerate(triplane_lst):
            lbound,ggrid = triplane.get_grid()
            grid_lst.append(ggrid)
            lbound_lst.append(lbound)
        grid = torch.cat(grid_lst,dim=0)
        if len(triplane_lst) > 1:
            lbound = torch.tensor(lbound_lst).view(-1, 1, 1, 1, 1).to(device=grid.device, dtype=grid.dtype)
        else:
            lbound = lbound_lst[0]

        cache['grid_alpha'] = grid
        cache['lbound'] = lbound

    def density_grid_func(pts,nerf_alpha):
        N_rays, N_samples, ch = pts.shape
        assert ch == 3
        # with torch.no_grad():
        pts = pts.view(len(triplane_lst),1, -1, N_samples, ch)
        if (enable_caching == False) or (len(cache)==0):
            grid_features_lst = []
            lbound_lst = []
            for idx, triplane in enumerate(triplane_lst):
                hr_plane = None
                if hr_planes_lst is not None:
                    hr_plane = hr_planes_lst[idx]
                lbound, grid_features, grid = triplane.get_grid_features(grid_resolution,hr_plane)
                # grid_features = grid_features.permute(3,0,1,2)# C,D,H,W
                grid_features_lst.append(grid_features)
                lbound_lst.append(lbound)
            grid_features = torch.stack(grid_features_lst,dim=0)
            if len(triplane_lst) > 1:
                lbound = torch.tensor(lbound_lst).view(-1,1,1,1,1).to(device=pts.device,dtype=pts.dtype)
            else:
                lbound = lbound_lst[0]

            # grid_alpha = nerf.get_alpha(grid_features,mode='fine').permute(0,4,1,2,3)
            grid_features_shape = grid_features.shape
            grid_alpha = nerf_alpha(grid_features.view(-1,grid_features_shape[-2],grid_features_shape[-1])
                                    ).view(*grid_features_shape[:-1],-1).permute(0,4,1,2,3)
            if enable_caching:
                cache['grid_alpha'] = grid_alpha
                cache['lbound'] = lbound
        else:
            grid_alpha = cache['grid_alpha']
            lbound = cache['lbound']

        sampled_alpha = F.grid_sample(grid_alpha,pts/lbound,padding_mode="border",align_corners=True)
        sampled_alpha = sampled_alpha.permute(0,2,3,4,1).reshape(N_rays, N_samples,-1)
        return sampled_alpha
    return density_grid_func,grid_resolution







# for test uses
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires,input_dims = 3, i=0):
    if i == -1:
        return nn.Identity(), 3

    embed_kwargs = {
        'include_input': True,
        'input_dims': input_dims,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim
def get_freq_sample_function(tmp1=None,tmp2=None):
    embed_fn, input_ch = get_embedder(10) #63
    embeddirs_fn, input_ch_views = get_embedder(4) #27
    # print('hh')
    def sample_func(pts, viewdirs):
        N_rays, N_samples, ch = pts.shape
        # assert ch == 3
        coded_features = embed_fn(pts.view(-1,ch)).view(N_rays, N_samples,input_ch)
        if viewdirs is not None:
            viewdirs_features = embeddirs_fn(viewdirs.view(-1,ch)).view(N_rays, N_samples,input_ch_views)
            coded_features = torch.cat([coded_features,viewdirs_features],dim=-1)
        return coded_features
    return sample_func


def set_lr(optimizer,lr):
    for idx,g in enumerate(optimizer.param_groups):
        if isinstance(lr,list):
            g['lr'] = lr[idx]
        else:
            g['lr'] = lr


def calculate_psnr(mse,values_range = 1):
    psnr = 10 * torch.log10((values_range**2)/mse)
    return psnr


def calculate_final_pixel_values(res,rand_offset,additional_spp,keys_to_edit =  ['rgb', 'rgb0']):
    assert additional_spp > 0
    rand_offset = rand_offset.view(-1, additional_spp + 1, rand_offset.shape[-1])
    # rand_offset_dist = rand_offset.norm(dim=-1, keepdim=True)
    # max_dist = math.sqrt(2)
    # assert rand_offset_dist.max() <= max_dist
    # weights = 1 - rand_offset_dist / max_dist  # 1 is at the point, 0 is the farthest
    # weights = F.normalize(weights, dim=-2)

    rand_offset_dist = rand_offset.pow(2).sum(dim=-1,keepdim=True)
    sigma = 0.5
    weights = torch.exp(-0.5*rand_offset_dist/(sigma**2))


    for key in keys_to_edit:
        if key in res:
            val = res[key]
            dim = val.shape[-1]
            if len(val.shape) == 1:
                dim = 1
            val = val.view(-1, additional_spp + 1, dim)
            val = (val * weights).sum(dim=-2) / weights.sum(dim=-2) #TODO
            # val = val.mean(dim=-2) # regular mean
            res[key] = val
    return res




# def get_triplane_freq_hybrid_sample_func(triplane_lst,hr_planes_lst = None):
#     triplane_sample = get_triplane_sample_func(triplane_lst,hr_planes_lst = hr_planes_lst)
#     embed_fn, input_ch = get_embedder(6, 32)

def get_levels(upscale_factor):
    wavelet_levels = math.log2(upscale_factor)
    if (abs(wavelet_levels - round(wavelet_levels)) > 1e-5):
        raise ValueError('Unsupported res. should be 2^')
    wavelet_levels = round(wavelet_levels)
    return wavelet_levels

if __name__ == "__main__":
    embder = get_freq_sample_function()
    # x = torch.randn(10,100,8)
    # b = embder(x,None)
    # print(b.shape)
    embed_fn, input_ch = get_embedder(6,8)
    print(input_ch)