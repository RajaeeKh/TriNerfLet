import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import threestudio.models.triplaneencoder.utils as utils
from threestudio.models.triplaneencoder.grid_backward import grid_sample

DEBUG_MODE = False

def cartesian_to_sperical_deg(x,y,z):
    y,z,x = x,y,z
    radius = torch.sqrt(x.pow(2) + y.pow(2) + z.pow(2))

    # phi = torch.arccos(z/radius.clamp(min=1e-6))
    # theta = torch.arcsin((y/(radius*torch.sin(phi)).clamp(min=1e-6)).clamp(min=-1,max=1))

    phi = torch.arccos(z/radius.clamp(min=1e-6))
    theta = torch.sign(y)*torch.arccos(x/torch.sqrt(x.pow(2) + y.pow(2)).clamp(min=1e-6))

    elevation = 90 - torch.rad2deg(phi)
    azimuth = torch.rad2deg(theta)
    if DEBUG_MODE:
        assert (elevation <= 90).all() and (elevation >= -90).all()

    return azimuth,elevation,radius

class TriPlaneVolume(torch.nn.Module):
    def __init__(self,number_of_features = 3,plane_resolution=224,init_sigma=0.1,lbound=1,
                 viewdir_plane_resolution = 32,
                 two_planes_per_axis=False
                 ,planes_features = None,viewdir_plane = None, apply_activation_on_features = False
                 ,inner_multi_res_scale = 1, inner_multi_res_viewdir_scale = 1,viewdir_mode = 'plane'
                 ,inner_multi_res_scale_current = 1
                 ,low_res_scale = 1
                 ,high_res_scale = 1
                 ,input_pts_in_unit_cube = True
                 ,wavelet_type = 'bior6.8'
                 ,wavelet_base_resolution=0
                 ,init_fn = None
                 ):
        super().__init__()
        self.number_of_features = number_of_features
        self.plane_resolution = plane_resolution
        self.init_sigma = init_sigma

        self.lbound = lbound
        self.input_pts_in_unit_cube = input_pts_in_unit_cube
        # self.lbound = nn.Parameter((torch.zeros(1)+lbound).clone().detach()) #TODO: remove
        self.lbound_viewdir = 1
        # self.lbound_viewdir = nn.Parameter((torch.zeros(1)+self.lbound_viewdir).clone().detach()) #TODO: remove
        self.n_output_dims = 3*self.number_of_features
        self.output_dim = self.n_output_dims
        self.n_input_dims = 3
        self.viewdir_plane_resolution = viewdir_plane_resolution
        self.two_planes_per_axis = two_planes_per_axis
        self.apply_activation_on_features = apply_activation_on_features

        plane_axes,plane_normals,plane_direction = self.create_subplanes_trivial_base(two_planes_per_axis = two_planes_per_axis)
        self.register_buffer('plane_axes', plane_axes.clone().detach())
        self.register_buffer('plane_normals', plane_normals.clone().detach())
        self.plane_direction = plane_direction
        if DEBUG_MODE:
            assert (torch.matmul(self.plane_axes.transpose(-2,-1), self.plane_axes) - torch.eye(self.plane_axes.shape[2]).unsqueeze(0)).abs().max() <= 1e-5

        self.inner_wavelet_scale = inner_multi_res_scale
        self.inner_wavelet_viewdir_scale = inner_multi_res_viewdir_scale
        self.inner_multi_res_scale_current = inner_multi_res_scale_current
        self.wavelet_base_resolution = wavelet_base_resolution
        assert self.inner_wavelet_scale >= self.inner_multi_res_scale_current

        self.low_res_scale = low_res_scale
        self.high_res_scale = high_res_scale
        assert low_res_scale >= high_res_scale
        self.double_resolution_mode = False
        self.current_resolution_mode = 'low_res'
        self.enable_cache = False
        self.wavelet_type = wavelet_type
        self.init_fn = init_fn

        self.init_plane_features(planes_features)


        self.enable_grid_acc = False








    def init_plane_features(self,planes_features):
        plane_resolution = self.plane_resolution
        self.last_used_planes = None
        if self.inner_wavelet_scale <= 1:
            if planes_features is None:
                planes_features = self.init_sigma * torch.randn(len(self.plane_direction), self.number_of_features,
                                                                plane_resolution, plane_resolution)
                if self.init_fn is not None:
                    planes_features = self.init_fn(planes_features)
            self.planes_features = nn.Parameter(planes_features.clone().detach())
            return

        from pytorch_wavelets import DWTForward, DWTInverse
        upscale_factor = self.inner_wavelet_scale
        wavelet_levels = utils.get_levels(upscale_factor)

        wavelet_coef_lst = []

        wave_type = self.wavelet_type
        print('************ selected wavelet: {} ************'.format(wave_type))
        pad_dict = {
            'bior6.8': 4,
            'bior2.6': 3,
            'bior4.4': 2,
            'bior2.2': 1,
            'haar': 0
        }
        pad = pad_dict[wave_type]

        # wave_type = 'bior2.6'
        # pad = 3
        # wave_type = 'bior6.8'
        # pad = 4

        levels = wavelet_levels
        xfm = DWTForward(J=1, wave=wave_type, mode='zero')
        self.idwt = DWTInverse(wave=wave_type, mode='zero')
        # self.xfm = xfm


        with torch.no_grad():
            yh = []
            tmp_wavelets = torch.ones(len(self.plane_direction), self.number_of_features,
                                      plane_resolution, plane_resolution)
            yl = tmp_wavelets
            for lvl in range(levels):
                yl, yh_tmp_lst = xfm(yl)
                yh_tmp = yh_tmp_lst[0]
                if (pad > 0) and (yl.shape[3] > self.wavelet_base_resolution):
                    yl = yl[..., pad:-pad, pad:-pad]
                    yh_tmp = yh_tmp_lst[0][..., pad:-pad, pad:-pad]
                yh.append(yh_tmp)
            yh = yh[::-1]
            self.planes_features_wavelet_yh_shapes = [val.shape for val in yh]
            self.planes_features_wavelet_yh_zeros = [torch.zeros_like(val) for val in yh]
            self.planes_features_wavelet_pad = pad
            print('wavelet levels: ')
            print(self.planes_features_wavelet_yh_shapes)



        if planes_features is None:
            planes_features = self.init_sigma*torch.randn_like(yl)
        self.planes_features = nn.Parameter(planes_features.clone().detach())

        self.planes_features_wavelet_current_level = utils.get_levels(self.inner_multi_res_scale_current)
        self.planes_features_wavelet_all_level = wavelet_levels
        for level_idx in range(wavelet_levels):
            if level_idx < (wavelet_levels - self.planes_features_wavelet_current_level):
                level_coefs = torch.zeros_like(yh[level_idx])
                wavelet_coef_lst.append(nn.Parameter(level_coefs.clone().detach()))

        self.planes_features_wavelet_coefs = nn.ParameterList(wavelet_coef_lst)

        # self.planes_features.requires_grad = False
        # for val in self.planes_features_wavelet_coefs:
        #     val.requires_grad = False
        # self.planes_features_wavelet_coefs[-1].requires_grad = True
        # self.planes_features_wavelet_coefs[-2].requires_grad = True
        # self.planes_features_wavelet_coefs[-3].requires_grad = True


        print('learnable_wavelets: ')
        print([val.shape for val in self.planes_features_wavelet_coefs if val.requires_grad])
        print('learned params')
        print([val.shape for val in self.parameters() if val.requires_grad])









    def get_wavelet_features(self):
        res = []
        if self.inner_wavelet_scale > 1:
            res = list(self.planes_features_wavelet_coefs)
        if self.enable_grid_acc:
            res += self.grid_wavelets
        return res



    def create_subplanes_trivial_base(self,two_planes_per_axis,dim=3):
        euclidean_base = torch.eye(dim)
        subplanes_basis = []
        normals = []
        normals_dir = []

        #up
        current_base1 =  euclidean_base[:, 0:1]  #x
        current_base2 =  euclidean_base[:, 2:] #z
        normal = euclidean_base[:, 1:2]  # y
        current_base = torch.cat([current_base1, current_base2], dim=1)
        subplanes_basis.append(current_base)
        normals.append(normal)
        normals_dir.append('up')


        # front
        current_base1 = euclidean_base[:, 0:1]  # x
        current_base2 = euclidean_base[:, 1:2]  # y
        normal = euclidean_base[:, 2:]  # z
        current_base = torch.cat([current_base1, current_base2], dim=1)
        subplanes_basis.append(current_base)
        normals.append(normal)
        normals_dir.append('front')


        # right
        current_base1 = euclidean_base[:, 1:2]  # y
        current_base2 = euclidean_base[:, 2:]  # z
        normal = euclidean_base[:, 0:1]  # x
        current_base = torch.cat([current_base1, current_base2], dim=1)
        subplanes_basis.append(current_base)
        normals.append(normal)
        normals_dir.append('right')



        subplanes_basis = torch.stack(subplanes_basis,dim=0)
        normals = torch.stack(normals,dim=0)
        return subplanes_basis,normals,normals_dir

    # def get_number_of_channels(self):
    #     return self.planes_features.shape[-3]

    @staticmethod
    def project_into_planes(planes,coords):
        if DEBUG_MODE:
            assert coords.shape[-1] == planes.shape[1]
        coords = coords.unsqueeze(-1).unsqueeze(1)
        planes = planes.unsqueeze(0)
        projected_coords = torch.matmul(planes.transpose(-1,-2),coords)
        return projected_coords.squeeze(-1)



    def sample_from_planes_aux(self,coordinates,plane_features,plane_axes,
                               lbound = 1,
                               mode='bilinear', padding_mode='border'):
        # plane_features : Np,C,H,W
        # plane_axes: Np,dim,dim-1
        # coordinates: N,dim
        projected_coords = TriPlaneVolume.project_into_planes(plane_axes,coordinates/lbound) # N,Np,dim-1
        # projected_normals = plane_normals.view(1,1,projected_coords.shape[2],plane_normals.shape[-1])
        projected_coords = projected_coords.transpose(0,1).unsqueeze(2)  # Np,N,1,dim-1
        # Np,C,N,1
        sampled_vals = F.grid_sample(plane_features, projected_coords, mode=mode, padding_mode=padding_mode, align_corners=True)
        # sampled_vals = grid_sample(plane_features,projected_coords,padding_mode=padding_mode,align_corners=True)
        sampled_vals = sampled_vals.permute(2,0,1,3)# N,Np,C,1
        sampled_vals = sampled_vals.squeeze(-1) # N,Np,C
        return sampled_vals

    def get_planes(self,max_res = -1,max_scale = -1):
        if self.last_used_planes is not None:
            # print('entered here')
            if self.double_resolution_mode:
                res = self.last_used_planes[self.current_resolution_mode]
            else:
                res =  self.last_used_planes['low_res']
            # print('pre here, {}'.format(res.shape))
            return res

        planes = self.planes_features
        if self.apply_activation_on_features:
            planes = F.tanh(planes)

        current_scale = 1
        x_low_res = None
        x_high_res = None
        low_res = self.plane_resolution / self.low_res_scale
        high_res = self.plane_resolution / self.high_res_scale
        # print('**************')
        if self.inner_wavelet_scale > 1:
            x = self.planes_features
            pad = self.planes_features_wavelet_pad
            # print('x ', x.requires_grad)
            # for wavelet_coefs in self.planes_features_wavelet_coefs:
            for level_idx in range(self.planes_features_wavelet_all_level):
                yl = 2*x
                # print('yl ', yl.requires_grad)
                if (min(x.shape[2:]) >= low_res) and (x_low_res is None):
                    x_low_res = x
                    if self.double_resolution_mode==False:
                        break
                if (min(x.shape[2:]) >= high_res) and (x_high_res is None):
                    x_high_res = x
                    break
                elif level_idx < len(self.planes_features_wavelet_coefs):
                    yh = self.planes_features_wavelet_coefs[level_idx]
                else:
                    print('heereeee')
                    yh = self.planes_features_wavelet_yh_zeros[level_idx].to(device = yl.device,dtype=yl.dtype)

                # print('yl ', yl.requires_grad)
                # print('yh ', yh.requires_grad)
                if yl.shape[3] >= self.wavelet_base_resolution:
                    yl = F.pad(yl, (pad, pad, pad, pad))
                    yh = F.pad(yh, (pad, pad, pad, pad))
                # print('yl ', yl.requires_grad)
                # print('yh ', yh.requires_grad)
                x = self.idwt((yl,[yh]))
                current_scale *= 2
                # print('x ' ,x.requires_grad)


            if self.apply_activation_on_features:
                # x = x.clamp(min=-1, max=1)
                x = F.tanh(x)

            # planes = x
            if (x_low_res is None):
                x_low_res = x
            if (x_high_res is None) and self.double_resolution_mode:
                x_high_res = x
            planes = x_low_res
            if self.double_resolution_mode and (self.current_resolution_mode == 'high_res'):
                planes = x_high_res
        if self.enable_cache:
            self.last_used_planes = {'low_res' : x_low_res,'high_res':x_high_res}

        # print('**************')
        # print('planes.shape = {}'.format(planes.shape))
        # if x_high_res is not None:
            # print('x_high_res.shape :{}'.format(x_high_res.shape))
        return planes

    def set_resolution_mode(self,val):
        assert val in ['low_res','high_res']
        self.current_resolution_mode = val
    def set_double_mode(self,val : bool):
        self.double_resolution_mode = val
    def reset_cahce(self):
        self.last_used_planes = None

    def sample_from_planes(self,coordinates,plane_features = None, lbound = None):
        if plane_features is None:
            plane_features = self.get_planes()
        plane_axes = self.plane_axes
        if lbound is None:
            lbound = self.lbound
        # print('coordinates.max = {}'.format(coordinates.max()))
        # print('coordinates.min = {}'.format(coordinates.min()))
        # print('plane_features.shape : {}'.format(plane_features.shape))
        # print('coordinates.shape = {}'.format(coordinates.shape))

        # assert coordinates.max() <= 1
        # assert coordinates.min() >= 0

        if self.input_pts_in_unit_cube:
            coordinates = (coordinates*2 - 1)*lbound
        # print('coordinates.max = {}'.format(coordinates.max()))
        # print('coordinates.min = {}'.format(coordinates.min()))
        sampled_vals = self.sample_from_planes_aux(coordinates,plane_features,plane_axes,lbound)
        return sampled_vals

    def get_grid_features(self,grid_res,plane_features=None,grid=None):
        if grid is None:
            grid_axis = torch.arange(grid_res)
            grid_x, grid_y, grid_z = torch.meshgrid(grid_axis, grid_axis,grid_axis, indexing='xy') #TODO: check correctness
            grid = torch.stack([grid_x,grid_y,grid_z],dim=-1)
            grid = grid / (grid_res-1)


        assert grid.max() <= 1
        assert grid.min() >= 0
        grid = 2*self.lbound*(grid) - self.lbound #[-lboud,lboud]
        grid = grid[..., [2, 0, 1]]

        if plane_features is None:
            plane_features = self.get_planes(2*grid_res)
            # plane_features = self.get_planes(grid_res) #TODO
            # print(plane_features.shape)
        grid = grid.to(device=plane_features.device,dtype=plane_features.dtype)
        grid_shape = grid.shape
        grid = grid.view(-1,grid.shape[-1])
        grid_features = self.sample_from_planes(grid,plane_features = plane_features)
        grid = grid.view(*grid_shape)
        grid_features = grid_features.view(*grid_shape[:-1],-1)

        # # grid_idx = grid.unsqueeze(0)
        # # grid_input = grid_idx.permute(0,4,1,2,3)
        # # test_grid_idx = torch.tensor([[-2,-2,-2],[-1,-1,-1],[0,0,0],[1,1,1],[2,2,2]]).view(1,1,1,-1,3).to(device=grid.device,dtype=grid.dtype)
        # test_grid_idx = torch.tensor([[-1, -1, -1],[-1,-1,1],[-1,1,-1],[-1,1,1],[1,-1,-1],[1,-1,1],[1,1,-1],[1,1,1]]).view(1, 1, 1, -1,
        #                                                                                                  3).to(
        #     device=grid.device, dtype=grid.dtype)
        # # tmp = F.grid_sample(grid_input,grid_idx[...,[2,1,0]]/self.lbound,align_corners=True)
        # grid_input = torch.arange(1,9).view(1,1,2,2,2).to(device=grid.device,dtype=grid.dtype)
        # tmp = F.grid_sample(grid_input,test_grid_idx,align_corners=True)
        # tmp2 = F.grid_sample(grid_input, test_grid_idx[...,[2,1,0]], align_corners=True)
        # print('hhh')

        return self.lbound,grid_features,grid




    def get_params(self,opt_cfg):
        return self.parameters()
        # params = []
        # lr = opt_cfg['lr']
        # lr_planes = opt_cfg['lr_planes'] if (opt_cfg['lr_planes'] is not None) else lr
        # params += [{'params': val, 'lr': lr_planes} for val in self.parameters()]
        # return params


    def forward(self,coordinates,bound = None):
        # print('coordinates.shape = {}'.format(coordinates.shape))
        # # print('bound = {}'.format(bound))
        # print((coordinates.abs().max(dim=-1).values <= 1).sum() / coordinates.shape[0])

        # with torch.cuda.amp.autocast(enabled=False): #TODO: may need to change it
        sampled_vals = self.sample_from_planes(coordinates,lbound = bound)
        # sampled_view_radiance = self.sample_view_radiance(viewdir)
        # res = sampled_vals.view(sampled_vals.shape[0],-1) if (sampled_vals.numel() > 0) else sampled_vals

        if (sampled_vals.numel() > 0):
            res = sampled_vals.view(sampled_vals.shape[0], -1)
        else:
            res = torch.zeros(0,self.number_of_features*3,device=coordinates.device,dtype=coordinates.dtype)

        return res





def kplanes_init_mul(x):
    print('************************* kplanes_init called ***********************')
    return 2*torch.rand_like(x) - 1
class KPlaneVolume(torch.nn.Module):
    def __init__(self,base_resolution,levels,channels,features_mode,func_init = False):
        super().__init__()
        assert levels >= 1
        assert features_mode in ['mul','concatination']
        self.features_mode = features_mode
        triplane_lst = []
        for current_lvl in range(levels):
            current_res = base_resolution*(2**current_lvl)
            current_tri = TriPlaneVolume(
            number_of_features=channels,
            plane_resolution=current_res,
            init_sigma = 0.1,
            lbound=1,
            viewdir_plane_resolution=-1,
            apply_activation_on_features=False,
            inner_multi_res_scale=1,
            inner_multi_res_scale_current=1,
            low_res_scale=1,
            high_res_scale=1,
            wavelet_type='haar',
            wavelet_base_resolution=current_res,
            init_fn = (None if (self.features_mode == 'concatination' and func_init==False) else kplanes_init_mul)
            )
            triplane_lst.append(current_tri)
            print(current_tri.planes_features.shape)
        self.triplane_lst = nn.ModuleList(triplane_lst)
        self.n_output_dims = levels*channels*(3 if self.features_mode == 'concatination' else 1)
        self.output_dim = self.n_output_dims
        self.n_input_dims = 3
        self.channels = channels
        if self.features_mode == 'concatination':
            assert self.n_output_dims == sum([tri.n_output_dims for tri in self.triplane_lst])

    def forward(self, coordinates, bound=None):
        res_lst = []
        for tri in self.triplane_lst:
            tmp_res = tri(coordinates, bound=bound)
            if self.features_mode == 'mul':
                tmp_res = tmp_res.view(tmp_res.shape[0],3,self.channels)
                tmp_res = tmp_res[:,0] * tmp_res[:,1] * tmp_res[:,2]
            res_lst.append(tmp_res)
        res = torch.cat(res_lst,dim=-1)
        # print(res.shape)
        return res

class MultiscaleKPlaneVolume(torch.nn.Module):
    def __init__(self,base_resolution,low_res_levels,high_res_levels,channels,features_mode):
        super().__init__()
        assert high_res_levels>=low_res_levels
        self.low_res_vol = KPlaneVolume(base_resolution,low_res_levels,channels,features_mode)
        high_res_base_resolution = base_resolution*(2**low_res_levels)
        self.high_res_vol = KPlaneVolume(high_res_base_resolution,high_res_levels-low_res_levels,channels,features_mode)
        self.enable_cache = True
        self.double_mode = False
        self.resolution_mode = 'low_res'

        self.n_output_dims = self.low_res_vol.n_output_dims
        self.n_output_dims_high_res = self.low_res_vol.n_output_dims + self.high_res_vol.n_output_dims
        # self.output_dim = self.n_output_dims
        self.n_input_dims = 3

    def reset_cahce(self):
        pass

    def set_double_mode(self,val):
        self.double_mode = val

    def set_resolution_mode(self,val):
        assert val in ['low_res','high_res']
        self.resolution_mode = val

    def get_planes(self):
        return torch.zeros(1,3,50,50)

    def get_wavelet_features(self):
        return []

    def forward(self, coordinates, bound=None):
        res = self.low_res_vol(coordinates, bound=bound)
        if self.double_mode and (self.resolution_mode == 'high_res'):
            high_resolution_feat = self.high_res_vol(coordinates, bound=bound)
            res = torch.cat([res,high_resolution_feat],dim=-1)
        return res

class MultiscaleKPlaneMulVolume(torch.nn.Module):
    def __init__(self,base_resolution,low_res_levels,high_res_levels,channels,features_mode):
        super().__init__()
        assert high_res_levels>=low_res_levels
        self.low_res_vol = KPlaneVolume(base_resolution,low_res_levels,
                                        channels,features_mode,func_init=True)
        high_res_base_resolution = base_resolution*(2**low_res_levels)
        self.high_res_vol = KPlaneVolume(high_res_base_resolution,high_res_levels-low_res_levels,
                                         channels,features_mode,func_init=True)
        self.enable_cache = True
        self.double_mode = False
        self.resolution_mode = 'low_res'

        self.n_output_dims = channels * 3
        # self.n_output_dims_high_res = self.low_res_vol.n_output_dims + self.high_res_vol.n_output_dims
        # self.output_dim = self.n_output_dims
        self.n_input_dims = 3

    def reset_cahce(self):
        pass

    def set_double_mode(self,val):
        self.double_mode = val

    def set_resolution_mode(self,val):
        assert val in ['low_res','high_res']
        self.resolution_mode = val

    def get_planes(self):
        return torch.zeros(1,3,50,50)

    def get_wavelet_features(self):
        return []

    def mul_tensor(self,x):
        res = x[...,0,:]
        for i in range(1,x.shape[-2]):
            res = res * x[...,i,:]
        return res

    def forward(self, coordinates, bound=None):
        res = self.low_res_vol(coordinates, bound=bound)
        res = res.view(res.shape[0],-1,self.n_output_dims)
        res = self.mul_tensor(res)
        if self.double_mode and (self.resolution_mode == 'high_res'):
            high_resolution_feat = self.high_res_vol(coordinates, bound=bound)
            high_resolution_feat = high_resolution_feat.view(res.shape[0], -1, self.n_output_dims)
            high_resolution_feat = self.mul_tensor(high_resolution_feat)
            res = res * high_resolution_feat
        return res


if __name__ == "__main__":
    import json
    with open('configs/train_config.json') as file:
        cfg = json.load(file)
    model = TriPlaneNerf(cfg['Nerf'])
    pts = torch.randn(100,3)
    viewdirs = torch.rand(100,3)
    res = model(pts,viewdirs)
