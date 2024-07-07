import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import triplaneencoder.utils as utils

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
                 ,learn_rotation_axis = False
                 ,dropout = 0
                 ,wavelet_type = 'bior6.8'
                 ,lbound_auto_scale = False
                 ,upscale_ratio_bound = -1
                 ,upscale_levels = 2
                 ,wavelet_base_resolution = 0
                 ):
        super().__init__()
        self.number_of_features = number_of_features
        self.plane_resolution = plane_resolution
        self.init_sigma = init_sigma

        self.lbound = lbound
        self.lbound_viewdir = 1
        self.output_dim = 3*self.number_of_features
        self.viewdir_plane_resolution = viewdir_plane_resolution
        self.two_planes_per_axis = two_planes_per_axis
        self.apply_activation_on_features = apply_activation_on_features

        plane_axes,plane_normals,plane_direction = self.create_subplanes_trivial_base(two_planes_per_axis = two_planes_per_axis)
        self.register_buffer('plane_axes', plane_axes.clone().detach())
        self.register_buffer('plane_normals', plane_normals.clone().detach())
        self.plane_direction = plane_direction
        if DEBUG_MODE:
            assert (torch.matmul(self.plane_axes.transpose(-2,-1), self.plane_axes) - torch.eye(self.plane_axes.shape[2]).unsqueeze(0)).abs().max() <= 1e-5

        self.wavelet_type = wavelet_type

        self.inner_wavelet_scale = inner_multi_res_scale
        self.inner_wavelet_viewdir_scale = inner_multi_res_viewdir_scale
        self.inner_multi_res_scale_current = inner_multi_res_scale_current
        self.wavelet_base_resolution = wavelet_base_resolution
        assert self.inner_wavelet_scale >= self.inner_multi_res_scale_current

        self.init_plane_features(planes_features)


        self.learn_rotation_axis = learn_rotation_axis
        self.rotation_matrix = None
        if self.learn_rotation_axis:
            self.rotation_matrix = nn.Parameter(torch.randn(number_of_features,3,3))
            self.register_buffer('eye_matrix',torch.eye(3).unsqueeze(0))

        self.dropout = None
        if (dropout > 0) and (dropout < 1):
            print('********* dropout enable with p={} ****************'.format(dropout))
            self.dropout = nn.Dropout(dropout)

        self.lbound_auto_scale = lbound_auto_scale
        self.lbound_scale = None
        if self.lbound_auto_scale:
            print('******************** lbound_auto_scale enabled *************************')
            # self.lbound_scale = nn.Parameter(0.1*torch.randn(3))
            self.lbound_scale = nn.Parameter(0.5 * torch.ones(3))
            # self.lbound_scale = nn.Parameter(2 * torch.zeros(3))

        self.upscale_ratio_bound = upscale_ratio_bound
        self.upscale_levels = upscale_levels
        self.init_upscale()



    def init_upscale(self):
        self.upscale_enabled = False
        if (self.upscale_ratio_bound) > 0 and (self.upscale_ratio_bound < 1):
            assert self.upscale_levels > 0
            print('********** Upscale enabled *************')
            self.upscale_enabled = True
            plane_resolution = self.plane_resolution

            upscale_wavelet_lst = []
            upscale_base_resolution_lst = []
            upscale_base_corner_lst = []
            upscale_bound_ratio_lst = []
            for level in range(self.upscale_levels):
                upscale_base_resolution = round(plane_resolution*self.upscale_ratio_bound)
                assert (plane_resolution % upscale_base_resolution == 0)
                upscale_base_corner = round(plane_resolution / 2 - upscale_base_resolution / 2)

                plane_resolution = 2*upscale_base_resolution
                ratio_bound = self.upscale_ratio_bound ** (level+1)
                upscale_bound_ratio_lst.append(ratio_bound)

                upscale_base_resolution_lst.append(upscale_base_resolution)
                upscale_base_corner_lst.append(upscale_base_corner)
                current_dim = upscale_base_resolution
                current_wavelet = nn.Parameter(torch.zeros(3, self.number_of_features, 3, current_dim, current_dim))
                upscale_wavelet_lst.append(current_wavelet)

            self.upscale_wavelet_lst = nn.ParameterList(upscale_wavelet_lst)
            self.upscale_base_resolution_lst = upscale_base_resolution_lst
            self.upscale_base_corner_lst = upscale_base_corner_lst
            self.upscale_bound_ratio_lst = upscale_bound_ratio_lst
            print('****************************** ratio bound **********************')
            print(self.upscale_bound_ratio_lst)





    def get_params2(self, lr):
        params = self.named_parameters()
        res_1 = []
        res_2 = []
        for kv in params:
            if 'lbound_scale' in kv[0]:
                res_1.append(kv[1])
                print('**************************')
                print(kv[0])
            else:
                res_2.append(kv[1])

        params = [
            {'params': res_1, 'lr': 10*lr},
            {'params': res_2, 'lr': lr},
        ]
        # print(len(params))
        return params



    def init_plane_features(self,planes_features):
        plane_resolution = self.plane_resolution
        init_sigma = self.init_sigma
        # init_sigma = 0.001
        self.last_used_planes = None
        if self.inner_wavelet_scale <= 1:
            if planes_features is None:
                planes_features = init_sigma * torch.randn(len(self.plane_direction), self.number_of_features,
                                                                plane_resolution, plane_resolution)
            self.planes_features = nn.Parameter(planes_features.clone().detach())
            return

        from pytorch_wavelets import DWTForward, DWTInverse
        upscale_factor = self.inner_wavelet_scale
        wavelet_levels = utils.get_levels(upscale_factor)

        wavelet_coef_lst = []
        wave_type = self.wavelet_type
        print('************ selected wavelet: {} ************'.format(wave_type))
        pad_dict = {
            'bior6.8' : 4,
            'bior2.6' : 3,
            'bior4.4' : 2,
            'bior2.2' : 1,
            'haar' : 0
        }

        pad = pad_dict[wave_type]
        levels = wavelet_levels
        xfm = DWTForward(J=1, wave=wave_type, mode='zero')
        self.idwt = DWTInverse(wave=wave_type, mode='zero')


        with torch.no_grad():
            yh = []
            tmp_wavelets = torch.ones(len(self.plane_direction), self.number_of_features,
                                      plane_resolution, plane_resolution)
            yl = tmp_wavelets
            for lvl in range(levels):
                yl, yh_tmp_lst = xfm(yl)
                if (pad > 0) and (yl.shape[3] > self.wavelet_base_resolution):
                    yl = yl[..., pad:-pad, pad:-pad]
                    yh_tmp = yh_tmp_lst[0][..., pad:-pad, pad:-pad]
                else:
                    yh_tmp = yh_tmp_lst[0]
                yh.append(yh_tmp)
            yh = yh[::-1]
            self.planes_features_wavelet_yh_shapes = [val.shape for val in yh]
            self.planes_features_wavelet_yh_zeros = [torch.zeros_like(val) for val in yh]
            self.planes_features_wavelet_pad = pad
            print('wavelet levels: ')
            print(self.planes_features_wavelet_yh_shapes)



        if planes_features is None:
            # planes_features = self.init_sigma*torch.randn_like(yl)
            planes_features = init_sigma * torch.randn_like(yl)
            # planes_features = torch.zeros_like(yl)
        self.planes_features = nn.Parameter(planes_features.clone().detach())

        self.planes_features_wavelet_current_level = utils.get_levels(self.inner_multi_res_scale_current)
        self.planes_features_wavelet_all_level = wavelet_levels
        for level_idx in range(wavelet_levels):
            if level_idx < (wavelet_levels - self.planes_features_wavelet_current_level):
                level_coefs = torch.zeros_like(yh[level_idx])
                wavelet_coef_lst.append(nn.Parameter(level_coefs.clone().detach()))

        self.planes_features_wavelet_coefs = nn.ParameterList(wavelet_coef_lst)



        print('learnable_wavelets: ')
        print([val.shape for val in self.planes_features_wavelet_coefs if val.requires_grad])
        print('learned params')
        print([val.shape for val in self.parameters() if val.requires_grad])





    def get_wavelet_features(self):
        res = []
        if self.inner_wavelet_scale > 1:
            res = list(self.planes_features_wavelet_coefs)
        return res

    def get_wavelet_features_upscaled(self):
        res = []
        if self.upscale_enabled:
            res = self.upscale_wavelet_lst
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



    @staticmethod
    def project_into_planes(planes,coords):
        if DEBUG_MODE:
            assert coords.shape[-1] == planes.shape[1]
        coords = coords.unsqueeze(-1).unsqueeze(1)
        planes = planes.unsqueeze(0)
        projected_coords = torch.matmul(planes.transpose(-1,-2),coords)
        return projected_coords.squeeze(-1)



    def get_lbound_scale(self):
        if self.lbound_scale is None:
            return None
        # lbound_scale = (self.lbound_scale.pow(2)+1) # [1,inf], we only want to zoom in
        # lbound_scale = (self.lbound_scale.abs() + 1)  # [1,inf], we only want to zoom in
        # lbound_scale = (torch.exp(self.lbound_scale) + 1)
        # lbound_scale = (torch.exp(self.lbound_scale.pow(2)))
        lbound_scale = (torch.exp(self.lbound_scale.abs()))
        return lbound_scale

    def sample_from_planes_aux(self,coordinates,plane_features,plane_axes,
                               lbound = 1,
                               mode='bilinear', padding_mode='border'):
        # plane_features : Np,C,H,W
        # plane_axes: Np,dim,dim-1
        # coordinates: N,dim
        projected_coords = TriPlaneVolume.project_into_planes(plane_axes,coordinates/lbound) # N,Np,dim-1
        # projected_normals = plane_normals.view(1,1,projected_coords.shape[2],plane_normals.shape[-1])
        projected_coords = projected_coords.transpose(0,1).unsqueeze(2)  # Np,N,1,dim-1

        if self.lbound_auto_scale:
            lbound_scale = self.get_lbound_scale()
            projected_coords = (projected_coords*lbound_scale.view(-1,1,1,1)).clamp(-1,1)

        # Np,C,N,1
        sampled_vals = F.grid_sample(plane_features, projected_coords, mode=mode, padding_mode=padding_mode, align_corners=True)
        sampled_vals = sampled_vals.permute(2,0,1,3)# N,Np,C,1
        sampled_vals = sampled_vals.squeeze(-1) # N,Np,C
        return sampled_vals


    def sample_from_planes_aux_rotation(self, coordinates, plane_features, plane_axes,
                               lbound=1,
                               mode='bilinear', padding_mode='border'):
        # plane_features : Np,C,H,W
        # plane_axes: Np,dim,dim-1
        # coordinates: N,dim
        Np,C,H,W = plane_features.shape
        dim = plane_axes.shape[1]
        #TODO: learn parameteric rotation matrix instaed of this dirty heck
        rotation_matrix = torch.matmul(self.rotation_matrix.transpose(1,2),self.rotation_matrix) + 1e-6*self.eye_matrix
        #make it basis
        rotation_matrix , tmp = torch.linalg.qr(rotation_matrix) # C,dim,dim
        plane_axes = torch.matmul(rotation_matrix.unsqueeze(1),plane_axes.unsqueeze(0)) #C,Np,dim,dim-1
        plane_axes = plane_axes.transpose(0,1) #Np,C,dim,dim-1
        plane_axes = plane_axes.reshape(-1,dim,dim-1) #Np*C,dim,dim-1


        projected_coords = TriPlaneVolume.project_into_planes(plane_axes, coordinates / lbound)  # N,Np*C,dim-1
        projected_coords = projected_coords.transpose(0,1).unsqueeze(2)  # Np*C,N,1,dim-1

        plane_features = plane_features.view(-1,1,H,W) # Np*C,1,H,W
        # Np*C,1,N,1
        sampled_vals = F.grid_sample(plane_features, projected_coords, mode=mode, padding_mode=padding_mode,
                                     align_corners=True)
        sampled_vals = sampled_vals.view(Np,C,sampled_vals.shape[2],sampled_vals.shape[3]) # Np,C,N,1
        sampled_vals = sampled_vals.permute(2, 0, 1, 3)  # N,Np,C,1
        sampled_vals = sampled_vals.squeeze(-1)  # N,Np,C
        return sampled_vals

    def build_planes(self,inner_wavelet_scale,planes_features,planes_features_wavelet_all_level
                     ,planes_features_wavelet_coefs,planes_features_wavelet_yh_zeros,get_all_resolutions
                     ,max_res,max_scale ,wavelet_base_resolution
                     ):
        all_res = []
        current_scale = 1
        planes = planes_features
        if inner_wavelet_scale > 1:
            # x = planes
            x = planes_features
            pad = self.planes_features_wavelet_pad
            # for wavelet_coefs in self.planes_features_wavelet_coefs:
            for level_idx in range(planes_features_wavelet_all_level):
                if get_all_resolutions:
                    all_res.append(x)
                yl = 2 * x
                if ((max_res > 0) and (min(x.shape[2:]) >= max_res)) or (
                        (max_scale > 0) and (current_scale >= max_scale)):
                    # yh = self.planes_features_wavelet_yh_zeros[level_idx].to(device=yl.device, dtype=yl.dtype)
                    break
                elif level_idx < len(planes_features_wavelet_coefs):
                    yh = planes_features_wavelet_coefs[level_idx]
                else:
                    # print('x.shape = ',x.shape)
                    # yh = torch.zeros_like(x).unsqueeze(dim=2).expand(-1,-1,3,-1,-1)
                    yh = planes_features_wavelet_yh_zeros[level_idx].to(device=yl.device, dtype=yl.dtype)

                if yl.shape[3] >= wavelet_base_resolution:
                    yl = F.pad(yl, (pad, pad, pad, pad))
                    yh = F.pad(yh, (pad, pad, pad, pad))
                x = self.idwt((yl, [yh]))
                current_scale *= 2

            if get_all_resolutions:
                all_res.append(x)

            # if self.apply_activation_on_features:
            #     x = F.tanh(x)
            planes = x
        if self.apply_activation_on_features:
            planes = F.tanh(planes)
        return planes,all_res

    def get_planes(self,max_res = -1,max_scale = -1,get_all_resolutions = False):
        # print('************ get_planes ***************')
        if self.last_used_planes is not None:
            return self.last_used_planes
        planes, all_res = self.build_planes(self.inner_wavelet_scale,self.planes_features,
                                             self.planes_features_wavelet_all_level
                     ,self.planes_features_wavelet_coefs,self.planes_features_wavelet_yh_zeros
                                             ,get_all_resolutions,max_res,max_scale,self.wavelet_base_resolution
                                             )
        self.last_used_planes = planes
        if self.upscale_enabled:
            planes_features_upscale = planes
            planes = [planes]
            all_res = [all_res]
            for level in range(self.upscale_levels):
                planes_features_upscale = planes_features_upscale[:,:,self.upscale_base_corner_lst[level] :
                                         (self.upscale_base_corner_lst[level] + self.upscale_base_resolution_lst[level])
                                        , self.upscale_base_corner_lst[level] :
                                         (self.upscale_base_corner_lst[level] + self.upscale_base_resolution_lst[level])
                                          ]
                planes_features_upscale, all_res_upscale = self.build_planes(2, planes_features_upscale,
                                                    1
                                                    , [self.upscale_wavelet_lst[level]],
                                                    None
                                                    , get_all_resolutions, max_res, max_scale,self.wavelet_base_resolution
                                                    )
                planes.append(planes_features_upscale)
                all_res.append(all_res_upscale)

            self.last_used_planes = planes
        if get_all_resolutions:
            return all_res
        return planes

    def reset_cahce(self):
        self.last_used_planes = None
    def sample_from_planes(self,coordinates,plane_features = None, lbound = None):
        N, C = coordinates.shape
        if plane_features is None:
            plane_features = self.get_planes()
        plane_axes = self.plane_axes
        if lbound is None:
            lbound = self.lbound
        if self.learn_rotation_axis:
            sampled_vals = self.sample_from_planes_aux_rotation(coordinates, plane_features, plane_axes, lbound)
        else:
            if self.upscale_enabled:
                plane_features_all = plane_features
                plane_features = plane_features_all[0]
                coords_max_abs = coordinates.abs().max(dim=-1).values
                res = torch.zeros(N, plane_features.shape[0], plane_features.shape[1], device=plane_features.device,
                                  dtype=plane_features.dtype)
                upscale_coords_flag_all = None
                for level in range(self.upscale_levels):
                    plane_features_upscale = plane_features_all[level+1]
                    lbound_upscaled = self.upscale_bound_ratio_lst[level]*lbound
                    if level < (self.upscale_levels - 1):
                        lbound_upscaled_next = (self.upscale_bound_ratio_lst[level + 1] * lbound)
                        upscale_coords_flag =  torch.logical_and(coords_max_abs <= lbound_upscaled, coords_max_abs > lbound_upscaled_next)
                    else:
                        upscale_coords_flag = coords_max_abs <= lbound_upscaled
                    coordinates_upscaled = coordinates[upscale_coords_flag]
                    # coordinates = coordinates[upscale_coords_flag == False]
                    sampled_vals_upscaled = self.sample_from_planes_aux(coordinates_upscaled,
                                                                        plane_features_upscale, plane_axes, lbound_upscaled)
                    res[upscale_coords_flag] = sampled_vals_upscaled

                    if upscale_coords_flag_all is None:
                        upscale_coords_flag_all = upscale_coords_flag
                    else:
                        upscale_coords_flag_all = torch.logical_or(upscale_coords_flag_all,upscale_coords_flag)

                coordinates = coordinates[upscale_coords_flag_all == False]
            sampled_vals = self.sample_from_planes_aux(coordinates,plane_features,plane_axes,lbound)
            if self.upscale_enabled:
                res[upscale_coords_flag_all == False] = sampled_vals
                sampled_vals = res
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

        return self.lbound,grid_features,grid




    def get_params(self,opt_cfg):
        return self.parameters()
        # params = []
        # lr = opt_cfg['lr']
        # lr_planes = opt_cfg['lr_planes'] if (opt_cfg['lr_planes'] is not None) else lr
        # params += [{'params': val, 'lr': lr_planes} for val in self.parameters()]
        # return params

    def forward(self,coordinates,bound):
        sampled_vals = self.sample_from_planes(coordinates,lbound = bound)
        # sampled_view_radiance = self.sample_view_radiance(viewdir)
        res = sampled_vals.view(sampled_vals.shape[0],-1)
        if self.dropout is not None:
            res = self.dropout(res)

        return res

