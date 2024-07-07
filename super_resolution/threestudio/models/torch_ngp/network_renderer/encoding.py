import torch
import torch.nn as nn
import torch.nn.functional as F

class FreqEncoder(nn.Module):
    def __init__(self, input_dim, max_freq_log2, N_freqs,
                 log_sampling=True, include_input=True,
                 periodic_fns=(torch.sin, torch.cos)):
    
        super().__init__()

        self.input_dim = input_dim
        self.include_input = include_input
        self.periodic_fns = periodic_fns

        self.output_dim = 0
        if self.include_input:
            self.output_dim += self.input_dim

        self.output_dim += self.input_dim * N_freqs * len(self.periodic_fns)

        if log_sampling:
            self.freq_bands = 2. ** torch.linspace(0., max_freq_log2, N_freqs)
        else:
            self.freq_bands = torch.linspace(2. ** 0., 2. ** max_freq_log2, N_freqs)

        self.freq_bands = self.freq_bands.numpy().tolist()

    def forward(self, input, **kwargs):

        out = []
        if self.include_input:
            out.append(input)

        for i in range(len(self.freq_bands)):
            freq = self.freq_bands[i]
            for p_fn in self.periodic_fns:
                out.append(p_fn(input * freq))

        out = torch.cat(out, dim=-1)


        return out

def get_encoder(encoding, cfg
                ):

    if encoding == 'None':
        return lambda x, **kwargs: x, cfg.input_dim
    
    elif encoding == 'frequency':
        #encoder = FreqEncoder(input_dim=input_dim, max_freq_log2=multires-1, N_freqs=multires, log_sampling=True)
        from threestudio.models.torch_ngp.freqencoder import FreqEncoder
        encoder = FreqEncoder(input_dim=cfg.input_dim, degree=cfg.multires)

    elif encoding == 'sphere_harmonics':
        from threestudio.models.torch_ngp.shencoder import SHEncoder
        encoder = SHEncoder(input_dim=cfg.input_dim, degree=cfg.dir_degree)

    elif encoding == 'triplane_wavelet':
        from threestudio.models.triplaneencoder.triplane_encoder import TriPlaneVolume
        # encoder = TriPlaneVolume(
        #     number_of_features=kwargs['triplane_channels'],
        #     plane_resolution=kwargs['triplane_resolution'],
        #     init_sigma=0.1,
        #     lbound=bound, #anyway lbound is orerriden in forward later
        #     viewdir_plane_resolution=-1,
        #     apply_activation_on_features=False,
        #     inner_multi_res_scale=kwargs['triplane_wavelet_levels'],
        #     inner_multi_res_scale_current=1,
        #     learn_rotation_axis = kwargs['learn_rotation_axis'],
        #     dropout=kwargs['dropout'],
        #     wavelet_type = kwargs['wavelet_type']
        # )
        encoder = TriPlaneVolume(
            number_of_features=cfg.triplane_channels,
            plane_resolution=cfg.triplane_resolution,
            init_sigma=0.1,
            lbound=cfg.bound,
            viewdir_plane_resolution=-1,
            apply_activation_on_features=False,
            inner_multi_res_scale=cfg.triplane_wavelet_levels,
            inner_multi_res_scale_current=1,
            low_res_scale=cfg.triplane_low_res_scale,
            high_res_scale=cfg.triplane_high_res_scale,
            input_pts_in_unit_cube = False
        )
    else:
        raise NotImplementedError('Unknown encoding mode, choose from [None, frequency, sphere_harmonics, hashgrid, tiledgrid]')

    return encoder, encoder.output_dim