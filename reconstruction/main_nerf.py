import torch
import argparse

from nerf.provider import NeRFDataset , get_dataset
# from nerf.gui import NeRFGUI
from nerf.utils import *

from functools import partial
from loss import huber_loss
import copy
import sys
from run_utils import get_params

import sys,os
dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.join(dir_path,'aux_libs'))

#torch.autograd.set_detect_anomaly(True)

def run_aux(opt):
    if opt.O:
        opt.fp16 = True
        opt.cuda_ray = True
        opt.preload = True

    if opt.patch_size > 1:
        opt.error_map = False  # do not use error_map if use patch-based training
        # assert opt.patch_size > 16, "patch_size should > 16 to run LPIPS loss."
        assert opt.num_rays % (opt.patch_size ** 2) == 0, "patch_size ** 2 should be dividable by num_rays."

    if opt.ff:
        opt.fp16 = True
        assert opt.bg_radius <= 0, "background model is not implemented for --ff"
        from nerf.network_ff import NeRFNetwork
    elif opt.tcnn:
        opt.fp16 = True
        assert opt.bg_radius <= 0, "background model is not implemented for --tcnn"
        from nerf.network_tcnn import NeRFNetwork
    else:
        from nerf.network import NeRFNetwork

    print(opt)

    seed_everything(opt.seed)

    keys_to_pass_to_nerf = ['triplane_channels', 'triplane_resolution', 'triplane_wavelet_levels',
                            'wavelet_type',
                            'hidden_dim' , 'hidden_dim_color', 'hidden_dim_bg', 'learn_rotation_axis',
                            'dropout','inner_bound',
                            'lbound_auto_scale',
                            'upscale_ratio_bound',
                            'upscale_levels',
                            'density_blob_scale',
                            'density_blob_std',
                            'mlp_weight_decay',
                            'wavelet_base_resolution',
                            'nerfacc_renderer'
                            ]
    additional_nerf_params = {}
    tmp_opt = vars(opt)
    for key in keys_to_pass_to_nerf:
        additional_nerf_params[key] = tmp_opt[key]
    model = NeRFNetwork(
        encoding="triplane_wavelet" if opt.triplane_wavelet else "hashgrid",
        bound=opt.bound,
        cuda_ray=opt.cuda_ray,
        density_scale=opt.density_scale,
        min_near=opt.min_near,
        density_thresh=opt.density_thresh,
        bg_radius=opt.bg_radius,
        **additional_nerf_params
    )

    print(model)
    print('***************************')
    print(model.encoder.state_dict().keys())
    print('***************************')

    criterion = torch.nn.MSELoss(reduction='none')
    if opt.huber_loss:
        # criterion = partial(huber_loss, reduction='none')
        criterion = torch.nn.HuberLoss(reduction='none', delta=0.1) # only available after torch 1.10 ?

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if opt.test or opt.save_planes:

        metrics = [PSNRMeter(), LPIPSMeter(device=device)]
        ema_decay = None
        if opt.test_with_ema and (opt.ema_decay >= 0):
            print('***** testing with ema *******')
            ema_decay = opt.ema_decay
        else:
            print('***** testing withot ema *******')
        trainer = Trainer('trinerflet', opt, model, device=device, workspace=opt.workspace, criterion=criterion, fp16=opt.fp16,
                          metrics=metrics, use_checkpoint=opt.ckpt, ema_decay=ema_decay,mute=opt.mute)
        if (opt.save_planes):
            trainer.save_triplane(True,save_wavelet = opt.save_wavelet)
        else:
            if opt.gui:
                gui = NeRFGUI(opt, trainer)
                gui.render()

            else:
                test_loader = get_dataset(opt)(opt, device=device, type='test').dataloader()

                if test_loader.has_gt:
                    trainer.evaluate(test_loader)  # blender has gt, so evaluate it.

                trainer.test(test_loader, write_video=True)  # test and save video

                trainer.save_mesh(resolution=256, threshold=10)

    else:
        if opt.mlp_weight_decay > 0:
            optimizer = lambda model: torch.optim.AdamW(model.get_params(opt.lr), betas=(0.9, 0.99), eps=1e-15
                                                        ,weight_decay=0) #weight decay are overriden inside get_params
        else:
            optimizer = lambda model: torch.optim.Adam(model.get_params(opt.lr), betas=(0.9, 0.99), eps=1e-15)

        # TODO
        # train_loader = get_dataset(opt)(opt, device=device, type='train').dataloader()
        opt_data = copy.deepcopy(opt)
        opt_data.num_rays = -1
        train_dataset = get_dataset(opt)(opt_data, device=torch.device('cpu'), type='train')
        train_loader = train_dataset.dataloader()

        # decay to 0.1 * init_lr at last iter step
        scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer, lambda iter: decay_function(iter, opt))

        metrics = [PSNRMeter(), LPIPSMeter(device=device)]
        ema_decay = opt.ema_decay
        if ema_decay <= 0:
            print(' ************* no ema ***************')
            ema_decay = None
        trainer = Trainer('trinerflet', opt, model, device=device, workspace=opt.workspace, optimizer=optimizer,
                          criterion=criterion, ema_decay=ema_decay, fp16=opt.fp16, lr_scheduler=scheduler,
                          scheduler_update_every_step=True, metrics=metrics, use_checkpoint=opt.ckpt,
                          eval_interval=opt.save_every,mute=opt.mute)

        if opt.gui:
            gui = NeRFGUI(opt, trainer, train_loader)
            gui.render()

        else:
            valid_loader = get_dataset(opt)(opt, device=device, type='val').dataloader()
            # TODO
            # step_per_epoch = len(train_loader)
            step_per_epoch = len(train_loader) * (train_dataset.H * train_dataset.W) / opt.num_rays
            max_epoch = np.ceil((opt.iters + max(opt.warmup_steps, 0)) / step_per_epoch).astype(np.int32)

            # also test
            test_loader = get_dataset(opt)(opt, device=device, type='test').dataloader()

            trainer.train(train_loader, valid_loader, max_epoch,
                          test_loader=test_loader if test_loader.has_gt else None)

            if not opt.test_with_ema:
                trainer.ema = None
            if test_loader.has_gt:
                trainer.evaluate(test_loader)  # blender has gt, so evaluate it.

            trainer.test(test_loader, write_video=True)  # test and save video

            trainer.save_mesh(resolution=256, threshold=10)


def run(opt):
    assert opt.path is not None
    assert os.path.exists(opt.path)
    opt.command_line = ' '.join(sys.argv)
    opt_keys = ['iters', 'num_rays', 'triplane_resolution', 'triplane_wavelet_levels', 'downscale', 'warmup_steps'
                ,'lr' , 'wavelet_regularization', 'upscale_ratio_bound', 'upscale_levels'
                ]
    print(opt.iters)
    print(opt.num_rays)
    print(opt.triplane_resolution)
    print(opt.triplane_wavelet_levels)
    print(opt.downscale)
    print(opt.warmup_steps)
    print(opt.lr)
    print(opt.wavelet_regularization)
    print(opt.upscale_ratio_bound)
    print(opt.upscale_levels)
    opt_vars = vars(opt)
    # opt_lst = [opt.iters,opt.num_rays,opt.triplane_resolution,opt.triplane_wavelet_levels,opt.downscale,opt.warmup_steps]
    length = max([len(opt_vars[key]) for key in opt_keys])
    assert sum([(len(opt_vars[key]) == length) or (len(opt_vars[key]) == 1) for key in opt_keys]) == len(opt_keys)

    if opt.test:
        for key in opt_keys:
            opt_vars[key] = opt_vars[key][-1]
        # print(opt)
        run_aux(opt)
    else:
        for i in range(length):
            tmp_opt = copy.deepcopy(opt)
            tmp_opt_vars = vars(tmp_opt)
            for key in opt_keys:
                if len(tmp_opt_vars[key]) == length:
                    tmp_opt_vars[key] = tmp_opt_vars[key][i]
                else:
                    tmp_opt_vars[key] = tmp_opt_vars[key][0]
            # print(tmp_opt)
            run_aux(tmp_opt)



if __name__ == '__main__':
    opt = get_params()
    run(opt)





