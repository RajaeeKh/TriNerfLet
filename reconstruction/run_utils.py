
import argparse

def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str,default=None)
    parser.add_argument('-O', action='store_true', help="equals --fp16 --cuda_ray --preload")
    parser.add_argument('--test', action='store_true', help="test mode")
    parser.add_argument('--workspace', type=str, default='workspace')
    parser.add_argument('--seed', type=int, default=0)

    ### training options
    parser.add_argument('--iters', type=int, nargs='+', default=[30000], help="training iters")
    parser.add_argument('--lr', type=float, nargs='+', default=[1e-2], help="initial learning rate")
    parser.add_argument('--ckpt', type=str, default='latest')
    parser.add_argument('--num_rays', type=int, nargs='+', default=[4096],
                        help="num rays sampled per image for each training step")
    parser.add_argument('--cuda_ray', action='store_true', help="use CUDA raymarching instead of pytorch")
    parser.add_argument('--max_steps', type=int, default=1024,
                        help="max num steps sampled per ray (only valid when using --cuda_ray)")
    parser.add_argument('--num_steps', type=int, default=512,
                        help="num steps sampled per ray (only valid when NOT using --cuda_ray)")
    parser.add_argument('--upsample_steps', type=int, default=0,
                        help="num steps up-sampled per ray (only valid when NOT using --cuda_ray)")
    parser.add_argument('--update_extra_interval', type=int, default=16,
                        help="iter interval to update extra status (only valid when using --cuda_ray)")
    parser.add_argument('--max_ray_batch', type=int, default=4096,
                        help="batch size of rays at inference to avoid OOM (only valid when NOT using --cuda_ray)")
    parser.add_argument('--patch_size', type=int, default=1,
                        help="[experimental] render patches in training, so as to apply LPIPS loss. 1 means disabled, use [64, 32, 16] to enable")

    ### network backbone options
    parser.add_argument('--fp16', action='store_true', help="use amp mixed precision training")
    parser.add_argument('--ff', action='store_true', help="use fully-fused MLP")
    parser.add_argument('--tcnn', action='store_true', help="use TCNN backend")

    ### dataset options
    parser.add_argument('--color_space', type=str, default='srgb', help="Color space, supports (linear, srgb)")
    parser.add_argument('--preload', action='store_true',
                        help="preload all data into GPU, accelerate training but use more GPU memory")
    # (the default value is for the fox dataset)
    parser.add_argument('--bound', type=float, default=2,
                        help="assume the scene is bounded in box[-bound, bound]^3, if > 1, will invoke adaptive ray marching.")
    parser.add_argument('--scale', type=float, default=0.33, help="scale camera location into box[-bound, bound]^3")
    parser.add_argument('--offset', type=float, nargs='*', default=[0, 0, 0], help="offset of camera location")
    parser.add_argument('--dt_gamma', type=float, default=1 / 128,
                        help="dt_gamma (>=0) for adaptive ray marching. set to 0 to disable, >0 to accelerate rendering (but usually with worse quality)")
    parser.add_argument('--min_near', type=float, default=0.2, help="minimum near distance for camera")
    parser.add_argument('--density_thresh', type=float, default=10, help="threshold for density grid to be occupied")
    parser.add_argument('--bg_radius', type=float, default=-1,
                        help="if positive, use a background model at sphere(bg_radius)")

    ### GUI options
    parser.add_argument('--gui', action='store_true', help="start a GUI")
    parser.add_argument('--W', type=int, default=1920, help="GUI width")
    parser.add_argument('--H', type=int, default=1080, help="GUI height")
    parser.add_argument('--radius', type=float, default=5, help="default GUI camera radius from center")
    parser.add_argument('--fovy', type=float, default=50, help="default GUI camera fovy")
    parser.add_argument('--max_spp', type=int, default=64, help="GUI rendering max sample per pixel")

    ### experimental
    parser.add_argument('--error_map', action='store_true', help="use error map to sample rays")
    parser.add_argument('--clip_text', type=str, default='', help="text input for CLIP guidance")
    parser.add_argument('--rand_pose', type=int, default=-1,
                        help="<0 uses no rand pose, =0 only uses rand pose, >0 sample one rand pose every $ known poses")

    ### added by rajaee
    parser.add_argument('--triplane_wavelet', action='store_true', help="use triplane wavelet encoder")
    parser.add_argument('--wavelet_regularization', type=float, nargs='+', default=[0.1], help="wavelet coefs regularization")
    parser.add_argument('--weighted_regularization', action='store_true',
                        help="whether to activate weighted regularization")
    parser.add_argument('--save_every', type=int, default=1, help="save every x epochs")
    parser.add_argument('--background_color', type=float, default=0, help="background color")
    parser.add_argument('--train_rand_bg', action='store_true', help="random background color for training")
    parser.add_argument('--triplane_channels', type=int, default=16, help="triplane number of channels per plane")
    parser.add_argument('--triplane_resolution', type=int, nargs='+', default=[2048], help="triplane resolution")
    parser.add_argument('--triplane_wavelet_levels', type=int, nargs='+', default=[128], help="triplane wavelet levels")
    parser.add_argument('--hidden_dim', type=int, default=64, help="hidden dim for sigma")
    parser.add_argument('--hidden_dim_color', type=int, default=64, help="hidden dim for color")
    parser.add_argument('--hidden_dim_bg', type=int, default=64, help="hidden dim for bg")
    parser.add_argument('--save_planes', action='store_true', help="whether to save all planes")
    parser.add_argument('--accumelate_steps', type=int, default=1, help="accumulate steps")
    parser.add_argument('--learn_rotation_axis', action='store_true', help="learn rotation for triplane")
    parser.add_argument('--dropout', type=float, default=0, help="dropout")
    parser.add_argument('--sched_base', type=float, default=0.1, help="base for scheduler")
    parser.add_argument('--sched_exp', type=float, default=2.5, help="exp for scheduler")
    parser.add_argument('--downscale', type=int, nargs='+', default=[1], help="downscale factor")
    parser.add_argument('--min_wavelet_resolution_to_learn', type=int, default=-1,
                        help="if > 0 it will learn wavelet resolution")
    parser.add_argument('--save_wavelet', action='store_true', help="saving all wavelet features")
    parser.add_argument('--warmup_steps', type=int, nargs='+', default=[0], help="warmup steps")
    parser.add_argument('--warmup_factor', type=float, default=1e-3, help="warmup factor")
    parser.add_argument('--ema_decay', type=float, default=0.95, help="ema decay")
    parser.add_argument('--test_with_ema', action='store_true', help="whether to enable ema in test")
    parser.add_argument('--fast_training', action='store_true', help="enable fast trainig")
    parser.add_argument('--training_evaluate_test', action='store_true',
                        help="evaluate test insted of validation in training")
    parser.add_argument('--mute', action='store_true', help="muting results")

    parser.add_argument('--inner_bound', type=float, default=-1,
                        help="inner bound")
    parser.add_argument('--wavelet_type', type=str, default='bior6.8')

    parser.add_argument('--lbound_auto_scale', action='store_true', help="autoscaling bound")

    parser.add_argument('--upscale_ratio_bound', type=float, nargs='+', default=[-1], help="upscale ratio")
    parser.add_argument('--upscale_levels', type=int, nargs='+', default=[2], help="upscale levels")

    parser.add_argument("--huber_loss", action='store_true',
                        help='use huber loss instead of mse')
    parser.add_argument('--density_scale', type=int, default=1, help="density scale")
    parser.add_argument('--alpha_bce', type=float, default=0, help="density alpha_bce")

    parser.add_argument('--density_blob_scale', type=float, default=0,
                        help="density_blob_scale")
    parser.add_argument('--density_blob_std', type=float, default=0.5,
                        help="density_blob_std")

    parser.add_argument('--mlp_weight_decay', type=float, default=-1,
                        help="mlp_weight_decay")
    parser.add_argument('--wavelet_base_resolution', type=int, default=0,
                        help="wavelet_base_resolution")

    parser.add_argument('--nerfacc_renderer', action='store_true', help="use nerfacc renderer")
    parser.add_argument('--z_variance_reg', type=float, default=-1,
                        help="z_variance reg")


    #LLFF data
    parser.add_argument('--llff_dataset', action='store_true', help="llff dataset flag")
    parser.add_argument("--llff_spherify", action='store_true',
                        help='set for spherical 360 scenes')
    parser.add_argument("--llff_hold", type=int, default=8,
                        help='will take every 1/N images as LLFF test set, paper uses 8')
    parser.add_argument("--llff_render_mode", action='store_true',
                        help='LLFF render data')
    parser.add_argument("--llff_render_all_test", action='store_true',
                        help='LLFF train on  all images')
    parser.add_argument("--llff_ndc", action='store_true',
                        help='whether to use ndc coords')

    parser.add_argument('--topia_dataset', action='store_true', help="3dtopia dataset flag")
    parser.add_argument('--topia_poses_fname', type=str, default='')

    opt = parser.parse_args()
    return opt