# TriNeRFLet: A Wavelet Based Triplane NeRF Representation

<p align="center">

<a href="https://rajaeekh.github.io/">Rajaei Khatib</a>,
<a href="https://www.giryes.sites.tau.ac.il/">Raja Giryes</a>

<a href="https://rajaeekh.github.io/trinerflet-web"><img src="https://img.shields.io/static/v1?label=Project&message=Website&color=blue"></a>
<a href="https://arxiv.org/abs/2401.06191"><img src="https://img.shields.io/badge/arXiv-2401.06191-b31b1b.svg"></a>

In this work, we propose TriNeRFLet, a 2D wavelet-based multiscale triplane representation for NeRF, which closes the 3D
recovery performance gap and is competitive with current state-of-the-art methods. Building upon the triplane framework,
we also propose a novel super-resolution (SR) technique that combines a diffusion model with TriNeRFLet for improving
NeRF resolution.

</p>
<br>

# Setup
## Clone Project
```
git clone https://github.com/RajaeeKh/TriNerfLet.git
cd TriNerfLet
```

## Using Environment
To set up our environment, please run:
```
# create an environment and activate it
pip install -r requirements2.txt
cd aux_libs
bash scripts/install_ext.sh
```
## Using Docker
```
docker run --rm --gpus all -it --name tri_run --entrypoint bash --shm-size=64g  --mount type=bind,src=<src_dir>,dst=<target_dir_docker>  rajaeekh95/trinerflet
```
<br>

# Running Reconstruction
Training:
```
cd reconstruction

## small version:
CUDA_VISIBLE_DEVICES=0 python main_nerf.py --path <data_path>  --workspace <output_path>  --fp16 --cuda_ray --bound 1.5 --scale 1 --dt_gamma 0 --iters 1000 5000 --num_rays 20000 60000 --background_color 0 --triplane_wavelet --triplane_channels 16 --triplane_wavelet_levels 8 16 --triplane_resolution 512 1024 --wavelet_regularization 0.2 --downscale 1 --ckpt latest_model --ema_decay -1 --training_evaluate_test --warmup_steps 0 200 --fast_training

## base light version:
CUDA_VISIBLE_DEVICES=0 python main_nerf.py --path <data_path>  --workspace <output_path>  --fp16 --cuda_ray --bound 1.5 --scale 1 --dt_gamma 0 --iters 1000 2000 6000 --num_rays 60000 --background_color 0 --triplane_wavelet --triplane_channels 32 --triplane_wavelet_levels 8 16 32 --triplane_resolution 512 1024 2048 --wavelet_regularization 0.4 --downscale 1 --ckpt latest_model --ema_decay -1 --training_evaluate_test --warmup_steps 0 100 400 --fast_training

## base version:
CUDA_VISIBLE_DEVICES=0 python main_nerf.py --path <data_path>  --workspace <output_path> --fp16 --cuda_ray --bound 1.5 --scale 1 --dt_gamma 0 --iters 1000 2000 40000 --num_rays 60000 --background_color 0 --triplane_wavelet --triplane_channels 32 --triplane_wavelet_levels 8 16 32 --triplane_resolution 512 1024 2048 --wavelet_regularization 0.4 --downscale 1 --ckpt latest_model --ema_decay -1 --training_evaluate_test --warmup_steps 0 100 500 --fast_training

## large version:
CUDA_VISIBLE_DEVICES=0 python main_nerf.py --path <data_path>  --workspace <output_path> --fp16 --cuda_ray --bound 1.5 --scale 1 --dt_gamma 0 --iters 1000 2000 80000 --num_rays 30000 60000 60000 --background_color 0 --triplane_wavelet --triplane_channels 48 --triplane_wavelet_levels 8 16 32 --triplane_resolution 512 1024 2048 --wavelet_regularization 0.6 --hidden_dim 128 --hidden_dim_color 128 --downscale 1 --ckpt latest_model --ema_decay -1 --training_evaluate_test --warmup_steps 0 100 1000

```
Rendering: for rendering, you need to run the same training command and adding to it: --test --max_steps 4096.

In case you are having memory issues in the base versions try to replace --num_rays to: 20000 60000 60000


<br>

# Running Super-Resolution
```
cd super_resolution

## blender 100->400
python launch.py --train --gpu 3 --config configs/triplane-sr100_400_2.yaml tag=materials data.dataroot=/home/rajaee/datasets/nerf_synthetic/nerf_synthetic/materials exp_root_dir=/home/rajaee/exps/final_check/sr

## blender 200->800
python launch.py --train --gpu 3 --config configs/triplane-sr200_800_6.yaml tag=materials data.dataroot=/home/rajaee/datasets/nerf_synthetic/nerf_synthetic/materials exp_root_dir=/home/rajaee/exps/final_check/sr

## LLFF 378x504 to 1512x2016
python launch.py --train --gpu 3 --config configs/triplane-sr_llff_best10_2.yaml tag=materials data.dataroot=/home/rajaee/datasets/nerf_synthetic/nerf_synthetic/materials exp_root_dir=/home/rajaee/exps/final_check/sr
```

All code here was tested on A6000RTX.