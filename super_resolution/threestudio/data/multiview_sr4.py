import copy
import json
import math
import os
import random
from dataclasses import dataclass

import cv2
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp
from torch.utils.data import DataLoader, Dataset, IterableDataset
from tqdm import tqdm

import threestudio
from threestudio import register
from threestudio.utils.config import parse_structured
from threestudio.utils.ops import get_mvp_matrix, get_ray_directions, get_rays,get_projection_matrix
from threestudio.utils.typing import *

from packaging import version as pver

def convert_pose(C2W):
    flip_yz = torch.eye(4)
    flip_yz[1, 1] = -1
    flip_yz[2, 2] = -1
    C2W = torch.matmul(C2W, flip_yz)
    return C2W


def convert_proj(K, H, W, near, far):
    return [
        [2 * K[0, 0] / W, -2 * K[0, 1] / W, (W - 2 * K[0, 2]) / W, 0],
        [0, -2 * K[1, 1] / H, (H - 2 * K[1, 2]) / H, 0],
        [0, 0, (-far - near) / (far - near), -2 * far * near / (far - near)],
        [0, 0, -1, 0],
    ]

def nerf_matrix_to_ngp(pose, scale=1, offset=[0, 0, 0]):
    new_pose = np.array([
        [pose[1, 0], -pose[1, 1], -pose[1, 2], pose[1, 3] * scale + offset[0]],
        [pose[2, 0], -pose[2, 1], -pose[2, 2], pose[2, 3] * scale + offset[1]],
        [pose[0, 0], -pose[0, 1], -pose[0, 2], pose[0, 3] * scale + offset[2]],
        [0, 0, 0, 1],
    ], dtype=np.float32)
    return new_pose

def custom_meshgrid(*args):
    # ref: https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid
    if pver.parse(torch.__version__) < pver.parse('1.10'):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing='ij')

@torch.cuda.amp.autocast(enabled=False)
def get_rays_ngp(poses, intrinsics, H, W, N=-1, error_map=None, patch_size=1):
    ''' get rays
    Args:
        poses: [B, 4, 4], cam2world
        intrinsics: [4]
        H, W, N: int
        error_map: [B, 128 * 128], sample probability based on training error
    Returns:
        rays_o, rays_d: [B, N, 3]
        inds: [B, N]
    '''

    device = poses.device
    B = poses.shape[0]
    fx, fy, cx, cy = intrinsics

    i, j = custom_meshgrid(torch.linspace(0, W-1, W, device=device), torch.linspace(0, H-1, H, device=device)) # float
    i = i.t().reshape([1, H*W]).expand([B, H*W]) + 0.5
    j = j.t().reshape([1, H*W]).expand([B, H*W]) + 0.5

    results = {}

    if N > 0:
        N = min(N, H*W)

        # if use patch-based sampling, ignore error_map
        if patch_size > 1:

            # random sample left-top cores.
            # NOTE: this impl will lead to less sampling on the image corner pixels... but I don't have other ideas.
            num_patch = N // (patch_size ** 2)
            inds_x = torch.randint(0, H - patch_size, size=[num_patch], device=device)
            inds_y = torch.randint(0, W - patch_size, size=[num_patch], device=device)
            inds = torch.stack([inds_x, inds_y], dim=-1) # [np, 2]

            # create meshgrid for each patch
            pi, pj = custom_meshgrid(torch.arange(patch_size, device=device), torch.arange(patch_size, device=device))
            offsets = torch.stack([pi.reshape(-1), pj.reshape(-1)], dim=-1) # [p^2, 2]

            inds = inds.unsqueeze(1) + offsets.unsqueeze(0) # [np, p^2, 2]
            inds = inds.view(-1, 2) # [N, 2]
            inds = inds[:, 0] * W + inds[:, 1] # [N], flatten

            inds = inds.expand([B, N])

        elif error_map is None:
            inds = torch.randint(0, H*W, size=[N], device=device) # may duplicate
            inds = inds.expand([B, N])
        else:

            # weighted sample on a low-reso grid
            inds_coarse = torch.multinomial(error_map.to(device), N, replacement=False) # [B, N], but in [0, 128*128)

            # map to the original resolution with random perturb.
            inds_x, inds_y = inds_coarse // 128, inds_coarse % 128 # `//` will throw a warning in torch 1.10... anyway.
            sx, sy = H / 128, W / 128
            inds_x = (inds_x * sx + torch.rand(B, N, device=device) * sx).long().clamp(max=H - 1)
            inds_y = (inds_y * sy + torch.rand(B, N, device=device) * sy).long().clamp(max=W - 1)
            inds = inds_x * W + inds_y

            results['inds_coarse'] = inds_coarse # need this when updating error_map

        i = torch.gather(i, -1, inds)
        j = torch.gather(j, -1, inds)

        results['inds'] = inds

    else:
        inds = torch.arange(H*W, device=device).expand([B, H*W])
        results['inds'] = inds

    zs = torch.ones_like(i)
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    directions = torch.stack((xs, ys, zs), dim=-1)
    directions = directions / torch.norm(directions, dim=-1, keepdim=True)
    rays_d = directions @ poses[:, :3, :3].transpose(-1, -2) # (B, N, 3)

    rays_o = poses[..., :3, 3] # [B, 3]
    rays_o = rays_o[..., None, :].expand_as(rays_d) # [B, N, 3]

    results['rays_o'] = rays_o
    results['rays_d'] = rays_d

    return results

def inter_pose(pose_0, pose_1, ratio):
    pose_0 = pose_0.detach().cpu().numpy()
    pose_1 = pose_1.detach().cpu().numpy()
    pose_0 = np.linalg.inv(pose_0)
    pose_1 = np.linalg.inv(pose_1)
    rot_0 = pose_0[:3, :3]
    rot_1 = pose_1[:3, :3]
    rots = Rot.from_matrix(np.stack([rot_0, rot_1]))
    key_times = [0, 1]
    slerp = Slerp(key_times, rots)
    rot = slerp(ratio)
    pose = np.diag([1.0, 1.0, 1.0, 1.0])
    pose = pose.astype(np.float32)
    pose[:3, :3] = rot.as_matrix()
    pose[:3, 3] = ((1.0 - ratio) * pose_0 + ratio * pose_1)[:3, 3]
    pose = np.linalg.inv(pose)
    return pose


@dataclass
class MultiviewsDataModuleConfig:
    dataroot: str = ""
    low_resolution: int = 64
    high_resolution: int = 256

    batch_size: int = 1
    eval_batch_size: int = 1
    latent_scale: int=1
    load_high_res_gt: bool = False
    bg_color: Tuple = (0.0, 0.0, 0.0)

    shuffle_batch: int = -1
    shuffle_steps: int = -1

    n_val_views: int = 1
    n_test_views: int = 120
    eval_elevation_deg: float = 15.0
    eval_camera_distance: float = 1.5
    eval_fovy_deg: float = 70.0



class MultiviewDataset(Dataset):
    def __init__(self, cfg: Any,mode,split = 'train',shuffle=False) -> None:
        super().__init__()
        self.cfg: MultiviewsDataModuleConfig = cfg

        assert self.cfg.batch_size == 1
        assert mode in ['low_res','high_res']
        assert split == 'train'
        self.shuffle = shuffle

        if mode == 'low_res':
            height = self.cfg.low_resolution
            width = self.cfg.low_resolution
        else:
            height = self.cfg.high_resolution
            width = self.cfg.high_resolution

        self.heights: List[int] = (
            [height] if isinstance(height, int) else height
        )
        self.widths: List[int] = (
            [width] if isinstance(width, int) else width
        )
        self.batch_sizes: List[int] = (
            [self.cfg.batch_size]
            if isinstance(self.cfg.batch_size, int)
            else self.cfg.batch_size
        )
        assert len(self.heights) == len(self.widths) == len(self.batch_sizes)
        self.resolution_milestones: List[int]


        self.directions_unit_focals = [
            get_ray_directions(H=height, W=width, focal=1.0)
            for (height, width) in zip(self.heights, self.widths)
        ]
        self.height: int = self.heights[0]
        self.width: int = self.widths[0]
        assert self.height == self.width
        self.batch_size: int = self.batch_sizes[0]
        self.directions_unit_focal = self.directions_unit_focals[0]


        self.batch_idx = 0
        mdata_f = os.path.join(self.cfg.dataroot,'metadata.json')
        with open(mdata_f) as file:
            mdata = json.load(file)
        self.num_views = mdata['num_views']
        img_lst = []
        # self.batch_lst = []
        rays_o_lst = []
        rays_d_lst = []
        self.imgs = None
        for img_idx in range(self.num_views):
            img_f = os.path.join(self.cfg.dataroot,'images',f'{img_idx}.png')
            cam_f = os.path.join(self.cfg.dataroot,'cameras',f'{img_idx}.pt')
            batch = torch.load(cam_f,map_location='cpu')

            c2w = batch['c2w']
            fovy = batch['fovy']
            focal_length: Float[Tensor, "B"] = 0.5 * self.height / torch.tan(0.5 * fovy)
            directions: Float[Tensor, "B H W 3"] = self.directions_unit_focal[
                                                   None, :, :, :
                                                   ].repeat(self.batch_size, 1, 1, 1)
            directions[:, :, :, :2] = (
                    directions[:, :, :, :2] / focal_length[:, None, None, None]
            )

            # Importance note: the returned rays_d MUST be normalized!
            rays_o, rays_d = get_rays(directions, c2w, keepdim=True)
            rays_o_lst.append(rays_o)
            rays_d_lst.append(rays_d)

            # self.batch_lst.append(batch)
            if mode == 'low_res':
                img = cv2.imread(img_f,cv2.IMREAD_UNCHANGED) / 255
                H, W, C = img.shape
                assert W==2*H
                assert C==3
                alpha = img[:,H:]
                bg_color = np.array(self.cfg.bg_color).reshape((1, 1, 3))
                img = img[:,:H] * alpha + bg_color * (1-alpha)
                assert self.height <= H
                img = cv2.resize(img, (self.width,self.height), interpolation=cv2.INTER_AREA)
                img = torch.from_numpy(img).float()
                img_lst.append(img)

        self.rays_o = torch.stack(rays_o_lst)
        self.rays_d = torch.stack(rays_d_lst)
        if len(img_lst) > 0:
            self.imgs = torch.stack(img_lst)
        self.light_positions = torch.zeros(self.rays_o.shape[0],3)

        if self.shuffle:
            if (self.cfg.shuffle_batch > 0) and (self.cfg.shuffle_steps > 0):
                number_of_imgs = self.rays_o.shape[0]
                all_pixels = (number_of_imgs * self.height * self.width)
                num_of_perms = math.ceil(self.cfg.shuffle_batch*self.cfg.shuffle_steps / all_pixels)
                self.randperm = torch.cat([torch.randperm(all_pixels).view(-1) for i in range(num_of_perms)],dim=0)
                self.perm_idx = 0

            self.rays_o = self.rays_o.reshape(-1,self.rays_o.shape[-1])
            self.rays_d = self.rays_d.reshape(-1, self.rays_d.shape[-1])
            self.frames_img = self.imgs.reshape(-1, self.imgs.shape[-1])




    def __len__(self):
        return self.rays_o.shape[0]

    def __getitem__(self, index):
        if self.shuffle:
            if (self.cfg.shuffle_batch > 0) and (self.cfg.shuffle_steps > 0):
                # print('perm_idx = {}'.format(self.perm_idx))
                inds = self.randperm[self.perm_idx*self.cfg.shuffle_batch:(self.perm_idx+1)*self.cfg.shuffle_batch]
                self.perm_idx += 1
            else:
                inds = np.random.choice(self.rays_o.shape[0],self.height*self.width,replace=False)
            res = {
                "index": index,
                "rays_o": self.rays_o[inds],
                "rays_d": self.rays_d[inds],
                'gt_rgb': self.frames_img[inds],
                "light_positions": self.light_positions[index: index + 1],
            }
            return res

        res = {
            "index": index,
            "rays_o": self.rays_o[index: index + 1],
            "rays_d": self.rays_d[index: index + 1],
            "light_positions": self.light_positions[index: index + 1],
            "height": self.height,
            "width": self.width,
        }
        # print('rays_o.shape1 = {}'.format(res['rays_o'].shape))
        # print('light_positions.shape1 = {}'.format(res['light_positions'].shape))
        if self.imgs is not None:
            res['gt_rgb'] = self.imgs[index: index + 1]
        return res


class MultiviewDatasetDoubleResolution(Dataset):
    def __init__(self, cfg: Any, split='train', add_shuffle=False) -> None:
        super().__init__()
        self.low_res_dataset = MultiviewDataset(cfg, 'low_res', split)
        self.high_res_dataset = MultiviewDataset(cfg, 'high_res', split)
        self.low_res_dataset_shuf = None
        if add_shuffle:
            self.low_res_dataset_shuf = MultiviewDataset(cfg, 'low_res', split, shuffle=True)
        assert len(self.low_res_dataset) == len(self.high_res_dataset)

    def __len__(self):
        return len(self.low_res_dataset)

    def __getitem__(self, index):
        res = {'high_res' : self.high_res_dataset[index],'low_res':self.low_res_dataset[index]}
        if self.low_res_dataset_shuf is not None:
            res['low_res_shuffled'] = self.low_res_dataset_shuf[index]
        return res

class MultiviewDatasetDoubleResolutionTraining(IterableDataset):
    def __init__(self, cfg: Any) -> None:
        super().__init__()
        self.cfg: MultiviewsDataModuleConfig = cfg
        split = 'train'
        self.dataset = MultiviewDatasetDoubleResolution(cfg,split,
                                                        add_shuffle=(self.cfg.shuffle_batch > 0) and (
                                                                    self.cfg.shuffle_steps > 0)
                                                        )

    def __iter__(self):
        while True:
            yield {}
    def collate(self, batch):
        index = torch.randint(0, len(self.dataset), (1,)).item()
        return self.dataset[index]



class RandomCameraDataset(Dataset):
    def __init__(self, cfg: Any, mode, split: str) -> None:
        super().__init__()
        self.cfg: MultiviewsDataModuleConfig = cfg
        self.split = split

        if split == "val":
            self.n_views = self.cfg.n_val_views
        else:
            self.n_views = self.cfg.n_test_views

        assert mode in ['low_res', 'high_res']

        if mode == 'low_res':
            self.height = self.cfg.low_resolution
            self.width = self.cfg.low_resolution
        else:
            self.height = self.cfg.high_resolution
            self.width = self.cfg.high_resolution

        azimuth_deg: Float[Tensor, "B"]
        if self.split == "val":
            # make sure the first and last view are not the same
            azimuth_deg = torch.linspace(0, 360.0, self.n_views + 1)[: self.n_views]
        else:
            azimuth_deg = torch.linspace(0, 360.0, self.n_views)
        elevation_deg: Float[Tensor, "B"] = torch.full_like(
            azimuth_deg, self.cfg.eval_elevation_deg
        )
        camera_distances: Float[Tensor, "B"] = torch.full_like(
            elevation_deg, self.cfg.eval_camera_distance
        )

        elevation = elevation_deg * math.pi / 180
        azimuth = azimuth_deg * math.pi / 180

        # convert spherical coordinates to cartesian coordinates
        # right hand coordinate system, x back, y right, z up
        # elevation in (-90, 90), azimuth from +x to +y in (-180, 180)
        camera_positions: Float[Tensor, "B 3"] = torch.stack(
            [
                camera_distances * torch.cos(elevation) * torch.cos(azimuth),
                camera_distances * torch.cos(elevation) * torch.sin(azimuth),
                camera_distances * torch.sin(elevation),
            ],
            dim=-1,
        )

        # default scene center at origin
        center: Float[Tensor, "B 3"] = torch.zeros_like(camera_positions)
        # default camera up direction as +z
        up: Float[Tensor, "B 3"] = torch.as_tensor([0, 0, 1], dtype=torch.float32)[
            None, :
        ].repeat(self.cfg.eval_batch_size, 1)

        fovy_deg: Float[Tensor, "B"] = torch.full_like(
            elevation_deg, self.cfg.eval_fovy_deg
        )
        fovy = fovy_deg * math.pi / 180
        light_positions: Float[Tensor, "B 3"] = camera_positions

        lookat: Float[Tensor, "B 3"] = F.normalize(center - camera_positions, dim=-1)
        right: Float[Tensor, "B 3"] = F.normalize(torch.cross(lookat, up), dim=-1)
        up = F.normalize(torch.cross(right, lookat), dim=-1)
        c2w3x4: Float[Tensor, "B 3 4"] = torch.cat(
            [torch.stack([right, up, -lookat], dim=-1), camera_positions[:, :, None]],
            dim=-1,
        )
        c2w: Float[Tensor, "B 4 4"] = torch.cat(
            [c2w3x4, torch.zeros_like(c2w3x4[:, :1])], dim=1
        )
        c2w[:, 3, 3] = 1.0

        # get directions by dividing directions_unit_focal by focal length
        focal_length: Float[Tensor, "B"] = (
            0.5 * self.height / torch.tan(0.5 * fovy)
        )
        directions_unit_focal = get_ray_directions(
            H=self.height, W=self.width, focal=1.0
        )
        directions: Float[Tensor, "B H W 3"] = directions_unit_focal[
            None, :, :, :
        ].repeat(self.n_views, 1, 1, 1)
        directions[:, :, :, :2] = (
            directions[:, :, :, :2] / focal_length[:, None, None, None]
        )

        rays_o, rays_d = get_rays(directions, c2w, keepdim=True)
        proj_mtx: Float[Tensor, "B 4 4"] = get_projection_matrix(
            fovy, self.width / self.height, 0.1, 1000.0
        )  # FIXME: hard-coded near and far
        mvp_mtx: Float[Tensor, "B 4 4"] = get_mvp_matrix(c2w, proj_mtx)

        self.rays_o, self.rays_d = rays_o, rays_d
        self.mvp_mtx = mvp_mtx
        self.c2w = c2w
        self.camera_positions = camera_positions
        self.light_positions = light_positions
        self.elevation, self.azimuth = elevation, azimuth
        self.elevation_deg, self.azimuth_deg = elevation_deg, azimuth_deg
        self.camera_distances = camera_distances

    def __len__(self):
        return self.n_views

    def __getitem__(self, index):
        return {
            "index": index,
            "rays_o": self.rays_o[index],
            "rays_d": self.rays_d[index],
            "mvp_mtx": self.mvp_mtx[index],
            "c2w": self.c2w[index],
            "camera_positions": self.camera_positions[index],
            "light_positions": self.light_positions[index],
            "elevation": self.elevation_deg[index],
            "azimuth": self.azimuth_deg[index],
            "camera_distances": self.camera_distances[index],
            "height": self.height,
            "width": self.width,
        }

    def collate(self, batch):
        batch = torch.utils.data.default_collate(batch)
        batch.update({"height": self.height, "width": self.width})
        return batch

class MultiviewDatasetDoubleResolutionEval(Dataset):
    def __init__(self, cfg: Any, split: str) -> None:
        super().__init__()
        self.cfg: MultiviewsDataModuleConfig = cfg
        self.low_res_dataset = RandomCameraDataset(cfg, 'low_res', split)
        self.high_res_dataset = RandomCameraDataset(cfg, 'high_res', split)

    def __iter__(self):
        while True:
            yield {}

    def __len__(self):
        return len(self.low_res_dataset)

    def __getitem__(self, index):
        res = {'high_res': self.high_res_dataset[index], 'low_res': self.low_res_dataset[index]}
        return res

    def collate(self, batch):
        # batch = torch.utils.data.default_collate(batch)
        # for key in ['low_res','high_res']:
        #     batch[key]['height'] = batch[key]['height'][0]
        #     batch[key]['width'] = batch[key]['width'][0]
        res = batch[0]
        return res


@register("multiview-camera-sr-datamodule2")
class MultiviewSRDataModule(pl.LightningDataModule):
    cfg: MultiviewsDataModuleConfig

    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None) -> None:
        super().__init__()
        self.cfg = parse_structured(MultiviewsDataModuleConfig, cfg)

    def setup(self, stage=None) -> None:
        if stage in [None, "fit"]:
            cfg = copy.deepcopy(self.cfg)
            #training must not contain th gt images
            cfg.load_high_res_gt = False
            self.train_dataset = MultiviewDatasetDoubleResolutionTraining(cfg)
        if stage in [None, "fit", "validate"]:
            self.val_dataset = MultiviewDatasetDoubleResolutionEval(self.cfg, "val")
        if stage in [None, "test", "predict"]:
            self.test_dataset = MultiviewDatasetDoubleResolutionEval(self.cfg, "test")

    def prepare_data(self):
        pass

    def general_loader(self, dataset, batch_size, collate_fn=None) -> DataLoader:
        return DataLoader(
            dataset,
            num_workers=1,  # type: ignore
            batch_size=batch_size,
            collate_fn=collate_fn,
        )

    def train_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.train_dataset, batch_size=None, collate_fn=self.train_dataset.collate
        )

    def val_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.val_dataset, batch_size=1, collate_fn=self.val_dataset.collate
        )
        # return self.general_loader(self.train_dataset, batch_size=None, collate_fn=self.train_dataset.collate)

    def test_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.test_dataset, batch_size=1, collate_fn=self.test_dataset.collate
        )

    def predict_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.test_dataset, batch_size=1, collate_fn=self.test_dataset.collate
        )
