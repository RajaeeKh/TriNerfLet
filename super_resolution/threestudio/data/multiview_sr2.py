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
from threestudio.utils.ops import get_mvp_matrix, get_ray_directions, get_rays
from threestudio.utils.typing import *

from threestudio.data.dataset_llff.colmap import ColmapDataset
from threestudio.data.dataset_llff.ray_utils import get_rays as get_rays_llff2


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
    low_resolution_factor: int = 8
    high_resolution_factor: int = 2
    batch_size: int = 1
    eval_batch_size: int = 1
    latent_scale: int=1
    load_high_res_gt: bool = False
    bg_color: Tuple = (0.0, 0.0, 0.0)
    # camera_layout: str = "around"
    # camera_distance: float = -1
    # eval_interpolation: Optional[Tuple[int, int, int]] = None  # (0, 1, 30)



class MultiviewDataset(Dataset):
    def __init__(self, cfg: Any,mode,split = 'train') -> None:
        super().__init__()
        self.cfg: MultiviewsDataModuleConfig = cfg

        assert self.cfg.batch_size == 1
        assert mode in ['low_res','high_res']
        if mode == 'low_res':
            downscale = self.cfg.low_resolution_factor
        else:
            downscale = self.cfg.high_resolution_factor
        self.downscale = downscale
        self.aux_dataset = ColmapDataset(self.cfg.dataroot, split=split, downsample=1 / self.downscale)

        W, H = self.aux_dataset.img_wh
        K = self.aux_dataset.K
        self.frame_w = W
        self.frame_h = H
        intrinsic = K

        frames_c2w = []
        frames_proj = []
        frames_position = []
        rays_o = []
        rays_d = []
        frames_img = []


        for img_idx in range(len(self.aux_dataset)):
            item = self.aux_dataset[img_idx]
            pose = item['pose']
            rays_o_tmp, rays_d_tmp = get_rays_llff2(self.aux_dataset.directions, pose)

            c2w = pose
            camera_position: Float[Tensor, "3"] = c2w[:3, 3:].reshape(-1)

            near = 0.1
            far = 1000.0
            proj = convert_proj(intrinsic, self.frame_h, self.frame_w, near, far)
            proj: Float[Tensor, "4 4"] = torch.FloatTensor(proj)

            frames_proj.append(proj)
            frames_c2w.append(c2w)
            frames_position.append(camera_position)

            rays_o.append(rays_o_tmp.view(self.frame_h, self.frame_w,-1 ))
            rays_d.append(rays_d_tmp.view(self.frame_h, self.frame_w,-1 ))
            if (mode == 'low_res') or self.cfg.load_high_res_gt:
                frames_img.append(item['rgb'].view(self.frame_h, self.frame_w,-1 ))

        self.frames_proj: Float[Tensor, "B 4 4"] = torch.stack(frames_proj, dim=0)
        self.frames_c2w: Float[Tensor, "B 4 4"] = torch.stack(frames_c2w, dim=0)
        self.frames_position: Float[Tensor, "B 3"] = torch.stack(frames_position, dim=0)
        self.frames_img = None
        if (mode == 'low_res') or self.cfg.load_high_res_gt:
            self.frames_img: Float[Tensor, "B H W 3"] = torch.stack(frames_img, dim=0)

        self.rays_o = torch.stack(rays_o, dim=0)
        self.rays_d = torch.stack(rays_d, dim=0)
        self.mvp_mtx: Float[Tensor, "B 4 4"] = get_mvp_matrix(
            self.frames_c2w, self.frames_proj
        )
        self.light_positions: Float[Tensor, "B 3"] = torch.zeros_like(
            self.frames_position
        )

    def __len__(self):
        return self.frames_proj.shape[0]

    def __getitem__(self, index):
        res = {
            "index": index,
            "rays_o": self.rays_o[index: index + 1],
            "rays_d": self.rays_d[index: index + 1],
            "mvp_mtx": self.mvp_mtx[index: index + 1],
            "c2w": self.frames_c2w[index: index + 1],
            "camera_positions": self.frames_position[index: index + 1],
            "light_positions": self.light_positions[index: index + 1],
            "height": self.frame_h,
            "width": self.frame_w,
        }
        # print('rays_o.shape1 = {}'.format(res['rays_o'].shape))
        # print('light_positions.shape1 = {}'.format(res['light_positions'].shape))
        if self.frames_img is not None:
            res['gt_rgb'] = self.frames_img[index: index + 1]
        return res


class MultiviewDatasetDoubleResolution(Dataset):
    def __init__(self, cfg: Any,split = 'train') -> None:
        super().__init__()
        self.low_res_dataset = MultiviewDataset(cfg,'low_res',split)
        self.high_res_dataset = MultiviewDataset(cfg, 'high_res', split)
        assert len(self.low_res_dataset) == len(self.high_res_dataset)

    def __len__(self):
        return len(self.low_res_dataset)

    def __getitem__(self, index):
        res = {'high_res' : self.high_res_dataset[index],'low_res':self.low_res_dataset[index]}
        return res

class MultiviewDatasetDoubleResolutionTraining(IterableDataset):
    def __init__(self, cfg: Any) -> None:
        super().__init__()
        self.cfg: MultiviewsDataModuleConfig = cfg
        split = 'train'
        self.dataset = MultiviewDatasetDoubleResolution(cfg,split)

    def __iter__(self):
        while True:
            yield {}
    def collate(self, batch):
        index = torch.randint(0, len(self.dataset), (1,)).item()
        return self.dataset[index]


class MultiviewDatasetDoubleResolutionEval(Dataset):
    def __init__(self, cfg: Any, split: str) -> None:
        super().__init__()
        self.cfg: MultiviewsDataModuleConfig = cfg
        self.dataset = MultiviewDatasetDoubleResolution(cfg,split)

    def __iter__(self):
        while True:
            yield {}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        res = self.dataset[index]
        return res

    def collate(self, batch):
        # batch = torch.utils.data.default_collate(batch)
        # for key in ['low_res','high_res']:
        #     batch[key]['height'] = batch[key]['height'][0]
        #     batch[key]['width'] = batch[key]['width'][0]
        res = batch[0]
        return res


@register("multiview-camera-sr-datamodule_llff")
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
