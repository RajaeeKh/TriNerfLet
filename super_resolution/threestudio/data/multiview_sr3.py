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

# from threestudio.data.dataset_llff.colmap import ColmapDataset
# from threestudio.data.dataset_llff.ray_utils import get_rays as get_rays_llff2

from threestudio.data.load_llff import load_llff_data

# Ray helpers
def get_rays_llff(H, W, K, c2w):
    i, j = torch.meshgrid(torch.linspace(0, W - 1, W),
                          torch.linspace(0, H - 1, H))  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i - K[0][2]) / K[0][0], -(j - K[1][2]) / K[1][1], -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3, :3],
                       -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    return rays_o, rays_d



def ndc_rays_llff(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d

    # Projection
    o0 = -1. / (W / (2. * focal)) * rays_o[..., 0] / rays_o[..., 2]
    o1 = -1. / (H / (2. * focal)) * rays_o[..., 1] / rays_o[..., 2]
    o2 = 1. + 2. * near / rays_o[..., 2]

    d0 = -1. / (W / (2. * focal)) * (rays_d[..., 0] / rays_d[..., 2] - rays_o[..., 0] / rays_o[..., 2])
    d1 = -1. / (H / (2. * focal)) * (rays_d[..., 1] / rays_d[..., 2] - rays_o[..., 1] / rays_o[..., 2])
    d2 = -2. * near / rays_o[..., 2]

    rays_o = torch.stack([o0, o1, o2], -1)
    rays_d = torch.stack([d0, d1, d2], -1)

    return rays_o, rays_d

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

    llffhold: int = 8
    llff_render_mode: bool = False
    llff_spherify: bool = False

    shuffle_batch: int = -1
    shuffle_steps: int = -1
    train_test_on_all_images: bool = False
    # camera_layout: str = "around"
    # camera_distance: float = -1
    # eval_interpolation: Optional[Tuple[int, int, int]] = None  # (0, 1, 30)



class MultiviewDataset(Dataset):
    def __init__(self, cfg: Any,mode,split = 'train',shuffle=False) -> None:
        super().__init__()
        self.cfg: MultiviewsDataModuleConfig = cfg

        assert self.cfg.batch_size == 1
        assert mode in ['low_res','high_res']
        if mode == 'low_res':
            downscale = self.cfg.low_resolution_factor
        else:
            downscale = self.cfg.high_resolution_factor
        self.downscale = downscale
        self.type = split
        self.shuffle = shuffle
        # self.aux_dataset = ColmapDataset(self.cfg.dataroot, split=split, downsample=1 / self.downscale)

        images, poses, bds, render_poses, i_test = load_llff_data(self.cfg.dataroot, self.downscale,
                                                                  recenter=True, bd_factor=.75,
                                                                  spherify=self.cfg.llff_spherify)

        # TODO: self.fp16 and alpha chnnel

        hwf = poses[0, :3, -1]
        poses = poses[:, :3, :4]
        print('Loaded llff', images.shape, render_poses.shape, hwf, self.cfg.dataroot)

        if images.shape[3] == 4:
            alpha = images[:,:,:,3:]
            rgb = images[:,:,:,:3]
            bg_color = np.array(self.cfg.bg_color).reshape((1, 1, 1, 3))
            images = rgb * alpha + (1 - alpha) * bg_color

        if not isinstance(i_test, list):
            i_test = [i_test]

        if self.cfg.llffhold > 0:
            print('Auto LLFF holdout,', self.cfg.llffhold)
            i_test = np.arange(images.shape[0])[::self.cfg.llffhold]

        i_val = i_test
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                            (i not in i_test and i not in i_val)])

        if self.cfg.train_test_on_all_images:
            i_train = np.arange(images.shape[0])
            i_val = np.arange(images.shape[0])
            i_test = np.arange(images.shape[0])

        print('DEFINING BOUNDS')
        near = 0.
        far = 1.
        print('NEAR FAR', near, far)
        self.near = near

        H, W, focal = hwf
        H, W = int(H), int(W)
        self.hwf = [H, W, focal]
        self.H = H
        self.W = W

        self.K = np.array([
            [focal, 0, 0.5 * W],
            [0, focal, 0.5 * H],
            [0, 0, 1]
        ])

        self.frame_w = W
        self.frame_h = H
        intrinsic = self.K

        self.calculate_limit(torch.Tensor(poses[i_train][:, :3, :4]))
        self.select_data(images, poses, render_poses, i_train, i_val, i_test)
        self.generate_rays()

        frames_c2w = []
        frames_proj = []
        frames_position = []
        rays_o = []
        rays_d = []
        frames_img = []


        for img_idx in range(self.poses.shape[0]):

            pose = self.poses[img_idx]
            # rays_o_tmp, rays_d_tmp = self.rays_o[img_idx],self.rays_d[img_idx]

            c2w = pose
            camera_position: Float[Tensor, "3"] = c2w[:3, 3:].reshape(-1)

            proj = convert_proj(intrinsic, self.frame_h, self.frame_w, near, far)
            proj: Float[Tensor, "4 4"] = torch.FloatTensor(proj)

            frames_proj.append(proj)
            frames_c2w.append(c2w)
            frames_position.append(camera_position)

            # rays_o.append(rays_o_tmp.view(self.frame_h, self.frame_w,-1 ))
            # rays_d.append(rays_d_tmp.view(self.frame_h, self.frame_w,-1 ))
            # if ((mode == 'low_res') or self.cfg.load_high_res_gt) and (self.images is not None):
                # frames_img.append(self.images[img_idx].view(self.frame_h, self.frame_w,-1 ))

        self.frames_proj: Float[Tensor, "B 4 4"] = torch.stack(frames_proj, dim=0)
        self.frames_c2w: Float[Tensor, "B 4 4"] = torch.stack(frames_c2w, dim=0)
        self.frames_position: Float[Tensor, "B 3"] = torch.stack(frames_position, dim=0)
        self.frames_img = None
        if ((mode == 'low_res') or self.cfg.load_high_res_gt) and (self.images is not None):
            # self.frames_img: Float[Tensor, "B H W 3"] = torch.stack(frames_img, dim=0)
            self.frames_img = self.images
            self.images = None

        # self.rays_o = torch.stack(rays_o, dim=0)
        # self.rays_d = torch.stack(rays_d, dim=0)
        self.mvp_mtx: Float[Tensor, "B 4 4"] = get_mvp_matrix(
            self.frames_c2w, self.frames_proj
        )
        self.light_positions: Float[Tensor, "B 3"] = torch.zeros_like(
            self.frames_position
        )
        if self.shuffle:
            if (self.cfg.shuffle_batch > 0) and (self.cfg.shuffle_steps > 0):
                number_of_imgs = self.rays_o.shape[0]
                all_pixels = (number_of_imgs * self.frame_h * self.frame_w)
                num_of_perms = math.ceil(self.cfg.shuffle_batch*self.cfg.shuffle_steps / all_pixels)
                self.randperm = torch.cat([torch.randperm(all_pixels).view(-1) for i in range(num_of_perms)],dim=0)
                self.perm_idx = 0

            self.rays_o = self.rays_o.view(-1,self.rays_o.shape[-1])
            self.rays_d = self.rays_d.view(-1, self.rays_d.shape[-1])
            self.frames_img = self.frames_img.view(-1, self.frames_img.shape[-1])



    def calculate_limit(self,poses):
        rays_o, rays_d = self.get_rays(poses)
        limit = torch.cat([rays_o, rays_o + rays_d], dim=0).abs().max()
        print('all limit = {}'.format(limit))
        self.limit = limit

    def get_rays(self,poses):
        rays_o_lst = []
        rays_d_lst = []
        camera_max_dist = poses[:, :3, -1].norm(dim=-1).max()
        print('camera_max_dist = {}'.format(camera_max_dist.item()))
        for i in range(poses.shape[0]):
            pose = poses[i]
            # pose = torch.from_numpy(nerf_matrix_to_ngp(pose, scale=self.scale, offset=self.offset))
            rays_o, rays_d = get_rays_llff(self.H, self.W, self.K, pose)  # (H, W, 3), (H, W, 3)
            rays_o, rays_d = ndc_rays_llff(self.H, self.W, self.K[0][0], 1., rays_o,
                                           rays_d)
            if torch.isnan(rays_o).any() or torch.isinf(rays_o).any() or torch.isnan(rays_d).any() or torch.isinf(rays_d).any():
                print('*********************************************************************************************')
                print('found nan in frame {}'.format(i))
                print('*********************************************************************************************')
            # rays_d = F.normalize(rays_d,dim=-1)
            rays_o_lst.append(rays_o)
            rays_d_lst.append(rays_d)
        rays_o = torch.stack(rays_o_lst)
        rays_d = torch.stack(rays_d_lst)
        return rays_o,rays_d
    def generate_rays(self):


        camera_max_dist = self.poses[:, :3, -1].norm(dim=-1).max()
        print('camera_max_dist = {}'.format(camera_max_dist.item()))

        self.rays_o,self.rays_d = self.get_rays(self.poses)
        self.rays_o /= (self.limit) # all scene now resides in [-1,1]
        self.rays_d /= (self.limit)
        limit_post = torch.cat([self.rays_o, self.rays_o + self.rays_d], dim=0).abs().max()
        print('limit_post = {}'.format(limit_post))
        # print('hhhh')
    def select_data(self,images, poses, render_poses,i_train,i_val,i_test):
        self.images = None
        if self.type == 'train':
            self.images = images[i_train]
            self.poses = poses[i_train]
        elif self.type == 'val':
            self.images = images[i_val]
            self.poses = poses[i_val]
        elif self.type == 'test':
            if self.cfg.llff_render_mode:
                self.poses = render_poses
            else:
                self.images = images[i_test]
                self.poses = poses[i_test]
        else:
            print('unverified mode: {}'.format(self.type))
            self.images = images
            self.poses = poses

        self.poses = torch.Tensor(self.poses[:,:3,:4])
        if self.images is not None:
            self.images = torch.Tensor(self.images)

    def __len__(self):
        return self.frames_proj.shape[0]

    def __getitem__(self, index):
        if self.shuffle:
            if (self.cfg.shuffle_batch > 0) and (self.cfg.shuffle_steps > 0):
                # print('perm_idx = {}'.format(self.perm_idx))
                inds = self.randperm[self.perm_idx*self.cfg.shuffle_batch:(self.perm_idx+1)*self.cfg.shuffle_batch]
                self.perm_idx += 1
            else:
                inds = np.random.choice(self.rays_o.shape[0],self.frame_h*self.frame_w,replace=False)
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
    def __init__(self, cfg: Any,split = 'train',add_shuffle = False) -> None:
        super().__init__()
        self.low_res_dataset = MultiviewDataset(cfg,'low_res',split)
        self.high_res_dataset = MultiviewDataset(cfg, 'high_res', split)
        self.low_res_dataset_shuf = None
        if add_shuffle:
            self.low_res_dataset_shuf = MultiviewDataset(cfg,'low_res',split,shuffle=True)
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
                                    add_shuffle = (self.cfg.shuffle_batch > 0) and (self.cfg.shuffle_steps > 0))

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


@register("multiview-camera-sr-datamodule_llff2")
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
