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
    low_resolution: int = 128
    high_resolution: int = 512
    pose_folder: str = ''
    batch_size: int = 1
    eval_batch_size: int = 1

    load_high_res_gt: bool = False
    bg_color: Tuple = (0.0, 0.0, 0.0)
    ngp_convention: bool = False

    shuffle_batch: int = -1
    shuffle_steps: int = -1



class MultiviewDataset(Dataset):
    def __init__(self, cfg: Any,mode,split = 'train',shuffle=False) -> None:
        super().__init__()
        self.cfg: MultiviewsDataModuleConfig = cfg

        assert self.cfg.batch_size == 1
        assert mode in ['low_res','high_res']

        # if os.path.exists(os.path.join(self.cfg.dataroot, 'transforms.json')):
        #     self.mode = 'colmap' # manually split, use view-interpolation for test.
        # elif os.path.exists(os.path.join(self.cfg.dataroot, 'transforms_train.json')):
        #     self.mode = 'blender' # provided split
        # else:
        #     raise NotImplementedError(f'[NeRFDataset] Cannot find transforms*.json under {self.cfg.dataroot}')
        # camera_dict = json.load(
        #     open(os.path.join(self.cfg.dataroot, "transforms_{}.json".format(split)), "r")
        # )

        poses_fname = sorted([os.path.join(self.cfg.pose_folder, f) for f in os.listdir(self.cfg.pose_folder)])
        batch_rays_list = []
        # H = args.render_res
        tmp_H = 128
        H = None
        ratio = 512 // tmp_H
        # ratio = 4
        c2w_lst = []
        K_lst = []
        img_f_lst = []
        for idx,p in enumerate(poses_fname):
            c2w = np.loadtxt(p).reshape(4, 4)
            c2w[:3, 3] *= 2.2
            c2w = np.array([
                [1, 0, 0, 0],
                [0, 0, -1, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1]
            ]) @ c2w
            c2w = np.array(c2w, dtype=np.float32)

            K = np.array([
                [560 / ratio, 0, tmp_H * 0.5],
                [0, 560 / ratio, tmp_H * 0.5],
                [0, 0, 1]
            ])
            img_f = os.path.join(self.cfg.dataroot,'{}.png'.format(idx))
            assert os.path.exists(img_f)
            c2w_lst.append(c2w)
            K_lst.append(K)
            img_f_lst.append(img_f)



        self.shuffle = shuffle
        # assert camera_dict["camera_model"] == "OPENCV"

        # frames = camera_dict["frames"]
        frames_proj = []
        frames_c2w = []
        frames_position = []
        frames_direction = []
        frames_img = []
        assert abs(self.cfg.high_resolution/self.cfg.low_resolution - 4) <= 1e-5
        img = cv2.imread(img_f_lst[0])[:, :, ::-1]
        H,W,C = img.shape
        assert H==W
        if mode == 'low_res':
            scale = W / self.cfg.low_resolution
        else:
            scale = W / self.cfg.high_resolution

        self.frame_w = round(W // scale)
        self.frame_h = round(H // scale)

        threestudio.info("Loading frames...")
        self.n_frames = len(img_f_lst)

        c2w_list = []
        for pose in tqdm(c2w_lst):
            # pose = np.array(frame['transform_matrix'], dtype=np.float32)
            # if self.cfg.ngp_convention:
            #     pose = nerf_matrix_to_ngp(pose)
            extrinsic: Float[Tensor, "4 4"] = torch.as_tensor(
                pose, dtype=torch.float32
            )
            c2w = extrinsic
            c2w_list.append(c2w)
        c2w_list = torch.stack(c2w_list, dim=0)


        for idx in tqdm(range(len(img_f_lst))):
            intrinsic: Float[Tensor, "4 4"] = torch.eye(4)
            # if 'fl_x' in frame:
            #     intrinsic[0, 0] = frame["fl_x"] / scale
            #     intrinsic[1, 1] = frame["fl_y"] / scale
            #     intrinsic[0, 2] = frame["cx"] / scale
            #     intrinsic[1, 2] = frame["cy"] / scale
            # else:
            # blender, assert in radians. already downscaled since we use H/W
            # fl_x = self.frame_w / (2 * np.tan(camera_dict['camera_angle_x'] / 2)) if 'camera_angle_x' in camera_dict else None
            # fl_y = self.frame_h / (2 * np.tan(camera_dict['camera_angle_y'] / 2)) if 'camera_angle_y' in camera_dict else None
            # if fl_x is None: fl_x = fl_y
            # if fl_y is None: fl_y = fl_x
            cx = self.frame_w / 2
            cy = self.frame_h / 2
            intrinsic[0, 0] = K_lst[idx][0,0]  / scale
            intrinsic[1, 1] = K_lst[idx][1,1]  / scale
            intrinsic[0, 2] = cx
            intrinsic[1, 2] = cy



            if (mode == 'low_res') or self.cfg.load_high_res_gt: #only load images in low res
                # frame_path = os.path.join(self.cfg.dataroot, frame["file_path"])
                # if self.mode == 'blender' and '.' not in os.path.basename(frame_path):
                #     frame_path += '.png'  # so silly...
                frame_path = img_f_lst[idx]
                img = cv2.imread(frame_path,cv2.IMREAD_UNCHANGED) / 255
                if img.shape[2] == 4:
                    img = img[:,:,[2,1,0,3]]
                    alpha = img[:,:,3:]
                    rgb = img[:,:,:3]
                    bg_color = np.array(self.cfg.bg_color).reshape((1,1,3))
                    img = rgb * alpha + (1-alpha)*bg_color
                elif img.shape[2] == 3:
                    img = img[:, :, ::-1].copy()
                else:
                    raise ValueError('Error')
                img = cv2.resize(img, (self.frame_w, self.frame_h), interpolation=cv2.INTER_AREA) #TODO: resize mode better be area
                img: Float[Tensor, "H W 3"] = torch.FloatTensor(img)


                frames_img.append(img)

            direction: Float[Tensor, "H W 3"] = get_ray_directions(
                self.frame_h,
                self.frame_w,
                (intrinsic[0, 0], intrinsic[1, 1]),
                (intrinsic[0, 2], intrinsic[1, 2]),
                use_pixel_centers=True,
            )

            c2w = c2w_list[idx]
            camera_position: Float[Tensor, "3"] = c2w[:3, 3:].reshape(-1)

            near = 0.1
            far = 1000.0
            proj = convert_proj(intrinsic, self.frame_h, self.frame_w, near, far)
            proj: Float[Tensor, "4 4"] = torch.FloatTensor(proj)
            frames_proj.append(proj)
            frames_c2w.append(c2w)
            frames_position.append(camera_position)
            frames_direction.append(direction)
        threestudio.info("Loaded frames.")

        self.frames_proj: Float[Tensor, "B 4 4"] = torch.stack(frames_proj, dim=0)
        self.frames_c2w: Float[Tensor, "B 4 4"] = torch.stack(frames_c2w, dim=0)
        self.frames_position: Float[Tensor, "B 3"] = torch.stack(frames_position, dim=0)
        self.frames_direction: Float[Tensor, "B H W 3"] = torch.stack(
            frames_direction, dim=0
        )
        self.frames_img = None
        if (mode == 'low_res') or self.cfg.load_high_res_gt:
            self.frames_img: Float[Tensor, "B H W 3"] = torch.stack(frames_img, dim=0)

        if self.cfg.ngp_convention:
            intrinsics_ngp = np.array([intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]])
            rays_ngp = get_rays_ngp(self.frames_c2w, intrinsics_ngp, self.frame_h, self.frame_w)
            self.rays_o = rays_ngp['rays_o'].reshape(self.frames_c2w.shape[0],self.frame_h, self.frame_w,-1)
            self.rays_d = rays_ngp['rays_d'].reshape(self.frames_c2w.shape[0],self.frame_h, self.frame_w,-1)
        else:
            self.rays_o, self.rays_d = get_rays(
                self.frames_direction, self.frames_c2w, keepdim=True
            )
        # print(self.rays_o[:,0,0])
        self.mvp_mtx: Float[Tensor, "B 4 4"] = get_mvp_matrix(
            self.frames_c2w, self.frames_proj
        )
        self.light_positions: Float[Tensor, "B 3"] = torch.zeros_like(
            self.frames_position
        )
        # self.rays_d *= (-1)
        if self.shuffle:
            if (self.cfg.shuffle_batch > 0) and (self.cfg.shuffle_steps > 0):
                number_of_imgs = self.rays_o.shape[0]
                all_pixels = (number_of_imgs * self.frame_h * self.frame_w)
                num_of_perms = math.ceil(self.cfg.shuffle_batch*self.cfg.shuffle_steps / all_pixels)
                self.randperm = torch.cat([torch.randperm(all_pixels).view(-1) for i in range(num_of_perms)],dim=0)
                self.perm_idx = 0

            self.rays_o = self.rays_o.reshape(-1,self.rays_o.shape[-1])
            self.rays_d = self.rays_d.reshape(-1, self.rays_d.shape[-1])
            self.frames_img = self.frames_img.reshape(-1, self.frames_img.shape[-1])

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
        # print(self.rays_o.shape)
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


@register("multiview-camera-sr-datamodule6")
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
