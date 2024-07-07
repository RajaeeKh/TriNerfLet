import os
import cv2
import glob
import json
from cv2 import transform
import tqdm
import numpy as np
from scipy.spatial.transform import Slerp, Rotation
import torch.nn.functional as F
import math
import trimesh

import torch
from torch.utils.data import DataLoader

try:
    from .utils import get_rays
except:
    from nerf.utils import get_rays


# ref: https://github.com/NVlabs/instant-ngp/blob/b76004c8cf478880227401ae763be4c02f80b62f/include/neural-graphics-primitives/nerf_loader.h#L50
def nerf_matrix_to_ngp(pose, scale=0.33, offset=[0, 0, 0]):
    # for the fox dataset, 0.33 scales camera radius to ~ 2
    new_pose = np.array([
        [pose[1, 0], -pose[1, 1], -pose[1, 2], pose[1, 3] * scale + offset[0]],
        [pose[2, 0], -pose[2, 1], -pose[2, 2], pose[2, 3] * scale + offset[1]],
        [pose[0, 0], -pose[0, 1], -pose[0, 2], pose[0, 3] * scale + offset[2]],
        [0, 0, 0, 1],
    ], dtype=np.float32)
    return new_pose


def visualize_poses(poses, size=0.1):
    # poses: [B, 4, 4]

    axes = trimesh.creation.axis(axis_length=4)
    box = trimesh.primitives.Box(extents=(2, 2, 2)).as_outline()
    box.colors = np.array([[128, 128, 128]] * len(box.entities))
    objects = [axes, box]

    for pose in poses:
        # a camera is visualized with 8 line segments.
        pos = pose[:3, 3]
        a = pos + size * pose[:3, 0] + size * pose[:3, 1] + size * pose[:3, 2]
        b = pos - size * pose[:3, 0] + size * pose[:3, 1] + size * pose[:3, 2]
        c = pos - size * pose[:3, 0] - size * pose[:3, 1] + size * pose[:3, 2]
        d = pos + size * pose[:3, 0] - size * pose[:3, 1] + size * pose[:3, 2]

        dir = (a + b + c + d) / 4 - pos
        dir = dir / (np.linalg.norm(dir) + 1e-8)
        o = pos + dir * 3

        segs = np.array([[pos, a], [pos, b], [pos, c], [pos, d], [a, b], [b, c], [c, d], [d, a], [pos, o]])
        segs = trimesh.load_path(segs)
        objects.append(segs)

    trimesh.Scene(objects).show()


def rand_poses(size, device, radius=1, theta_range=[np.pi/3, 2*np.pi/3], phi_range=[0, 2*np.pi]):
    ''' generate random poses from an orbit camera
    Args:
        size: batch size of generated poses.
        device: where to allocate the output.
        radius: camera radius
        theta_range: [min, max], should be in [0, \pi]
        phi_range: [min, max], should be in [0, 2\pi]
    Return:
        poses: [size, 4, 4]
    '''
    
    def normalize(vectors):
        return vectors / (torch.norm(vectors, dim=-1, keepdim=True) + 1e-10)

    thetas = torch.rand(size, device=device) * (theta_range[1] - theta_range[0]) + theta_range[0]
    phis = torch.rand(size, device=device) * (phi_range[1] - phi_range[0]) + phi_range[0]

    centers = torch.stack([
        radius * torch.sin(thetas) * torch.sin(phis),
        radius * torch.cos(thetas),
        radius * torch.sin(thetas) * torch.cos(phis),
    ], dim=-1) # [B, 3]

    # lookat
    forward_vector = - normalize(centers)
    up_vector = torch.FloatTensor([0, -1, 0]).to(device).unsqueeze(0).repeat(size, 1) # confused at the coordinate system...
    right_vector = normalize(torch.cross(forward_vector, up_vector, dim=-1))
    up_vector = normalize(torch.cross(right_vector, forward_vector, dim=-1))

    poses = torch.eye(4, dtype=torch.float, device=device).unsqueeze(0).repeat(size, 1, 1)
    poses[:, :3, :3] = torch.stack((right_vector, up_vector, forward_vector), dim=-1)
    poses[:, :3, 3] = centers

    return poses


class NeRFDataset:
    def __init__(self, opt, device, type='train', n_test=120):
        super().__init__()
        
        self.opt = opt
        self.device = device
        self.type = type # train, val, test
        self.downscale = opt.downscale
        self.root_path = opt.path
        self.preload = opt.preload # preload data into GPU
        self.scale = opt.scale # camera radius scale to make sure camera are inside the bounding box.
        self.offset = opt.offset # camera offset
        self.bound = opt.bound # bounding box half length, also used as the radius to random sample poses.
        self.fp16 = opt.fp16 # if preload, load into fp16.

        self.training = self.type in ['train', 'all', 'trainval']
        self.num_rays = self.opt.num_rays if self.training else -1

        self.rand_pose = opt.rand_pose

        downscale = self.downscale

        # auto-detect transforms.json and split mode.
        if os.path.exists(os.path.join(self.root_path, 'transforms.json')):
            self.mode = 'colmap' # manually split, use view-interpolation for test.
        elif os.path.exists(os.path.join(self.root_path, 'transforms_train.json')):
            self.mode = 'blender' # provided split
        else:
            raise NotImplementedError(f'[NeRFDataset] Cannot find transforms*.json under {self.root_path}')

        # load nerf-compatible format data.
        if self.mode == 'colmap':
            with open(os.path.join(self.root_path, 'transforms.json'), 'r') as f:
                transform = json.load(f)
        elif self.mode == 'blender':
            # load all splits (train/valid/test), this is what instant-ngp in fact does...
            if type == 'all':
                transform_paths = glob.glob(os.path.join(self.root_path, '*.json'))
                transform = None
                for transform_path in transform_paths:
                    with open(transform_path, 'r') as f:
                        tmp_transform = json.load(f)
                        if transform is None:
                            transform = tmp_transform
                        else:
                            transform['frames'].extend(tmp_transform['frames'])
            # load train and val split
            elif type == 'trainval':
                with open(os.path.join(self.root_path, f'transforms_train.json'), 'r') as f:
                    transform = json.load(f)
                with open(os.path.join(self.root_path, f'transforms_val.json'), 'r') as f:
                    transform_val = json.load(f)
                transform['frames'].extend(transform_val['frames'])
            # only load one specified split
            else:
                with open(os.path.join(self.root_path, f'transforms_{type}.json'), 'r') as f:
                    transform = json.load(f)

        else:
            raise NotImplementedError(f'unknown dataset mode: {self.mode}')

        # load image size
        if 'h' in transform and 'w' in transform:
            self.H = int(transform['h']) // downscale
            self.W = int(transform['w']) // downscale
        else:
            # we have to actually read an image to get H and W later.
            self.H = self.W = None
        
        # read images
        frames = transform["frames"]
        #frames = sorted(frames, key=lambda d: d['file_path']) # why do I sort...
        
        # for colmap, manually interpolate a test set.
        if self.mode == 'colmap' and type == 'test':
            
            # choose two random poses, and interpolate between.
            # f0, f1 = np.random.choice(frames, 2, replace=False)
            f0,f1 = frames[0],frames[16]
            pose0 = nerf_matrix_to_ngp(np.array(f0['transform_matrix'], dtype=np.float32), scale=self.scale, offset=self.offset) # [4, 4]
            pose1 = nerf_matrix_to_ngp(np.array(f1['transform_matrix'], dtype=np.float32), scale=self.scale, offset=self.offset) # [4, 4]
            rots = Rotation.from_matrix(np.stack([pose0[:3, :3], pose1[:3, :3]]))
            slerp = Slerp([0, 1], rots)

            self.poses = []
            self.images = None
            for i in range(n_test + 1):
                ratio = np.sin(((i / n_test) - 0.5) * np.pi) * 0.5 + 0.5
                pose = np.eye(4, dtype=np.float32)
                pose[:3, :3] = slerp(ratio).as_matrix()
                pose[:3, 3] = (1 - ratio) * pose0[:3, 3] + ratio * pose1[:3, 3]
                self.poses.append(pose)

        else:
            # for colmap, manually split a valid set (the first frame).
            if self.mode == 'colmap':
                if type == 'train':
                    frames = frames[1:]
                elif type == 'val':
                    frames = frames[:1]
                # else 'all' or 'trainval' : use all frames
            
            self.poses = []
            self.images = []
            for f in tqdm.tqdm(frames, desc=f'Loading {type} data'):
                f_path = os.path.join(self.root_path, f['file_path'])
                if self.mode == 'blender' and '.' not in os.path.basename(f_path):
                    f_path += '.png' # so silly...

                # there are non-exist paths in fox...
                if not os.path.exists(f_path):
                    continue
                
                pose = np.array(f['transform_matrix'], dtype=np.float32) # [4, 4]
                pose = nerf_matrix_to_ngp(pose, scale=self.scale, offset=self.offset)

                image = cv2.imread(f_path, cv2.IMREAD_UNCHANGED) # [H, W, 3] o [H, W, 4]
                if self.H is None or self.W is None:
                    self.H = image.shape[0] // downscale
                    self.W = image.shape[1] // downscale

                # add support for the alpha channel as a mask.
                if image.shape[-1] == 3: 
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)

                if image.shape[0] != self.H or image.shape[1] != self.W:
                    image = cv2.resize(image, (self.W, self.H), interpolation=cv2.INTER_AREA)
                    
                image = image.astype(np.float32) / 255 # [H, W, 3/4]

                self.poses.append(pose)
                self.images.append(image)
            
        self.poses = torch.from_numpy(np.stack(self.poses, axis=0)) # [N, 4, 4]
        if self.images is not None:
            self.images = torch.from_numpy(np.stack(self.images, axis=0)) # [N, H, W, C]
        
        # calculate mean radius of all camera poses
        self.radius = self.poses[:, :3, 3].norm(dim=-1).mean(0).item()
        #print(f'[INFO] dataset camera poses: radius = {self.radius:.4f}, bound = {self.bound}')

        # initialize error_map
        if self.training and self.opt.error_map:
            self.error_map = torch.ones([self.images.shape[0], 128 * 128], dtype=torch.float) # [B, 128 * 128], flattened for easy indexing, fixed resolution...
        else:
            self.error_map = None

        # [debug] uncomment to view all training poses.
        # visualize_poses(self.poses.numpy())

        # [debug] uncomment to view examples of randomly generated poses.
        # visualize_poses(rand_poses(100, self.device, radius=self.radius).cpu().numpy())

        if self.preload:
            self.poses = self.poses.to(self.device)
            if self.images is not None:
                # TODO: linear use pow, but pow for half is only available for torch >= 1.10 ?
                if self.fp16 and self.opt.color_space != 'linear':
                    dtype = torch.half
                else:
                    dtype = torch.float
                self.images = self.images.to(dtype).to(self.device)
            if self.error_map is not None:
                self.error_map = self.error_map.to(self.device)

        # load intrinsics
        if 'fl_x' in transform or 'fl_y' in transform:
            fl_x = (transform['fl_x'] if 'fl_x' in transform else transform['fl_y']) / downscale
            fl_y = (transform['fl_y'] if 'fl_y' in transform else transform['fl_x']) / downscale
        elif 'camera_angle_x' in transform or 'camera_angle_y' in transform:
            # blender, assert in radians. already downscaled since we use H/W
            fl_x = self.W / (2 * np.tan(transform['camera_angle_x'] / 2)) if 'camera_angle_x' in transform else None
            fl_y = self.H / (2 * np.tan(transform['camera_angle_y'] / 2)) if 'camera_angle_y' in transform else None
            if fl_x is None: fl_x = fl_y
            if fl_y is None: fl_y = fl_x
        else:
            raise RuntimeError('Failed to load focal length, please check the transforms.json!')

        cx = (transform['cx'] / downscale) if 'cx' in transform else (self.W / 2)
        cy = (transform['cy'] / downscale) if 'cy' in transform else (self.H / 2)
    
        self.intrinsics = np.array([fl_x, fl_y, cx, cy])


    def collate(self, index):

        B = len(index) # a list of length 1

        # random pose without gt images.
        if self.rand_pose == 0 or index[0] >= len(self.poses):

            poses = rand_poses(B, self.device, radius=self.radius)

            # sample a low-resolution but full image for CLIP
            s = np.sqrt(self.H * self.W / self.num_rays) # only in training, assert num_rays > 0
            rH, rW = int(self.H / s), int(self.W / s)
            rays = get_rays(poses, self.intrinsics / s, rH, rW, -1)

            return {
                'H': rH,
                'W': rW,
                'rays_o': rays['rays_o'],
                'rays_d': rays['rays_d'],    
            }

        poses = self.poses[index].to(self.device) # [B, 4, 4]

        error_map = None if self.error_map is None else self.error_map[index]
        
        rays = get_rays(poses, self.intrinsics, self.H, self.W, self.num_rays, error_map, self.opt.patch_size)

        results = {
            'H': self.H,
            'W': self.W,
            'rays_o': rays['rays_o'],
            'rays_d': rays['rays_d'],
        }

        if self.images is not None:
            images = self.images[index].to(self.device) # [B, H, W, 3/4]
            if self.training:
                C = images.shape[-1]
                images = torch.gather(images.view(B, -1, C), 1, torch.stack(C * [rays['inds']], -1)) # [B, N, 3/4]
            results['images'] = images
        
        # need inds to update error_map
        if error_map is not None:
            results['index'] = index
            results['inds_coarse'] = rays['inds_coarse']
            
        return results

    def dataloader(self):
        size = len(self.poses)
        if self.training and self.rand_pose > 0:
            size += size // self.rand_pose # index >= size means we use random pose.
        loader = DataLoader(list(range(size)), batch_size=1, collate_fn=self.collate, shuffle=self.training, num_workers=0)
        loader._data = self # an ugly fix... we need to access error_map & poses in trainer.
        loader.has_gt = self.images is not None
        return loader


try:
    from .load_llff import load_llff_data
except:
    from nerf.load_llff import load_llff_data
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


def get_dataset(opt):
    if opt.llff_dataset:
        return NeRFDatasetLLFF
        # return NeRFDatasetLLFF2
    if opt.topia_dataset:
        return NeRFDatasetTopia
    return NeRFDataset
#TODO: check bound box and near-far
class NeRFDatasetLLFF:
    def __init__(self, opt, device, type='train', n_test=10):
        super().__init__()

        self.opt = opt
        # self.device = device
        self.device = torch.device('cpu')
        self.type = type  # train, val, test
        self.downscale = opt.downscale
        self.root_path = opt.path
        self.preload = opt.preload  # preload data into GPU
        self.scale = opt.scale  # camera radius scale to make sure camera are inside the bounding box.
        self.offset = opt.offset  # camera offset
        self.bound = opt.bound  # bounding box half length, also used as the radius to random sample poses.
        self.fp16 = opt.fp16  # if preload, load into fp16.

        self.llff_spherify = opt.llff_spherify
        self.llffhold = opt.llff_hold
        self.llff_render_mode = opt.llff_render_mode
        self.llff_render_all_test = opt.llff_render_all_test
        self.llff_ndc = opt.llff_ndc

        self.training = self.type in ['train', 'all', 'trainval']
        self.num_rays = self.opt.num_rays if self.training else -1


        images, poses, bds, render_poses, i_test = load_llff_data(self.root_path, self.downscale,
                                                                  recenter=True, bd_factor=.75,
                                                                  spherify=self.llff_spherify)

        #TODO: self.fp16 and alpha chnnel

        hwf = poses[0, :3, -1]
        poses = poses[:, :3, :4]
        print('Loaded llff', images.shape, render_poses.shape, hwf, self.root_path)
        if not isinstance(i_test, list):
            i_test = [i_test]

        if self.llffhold > 0:
            print('Auto LLFF holdout,', self.llffhold)
            i_test = np.arange(images.shape[0])[::self.llffhold]



        i_val = i_test
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                            (i not in i_test and i not in i_val)])

        if self.llff_render_all_test:
            i_test = np.arange(images.shape[0])
            i_val = i_test

        print('DEFINING BOUNDS')
        # assert opt.min_near < 1e-5
        # assert abs(opt.bound - 1) < 1e-5
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

        #TODO: make sure
        # self.intrinsics = np.array([focal, focal, W // 2, H // 2])
        self.error_map = None

        self.calculate_limit(torch.Tensor(poses[i_train][:, :3, :4]).to(self.device))
        self.select_data(images, poses, render_poses,i_train,i_val,i_test)
        self.generate_rays()

        self.device = device


    def calculate_limit(self,poses):
        self.limit = 1
        if self.llff_ndc:
            # assert self.bound == 1
            rays_o, rays_d = self.get_rays(poses)
            limit = torch.cat([rays_o, rays_o + rays_d], dim=0).abs().max()
            print('all limit = {}'.format(limit))
            self.limit = limit

    def get_rays(self,poses):
        rays_o_lst = []
        rays_d_lst = []
        camera_dist = poses[:, :3, -1].norm(dim=-1).view(-1)
        print('camera dists:')
        print(camera_dist.tolist())
        print('camera_max_dist = {}'.format(camera_dist.max().item()))
        for i in range(poses.shape[0]):
            pose = poses[i]
            rays_o, rays_d = get_rays_llff(self.H, self.W, self.K, pose)  # (H, W, 3), (H, W, 3)
            if self.llff_ndc:
                rays_o, rays_d = ndc_rays_llff(self.H, self.W, self.K[0][0], 1., rays_o,
                                               rays_d)
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
        # print(self.rays_o.shape)
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
            if self.llff_render_mode:
                self.poses = render_poses
            else:
                self.images = images[i_test]
                self.poses = poses[i_test]
        else:
            print('unverified mode: {}'.format(self.type))
            self.images = images
            self.poses = poses

        # assert self.scale == 1
        assert sum([val == 0 for val in self.offset]) == 3

        # # TODO why not working
        # poses_lst = []
        # for i in range(self.poses.shape[0]):
        #     pose = nerf_matrix_to_ngp(self.poses[i], scale=self.scale, offset=self.offset)
        #     poses_lst.append(pose)
        # self.poses = np.stack(poses_lst,0)

        self.poses = torch.Tensor(self.poses[:,:3,:4]).to(self.device)
        if self.images is not None:
            self.images = torch.Tensor(self.images).to(self.device)

    def collate(self, index):

        B = len(index)  # a list of length 1
        assert B==1
        rays_o = self.rays_o[index]
        rays_o = rays_o.view(B,-1,rays_o.shape[-1])
        rays_d = self.rays_d[index]
        rays_d = rays_d.view(B, -1, rays_d.shape[-1])

        results = {
            'H': self.H,
            'W': self.W,
            'rays_o': rays_o,
            'rays_d': rays_d,
        }

        if self.images is not None:
            images = self.images[index]  # [ H, W, 3/4]
            C = images.shape[-1]
            if self.training:
                images = images.view(B, -1, C)
            results['images'] = images

        if self.num_rays > 0:
            inds = torch.randperm(rays_o.shape[-1])[:self.num_rays]
            for key in ['rays_o', 'rays_d', 'images']:
                if key in results:
                    results[key] = results[key][:,inds]

        for key in ['rays_o', 'rays_d', 'images']:
            if key in results:
                results[key] = results[key].to(self.device)
        return results

    def dataloader(self):
        size = len(self.poses)
        loader = DataLoader(list(range(size)), batch_size=1, collate_fn=self.collate, shuffle=self.training,
                            num_workers=0)
        loader._data = self  # an ugly fix... we need to access error_map & poses in trainer.
        loader.has_gt = self.images is not None
        return loader



class NeRFDatasetTopia:
    def __init__(self, opt, device, type='train', n_test=10):
        super().__init__()

        self.opt = opt
        # self.device = device
        self.device = torch.device('cpu')
        self.type = type  # train, val, test
        self.downscale = opt.downscale
        self.root_path = opt.path
        self.preload = opt.preload  # preload data into GPU
        self.scale = opt.scale  # camera radius scale to make sure camera are inside the bounding box.
        self.offset = opt.offset  # camera offset
        self.bound = opt.bound  # bounding box half length, also used as the radius to random sample poses.
        self.fp16 = opt.fp16  # if preload, load into fp16.

        self.topia_poses_fname = opt.topia_poses_fname

        self.training = self.type in ['train', 'all', 'trainval']
        self.num_rays = self.opt.num_rays if self.training else -1

        self.error_map = None

        downscale = self.downscale #TODO
        # downscale = 0.25

        self.H = None
        self.W = None

        poses_fname = sorted([os.path.join(self.topia_poses_fname, f) for f in os.listdir(self.topia_poses_fname)])
        # H = args.render_res
        tmp_H = 128
        H = None
        ratio = 512 // tmp_H
        # ratio = 4

        self.poses = []
        self.images = []

        for idx, p in enumerate(poses_fname):
            c2w = np.loadtxt(p).reshape(4, 4)
            c2w[:3, 3] *= 2.2
            c2w = np.array([
                [1, 0, 0, 0],
                [0, 0, -1, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1]
            ]) @ c2w

            pose = np.array(c2w, dtype=np.float32)

            K = np.array([
                [560 / ratio, 0, tmp_H * 0.5],
                [0, 560 / ratio, tmp_H * 0.5],
                [0, 0, 1]
            ])
            img_f = os.path.join(self.root_path, '{}.png'.format(idx))
            # assert os.path.exists(img_f)
            # c2w_lst.append(c2w)
            # K_lst.append(K)
            # img_f_lst.append(img_f)

            image = cv2.imread(img_f, cv2.IMREAD_UNCHANGED)  # [H, W, 3] o [H, W, 4]
            if self.H is None or self.W is None:
                self.H = round(image.shape[0] // downscale)
                self.W = round(image.shape[1] // downscale)

            # add support for the alpha channel as a mask.
            if image.shape[-1] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)

            if image.shape[0] != self.H or image.shape[1] != self.W:
                image = cv2.resize(image, (self.W, self.H), interpolation=cv2.INTER_AREA)

            image = image.astype(np.float32) / 255  # [H, W, 3/4]

            self.poses.append(pose)
            self.images.append(image)

        self.poses = torch.from_numpy(np.stack(self.poses, axis=0))  # [N, 4, 4]
        if self.images is not None:
            self.images = torch.from_numpy(np.stack(self.images, axis=0))  # [N, H, W, C]

        # calculate mean radius of all camera poses
        self.radius = self.poses[:, :3, 3].norm(dim=-1).mean(0).item()

        cx =  (self.W / 2)
        cy =  (self.H / 2)
        self.intrinsics = np.array([K[0,0]  / downscale, K[1,1]  / downscale, cx, cy])
        self.device = device

        rays = get_rays(self.poses.to(self.device) , self.intrinsics, self.H, self.W, self.num_rays,
                        None, self.opt.patch_size)
        # print('******************************')
        # print(rays['rays_o'].shape)
        # print(rays['rays_o'][:,0])
        # print(rays['rays_d'].norm(dim=-1))

    def collate(self, index):

        B = len(index)  # a list of length 1


        poses = self.poses[index].to(self.device)  # [B, 4, 4]

        error_map = None

        rays = get_rays(poses, self.intrinsics, self.H, self.W, self.num_rays, error_map, self.opt.patch_size)

        results = {
            'H': self.H,
            'W': self.W,
            'rays_o': rays['rays_o'],
            'rays_d': rays['rays_d'],
        }

        if self.images is not None:
            images = self.images[index].to(self.device)  # [B, H, W, 3/4]
            if self.training:
                C = images.shape[-1]
                images = torch.gather(images.view(B, -1, C), 1, torch.stack(C * [rays['inds']], -1))  # [B, N, 3/4]
            results['images'] = images

        # need inds to update error_map
        if error_map is not None:
            results['index'] = index
            results['inds_coarse'] = rays['inds_coarse']

        return results




    def dataloader(self):
        size = len(self.poses)
        loader = DataLoader(list(range(size)), batch_size=1, collate_fn=self.collate, shuffle=self.training,
                            num_workers=0)
        loader._data = self  # an ugly fix... we need to access error_map & poses in trainer.
        loader.has_gt = self.images is not None
        return loader

try:
    from .dataset_llff.colmap import ColmapDataset
    from .dataset_llff.ray_utils import get_rays as get_rays_llff2
except:
    from nerf.dataset_llff.colmap import ColmapDataset
    from nerf.dataset_llff.ray_utils import get_rays as get_rays_llff2
class NeRFDatasetLLFF2:
    def __init__(self, opt, device, type='train', n_test=10):
        super().__init__()

        self.opt = opt
        # self.device = device
        self.device = torch.device('cpu')
        self.type = type  # train, val, test
        self.downscale = opt.downscale
        self.root_path = opt.path
        self.preload = opt.preload  # preload data into GPU
        self.scale = opt.scale  # camera radius scale to make sure camera are inside the bounding box.
        self.offset = opt.offset  # camera offset
        self.bound = opt.bound  # bounding box half length, also used as the radius to random sample poses.
        self.fp16 = opt.fp16  # if preload, load into fp16.

        self.llff_spherify = opt.llff_spherify
        self.llffhold = opt.llff_hold
        self.llff_render_mode = opt.llff_render_mode

        self.training = self.type in ['train', 'all', 'trainval']
        self.num_rays = self.opt.num_rays if self.training else -1


        self.aux_dataset = ColmapDataset(self.root_path,split= self.type,downsample=1/self.downscale)
        self.generate_rays()

        W, H = self.aux_dataset.img_wh
        K = self.aux_dataset.K
        self.W = W
        self.H = H
        fl_x, fl_y, cx, cy = K[0,0],K[1,1], K[0,2],K[1,2]
        self.intrinsics = np.array([fl_x, fl_y, cx, cy])
        self.error_map = None

        self.device = device


    def generate_rays(self):
        rays_o_lst = []
        rays_d_lst = []
        images_lst = []
        poses_lst = []
        for img_idx in range(len(self.aux_dataset)):
            item = self.aux_dataset[img_idx]
            pose = item['pose']

            # TODO: maybe use ours get rays
            # pose = torch.from_numpy(nerf_matrix_to_ngp(pose, scale=self.scale, offset=self.offset))[:3,:4]

            rays_o, rays_d = get_rays_llff2(self.aux_dataset.directions, pose)
            # rays_d = F.normalize(rays_d,dim=-1)
            rays_o_lst.append(rays_o)
            rays_d_lst.append(rays_d)
            poses_lst.append(pose)
            if 'rgb' in item:
                images_lst.append(item['rgb'])
        self.rays_o = torch.stack(rays_o_lst)
        self.rays_d = torch.stack(rays_d_lst)
        self.poses = torch.stack(poses_lst)
        camera_max_dist = self.rays_o.norm(dim=-1).max()
        print('camera_max_dist2 = {}'.format(camera_max_dist.item()))
        # self.rays_o /= camera_max_dist
        if len(images_lst) > 0:
            self.images = torch.stack(images_lst)

    def collate(self, index):

        B = len(index)  # a list of length 1
        assert B==1
        rays_o = self.rays_o[index]
        rays_o = rays_o.view(B,-1,rays_o.shape[-1])
        rays_d = self.rays_d[index]
        rays_d = rays_d.view(B, -1, rays_d.shape[-1])

        results = {
            'H': self.H,
            'W': self.W,
            'rays_o': rays_o,
            'rays_d': rays_d,
        }

        if self.images is not None:
            images = self.images[index]  # [ H * W, 3/4]
            C = images.shape[-1]
            if self.training:
                images = images.view(B, -1, C)
            else:
                images = images.view(B, self.H,self.W, C)
            results['images'] = images

        if self.num_rays > 0:
            inds = torch.randperm(rays_o.shape[-1])[:self.num_rays]
            for key in ['rays_o', 'rays_d', 'images']:
                if key in results:
                    results[key] = results[key][:,inds]

        for key in ['rays_o', 'rays_d', 'images']:
            if key in results:
                results[key] = results[key].to(self.device)
        return results

    def dataloader(self):
        size = len(self.aux_dataset)
        loader = DataLoader(list(range(size)), batch_size=1, collate_fn=self.collate, shuffle=self.training,
                            num_workers=0)
        loader._data = self  # an ugly fix... we need to access error_map & poses in trainer.
        loader.has_gt = self.images is not None
        return loader


if __name__ == "__main__":
    path = '/data/datasets/nerf/360_v2/bonsai'
    # images, poses, bds, render_poses, i_test = load_llff_data(path, 8,
    #                                                           recenter=True, bd_factor=.75,
    #                                                           spherify=False)
    # print(images.shape)
    # print(poses.shape)
    # print(render_poses.shape)
    # print(i_test)

    img_idx = 0
    dataset = ColmapDataset(path, split='train', downsample=1 / 8)
    item = dataset[img_idx]
    print(item.keys())
    print(len(dataset))

    rays_o, rays_d = get_rays_llff2(dataset.directions.cuda(), item['pose'].cuda())
    print(rays_o.shape)
    print(rays_d.shape)
    img = item['rgb']
    print(img.shape)
    print('h')

