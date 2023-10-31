import json
import os

import cv2
import imageio
import matplotlib.pyplot as plt
import torch

# import matplotlib.pyplot as plt
from icecream import ic
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from tqdm import tqdm

from modules.tonemap import SRGBTonemap

from .ray_utils import *


class BlenderDataset(Dataset):
    def __init__(
        self,
        datadir,
        stack_norms=False,
        split="train",
        downsample=1.0,
        is_stack=False,
        N_vis=-1,
        white_bg=True,
        is_testing=False,
    ):
        self.downsample = downsample
        self.N_vis = N_vis
        self.root_dir = datadir
        self.split = split
        self.is_stack = is_stack
        self.stack_norms = stack_norms
        self.white_bg = white_bg
        self.is_testing = is_testing or split == "test"
        self.define_transforms()
        if self.stack_norms:
            print("Stacking normals")

        self.scene_bbox = torch.tensor([[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]])
        self.blender2opencv = np.array(
            [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
        )
        self.center = torch.mean(self.scene_bbox, axis=0).float().view(1, 1, 3)
        self.radius = (self.scene_bbox[1] - self.center).float().view(1, 1, 3)
        self.read_meta()
        self.define_proj_mat()

    def read_depth(self, filename):
        depth = np.array(read_pfm(filename)[0], dtype=np.float32)  # (800, 800)
        return depth

    def read_meta(self):
        with open(
            os.path.join(self.root_dir, f"transforms_{self.split}.json"), "r"
        ) as f:
            self.meta = json.load(f)

        if "ext" in self.meta:
            ext = self.meta["ext"]
        elif self.meta['frames'][0]['file_path'].lower().endswith('jpg') or \
                self.meta['frames'][0]['file_path'].lower().endswith('jpeg') or \
                self.meta['frames'][0]['file_path'].lower().endswith('png'):
            ext = ""
        else:
            ext = ".png"
        if "normal_ext" in self.meta:
            normal_ext = self.meta["normal_ext"]
        else:
            normal_ext = ext

        if "near_far" in self.meta:
            self.near_far = self.meta["near_far"]
        else:
            self.near_far = [2.0, 6.0]
        if "white_bg" in self.meta:
            self.white_bg = self.meta["white_bg"]
        else:
            self.white_bg = self.white_bg
        if "w" not in self.meta:
            self.meta["w"] = 800
        if "h" not in self.meta:
            self.meta["h"] = 800

        w, h = int(self.meta["w"] / self.downsample), int(self.meta["h"] / self.downsample)
        self.img_wh = [w, h]
        print(f"Original Image size: {self.meta['w']} x {self.meta['h']}")
        print(f"Image size: {w} x {h}")
        if "aabb_scale" in self.meta:
            aabb_scale = self.meta["aabb_scale"]
            self.scene_bbox *= aabb_scale
            self.radius *= aabb_scale

        if "camera_angle_x" in self.meta:
            self.fx = (
                0.5 * w / np.tan(0.5 * self.meta["camera_angle_x"])
            )  # original focal length
            self.fy = self.fx
        else:
            self.fx = self.meta["fl_x"]
            self.fy = self.meta["fl_y"]
        ic(self.fx, self.fy)

        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = get_ray_directions(h, w, [self.fx, self.fy])  # (h, w, 3)
        cam_right = torch.tensor([1.0, 0.0, 0.0]).reshape(1, -1)
        # self.rays_up = torch.linalg.cross(self.directions.reshape(-1, 3), cam_right, dim=-1).reshape(h, w, 3)
        self.rays_up = torch.tensor([0.0, 1.0, 0.0]).reshape(1, 1, 3).expand(h, w, 3)
        # self.rays_up = -torch.stack([
        #     torch.zeros_like(self.directions[..., 0]),
        #     self.directions[..., 2],
        #     -self.directions[..., 1],
        # ], dim=2)
        # self.rays_up /= torch.norm(self.rays_up, dim=2, keepdim=True)
        self.directions = self.directions / torch.norm(
            self.directions, dim=-1, keepdim=True
        )
        self.intrinsics = torch.tensor(
            [[self.fx, 0, w / 2], [0, self.fy, h / 2], [0, 0, 1]]
        ).float()
        ic(self.intrinsics)

        self.image_paths = []
        self.poses = []
        self.all_rays = []
        self.all_norms = []
        self.all_rgbs = []
        self.all_masks = []
        self.all_depth = []
        self.normal_paths = []
        self.tint_paths = []
        self.acc_paths = []
        self.acc_maps = []

        tonemap = SRGBTonemap()
        img_eval_interval = (
            1 if self.N_vis < 0 else len(self.meta["frames"]) // self.N_vis
        )
        idxs = list(range(0, len(self.meta["frames"]), img_eval_interval))
        for i in tqdm(
            idxs, desc=f"Loading data {self.split} ({len(idxs)})"
        ):  # img_list:#
            frame = self.meta["frames"][i]
            pose = np.array(frame["transform_matrix"]) @ self.blender2opencv
            c2w = torch.FloatTensor(pose)
            self.poses += [c2w]

            image_path = os.path.join(self.root_dir, f"{frame['file_path']}{ext}")
            normal_path = os.path.join(self.root_dir, f"{frame['file_path']}_normal{normal_ext}")
            tint_path = os.path.join(self.root_dir, "test", "tint", f"r_{i}{ext}")
            acc_path = os.path.join(self.root_dir, "test", f"r_{i}_alpha{ext}")
            self.image_paths += [image_path]
            self.normal_paths += [normal_path]
            self.tint_paths += [tint_path]
            self.acc_paths += [acc_path]
            img = Image.open(image_path)
            # img = imageio.imread(image_path)

            if self.downsample != 1.0:
                img.resize(self.img_wh, Image.LANCZOS)
            img = self.transform(img)  # (4, h, w)
            # plt.imshow(img.permute(1, 2, 0))
            # plt.show()
            if img.shape[0] == 4:
                self.acc_maps += [img[-1]]
            elif os.path.exists(self.acc_paths[-1]):
                self.acc_maps += [np.array(Image.open(self.acc_paths[-1]))[..., None]]

            img = img.view(img.shape[0], -1).permute(1, 0)  # (h*w, 4) RGBA

            rays_o, rays_d = get_rays(self.directions, c2w)  # both (h*w, 3)
            _, rays_up = get_rays(self.rays_up, c2w)  # both (h*w, 3)
            # rays = torch.cat([rays_o, rays_d, rays_up], 1)
            rays = torch.cat([rays_o, rays_d], 1)

            c = img.shape[1]
            if c == 4 and self.is_testing:
                img[:, :3] = img[:, :3] * img[:, -1:] + (
                    1 - img[:, -1:]
                )  # blend A to RGB
                # this is automatic during training
            # this is really stupid
            if "exr" in str(image_path):
                img[:, :3] = tonemap(img[:, :3])
                self.hdr = True
            else:
                self.hdr = False
            # img = img.clip(0, 1)
            self.all_rgbs += [img]

            if self.stack_norms:
                self.all_norms += [self.get_normal(i).reshape(-1, 3)]

            self.all_rays += [rays]  # (h*w, 6)

        self.poses = torch.stack(self.poses)
        if self.stack_norms:
            self.all_norms = torch.cat(self.all_norms)
        if not self.is_stack:
            self.all_rays = torch.cat(
                self.all_rays, 0
            )  # (len(self.meta['frames])*h*w, 3)
            self.all_rgbs = torch.cat(
                self.all_rgbs, 0
            )  # (len(self.meta['frames])*h*w, 3)

        #             self.all_depth = torch.cat(self.all_depth, 0)  # (len(self.meta['frames])*h*w, 3)
        else:
            self.all_rays = torch.stack(
                self.all_rays, 0
            )  # (len(self.meta['frames]),h*w, 3)

            self.all_rgbs = torch.stack(self.all_rgbs, 0).reshape(
                -1, *self.img_wh[::-1], c
            )[
                ..., :3
            ]  # (len(self.meta['frames]),h,w,3)
            # self.all_masks = torch.stack(self.all_masks, 0).reshape(-1,*self.img_wh[::-1])  # (len(self.meta['frames]),h,w,3)

    def define_transforms(self):
        self.transform = T.ToTensor()

    def define_proj_mat(self):
        self.proj_mat = self.intrinsics.unsqueeze(0) @ torch.inverse(self.poses)[:, :3]

    def world2ndc(self, points, lindisp=None):
        device = points.device
        return (points - self.center.to(device)) / self.radius.to(device)

    def __len__(self):
        return len(self.all_rgbs)

    def get_tint(self, idx):
        tint = imageio.imread(self.tint_paths[idx])
        return torch.as_tensor(tint)[..., :3]

    def get_normal(self, idx):
        norms = imageio.imread(self.normal_paths[idx])
        norms = torch.as_tensor(norms)[..., :3].float()
        if norms.max() > 2:
            norms = (norms - 128) / 127
            norms = norms / torch.linalg.norm(norms, dim=-1, keepdim=True).clip(
                min=torch.finfo(torch.float32).eps
            )
        else:
            norms = (norms - 0.5) * 2
        return norms

    def __getitem__(self, idx):
        if self.split == "train":  # use data in the buffers
            sample = {"rays": self.all_rays[idx], "rgbs": self.all_rgbs[idx]}

        else:  # create data for each image separately
            img = self.all_rgbs[idx]
            rays = self.all_rays[idx]
            mask = self.all_masks[idx]  # for quantity evaluation

            sample = {"rays": rays, "rgbs": img, "mask": mask}
        return sample
