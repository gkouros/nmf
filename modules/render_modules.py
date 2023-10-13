import math
from math import pi
from typing import List

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from icecream import ic

from mutils import inv_sigmoid, normalize

from . import safemath, util
from .grid_sample_Cinf import gkern
from .sh import eval_sh_bases


def get_dim(encoder, extra=0):
    return 0 if encoder is None else encoder.dim() + extra


def str2fn(name):
    if name == "sigmoid":
        return torch.nn.Sigmoid()
    if name == "exp":
        return torch.exp
    elif name == "softplus":
        return torch.nn.Softplus()
    elif name == "identity":
        return torch.nn.Identity()
    elif name == "clamp":
        return Clamp(0, 1)
    else:
        raise Exception(f"Unknown function {name}")


def positional_encoding(positions, freqs):
    freq_bands = (2 ** torch.arange(freqs).float()).to(positions.device)  # (F,)
    pts = (positions[..., None] * freq_bands).reshape(
        positions.shape[:-1] + (freqs * positions.shape[-1],)
    )  # (..., DF)
    pts = torch.cat([torch.sin(pts), torch.cos(pts)], dim=-1)
    return pts


def spherical_encoding(refdirs, roughness, pe, ind_order=[0, 1, 2]):
    i, j, k = ind_order
    norm2d = torch.sqrt(refdirs[..., i] ** 2 + refdirs[..., j] ** 2)
    refangs = torch.stack(
        [
            safemath.atan2(refdirs[..., j], refdirs[..., i]) * norm2d,
            safemath.atan2(refdirs[..., k], norm2d),
        ],
        dim=-1,
    )
    return [
        safemath.integrated_pos_enc((refangs[..., 0:1], roughness), 0, pe),
        safemath.integrated_pos_enc((refangs[..., 1:2], roughness), 0, pe),
    ]


def normal_dist(x, sigma: float):
    SQ2PI = 2.50662827463
    return torch.exp(-((x / sigma) ** 2) / 2) / SQ2PI / sigma


def SHRender(xyz_sampled, viewdirs, features):
    sh_mult = eval_sh_bases(2, viewdirs)[:, None]
    rgb_sh = features.view(-1, 3, sh_mult.shape[-1])
    rgb = torch.relu(torch.sum(sh_mult * rgb_sh, dim=-1) + 0.5)
    return rgb


def RGBRender(xyz_sampled, viewdirs, features):
    rgb = features
    return rgb


class Clamp(torch.nn.Module):
    def __init__(self, min=None, max=None):
        super().__init__()
        self.min = min
        self.max = max

    def forward(self, x):
        return x.clamp(self.min, self.max)


class IPE(torch.nn.Module):
    def __init__(self, max_degree=8, in_dim=3) -> None:
        super().__init__()
        self.max_degree = max_degree
        self.in_dim = in_dim

    def dim(self):
        return 2 * self.in_dim * self.max_degree

    def forward(self, viewdirs, roughness, **kwargs):
        size = roughness.reshape(-1, 1).expand(viewdirs.shape)
        return safemath.integrated_pos_enc((viewdirs, size), 0, self.max_degree)


class PE(torch.nn.Module):
    def __init__(self, max_degree=8, in_dim=3) -> None:
        super().__init__()
        self.max_degree = max_degree
        self.in_dim = in_dim

    def dim(self):
        return 2 * self.in_dim * self.max_degree

    def forward(self, x, roughness, **kwargs):
        return positional_encoding(x, self.max_degree)


class VisibilityMLP(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        view_encoder=None,
        feape=2,
        featureC=128,
        num_layers=4,
        lr=1e-3,
    ):
        super().__init__()

        self.lr = lr
        self.in_mlpC = 3
        if feape > -1:
            self.in_mlpC += 2 * feape * in_channels + in_channels
        self.view_encoder = view_encoder
        self.in_mlpC += get_dim(self.view_encoder)
        self.feape = feape

        self.mlp = torch.nn.Sequential(
            # torch.nn.BatchNorm1d(self.in_mlpC),
            torch.nn.Linear(self.in_mlpC, featureC),
            # torch.nn.BatchNorm1d(featureC),
            *sum(
                [
                    [
                        torch.nn.ReLU(inplace=True),
                        torch.nn.Linear(featureC, featureC),
                        # torch.nn.BatchNorm1d(featureC),
                    ]
                    for _ in range(num_layers - 2)
                ],
                [],
            ),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(featureC, 2),
        )
        torch.nn.init.constant_(self.mlp[-1].bias, -2)
        self.mlp.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=np.sqrt(2))

    def mask(self, norm_ray_origins, viewdirs, world_bounces, features):
        eterm, sigvis = self.visibility_module(norm_ray_origins, viewdirs, features)
        p = max(min(1 - world_bounces / sigvis.numel(), 1.0), 0.0)
        t = torch.quantile(sigvis.flatten(), p).clip(min=0.9)
        vis_mask = sigvis > t
        return vis_mask

    def update(
        self, norm_ray_origins, viewdirs, app_features, termination, bgvisibility
    ):
        # bgvisibility is 1 if it reaches the BG and 0 if not
        eterm, sigvis = self.visibility_module(norm_ray_origins, viewdirs, app_features)
        # loss = ((termination - eterm)**2 + (sigvis-visibility.float())**2).sum()
        loss = ((sigvis - (~visibility).float()) ** 2).mean()
        return loss

    def forward(self, pts, viewdirs, features, **kwargs):
        B = pts.shape[0]
        pts = pts[..., :3]

        indata = [viewdirs]
        if self.feape > -1:
            indata.append(features)
        if self.feape > 0:
            indata += [positional_encoding(features, self.feape)]
        if self.view_encoder is not None:
            ise_enc = self.view_encoder(
                viewdirs, torch.tensor(1e-2, device=pts.device).expand(B)
            ).reshape(B, -1)
            indata += [ise_enc]

        mlp_in = torch.cat(indata, dim=-1)
        out = self.mlp(mlp_in)
        sigvis = torch.sigmoid(out[..., 0])
        eterm = torch.exp(out[..., 1])

        return eterm, sigvis


class MLPRender_Fea(torch.nn.Module):
    def __init__(self, in_channels, viewpe=6, feape=6, featureC=128, lr=1e-3):
        super(MLPRender_Fea, self).__init__()

        self.lr = lr
        self.in_mlpC = 2 * viewpe * 3 + 2 * feape * in_channels + 3 + in_channels
        self.viewpe = viewpe
        self.feape = feape
        layer1 = torch.nn.Linear(self.in_mlpC, featureC)
        layer2 = torch.nn.Linear(featureC, featureC)
        layer3 = torch.nn.Linear(featureC, 3)

        self.mlp = torch.nn.Sequential(
            layer1,
            torch.nn.ReLU(inplace=True),
            layer2,
            torch.nn.ReLU(inplace=True),
            layer3,
        )
        torch.nn.init.constant_(self.mlp[-1].bias, 0)

    def calibrate(self, *args):
        return

    def forward(self, pts, viewdirs, features, *args, **kwargs):
        indata = [features, viewdirs]
        if self.feape > 0:
            indata += [positional_encoding(features, self.feape)]
        if self.viewpe > 0:
            indata += [positional_encoding(viewdirs, self.viewpe)]
        mlp_in = torch.cat(indata, dim=-1)
        rgb = self.mlp(mlp_in)
        rgb = torch.sigmoid(rgb)

        return rgb


class MLPRender_FP(torch.nn.Module):
    in_channels: int
    feape: int
    featureC: int
    num_layers: int

    def __init__(
        self,
        in_channels,
        view_encoder=None,
        ref_encoder=None,
        feape=6,
        activation="softplus",
        lr=1e-3,
        offset=0,
        **kwargs,
    ):
        super().__init__()

        self.lr = lr
        self.ref_encoder = ref_encoder
        self.in_mlpC = 3 + 1
        if feape > -1:
            self.in_mlpC += 2 * feape * in_channels + in_channels
        self.view_encoder = view_encoder
        if view_encoder is not None:
            self.in_mlpC += self.view_encoder.dim()
        if ref_encoder is not None:
            self.in_mlpC += self.ref_encoder.dim()
        self.feape = feape
        self.offset = offset

        self.mlp = util.create_mlp(self.in_mlpC, 3, **kwargs)
        self.activation = str2fn(activation)

    def init_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=np.sqrt(2))

    def forward(
        self, pts, viewdirs, features, refdirs, roughness, viewdotnorm, **kwargs
    ):
        B = pts.shape[0]
        pts = pts[..., :3]

        indata = [refdirs, viewdotnorm]
        if self.feape > -1:
            indata.append(features)
        if self.feape > 0:
            indata += [positional_encoding(features, self.feape)]
        if self.view_encoder is not None:
            ise_enc = self.view_encoder(viewdirs).reshape(B, -1)
            indata += [ise_enc]
        if self.ref_encoder is not None:
            ise_enc = self.ref_encoder(refdirs, roughness).reshape(B, -1)
            indata += [ise_enc]

        mlp_in = torch.cat(indata, dim=-1)
        rgb = self.mlp(mlp_in)
        rgb = self.activation(rgb + self.offset)

        return rgb


class PassthroughDiffuse(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.allocation = 8
        self.lr = 0

    def forward(self, pts, viewdirs, features, **kwargs):
        B = pts.shape[0]
        mlp_out = features
        # max 0.5 roughness
        i = 0
        diffuse = torch.sigmoid(mlp_out[..., : i + 3] - 3)
        i += 3
        roughness = torch.sigmoid(mlp_out[..., i : i + 1] + 2).clip(min=1e-2) / 2
        i += 1
        ambient = torch.sigmoid(mlp_out[..., i : i + 1] - 2)
        i += 1
        tint = torch.sigmoid(mlp_out[..., i : i + 3])
        i += 3
        return (
            diffuse,
            tint,
            dict(
                ambient=ambient,
                diffuse=diffuse,
                roughness=roughness,
            ),
        )


class MLPRender(torch.nn.Module):
    in_channels: int
    viewpe: int
    feape: int
    refpe: int
    featureC: int
    num_layers: int

    def __init__(
        self,
        in_channels,
        pospe=12,
        view_encoder=None,
        feape=6,
        featureC=128,
        num_layers=0,
        allocation=0,
        unlit_tint=False,
        lr=1e-4,
    ):
        super().__init__()

        in_channels = in_channels if allocation <= 0 else allocation
        self.in_mlpC = (
            +2 * max(feape, 0) * in_channels + in_channels if feape >= 0 else 0
        )
        if pospe >= 0:
            self.in_mlpC += 2 * pospe * 3 + 3
        self.unlit_tint = unlit_tint
        self.lr = lr
        self.allocation = allocation

        self.view_encoder = view_encoder
        if view_encoder is not None:
            self.in_mlpC += self.view_encoder.dim() + 3
        self.feape = feape
        self.pospe = pospe
        if num_layers > 0:
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(self.in_mlpC, featureC),
                # torch.nn.ReLU(inplace=True),
                # torch.nn.Linear(featureC, featureC),
                # torch.nn.BatchNorm1d(featureC),
                *sum(
                    [
                        [
                            torch.nn.ReLU(inplace=True),
                            torch.nn.Linear(featureC, featureC),
                            # torch.nn.BatchNorm1d(featureC)
                        ]
                        for _ in range(num_layers - 2)
                    ],
                    [],
                ),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(featureC, 3),
            )
            torch.nn.init.constant_(self.mlp[-1].bias, 0)
            # self.mlp.apply(self.init_weights)
        else:
            self.mlp = torch.nn.Identity()
        # to the neural network, roughness is unitless

    def forward(self, pts, viewdirs, features, **kwargs):
        if self.allocation > 0:
            features = features[..., : self.allocation]
        B = pts.shape[0]
        size = pts[..., 3:4].expand(pts[..., :3].shape)
        pts = pts[..., :3]
        indata = []
        if self.pospe >= 0:
            indata.append(pts)
        if self.pospe > 0:
            indata += [safemath.integrated_pos_enc((pts, size), 0, self.pospe)]

        if self.feape >= 0:
            indata.append(features)
        if self.feape > 0:
            indata += [positional_encoding(features, self.feape)]
        if self.view_encoder is not None:
            indata += [
                self.view_encoder(
                    viewdirs, torch.tensor(20.0, device=pts.device).expand(B)
                ).reshape(B, -1),
                viewdirs,
            ]
        mlp_in = torch.cat(indata, dim=-1)
        mlp_out = self.mlp(mlp_in)

        ambient = torch.sigmoid(mlp_out[..., 2] - 2)
        roughness = torch.sigmoid(mlp_out[..., 2] - 1).clip(min=1e-2)  # /2
        # ic(mlp_out[..., 0:6])
        tint = torch.sigmoid((mlp_out[..., :3]).clip(min=-10, max=10))
        # diffuse = rgb[..., :3]
        # tint = F.softplus(mlp_out[..., 3:6])
        diffuse = torch.sigmoid((mlp_out[..., :3] - math.log(3)))

        # ic(f0)
        return (
            diffuse,
            tint,
            dict(
                # refraction_index = refraction_index,
                # ratio_diffuse = ratio_diffuse,
                # reflectivity = reflectivity,
                ambient=ambient,
                # albedo=albedo,
                diffuse=diffuse,
                roughness=roughness,
                # f0 = f0,
                tint=tint,
            ),
        )


class RandHydraMLPDiffuse(torch.nn.Module):
    in_channels: int
    viewpe: int
    feape: int
    refpe: int
    featureC: int
    num_layers: int

    def __init__(
        self,
        in_channels,
        pospe=12,
        view_encoder=None,
        roughness_view_encoder=None,
        roughness_cfg=None,
        feape=6,
        allocation=0,
        unlit_tint=False,
        lr=1e-4,
        tint_bias=-1,
        diffuse_bias=-2,
        diffuse_mul=1,
        roughness_bias=1,
        start_roughness=0.35,
        f0_bias=0,
        **kwargs,
    ):
        super().__init__()

        in_channels = in_channels if allocation <= 0 else allocation
        self.in_mlpC = (
            +2 * max(feape, 0) * in_channels + in_channels if feape >= 0 else 0
        )
        if pospe >= 0:
            self.in_mlpC += 2 * pospe * 3 + 3
        self.unlit_tint = unlit_tint
        self.tint_bias = tint_bias
        self.diffuse_bias = diffuse_bias
        self.roughness_bias = roughness_bias
        self.lr = lr
        self.allocation = allocation
        self.diffuse_mul = diffuse_mul
        self.start_roughness = start_roughness
        self.f0_bias = f0_bias

        self.view_encoder = view_encoder
        self.roughness_view_encoder = roughness_view_encoder
        self.in_mlpC += get_dim(self.view_encoder, 3)
        self.feape = feape
        self.pospe = pospe
        self.diffuse_mlp = util.create_mlp(self.in_mlpC, 3, **kwargs)
        self.tint_mlp = util.create_mlp(self.in_mlpC, 3, **kwargs)
        self.f0_mlp = util.create_mlp(self.in_mlpC, 3, **kwargs)
        roughness_cfg = roughness_cfg if roughness_cfg is not None else kwargs
        self.roughness_mlp = util.create_mlp(
            self.in_mlpC + get_dim(self.roughness_view_encoder, 3), 2, **roughness_cfg
        )

    def calibrate(self, mean_brightness, conserve_energy, *args, **kwargs):
        diffuse, tint, extra = self(*args, **kwargs)
        diffuse_v = inv_sigmoid(diffuse).mean().detach().item()
        # tint_v = (tint / (1-tint)).log()
        v = (0.25 if not conserve_energy else 0.5) / float(mean_brightness)
        self.diffuse_bias += inv_sigmoid(v) - diffuse_v
        ic(self.diffuse_bias, mean_brightness, v)

        roughness = (extra["r1"] + extra["r2"]) / 2 / 2
        roughness_v = inv_sigmoid(roughness).mean().detach().item()
        self.roughness_bias += inv_sigmoid(self.start_roughness) - roughness_v

        # self.tint_bias += 1.1 - diffuse_v

    def forward(self, pts, viewdirs, features, std=0, **kwargs):
        if self.allocation > 0:
            features = features[..., : self.allocation]
        device = features.device
        B = pts.shape[0]
        size = pts[..., 3:4].expand(pts[..., :3].shape)
        pts = pts[..., :3]
        indata = []
        if self.pospe >= 0:
            indata.append(pts)
        if self.pospe > 0:
            indata += [safemath.integrated_pos_enc((pts, size), 0, self.pospe)]

        if self.feape >= 0:
            indata.append(features)
        if self.feape > 0:
            indata += [positional_encoding(features, self.feape)]
        if self.view_encoder is not None:
            indata += [
                self.view_encoder(
                    viewdirs, torch.tensor(1e-3, device=device).expand(B)
                ).reshape(B, -1),
                viewdirs,
            ]
        mlp_in = torch.cat(indata, dim=-1)

        if self.roughness_view_encoder is not None:
            indata += [
                self.roughness_view_encoder(
                    viewdirs, torch.tensor(1e-3, device=device).expand(B)
                ).reshape(B, -1),
                viewdirs,
            ]
        rough_mlp_in = torch.cat(indata, dim=-1)
        diffuse = torch.sigmoid(
            self.diffuse_mul * self.diffuse_mlp(mlp_in) + self.diffuse_bias
        )
        diffuse = (diffuse + torch.randn_like(diffuse) * std).clip(min=0, max=1)
        r = torch.sigmoid(self.roughness_mlp(rough_mlp_in) + self.roughness_bias) / 2
        r = (r + torch.randn_like(r) * std / 2).clip(min=1e-2, max=1)
        tint = torch.sigmoid(self.tint_mlp(mlp_in) + self.tint_bias)
        f0 = torch.sigmoid(self.f0_mlp(mlp_in) + self.f0_bias)
        # tint = (tint + torch.randn_like(tint) * std).clip(min=0, max=1)

        # ic(f0)
        return (
            diffuse,
            tint,
            dict(
                diffuse=diffuse,
                r1=r[:, 0:1],
                r2=r[:, 1:2],
                f0=f0,
                tint=tint,
            ),
        )


class HydraMLPDiffuse(torch.nn.Module):
    in_channels: int
    viewpe: int
    feape: int
    refpe: int
    featureC: int
    num_layers: int

    def __init__(
        self,
        in_channels,
        pospe=12,
        view_encoder=None,
        roughness_view_encoder=None,
        roughness_cfg=None,
        feape=6,
        allocation=0,
        unlit_tint=False,
        lr=1e-4,
        tint_bias=-1,
        diffuse_bias=-2,
        diffuse_mul=1,
        roughness_bias=1,
        start_roughness=0.35,
        **kwargs,
    ):
        super().__init__()

        in_channels = in_channels if allocation <= 0 else allocation
        self.in_mlpC = (
            +2 * max(feape, 0) * in_channels + in_channels if feape >= 0 else 0
        )
        if pospe >= 0:
            self.in_mlpC += 2 * pospe * 3 + 3
        self.unlit_tint = unlit_tint
        self.tint_bias = tint_bias
        self.diffuse_bias = diffuse_bias
        self.roughness_bias = roughness_bias
        self.lr = lr
        self.allocation = allocation
        self.diffuse_mul = diffuse_mul
        self.start_roughness = start_roughness

        self.view_encoder = view_encoder
        self.roughness_view_encoder = roughness_view_encoder
        self.in_mlpC += get_dim(self.view_encoder, 3)
        self.feape = feape
        self.pospe = pospe
        self.diffuse_mlp = util.create_mlp(self.in_mlpC, 3, **kwargs)
        self.tint_mlp = util.create_mlp(self.in_mlpC, 3, **kwargs)
        roughness_cfg = roughness_cfg if roughness_cfg is not None else kwargs
        self.roughness_mlp = util.create_mlp(
            self.in_mlpC + get_dim(self.roughness_view_encoder, 3), 2, **roughness_cfg
        )

    def calibrate(self, mean_brightness, conserve_energy, *args, **kwargs):
        diffuse, tint, extra = self(*args, **kwargs)
        diffuse_v = inv_sigmoid(diffuse).mean().detach().item()
        # tint_v = (tint / (1-tint)).log()
        v = (0.25 if not conserve_energy else 0.5) / float(mean_brightness)
        self.diffuse_bias += inv_sigmoid(v) - diffuse_v
        ic(self.diffuse_bias, mean_brightness, v)

        roughness = (extra["r1"] + extra["r2"]) / 2 / 2
        roughness_v = inv_sigmoid(roughness).mean().detach().item()
        self.roughness_bias += inv_sigmoid(self.start_roughness) - roughness_v

        # self.tint_bias += 1.1 - diffuse_v

    def forward(self, pts, viewdirs, features, **kwargs):
        if self.allocation > 0:
            features = features[..., : self.allocation]
        device = features.device
        B = pts.shape[0]
        size = pts[..., 3:4].expand(pts[..., :3].shape)
        pts = pts[..., :3]
        indata = []
        if self.pospe >= 0:
            indata.append(pts)
        if self.pospe > 0:
            indata += [safemath.integrated_pos_enc((pts, size), 0, self.pospe)]

        if self.feape >= 0:
            indata.append(features)
        if self.feape > 0:
            indata += [positional_encoding(features, self.feape)]
        if self.view_encoder is not None:
            indata += [
                self.view_encoder(
                    viewdirs, torch.tensor(1e-3, device=device).expand(B)
                ).reshape(B, -1),
                viewdirs,
            ]
        mlp_in = torch.cat(indata, dim=-1)

        if self.roughness_view_encoder is not None:
            indata += [
                self.roughness_view_encoder(
                    viewdirs, torch.tensor(1e-3, device=device).expand(B)
                ).reshape(B, -1),
                viewdirs,
            ]
        rough_mlp_in = torch.cat(indata, dim=-1)
        diffuse = torch.sigmoid(
            self.diffuse_mul * self.diffuse_mlp(mlp_in) + self.diffuse_bias
        )
        r = torch.sigmoid(self.roughness_mlp(rough_mlp_in) + self.roughness_bias) / 2
        tint = torch.sigmoid(self.tint_mlp(mlp_in) + self.tint_bias)

        return (
            diffuse,
            tint,
            dict(
                diffuse=diffuse,
                r1=r[:, 0:1],
                r2=r[:, 1:2],
                tint=tint,
            ),
        )


class MLPDiffuse(torch.nn.Module):
    in_channels: int
    viewpe: int
    feape: int
    refpe: int
    featureC: int
    num_layers: int

    def __init__(
        self,
        in_channels,
        pospe=12,
        view_encoder=None,
        feape=6,
        allocation=0,
        unlit_tint=False,
        lr=1e-4,
        tint_bias=-1,
        diffuse_bias=-2,
        diffuse_mul=1,
        roughness_bias=1,
        **kwargs,
    ):
        super().__init__()

        in_channels = in_channels if allocation <= 0 else allocation
        self.in_mlpC = (
            +2 * max(feape, 0) * in_channels + in_channels if feape >= 0 else 0
        )
        if pospe >= 0:
            self.in_mlpC += 2 * pospe * 3 + 3
        self.unlit_tint = unlit_tint
        self.tint_bias = tint_bias
        self.diffuse_bias = diffuse_bias
        self.roughness_bias = roughness_bias
        self.diffuse_mul = diffuse_mul
        self.lr = lr
        self.allocation = allocation

        self.view_encoder = view_encoder
        if view_encoder is not None:
            self.in_mlpC += self.view_encoder.dim() + 3
        self.feape = feape
        self.pospe = pospe
        self.mlp = util.create_mlp(self.in_mlpC, 10, **kwargs)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            # if m.weight.shape[0] <= 60:
            #     torch.nn.init.constant_(m.weight, np.sqrt(2) / m.weight.shape[1])
            # else:
            #     torch.nn.init.xavier_uniform_(m.weight, gain=np.sqrt(2))
            torch.nn.init.xavier_uniform_(
                m.weight, gain=torch.nn.init.calculate_gain("relu")
            )

    def forward(self, pts, viewdirs, features, **kwargs):
        if self.allocation > 0:
            features = features[..., : self.allocation]
        B = pts.shape[0]
        size = pts[..., 3:4].expand(pts[..., :3].shape)
        pts = pts[..., :3]
        indata = []
        if self.pospe >= 0:
            indata.append(pts)
        if self.pospe > 0:
            indata += [safemath.integrated_pos_enc((pts, size), 0, self.pospe)]

        if self.feape >= 0:
            indata.append(features)
        if self.feape > 0:
            indata += [positional_encoding(features, self.feape)]
        if self.view_encoder is not None:
            indata += [
                self.view_encoder(
                    viewdirs, torch.tensor(1e-3, device=pts.device).expand(B)
                ).reshape(B, -1),
                viewdirs,
            ]
        mlp_in = torch.cat(indata, dim=-1)
        mlp_out = self.mlp(mlp_in)

        ambient = torch.sigmoid(mlp_out[..., 6:7] - 2)
        # r1 = F.softplus(mlp_out[..., 7:8]+self.roughness_bias) + 1e-3
        # r2 = F.softplus(mlp_out[..., 8:9]+self.roughness_bias) + 1e-3
        r1 = torch.sigmoid(mlp_out[..., 7:8] + self.roughness_bias) * (1 - 1e-3) + 1e-3
        r2 = torch.sigmoid(mlp_out[..., 8:9] + self.roughness_bias) * (1 - 1e-3) + 1e-3
        tint = torch.sigmoid((mlp_out[..., 3:6] + self.tint_bias))
        f0 = torch.sigmoid((mlp_out[..., 9:10] + 3)) * (1 - 0.001) + 0.001
        diffuse = torch.sigmoid(
            (self.diffuse_mul * mlp_out[..., :3] + self.diffuse_bias)
        )

        # ic(f0)
        return (
            diffuse,
            tint,
            dict(
                ambient=ambient,
                diffuse=diffuse,
                r1=r1,
                r2=r2,
                f0=f0,
                tint=tint,
            ),
        )


def init_weights_multiply(m):
    if isinstance(m, torch.nn.Linear):
        # torch.nn.init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        m.weight.data *= 10


def init_weights_kaiming_fan_out(m):
    if isinstance(m, torch.nn.Linear):
        # torch.nn.init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        torch.nn.init.kaiming_uniform_(m.weight, mode="fan_out")
        m.weight.data *= 2
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)


def init_weights_final_norm(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.uniform_(m.weight, -1e-5, 1e-5)


class MLPNormal(torch.nn.Module):
    in_channels: int
    feape: int
    featureC: int
    num_layers: int

    def __init__(
        self,
        in_channels,
        pospe=6,
        feape=6,
        allocation=0,
        lr=1e-4,
        size_multi=2.5e-3,
        offset_geometry=False,
        **kwargs,
    ):
        super().__init__()

        in_channels = in_channels if allocation <= 0 else allocation
        self.in_mlpC = 0
        if pospe >= 0:
            self.in_mlpC += 2 * pospe * 3 + 3
        if feape >= 0:
            self.in_mlpC += 2 * max(feape, 0) * in_channels + in_channels
        self.pospe = pospe
        self.feape = feape
        self.lr = lr
        self.allocation = allocation
        self.size_multi = size_multi
        self.mlp = util.create_mlp(self.in_mlpC, 3, bias=False, **kwargs)
        self.mlp[-1].apply(init_weights_final_norm)
        # self.mlp.apply(init_weights_multiply)
        self.offset_geometry = offset_geometry

    def forward(self, pts, features, geo_norms, **kwargs):
        size = pts[..., 3:4].expand(pts[..., :3].shape)
        pts = pts[..., :3]
        indata = []
        if self.pospe >= 0:
            indata.append(pts)
        if self.allocation > 0:
            features = features[..., : self.allocation]
        if self.feape >= 0:
            indata.append(features)

        if self.pospe > 0:
            indata += [
                safemath.integrated_pos_enc(
                    (pts, self.size_multi * size), 0, self.pospe
                )
            ]
            # ic(safemath.integrated_pos_enc((pts, 2.5e-3*size), 0, self.pospe), size.min(), size.max())
        if self.feape > 0:
            indata += [positional_encoding(features, self.feape)]
        mlp_in = torch.cat(indata, dim=-1)
        mlp_out = self.mlp(mlp_in)
        # mlp_out = mlp_out + 1e-1*torch.randn_like(mlp_out)
        # ic(indata, mlp_out)
        # ic(self.mlp[0].weight.grad, self.mlp[0].weight, self.mlp[-1].weight.grad, self.mlp[-1].weight)
        # for layer in self.mlp.children():
        #     if hasattr(layer, 'weight'):
        #         ic(layer.weight, layer.bias)

        normals = normalize(mlp_out)

        return normals


class AppDimNormal(torch.nn.Module):
    def __init__(self, in_channels=0, activation=torch.nn.Identity):
        super().__init__()
        self.activation = activation()
        self.lr = 1
        self.allocation = 3

    def forward(self, pts, features, **kwargs):
        start_ind = 0
        # raw_norms = features[..., start_ind:start_ind+3]
        raw_norms = features[..., start_ind : start_ind + 3]
        # raw_norms = 2*torch.sigmoid(raw_norms)-1
        raw_norms = self.activation(raw_norms)
        normals = raw_norms / (torch.norm(raw_norms, dim=-1, keepdim=True) + 1e-8)
        return normals


class MLPRender_PE(torch.nn.Module):
    def __init__(self, in_channels, viewpe=6, pospe=6, featureC=128):
        super().__init__()

        self.in_mlpC = (3 + 2 * viewpe * 3) + (3 + 2 * pospe * 3) + in_channels
        self.viewpe = viewpe
        self.pospe = pospe
        layer1 = torch.nn.Linear(self.in_mlpC, featureC)
        layer2 = torch.nn.Linear(featureC, featureC)
        layer3 = torch.nn.Linear(featureC, 3)

        self.mlp = torch.nn.Sequential(
            layer1,
            torch.nn.ReLU(inplace=True),
            layer2,
            torch.nn.ReLU(inplace=True),
            layer3,
        )
        torch.nn.init.constant_(self.mlp[-1].bias, 0)

    def forward(self, pts, viewdirs, features):
        indata = [features, viewdirs]
        if self.pospe > 0:
            indata += [positional_encoding(pts, self.pospe)]
        if self.viewpe > 0:
            indata += [positional_encoding(viewdirs, self.viewpe)]
        mlp_in = torch.cat(indata, dim=-1)
        rgb = self.mlp(mlp_in)
        rgb = torch.sigmoid(rgb)

        return rgb


class LearnableSphericalEncoding(torch.nn.Module):
    def __init__(self, out_channels, out_res):
        super().__init__()
        # out_res is the number of points used to represent the sphere
        # out channels is the number of channels per a point
        self.out_res = out_res
        self.out_channels = out_channels

        # http://extremelearning.com.au/how-to-evenly-distribute-points-on-a-sphere-more-effectively-than-the-canonical-fibonacci-lattice/#more-3069
        if out_res < 24:
            eps = 0.33
        elif out_res < 177:
            eps = 1.33
        elif out_res < 890:
            eps = 3.33

        weights = torch.rand((1, out_res, out_channels))
        # weights = torch.ones((1, out_res, out_channels))
        self.register_parameter("weights", torch.nn.Parameter(weights))

        indices = torch.arange(0, out_res, dtype=float)
        goldenRatio = (1 + 5**0.5) / 2

        phi = torch.arccos(1 - 2 * (indices + eps) / (out_res - 1 + 2 * eps))
        theta = 2 * pi * indices / goldenRatio

        x, y, z = (
            torch.cos(theta) * torch.sin(phi),
            torch.sin(theta) * torch.sin(phi),
            torch.cos(phi),
        )
        self.register_buffer("sphere_pos", torch.stack([x, y, z], dim=0).float())

    def forward(self, vec, sigma):
        # vec: N, 3 normal vectors representing input directions
        # output: N, C

        # cos_dist: N, M
        cos_dist = (vec @ self.sphere_pos).clip(min=-1 + 1e-5, max=1 - 1e-5)
        # weights: 1, M, C
        # output: (N, 1, M) @ (1, M, C) -> (N, 1, C)
        prob = normal_dist(torch.arccos(cos_dist), sigma)
        prob /= prob.sum(dim=1, keepdim=True) + 1e-8
        output = torch.matmul(prob.unsqueeze(1), self.weights)
        # import matplotlib.pyplot as plt
        # fig = plt.figure()
        # ax = fig.add_subplot(projection='3d')
        # # ax.scatter(self.sphere_pos[0].cpu(), self.sphere_pos[1].cpu(), self.sphere_pos[2].cpu(), c=prob[0].detach().cpu())
        # ic(self.weights.max(), sigma.min(), sigma.max())
        # col = self.weights[0].detach().cpu()
        # ax.scatter(self.sphere_pos[0].cpu(), self.sphere_pos[1].cpu(), self.sphere_pos[2].cpu(), c=torch.sigmoid(col))
        # plt.show()
        return output.squeeze(1)
