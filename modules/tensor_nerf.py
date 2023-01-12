import torch
import torch.nn.functional as F
import numpy as np
import time
from icecream import ic

from . import render_modules
from .tonemap import SRGBTonemap
import plotly.express as px
import plotly.graph_objects as go
import random
import hydra

from torch.autograd import grad
import matplotlib.pyplot as plt
import math
from .logger import Logger
import utils
from modules.row_mask_sum import row_mask_sum
from modules.distortion_loss_warp import calc_distortion_loss
from mutils import normalize

LOGGER = Logger(enable=False)
FIXED_SPHERE = False
FIXED_RETRO = False

def raw2alpha(sigma, dist):
    # sigma, dist  [N_rays, N_samples]
    # alpha = 1. - torch.exp((-sigma*dist).clip(min=-1e10, max=0))
    alpha = 1. - torch.exp(-sigma*dist)

    # T is the term that integrates the alpha backwards to prevent occluded objects from influencing things
    # multiply in exponential space to take exponential of integral
    T = torch.cumprod(torch.cat([
        torch.ones(alpha.shape[0], 1, device=alpha.device),
        1. - alpha + 1e-10
    ], dim=-1), dim=-1)

    weights = alpha * T[:, :-1]  # [N_rays, N_samples]
    return alpha, weights, T[:, -1:]

class TensorNeRF(torch.nn.Module):
    def __init__(self, rf, model, aabb, near_far,
                 sampler, tonemap=None, bg_module=None, normal_module=None,
                 alphaMask=None,
                 infinity_border=False, recur_stepmul=1,
                 rayMarch_weight_thres=0.0001, recur_weight_thres=1e-3,detach_inter=False, bg_noise=0, bg_noise_decay=0.999, use_predicted_normals=True,
                 orient_world_normals=False, align_pred_norms=True,
                 eval_batch_size=512, geonorm_iters=-1,
                 lr_scale=1,
                 **kwargs):
        super(TensorNeRF, self).__init__()
        self.rf = rf(aabb=aabb)
        self.normal_module = normal_module(self.rf.app_dim) if normal_module is not None else None
        self.sampler = sampler(near_far=near_far, aabb=aabb)
        self.model = model(self.rf.app_dim)
        self.bg_module = bg_module
        self.visibility_module = None
        if tonemap is None:
            self.tonemap = SRGBTonemap()
        else:
            self.tonemap = tonemap

        self.lr_scale = lr_scale
        self.bg_noise = bg_noise
        self.bg_noise_decay = bg_noise_decay
        self.alphaMask = alphaMask
        self.infinity_border = infinity_border
        self.rayMarch_weight_thres = rayMarch_weight_thres
        self.recur_weight_thres = recur_weight_thres
        self.eval_batch_size = eval_batch_size
        self.geonorm_iters = geonorm_iters
        self.recur_stepmul = recur_stepmul

        self.detach_inter = detach_inter

        self.use_predicted_normals = use_predicted_normals and self.normal_module is not None
        self.align_pred_norms = use_predicted_normals | align_pred_norms
        self.orient_world_normals = orient_world_normals | (not self.align_pred_norms)
        ic(self.use_predicted_normals, self.align_pred_norms, self.orient_world_normals)

    def get_device(self):
        return self.rf.units.device

    def get_optparam_groups(self):
        grad_vars = []
        grad_vars += self.rf.get_optparam_groups(self.lr_scale)
        grad_vars += self.model.get_optparam_groups(self.lr_scale)
        if isinstance(self.normal_module, torch.nn.Module):
            grad_vars += [{'params': self.normal_module.parameters(),
                           'lr': self.normal_module.lr*self.lr_scale}]
        if isinstance(self.bg_module, torch.nn.Module):
            grad_vars += self.bg_module.get_optparam_groups(self.lr_scale)
        return grad_vars

    def save(self, path, config):
        print(f"Saving nerf to {path}")
        if self.bg_module is not None:
            config['bg_module']['bg_resolution'] = self.bg_module.bg_resolution
        config['use_predicted_normals'] = self.use_predicted_normals
        ckpt = {'config': config, 'state_dict': self.state_dict()}
        # if self.alphaMask is not None:
        #     alpha_volume = self.alphaMask.alpha_volume.cpu()
        #     ckpt.update({'alphaMask': alpha_volume})
        #     #  alpha_volume = self.alphaMask.alpha_volume.cpu().numpy()
        #     #  ckpt.update({'alphaMask.shape': alpha_volume.shape})
        #     #  ckpt.update(
        #     #      {'alphaMask.mask': np.packbits(alpha_volume.reshape(-1))})
        #     ckpt.update({'alphaMask.aabb': self.alphaMask.aabb.cpu()})
        torch.save(ckpt, path)

    @staticmethod
    def load(ckpt, config=None, near_far=None, **kwargs):
        config = ckpt['config'] if config is None else config
        aabb = ckpt['state_dict']['rf.aabb']
        # if 'model.brdf_sampler.angs' in ckpt['state_dict']:
        #     c = ckpt['state_dict']['model.brdf_sampler.angs'].shape[0]
        #     ic(c)
        del ckpt['state_dict']['model.brdf_sampler.angs']
        # del ckpt['state_dict']['sampler.occupancy_grid._binary']
        # del ckpt['state_dict']['sampler.occupancy_grid.occs']
        # ic(ckpt['state_dict'].keys())
        near_far = near_far if near_far is not None else [1, 6]
        if 'rf.grid_size' in ckpt['state_dict']:
            grid_size = list(ckpt['state_dict']['rf.grid_size'])
            ic(grid_size)
        else:
            grid_size = None
        rf = hydra.utils.instantiate(config)(aabb=aabb, near_far=near_far, grid_size=grid_size)
        # if 'alphaMask.aabb' in ckpt.keys():
        #     #  length = np.prod(ckpt['alphaMask.shape'])
        #     #  alpha_volume = torch.from_numpy(np.unpackbits(ckpt['alphaMask.mask'])[
        #     #                                  :length].reshape(ckpt['alphaMask.shape'])).float()
        #     alpha_volume = ckpt['alphaMask']
        #     rf.alphaMask = utils.AlphaGridMask(
        #         ckpt['alphaMask.aabb'], alpha_volume)
        rf.load_state_dict(ckpt['state_dict'], **kwargs)
        return rf

    def check_schedule(self, iter, batch_mul):
        require_reassignment = False
        require_reassignment |= self.model.check_schedule(iter, batch_mul, bg_module=self.bg_module)
        require_reassignment |= self.sampler.check_schedule(iter, batch_mul, self.rf)
        require_reassignment |= self.rf.check_schedule(iter, batch_mul)
        if require_reassignment:
            self.sampler.update(self.rf, init=True)
        self.bg_noise *= self.bg_noise_decay
        if self.geonorm_iters > 0:
            self.use_predicted_normals = self.geonorm_iters*batch_mul < iter
        return require_reassignment

    def at_infinity(self, xyz_sampled, max_dist=10):
        margin = 1 - 1/max_dist/2
        at_infinity = torch.linalg.norm(
            xyz_sampled, dim=-1, ord=torch.inf).abs() >= margin
        return at_infinity

    def calculate_normals(self, xyz):
        with torch.enable_grad():
            xyz_g = xyz.clone()
            xyz_g.requires_grad = True

            # compute sigma
            validsigma = self.rf.compute_densityfeature(xyz_g, activate=False)

            # compute normal
            grad_outputs = torch.ones_like(validsigma)
            g = grad(validsigma, xyz_g, grad_outputs=grad_outputs, create_graph=True, allow_unused=True)
            # n = torch.linalg.norm(g[0][:, :3], dim=-1)
            # ic(g[0][:, :3].abs().max())
            norms = normalize(-g[0][:, :3])
            return norms

    def render_just_bg(self, rays, roughness):
        if rays.shape[0] == 0:
            return torch.empty((0, 3), device=rays.device)
        viewdirs = rays[:, 3:6]
        bg = self.bg_module(viewdirs[:, :], roughness)
        return bg.reshape(-1, 3)

    def forward(self, rays, focal, start_mipval=None, bg_col=torch.tensor([1, 1, 1]), stepmul=1,
                recur=0, override_near=None, output_alpha=None, dynamic_batch_size=True, gt_normals=None,
                is_train=False, ndc_ray=False, N_samples=-1, tonemap=True, draw_debug=True):
        # rays: (N, (origin, viewdir, ray_up))
        output = {}
        eps = torch.finfo(torch.float32).eps

        # sample points
        device = rays.device

        xyz_sampled, ray_valid, max_samps, z_vals, dists, whole_valid, extras = self.sampler.sample(
            rays, focal, ndc_ray=ndc_ray, override_near=override_near, is_train=is_train,
            stepmul=stepmul, dynamic_batch_size=dynamic_batch_size,
            N_samples=N_samples, rf=self.rf)
        # xyz_sampled: (M, 4) float. premasked valid sample points
        # ray_valid: (b, N) bool. mask of which samples are valid
        # max_samps = N
        # z_vals: (b, N) float. distance along ray to sample
        # dists: (b, N) float. distance between samples
        # whole_valid: mask into origin rays of which B rays where able to be fully sampled.
        B = ray_valid.shape[0]

        xyz_normed = self.rf.normalize_coord(xyz_sampled)
        full_shape = (B, max_samps, 3)
        n_samples = full_shape[1]

        M = xyz_sampled.shape[0]
 
        device = xyz_sampled.device

        viewdirs = rays[whole_valid, 3:6].view(-1, 1, 3).expand(full_shape)
        # rays_up = rays[:, 6:9]
        # rays_up = rays_up.view(-1, 1, 3).expand(full_shape)
        n_samples = full_shape[1]

        # sigma.shape: (N, N_samples)
        sigma = torch.zeros(full_shape[:-1], device=device)

        world_normal = torch.zeros((M, 3), device=device)

        all_app_features = None
        pred_norms = torch.zeros((M, 3), device=device)
        if ray_valid.any():
            if self.rf.separate_appgrid:
                psigma = self.rf.compute_densityfeature(xyz_sampled)
            else:
                psigma, all_app_features = self.rf.compute_feature(xyz_sampled)
            sigma[ray_valid] = psigma

        def recur_forward(rays, start_mipval):
            if recur < len(self.model.max_brdf_rays)-1:
                incoming_data, incoming_stats = self(rays, focal, recur=recur+1, bg_col=None, dynamic_batch_size=False, stepmul=self.recur_stepmul,
                                     start_mipval=start_mipval.reshape(-1), override_near=self.rf.stepSize*5, is_train=is_train,
                                     ndc_ray=False, N_samples=N_samples, tonemap=False, draw_debug=False)
                incoming_light = incoming_data['rgb_map']
            else:
                incoming_light = self.render_just_bg(rays, start_mipval.reshape(-1))
            return incoming_light


        # weight: [N_rays, N_samples]
        # ic((dists * self.rf.distance_scale).mean())
        alpha, weight, bg_weight = raw2alpha(sigma, dists * self.rf.distance_scale)

        # app stands for appearance
        pweight = weight[ray_valid]
        thres = self.rayMarch_weight_thres if recur == 0 else self.recur_weight_thres
        # app_mask = weight > thres
        app_mask = ray_valid
        papp_mask = app_mask[ray_valid]
        # ic(recur, ray_valid.shape, ray_valid.sum(), app_mask.shape, app_mask.sum())

        # if self.visibility_module is not None:
        #     self.visibility_module.ray_update(xyz_normed, viewdirs[ray_valid], app_mask, ray_valid)

        if app_mask.any():
            #  Compute normals for app mask
            app_xyz = xyz_sampled[papp_mask]

            # TODO REMOVE
            norms = self.calculate_normals(app_xyz)
            world_normal[papp_mask] = norms

            app_norm_xyz = xyz_normed[papp_mask]

            if all_app_features is None:
                app_features = self.rf.compute_appfeature(app_xyz)
            else:
                app_features = all_app_features[papp_mask]
                # _, app_features = self.rf.compute_feature(app_norm_xyz)

            # interpolate between the predicted and world normals
            if self.normal_module is not None:
                pred_norms = torch.zeros_like(pred_norms)
                pred_norms[papp_mask] = self.normal_module(app_norm_xyz, app_features, world_normal[papp_mask])
                v_world_normal = pred_norms if self.use_predicted_normals else world_normal
            else:
                v_world_normal = world_normal

            rgb, debug = self.model(app_xyz, app_features, viewdirs[app_mask], v_world_normal[papp_mask], weight, app_mask, weight.shape[0], recur, recur_forward)

        else:
            debug = {k: torch.empty((0, v), device=device, dtype=weight.dtype) for k, v in self.model.outputs.items()}
            rgb = torch.empty((0, 3), device=device, dtype=weight.dtype)
            v_world_normal = world_normal

        # calculate depth

        # shadow_map = torch.sum(weight * shadows, 1)
        # (N, bundle_size, bundle_size)
        if recur > 0 and self.detach_inter:
            weight = weight.detach()

        acc_map = torch.sum(weight, 1)
        # rgb_map = torch.sum(weight[..., None] * rgb.clip(min=0, max=1), -2)
        eweight = weight[app_mask][..., None]
        tmap_rgb = self.tonemap(rgb.clip(min=0, max=1))
        # rgb_map = row_mask_sum(eweight * tmap_rgb, app_mask)

        D = tmap_rgb.shape[1]
        index = extras['ray_indices'][:, None].expand(-1, D)
        rgb_map = torch.zeros((B, D), device=device, dtype=eweight.dtype)
        # ic(index.shape, (eweight * tmap_rgb).shape, rgb_map.shape, index.max())
        rgb_map.scatter_add_(0, index, eweight * tmap_rgb)

        images = {}
        statistics = dict(
            recur=recur,
            whole_valid=whole_valid, 
        )

        if not is_train and draw_debug:
            with torch.no_grad():
                depth_map = torch.sum(weight * z_vals, 1)
                # depth_map = depth_map + (1. - acc_map) * rays[whole_valid, -1]
                depth_map = depth_map + (1. - acc_map) * 0

                # view dependent normal map
                # N, 3, 3
                # row_basis = -torch.stack([
                #     -torch.linalg.cross(viewdirs[:, 0], rays_up[:, 0], dim=-1),
                #     viewdirs[:, 0],
                #     rays_up[:, 0],
                # ], dim=1)
                # d_normal_map = torch.matmul(row_basis, d_world_normal_map.unsqueeze(-1)).squeeze(-1)

                # world_normal_map = row_mask_sum(world_normal*pweight[..., None], ray_valid)
                world_normal_map = torch.zeros((B, 3), device=device, dtype=eweight.dtype)
                world_normal_map.scatter_add_(0, index, eweight * world_normal)
                world_normal_map = acc_map[..., None] * world_normal_map + (1 - acc_map[..., None])

                # pred_norm_map = row_mask_sum(pred_norms*pweight[..., None], ray_valid)
                pred_norm_map = torch.zeros((B, 3), device=device, dtype=eweight.dtype)
                pred_norm_map.scatter_add_(0, index, eweight * pred_norms)
                pred_norm_map = acc_map[..., None] * pred_norm_map + (1 - acc_map[..., None])

                if weight.shape[1] > 0:
                    inds = ((weight).max(dim=1).indices).clip(min=0)
                    full_xyz_sampled = torch.zeros((B, max_samps, 4), device=device)
                    full_xyz_sampled[ray_valid] = xyz_sampled

                    termination_xyz = full_xyz_sampled[range(full_shape[0]), inds].cpu()
                else:
                    termination_xyz = torch.empty(0, 4)

                # collect statistics about the surface
                # surface width in voxels
                surface_width = ray_valid.sum(dim=1)

                # TODO REMOVE
                LOGGER.log_norms_n_rays(xyz_sampled[papp_mask], v_world_normal[papp_mask], weight[app_mask])
                # o_world_normal = world_normal if self.orient_world_normals else pred_norms
                # ori_loss = torch.matmul(viewdirs[ray_valid].reshape(-1, 1, 3).detach(), o_world_normal.reshape(-1, 3, 1)).reshape(pweight.shape).clamp(min=0)**2
                # debug[ray_valid] = ori_loss.reshape(-1, 1).expand(-1, 3)
                # calculate cross section
                pcs_mask = xyz_normed[..., 2] < 0
                full_cs_mask = torch.zeros_like(weight, dtype=bool)
                full_cs_mask[ray_valid] = pcs_mask
                cross_section = row_mask_sum(full_cs_mask[app_mask][..., None] * eweight * rgb.clip(min=0, max=1), app_mask)
            images['cross_section'] = cross_section.detach().cpu()
            images['depth'] = depth_map.detach().cpu()
            images['world_normal'] = world_normal_map.detach().cpu()
            images['normal'] = pred_norm_map.detach().cpu()
            images['termination_xyz'] = termination_xyz
            images['surf_width'] = surface_width
            for k, v in debug.items():
                images[k] = row_mask_sum(v*eweight, app_mask)
        elif recur == 0:
            # viewdirs point inward. -viewdirs aligns with pred_norms. So we want it below 0
            o_world_normal = world_normal if self.orient_world_normals else pred_norms
            aweight = weight[app_mask]
            NdotV1 = (-viewdirs[app_mask].reshape(-1, 3).detach() * pred_norms[papp_mask].reshape(-1, 3)).sum(dim=-1)

            aweight = weight[app_mask]
            NdotV2 = (-viewdirs[app_mask].reshape(-1, 3).detach() * world_normal[papp_mask].reshape(-1, 3)).sum(dim=-1)
            ori_loss = (aweight * (NdotV1.clamp(max=0)**2 + NdotV2.clamp(max=0)**2)).sum()# / B

            midpoint = torch.cat([
                z_vals,
                (2*z_vals[:, -1] - z_vals[:, -2])[:, None],
            ], dim=1)
            # extend the dt artifically to the background
            dt = torch.cat([
                dists,
                0*dists[:, -2:-1]
            ], dim=1)
            full_weight = torch.cat([weight, 1-weight.sum(dim=1, keepdim=True)], dim=1)
            # TODO REMOVE
            distortion_loss = calc_distortion_loss(midpoint, full_weight, dt)
            # distortion_loss = torch.tensor(0.0, device=device) 

            if self.align_pred_norms:
                align_world_loss = 2*(1-(pred_norms[papp_mask] * world_normal[papp_mask]).sum(dim=-1))#**0.5
                prediction_loss = (aweight * align_world_loss).sum()# / B
            else:
                prediction_loss = torch.tensor(0.0)

            # output['diffuse_reg'] = (roughness-0.5).clip(min=0).mean() + tint.clip(min=1e-3).mean()
            # output['diffuse_reg'] = tint.clip(min=1e-3).mean()
            if self.bg_module is not None:
                envmap_brightness = self.bg_module.mean_color().mean()
                statistics['envmap_reg'] = (envmap_brightness).clip(min=0)
            else:
                statistics['envmap_reg'] = torch.tensor(0.0)

            if gt_normals is not None:
                gt_normals_mask = gt_normals[whole_valid].view(-1, 1, 3).expand(full_shape)[app_mask]
                gt_mask = gt_normals_mask.sum(dim=1) > 0.9
                pred_norm_err_a = 2*(1-(pred_norms[papp_mask][gt_mask] * gt_normals_mask[gt_mask]).sum(dim=-1))
                pred_norm_err_b = 2*(1-(world_normal[papp_mask][gt_mask] * gt_normals_mask[gt_mask]).sum(dim=-1))
                pred_norm_err = (aweight[gt_mask] * (pred_norm_err_a + pred_norm_err_b)).sum()# / B
                statistics['normal_err'] = pred_norm_err
            statistics['brdf_reg'] = -debug['tint'].mean() if 'tint' in debug else torch.tensor(0.0)
            statistics['diffuse_reg'] = debug['roughness'].sum() if 'roughness' in debug else torch.tensor(0.0)
            statistics['prediction_loss'] = prediction_loss
            statistics['ori_loss'] = ori_loss
            statistics['distortion_loss'] = distortion_loss
            for k, v in debug.items():
                images[k] = (v)

        # if recur > 0:
        #     ic(weight.sum(), z_vals, xyz_sampled[..., :3], rgb_map.max())

        # ic(weight.mean(), rgb.mean(), rgb_map.mean(), v_world_normal.mean(), sigma.mean(), dists.mean(), alpha.mean())

        # ic(rgb[app_mask].mean(dim=0), rgb_map[acc_map>0.1].mean(dim=0), weight.shape)
        if self.bg_module is not None and bg_col is None:
            bg_roughness = -100*torch.ones(B, 1, device=device) if start_mipval is None else start_mipval
            bg = self.bg_module(viewdirs[:, 0, :], bg_roughness).reshape(-1, 3)
            if tonemap:
                bg = self.tonemap(bg, noclip=True)
        else:
            # if white_bg or (is_train and torch.rand((1,)) < 0.5):
            #     if output_alpha is not None and self.bg_noise > 0:
            #         noise = (torch.rand((1, 1), device=device) > 0.5).float()*output_alpha[:, None] + (1-output_alpha[:, None])
            #         # noise = (torch.rand((*acc_map.shape, 1), device=device) > 0.5).float()*output_alpha[:, None] + (1-output_alpha[:, None])
            #     else:
            #         noise = 1-torch.rand((*acc_map.shape, 3), device=device)*self.bg_noise
            #     # noise = 1-torch.rand((*acc_map.shape, 3), device=device)*self.bg_noise
            #     rgb_map = rgb_map + (1 - acc_map[..., None]) * noise
            bg = bg_col.to(device).reshape(1, 3)
        rgb_map = rgb_map + (1 - acc_map[..., None]) * bg
            # if white_bg:
            #     if True:
            #         bg_col = torch.rand((1, 3), device=device).clip(min=torch.finfo(torch.float32).eps).sqrt()
            #         # rgb_map = rgb_map + (1 - acc_map[..., None]) * torch.rand_like(rgb_map)
            #         rgb_map = rgb_map + (1 - acc_map[..., None]) * torch.where(torch.rand_like(acc_map[..., None]) < 0.5, 0, 1)
            #         # rgb_map = rgb_map + (1 - acc_map[..., None]) * bg_col
            #     else:
            #         rgb_map = rgb_map + (1 - acc_map[..., None])

        # ic(rgb, rgb_map)
        # ic(opacity, acc_map)

        images['rgb_map'] = rgb_map
        images['acc_map'] = acc_map.detach().cpu()
        return images, statistics
