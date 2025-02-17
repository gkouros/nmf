import torch
import torch.nn.functional as F
from icecream import ic


class AlphaGridMask(torch.nn.Module):
    def __init__(self, aabb, alpha_volume, align_corners=True):
        super(AlphaGridMask, self).__init__()
        self.register_buffer("aabb", aabb)

        aabbSize = self.aabb[1] - self.aabb[0]
        invgrid_size = 1.0 / aabbSize * 2
        grid_size = torch.LongTensor(
            [alpha_volume.shape[-1], alpha_volume.shape[-2], alpha_volume.shape[-3]]
        )
        self.align_corners = align_corners
        self.register_buffer("grid_size", grid_size)
        self.register_buffer("invgrid_size", invgrid_size)
        self.register_buffer(
            "alpha_volume", alpha_volume.view(1, 1, *alpha_volume.shape[-3:])
        )

    def sample_alpha(self, xyz_sampled, contract_space=False):
        xyz_sampled = self.normalize_coord(xyz_sampled, contract_space)
        # ic(xyz_sampled.shape, self.alpha_volume.shape)
        alpha_vals = F.grid_sample(
            self.alpha_volume,
            xyz_sampled[..., :3].view(1, -1, 1, 1, 3),
            align_corners=True,
        ).view(-1)

        # H, W, D = self.alpha_volume.shape
        # i = ((xyz_sampled[..., 2] / 2 + 0.5) * (H - 1)).long().clip(0, H - 1)
        # j = ((xyz_sampled[..., 1] / 2 + 0.5) * (W - 1)).long().clip(0, W - 1)
        # k = ((xyz_sampled[..., 0] / 2 + 0.5) * (D - 1)).long().clip(0, D - 1)
        # alpha_vals = self.alpha_volume[i, j, k]

        # ic(alpha_vals.sum(), alpha_vals.shape)
        # alpha_vals = F.grid_sample(
        #         self.alpha_volume.reshape(1, 1, H, W, D).float(),
        #         xyz_sampled[..., :3].view(1, -1, 1, 1, 3),
        #         align_corners=self.align_corners,
        #         mode='nearest').view(-1)

        return alpha_vals

    def normalize_coord(self, xyz_sampled, contract_space):
        coords = (xyz_sampled[..., :3] - self.aabb[0]) * self.invgrid_size - 1
        size = xyz_sampled[..., 3:4]
        normed = torch.cat((coords, size), dim=-1)
        if contract_space:
            dist = (
                torch.linalg.norm(normed[..., :3], dim=-1, keepdim=True, ord=torch.inf)
                + 1e-8
            )
            direction = normed[..., :3] / dist
            contracted = torch.where(dist > 1, (2 - 1 / dist), dist) / 2 * direction
            return torch.cat([contracted, xyz_sampled[..., 3:]], dim=-1)
        else:
            return normed


class AlphaGridSampler(torch.nn.Module):
    def __init__(
        self,
        aabb,
        enable_alpha_mask=False,
        threshold=1e-4,
        multiplier=1,
        near_far=[2, 6],
        nEnvSamples=0,
        alphaMask_thres=0.001,
        update_list=[],
        max_samples=-1,
    ):
        super().__init__()
        self.aabb = aabb
        self.enable_alpha_mask = enable_alpha_mask
        self.alphaMask = None
        self.threshold = threshold
        self.nEnvSamples = nEnvSamples
        self.multiplier = int(multiplier)
        self.near_far = near_far
        self.update_list = update_list
        self.grid_size = [128, 128, 128]
        self.cumrand = True
        self.single_jitter = False
        self.alphaMask_thres = alphaMask_thres
        self.max_samples = max_samples

    def check_schedule(self, iteration, batch_mul, rf):
        if iteration in self.update_list:
            self.update(rf)
        return False

    @torch.no_grad()
    def update(self, rf, init=False):
        self.aabb = rf.aabb
        self.units = rf.units
        self.contract_space = rf.contract_space
        # reso_mask = reso_cur
        self.nSamples = rf.nSamples * self.multiplier
        self.stepsize = rf.stepsize / self.multiplier
        if not init:
            new_aabb = self.updateAlphaMask(rf, rf.grid_size)
            apply_correction = not torch.all(
                torch.tensor(self.grid_size).to(rf.grid_size.device) == rf.grid_size
            )
            # rf.shrink(new_aabb, apply_correction)
            self.grid_size = rf.grid_size
        ic(self.nSamples, self.stepsize)

    def sample_ray_ndc(self, rays_o, rays_d, focal, is_train=True, N_samples=-1):
        N_samples = N_samples if N_samples > 0 else self.nSamples
        near, far = self.near_far
        interpx = torch.linspace(near, far, N_samples).unsqueeze(0).to(rays_o)
        if is_train:
            l = torch.rand_like(interpx)
            interpx += l.to(rays_o) * ((far - near) / N_samples)

        rays_pts = rays_o[..., None, :] + rays_d[..., None, :] * interpx[..., None]
        mask_outbbox = ((self.aabb[0] > rays_pts) | (rays_pts > self.aabb[1])).any(
            dim=-1
        )

        # add size
        rays_pts = torch.cat([rays_pts, interpx.unsqueeze(-1) / focal], dim=-1)

        return rays_pts, interpx, ~mask_outbbox

    def sample_ray(
        self,
        rays_o,
        rays_d,
        focal,
        is_train=True,
        override_near=None,
        N_samples=-1,
        N_env_samples=-1,
    ):
        # focal: ratio of meters to pixels at a distance of 1 meter
        N_samples = N_samples if N_samples > 0 else self.nSamples
        N_env_samples = N_env_samples if N_env_samples > 0 else self.nEnvSamples
        device = rays_o.device
        stepsize = self.stepsize
        near, far = self.near_far
        if override_near is not None:
            near = override_near
        vec = torch.where(rays_d == 0, torch.full_like(rays_d, 1e-6), rays_d)
        rate_a = (self.aabb[1].to(rays_o) - rays_o) / vec
        rate_b = (self.aabb[0].to(rays_o) - rays_o) / vec
        t_min = torch.minimum(rate_a, rate_b).amax(-1).clamp(min=near, max=far)
        # t_min = near * torch.ones_like(t_min)

        rng = torch.arange(N_samples, device=rays_o.device)[None].float()
        # extend rng to sample towards infinity
        if N_env_samples > 0:
            ext_rng = (
                N_samples
                + N_env_samples
                / torch.linspace(
                    1, 1 / N_env_samples, N_env_samples, device=rays_o.device
                )[None].float()
            )
            rng = torch.cat([rng, ext_rng], dim=1)

        if is_train:
            if self.cumrand:
                steps = (
                    torch.rand((rays_d.shape[-2], N_samples), device=device) * stepsize
                    + stepsize / 2
                )
                step = torch.cumsum(steps, dim=1)
            else:
                rng = rng.repeat(rays_d.shape[-2], 1)
                # N, N_samples
                # add noise along each ray
                brng = rng.reshape(-1, N_samples + N_env_samples)
                # brng = brng + torch.rand_like(brng[:, [0], [0]])
                # r = torch.rand_like(brng[:, 0:1, 0:1])
                # r = torch.rand_like(brng[:, 0:1])
                if self.single_jitter:
                    r = torch.rand_like(brng[:, 0:1])
                else:
                    r = torch.rand_like(brng)
                brng = brng + r
                rng = brng.reshape(-1, N_samples + N_env_samples)
                step = stepsize * rng
        else:
            step = stepsize * rng
        # steps = torch.rand((rays_d.shape[-2], N_samples), device=device) * stepsize * 2
        interpx = t_min[..., None] + step

        rays_pts = rays_o[..., None, :] + rays_d[..., None, :] * interpx[..., None]
        mask_outbbox = ((self.aabb[0] > rays_pts) | (rays_pts > self.aabb[1])).any(
            dim=-1
        )

        # add size
        rays_pts = torch.cat([rays_pts, interpx.unsqueeze(-1) / focal], dim=-1)
        env_mask = torch.zeros_like(mask_outbbox)
        env_mask[:, N_samples:] = 1

        if self.contract_space:
            mask_outbbox = torch.zeros_like(mask_outbbox)

        return rays_pts, interpx, ~mask_outbbox, env_mask

    def compute_alpha(self, xyz_locs, rf, length=1):
        if self.alphaMask is not None:
            alphas = self.alphaMask.sample_alpha(xyz_locs)
            alpha_mask = alphas > 0
        else:
            alpha_mask = torch.ones_like(xyz_locs[:, 0], dtype=bool)

        sigma = torch.zeros(xyz_locs.shape[:-1], device=xyz_locs.device)

        if alpha_mask.any():
            validsigma = rf.compute_densityfeature(xyz_locs[alpha_mask])
            sigma[alpha_mask] = validsigma

        alpha = 1 - torch.exp(-sigma * length).view(xyz_locs.shape[:-1])

        return alpha

    @torch.no_grad()
    def getDenseAlpha(self, rf, grid_size=None):
        grid_size = self.grid_size if grid_size is None else grid_size

        samples = torch.stack(
            torch.meshgrid(
                torch.linspace(0, 1, grid_size[0]),
                torch.linspace(0, 1, grid_size[1]),
                torch.linspace(0, 1, grid_size[2]),
            ),
            -1,
        ).to(rf.get_device())
        dense_xyz = self.aabb[0] * (1 - samples) + self.aabb[1] * samples

        # dense_xyz = dense_xyz
        # print(self.stepsize, self.distance_scale*self.aabbDiag)
        alpha = torch.zeros_like(dense_xyz[..., 0])
        for i in range(grid_size[0]):
            alpha[i] = self.compute_alpha(
                dense_xyz[i].view(-1, 3), rf, self.stepsize
            ).view((grid_size[1], grid_size[2]))
        return alpha, dense_xyz

    @torch.no_grad()
    def updateAlphaMask(self, rf, grid_size=(200, 200, 200)):
        alpha, dense_xyz = self.getDenseAlpha(rf, grid_size)
        dense_xyz = dense_xyz.transpose(0, 2).contiguous()
        alpha = alpha.clamp(0, 1).transpose(0, 2).contiguous()[None, None]
        total_voxels = grid_size[0] * grid_size[1] * grid_size[2]

        ks = 3
        alpha = F.max_pool3d(alpha, kernel_size=ks, padding=ks // 2, stride=1).view(
            list(grid_size)[::-1]
        )
        alpha[alpha >= self.alphaMask_thres] = 1
        alpha[alpha < self.alphaMask_thres] = 0

        self.alphaMask = AlphaGridMask(self.aabb, alpha).to(rf.get_device())

        valid_xyz = dense_xyz[alpha > 0.5]

        xyz_min = valid_xyz.amin(0)
        xyz_max = valid_xyz.amax(0)

        new_aabb = torch.stack((xyz_min, xyz_max))

        total = torch.sum(alpha)
        print(
            f"bbox: {xyz_min, xyz_max} alpha rest %%%f" % (total / total_voxels * 100)
        )
        return new_aabb

    @torch.no_grad()
    def sample(
        self,
        rays_chunk,
        focal,
        rf,
        override_near=None,
        is_train=False,
        dynamic_batch_size=True,
        override_alpha_thres=None,
        stepmul=1,
        ndc_ray=False,
        **args,
    ):
        """
        Parameters:
            rays_chunk: (B, 6) float. (ox, oy, oz, dx, dy, dz) ray origin and direction
            focal: focal length of camera projecting ray. Unused
            ndc_ray: whether to sample ray in normalized device coordinates
            override_near: an override on what the closest point sampled can be
            is_train: varies sampler based sampler_mode
            N_samples: override how many points are sampled
        Returns:
            xyz_sampled: (M, 4) float. premasked valid sample points
            ray_valid: (b, N) bool. mask of which samples are valid
            max_samps = N
            z_vals: (b, N) float. distance along ray to sample
            dists: (b, N) float. distance between samples
            whole_valid: mask into origin rays_chunk of which b rays where able to be fully sampled.
        """
        N_samples = -1
        viewdirs = rays_chunk[:, 3:6]
        if ndc_ray:
            xyz_sampled, z_vals, ray_valid = self.sample_ray_ndc(
                rays_chunk[:, :3],
                viewdirs,
                focal,
                is_train=is_train,
                N_samples=int(N_samples * stepmul),
            )
            dists = torch.cat(
                (z_vals[:, 1:] - z_vals[:, :-1], torch.zeros_like(z_vals[:, :1])),
                dim=-1,
            )
            rays_norm = torch.norm(viewdirs, dim=-1, keepdim=True)
            dists = dists * rays_norm
            viewdirs = viewdirs / rays_norm
        else:
            xyz_sampled, z_vals, ray_valid, env_mask = self.sample_ray(
                rays_chunk[:, :3],
                viewdirs,
                focal,
                is_train=is_train,
                N_samples=int(N_samples * stepmul),
                override_near=override_near,
            )
            dists = torch.cat(
                (z_vals[:, 1:] - z_vals[:, :-1], torch.zeros_like(z_vals[:, :1])),
                dim=-1,
            )

        device = rays_chunk.device
        # sample alphas and cull samples from the ray
        if self.alphaMask is not None and self.enable_alpha_mask:
            alphas = self.alphaMask.sample_alpha(xyz_sampled[ray_valid])
            alpha_mask = alphas > 0
            ray_invalid = ~ray_valid
            ray_invalid[ray_valid] |= ~alpha_mask
            ray_valid = ~ray_invalid

        dists = torch.cat(
            (z_vals[:, 1:] - z_vals[:, :-1], torch.zeros_like(z_vals[:, :1])), dim=-1
        )
        N, M = xyz_sampled.shape[:2]

        if (
            self.max_samples > 0
            and is_train
            and dynamic_batch_size
            and ray_valid.sum() > self.max_samples
        ):
            whole_valid = torch.cumsum(ray_valid.sum(dim=1), dim=0) < self.max_samples
            ray_valid = ray_valid[whole_valid, :]
            xyz_sampled = xyz_sampled[whole_valid, :]
            N, M = xyz_sampled.shape[:2]
            z_vals = z_vals[whole_valid, :]
            dists = dists[whole_valid, :]
            xyzs = xyz_sampled[ray_valid]
        else:
            whole_valid = torch.ones((N), dtype=bool, device=device)
            xyzs = xyz_sampled[ray_valid]

        return (xyzs, ray_valid, M, z_vals, dists, whole_valid)
