import datetime
import math
import os
import sys
from pathlib import Path

import hydra
import torch
from icecream import ic
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from dataLoader import dataset_dict
from modules.integral_equirect import IntegralEquirect
from modules.tensor_nerf import TensorNeRF
from samplers.simple_sampler import SimpleSampler
from samplers.patch_sampler import PatchSampler
from mutils import normalize
from renderer import *
from utils import *

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

# torch.autograd.set_detect_anomaly(True)

# from torch.profiler import profile, record_function, ProfilerActivity


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

renderer = chunk_renderer


@torch.no_grad()
def render_test(args):
    params = args.model.params
    params = args.model.params
    expname = f"{args.dataset.scenedir.split('/')[-1]}_{args.expname}"
    ic(expname)

    if args.ckpt is None:
        args.ckpt = os.path.join(args.basedir, expname, expname + '.th')

    if not os.path.exists(args.ckpt):
        logger.info("the ckpt path does not exists!!")
        return

    white_bg = True

    # init dataset
    dataset = dataset_dict[args.dataset.dataset_name]
    test_dataset = dataset(
        os.path.join(args.datadir, args.dataset.scenedir),
        split="test",
        downsample=args.dataset.downsample_train,
        is_stack=True,
        white_bg=white_bg,
    )
    ic(test_dataset.near_far)
    test_dataset.near_far = args.dataset.near_far
    ndc_ray = args.dataset.ndc_ray

    ckpt = torch.load(args.ckpt)
    tensorf = TensorNeRF.load(
        ckpt, args.model.arch, near_far=test_dataset.near_far, strict=False
    )

    tensorf = tensorf.to(device)
    tensorf.train()
    tensorf.sampler.update(tensorf.rf, init=True)
    tensorf.sampler.updateAlphaMask(tensorf.rf, grid_size=[266] * 3)
    tensorf.load_state_dict(ckpt["state_dict"], strict=False)
    ic(tensorf.sampler.near_far)
    # for i in range(1000):
    #     tensorf.sampler.check_schedule(i, 1, tensorf.rf)

    # if tensorf.bright_sampler is not None:
    #     tensorf.bright_sampler.update(tensorf.bg_module)
    if args.fixed_bg is not None:
        bg_sd = torch.load(args.fixed_bg)
        bg_module = IntegralEquirect(
            bg_resolution=512,
            mipbias=0,
            activation="exp",
            lr=0.001,
            init_val=-1.897,
            mul_lr=0.001,
            brightness_lr=0,
            betas=[0.0, 0.0],
            mul_betas=[0.9, 0.9],
            mipbias_lr=1e-4,
            mipnoise=0.0,
        )
        bg_module.load_state_dict(bg_sd)
        bg_module.lr = 0
        bg_module.mul_lr = 0
        bg_module.brightness_lr = 0
        a = bg_module.bg_mat.reshape(-1, 3).mean(dim=-1)
        b = tensorf.bg_module.bg_mat.reshape(-1, 3).mean(dim=-1)
        a.sort()
        b.sort()
        a0 = a[500]
        a1 = a[-500]
        b0 = b[500]
        b1 = b[-500]
        # a0 = torch.quantile(a, 0.05)
        # a1 = torch.quantile(a, 0.95)
        # b0 = torch.quantile(b, 0.05)
        # b1 = torch.quantile(b, 0.95)
        new_mul = (tensorf.bg_module.mul * (b1 - b0)) / (bg_module.mul * (a1 - a0))
        new_mul = 1
        bg_module.mul *= new_mul
        # offset = tensorf.bg_module.mean_color().mean() / bg_module.mean_color().mean()
        # ic(new_mul, offset, torch.log(offset))
        # bg_module.brightness += torch.log(offset)

        # bg_module.mul += 1
        tensorf.bg_module = bg_module.to(device)

    logfolder = os.path.dirname(args.ckpt)

    """ Render train path """
    if args.render_train:
        os.makedirs(f"{logfolder}/imgs_train_all", exist_ok=True)
        train_dataset = dataset(
            os.path.join(args.datadir, args.dataset.scenedir),
            split="train",
            downsample=args.dataset.downsample_train,
            is_stack=True,
            white_bg=white_bg,
            is_testing=True,
        )
        test_res = evaluation(
            train_dataset,
            tensorf,
            args,
            renderer,
            f"{logfolder}/imgs_train_all/",
            N_vis=-1,
            N_samples=-1,
            white_bg=white_bg,
            ndc_ray=ndc_ray,
            device=device,
        )
        logger.info(
            f'======> {expname} train all psnr: {np.mean(test_res["psnrs"])} <========================'
        )

    """ Render test views """
    if args.render_test:
        folder = f"{logfolder}/imgs_test_all"
        os.makedirs(folder, exist_ok=True)
        logger.info(f"Saving test to {folder}")
        evaluation(
            test_dataset,
            tensorf,
            args,
            renderer,
            folder,
            N_vis=-1,
            N_samples=-1,
            white_bg=white_bg,
            ndc_ray=ndc_ray,
            device=device,
        )

    """ Render test path """
    torch.cuda.empty_cache()
    if args.render_path:
        c2ws = test_dataset.render_path
        # c2ws = test_dataset.poses
        logger.info("========>", c2ws.shape)
        os.makedirs(f"{logfolder}/imgs_path_all", exist_ok=True)
        evaluation_path(
            test_dataset,
            tensorf,
            c2ws,
            renderer,
            f"{logfolder}/imgs_path_all/",
            N_vis=-1,
            N_samples=-1,
            white_bg=white_bg,
            ndc_ray=ndc_ray,
            device=device,
            gt_bg=gt_bg,
            bundle_size=args.bundle_size
        )

def reconstruction(args):
    params = args.model.params
    expname = f"{args.dataset.scenedir.split('/')[-1]}_{args.expname}"
    ic(expname)

    # init training and testing datasets
    dataset = dataset_dict[args.dataset.dataset_name]
    stack_norms = args.dataset.get("stack_norms", False)
    white_bg = args.dataset.get("white_bg", True)
    patch_size = args.dataset.get("patch_size", 1)
    train_dataset = dataset(
        os.path.join(args.datadir, args.dataset.scenedir),
        split="train",
        downsample=args.dataset.downsample_train,
        is_stack=False,
        stack_norms=stack_norms,
        white_bg=white_bg,
        patch_size=patch_size,
    )
    test_dataset = dataset(
        os.path.join(args.datadir, args.dataset.scenedir),
        split="test",
        downsample=args.dataset.downsample_train,
        is_stack=True,
        white_bg=white_bg,
        is_testing=True,
    )
    if hasattr(args.dataset, 'near_far'):
        train_dataset.near_far = args.dataset.near_far
    ndc_ray = args.dataset.ndc_ray

    if args.add_timestamp:
        logfolder = f'{args.basedir}/{expname}{datetime.datetime.now().strftime("-%Y%m%d-%H%M%S")}'
    else:
        logfolder = f"{args.basedir}/{expname}"
    logger.add(logfolder + "/{time}.log", level="INFO", rotation="100 MB")

    # init log file
    os.makedirs(logfolder, exist_ok=True)
    os.makedirs(f"{logfolder}/imgs_vis", exist_ok=True)
    summary_writer = SummaryWriter(logfolder)

    aabb_scale = (1 if not hasattr(args.dataset, "aabb_scale") else args.dataset.aabb_scale)
    aabb = train_dataset.scene_bbox.to(device) * aabb_scale

    tensorf = hydra.utils.instantiate(args.model.arch)(aabb=aabb, near_far=train_dataset.near_far)
    if args.ckpt is not None:
        # TODO REMOVE
        ckpt = torch.load(args.ckpt)
        tensorf = TensorNeRF.load(ckpt, args.model.arch, strict=False)

    # use fixed GT envmap
    if args.fixed_bg is not None:
        tensorf.set_fixed_bg(args.fixed_bg)

    # send model to device and set in training mode
    tensorf = tensorf.to(device)
    tensorf.train()

    # setup lr schedule
    lr_bg = 1e-5
    grad_vars = tensorf.get_optparam_groups()
    if args.lr_decay_iters > 0:
        lr_factor = args.lr_decay_target_ratio ** (1 / args.lr_decay_iters)
    else:
        args.lr_decay_iters = params.n_iters
        lr_factor = args.lr_decay_target_ratio ** (1 / params.n_iters)

    # upsampling scheme for envmap
    upsamp_bg = hasattr(params, "bg_upsamp_res") and tensorf.bg_module is not None
    if upsamp_bg:
        res = params.bg_upsamp_res.pop(0)
        lr_bg = params.bg_upsamp_lr.pop(0)
        logger.info(f"Upsampling bg to {res}")
        tensorf.bg_module.upsample(res)
        ind = [i for i, d in enumerate(grad_vars) if "name" in d and d["name"] == "bg"][0]
        grad_vars[ind]["params"] = tensorf.bg_module.parameters()
        grad_vars[ind]["lr"] = lr_bg

    torch.cuda.empty_cache()
    PSNRs, PSNRs_test = [], [0]

    # get rays and gt colors
    allrays = train_dataset.all_rays.to(device)  # [N*H*W, 6]
    allrgbs = train_dataset.all_rgbs.to(device)  # [N*H*W, 4]

    # get width and height of images
    width, height = train_dataset.img_wh

    # create ray sampler
    # trainingSampler = SimpleSampler(allrays.shape[0], params.batch_size, device)
    trainingSampler = PatchSampler(allrays.shape[0], params.batch_size, width, height, device,
                                   patch_size=args.dataset.get("patch_size", 1))

    # init total variation
    TV_weight_density, TV_weight_app = params.TV_weight_density, params.TV_weight_app
    tvreg = TVLoss()
    logger.info(f"initial TV_weight density: {TV_weight_density} appearance: {TV_weight_app}")

    # ratio of meters to pixels at a distance of 1 meter
    focal = train_dataset.focal[0] if ndc_ray else train_dataset.fx

    # update sampler
    tensorf.sampler.update(tensorf.rf, init=True)

    # TODO REMOVE
    # initialize density field with random values
    if args.ckpt is None:
        # pretraining
        if tensorf.rf.num_pretrain > 0:
            # dparams = tensorf.parameters()
            # space_optim = torch.optim.Adam(tensorf.rf.dbasis_mat.parameters(), lr=0.5, betas=(0.9,0.99))
            space_optim = torch.optim.Adam(tensorf.parameters(), lr=0.005, betas=(0.9, 0.99))
            pbar = tqdm(range(tensorf.rf.num_pretrain))
            for _ in pbar:
                xyz = (torch.rand(20000, 3, device=device) * 2 - 1) * tensorf.rf.aabb[1].reshape(1, 3)
                sigma_feat = tensorf.rf.compute_densityfeature(xyz)
                step_size = tensorf.sampler.stepsize
                alpha = 1 - torch.exp(-sigma_feat * step_size * tensorf.rf.distance_scale)
                # sigma = 1-torch.exp(-sigma_feat)
                # loss = (sigma-torch.rand_like(sigma)*args.start_density).abs().mean()
                # target_alpha = (params.start_density+params.start_density*(2*torch.rand_like(alpha)-1))
                target_alpha = (params.start_density + 0.1 * params.start_density * torch.randn_like(alpha))
                # target_alpha = target_alpha.clip(min=params.start_density/2, max=params.start_density*2)
                # target_alpha = params.start_density
                loss = (alpha - target_alpha).abs().mean()
                # loss = (-sigma[mask].clip(max=1).sum() + sigma[~mask].clip(min=1e-8).sum())
                space_optim.zero_grad()
                loss.backward()
                pbar.set_description(f"Mean alpha: {alpha.detach().mean().item():.06f}.")
                space_optim.step()
        elif tensorf.rf.calibrate:
            # calculate alpha mean
            xyz = (torch.rand(20000, 3, device=device) * 2 - 1) * tensorf.rf.aabb[1].reshape(1, 3)
            sigma_feat = tensorf.rf.compute_densityfeature(xyz)
            target_sigma = -math.log(1 - params.start_density) / (tensorf.sampler.stepsize * tensorf.rf.distance_scale)

            # compute density_shift assume exponential activation
            density_shift = math.log(target_sigma) - math.log(sigma_feat.mean().item())
            # ic(target_sigma, sigma_feat.mean(), density_shift, sigma_feat.mean())
            tensorf.rf.density_shift += density_shift
            args.field.density_shift = tensorf.rf.density_shift

    # tensorf.sampler.mark_untrained_grid(train_dataset.poses, train_dataset.intrinsics)
    xyz = (torch.rand(20000, 3, device=device) * 2 - 1) * tensorf.rf.aabb[1].reshape(1, 3)
    sigma_feat = tensorf.rf.compute_densityfeature(xyz)
    torch.cuda.empty_cache()
    tensorf.sampler.update(tensorf.rf, init=True)
    torch.cuda.empty_cache()

    # calibrate model
    xyz = torch.rand(100000, 4, device=device) * 2 - 1
    xyz[:, 3] *= 0
    sigma_feat = tensorf.rf.compute_densityfeature(xyz)
    alpha = 1 - torch.exp(-sigma_feat * tensorf.sampler.stepsize * tensorf.rf.distance_scale)
    feat = tensorf.rf.compute_appfeature(xyz)
    bg_brightness = tensorf.bg_module.mean_color().detach().mean()
    args = tensorf.model.calibrate(args, xyz, feat, bg_brightness)

    # initialize optimizer and LR scheduler
    optimizer, scheduler = init_optimizer(tensorf, grad_vars, params)

    # decay of orientation loss weight
    ori_decay = (
        math.exp(math.log(params.final_ori_lambda / params.ori_lambda) / params.n_iters)
        if params.ori_lambda > 0 and params.final_ori_lambda is not None
        else 1
    )
    # decay of predicted normal loss weight
    normal_decay = (
        math.exp(math.log(params.final_pred_lambda / params.pred_lambda) / params.n_iters)
        if params.pred_lambda > 0 and params.final_pred_lambda is not None
        else 1
    )

    # save config in log dir of experiment
    OmegaConf.save(config=args, f=f"{logfolder}/config.yaml")

    # initialize training params
    num_rays = max(params.starting_batch_size, patch_size ** 2)
    prev_n_samples = None
    hist_n_samples = None

    # load gt background envmap for evaluation
    gt_bg_path = args.gt_bg if args.gt_bg is not None else None
    if hasattr(args.dataset, "gt_bg") and args.dataset.gt_bg is not None:
        gt_bg_path = Path("backgrounds") / args.dataset.gt_bg
    ic(gt_bg_path)
    gt_bg = cv2.imread(str(gt_bg_path)) if gt_bg_path is not None else None

    """ Main training loop """
    if True:
        # with torch.profiler.profile(record_shapes=True, schedule=torch.profiler.schedule(wait=1, warmup=1, active=params.n_iters-1), with_stack=True) as p:
        # with torch.autograd.detect_anomaly():
        pbar = tqdm(range(params.n_iters), miniters=args.progress_refresh_rate, file=sys.stdout)
        for iteration in pbar:
            optimizer.zero_grad(set_to_none=True)
            losses, roughnesses, envmap_regs, diffuse_regs = [], [], [], []
            brdf_regs = []
            pred_losses, ori_losses = [], []
            smoothness_losses = []
            TVs = []
            lbatch_size = min(
                params.min_batch_size if num_rays < params.min_batch_size else num_rays,
                params.max_batch_size,
            )
            num_remaining = lbatch_size
            while num_remaining > 0:
                lnum_rays = min(num_rays, num_remaining)
                # lnum_rays must be a multiple of patch_size
                lnum_rays = max(lnum_rays // patch_size**2 * patch_size**2, patch_size ** 2)
                num_remaining = max(num_remaining - lnum_rays, 0)

                # sample rays
                ray_idx = trainingSampler.nextids(lnum_rays)
                rays_train = allrays[ray_idx].to(device)
                rgba_train = allrgbs[ray_idx].reshape(-1, allrgbs.shape[-1]).to(device)
                match params.bg_col:
                    case "rand":
                        bg_col = torch.rand(3, device=device)
                    case "white":
                        bg_col = torch.ones((3), device=device)
                    case "black":
                        bg_col = torch.zeros((3), device=device)
                    case _:
                        raise Exception(f"Unknown bg col: {params.bg_col}")
                if rgba_train.shape[-1] == 4:
                    # blend A to RGB
                    rgb_train = (rgba_train[:, :3] * rgba_train[:, -1:] + (1 - rgba_train[:, -1:]) * bg_col)
                    alpha_train = rgba_train[..., 3]
                else:
                    rgb_train = rgba_train
                    alpha_train = None
                gt_normal_map = (
                    train_dataset.all_norms[ray_idx].to(device)
                    if train_dataset.stack_norms
                    else None
                )
                with torch.cuda.amp.autocast(enabled=args.fp16):
                    ims, stats = renderer(
                        rays_train,
                        tensorf,
                        gt_normals=gt_normal_map,
                        keys=[
                            "rgb_map",
                            "depth_map",
                            "world_normal_map",
                            "pred_normal_map",
                            "normal_err",
                            "distortion_loss",
                            "prediction_loss",
                            "ori_loss",
                            "diffuse_reg",
                            "roughness",
                            "whole_valid",
                            "envmap_reg",
                            "brdf_reg",
                            "n_samples",
                        ],
                        focal=focal,
                        output_alpha=alpha_train,
                        chunk=num_rays,
                        bg_col=bg_col,
                        is_train=True,
                        ndc_ray=ndc_ray,
                    )
                    n_samples = stats["n_samples"]
                    if n_samples[0] == 0:
                        continue
                    prediction_loss = stats["prediction_loss"].sum()
                    ori_loss = stats["ori_loss"].sum()
                    distortion_loss = stats["distortion_loss"].sum()
                    diffuse_reg = stats["diffuse_reg"].sum()
                    envmap_reg = stats["envmap_reg"].sum()
                    brdf_reg = stats["brdf_reg"].sum()
                    rgb_map = ims["rgb_map"]
                    depth_map_valid = ims["depth_map"]
                    world_normal_map_valid = ims["world_normal_map"]
                    pred_normal_map_valid = ims["pred_normal_map"]
                    whole_valid = stats["whole_valid"]
                    if not train_dataset.hdr:
                        rgb_map = rgb_map.clip(max=1)

                    """ Photometric Loss """
                    if params.charbonier_loss:
                        loss = torch.sqrt((rgb_map - rgb_train[whole_valid]) ** 2 + params.charbonier_eps**2).sum()
                    else:
                        if tensorf.hdr:
                            loss = (F.huber_loss(rgb_map, rgb_train[whole_valid], delta=1, reduction="none").sum())
                        else:
                            loss = ((rgb_map.clip(0, 1) - rgb_train[whole_valid].clip(0, 1)) ** 2).sum()
                    norm_err = (
                        sum(stats["normal_err"])
                        if type(stats["normal_err"]) == list
                        else stats["normal_err"].sum()
                    )
                    photo_loss = (((rgb_map.clip(0, 1) - rgb_train[whole_valid].clip(0, 1)) ** 2).mean().detach())

                    """ Depth smoothness loss """
                    if patch_size > 1 and params.smoothness_gamma > 0:
                        num_channels = 3 if params.smooth_normals else 1
                        patches = torch.zeros(whole_valid.shape + (num_channels,), dtype=depth_map_valid.dtype, device=device)
                        patches[whole_valid] = depth_map_valid[None] if not params.smooth_normals else world_normal_map_valid
                        patches = patches.reshape(-1, patch_size ** 2, num_channels)  # (B/(P*P), P*P, C)
                        patch_mid_offset = patch_size ** 2 // 2  # offset to middle point of patch
                        delta_patch = (patches - patches[:, None, patch_mid_offset]).abs().sum(axis=-1)  # patch depth differences from mid
                        if params.bilateral_smoothness:
                            rgb_patches = rgb_train.reshape(-1, patch_size ** 2, rgb_train.shape[-1])  # (B/(P*P), P*P, 3)
                            delta_rgb = rgb_patches - rgb_patches[:, None, patch_mid_offset]  # (B/(P*P), P*P, 3) color difference of patch from mid
                            weights = torch.exp(-delta_rgb.abs().sum(axis=-1) / params.smoothness_gamma)  # patch color consistency weights
                        else:
                            weights = torch.ones_like(delta_depth)
                        smoothness_loss = (weights * delta_patch).sum()  # the smoothness loss
                    else:
                        smoothness_loss = torch.tensor(0.0, device=device)

                    # adjust number of rays
                    # need to store mean ratios if I have any hope of stabilizing this
                    mean_samples = n_samples
                    ratio = int(whole_valid.sum()) / mean_samples[0]
                    mean_ratio = (
                        ratio
                        if prev_n_samples is None
                        else min(0.1 * ratio + 0.9 * prev_n_samples, ratio)
                    )
                    prev_n_samples = mean_ratio
                    num_rays = int(mean_ratio * params.target_num_samples + 1)
                    tensorf.model.update_n_samples(n_samples[1:])
                    # tensorf.eval_batch_size = num_rays // 4

                    # rays_remaining -= rgb_map.shape[0]
                    # rays_train = rays_train[~whole_valid]
                    # rgb_train = rgb_train[~whole_valid]
                    # if gt_normal_map is not None:
                    #     gt_normal_map = gt_normal_map[~whole_valid]

                    # loss
                    # print losses
                    # ic(pred_lambda, ori_lambda, loss,
                    #     params.distortion_lambda * distortion_loss,
                    #     params.ori_lambda * ori_loss,
                    #     params.envmap_lambda * (envmap_reg-0.05).clip(min=0),
                    #     params.diffuse_lambda * diffuse_reg,
                    #     params.brdf_lambda * brdf_reg,
                    #     params.pred_lambda * prediction_loss,
                    #     params.normal_err_lambda * norm_err)
                    total_loss = (
                        loss
                        + params.distortion_lambda * distortion_loss
                        + params.ori_lambda * ori_loss
                        + params.envmap_lambda * envmap_reg
                        + params.diffuse_lambda * diffuse_reg
                        + params.brdf_lambda * brdf_reg
                        + params.pred_lambda * prediction_loss
                        + params.normal_err_lambda * norm_err
                        + params.smoothness_lambda * smoothness_loss
                    )

                    # if tensorf.visibility_module is not None:
                    # pass
                    # if iteration % 1 == 0 and iteration > 250:
                    #     # if iteration < 100 or iteration % 1000 == 0:
                    #     if iteration % 250 == 0 and iteration < 2000:
                    #         tensorf.init_vis_module()
                    #         torch.cuda.empty_cache()
                    #     else:
                    #         tensorf.compute_visibility_loss(params.N_visibility_rays)

                    if params.ortho_weight > 0:
                        loss_reg = tensorf.rf.vector_comp_diffs()
                        total_loss += params.ortho_weight * loss_reg
                        summary_writer.add_scalar(
                            "train/reg", loss_reg.detach().item(), global_step=iteration
                        )
                    if params.L1_weight_initial > 0:
                        loss_reg_L1 = tensorf.rf.density_L1()
                        total_loss += params.L1_weight_initial * loss_reg_L1
                        summary_writer.add_scalar(
                            "train/reg_l1",
                            loss_reg_L1.detach().item(),
                            global_step=iteration,
                        )

                    loss_tv = 0
                    if TV_weight_density > 0:
                        TV_weight_density *= lr_factor
                        loss_tv = tensorf.rf.TV_loss_density(tvreg) * TV_weight_density
                        summary_writer.add_scalar(
                            "train/reg_tv_density",
                            loss_tv.detach().item(),
                            global_step=iteration,
                        )
                    if TV_weight_app > 0:
                        TV_weight_app *= lr_factor
                        loss_tv = (
                            loss_tv + tensorf.rf.TV_loss_app(tvreg) * TV_weight_app
                        )
                        summary_writer.add_scalar(
                            "train/reg_tv_app",
                            loss_tv.detach().item(),
                            global_step=iteration,
                        )
                    if params.TV_weight_bg > 0:
                        loss_tv = (
                            loss_tv + params.TV_weight_bg * tensorf.bg_module.tv_loss()
                        )
                    total_loss = total_loss + loss_tv

                    total_loss = total_loss / lbatch_size
                    if torch.isnan(total_loss).any():
                        continue
                    total_loss.backward()

                    photo_loss = photo_loss.detach().item()

                    TVs.append(float(loss_tv))
                    ori_losses.append(params.ori_lambda * ori_loss.detach().item())
                    pred_losses.append(params.pred_lambda * prediction_loss.detach().item())
                    losses.append(total_loss.detach().item())
                    # roughnesses.append(ims['roughness'].mean().detach().item())
                    diffuse_regs.append(params.diffuse_lambda * diffuse_reg.detach().item() / lbatch_size)
                    envmap_regs.append(params.envmap_lambda * envmap_reg.detach().item() / lbatch_size)
                    brdf_regs.append(params.brdf_lambda * brdf_reg.detach().item())
                    smoothness_losses.append(params.smoothness_lambda * smoothness_loss.detach().item())
                    PSNRs.append(-10.0 * np.log(photo_loss) / np.log(10.0))

                    # summary_writer.add_scalar('train/PSNR', PSNRs[-1], global_step=iteration)
                    # summary_writer.add_scalar('train/mse', photo_loss, global_step=iteration)
                    # summary_writer.add_scalar('train/ori_loss', ori_loss.detach().item(), global_step=iteration)
                    # summary_writer.add_scalar('train/distortion_loss', distortion_loss.detach().item(), global_step=iteration)
                    # summary_writer.add_scalar('train/prediction_loss', prediction_loss.detach().item(), global_step=iteration)
                    # summary_writer.add_scalar('train/diffuse_loss', diffuse_reg.detach().item(), global_step=iteration)
                    # summary_writer.add_scalar('train/lr', list(optimizer.param_groups)[0]['lr'], global_step=iteration)
                del ray_idx, rays_train, rgba_train, gt_normal_map, ims, stats

            if params.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(tensorf.parameters(), params.clip_grad)
            optimizer.step()
            scheduler.step()
            params.ori_lambda *= ori_decay
            params.pred_lambda *= normal_decay

            if iteration % args.vis_every == args.vis_every - 1 and args.N_vis != 0:
                # tensorf.save(f'{logfolder}/{expname}_{iteration}.th', args.model.arch)
                torch.cuda.empty_cache()
                test_res = evaluation(
                    test_dataset,
                    tensorf,
                    args,
                    renderer,
                    f"{logfolder}/imgs_vis/",
                    N_vis=args.N_vis,
                    prtx=f"{iteration:06d}_",
                    white_bg=white_bg,
                    ndc_ray=ndc_ray,
                    compute_extra_metrics=False,
                    gt_bg=gt_bg,
                )
                PSNRs_test = test_res["psnrs"]
                summary_writer.add_scalar(
                    "test/psnr", np.mean(test_res["psnrs"]), global_step=iteration
                )
                summary_writer.add_scalar(
                    "test/norm_err",
                    np.mean(test_res["norm_errs"]),
                    global_step=iteration,
                )
                logger.info(f"test_psnr = {float(np.mean(PSNRs_test)):.2f}")
                if args.save_often:
                    tensorf.save(
                        f"{logfolder}/{expname}_{iteration:06d}.th", args.model.arch
                    )

            # logger.info the current values of the losses.
            if iteration % args.progress_refresh_rate == 0:
                desc = (
                    f"psnr = {float(np.mean(PSNRs)):.2f}"
                    + f" test_psnr = {float(np.mean(PSNRs_test)):.2f}"
                    + f" loss = {float(np.sum(losses)):.5f}"
                    + f" envmap = {float(np.sum(envmap_regs)):.5f}"
                    + f" diffuse = {float(np.sum(diffuse_regs)):.5f}"
                    + f" brdf = {float(np.sum(brdf_regs)):.5f}"
                    + f" smooth = {float(np.sum(smoothness_losses)):.5f}"
                    + f" nrays = {[num_rays] + tensorf.model.max_retrace_rays}"
                )
                # f' rough = {float(np.mean(roughnesses)):.5f}' + \
                # f' diffuse = {float(np.sum(diffuse_regs)):.5f}' + \
                # f' tv = {float(np.mean(TVs)):.5f}' + \
                # f' ori loss = {float(np.mean(ori_losses) / num_rays):.5f}' + \
                # f' pred loss = {float(np.mean(pred_losses) / num_rays):.5f}' + \
                # + f' mse = {photo_loss:.6f}'
                if tensorf.bg_module is not None:
                    desc = desc + f" mipbias = {float(tensorf.bg_module.mipbias):.1e}"
                    # f' mul = {float(tensorf.bg_module.mul):.1e}' + \
                    # f' bright = {float(tensorf.bg_module.brightness):.1e}'
                pbar.set_description(desc)
                PSNRs = []

            if tensorf.check_schedule(iteration, 1):
                grad_vars = tensorf.get_optparam_groups()
                print("reinit optimizer")
                optimizer, scheduler = init_optimizer(tensorf, grad_vars, params)
                num_rays = params.starting_batch_size
                prev_n_samples = None
                hist_n_samples = None
                tensorf.model.reset_counter()
                # new_grad_vars = tensorf.get_optparam_groups()
                # for param_group, new_param_group in zip(optimizer.param_groups, new_grad_vars):
                #     param_group['params'] = new_param_group['params']

            # if iteration in update_alphamask_list:

            #  if reso_cur[0] * reso_cur[1] * reso_cur[2]<256**3:# update volume resolution
            # tensorVM.alphaMask = None
            # L1_reg_weight = params.L1_weight_rest
            # logger.info("continuing L1_reg_weight", L1_reg_weight)

            # if not ndc_ray and iteration == update_AlphaMask_list[-1] and args.filter_rays:
            #     # filter rays outside the bbox
            #     allrays, allrgbs, mask = tensorf.filtering_rays(allrays, allrgbs, focal)
            #     trainingSampler = SimpleSampler(allrays.shape[0], params.batch_size)

    #         p.step()
    # p.export_chrome_trace('p.trace')

    # prof.export_chrome_trace('trace.json')

    """ Save checkpoint """
    tensorf.save(f"{logfolder}/{expname}.th", args.model.arch)

    """ Render training views """
    torch.cuda.empty_cache()
    if args.render_train:
        os.makedirs(f"{logfolder}/imgs_train_all", exist_ok=True)
        train_dataset = dataset(
            args.datadir, split="train", downsample=args.downsample_train, is_stack=True
        )
        test_res = evaluation(
            train_dataset,
            tensorf,
            args,
            renderer,
            f"{logfolder}/imgs_train_all/",
            N_vis=-1,
            N_samples=-1,
            white_bg=white_bg,
            ndc_ray=ndc_ray,
            device=device,
            gt_bg=gt_bg,
        )
        logger.info(
            f'======> {expname} test all psnr: {np.mean(test_res["psnrs"])} <========================'
        )

    """ Render test views """
    torch.cuda.empty_cache()
    if args.render_test:
        os.makedirs(f"{logfolder}/imgs_test_all", exist_ok=True)
        test_res = evaluation(
            test_dataset,
            tensorf,
            args,
            renderer,
            f"{logfolder}/imgs_test_all/",
            N_vis=-1,
            N_samples=-1,
            white_bg=white_bg,
            ndc_ray=ndc_ray,
            device=device,
            gt_bg=gt_bg,
        )
        summary_writer.add_scalar(
            "test/psnr_all", np.mean(test_res["psnrs"]), global_step=iteration
        )
        logger.info(
            f'======> {expname} test all psnr: {np.mean(test_res["psnrs"])} <========================'
        )

    """ Render test path """
    torch.cuda.empty_cache()
    if args.render_path:
        c2ws = test_dataset.render_path
        # c2ws = test_dataset.poses
        logger.info("========>", c2ws.shape)
        os.makedirs(f"{logfolder}/imgs_path_all", exist_ok=True)
        evaluation_path(
            test_dataset,
            tensorf,
            c2ws,
            renderer,
            args,
            savePath=f"{logfolder}/imgs_path_all/",
            device=device,
            ndc_ray=ndc_ray,
            N_samples=-1,
            white_bg=white_bg,
            gt_bg=gt_bg,
        )


""" Train model """
@hydra.main(version_base=None, config_path="configs", config_name="default")
def train(cfg: DictConfig):
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    logger.info(cfg.dataset)
    # logger.info(cfg.model)
    cfg.model.arch.rf = cfg.field

    if cfg.render_only:
        render_test(cfg)
    else:
        reconstruction(cfg)
        # reconstruction(args)


if __name__ == "__main__":
    train()
