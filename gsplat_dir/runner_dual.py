import json
import math
import os
import time

import torch
import torch.nn.functional as F
import tqdm
from fused_ssim import fused_ssim
from torch.utils.data import DataLoader, Subset
from typing_extensions import assert_never

from gsplat_dir.runner import Runner, _seed_worker
from submodules.gsplat.gsplat.strategy import DefaultStrategy, MCMCStrategy

try:
    from fused_bilagrid import slice, total_variation_loss
except Exception:
    try:
        from lib_bilagrid import slice, total_variation_loss
    except Exception:
        slice = None
        total_variation_loss = None


class RunnerDual(Runner):
    """Runner with separate loaders for real images and nerf samples."""

    def _build_dual_subsets(self):
        real_items = []
        nerf_items = []
        for ds_item, parser_idx in enumerate(self.trainset.indices):
            name = os.path.basename(self.parser.image_names[parser_idx])
            if name.startswith("nerf_sample_"):
                nerf_items.append(ds_item)
            else:
                real_items.append(ds_item)
        return real_items, nerf_items

    @staticmethod
    def _next_batch(loader, it):
        if loader is None:
            return None, None
        if it is None:
            it = iter(loader)
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        return batch, it

    def train(self):
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank
        world_size = self.world_size
        if cfg.use_bilateral_grid and (slice is None or total_variation_loss is None):
            raise ImportError("Bilateral grid enabled but bilagrid ops are not available.")

        if world_rank == 0:
            with open(f"{cfg.result_dir}/cfg.yml", "w") as f:
                import yaml

                yaml.dump(vars(cfg), f)

        max_steps = cfg.max_steps
        init_step = 0

        schedulers = [
            torch.optim.lr_scheduler.ExponentialLR(
                self.optimizers["means"], gamma=0.01 ** (1.0 / max_steps)
            ),
        ]
        if cfg.pose_opt:
            schedulers.append(
                torch.optim.lr_scheduler.ExponentialLR(
                    self.pose_optimizers[0], gamma=0.01 ** (1.0 / max_steps)
                )
            )
        if cfg.use_bilateral_grid:
            schedulers.append(
                torch.optim.lr_scheduler.ChainedScheduler(
                    [
                        torch.optim.lr_scheduler.LinearLR(
                            self.bil_grid_optimizers[0], start_factor=0.01, total_iters=1000
                        ),
                        torch.optim.lr_scheduler.ExponentialLR(
                            self.bil_grid_optimizers[0], gamma=0.01 ** (1.0 / max_steps)
                        ),
                    ]
                )
            )

        real_items, nerf_items = self._build_dual_subsets()
        decay_steps = max(1, int(getattr(cfg, "dual_nerf_decay_steps_to_quarter", 1000)))
        nerf_decay_k = math.log(4.0) / float(decay_steps)
        nerf_disable_threshold = float(getattr(cfg, "dual_nerf_disable_threshold", 0.1))
        nerf_weight_start = float(getattr(cfg, "dual_nerf_loss_weight", 2.0))
        nerf_branch_enabled = len(nerf_items) > 0
        nerf_branch_disabled_printed = False
        use_nerf_depth = bool(getattr(cfg, "use_nerf_depth_supervision", False))
        nerf_depth_lambda = float(getattr(cfg, "nerf_depth_lambda", 0.0))
        nerf_depth_max_steps = int(getattr(cfg, "nerf_depth_max_steps", -1))
        nerf_depth_log_space = bool(getattr(cfg, "nerf_depth_log_space", True))
        depth_warmup_steps = int(getattr(cfg, "depth_warmup_steps", 0))
        depth_warmup_disable_nerf_branch = bool(
            getattr(cfg, "depth_warmup_disable_nerf_branch", True)
        )
        depth_warmup_force_depth = bool(
            getattr(cfg, "depth_warmup_force_depth_supervision", True)
        )
        if world_rank == 0:
            print(
                f"[DualLoader] real train images={len(real_items)}, nerf samples={len(nerf_items)}, "
                f"real_bs={cfg.batch_size}, nerf_bs={max(1, cfg.batch_size * cfg.nerf_batch_factor)}"
            )
            print(
                f"[DualLoss] real_weight={cfg.dual_real_loss_weight}, "
                f"nerf_weight_schedule: w0={nerf_weight_start}, quarter_at={decay_steps}, "
                f"disable_below={nerf_disable_threshold}"
            )
            if use_nerf_depth:
                print(
                    f"[DualDepth] enabled: lambda={nerf_depth_lambda}, "
                    f"log_space={nerf_depth_log_space}, max_steps={nerf_depth_max_steps}, "
                    f"include_real={bool(getattr(cfg, 'nerf_depth_include_real', True))}, "
                    f"include_nerf={bool(getattr(cfg, 'nerf_depth_include_nerf_samples', True))}"
                )
            if depth_warmup_steps > 0:
                print(
                    f"[DualDepthWarmup] steps={depth_warmup_steps}, "
                    f"disable_nerf_branch={depth_warmup_disable_nerf_branch}, "
                    f"force_depth_supervision={depth_warmup_force_depth}"
                )

        real_loader = None
        nerf_loader = None
        if len(real_items) > 0:
            real_gen = torch.Generator().manual_seed(self._loader_seed_base + 1)
            real_loader = DataLoader(
                Subset(self.trainset, real_items),
                batch_size=cfg.batch_size,
                shuffle=True,
                num_workers=4,
                persistent_workers=True,
                pin_memory=True,
                worker_init_fn=_seed_worker,
                generator=real_gen,
            )
        if len(nerf_items) > 0:
            nerf_gen = torch.Generator().manual_seed(self._loader_seed_base + 2)
            nerf_loader = DataLoader(
                Subset(self.trainset, nerf_items),
                batch_size=max(1, cfg.batch_size * cfg.nerf_batch_factor),
                shuffle=True,
                num_workers=4,
                persistent_workers=True,
                pin_memory=True,
                worker_init_fn=_seed_worker,
                generator=nerf_gen,
            )

        real_iter = None
        nerf_iter = None
        real_use_l1 = bool(getattr(cfg, "use_l1_for_real_samples", False))
        nerf_use_l1 = bool(getattr(cfg, "use_l1_for_nerf_samples", False))

        def _depth_supervision_active(cur_step: int) -> bool:
            if not use_nerf_depth:
                return False
            if nerf_depth_lambda <= 0.0:
                return False
            if nerf_depth_max_steps < 0:
                return True
            return cur_step < nerf_depth_max_steps

        def _compute_nerf_depth_loss(pred_depth_hw, data_batch):
            if pred_depth_hw is None or data_batch is None:
                return torch.tensor(0.0, device=device)
            if "nerf_depth" not in data_batch or "has_nerf_depth" not in data_batch:
                return torch.tensor(0.0, device=device)
            tgt = data_batch["nerf_depth"].to(device).float()
            has = data_batch["has_nerf_depth"].to(device).bool()
            if pred_depth_hw.ndim == 4 and pred_depth_hw.shape[-1] == 1:
                pred = pred_depth_hw[..., 0]
            elif pred_depth_hw.ndim == 3:
                pred = pred_depth_hw
            else:
                return torch.tensor(0.0, device=device)
            if pred.shape != tgt.shape:
                return torch.tensor(0.0, device=device)
            eps = 1e-6
            valid = has[:, None, None] & torch.isfinite(tgt) & torch.isfinite(pred) & (tgt > 0.0) & (pred > 0.0)
            if not bool(valid.any().item()):
                return torch.tensor(0.0, device=device)
            if nerf_depth_log_space:
                pred_v = torch.log(torch.clamp(pred[valid], min=eps))
                tgt_v = torch.log(torch.clamp(tgt[valid], min=eps))
            else:
                pred_v = pred[valid]
                tgt_v = tgt[valid]
            return F.l1_loss(pred_v, tgt_v)

        global_tic = time.time()
        pbar = tqdm.tqdm(range(init_step, max_steps))
        best_val_combo = float("inf")
        best_val_step = -1
        for step in pbar:
            if not cfg.disable_viewer:
                while self.viewer.state == "paused":
                    time.sleep(0.01)
                self.viewer.lock.acquire()
                tic = time.time()

            real_data, real_iter = self._next_batch(real_loader, real_iter)
            in_depth_warmup = depth_warmup_steps > 0 and step < depth_warmup_steps
            if world_rank == 0 and step == depth_warmup_steps and depth_warmup_steps > 0:
                print(f"[DualDepthWarmup] finished at step={step}; switching to standard dual training.")
            depth_supervision_active = _depth_supervision_active(step) or (
                in_depth_warmup
                and depth_warmup_force_depth
                and use_nerf_depth
                and nerf_depth_lambda > 0.0
            )
            nerf_weight_sched = nerf_weight_start * math.exp(-nerf_decay_k * float(step))
            if nerf_branch_enabled and nerf_weight_sched < nerf_disable_threshold:
                nerf_branch_enabled = False
                if world_rank == 0 and not nerf_branch_disabled_printed:
                    print(
                        f"[DualLoss] disabling nerf branch at step={step} "
                        f"(scheduled nerf weight={nerf_weight_sched:.6f} < {nerf_disable_threshold})"
                    )
                    nerf_branch_disabled_printed = True
            nerf_branch_enabled_this_step = nerf_branch_enabled
            if in_depth_warmup and depth_warmup_disable_nerf_branch:
                nerf_branch_enabled_this_step = False
            nerf_data, nerf_iter = (
                self._next_batch(nerf_loader, nerf_iter) if nerf_branch_enabled_this_step else (None, nerf_iter)
            )

            if real_data is None and nerf_data is None:
                raise RuntimeError("Both real and nerf loaders are empty.")

            sh_degree_to_use = min(step // cfg.sh_degree_interval, cfg.sh_degree)
            num_train_rays_per_step = 0

            real_loss = torch.tensor(0.0, device=device)
            real_recon_loss = torch.tensor(0.0, device=device)
            ssimloss = torch.tensor(0.0, device=device)
            depthloss = None
            depth_supervision_loss = torch.tensor(0.0, device=device)
            real_info = None
            real_Ks = None
            real_colors = None
            real_pixels = None
            pose_err = None
            real_depth_pred = None

            nerf_loss = torch.tensor(0.0, device=device)
            nerf_reconloss = torch.tensor(0.0, device=device)
            nerf_info = None
            nerf_Ks = None
            nerf_depth_pred = None

            if real_data is not None:
                camtoworlds = camtoworlds_gt = real_data["camtoworld"].to(device)
                Ks = real_data["K"].to(device)
                pixels = real_data["image"].to(device) / 255.0
                image_ids = real_data["image_id"].to(device)
                masks = real_data["mask"].to(device) if "mask" in real_data else None
                is_nerf_sample = real_data["is_nerf_sample"].to(device)
                if bool(is_nerf_sample.any().item()):
                    raise RuntimeError("Real loader produced nerf samples. Split mismatch.")
                if cfg.depth_loss:
                    points = real_data["points"].to(device)
                    depths_gt = real_data["depths"].to(device)

                num_train_rays_per_step += pixels.shape[0] * pixels.shape[1] * pixels.shape[2]
                height, width = pixels.shape[1:3]

                if cfg.pose_noise:
                    camtoworlds = self.pose_perturb(camtoworlds, image_ids)
                if cfg.pose_opt:
                    camtoworlds = self.pose_adjust(camtoworlds, image_ids)

                render_mode_real = "RGB+ED" if (cfg.depth_loss or depth_supervision_active) else "RGB"
                renders, alphas, info = self.rasterize_splats(
                    camtoworlds=camtoworlds,
                    Ks=Ks,
                    width=width,
                    height=height,
                    sh_degree=sh_degree_to_use,
                    near_plane=cfg.near_plane,
                    far_plane=cfg.far_plane,
                    image_ids=image_ids,
                    render_mode=render_mode_real,
                    masks=masks,
                )
                colors, depths = (renders[..., 0:3], renders[..., 3:4]) if renders.shape[-1] == 4 else (renders, None)
                real_depth_pred = depths

                if cfg.use_bilateral_grid:
                    grid_y, grid_x = torch.meshgrid(
                        (torch.arange(height, device=self.device) + 0.5) / height,
                        (torch.arange(width, device=self.device) + 0.5) / width,
                        indexing="ij",
                    )
                    grid_xy = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
                    colors = slice(
                        self.bil_grids,
                        grid_xy.expand(colors.shape[0], -1, -1, -1),
                        colors,
                        image_ids.unsqueeze(-1),
                    )["rgb"]

                if cfg.random_bkgd:
                    bkgd = torch.rand(1, 3, device=device)
                    colors = colors + bkgd * (1.0 - alphas)

                real_recon_loss = F.l1_loss(colors, pixels) if real_use_l1 else F.mse_loss(colors, pixels)
                ssimloss = 1.0 - fused_ssim(
                    colors.permute(0, 3, 1, 2), pixels.permute(0, 3, 1, 2), padding="valid"
                )
                real_loss = real_recon_loss * (1.0 - cfg.ssim_lambda) + ssimloss * cfg.ssim_lambda

                if cfg.depth_loss:
                    points_n = torch.stack(
                        [
                            points[:, :, 0] / (width - 1) * 2 - 1,
                            points[:, :, 1] / (height - 1) * 2 - 1,
                        ],
                        dim=-1,
                    )
                    grid = points_n.unsqueeze(2)
                    depths_s = F.grid_sample(depths.permute(0, 3, 1, 2), grid, align_corners=True)
                    depths_s = depths_s.squeeze(3).squeeze(1)
                    disp = torch.where(depths_s > 0.0, 1.0 / depths_s, torch.zeros_like(depths_s))
                    disp_gt = 1.0 / depths_gt
                    depthloss = F.l1_loss(disp, disp_gt) * self.scene_scale
                    real_loss = real_loss + depthloss * cfg.depth_lambda

                if cfg.pose_opt and cfg.pose_noise:
                    pose_err = F.l1_loss(camtoworlds_gt, camtoworlds)

                real_info = info
                real_Ks = Ks
                real_colors = colors
                real_pixels = pixels

            if nerf_data is not None:
                camtoworlds = nerf_data["camtoworld"].to(device)
                Ks = nerf_data["K"].to(device)
                pixels = nerf_data["image"].to(device) / 255.0
                image_ids = nerf_data["image_id"].to(device)
                masks = nerf_data["mask"].to(device) if "mask" in nerf_data else None
                is_nerf_sample = nerf_data["is_nerf_sample"].to(device)
                if not bool(is_nerf_sample.all().item()):
                    raise RuntimeError("NeRF loader produced real images. Split mismatch.")
                weight_maps = nerf_data["weight_map"].to(device)

                num_train_rays_per_step += pixels.shape[0] * pixels.shape[1] * pixels.shape[2]
                height, width = pixels.shape[1:3]

                if cfg.pose_noise:
                    camtoworlds = self.pose_perturb(camtoworlds, image_ids)
                if cfg.pose_opt:
                    camtoworlds = self.pose_adjust(camtoworlds, image_ids)

                renders, alphas, info = self.rasterize_splats(
                    camtoworlds=camtoworlds,
                    Ks=Ks,
                    width=width,
                    height=height,
                    sh_degree=sh_degree_to_use,
                    near_plane=cfg.near_plane,
                    far_plane=cfg.far_plane,
                    image_ids=image_ids,
                    render_mode="RGB+ED" if depth_supervision_active else "RGB",
                    masks=masks,
                )
                colors, nerf_depth_pred = (
                    (renders[..., 0:3], renders[..., 3:4])
                    if renders.shape[-1] == 4
                    else (renders, None)
                )

                if cfg.use_bilateral_grid:
                    grid_y, grid_x = torch.meshgrid(
                        (torch.arange(height, device=self.device) + 0.5) / height,
                        (torch.arange(width, device=self.device) + 0.5) / width,
                        indexing="ij",
                    )
                    grid_xy = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
                    colors = slice(
                        self.bil_grids,
                        grid_xy.expand(colors.shape[0], -1, -1, -1),
                        colors,
                        image_ids.unsqueeze(-1),
                    )["rgb"]

                if cfg.random_bkgd:
                    bkgd = torch.rand(1, 3, device=device)
                    colors = colors + bkgd * (1.0 - alphas)

                if nerf_use_l1:
                    recon_map = torch.abs(colors - pixels).mean(dim=-1)
                else:
                    recon_map = ((colors - pixels) ** 2).mean(dim=-1)
                w = torch.clamp(weight_maps, min=0.0)
                denom = torch.clamp(w.sum(dim=(1, 2)), min=1e-8)
                per_img = (recon_map * w).sum(dim=(1, 2)) / denom
                nerf_reconloss = per_img.mean()
                nerf_loss = nerf_reconloss

                nerf_info = info
                nerf_Ks = Ks

            info_main = real_info if real_info is not None else nerf_info
            Ks_main = real_Ks if real_Ks is not None else nerf_Ks

            self.cfg.strategy.step_pre_backward(
                params=self.splats,
                optimizers=self.optimizers,
                state=self.strategy_state,
                step=step,
                info=info_main,
            )

            wr = float(cfg.dual_real_loss_weight) if real_data is not None else 0.0
            wn = float(nerf_weight_sched) if nerf_data is not None else 0.0
            wsum = wr + wn
            if wsum <= 0:
                raise RuntimeError("dual_real_loss_weight + dual_nerf_loss_weight must be > 0.")
            loss = (wr * real_loss + wn * nerf_loss) / wsum
            if depth_supervision_active:
                depth_terms = []
                if bool(getattr(cfg, "nerf_depth_include_real", True)):
                    depth_terms.append(_compute_nerf_depth_loss(real_depth_pred, real_data))
                if bool(getattr(cfg, "nerf_depth_include_nerf_samples", True)):
                    depth_terms.append(_compute_nerf_depth_loss(nerf_depth_pred, nerf_data))
                if len(depth_terms) > 0:
                    depth_supervision_loss = torch.stack(depth_terms).mean()
                    loss = loss + nerf_depth_lambda * depth_supervision_loss

            if cfg.use_bilateral_grid:
                tvloss = 10 * total_variation_loss(self.bil_grids.grids)
                loss += tvloss

            if cfg.opacity_reg > 0.0:
                loss += cfg.opacity_reg * torch.sigmoid(self.splats["opacities"]).mean()
            if cfg.scale_reg > 0.0:
                loss += cfg.scale_reg * torch.exp(self.splats["scales"]).mean()

            loss.backward()

            desc = f"loss={loss.item():.3f}| real={real_loss.item():.3f}| nerf={nerf_loss.item():.3f}| sh degree={sh_degree_to_use}| "
            if depthloss is not None:
                desc += f"depth loss={depthloss.item():.6f}| "
            if depth_supervision_active:
                desc += f"nerf-depth={depth_supervision_loss.item():.6f}| "
            if pose_err is not None:
                desc += f"pose err={pose_err.item():.6f}| "
            pbar.set_description(desc)

            if world_rank == 0 and cfg.tb_every > 0 and step % cfg.tb_every == 0:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                self.writer.add_scalar("train/loss", loss.item(), step)
                self.writer.add_scalar("train/real_loss", real_loss.item(), step)
                self.writer.add_scalar("train/nerf_loss", nerf_loss.item(), step)
                self.writer.add_scalar("train/nerf_weight_schedule", float(nerf_weight_sched), step)
                self.writer.add_scalar("train/reconloss", real_recon_loss.item(), step)
                if real_use_l1:
                    self.writer.add_scalar("train/l1loss", real_recon_loss.item(), step)
                else:
                    self.writer.add_scalar("train/l2loss", real_recon_loss.item(), step)
                self.writer.add_scalar("train/ssimloss", ssimloss.item(), step)
                self.writer.add_scalar("train/nerf_weighted_reconloss", nerf_reconloss.item(), step)
                if nerf_use_l1:
                    self.writer.add_scalar("train/nerf_weighted_l1loss", nerf_reconloss.item(), step)
                else:
                    self.writer.add_scalar("train/nerf_weighted_l2loss", nerf_reconloss.item(), step)
                self.writer.add_scalar(
                    "train/nerf_sample_fraction",
                    float(1.0 if real_data is None else 0.0 if nerf_data is None else 0.5),
                    step,
                )
                self.writer.add_scalar("train/num_GS", len(self.splats["means"]), step)
                self.writer.add_scalar("train/mem", mem, step)
                if depthloss is not None:
                    self.writer.add_scalar("train/depthloss", depthloss.item(), step)
                if cfg.use_bilateral_grid:
                    self.writer.add_scalar("train/tvloss", tvloss.item(), step)
                if depth_supervision_active:
                    self.writer.add_scalar(
                        "train/nerf_depth_supervision_loss",
                        depth_supervision_loss.item(),
                        step,
                    )
                if cfg.tb_save_image and real_colors is not None and real_pixels is not None:
                    canvas = torch.cat([real_pixels, real_colors], dim=2).detach().cpu().numpy()
                    canvas = canvas.reshape(-1, *canvas.shape[2:])
                    self.writer.add_image("train/render", canvas, step)
                self.writer.flush()

            if step in [i - 1 for i in cfg.save_steps]:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                stats = {"mem": mem, "ellipse_time": time.time() - global_tic, "num_GS": len(self.splats["means"])}
                print("Step: ", step, stats)
                with open(f"{self.stats_dir}/train_step{step:04d}_rank{self.world_rank}.json", "w") as f:
                    json.dump(stats, f)
                data = {"step": step, "splats": self.splats.state_dict()}
                if cfg.pose_opt:
                    data["pose_adjust"] = self.pose_adjust.module.state_dict() if world_size > 1 else self.pose_adjust.state_dict()
                if cfg.app_opt:
                    data["app_module"] = self.app_module.module.state_dict() if world_size > 1 else self.app_module.state_dict()
                torch.save(data, f"{self.ckpt_dir}/ckpt_{step}_rank{self.world_rank}.pt")

            if cfg.sparse_grad:
                assert cfg.packed, "Sparse gradients only work with packed mode."
                gaussian_ids = info_main["gaussian_ids"]
                for k in self.splats.keys():
                    grad = self.splats[k].grad
                    if grad is None or grad.is_sparse:
                        continue
                    self.splats[k].grad = torch.sparse_coo_tensor(
                        indices=gaussian_ids[None],
                        values=grad[gaussian_ids],
                        size=self.splats[k].size(),
                        is_coalesced=len(Ks_main) == 1,
                    )

            if cfg.visible_adam:
                if cfg.packed:
                    visibility_mask = torch.zeros_like(self.splats["opacities"], dtype=bool)
                    visibility_mask.scatter_(0, info_main["gaussian_ids"], 1)
                else:
                    visibility_mask = (info_main["radii"] > 0).all(-1).any(0)

            for optimizer in self.optimizers.values():
                if cfg.visible_adam:
                    optimizer.step(visibility_mask)
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.pose_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.app_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.bil_grid_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for scheduler in schedulers:
                scheduler.step()

            if isinstance(self.cfg.strategy, DefaultStrategy):
                self.cfg.strategy.step_post_backward(
                    params=self.splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=info_main,
                    packed=cfg.packed,
                )
            elif isinstance(self.cfg.strategy, MCMCStrategy):
                self.cfg.strategy.step_post_backward(
                    params=self.splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=info_main,
                    lr=schedulers[0].get_last_lr()[0],
                )
            else:
                assert_never(self.cfg.strategy)

            if step % 100 == 0:
                eval_stats = self.eval_allstats()
                if world_rank == 0:
                    # Combined lower-is-better validation score:
                    # (10^(-PSNR/10) * sqrt(1-SSIM) * LPIPS)^(1/3)
                    if all(k in eval_stats for k in ("psnr", "ssim", "lpips")):
                        val_combo = (
                            (10.0 ** (-float(eval_stats["psnr"]) / 10.0))
                            * math.sqrt(max(0.0, 1.0 - float(eval_stats["ssim"])))
                            * float(eval_stats["lpips"])
                        ) ** (1.0 / 3.0)
                        self.writer.add_scalar("val/combined_score", val_combo, step)
                        self.writer.flush()
                        if val_combo < best_val_combo:
                            best_val_combo = val_combo
                            best_val_step = step
                            best_ckpt = {
                                "step": step,
                                "best_val_combo": best_val_combo,
                                "best_val_step": best_val_step,
                                "splats": self.splats.state_dict(),
                            }
                            if cfg.pose_opt:
                                best_ckpt["pose_adjust"] = (
                                    self.pose_adjust.module.state_dict()
                                    if world_size > 1
                                    else self.pose_adjust.state_dict()
                                )
                            if cfg.app_opt:
                                best_ckpt["app_module"] = (
                                    self.app_module.module.state_dict()
                                    if world_size > 1
                                    else self.app_module.state_dict()
                                )
                            best_path = f"{self.ckpt_dir}/best_ckpt_rank{self.world_rank}.pt"
                            torch.save(best_ckpt, best_path)
                            print(
                                f"[best-ckpt] saved {best_path} at step={step} "
                                f"with combined_score={best_val_combo:.6f}"
                            )
                    with open(os.path.join(cfg.result_dir, "gsplat_stats.json"), "w", encoding="utf-8") as f:
                        json.dump(self.stats_arr, f, indent=2, ensure_ascii=False)
                    training_time = time.time() - global_tic
                    out_path = os.path.join(cfg.result_dir, "time_logs.txt")
                    with open(out_path, "a") as f:
                        f.write(f"[gaussian-splatting-{step}] - {training_time}\n")

            if cfg.compression is not None and step in [i - 1 for i in cfg.eval_steps]:
                self.run_compression(step=step)

            if not cfg.disable_viewer:
                self.viewer.lock.release()
                num_train_steps_per_sec = 1.0 / (max(time.time() - tic, 1e-10))
                num_train_rays_per_sec = num_train_rays_per_step * num_train_steps_per_sec
                self.viewer.render_tab_state.num_train_rays_per_sec = num_train_rays_per_sec
                self.viewer.update(step, num_train_rays_per_step)
