import json
import math
import os
import time
from typing import Optional

import torch
import torch.nn.functional as F
import tqdm
from fused_ssim import fused_ssim
from torch.utils.data import DataLoader, Subset
from typing_extensions import assert_never

from gsplat_dir.runner import Runner
from submodules.gsplat.gsplat.strategy import DefaultStrategy, MCMCStrategy

try:
    from fused_bilagrid import slice, total_variation_loss
except Exception:
    try:
        from lib_bilagrid import slice, total_variation_loss
    except Exception:
        slice = None
        total_variation_loss = None


class RunnerStaged(Runner):
    """Two-phase training: nerf-only warmup, then real-only gsplat."""

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

    @staticmethod
    def _reset_optimizer_state(optim):
        optim.state.clear()
        optim.zero_grad(set_to_none=True)

    def _build_schedulers(self, phase_steps: int):
        cfg = self.cfg
        steps = max(1, int(phase_steps))
        schedulers = [
            torch.optim.lr_scheduler.ExponentialLR(
                self.optimizers["means"], gamma=0.01 ** (1.0 / steps)
            ),
        ]
        if cfg.pose_opt:
            schedulers.append(
                torch.optim.lr_scheduler.ExponentialLR(
                    self.pose_optimizers[0], gamma=0.01 ** (1.0 / steps)
                )
            )
        if cfg.use_bilateral_grid:
            schedulers.append(
                torch.optim.lr_scheduler.ChainedScheduler(
                    [
                        torch.optim.lr_scheduler.LinearLR(
                            self.bil_grid_optimizers[0], start_factor=0.01, total_iters=min(1000, steps)
                        ),
                        torch.optim.lr_scheduler.ExponentialLR(
                            self.bil_grid_optimizers[0], gamma=0.01 ** (1.0 / steps)
                        ),
                    ]
                )
            )
        return schedulers

    def _reset_phase_optimizers(self, real_phase_steps: int):
        # Reset optimizer state (momentum, etc.) at phase switch.
        for optimizer in self.optimizers.values():
            self._reset_optimizer_state(optimizer)
        for optimizer in self.pose_optimizers:
            self._reset_optimizer_state(optimizer)
        for optimizer in self.app_optimizers:
            self._reset_optimizer_state(optimizer)
        for optimizer in self.bil_grid_optimizers:
            self._reset_optimizer_state(optimizer)
        return self._build_schedulers(real_phase_steps)

    def _reset_opacity_to_value(self, value: float):
        value = float(max(1e-6, min(1.0 - 1e-6, value)))
        logit = math.log(value / (1.0 - value))
        with torch.no_grad():
            self.splats["opacities"].fill_(logit)

    def train(self):
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank
        world_size = self.world_size
        if cfg.use_bilateral_grid and (slice is None or total_variation_loss is None):
            raise ImportError("Bilateral grid enabled but bilagrid ops are not available.")

        nerf_phase_steps = int(getattr(cfg, "staged_nerf_phase_steps", 0))
        real_phase_steps = int(getattr(cfg, "staged_real_phase_steps", cfg.max_steps))
        max_steps = max(0, nerf_phase_steps) + max(0, real_phase_steps)
        if max_steps <= 0:
            raise RuntimeError("staged_nerf_phase_steps + staged_real_phase_steps must be > 0")

        if world_rank == 0:
            import yaml

            with open(f"{cfg.result_dir}/cfg.yml", "w") as f:
                yaml.dump(vars(cfg), f)
            print(
                f"[Staged] nerf_phase_steps={nerf_phase_steps}, real_phase_steps={real_phase_steps}, "
                f"total_steps={max_steps}"
            )

        real_items, nerf_items = self._build_dual_subsets()
        if world_rank == 0:
            print(
                f"[Staged] real train images={len(real_items)}, nerf samples={len(nerf_items)}, "
                f"real_bs={cfg.batch_size}, nerf_bs={max(1, cfg.batch_size * cfg.nerf_batch_factor)}"
            )

        real_loader = None
        nerf_loader = None
        if len(real_items) > 0:
            real_loader = DataLoader(
                Subset(self.trainset, real_items),
                batch_size=cfg.batch_size,
                shuffle=True,
                num_workers=4,
                persistent_workers=True,
                pin_memory=True,
            )
        if len(nerf_items) > 0:
            nerf_loader = DataLoader(
                Subset(self.trainset, nerf_items),
                batch_size=max(1, cfg.batch_size * cfg.nerf_batch_factor),
                shuffle=True,
                num_workers=4,
                persistent_workers=True,
                pin_memory=True,
            )

        real_iter = None
        nerf_iter = None
        real_use_l1 = bool(getattr(cfg, "use_l1_for_real_samples", False))
        schedulers = self._build_schedulers(max_steps)
        switched_to_real = False

        global_tic = time.time()
        pbar = tqdm.tqdm(range(max_steps))
        for step in pbar:
            if not cfg.disable_viewer:
                while self.viewer.state == "paused":
                    time.sleep(0.01)
                self.viewer.lock.acquire()
                tic = time.time()

            in_nerf_phase = (step < nerf_phase_steps)
            real_phase_step = max(0, step - nerf_phase_steps)
            strategy_active = not in_nerf_phase
            if in_nerf_phase and not switched_to_real and world_rank == 0 and step == 0:
                print("[Staged] phase=nerf-only")
            if (not in_nerf_phase) and not switched_to_real:
                switched_to_real = True
                if world_rank == 0:
                    print(f"[Staged] switching to phase=real-only at step={step}")
                if bool(getattr(cfg, "staged_reset_opacity_after_nerf", True)):
                    self._reset_opacity_to_value(float(getattr(cfg, "staged_reset_opacity_value", 0.1)))
                    if world_rank == 0:
                        print(
                            f"[Staged] reset opacities to {float(getattr(cfg, 'staged_reset_opacity_value', 0.1))}"
                        )
                schedulers = self._reset_phase_optimizers(real_phase_steps)

            data, real_iter_or_nerf_iter = (None, None)
            if in_nerf_phase:
                data, nerf_iter = self._next_batch(nerf_loader, nerf_iter)
                if data is None:
                    raise RuntimeError("Nerf phase requested but nerf loader is empty.")
            else:
                data, real_iter = self._next_batch(real_loader, real_iter)
                if data is None:
                    raise RuntimeError("Real phase requested but real loader is empty.")

            camtoworlds = camtoworlds_gt = data["camtoworld"].to(device)
            Ks = data["K"].to(device)
            pixels = data["image"].to(device) / 255.0
            image_ids = data["image_id"].to(device)
            masks = data["mask"].to(device) if "mask" in data else None
            is_nerf_sample = data["is_nerf_sample"].to(device)
            weight_maps = data["weight_map"].to(device) if "weight_map" in data else None
            num_train_rays_per_step = pixels.shape[0] * pixels.shape[1] * pixels.shape[2]
            if cfg.depth_loss and (not in_nerf_phase):
                points = data["points"].to(device)
                depths_gt = data["depths"].to(device)

            height, width = pixels.shape[1:3]

            if cfg.pose_noise:
                camtoworlds = self.pose_perturb(camtoworlds, image_ids)
            if cfg.pose_opt:
                camtoworlds = self.pose_adjust(camtoworlds, image_ids)

            # Keep SH/strategy schedule tied to real-image phase progression.
            sh_degree_to_use = min(real_phase_step // cfg.sh_degree_interval, cfg.sh_degree)
            renders, alphas, info = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=sh_degree_to_use,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                image_ids=image_ids,
                render_mode="RGB+ED" if (cfg.depth_loss and (not in_nerf_phase)) else "RGB",
                masks=masks,
            )
            if renders.shape[-1] == 4:
                colors, depths = renders[..., 0:3], renders[..., 3:4]
            else:
                colors, depths = renders, None

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

            if strategy_active:
                self.cfg.strategy.step_pre_backward(
                    params=self.splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=real_phase_step,
                    info=info,
                )

            depthloss: Optional[torch.Tensor] = None
            if in_nerf_phase:
                if not bool(is_nerf_sample.all().item()):
                    raise RuntimeError("Nerf phase batch contains real images.")
                l2_map = ((colors - pixels) ** 2).mean(dim=-1)
                w = torch.clamp(weight_maps, min=0.0)
                denom = torch.clamp(w.sum(dim=(1, 2)), min=1e-8)
                nerf_l2loss = ((l2_map * w).sum(dim=(1, 2)) / denom).mean()
                real_recon_loss = torch.tensor(0.0, device=device)
                ssimloss = torch.tensor(0.0, device=device)
                loss = nerf_l2loss
                real_loss = torch.tensor(0.0, device=device)
            else:
                if bool(is_nerf_sample.any().item()):
                    raise RuntimeError("Real phase batch contains nerf samples.")
                real_recon_loss = F.l1_loss(colors, pixels) if real_use_l1 else F.mse_loss(colors, pixels)
                ssimloss = 1.0 - fused_ssim(
                    colors.permute(0, 3, 1, 2), pixels.permute(0, 3, 1, 2), padding="valid"
                )
                real_loss = real_recon_loss * (1.0 - cfg.ssim_lambda) + ssimloss * cfg.ssim_lambda
                nerf_l2loss = torch.tensor(0.0, device=device)
                loss = real_loss
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
                    loss += depthloss * cfg.depth_lambda

            if cfg.use_bilateral_grid:
                tvloss = 10 * total_variation_loss(self.bil_grids.grids)
                loss += tvloss
            if cfg.opacity_reg > 0.0:
                loss += cfg.opacity_reg * torch.sigmoid(self.splats["opacities"]).mean()
            if cfg.scale_reg > 0.0:
                loss += cfg.scale_reg * torch.exp(self.splats["scales"]).mean()

            loss.backward()

            desc = f"loss={loss.item():.3f}| phase={'nerf' if in_nerf_phase else 'real'}| sh degree={sh_degree_to_use}| "
            if depthloss is not None:
                desc += f"depth loss={depthloss.item():.6f}| "
            if cfg.pose_opt and cfg.pose_noise:
                pose_err = F.l1_loss(camtoworlds_gt, camtoworlds)
                desc += f"pose err={pose_err.item():.6f}| "
            pbar.set_description(desc)

            if world_rank == 0 and cfg.tb_every > 0 and step % cfg.tb_every == 0:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                self.writer.add_scalar("train/loss", loss.item(), step)
                self.writer.add_scalar("train/real_loss", real_loss.item(), step)
                self.writer.add_scalar("train/nerf_weighted_l2loss", nerf_l2loss.item(), step)
                self.writer.add_scalar("train/reconloss", real_recon_loss.item(), step)
                if real_use_l1:
                    self.writer.add_scalar("train/l1loss", real_recon_loss.item(), step)
                else:
                    self.writer.add_scalar("train/l2loss", real_recon_loss.item(), step)
                self.writer.add_scalar("train/ssimloss", ssimloss.item(), step)
                self.writer.add_scalar("train/phase_is_nerf", float(1.0 if in_nerf_phase else 0.0), step)
                self.writer.add_scalar("train/num_GS", len(self.splats["means"]), step)
                self.writer.add_scalar("train/mem", mem, step)
                if depthloss is not None:
                    self.writer.add_scalar("train/depthloss", depthloss.item(), step)
                if cfg.use_bilateral_grid:
                    self.writer.add_scalar("train/tvloss", tvloss.item(), step)
                if cfg.tb_save_image:
                    canvas = torch.cat([pixels, colors], dim=2).detach().cpu().numpy()
                    canvas = canvas.reshape(-1, *canvas.shape[2:])
                    self.writer.add_image("train/render", canvas, step)
                self.writer.flush()

            if step in [i - 1 for i in cfg.save_steps] or step == max_steps - 1:
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
                gaussian_ids = info["gaussian_ids"]
                for k in self.splats.keys():
                    grad = self.splats[k].grad
                    if grad is None or grad.is_sparse:
                        continue
                    self.splats[k].grad = torch.sparse_coo_tensor(
                        indices=gaussian_ids[None],
                        values=grad[gaussian_ids],
                        size=self.splats[k].size(),
                        is_coalesced=len(Ks) == 1,
                    )

            if cfg.visible_adam:
                if cfg.packed:
                    visibility_mask = torch.zeros_like(self.splats["opacities"], dtype=bool)
                    visibility_mask.scatter_(0, info["gaussian_ids"], 1)
                else:
                    visibility_mask = (info["radii"] > 0).all(-1).any(0)

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

            if strategy_active:
                if isinstance(self.cfg.strategy, DefaultStrategy):
                    self.cfg.strategy.step_post_backward(
                        params=self.splats,
                        optimizers=self.optimizers,
                        state=self.strategy_state,
                        step=real_phase_step,
                        info=info,
                        packed=cfg.packed,
                    )
                elif isinstance(self.cfg.strategy, MCMCStrategy):
                    self.cfg.strategy.step_post_backward(
                        params=self.splats,
                        optimizers=self.optimizers,
                        state=self.strategy_state,
                        step=real_phase_step,
                        info=info,
                        lr=schedulers[0].get_last_lr()[0],
                    )
                else:
                    assert_never(self.cfg.strategy)

            # Eval timeline starts at real phase step 0 (nerf warmup steps are ignored).
            real_phase_step = step - nerf_phase_steps
            if real_phase_step >= 0 and real_phase_step % 100 == 0:
                self.eval_allstats()
                if len(self.stats_arr) > 0:
                    self.stats_arr[-1]["step"] = int(real_phase_step)
                    self.stats_arr[-1]["global_step"] = int(step)
                if world_rank == 0:
                    with open(os.path.join(cfg.result_dir, "gsplat_stats.json"), "w", encoding="utf-8") as f:
                        json.dump(self.stats_arr, f, indent=2, ensure_ascii=False)
                    training_time = time.time() - global_tic
                    out_path = os.path.join(cfg.result_dir, "time_logs.txt")
                    with open(out_path, "a") as f:
                        f.write(f"[gaussian-splatting-{real_phase_step}] - {training_time}\n")

            if cfg.compression is not None and step in [i - 1 for i in cfg.eval_steps]:
                self.run_compression(step=step)

            if not cfg.disable_viewer:
                self.viewer.lock.release()
                num_train_steps_per_sec = 1.0 / (max(time.time() - tic, 1e-10))
                num_train_rays_per_sec = num_train_rays_per_step * num_train_steps_per_sec
                self.viewer.render_tab_state.num_train_rays_per_sec = num_train_rays_per_sec
                self.viewer.update(step, num_train_rays_per_step)
