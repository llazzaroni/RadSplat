import json
import os
import sys
import time
from pathlib import Path

import torch
import tyro

from gsplat_dir.cfg import Config
from gsplat_dir.runner_staged import RunnerStaged
from submodules.gsplat.gsplat.distributed import cli
from submodules.gsplat.gsplat.strategy import DefaultStrategy, MCMCStrategy


def main(local_rank: int, world_rank, world_size: int, cfg: Config):
    if world_size > 1 and not cfg.disable_viewer:
        cfg.disable_viewer = True
        if world_rank == 0:
            print("Viewer is disabled in distributed training.")

    cfg.result_dir = os.path.abspath(os.path.expanduser(cfg.result_dir))
    print("DEBUG result_dir:", cfg.result_dir, file=sys.stderr)
    if cfg.result_dir in {"", "/"}:
        raise ValueError(
            f"Refusing to write to result_dir={cfg.result_dir!r}. "
            "Pass a user-writable directory with --result_dir."
        )
    os.makedirs(cfg.result_dir, exist_ok=True)

    runner = RunnerStaged(local_rank, world_rank, world_size, cfg)

    if cfg.ckpt:
        ckpts = [torch.load(file, map_location=runner.device, weights_only=True) for file in cfg.ckpt]
        for k in runner.splats.keys():
            runner.splats[k].data = torch.cat([ckpt["splats"][k] for ckpt in ckpts])
        step = ckpts[0]["step"]
        runner.eval(step=step)
        runner.render_traj(step=step)
        if cfg.compression is not None:
            runner.run_compression(step=step)
    else:
        if cfg.save_first_ckp:
            runner.save_ckpt()
        runner.train()
        Path(cfg.result_dir).mkdir(exist_ok=True)
        out_path = os.path.join(cfg.result_dir, "gsplat_stats.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(runner.stats_arr, f, indent=2, ensure_ascii=False)

    if not cfg.disable_viewer:
        runner.viewer.complete()
        print("Viewer running... Ctrl+C to exit.")
        time.sleep(1000000)


if __name__ == "__main__":
    configs = {
        "default": (
            "Staged gsplat training (nerf-only warmup then real-only).",
            Config(strategy=DefaultStrategy(verbose=True)),
        ),
        "mcmc": (
            "Staged gsplat training with MCMC densification strategy.",
            Config(
                init_opa=0.5,
                init_scale=0.1,
                opacity_reg=0.01,
                scale_reg=0.01,
                strategy=MCMCStrategy(verbose=True),
            ),
        ),
    }
    cfg = tyro.extras.overridable_config_cli(configs)
    cfg.adjust_steps(cfg.steps_scaler)

    if cfg.use_bilateral_grid or cfg.use_fused_bilagrid:
        if cfg.use_fused_bilagrid:
            cfg.use_bilateral_grid = True
            from fused_bilagrid import BilateralGrid, color_correct, slice, total_variation_loss
        else:
            cfg.use_bilateral_grid = True
            from lib_bilagrid import BilateralGrid, color_correct, slice, total_variation_loss

    if cfg.compression == "png":
        try:
            import plas  # noqa: F401
            import torchpq  # noqa: F401
        except Exception:
            raise ImportError("To use PNG compression, install torchpq and plas.")

    if cfg.with_ut:
        assert cfg.with_eval3d, "Training with UT requires setting `with_eval3d` flag."

    cli(main, cfg, verbose=True)
