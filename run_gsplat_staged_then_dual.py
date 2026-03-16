import copy
import json
import os
import sys
import time
from pathlib import Path

import torch
import tyro
from torch.nn.parallel import DistributedDataParallel as DDP

from gsplat_dir.cfg import Config
from gsplat_dir.runner_dual import RunnerDual
from gsplat_dir.runner_staged import RunnerStaged
from submodules.gsplat.gsplat.distributed import cli
from submodules.gsplat.gsplat.strategy import DefaultStrategy, MCMCStrategy


def _module_state_dict_to_cpu(module):
    target = module.module if isinstance(module, DDP) else module
    state = target.state_dict()
    out = {}
    for k, v in state.items():
        out[k] = v.detach().cpu().clone() if torch.is_tensor(v) else copy.deepcopy(v)
    return out


def _load_module_state_dict(module, state_dict):
    target = module.module if isinstance(module, DDP) else module
    target.load_state_dict(state_dict)


def _copy_splats_to_cpu(splats):
    state = splats.state_dict()
    out = {}
    for k, v in state.items():
        out[k] = v.detach().cpu().clone()
    return out


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
    if cfg.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        if world_rank == 0:
            print(
                f"[Determinism] enabled (seed={cfg.seed}); "
                "set CUBLAS_WORKSPACE_CONFIG=:4096:8"
            )

    if cfg.ckpt:
        raise ValueError(
            "--ckpt is not supported in staged-then-dual mode. "
            "This script always trains both phases end-to-end."
        )

    pre_steps = int(getattr(cfg, "staged_nerf_phase_steps", 0))
    if pre_steps <= 0:
        raise ValueError(
            "staged_nerf_phase_steps must be > 0 for staged-then-dual mode."
        )

    # Stage 1: nerf-only pretraining using staged runner.
    stage1_cfg = copy.deepcopy(cfg)
    stage1_cfg.result_dir = os.path.join(cfg.result_dir, "stage1_staged_pretrain")
    stage1_cfg.staged_nerf_phase_steps = pre_steps
    stage1_cfg.staged_real_phase_steps = 0
    os.makedirs(stage1_cfg.result_dir, exist_ok=True)
    if world_rank == 0:
        print(
            f"[Pipeline] Stage 1 (staged nerf-only) steps={pre_steps}, "
            f"result_dir={stage1_cfg.result_dir}"
        )

    stage1_runner = RunnerStaged(local_rank, world_rank, world_size, stage1_cfg)
    if stage1_cfg.save_first_ckp:
        stage1_runner.save_ckpt()
    stage1_runner.train()
    Path(stage1_cfg.result_dir).mkdir(exist_ok=True)
    with open(
        os.path.join(stage1_cfg.result_dir, "gsplat_stats.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(stage1_runner.stats_arr, f, indent=2, ensure_ascii=False)

    # Snapshot stage-1 weights on CPU before constructing stage-2 runner.
    splats_state_cpu = _copy_splats_to_cpu(stage1_runner.splats)
    pose_state_cpu = None
    app_state_cpu = None
    bil_state_cpu = None
    if cfg.pose_opt and hasattr(stage1_runner, "pose_adjust"):
        pose_state_cpu = _module_state_dict_to_cpu(stage1_runner.pose_adjust)
    if cfg.app_opt and hasattr(stage1_runner, "app_module"):
        app_state_cpu = _module_state_dict_to_cpu(stage1_runner.app_module)
    if cfg.use_bilateral_grid and hasattr(stage1_runner, "bil_grids"):
        bil_state_cpu = _module_state_dict_to_cpu(stage1_runner.bil_grids)

    del stage1_runner
    torch.cuda.empty_cache()

    # Stage 2: dual runner training initialized from stage-1 weights.
    stage2_cfg = copy.deepcopy(cfg)
    stage2_cfg.result_dir = cfg.result_dir
    os.makedirs(stage2_cfg.result_dir, exist_ok=True)
    if world_rank == 0:
        print(
            f"[Pipeline] Stage 2 (dual) steps={stage2_cfg.max_steps}, "
            f"result_dir={stage2_cfg.result_dir}"
        )

    stage2_runner = RunnerDual(local_rank, world_rank, world_size, stage2_cfg)
    with torch.no_grad():
        for k, v in splats_state_cpu.items():
            stage2_runner.splats[k].data.copy_(v.to(stage2_runner.device))
    if pose_state_cpu is not None:
        _load_module_state_dict(stage2_runner.pose_adjust, pose_state_cpu)
    if app_state_cpu is not None:
        _load_module_state_dict(stage2_runner.app_module, app_state_cpu)
    if bil_state_cpu is not None:
        _load_module_state_dict(stage2_runner.bil_grids, bil_state_cpu)

    if stage2_cfg.save_first_ckp:
        stage2_runner.save_ckpt()
    stage2_runner.train()
    Path(stage2_cfg.result_dir).mkdir(exist_ok=True)
    with open(
        os.path.join(stage2_cfg.result_dir, "gsplat_stats.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(stage2_runner.stats_arr, f, indent=2, ensure_ascii=False)

    if not stage2_cfg.disable_viewer:
        stage2_runner.viewer.complete()
        print("Viewer running... Ctrl+C to exit.")
        time.sleep(1000000)


if __name__ == "__main__":
    configs = {
        "default": (
            "Two-stage pipeline: staged nerf-only pretraining, then dual-loader training.",
            Config(strategy=DefaultStrategy(verbose=True)),
        ),
        "mcmc": (
            "Two-stage pipeline with MCMC densification strategy.",
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
