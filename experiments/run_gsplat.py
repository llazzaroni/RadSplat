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


class ExperimentConfig(Config):
    # Training mode selection:
    # --dual-runner only -> dual
    # --staged-runner only -> staged
    # both flags -> staged pretraining, then dual
    dual_runner: bool = False
    staged_runner: bool = False


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


def _prepare_cfg(world_rank: int, world_size: int, cfg: ExperimentConfig) -> None:
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


def _write_stats(cfg: Config, stats_arr) -> None:
    Path(cfg.result_dir).mkdir(exist_ok=True)
    out_path = os.path.join(cfg.result_dir, "gsplat_stats.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(stats_arr, f, indent=2, ensure_ascii=False)


def _run_single_mode(
    local_rank: int,
    world_rank: int,
    world_size: int,
    cfg: ExperimentConfig,
    mode: str,
):
    runner_cls = RunnerDual if mode == "dual" else RunnerStaged
    runner = runner_cls(local_rank, world_rank, world_size, cfg)

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
        _write_stats(cfg, runner.stats_arr)

    return runner


def _run_staged_then_dual(
    local_rank: int,
    world_rank: int,
    world_size: int,
    cfg: ExperimentConfig,
):
    if cfg.ckpt:
        raise ValueError(
            "--ckpt is not supported in staged+dual mode. "
            "This mode always trains both phases end-to-end."
        )

    pre_steps = int(getattr(cfg, "staged_nerf_phase_steps", 0))
    if pre_steps <= 0:
        raise ValueError(
            "staged_nerf_phase_steps must be > 0 when using both "
            "--staged-runner and --dual-runner."
        )

    stage1_cfg = copy.deepcopy(cfg)
    stage1_cfg.result_dir = os.path.join(cfg.result_dir, "stage1_staged_pretrain")
    stage1_cfg.staged_nerf_phase_steps = pre_steps
    stage1_cfg.staged_real_phase_steps = 0
    os.makedirs(stage1_cfg.result_dir, exist_ok=True)
    if world_rank == 0:
        print(
            f"[Pipeline] Stage 1 (staged) steps={pre_steps}, "
            f"result_dir={stage1_cfg.result_dir}"
        )

    stage1_runner = RunnerStaged(local_rank, world_rank, world_size, stage1_cfg)
    if stage1_cfg.save_first_ckp:
        stage1_runner.save_ckpt()
    stage1_runner.train()
    _write_stats(stage1_cfg, stage1_runner.stats_arr)

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
    _write_stats(stage2_cfg, stage2_runner.stats_arr)
    return stage2_runner


def main(local_rank: int, world_rank, world_size: int, cfg: ExperimentConfig):
    _prepare_cfg(world_rank, world_size, cfg)

    if not cfg.dual_runner and not cfg.staged_runner:
        raise ValueError(
            "Select a mode with --dual-runner and/or --staged-runner. "
            "Use both to run staged pretraining followed by dual training."
        )

    if cfg.staged_runner and cfg.dual_runner:
        runner = _run_staged_then_dual(local_rank, world_rank, world_size, cfg)
    elif cfg.dual_runner:
        runner = _run_single_mode(local_rank, world_rank, world_size, cfg, mode="dual")
    else:
        runner = _run_single_mode(local_rank, world_rank, world_size, cfg, mode="staged")

    if not cfg.disable_viewer:
        runner.viewer.complete()
        print("Viewer running... Ctrl+C to exit.")
        time.sleep(1000000)


if __name__ == "__main__":
    configs = {
        "default": (
            "Unified gsplat entrypoint. Select --dual-runner, --staged-runner, or both.",
            ExperimentConfig(strategy=DefaultStrategy(verbose=True)),
        ),
        "mcmc": (
            "Unified gsplat entrypoint with MCMC densification strategy.",
            ExperimentConfig(
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
