#!/usr/bin/env python3
"""Render a subset of nerf_sample_* views from a dataset using a GS checkpoint."""

import argparse
import os
import random
from pathlib import Path
import sys

import imageio.v2 as imageio
import numpy as np
import torch

_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from gsplat_dir.cfg import Config
from gsplat_dir.runner import Runner
from submodules.gsplat.gsplat.strategy import DefaultStrategy


def _load_runner(data_dir: str, result_dir: str, data_factor: int, nerf_samples_data_factor: int, deterministic: bool):
    cfg = Config(strategy=DefaultStrategy(verbose=False))
    cfg.disable_viewer = True
    cfg.data_dir = data_dir
    cfg.result_dir = result_dir
    cfg.data_factor = data_factor
    cfg.nerf_samples_data_factor = nerf_samples_data_factor
    cfg.nerf_init = False
    cfg.deterministic = deterministic
    cfg.max_steps = 1
    return Runner(local_rank=0, world_rank=0, world_size=1, cfg=cfg)


def _load_ckpt_into_runner(runner: Runner, ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location=runner.device, weights_only=True)
    for k in runner.splats.keys():
        runner.splats[k].data = ckpt["splats"][k].to(runner.device)
    return int(ckpt.get("step", -1))


def _is_nerf_sample(name: str) -> bool:
    return os.path.basename(name).startswith("nerf_sample_")


def _find_indices(runner: Runner, view_source: str):
    if view_source == "nerf_samples":
        return [idx for idx, name in enumerate(runner.parser.image_names) if _is_nerf_sample(name)]
    if view_source == "val":
        return [int(i) for i in runner.valset.indices if not _is_nerf_sample(runner.parser.image_names[int(i)])]
    if view_source == "train":
        return [int(i) for i in runner.trainset.indices if not _is_nerf_sample(runner.parser.image_names[int(i)])]
    if view_source == "all":
        return [idx for idx, name in enumerate(runner.parser.image_names) if not _is_nerf_sample(name)]
    raise ValueError(f"Unsupported view_source: {view_source}")


def _to_uint8(img: np.ndarray) -> np.ndarray:
    return (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description="Render dataset views from a GS checkpoint.")
    ap.add_argument("--ckpt", required=True, help="Path to GS checkpoint (*.pt).")
    ap.add_argument("--data-dir", required=True, help="Dataset root.")
    ap.add_argument("--out-dir", required=True, help="Output directory for rendered images.")
    ap.add_argument("--num-views", type=int, default=32, help="How many views to render.")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for view sampling.")
    ap.add_argument("--data-factor", type=int, default=1, help="Real-image factor for parser.")
    ap.add_argument(
        "--nerf-samples-data-factor",
        type=int,
        default=8,
        help="Factor used for nerf_sample_* images in dataset.",
    )
    ap.add_argument(
        "--view-source",
        choices=["nerf_samples", "val", "train", "all"],
        default="nerf_samples",
        help="Which views to render.",
    )
    ap.add_argument("--deterministic", action="store_true", help="Deterministic torch/cudnn.")
    ap.add_argument(
        "--save-side-by-side",
        action="store_true",
        help="Also save GT|Pred side-by-side images.",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_render = out_dir / "renders"
    out_sbs = out_dir / "side_by_side"
    out_render.mkdir(parents=True, exist_ok=True)
    if args.save_side_by_side:
        out_sbs.mkdir(parents=True, exist_ok=True)

    runner = _load_runner(
        data_dir=args.data_dir,
        result_dir=str(out_dir / "_tmp_runner"),
        data_factor=args.data_factor,
        nerf_samples_data_factor=args.nerf_samples_data_factor,
        deterministic=args.deterministic,
    )
    ckpt_step = _load_ckpt_into_runner(runner, args.ckpt)

    indices = _find_indices(runner, args.view_source)
    if len(indices) == 0:
        raise RuntimeError(f"No views found for view_source={args.view_source}.")

    rng = random.Random(args.seed)
    rng.shuffle(indices)
    indices = indices[: max(1, min(args.num_views, len(indices)))]

    print(f"[render-views] ckpt_step={ckpt_step} total_views={len(indices)} source={args.view_source}")
    print(f"[render-nerf-samples] rendering {len(indices)} views -> {out_render}")

    for i, idx in enumerate(indices, start=1):
        name = runner.parser.image_names[idx]
        gt_path = (
            runner.parser.image_paths_nerf[idx]
            if _is_nerf_sample(name)
            else runner.parser.image_paths[idx]
        )
        gt_u8 = imageio.imread(gt_path)[..., :3]
        gt_h, gt_w = gt_u8.shape[:2]
        camera_id = runner.parser.camera_ids[idx]
        K_np = runner.parser.Ks_dict[camera_id].copy()
        base_w, base_h = runner.parser.imsize_dict[camera_id]
        sx = gt_w / float(base_w)
        sy = gt_h / float(base_h)
        K_np[0, :] *= sx
        K_np[1, :] *= sy
        K = torch.from_numpy(K_np).float().to(runner.device)[None]
        c2w = torch.from_numpy(runner.parser.camtoworlds[idx]).float().to(runner.device)[None]
        width, height = gt_w, gt_h

        renders, _, _ = runner.rasterize_splats(
            camtoworlds=c2w,
            Ks=K,
            width=width,
            height=height,
            sh_degree=runner.cfg.sh_degree,
            near_plane=runner.cfg.near_plane,
            far_plane=runner.cfg.far_plane,
            render_mode="RGB",
        )
        pred = renders[..., 0:3] if renders.shape[-1] == 4 else renders
        pred_np = pred[0].detach().cpu().numpy()
        pred_u8 = _to_uint8(pred_np)

        stem = Path(name).stem
        imageio.imwrite(out_render / f"{stem}_pred.png", pred_u8)

        if args.save_side_by_side:
            sbs = np.concatenate([gt_u8, pred_u8], axis=1)
            imageio.imwrite(out_sbs / f"{stem}_gt_pred.png", sbs)

        if i % 10 == 0 or i == len(indices):
            print(f"[render-views] {i}/{len(indices)}")

    print("[render-views] done")


if __name__ == "__main__":
    main()
