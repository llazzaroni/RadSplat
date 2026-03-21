#!/usr/bin/env python3
"""Render dataset views by COLMAP index using the exact training camera/data path."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

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


def _to_uint8(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img)
    if arr.dtype == np.uint8:
        return arr
    if float(np.nanmax(arr)) > 1.5:
        return np.clip(arr, 0.0, 255.0).astype(np.uint8)
    return (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)


def _build_runner(
    data_dir: str,
    result_dir: str,
    data_factor: int,
    nerf_samples_data_factor: int,
    split_payload_path: str,
):
    cfg = Config(strategy=DefaultStrategy(verbose=False))
    cfg.disable_viewer = True
    cfg.data_dir = data_dir
    cfg.result_dir = result_dir
    cfg.data_factor = data_factor
    cfg.nerf_samples_data_factor = nerf_samples_data_factor
    cfg.pt_path = split_payload_path
    cfg.nerf_init = False
    cfg.max_steps = 1
    return Runner(local_rank=0, world_rank=0, world_size=1, cfg=cfg)


def _load_ckpt(runner: Runner, ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location=runner.device, weights_only=True)
    for k in runner.splats.keys():
        runner.splats[k].data = ckpt["splats"][k].to(runner.device)
    return int(ckpt.get("step", -1))


def _global_to_dataset_maps(runner: Runner):
    train_g2l = {int(g): i for i, g in enumerate(runner.trainset.indices)}
    val_g2l = {int(g): i for i, g in enumerate(runner.valset.indices)}
    return train_g2l, val_g2l


def _select_global_indices(runner: Runner, source: str, prefix: str):
    if source == "train":
        return [int(i) for i in runner.trainset.indices]
    if source == "val":
        return [int(i) for i in runner.valset.indices]

    all_idx = list(range(len(runner.parser.image_names)))
    if source == "all":
        return all_idx
    if source == "prefix":
        return [i for i in all_idx if Path(runner.parser.image_names[i]).name.startswith(prefix)]
    raise ValueError(f"Unsupported source: {source}")


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(
        description="Render by COLMAP-indexed poses using exactly the same camera tensors as training."
    )
    ap.add_argument("--ckpt", required=True, help="Path to gsplat checkpoint.")
    ap.add_argument("--dataset", required=True, help="Dataset root.")
    ap.add_argument("--out-dir", required=True, help="Output directory.")
    ap.add_argument("--sample-pt", default="", help="Optional split payload used in training.")
    ap.add_argument("--source", choices=["all", "train", "val", "prefix"], default="prefix")
    ap.add_argument("--prefix", default="nerf_sample_", help="Used when --source prefix.")
    ap.add_argument("--num-views", type=int, default=40, help="Max number of views to render.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data-factor", type=int, default=1)
    ap.add_argument("--nerf-samples-data-factor", type=int, default=8)
    ap.add_argument("--save-side-by-side", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_pred = out_dir / "renders"
    out_sbs = out_dir / "side_by_side"
    out_pred.mkdir(parents=True, exist_ok=True)
    if args.save_side_by_side:
        out_sbs.mkdir(parents=True, exist_ok=True)

    runner = _build_runner(
        data_dir=args.dataset,
        result_dir=str(out_dir / "_tmp_runner"),
        data_factor=args.data_factor,
        nerf_samples_data_factor=args.nerf_samples_data_factor,
        split_payload_path=args.sample_pt,
    )
    ckpt_step = _load_ckpt(runner, args.ckpt)
    train_g2l, val_g2l = _global_to_dataset_maps(runner)

    indices = _select_global_indices(runner, source=args.source, prefix=args.prefix)
    if not indices:
        raise RuntimeError(f"No indices found for source={args.source}, prefix={args.prefix}.")
    rng = random.Random(args.seed)
    rng.shuffle(indices)
    indices = indices[: max(1, min(args.num_views, len(indices)))]

    print(f"[render-colmap] ckpt_step={ckpt_step}, source={args.source}, views={len(indices)}")

    for i, idx in enumerate(indices, start=1):
        name = runner.parser.image_names[idx]
        if idx in train_g2l:
            data = runner.trainset[train_g2l[idx]]
        elif idx in val_g2l:
            data = runner.valset[val_g2l[idx]]
        else:
            continue  # should not happen with normal train/val splits

        gt_u8 = _to_uint8(data["image"].detach().cpu().numpy())
        h, w = gt_u8.shape[:2]
        K = data["K"].to(runner.device)[None]
        c2w = data["camtoworld"].to(runner.device)[None]

        renders, _, _ = runner.rasterize_splats(
            camtoworlds=c2w,
            Ks=K,
            width=w,
            height=h,
            sh_degree=runner.cfg.sh_degree,
            near_plane=runner.cfg.near_plane,
            far_plane=runner.cfg.far_plane,
            render_mode="RGB",
        )
        pred = renders[..., 0:3] if renders.shape[-1] == 4 else renders
        pred_u8 = _to_uint8(pred[0].detach().cpu().numpy())

        stem = Path(name).stem
        imageio.imwrite(out_pred / f"{stem}_pred.png", pred_u8)
        if args.save_side_by_side:
            imageio.imwrite(out_sbs / f"{stem}_gt_pred.png", np.concatenate([gt_u8, pred_u8], axis=1))

        if i % 10 == 0 or i == len(indices):
            print(f"[render-colmap] {i}/{len(indices)}")

    print("[render-colmap] done")


if __name__ == "__main__":
    main()
