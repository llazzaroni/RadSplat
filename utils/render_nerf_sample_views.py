#!/usr/bin/env python3
"""Render train/val views from a dataset using a GS checkpoint and split payload."""

import argparse
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


def _load_runner(
    data_dir: str,
    result_dir: str,
    data_factor: int,
    nerf_samples_data_factor: int,
    split_payload_path: str,
    deterministic: bool,
):
    cfg = Config(strategy=DefaultStrategy(verbose=False))
    cfg.disable_viewer = True
    cfg.data_dir = data_dir
    cfg.result_dir = result_dir
    cfg.data_factor = data_factor
    cfg.nerf_samples_data_factor = nerf_samples_data_factor
    cfg.pt_path = split_payload_path
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
    return Path(name).name.startswith("nerf_sample_")


def _select_indices(indices, max_views: int, seed: int):
    idx = [int(i) for i in indices]
    if max_views <= 0 or max_views >= len(idx):
        return idx
    rng = random.Random(seed)
    rng.shuffle(idx)
    return idx[:max_views]


def _write_split_manifest(path: Path, split_name: str, indices, runner: Runner):
    lines = []
    cam_ids = set()
    for idx in indices:
        name = runner.parser.image_names[idx]
        cam_id = int(runner.parser.camera_ids[idx])
        cam_ids.add(cam_id)
        lines.append(f"{idx}\t{cam_id}\t{name}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    print(f"[split] {split_name}: {len(indices)} images, camera_ids={sorted(cam_ids)}")


def _to_uint8(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img)
    if arr.dtype == np.uint8:
        return arr
    # Dataset samples are often float in [0, 255], while renders are float in [0, 1].
    if float(np.nanmax(arr)) > 1.5:
        return np.clip(arr, 0.0, 255.0).astype(np.uint8)
    return (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)


def _render_indices(
    runner: Runner,
    indices,
    out_dir: Path,
    save_side_by_side: bool,
    split_name: str,
):
    out_render = out_dir / "renders"
    out_sbs = out_dir / "side_by_side"
    out_render.mkdir(parents=True, exist_ok=True)
    if save_side_by_side:
        out_sbs.mkdir(parents=True, exist_ok=True)

    # Build global-index -> local-index maps so we can fetch the exact preprocessed
    # training/validation sample (same undistortion/scaling path used in training).
    train_g2l = {int(g): i for i, g in enumerate(runner.trainset.indices)}
    val_g2l = {int(g): i for i, g in enumerate(runner.valset.indices)}

    for i, idx in enumerate(indices, start=1):
        name = runner.parser.image_names[idx]
        if idx in train_g2l:
            data = runner.trainset[train_g2l[idx]]
        elif idx in val_g2l:
            data = runner.valset[val_g2l[idx]]
        else:
            raise RuntimeError(f"Global image index {idx} not found in train/val datasets.")

        gt = data["image"].detach().cpu().numpy()
        gt_u8 = _to_uint8(gt)
        gt_h, gt_w = gt_u8.shape[:2]
        K = data["K"].to(runner.device)[None]
        c2w = data["camtoworld"].to(runner.device)[None]

        renders, _, _ = runner.rasterize_splats(
            camtoworlds=c2w,
            Ks=K,
            width=gt_w,
            height=gt_h,
            sh_degree=runner.cfg.sh_degree,
            near_plane=runner.cfg.near_plane,
            far_plane=runner.cfg.far_plane,
            render_mode="RGB",
        )
        pred = renders[..., 0:3] if renders.shape[-1] == 4 else renders
        pred_u8 = _to_uint8(pred[0].detach().cpu().numpy())

        stem = Path(name).stem
        imageio.imwrite(out_render / f"{stem}_pred.png", pred_u8)
        if save_side_by_side:
            sbs = np.concatenate([gt_u8, pred_u8], axis=1)
            imageio.imwrite(out_sbs / f"{stem}_gt_pred.png", sbs)

        if i % 10 == 0 or i == len(indices):
            print(f"[render:{split_name}] {i}/{len(indices)}")


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description="Render train/val dataset views from a GS checkpoint.")
    ap.add_argument("--ckpt", required=True, help="Path to GS checkpoint (*.pt).")
    ap.add_argument("--data-dir", required=True, help="Dataset root.")
    ap.add_argument(
        "--sample-pt",
        required=True,
        help="Split payload path (.pt) used to define train/val camera membership.",
    )
    ap.add_argument("--out-dir", required=True, help="Output directory for rendered images.")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for view sampling.")
    ap.add_argument("--data-factor", type=int, default=1, help="Real-image factor for parser.")
    ap.add_argument(
        "--nerf-samples-data-factor",
        type=int,
        default=8,
        help="Factor used for nerf_sample_* images in dataset.",
    )
    ap.add_argument("--max-train-views", type=int, default=0, help="Max train views to render (0=all).")
    ap.add_argument("--max-val-views", type=int, default=0, help="Max val views to render (0=all).")
    ap.add_argument("--deterministic", action="store_true", help="Deterministic torch/cudnn.")
    ap.add_argument(
        "--save-side-by-side",
        action="store_true",
        help="Also save GT|Pred side-by-side images.",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runner = _load_runner(
        data_dir=args.data_dir,
        result_dir=str(out_dir / "_tmp_runner"),
        data_factor=args.data_factor,
        nerf_samples_data_factor=args.nerf_samples_data_factor,
        split_payload_path=args.sample_pt,
        deterministic=args.deterministic,
    )
    ckpt_step = _load_ckpt_into_runner(runner, args.ckpt)

    train_indices = _select_indices(runner.trainset.indices, args.max_train_views, args.seed)
    val_indices = _select_indices(runner.valset.indices, args.max_val_views, args.seed + 1)

    if not train_indices and not val_indices:
        raise RuntimeError("No train/val indices found from sample payload.")

    print(f"[render] ckpt_step={ckpt_step}")
    _write_split_manifest(out_dir / "train_manifest.txt", "train", train_indices, runner)
    _write_split_manifest(out_dir / "val_manifest.txt", "val", val_indices, runner)

    if train_indices:
        _render_indices(
            runner=runner,
            indices=train_indices,
            out_dir=out_dir / "train",
            save_side_by_side=args.save_side_by_side,
            split_name="train",
        )
    if val_indices:
        _render_indices(
            runner=runner,
            indices=val_indices,
            out_dir=out_dir / "val",
            save_side_by_side=args.save_side_by_side,
            split_name="val",
        )

    print("[render] done")


if __name__ == "__main__":
    main()
