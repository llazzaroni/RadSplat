#!/usr/bin/env python3
"""Render dataset poses from a gsplat checkpoint and compare against ground truth images."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
import yaml

_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from gsplat_dir.cfg import Config
from gsplat_dir.runner import Runner
from submodules.gsplat.gsplat.strategy import DefaultStrategy

try:
    from fused_bilagrid import slice as bilateral_slice
except Exception:
    try:
        from lib_bilagrid import slice as bilateral_slice
    except Exception:
        bilateral_slice = None


def _to_uint8(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img)
    if arr.dtype == np.uint8:
        return arr
    if float(np.nanmax(arr)) > 1.5:
        return np.clip(arr, 0.0, 255.0).astype(np.uint8)
    return (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)


def _psnr(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_f = pred.astype(np.float32) / 255.0
    gt_f = gt.astype(np.float32) / 255.0
    mse = float(np.mean((pred_f - gt_f) ** 2))
    if mse <= 1e-12:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse)


def _safe_float(v) -> float:
    if isinstance(v, torch.Tensor):
        return float(v.detach().cpu().item())
    return float(v)


def _make_compare_panel(gt_u8: np.ndarray, pred_u8: np.ndarray) -> np.ndarray:
    diff = np.abs(pred_u8.astype(np.int16) - gt_u8.astype(np.int16)).astype(np.uint8)
    return np.concatenate([gt_u8, pred_u8, diff], axis=1)


def _load_base_cfg(ckpt_path: Path) -> Config:
    cfg = Config(strategy=DefaultStrategy(verbose=False))
    cfg_path = ckpt_path.parents[1] / "cfg.yml"
    if not cfg_path.is_file():
        return cfg

    payload = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    allowed = set(Config.__dataclass_fields__.keys())
    for key, value in payload.items():
        if key in allowed and key != "strategy":
            setattr(cfg, key, value)
    cfg.strategy = DefaultStrategy(verbose=False)
    return cfg


def _build_runner(args, ckpt: dict) -> Runner:
    cfg = _load_base_cfg(Path(args.ckpt))
    cfg.disable_viewer = True
    cfg.disable_video = True
    cfg.deterministic = args.deterministic
    cfg.result_dir = str((Path(args.out_dir) / "_tmp_runner").resolve())
    cfg.data_dir = args.data_dir
    cfg.data_factor = args.data_factor
    if args.nerf_samples_data_factor is not None:
        cfg.nerf_samples_data_factor = args.nerf_samples_data_factor
    cfg.pt_path = args.sample_pt
    cfg.max_steps = 1
    cfg.nerf_init = False
    cfg.pose_opt = "pose_adjust" in ckpt
    cfg.app_opt = "app_module" in ckpt
    return Runner(local_rank=0, world_rank=0, world_size=1, cfg=cfg)


def _load_ckpt(path: str) -> dict:
    return torch.load(path, map_location="cpu", weights_only=True)


def _restore_model(runner: Runner, ckpt: dict) -> int:
    for key in runner.splats.keys():
        runner.splats[key].data = ckpt["splats"][key].to(runner.device)
    if runner.cfg.pose_opt and "pose_adjust" in ckpt:
        runner.pose_adjust.load_state_dict(ckpt["pose_adjust"])
    if runner.cfg.app_opt and "app_module" in ckpt:
        runner.app_module.load_state_dict(ckpt["app_module"])
    return int(ckpt.get("step", -1))


def _global_to_dataset_maps(runner: Runner):
    train_g2l = {int(g): i for i, g in enumerate(runner.trainset.indices)}
    val_g2l = {int(g): i for i, g in enumerate(runner.valset.indices)}
    return train_g2l, val_g2l


def _select_indices(runner: Runner, split: str, include_nerf_samples: bool, max_train: int, max_val: int, seed: int):
    parser_names = runner.parser.image_names
    train_idx = [int(i) for i in runner.trainset.indices]
    val_idx = [int(i) for i in runner.valset.indices]

    def _filter(indices):
        if include_nerf_samples:
            return indices
        return [i for i in indices if not Path(parser_names[i]).name.startswith("nerf_sample_")]

    train_idx = _filter(train_idx)
    val_idx = _filter(val_idx)

    rng = random.Random(seed)

    def _limit(indices, max_views):
        if max_views <= 0 or max_views >= len(indices):
            return indices
        chosen = list(indices)
        rng.shuffle(chosen)
        return sorted(chosen[:max_views])

    train_idx = _limit(train_idx, max_train)
    val_idx = _limit(val_idx, max_val)

    if split == "train":
        return [("train", i) for i in train_idx]
    if split == "val":
        return [("val", i) for i in val_idx]
    if split == "all":
        tags = {i: "train" for i in train_idx}
        for i in val_idx:
            tags.setdefault(i, "val")
        return [(tags[i], i) for i in sorted(tags)]
    raise ValueError(f"Unsupported split: {split}")


def _fetch_sample(runner: Runner, global_idx: int, train_g2l: dict[int, int], val_g2l: dict[int, int]):
    if global_idx in train_g2l:
        return "train", runner.trainset[train_g2l[global_idx]]
    if global_idx in val_g2l:
        return "val", runner.valset[val_g2l[global_idx]]
    raise RuntimeError(f"Global image index {global_idx} not found in train/val datasets.")


def _render_one(runner: Runner, sample: dict, sample_split: str) -> np.ndarray:
    gt = sample["image"].detach().cpu().numpy()
    gt_u8 = _to_uint8(gt)
    height, width = gt_u8.shape[:2]

    image_ids = sample["image_id"].to(runner.device)[None]
    K = sample["K"].to(runner.device)[None]
    c2w = sample["camtoworld"].to(runner.device)[None]
    masks = sample["mask"].to(runner.device)[None] if "mask" in sample else None
    use_train_image_state = sample_split == "train"

    if runner.cfg.pose_opt and use_train_image_state:
        c2w = runner.pose_adjust(c2w, image_ids)

    renders, _, _ = runner.rasterize_splats(
        camtoworlds=c2w,
        Ks=K,
        width=width,
        height=height,
        sh_degree=runner.cfg.sh_degree,
        near_plane=runner.cfg.near_plane,
        far_plane=runner.cfg.far_plane,
        masks=masks,
        image_ids=image_ids if (runner.cfg.app_opt and use_train_image_state) else None,
        render_mode="RGB",
    )
    pred = renders[..., 0:3] if renders.shape[-1] == 4 else renders
    if runner.cfg.use_bilateral_grid and bilateral_slice is not None and use_train_image_state:
        grid_y, grid_x = torch.meshgrid(
            (torch.arange(height, device=runner.device) + 0.5) / height,
            (torch.arange(width, device=runner.device) + 0.5) / width,
            indexing="ij",
        )
        grid_xy = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
        pred = bilateral_slice(
            runner.bil_grids,
            grid_xy.expand(pred.shape[0], -1, -1, -1),
            pred,
            image_ids.unsqueeze(-1),
        )["rgb"]
    pred = torch.clamp(pred[0], 0.0, 1.0).detach().cpu().numpy()
    return gt_u8, _to_uint8(pred)


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Render dataset poses from a gsplat checkpoint and compare against dataset images."
    )
    ap.add_argument("--ckpt", required=True, help="Path to gsplat checkpoint (*.pt).")
    ap.add_argument("--data-dir", required=True, help="Dataset root.")
    ap.add_argument("--out-dir", required=True, help="Output directory.")
    ap.add_argument("--sample-pt", default="", help="Optional split payload used during training.")
    ap.add_argument("--split", choices=["all", "train", "val"], default="all")
    ap.add_argument("--data-factor", type=int, default=1)
    ap.add_argument("--nerf-samples-data-factor", type=int, default=None)
    ap.add_argument("--max-train-views", type=int, default=0, help="0 means all.")
    ap.add_argument("--max-val-views", type=int, default=0, help="0 means all.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--include-nerf-samples", action="store_true")
    ap.add_argument("--save-side-by-side", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = _load_ckpt(args.ckpt)
    runner = _build_runner(args, ckpt)
    ckpt_step = _restore_model(runner, ckpt)
    train_g2l, val_g2l = _global_to_dataset_maps(runner)

    selected = _select_indices(
        runner=runner,
        split=args.split,
        include_nerf_samples=args.include_nerf_samples,
        max_train=args.max_train_views,
        max_val=args.max_val_views,
        seed=args.seed,
    )
    if not selected:
        raise RuntimeError("No images selected for rendering.")

    rows = []
    summary = {"train": [], "val": [], "all": []}
    metrics_out = out_dir / "metrics.csv"
    summary_out = out_dir / "summary.json"

    render_root = out_dir / "renders"
    compare_root = out_dir / "compare"
    render_root.mkdir(parents=True, exist_ok=True)
    if args.save_side_by_side:
        compare_root.mkdir(parents=True, exist_ok=True)

    print(f"[render-compare] ckpt_step={ckpt_step}, selected={len(selected)}, split={args.split}")
    for i, (requested_split, global_idx) in enumerate(selected, start=1):
        actual_split, sample = _fetch_sample(runner, global_idx, train_g2l, val_g2l)
        name = runner.parser.image_names[global_idx]
        gt_u8, pred_u8 = _render_one(runner, sample, actual_split)

        split_dir = render_root / actual_split
        split_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(name).stem
        pred_path = split_dir / f"{stem}_pred.png"
        imageio.imwrite(pred_path, pred_u8)

        compare_path = None
        if args.save_side_by_side:
            compare_dir = compare_root / actual_split
            compare_dir.mkdir(parents=True, exist_ok=True)
            compare_path = compare_dir / f"{stem}_gt_pred_diff.png"
            imageio.imwrite(compare_path, _make_compare_panel(gt_u8, pred_u8))

        gt_t = torch.from_numpy(gt_u8).float().permute(2, 0, 1)[None].to(runner.device) / 255.0
        pred_t = torch.from_numpy(pred_u8).float().permute(2, 0, 1)[None].to(runner.device) / 255.0
        psnr = _psnr(pred_u8, gt_u8)
        ssim = _safe_float(runner.ssim(pred_t, gt_t))
        lpips = _safe_float(runner.lpips(pred_t, gt_t))

        row = {
            "split": actual_split,
            "requested_split": requested_split,
            "global_idx": global_idx,
            "image_name": name,
            "psnr": psnr,
            "ssim": ssim,
            "lpips": lpips,
            "pred_path": str(pred_path.relative_to(out_dir)),
            "compare_path": "" if compare_path is None else str(compare_path.relative_to(out_dir)),
        }
        rows.append(row)
        summary[actual_split].append(row)
        summary["all"].append(row)

        if i % 10 == 0 or i == len(selected):
            print(f"[render-compare] {i}/{len(selected)}")

    with metrics_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "split",
                "requested_split",
                "global_idx",
                "image_name",
                "psnr",
                "ssim",
                "lpips",
                "pred_path",
                "compare_path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary_payload = {
        "ckpt": str(Path(args.ckpt).resolve()),
        "data_dir": str(Path(args.data_dir).resolve()),
        "ckpt_step": ckpt_step,
        "num_images": len(rows),
        "splits": {},
    }
    for split_name in ("train", "val", "all"):
        split_rows = summary[split_name]
        if not split_rows:
            continue
        summary_payload["splits"][split_name] = {
            "count": len(split_rows),
            "psnr_mean": float(np.mean([r["psnr"] for r in split_rows])),
            "ssim_mean": float(np.mean([r["ssim"] for r in split_rows])),
            "lpips_mean": float(np.mean([r["lpips"] for r in split_rows])),
        }

    summary_out.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    print(f"[render-compare] wrote {metrics_out}")
    print(f"[render-compare] wrote {summary_out}")


if __name__ == "__main__":
    main()
