#!/usr/bin/env python3
"""Render nerf_sample_* views using poses directly from COLMAP images.bin/cameras.bin."""

import argparse
import os
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
from submodules.nerfstudio.nerfstudio.data.utils.colmap_parsing_utils import (
    qvec2rotmat,
    read_cameras_binary,
    read_images_binary,
)


def _to_uint8(img: np.ndarray) -> np.ndarray:
    return (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)


def _build_runner(data_dir: str, result_dir: str, data_factor: int, nerf_samples_data_factor: int):
    cfg = Config(strategy=DefaultStrategy(verbose=False))
    cfg.disable_viewer = True
    cfg.data_dir = data_dir
    cfg.result_dir = result_dir
    cfg.data_factor = data_factor
    cfg.nerf_samples_data_factor = nerf_samples_data_factor
    cfg.nerf_init = False
    cfg.max_steps = 1
    return Runner(local_rank=0, world_rank=0, world_size=1, cfg=cfg)


def _load_ckpt(runner: Runner, ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location=runner.device, weights_only=True)
    for k in runner.splats.keys():
        runner.splats[k].data = ckpt["splats"][k].to(runner.device)
    return int(ckpt.get("step", -1))


def _camera_from_colmap(img, cam, device: str, out_w: int, out_h: int):
    model = str(cam.model)
    p = cam.params
    if model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL", "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE"):
        f, cx, cy = float(p[0]), float(p[1]), float(p[2])
        fx, fy = f, f
    elif model in ("PINHOLE", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV", "THIN_PRISM_FISHEYE", "FOV"):
        fx, fy, cx, cy = float(p[0]), float(p[1]), float(p[2]), float(p[3])
    else:
        raise RuntimeError(f"Unsupported camera model: {model}")

    # Scale intrinsics to output resolution.
    sx = out_w / float(cam.width)
    sy = out_h / float(cam.height)
    fx *= sx
    fy *= sy
    cx *= sx
    cy *= sy

    R = qvec2rotmat(img.qvec).astype(np.float32)
    t = img.tvec.astype(np.float32).reshape(3, 1)
    w2c = np.concatenate(
        [np.concatenate([R, t], axis=1), np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32)],
        axis=0,
    )
    c2w = np.linalg.inv(w2c).astype(np.float32)

    c2w_t = torch.from_numpy(c2w).float().unsqueeze(0).to(device)
    K = torch.tensor(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)
    return c2w_t, K


def _resolve_gt_path(dataset: Path, name: str):
    p8 = dataset / "images_8" / name
    if p8.exists():
        return p8
    p1 = dataset / "images" / name
    if p1.exists():
        return p1
    return None


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description="Render using exact COLMAP nerf_sample poses.")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--num-views", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data-factor", type=int, default=1)
    ap.add_argument("--nerf-samples-data-factor", type=int, default=8)
    ap.add_argument("--save-side-by-side", action="store_true")
    args = ap.parse_args()

    dataset = Path(args.dataset)
    sparse = dataset / "sparse" / "0"
    if not sparse.exists():
        sparse = dataset / "sparse"
    cams = read_cameras_binary(sparse / "cameras.bin")
    imgs = read_images_binary(sparse / "images.bin")
    nerf_imgs = [im for im in imgs.values() if os.path.basename(im.name).startswith("nerf_sample_")]
    if len(nerf_imgs) == 0:
        raise RuntimeError("No nerf_sample_* entries found in COLMAP images.bin")

    rng = random.Random(args.seed)
    rng.shuffle(nerf_imgs)
    nerf_imgs = nerf_imgs[: max(1, min(args.num_views, len(nerf_imgs)))]

    out_dir = Path(args.out_dir)
    out_pred = out_dir / "renders"
    out_sbs = out_dir / "side_by_side"
    out_pred.mkdir(parents=True, exist_ok=True)
    if args.save_side_by_side:
        out_sbs.mkdir(parents=True, exist_ok=True)

    runner = _build_runner(
        data_dir=str(dataset),
        result_dir=str(out_dir / "_tmp_runner"),
        data_factor=args.data_factor,
        nerf_samples_data_factor=args.nerf_samples_data_factor,
    )
    ckpt_step = _load_ckpt(runner, args.ckpt)
    print(f"[render-colmap-nerf] ckpt_step={ckpt_step}, rendering {len(nerf_imgs)} views")

    for i, im in enumerate(nerf_imgs, start=1):
        cam = cams[im.camera_id]
        gt_path = _resolve_gt_path(dataset, im.name)
        if gt_path is not None:
            gt_u8 = imageio.imread(gt_path)[..., :3]
            out_h, out_w = gt_u8.shape[:2]
        else:
            gt_u8 = None
            out_w, out_h = int(cam.width), int(cam.height)

        c2w_t, K = _camera_from_colmap(im, cam, runner.device, out_w=out_w, out_h=out_h)
        renders, _, _ = runner.rasterize_splats(
            camtoworlds=c2w_t,
            Ks=K,
            width=out_w,
            height=out_h,
            sh_degree=runner.cfg.sh_degree,
            near_plane=runner.cfg.near_plane,
            far_plane=runner.cfg.far_plane,
            render_mode="RGB",
        )
        pred = renders[..., 0:3] if renders.shape[-1] == 4 else renders
        pred_u8 = _to_uint8(pred[0].detach().cpu().numpy())

        stem = Path(im.name).stem
        imageio.imwrite(out_pred / f"{stem}_pred.png", pred_u8)
        if args.save_side_by_side and gt_u8 is not None and gt_u8.shape[:2] == pred_u8.shape[:2]:
            imageio.imwrite(out_sbs / f"{stem}_gt_pred.png", np.concatenate([gt_u8, pred_u8], axis=1))

        if i % 10 == 0 or i == len(nerf_imgs):
            print(f"[render-colmap-nerf] {i}/{len(nerf_imgs)}")

    print("[render-colmap-nerf] done")


if __name__ == "__main__":
    main()
