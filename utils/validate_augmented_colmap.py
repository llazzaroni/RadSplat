#!/usr/bin/env python3
"""Validate nerf_sample_* camera consistency inside a COLMAP model."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from submodules.nerfstudio.nerfstudio.data.utils.colmap_parsing_utils import (
    qvec2rotmat,
    read_cameras_binary,
    read_images_binary,
)


def camera_center_and_forward(qvec: np.ndarray, tvec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return camera center C and forward vector in world coordinates."""
    r_w2c = qvec2rotmat(qvec).astype(np.float64)  # world->camera
    t_w2c = tvec.astype(np.float64).reshape(3, 1)
    r_c2w = r_w2c.T
    c = (-r_c2w @ t_w2c).reshape(3)
    fwd = r_c2w[:, 2]  # +Z forward in COLMAP/OpenCV camera frame
    fwd /= np.linalg.norm(fwd) + 1e-12
    return c, fwd


def percentile(arr: np.ndarray, p: float) -> float:
    if arr.size == 0:
        return float("nan")
    return float(np.percentile(arr, p))


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate nerf_sample_* pose/intrinsics consistency.")
    ap.add_argument("--dataset", required=True, help="Dataset root with sparse/0 and images folders.")
    ap.add_argument(
        "--factor",
        type=int,
        default=8,
        help="Image factor folder to verify dimensions against (e.g., 8 -> images_8).",
    )
    ap.add_argument(
        "--prefix",
        default="nerf_sample_",
        help="Synthetic image name prefix.",
    )
    args = ap.parse_args()

    root = Path(args.dataset)
    sparse = root / "sparse" / "0"
    if not sparse.exists():
        sparse = root / "sparse"
    cams = read_cameras_binary(sparse / "cameras.bin")
    imgs = read_images_binary(sparse / "images.bin")

    all_items = list(imgs.values())
    syn = [im for im in all_items if Path(im.name).name.startswith(args.prefix)]
    real = [im for im in all_items if not Path(im.name).name.startswith(args.prefix)]

    print(f"images total={len(all_items)} real={len(real)} synthetic={len(syn)}")
    if len(syn) == 0:
        print("No synthetic images found.")
        return

    # Camera ID usage
    cam_hist = {}
    for im in syn:
        cam_hist[im.camera_id] = cam_hist.get(im.camera_id, 0) + 1
    print("synthetic camera_id histogram:", cam_hist)

    # Dimension checks at requested factor
    image_dir = root / ("images" if args.factor == 1 else f"images_{args.factor}")
    if not image_dir.exists():
        raise RuntimeError(f"Missing image folder: {image_dir}")

    missing = 0
    bad_size = 0
    for im in syn:
        p = image_dir / im.name
        if not p.exists():
            missing += 1
            continue
        h, w = imageio.imread(p)[..., :3].shape[:2]
        cam = cams[im.camera_id]
        exp_w = int(round(cam.width / args.factor))
        exp_h = int(round(cam.height / args.factor))
        if (w, h) != (exp_w, exp_h):
            bad_size += 1
    print(f"synthetic image files missing={missing} size_mismatch={bad_size} (factor={args.factor})")

    # Pose checks vs real cameras
    c_real = []
    f_real = []
    for im in real:
        c, f = camera_center_and_forward(im.qvec, im.tvec)
        c_real.append(c)
        f_real.append(f)
    c_real = np.stack(c_real, axis=0)
    f_real = np.stack(f_real, axis=0)

    c_syn = []
    f_syn = []
    for im in syn:
        c, f = camera_center_and_forward(im.qvec, im.tvec)
        c_syn.append(c)
        f_syn.append(f)
    c_syn = np.stack(c_syn, axis=0)
    f_syn = np.stack(f_syn, axis=0)

    # Nearest-real position distance for each synthetic
    dmat = np.linalg.norm(c_syn[:, None, :] - c_real[None, :, :], axis=-1)
    nn_idx = np.argmin(dmat, axis=1)
    nn_dist = dmat[np.arange(len(syn)), nn_idx]

    # Median nearest-neighbor spacing among real cameras (scale reference)
    d_real = np.linalg.norm(c_real[:, None, :] - c_real[None, :, :], axis=-1)
    np.fill_diagonal(d_real, np.inf)
    real_nn = np.min(d_real, axis=1)
    ref = float(np.median(real_nn))
    rel = nn_dist / max(ref, 1e-12)

    print(
        "synthetic->nearest-real distance:"
        f" abs p50={percentile(nn_dist,50):.4f} p90={percentile(nn_dist,90):.4f} max={float(np.max(nn_dist)):.4f}"
    )
    print(
        "distance normalized by real median nn:"
        f" p50={percentile(rel,50):.3f} p90={percentile(rel,90):.3f} max={float(np.max(rel)):.3f}"
    )

    # Orientation mismatch to nearest real camera
    dots = np.sum(f_syn * f_real[nn_idx], axis=1)
    dots = np.clip(dots, -1.0, 1.0)
    ang = np.degrees(np.arccos(dots))
    print(
        "forward-angle vs nearest-real:"
        f" p50={percentile(ang,50):.2f}deg p90={percentile(ang,90):.2f}deg max={float(np.max(ang)):.2f}deg"
    )

    # Bounding box sanity
    mn = c_real.min(axis=0)
    mx = c_real.max(axis=0)
    ext = mx - mn
    pad = 0.1 * ext
    inside = np.all((c_syn >= (mn - pad)) & (c_syn <= (mx + pad)), axis=1)
    outside_ratio = 1.0 - float(np.mean(inside))
    print(f"synthetic centers outside real bbox(+10% margin): {outside_ratio*100:.1f}%")

    # Heuristic warnings
    problems = []
    if bad_size > 0:
        problems.append("camera/image size mismatch for synthetic samples")
    if percentile(rel, 90) > 3.0:
        problems.append("synthetic camera centers far from real camera manifold (possible frame/scale mismatch)")
    if percentile(ang, 90) > 60.0:
        problems.append("synthetic camera orientations differ strongly from nearby real cameras")
    if outside_ratio > 0.4:
        problems.append("many synthetic centers outside real camera region")

    if problems:
        print("WARNING:")
        for p in problems:
            print(f" - {p}")
    else:
        print("OK: no obvious pose/intrinsics consistency issues detected.")


if __name__ == "__main__":
    main()
