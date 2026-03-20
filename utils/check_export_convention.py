#!/usr/bin/env python3
"""Check whether opencv or opengl is correct for export back to COLMAP."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from nerfstep.nerf_models import Nerfacto
from submodules.nerfstudio.nerfstudio.data.utils.colmap_parsing_utils import read_images_binary, qvec2rotmat


def c2w_from_colmap(qvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    r_w2c = qvec2rotmat(qvec).astype(np.float64)
    t_w2c = tvec.reshape(3, 1).astype(np.float64)
    w2c = np.eye(4, dtype=np.float64)
    w2c[:3, :3] = r_w2c
    w2c[:3, 3:] = t_w2c
    return np.linalg.inv(w2c)[:3, :]


def rot_deg(a: np.ndarray, b: np.ndarray) -> float:
    ra = a[:3, :3]
    rb = b[:3, :3]
    r = ra @ rb.T
    tr = np.clip((np.trace(r) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(tr)))


def trans_err(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a[:3, 3] - b[:3, 3]))


def summarize(name: str, r: list[float], t: list[float]) -> None:
    r = np.asarray(r, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    print(
        f"{name}: n={len(r)} "
        f"rot_deg[p50/p90/max]={np.percentile(r,50):.4f}/{np.percentile(r,90):.4f}/{np.max(r):.4f} "
        f"trans[p50/p90/max]={np.percentile(t,50):.6f}/{np.percentile(t,90):.6f}/{np.max(t):.6f}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare opencv vs opengl export convention against COLMAP.")
    ap.add_argument("--nerf-config", required=True, help="Path to nerfstudio config.yml")
    ap.add_argument("--dataset", required=True, help="Dataset root (for sparse/0/images.bin)")
    ap.add_argument("--max-images", type=int, default=1000, help="Max train images to compare")
    args = ap.parse_args()

    model = Nerfacto(args.nerf_config)
    dpo = model.pipeline.datamanager.train_dataparser_outputs
    cams = model.pipeline.datamanager.train_dataset.cameras.to("cpu")

    sparse = Path(args.dataset) / "sparse" / "0"
    if not sparse.exists():
        sparse = Path(args.dataset) / "sparse"
    images = read_images_binary(sparse / "images.bin")
    by_name = {Path(v.name).name: v for v in images.values()}

    fnames = [Path(p).name for p in dpo.image_filenames]
    n = min(len(fnames), args.max_images)

    rot_cv, tr_cv = [], []
    rot_gl, tr_gl = [], []
    missing = 0

    poses = cams.camera_to_worlds[:n].clone()
    poses_cv = dpo.transform_poses_to_original_space(poses, camera_convention="opencv").numpy()
    poses_gl = dpo.transform_poses_to_original_space(poses, camera_convention="opengl").numpy()

    for i in range(n):
        name = fnames[i]
        if name not in by_name:
            missing += 1
            continue
        im = by_name[name]
        ref = c2w_from_colmap(im.qvec, im.tvec)
        pcv = poses_cv[i]
        pgl = poses_gl[i]
        rot_cv.append(rot_deg(pcv, ref))
        tr_cv.append(trans_err(pcv, ref))
        rot_gl.append(rot_deg(pgl, ref))
        tr_gl.append(trans_err(pgl, ref))

    print(f"checked={len(rot_cv)} missing={missing}")
    if len(rot_cv) == 0:
        print("No overlapping image names found.")
        return
    summarize("opencv", rot_cv, tr_cv)
    summarize("opengl", rot_gl, tr_gl)
    better = "opencv" if np.median(rot_cv) + np.median(tr_cv) < np.median(rot_gl) + np.median(tr_gl) else "opengl"
    print(f"best_convention={better}")


if __name__ == "__main__":
    main()
