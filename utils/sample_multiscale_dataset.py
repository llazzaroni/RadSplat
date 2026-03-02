#!/usr/bin/env python3
"""Sample a subset of images across multiscale folders.

Expected source structure:
  <src>/
    images/
    images_2/
    images_4/
    images_8/

The script samples N files from <src>/images and copies matching relative paths
from all four folders into <dst>/images*, preserving relative paths.
"""

import argparse
import random
import shutil
from pathlib import Path
from typing import List

import numpy as np

try:
    from RadSplat.submodules.nerfstudio.nerfstudio.data.utils import colmap_parsing_utils as colmap
except ModuleNotFoundError:
    from submodules.nerfstudio.nerfstudio.data.utils import colmap_parsing_utils as colmap


SCALES = ("images", "images_2", "images_4", "images_8")


def list_relative_files(root: Path) -> List[Path]:
    return sorted([p.relative_to(root) for p in root.rglob("*") if p.is_file()])


def validate_source(src: Path) -> None:
    for folder in SCALES:
        p = src / folder
        if not p.exists() or not p.is_dir():
            raise ValueError(f"Missing required folder: {p}")
    sparse_0 = src / "sparse" / "0"
    sparse_root = src / "sparse"
    if not sparse_0.exists() and not sparse_root.exists():
        raise ValueError(f"Missing required COLMAP folder: {src / 'sparse'}")


def validate_destination(dst: Path) -> None:
    if dst.exists():
        if not dst.is_dir():
            raise ValueError(f"Destination exists and is not a directory: {dst}")
        if any(dst.iterdir()):
            raise ValueError(f"Destination directory must be empty: {dst}")
    else:
        dst.mkdir(parents=True, exist_ok=True)


def detect_sparse_dir(src: Path) -> Path:
    sparse_0 = src / "sparse" / "0"
    if sparse_0.exists():
        return sparse_0
    sparse_root = src / "sparse"
    if sparse_root.exists():
        return sparse_root
    raise ValueError(f"Could not find sparse model under {src / 'sparse'}")


def filter_and_write_sparse(src: Path, dst: Path, sampled_rel: List[Path]) -> None:
    sparse_in = detect_sparse_dir(src)
    cameras, images, points3D = colmap.read_model(str(sparse_in), ext=".bin")

    sampled_names = {p.as_posix() for p in sampled_rel}
    kept_images = {iid: img for iid, img in images.items() if img.name in sampled_names}
    kept_image_ids = set(kept_images.keys())

    if len(kept_images) == 0:
        raise RuntimeError("No sampled filenames matched entries in COLMAP images.bin.")

    # Keep only points observed in at least one selected image.
    kept_points = {}
    for pid, pt in points3D.items():
        mask = np.isin(pt.image_ids, list(kept_image_ids))
        if not np.any(mask):
            continue
        kept_points[pid] = colmap.Point3D(
            id=pt.id,
            xyz=pt.xyz,
            rgb=pt.rgb,
            error=pt.error,
            image_ids=pt.image_ids[mask],
            point2D_idxs=pt.point2D_idxs[mask],
        )

    kept_point_ids = set(kept_points.keys())

    # Keep all keypoints for selected images, but drop links to removed points.
    final_images = {}
    for iid, img in kept_images.items():
        point3d_ids = img.point3D_ids.copy()
        invalid_mask = (~np.isin(point3d_ids, list(kept_point_ids))) & (point3d_ids != -1)
        point3d_ids[invalid_mask] = -1
        final_images[iid] = colmap.Image(
            id=img.id,
            qvec=img.qvec,
            tvec=img.tvec,
            camera_id=img.camera_id,
            name=img.name,
            xys=img.xys,
            point3D_ids=point3d_ids,
        )

    sparse_out = dst / "sparse" / "0"
    sparse_out.mkdir(parents=True, exist_ok=True)
    colmap.write_model(cameras, final_images, kept_points, str(sparse_out), ext=".bin")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Randomly sample N images and copy matching files across images, images_2, images_4, images_8."
    )
    parser.add_argument("--src", required=True, type=Path, help="Source dataset directory")
    parser.add_argument("--dst", required=True, type=Path, help="Destination empty directory")
    parser.add_argument("--n", required=True, type=int, help="Number of images to sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    src = args.src.resolve()
    dst = args.dst.resolve()
    n = args.n

    if n <= 0:
        raise ValueError("--n must be > 0")

    validate_source(src)
    validate_destination(dst)

    base_files = list_relative_files(src / "images")
    if n > len(base_files):
        raise ValueError(f"Requested n={n} but only {len(base_files)} files found in {src / 'images'}")

    rng = random.Random(args.seed)
    sampled = sorted(rng.sample(base_files, n))

    # Ensure sampled paths exist in all scale folders before copying.
    for rel in sampled:
        for folder in SCALES:
            candidate = src / folder / rel
            if not candidate.is_file():
                raise FileNotFoundError(f"Missing matching file in {folder}: {candidate}")

    # Copy sampled files.
    for folder in SCALES:
        (dst / folder).mkdir(parents=True, exist_ok=True)

    for rel in sampled:
        for folder in SCALES:
            src_file = src / folder / rel
            dst_file = dst / folder / rel
            dst_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_file, dst_file)

    # Keep useful metadata if present.
    poses_bounds = src / "poses_bounds.npy"
    if poses_bounds.exists():
        shutil.copy2(poses_bounds, dst / "poses_bounds.npy")

    filter_and_write_sparse(src, dst, sampled)

    print(f"Sampled {n} files from {src / 'images'}")
    print(f"Wrote subset to {dst}")


if __name__ == "__main__":
    main()
