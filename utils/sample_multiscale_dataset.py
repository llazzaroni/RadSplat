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

SCALES = ("images", "images_2", "images_4", "images_8")


def list_relative_files(root: Path) -> List[Path]:
    return sorted([p.relative_to(root) for p in root.rglob("*") if p.is_file()])


def validate_source(src: Path) -> None:
    for folder in SCALES:
        p = src / folder
        if not p.exists() or not p.is_dir():
            raise ValueError(f"Missing required folder: {p}")


def validate_destination(dst: Path) -> None:
    if dst.exists():
        if not dst.is_dir():
            raise ValueError(f"Destination exists and is not a directory: {dst}")
        if any(dst.iterdir()):
            raise ValueError(f"Destination directory must be empty: {dst}")
    else:
        dst.mkdir(parents=True, exist_ok=True)


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

    print(f"Sampled {n} files from {src / 'images'}")
    print(f"Wrote image-only subset to {dst}")


if __name__ == "__main__":
    main()
