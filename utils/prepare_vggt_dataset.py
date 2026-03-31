#!/usr/bin/env python3
"""Build a COLMAP-style dataset from a folder of images using VGGT.

Expected input:
  <src>/
    image_000.png
    image_001.png
    ...

or:
  <src>/
    images/
      image_000.png
      image_001.png
      ...

Output:
  <dst>/
    images/
    images_2/
    images_4/
    images_8/
    sparse/
      cameras.bin
      images.bin
      points3D.bin
      points.ply
"""

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
SCALES = (1, 2, 4, 8)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _bootstrap_imports() -> None:
    repo_root = _repo_root()
    vggt_root = repo_root / "submodules" / "vggt"
    if str(vggt_root) not in sys.path:
        sys.path.insert(0, str(vggt_root))


def _list_source_images(src: Path, recursive: bool) -> tuple[Path, list[Path]]:
    candidates = []
    images_dir = src / "images"
    root = images_dir if images_dir.is_dir() else src
    iterator = root.rglob("*") if recursive else root.glob("*")
    for p in iterator:
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            candidates.append(p.relative_to(root))
    candidates = sorted(candidates)
    if not candidates:
        raise RuntimeError(f"No image files found under {root}")
    return root, candidates


def _ensure_empty_or_new(dst: Path, overwrite: bool) -> None:
    if dst.exists():
        if not overwrite:
            raise FileExistsError(f"Destination already exists: {dst}. Use --overwrite.")
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)


def _resized_dimensions(width: int, height: int, factor: int) -> tuple[int, int]:
    return (max(1, round(width / factor)), max(1, round(height / factor)))


def _write_multiscale_images(src_root: Path, rel_paths: list[Path], dst: Path) -> None:
    out_dirs = {1: dst / "images", 2: dst / "images_2", 4: dst / "images_4", 8: dst / "images_8"}
    for out_dir in out_dirs.values():
        out_dir.mkdir(parents=True, exist_ok=True)

    for idx, rel in enumerate(rel_paths, start=1):
        src_path = src_root / rel
        with Image.open(src_path) as img:
            img = img.convert("RGB")
            for factor, out_dir in out_dirs.items():
                out_path = out_dir / rel
                out_path.parent.mkdir(parents=True, exist_ok=True)
                if factor == 1:
                    img.save(out_path)
                else:
                    w, h = _resized_dimensions(img.width, img.height, factor)
                    img.resize((w, h), resample=Image.Resampling.LANCZOS).save(out_path)
        if idx % 25 == 0 or idx == len(rel_paths):
            print(f"[prepare-vggt-dataset] copied {idx}/{len(rel_paths)} images")


def _resize_depth(depth_hw: torch.Tensor, out_h: int, out_w: int) -> torch.Tensor:
    src_h, src_w = depth_hw.shape
    if src_h == out_h and src_w == out_w:
        return depth_hw
    return F.interpolate(
        depth_hw.unsqueeze(0).unsqueeze(0),
        size=(out_h, out_w),
        mode="bilinear",
        align_corners=False,
    )[0, 0]


def _crop_vggt_depth_to_original(depth_hw: torch.Tensor, width: int, height: int) -> torch.Tensor:
    target_size = int(depth_hw.shape[0])
    max_dim = max(width, height)
    left = (max_dim - width) // 2
    top = (max_dim - height) // 2
    scale = float(target_size) / float(max_dim)
    x1 = left * scale
    y1 = top * scale
    x2 = (left + width) * scale
    y2 = (top + height) * scale

    x1i = max(0, min(target_size - 1, int(np.floor(x1))))
    y1i = max(0, min(target_size - 1, int(np.floor(y1))))
    x2i = max(x1i + 1, min(target_size, int(np.ceil(x2))))
    y2i = max(y1i + 1, min(target_size, int(np.ceil(y2))))
    cropped = depth_hw[y1i:y2i, x1i:x2i]
    return _resize_depth(cropped, out_h=height, out_w=width)


def _run_vggt_depth_export(
    scene_dir: Path,
    rel_paths: list[Path],
    output_prefix: str,
) -> None:
    _bootstrap_imports()
    from demo_colmap import run_VGGT
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images_square

    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("VGGT depth export currently expects CUDA.")

    model = VGGT()
    url = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    model.load_state_dict(torch.hub.load_state_dict_from_url(url))
    model.eval()
    model = model.to(device)

    image_paths = [str(scene_dir / "images" / rel) for rel in rel_paths]
    img_load_resolution = 1024
    vggt_fixed_resolution = 518
    images, _ = load_and_preprocess_images_square(image_paths, img_load_resolution)
    images = images.to(device)

    _, _, depth_map, _ = run_VGGT(model, images, dtype, vggt_fixed_resolution)
    depth_map_t = torch.from_numpy(depth_map).float()

    depth_dirs = {
        1: scene_dir / output_prefix,
        2: scene_dir / f"{output_prefix}_2",
        4: scene_dir / f"{output_prefix}_4",
        8: scene_dir / f"{output_prefix}_8",
    }
    for d in depth_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    for idx, rel in enumerate(rel_paths, start=1):
        with Image.open(scene_dir / "images" / rel) as img:
            full_w, full_h = img.size
        depth_full = _crop_vggt_depth_to_original(depth_map_t[idx - 1], full_w, full_h)

        for factor, out_dir in depth_dirs.items():
            img_dir = scene_dir / ("images" if factor == 1 else f"images_{factor}")
            img_path = img_dir / rel
            if not img_path.exists():
                continue
            with Image.open(img_path) as img_scaled:
                out_w, out_h = img_scaled.size
            depth_scaled = _resize_depth(depth_full, out_h=out_h, out_w=out_w)
            out_path = (out_dir / rel).with_suffix(".npy")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(out_path, depth_scaled.cpu().numpy().astype(np.float32))

        if idx % 25 == 0 or idx == len(rel_paths):
            print(f"[prepare-vggt-dataset] wrote depths {idx}/{len(rel_paths)}")


def _run_vggt_reconstruction(
    scene_dir: Path,
    seed: int,
    use_ba: bool,
    shared_camera: bool,
    camera_type: str,
    vis_thresh: float,
    query_frame_num: int,
    max_query_pts: int,
    fine_tracking: bool,
    max_reproj_error: float,
    conf_thres_value: float,
) -> None:
    _bootstrap_imports()
    from demo_colmap import demo_fn

    class Args:
        pass

    args = Args()
    args.scene_dir = str(scene_dir)
    args.seed = seed
    args.use_ba = use_ba
    args.max_reproj_error = max_reproj_error
    args.shared_camera = shared_camera
    args.camera_type = camera_type
    args.vis_thresh = vis_thresh
    args.query_frame_num = query_frame_num
    args.max_query_pts = max_query_pts
    args.fine_tracking = fine_tracking
    args.conf_thres_value = conf_thres_value
    demo_fn(args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a COLMAP-style dataset using VGGT.")
    parser.add_argument("--src", required=True, type=Path, help="Input directory with images or an images/ subfolder.")
    parser.add_argument("--dst", required=True, type=Path, help="Output dataset directory.")
    parser.add_argument("--recursive", action="store_true", help="Recursively scan the input for images.")
    parser.add_argument("--overwrite", action="store_true", help="Delete dst if it already exists.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed passed to VGGT.")
    parser.add_argument(
        "--output-depth-prefix",
        default="depths_vggt",
        help="Depth output folder prefix under dst: prefix[, _2, _4, _8].",
    )
    parser.add_argument("--use-ba", action="store_true", help="Enable bundle adjustment in VGGT COLMAP export.")
    parser.add_argument("--shared-camera", action="store_true", help="Use a shared camera during BA mode.")
    parser.add_argument("--camera-type", default="SIMPLE_PINHOLE", help="COLMAP camera type used in BA mode.")
    parser.add_argument("--vis-thresh", type=float, default=0.2, help="VGGT visibility threshold for BA mode.")
    parser.add_argument("--query-frame-num", type=int, default=8, help="VGGT query frame count for BA mode.")
    parser.add_argument("--max-query-pts", type=int, default=4096, help="VGGT max query points for BA mode.")
    parser.add_argument("--fine-tracking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-reproj-error", type=float, default=8.0, help="BA reprojection threshold.")
    parser.add_argument(
        "--conf-thres-value",
        type=float,
        default=5.0,
        help="Depth confidence threshold in non-BA mode.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    src = args.src.resolve()
    dst = args.dst.resolve()
    _ensure_empty_or_new(dst, overwrite=args.overwrite)

    src_root, rel_paths = _list_source_images(src, recursive=args.recursive)
    print(f"[prepare-vggt-dataset] found {len(rel_paths)} source images under {src_root}")
    _write_multiscale_images(src_root, rel_paths, dst)

    print("[prepare-vggt-dataset] running VGGT COLMAP export")
    _run_vggt_reconstruction(
        scene_dir=dst,
        seed=args.seed,
        use_ba=args.use_ba,
        shared_camera=args.shared_camera,
        camera_type=args.camera_type,
        vis_thresh=args.vis_thresh,
        query_frame_num=args.query_frame_num,
        max_query_pts=args.max_query_pts,
        fine_tracking=args.fine_tracking,
        max_reproj_error=args.max_reproj_error,
        conf_thres_value=args.conf_thres_value,
    )
    print("[prepare-vggt-dataset] exporting VGGT depth maps")
    _run_vggt_depth_export(
        scene_dir=dst,
        rel_paths=rel_paths,
        output_prefix=args.output_depth_prefix,
    )
    print(f"[prepare-vggt-dataset] done: {dst}")


if __name__ == "__main__":
    main()
