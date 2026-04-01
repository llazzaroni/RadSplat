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
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import trimesh


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
SCALES = (1, 2, 4, 8)


def _log(msg: str) -> None:
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[prepare-vggt-dataset {now}] {msg}", flush=True)


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
            _log(f"copied {idx}/{len(rel_paths)} images")


def _ensure_depth_hw(depth: torch.Tensor) -> torch.Tensor:
    if depth.ndim == 2:
        return depth
    if depth.ndim == 3 and depth.shape[-1] == 1:
        return depth[..., 0]
    if depth.ndim == 3 and depth.shape[0] == 1:
        return depth[0]
    raise ValueError(f"expected depth with shape (H, W), (H, W, 1), or (1, H, W); got {tuple(depth.shape)}")


def _resize_depth(depth_hw: torch.Tensor, out_h: int, out_w: int) -> torch.Tensor:
    depth_hw = _ensure_depth_hw(depth_hw)
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
    depth_hw = _ensure_depth_hw(depth_hw)
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
    img_load_resolution: int,
    vggt_resolution: int,
) -> None:
    _bootstrap_imports()
    _log("imported VGGT depth-export modules")
    from demo_colmap import run_VGGT
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images_square

    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("VGGT depth export currently expects CUDA.")
    _log(f"starting depth export on device={device} dtype={dtype}")

    model = VGGT()
    url = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    _log("loading VGGT weights for depth export")
    model.load_state_dict(torch.hub.load_state_dict_from_url(url))
    model.eval()
    model = model.to(device)
    _log("VGGT weights loaded for depth export")

    image_paths = [str(scene_dir / "images" / rel) for rel in rel_paths]
    vggt_fixed_resolution = int(vggt_resolution)
    _log(f"loading and preprocessing {len(image_paths)} images for depth export")
    images, _ = load_and_preprocess_images_square(image_paths, img_load_resolution)
    images = images.to(device)
    _log("running VGGT forward pass for depth export")

    _, _, depth_map, _ = run_VGGT(model, images, dtype, vggt_fixed_resolution)
    depth_map_t = torch.from_numpy(depth_map).float()
    _log("VGGT forward pass completed for depth export")

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
            _log(f"wrote depths {idx}/{len(rel_paths)}")


def _run_vggt_pointcloud_export(
    scene_dir: Path,
    rel_paths: list[Path] | None,
    img_load_resolution: int,
    vggt_resolution: int,
    conf_thres_value: float,
    max_points: int,
) -> None:
    _bootstrap_imports()
    _log("imported VGGT pointcloud-export modules")
    from demo_colmap import run_VGGT
    from vggt.models.vggt import VGGT
    from vggt.utils.geometry import unproject_depth_map_to_point_map
    from vggt.utils.helper import randomly_limit_trues
    from vggt.utils.load_fn import load_and_preprocess_images_square

    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("VGGT pointcloud export currently expects CUDA.")
    _log(f"starting pointcloud export on device={device} dtype={dtype}")

    model = VGGT()
    url = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    _log("loading VGGT weights for pointcloud export")
    model.load_state_dict(torch.hub.load_state_dict_from_url(url))
    model.eval()
    model = model.to(device)
    _log("VGGT weights loaded for pointcloud export")

    image_dir = scene_dir / "images"
    if rel_paths is None:
        chosen_rel_paths = sorted(p.relative_to(image_dir) for p in image_dir.rglob("*") if p.is_file())
    else:
        chosen_rel_paths = list(rel_paths)
    image_paths = [str(image_dir / rel) for rel in chosen_rel_paths]
    if not image_paths:
        raise RuntimeError(f"No images found under {image_dir}")

    _log(f"loading and preprocessing {len(image_paths)} images for pointcloud export")
    images, _ = load_and_preprocess_images_square(image_paths, img_load_resolution)
    images = images.to(device)
    _log("running VGGT forward pass for pointcloud export")

    extrinsic, intrinsic, depth_map, depth_conf = run_VGGT(model, images, dtype, int(vggt_resolution))
    points_3d = unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic)
    _log("VGGT forward pass completed for pointcloud export")

    points_rgb = F.interpolate(
        images, size=(int(vggt_resolution), int(vggt_resolution)), mode="bilinear", align_corners=False
    )
    points_rgb = (points_rgb.cpu().numpy() * 255).astype(np.uint8).transpose(0, 2, 3, 1)

    conf_mask = depth_conf >= conf_thres_value
    if max_points > 0:
        conf_mask = randomly_limit_trues(conf_mask, max_points)

    points_3d = points_3d[conf_mask]
    points_rgb = points_rgb[conf_mask]
    if len(points_3d) == 0:
        raise RuntimeError("VGGT pointcloud export produced zero points after confidence filtering.")

    sparse_dir = scene_dir / "sparse" / "0"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    out_path = sparse_dir / "points.ply"
    _log(f"writing point cloud with {len(points_3d)} points to {out_path}")
    trimesh.PointCloud(points_3d, colors=points_rgb).export(out_path)
    _log("pointcloud export completed")


def _run_vggt_reconstruction(
    scene_dir: Path,
    seed: int,
    img_load_resolution: int,
    vggt_resolution: int,
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
    _log("importing VGGT reconstruction entrypoint")
    from demo_colmap import demo_fn
    _log("VGGT reconstruction entrypoint imported")

    class Args:
        pass

    args = Args()
    args.scene_dir = str(scene_dir)
    args.seed = seed
    args.img_load_resolution = int(img_load_resolution)
    args.vggt_resolution = int(vggt_resolution)
    args.use_ba = use_ba
    args.max_reproj_error = max_reproj_error
    args.shared_camera = shared_camera
    args.camera_type = camera_type
    args.vis_thresh = vis_thresh
    args.query_frame_num = query_frame_num
    args.max_query_pts = max_query_pts
    args.fine_tracking = fine_tracking
    args.conf_thres_value = conf_thres_value
    _log("starting VGGT reconstruction")
    demo_fn(args)
    _log("VGGT reconstruction completed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a COLMAP-style dataset using VGGT.")
    parser.add_argument("--src", required=True, type=Path, help="Input directory with images or an images/ subfolder.")
    parser.add_argument("--dst", required=True, type=Path, help="Output dataset directory.")
    parser.add_argument("--recursive", action="store_true", help="Recursively scan the input for images.")
    parser.add_argument("--overwrite", action="store_true", help="Delete dst if it already exists.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed passed to VGGT.")
    parser.add_argument(
        "--img-load-resolution",
        type=int,
        default=1024,
        help="Square resolution used when loading/padding images before VGGT.",
    )
    parser.add_argument(
        "--vggt-resolution",
        type=int,
        default=518,
        help="Internal square resolution used for the VGGT forward pass.",
    )
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
    parser.add_argument(
        "--points-only",
        action="store_true",
        help="Write only sparse/0/points.ply from the feed-forward VGGT depth output and skip COLMAP/depth export.",
    )
    parser.add_argument(
        "--max-pointcloud-points",
        type=int,
        default=100000,
        help="Maximum number of points to keep when --points-only is enabled (<=0 keeps all).",
    )
    parser.add_argument(
        "--pointcloud-subset-size",
        type=int,
        default=0,
        help="If >0, sample this many images for the pointcloud pass while keeping full-scene reconstruction outputs.",
    )
    parser.add_argument(
        "--pointcloud-seed",
        type=int,
        default=42,
        help="Random seed used when sampling a pointcloud subset.",
    )
    parser.add_argument(
        "--skip-depth-export",
        action="store_true",
        help="Skip depth map export after reconstruction.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    src = args.src.resolve()
    dst = args.dst.resolve()
    _log(f"src={src}")
    _log(f"dst={dst}")
    _ensure_empty_or_new(dst, overwrite=args.overwrite)

    src_root, rel_paths = _list_source_images(src, recursive=args.recursive)
    _log(f"found {len(rel_paths)} source images under {src_root}")
    _write_multiscale_images(src_root, rel_paths, dst)

    pointcloud_rel_paths = rel_paths
    if args.pointcloud_subset_size > 0:
        if args.pointcloud_subset_size > len(rel_paths):
            raise ValueError(
                f"--pointcloud-subset-size={args.pointcloud_subset_size} exceeds available image count {len(rel_paths)}"
            )
        rng = random.Random(args.pointcloud_seed)
        pointcloud_rel_paths = sorted(rng.sample(rel_paths, args.pointcloud_subset_size))
        _log(
            f"sampled {len(pointcloud_rel_paths)}/{len(rel_paths)} images for pointcloud export "
            f"(seed={args.pointcloud_seed})"
        )

    if args.points_only:
        _log("exporting VGGT point cloud only")
        _run_vggt_pointcloud_export(
            scene_dir=dst,
            rel_paths=pointcloud_rel_paths,
            img_load_resolution=args.img_load_resolution,
            vggt_resolution=args.vggt_resolution,
            conf_thres_value=args.conf_thres_value,
            max_points=args.max_pointcloud_points,
        )
    else:
        _log("running VGGT COLMAP export")
        _run_vggt_reconstruction(
            scene_dir=dst,
            seed=args.seed,
            img_load_resolution=args.img_load_resolution,
            vggt_resolution=args.vggt_resolution,
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
        if args.pointcloud_subset_size > 0:
            _log("exporting sampled VGGT point cloud")
            _run_vggt_pointcloud_export(
                scene_dir=dst,
                rel_paths=pointcloud_rel_paths,
                img_load_resolution=args.img_load_resolution,
                vggt_resolution=args.vggt_resolution,
                conf_thres_value=args.conf_thres_value,
                max_points=args.max_pointcloud_points,
            )
        if not args.skip_depth_export:
            _log("exporting VGGT depth maps")
            _run_vggt_depth_export(
                scene_dir=dst,
                rel_paths=rel_paths,
                output_prefix=args.output_depth_prefix,
                img_load_resolution=args.img_load_resolution,
                vggt_resolution=args.vggt_resolution,
            )
    _log(f"done: {dst}")


if __name__ == "__main__":
    main()
