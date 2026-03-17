#!/usr/bin/env python3
"""Generate depth maps for a dataset using a trained Nerfstudio NeRF model."""

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

try:
    from RadSplat.nerfstep.nerf_models import Nerfacto
except ModuleNotFoundError:
    from nerfstep.nerf_models import Nerfacto


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
SCALE_FOLDERS = {1: "images", 2: "images_2", 4: "images_4", 8: "images_8"}


def rel_to_images_root(p: Path) -> Tuple[str, Path]:
    parts = p.parts
    for i, part in enumerate(parts):
        if part == "images" or part.startswith("images_"):
            tail = Path(*parts[i + 1 :]) if i + 1 < len(parts) else Path(p.name)
            return part, tail
    return "images", Path(p.name)


def first_image_size(folder: Path) -> Optional[Tuple[int, int]]:
    if not folder.exists():
        return None
    for p in sorted(folder.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            with Image.open(p) as img:
                return img.size
    return None


def dataset_scale_sizes(dataset_root: Path) -> Dict[int, Tuple[int, int]]:
    sizes = {}
    for scale, folder_name in SCALE_FOLDERS.items():
        size = first_image_size(dataset_root / folder_name)
        if size is not None:
            sizes[scale] = size
    return sizes


def detect_base_scale(render_size: Tuple[int, int], dataset_root: Path) -> int:
    scale_sizes = dataset_scale_sizes(dataset_root)
    if not scale_sizes:
        raise RuntimeError("Could not find image files in dataset scale folders.")

    exact = [s for s, sz in scale_sizes.items() if sz == render_size]
    if exact:
        return min(exact)

    close = [
        (s, abs(sz[0] - render_size[0]) + abs(sz[1] - render_size[1]))
        for s, sz in scale_sizes.items()
        if abs(sz[0] - render_size[0]) <= 1 and abs(sz[1] - render_size[1]) <= 1
    ]
    if close:
        close.sort(key=lambda x: x[1])
        return close[0][0]

    raise RuntimeError(
        f"Rendered size {render_size} does not match dataset scale sizes: {scale_sizes}"
    )


def _select_depth_tensor(outputs: dict, depth_key: str) -> torch.Tensor:
    if depth_key != "auto":
        if depth_key not in outputs:
            raise KeyError(f"Requested depth key '{depth_key}' not in model outputs: {list(outputs.keys())}")
        return outputs[depth_key]

    for k in ("depth", "expected_depth"):
        if k in outputs:
            return outputs[k]
    raise KeyError(f"No depth-like key found in outputs: {list(outputs.keys())}")


def _to_hw_depth(depth: torch.Tensor) -> torch.Tensor:
    d = depth.detach().float()
    # Common shapes: [H,W,1], [1,H,W,1], [H,W], [1,H,W]
    if d.ndim == 4:
        d = d[0]
    if d.ndim == 3 and d.shape[-1] == 1:
        d = d[..., 0]
    elif d.ndim == 3 and d.shape[0] == 1:
        d = d[0]
    if d.ndim != 2:
        raise RuntimeError(f"Unexpected depth tensor shape: {tuple(depth.shape)}")
    return d


def _resize_depth(depth_hw: torch.Tensor, out_h: int, out_w: int) -> torch.Tensor:
    src_h, src_w = depth_hw.shape
    if src_h == out_h and src_w == out_w:
        return depth_hw
    t = depth_hw.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    t = F.interpolate(t, size=(out_h, out_w), mode="bilinear", align_corners=False)
    return t[0, 0]


def _collect_entries(model: Nerfacto, split_mode: str) -> List[Tuple[Path, torch.Tensor]]:
    dataparser = model.pipeline.datamanager.dataparser
    entries: List[Tuple[Path, torch.Tensor]] = []

    split_names: Iterable[str]
    if split_mode == "all":
        split_names = ("train", "test")
    else:
        split_names = (split_mode,)

    seen = set()
    for split in split_names:
        outputs = dataparser.get_dataparser_outputs(split=split)
        image_filenames = outputs.image_filenames
        cameras = outputs.cameras.to(model.device)
        if cameras.shape[0] != len(image_filenames):
            raise RuntimeError(
                f"Mismatch in split '{split}': cameras={cameras.shape[0]} filenames={len(image_filenames)}"
            )
        for i, filename in enumerate(image_filenames):
            rel = rel_to_images_root(Path(str(filename)))[1]
            key = str(rel)
            if key in seen:
                continue
            seen.add(key)
            entries.append((rel, cameras[i : i + 1]))
    return entries


def generate_depth_maps(
    nerf_folder: Path,
    dataset_root: Path,
    output_prefix: str,
    split_mode: str,
    depth_key: str,
    overwrite: bool,
) -> None:
    model = Nerfacto(str(nerf_folder))
    model.pipeline.model.eval()

    entries = _collect_entries(model, split_mode=split_mode)
    if len(entries) == 0:
        raise RuntimeError("No images found in selected split(s).")

    # Render first image to infer base scale.
    with torch.no_grad():
        _, first_cam = entries[0]
        out0 = model.pipeline.model.get_outputs_for_camera(first_cam)
        dep0 = _to_hw_depth(_select_depth_tensor(out0, depth_key))
        base_scale = detect_base_scale((dep0.shape[1], dep0.shape[0]), dataset_root)

    available_scales = sorted(dataset_scale_sizes(dataset_root).keys())
    write_scales = [s for s in available_scales if s >= base_scale]
    if not write_scales:
        raise RuntimeError("No target scales available to write depth maps.")

    out_dirs = {s: dataset_root / f"{output_prefix}{'' if s == 1 else f'_{s}'}" for s in write_scales}
    for d in out_dirs.values():
        if d.exists() and not overwrite:
            raise FileExistsError(f"Output depth directory already exists: {d}. Use --overwrite.")
        d.mkdir(parents=True, exist_ok=True)

    print(
        f"[depth-gen] entries={len(entries)} base_scale={base_scale} "
        f"write_scales={write_scales} depth_key={depth_key}"
    )

    with torch.no_grad():
        for idx, (rel_tail, cam) in enumerate(entries, start=1):
            outputs = model.pipeline.model.get_outputs_for_camera(cam)
            depth_hw = _to_hw_depth(_select_depth_tensor(outputs, depth_key)).to("cpu")

            for scale in write_scales:
                img_folder = SCALE_FOLDERS[scale]
                img_path = dataset_root / img_folder / rel_tail
                if not img_path.exists():
                    # If an image is missing at this scale, skip writing this scale for this frame.
                    continue
                with Image.open(img_path) as im:
                    out_w, out_h = im.size
                depth_scaled = _resize_depth(depth_hw, out_h=out_h, out_w=out_w).numpy().astype(np.float32)

                out_path = out_dirs[scale] / rel_tail
                out_path = out_path.with_suffix(".npy")
                out_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(out_path, depth_scaled)

            if idx % 20 == 0 or idx == len(entries):
                print(f"[depth-gen] rendered {idx}/{len(entries)}")

    print("[depth-gen] done")
    for s in write_scales:
        print(f"[depth-gen] wrote: {out_dirs[s]}")


def parse_args():
    p = argparse.ArgumentParser(description="Generate NeRF depth maps aligned to dataset images.")
    p.add_argument("--nerf-folder", required=True, type=Path, help="Nerfstudio config.yml or run folder.")
    p.add_argument("--dataset", required=True, type=Path, help="Dataset root containing images[/_2/_4/_8].")
    p.add_argument(
        "--output-prefix",
        default="depths_nerf",
        help="Output folder prefix under dataset root: prefix[, _2, _4, _8].",
    )
    p.add_argument(
        "--split",
        choices=["train", "test", "all"],
        default="all",
        help="Which dataparser split(s) to render.",
    )
    p.add_argument(
        "--depth-key",
        default="auto",
        help="Model output key for depth. Use 'auto' (default), 'depth', or 'expected_depth'.",
    )
    p.add_argument("--overwrite", action="store_true", help="Allow writing into existing output folders.")
    return p.parse_args()


def main():
    args = parse_args()
    generate_depth_maps(
        nerf_folder=args.nerf_folder,
        dataset_root=args.dataset,
        output_prefix=args.output_prefix,
        split_mode=args.split,
        depth_key=args.depth_key,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
