import argparse
import shutil
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image
import torch

try:
    from RadSplat.nerfstep.nerf_models import Nerfacto
except ModuleNotFoundError:
    from nerfstep.nerf_models import Nerfacto


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
SCALE_FOLDERS = {1: "images", 2: "images_2", 4: "images_4", 8: "images_8"}
SUPPORTED_METHODS = {"nerfacto", "mipnerf"}


def copy_dataset(input_dataset: Path, output_dataset: Path) -> None:
    if not input_dataset.exists():
        raise FileNotFoundError(f"Input dataset does not exist: {input_dataset}")
    if output_dataset.exists():
        raise FileExistsError(f"Output dataset already exists: {output_dataset}")
    shutil.copytree(input_dataset, output_dataset)


def clear_images(root: Path) -> None:
    for folder_name in SCALE_FOLDERS.values():
        folder = root / folder_name
        if not folder.exists():
            continue
        for p in folder.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                p.unlink()


def rel_to_images_root(p: Path) -> Tuple[str, Path]:
    parts = p.parts
    for i, part in enumerate(parts):
        if part == "images" or part.startswith("images_"):
            tail = Path(*parts[i + 1 :]) if i + 1 < len(parts) else Path(p.name)
            return part, tail
    return "images", Path(p.name)


def tensor_rgb_to_pil(rgb: torch.Tensor) -> Image.Image:
    rgb = rgb.detach().float().cpu().clamp(0.0, 1.0).numpy()
    arr = (rgb * 255.0).astype(np.uint8)
    return Image.fromarray(arr)


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


def detect_base_scale(render_size: Tuple[int, int], input_dataset: Path) -> int:
    scale_sizes = dataset_scale_sizes(input_dataset)
    if not scale_sizes:
        raise RuntimeError("Could not find any image files in input dataset scale folders.")

    exact = [s for s, sz in scale_sizes.items() if sz == render_size]
    if exact:
        return min(exact)

    # Fallback: allow a tiny tolerance (e.g., odd-size rounding differences).
    close = [
        (s, abs(sz[0] - render_size[0]) + abs(sz[1] - render_size[1]))
        for s, sz in scale_sizes.items()
        if abs(sz[0] - render_size[0]) <= 1 and abs(sz[1] - render_size[1]) <= 1
    ]
    if close:
        close.sort(key=lambda x: x[1])
        return close[0][0]

    raise RuntimeError(
        f"Rendered size {render_size} does not match any input scale sizes: {scale_sizes}"
    )


def keep_scale_folders(output_dataset: Path, keep_scales) -> None:
    keep_names = {SCALE_FOLDERS[s] for s in keep_scales}
    for folder_name in SCALE_FOLDERS.values():
        folder = output_dataset / folder_name
        if folder_name in keep_names:
            folder.mkdir(parents=True, exist_ok=True)
        elif folder.exists():
            shutil.rmtree(folder)


def resolve_config_and_method(nerf_path: Path) -> Tuple[Path, str]:
    if nerf_path.is_dir():
        config_path = nerf_path / "config.yml"
    else:
        config_path = nerf_path

    if not config_path.exists():
        raise FileNotFoundError(f"Could not find config.yml at: {config_path}")
    if config_path.suffix.lower() not in {".yml", ".yaml"}:
        raise ValueError(f"Nerf path must point to a config .yml/.yaml file or run directory: {nerf_path}")

    method = None

    # Standard nerfstudio layout: outputs/<exp>/<method>/<timestamp>/config.yml
    try:
        candidate = config_path.parents[1].name
        if candidate in SUPPORTED_METHODS:
            method = candidate
    except IndexError:
        pass

    # Fallback search in full path.
    if method is None:
        for part in config_path.parts:
            if part in SUPPORTED_METHODS:
                method = part
                break

    if method is None:
        raise ValueError(
            f"Could not infer method from path: {config_path}. "
            f"Expected one of {sorted(SUPPORTED_METHODS)} in the path."
        )

    return config_path, method


def save_multiscale(
    img: Image.Image, out_root: Path, rel_tail: Path, base_scale: int, keep_scales
) -> None:
    for scale in keep_scales:
        folder_name = SCALE_FOLDERS[scale]
        out_path = out_root / folder_name / rel_tail
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if scale == base_scale:
            img_scale = img
        else:
            ratio = scale // base_scale
            w = max(1, img.width // ratio)
            h = max(1, img.height // ratio)
            img_scale = img.resize((w, h), resample=Image.Resampling.LANCZOS)
        ext = out_path.suffix.lower()
        if ext in {".jpg", ".jpeg"}:
            img_scale.save(out_path, quality=100, subsampling=0)
        else:
            img_scale.save(out_path)


def regenerate_images(nerf_path: Path, input_dataset: Path, output_dataset: Path) -> None:
    config_path, method = resolve_config_and_method(nerf_path)
    print(f"Loading NeRF method '{method}' from config: {config_path}")
    model = Nerfacto(str(config_path))
    datamanager = model.pipeline.datamanager

    # Regenerate only training-set images. Validation/test images are kept as-is
    # from the copied original dataset.
    entries = []
    split = "train"
    outputs = datamanager.dataparser.get_dataparser_outputs(split=split)
    image_filenames = outputs.image_filenames
    cameras = outputs.cameras.to(model.device)
    if cameras.shape[0] != len(image_filenames):
        raise RuntimeError(
            f"Mismatch in split '{split}': cameras={cameras.shape[0]} filenames={len(image_filenames)}"
        )
    for i, filename in enumerate(image_filenames):
        entries.append((filename, cameras[i : i + 1]))

    if len(entries) == 0:
        raise RuntimeError("No image filenames found in dataparser outputs for split='train'.")

    model.pipeline.model.eval()
    with torch.no_grad():
        base_scale = None
        keep_scales = None

        for filename, camera in entries:
            outputs = model.pipeline.model.get_outputs_for_camera(camera)
            rgb = outputs["rgb"]
            if rgb.ndim == 4:
                rgb = rgb[0]
            pil_img = tensor_rgb_to_pil(rgb)
            _, rel_tail = rel_to_images_root(Path(str(filename)))

            if base_scale is None:
                base_scale = detect_base_scale((pil_img.width, pil_img.height), input_dataset)
                keep_scales = [s for s in sorted(SCALE_FOLDERS) if s >= base_scale]
                keep_scale_folders(output_dataset, keep_scales)
                print(
                    f"Detected NeRF training scale: images_{base_scale if base_scale > 1 else 1} "
                    f"(render size {pil_img.width}x{pil_img.height}). Writing scales: "
                    f"{[SCALE_FOLDERS[s] for s in keep_scales]}"
                )

            save_multiscale(pil_img, output_dataset, rel_tail, base_scale, keep_scales)


def parse_args():
    parser = argparse.ArgumentParser(description="Regenerate dataset images from a trained NeRF model.")
    parser.add_argument("--input-dataset", required=True, type=Path)
    parser.add_argument("--output-dataset", required=True, type=Path)
    parser.add_argument("--nerf-folder", required=True, type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    copy_dataset(args.input_dataset, args.output_dataset)
    regenerate_images(args.nerf_folder, args.input_dataset, args.output_dataset)
    print(f"Regenerated dataset saved to: {args.output_dataset}")


if __name__ == "__main__":
    main()
