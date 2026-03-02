import argparse
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
import torch

try:
    from RadSplat.nerfstep.nerf_models import Nerfacto
except ModuleNotFoundError:
    from nerfstep.nerf_models import Nerfacto


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def copy_dataset(input_dataset: Path, output_dataset: Path) -> None:
    if not input_dataset.exists():
        raise FileNotFoundError(f"Input dataset does not exist: {input_dataset}")
    if output_dataset.exists():
        raise FileExistsError(f"Output dataset already exists: {output_dataset}")
    shutil.copytree(input_dataset, output_dataset)


def clear_images(root: Path) -> None:
    for folder_name in ["images", "images_2", "images_4", "images_8"]:
        folder = root / folder_name
        if not folder.exists():
            continue
        for p in folder.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                p.unlink()


def rel_to_images_root(p: Path) -> Path:
    parts = p.parts
    if "images" in parts:
        i = parts.index("images")
        return Path(*parts[i:])
    return Path("images") / p.name


def tensor_rgb_to_pil(rgb: torch.Tensor) -> Image.Image:
    rgb = rgb.detach().float().cpu().clamp(0.0, 1.0).numpy()
    arr = (rgb * 255.0).astype(np.uint8)
    return Image.fromarray(arr)


def save_multiscale(img: Image.Image, out_root: Path, rel_images_path: Path) -> None:
    out_paths = {
        1: out_root / rel_images_path,
        2: out_root / str(rel_images_path).replace("images/", "images_2/", 1),
        4: out_root / str(rel_images_path).replace("images/", "images_4/", 1),
        8: out_root / str(rel_images_path).replace("images/", "images_8/", 1),
    }

    for scale, out_path in out_paths.items():
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if scale == 1:
            img_scale = img
        else:
            w = max(1, img.width // scale)
            h = max(1, img.height // scale)
            img_scale = img.resize((w, h), resample=Image.Resampling.LANCZOS)
        img_scale.save(out_path)


def regenerate_images(nerf_folder: Path, output_dataset: Path) -> None:
    model = Nerfacto(str(nerf_folder))
    datamanager = model.pipeline.datamanager

    # Collect entries from both train and eval/test splits to cover all frames.
    entries = []
    seen = set()
    split_names = ["train", getattr(datamanager, "test_split", "test")]
    for split in split_names:
        outputs = datamanager.dataparser.get_dataparser_outputs(split=split)
        image_filenames = outputs.image_filenames
        cameras = outputs.cameras.to(model.device)
        if cameras.shape[0] != len(image_filenames):
            raise RuntimeError(
                f"Mismatch in split '{split}': cameras={cameras.shape[0]} filenames={len(image_filenames)}"
            )
        for i, filename in enumerate(image_filenames):
            key = str(filename)
            if key in seen:
                continue
            seen.add(key)
            entries.append((filename, cameras[i : i + 1]))

    if len(entries) == 0:
        raise RuntimeError("No image filenames found in dataparser outputs (train/eval).")

    model.pipeline.model.eval()
    with torch.no_grad():
        for filename, camera in entries:
            outputs = model.pipeline.model.get_outputs_for_camera(camera)
            rgb = outputs["rgb"]
            if rgb.ndim == 4:
                rgb = rgb[0]
            pil_img = tensor_rgb_to_pil(rgb)
            rel_path = rel_to_images_root(Path(str(filename)))
            save_multiscale(pil_img, output_dataset, rel_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Regenerate dataset images from a trained NeRF model.")
    parser.add_argument("--input-dataset", required=True, type=Path)
    parser.add_argument("--output-dataset", required=True, type=Path)
    parser.add_argument("--nerf-folder", required=True, type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    copy_dataset(args.input_dataset, args.output_dataset)
    clear_images(args.output_dataset)
    regenerate_images(args.nerf_folder, args.output_dataset)
    print(f"Regenerated dataset saved to: {args.output_dataset}")


if __name__ == "__main__":
    main()
