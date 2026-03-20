import torch
import numpy as np
import argparse
import os
from contextlib import contextmanager
import cv2
from PIL import Image
import shutil
from pathlib import Path
import matplotlib.pyplot as plt

from nerfstep.nerf_models import Nerfacto
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.utils import colmap_parsing_utils as colmap
from nerfstudio.data.utils.colmap_parsing_utils import rotmat2qvec
from nerfstudio.data.utils.colmap_parsing_utils import Image as ColmapImage

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


@contextmanager
def pushd(new_dir):
    old_dir = os.getcwd()
    os.chdir(new_dir)
    try:
        yield
    finally:
        os.chdir(old_dir)



def rot_x(angle):
    c = torch.cos(angle); s = torch.sin(angle)
    R = torch.zeros(angle.shape[0], 3, 3, device=angle.device)
    R[:, 0, 0] = 1
    R[:, 1, 1] = c; R[:, 1, 2] = -s
    R[:, 2, 1] = s; R[:, 2, 2] = c
    return R

def rot_y(angle):
    c = torch.cos(angle); s = torch.sin(angle)
    R = torch.zeros(angle.shape[0], 3, 3, device=angle.device)
    R[:, 1, 1] = 1
    R[:, 0, 0] = c; R[:, 0, 2] = s
    R[:, 2, 0] = -s; R[:, 2, 2] = c
    return R

def rot_z(angle):
    c = torch.cos(angle); s = torch.sin(angle)
    R = torch.zeros(angle.shape[0], 3, 3, device=angle.device)
    R[:, 2, 2] = 1
    R[:, 0, 0] = c; R[:, 0, 1] = -s
    R[:, 1, 0] = s; R[:, 1, 1] = c
    return R

def sample_small_rotation(k, max_yaw_deg=10.0, max_pitch_deg=6.0, max_roll_deg=2.0, device="cpu"):
    yaw   = (torch.rand(k, device=device) * 2 - 1) * (max_yaw_deg   * torch.pi / 180.0)
    pitch = (torch.rand(k, device=device) * 2 - 1) * (max_pitch_deg * torch.pi / 180.0)
    roll  = (torch.rand(k, device=device) * 2 - 1) * (max_roll_deg  * torch.pi / 180.0)
    dR = rot_y(yaw) @ rot_x(pitch) @ rot_z(roll)
    return dR

def sample_uniform_ball(num, radius):

    # Sample uniformly on the sphere surface
    v = torch.randn(num, 3, device='cpu')
    v = v / (v.norm(dim=-1, keepdim=True) + 1e-9)

    u = torch.rand(num, 1, device='cpu')
    rad = radius * (u ** (1.0 / 3.0))

    # returns a torch tensor in (num, 3)
    return v * rad


def sample_new_cameras_jitter(cam_positions, c2w, n_samples_jitter):

    # First step: compute median neighbor distance
    dists = torch.cdist(cam_positions, cam_positions, p=2)
    dists.fill_diagonal_(float('inf'))
    nn_dists = dists.min(dim=1).values
    median_dist = nn_dists.median()

    translation_radius = 0.2 * median_dist

    # Second step: sample cams within the radii
    offsets = sample_uniform_ball(n_samples_jitter, translation_radius)
    dR = sample_small_rotation(n_samples_jitter)
    
    N = cam_positions.shape[0]
    n_samples_per_cam = n_samples_jitter // N
    rem = n_samples_jitter % N

    positions_list = []
    rotations_list = []

    for i in range(N):
        cam_center = cam_positions[i]
        new_positions = cam_center + offsets[i * n_samples_per_cam:(i + 1) * n_samples_per_cam, :]
        positions_list.append(new_positions)

        rotation = c2w[i, :3, :3]
        dR_slice = dR[i * n_samples_per_cam:(i + 1) * n_samples_per_cam, :, :]
        new_rotation = rotation.unsqueeze(0) @ dR_slice
        rotations_list.append(new_rotation)

    if rem > 0:
        rem_cams = torch.randint(low=0, high=N, size=(rem,))
        positions_list.append(cam_positions[rem_cams] + offsets[-rem:])
        rotations_list.append(c2w[rem_cams, :3, :3] @ dR[-rem:])
    
    final_positions = torch.cat(positions_list, dim=0)
    final_rotations = torch.cat(rotations_list, dim=0)

    c2w_jitter = torch.zeros((n_samples_jitter, 3, 4), dtype=c2w.dtype, device=c2w.device)
    c2w_jitter[:, :3, :3] = final_rotations
    c2w_jitter[:, :3, 3] = final_positions

    return c2w_jitter


def sample_new_cameras_interpolation(cam_positions, c2w, n_samples_interpolation):

    # First, build a kNN graph of the cameras
    dists = torch.cdist(cam_positions, cam_positions)
    knn = dists.topk(k=4, largest=False).indices[:, 1:]

    N = cam_positions.shape[0]

    new_positions = []
    new_orientations = []

    for i in range(N):
        for j in knn[i]:
            ti = cam_positions[i]
            tj = cam_positions[j]

            for alpha in [0.25, 0.5, 0.75]:
                if alpha == 0.5 or alpha == 0.25:
                    R_new = c2w[i, :3, :3]
                else:
                    R_new = c2w[j, :3, :3]
                new_positions.append((1 - alpha) * ti + alpha * tj)
                new_orientations.append(R_new)

    final_positions = torch.stack(new_positions)
    final_orientations = torch.stack(new_orientations)

    c2w_interpolation = torch.zeros((final_positions.shape[0], 3, 4), dtype=c2w.dtype, device=c2w.device)
    c2w_interpolation[:, :3, :3] = final_orientations
    c2w_interpolation[:, :3, 3] = final_positions

    return c2w_interpolation

def render_one_view_uint8(model, c2w_1, scale=0.5):

    cams_all = model.pipeline.datamanager.train_dataset.cameras.to(model.device)
    cams0 = cams_all[0:1]  # Cameras object with 1 camera

    # Clone cams0 into a new Cameras object
    cams0_low = Cameras(
        camera_to_worlds=cams0.camera_to_worlds.clone(),
        fx=cams0.fx.clone(),
        fy=cams0.fy.clone(),
        cx=cams0.cx.clone(),
        cy=cams0.cy.clone(),
        width=cams0.width.clone(),
        height=cams0.height.clone(),
        distortion_params=(
            cams0.distortion_params.clone()
            if cams0.distortion_params is not None
            else None
        ),
        camera_type=cams0.camera_type.clone(),
        times=(cams0.times.clone() if cams0.times is not None else None),
        metadata=cams0.metadata,  # usually fine to share
    ).to(model.device)

    # Go from Nerf factor-2 → GSplat factor-4 ⇒ scale by 0.5
    cams0_low.rescale_output_resolution(scale)

    model.pipeline.model.eval()
    with torch.no_grad():
        c2w_new_batched = c2w_1.unsqueeze(0).to(model.device)  # [1,3,4]

        cam_i = Cameras(
            camera_to_worlds=c2w_new_batched,
            fx=cams0_low.fx,
            fy=cams0_low.fy,
            cx=cams0_low.cx,
            cy=cams0_low.cy,
            width=cams0_low.width,
            height=cams0_low.height,
            distortion_params=cams0_low.distortion_params,
            camera_type=cams0_low.camera_type,
            times=cams0_low.times,
            metadata=cams0_low.metadata,
        ).to(model.device)

        outputs = model.pipeline.model.get_outputs_for_camera(cam_i)
        rgb = outputs["rgb"]

        if rgb.ndim == 4:
            rgb_np = rgb[0].cpu().numpy()
        elif rgb.ndim == 3:
            rgb_np = rgb.cpu().numpy()
        else:
            raise RuntimeError(f"Unexpected rgb ndim: {rgb.ndim}")
        img = (rgb_np * 255).astype(np.uint8)
    return img

def filter_candidates_by_ensemble_variance(
    images_models_low_res,
    new_c2w,
    alpha=10.0,
    tau_mean=0.55,
    tau_pix=0.30,
    kappa=0.70,
    eps=1e-8,
):
    M = len(images_models_low_res)
    assert M > 1, "Need at least 2 models for variance-based filtering."
    N = len(images_models_low_res[0])
    assert all(len(lst) == N for lst in images_models_low_res), "All models must render the same number of candidates."
    assert new_c2w.shape[0] == N, "new_c2w and rendered images disagree on number of candidates."

    # Stack to tensor: (M, N, H, W, 3) in [0,1]
    imgs = torch.stack([
        torch.stack([torch.from_numpy(img) for img in images_models_low_res[m]], dim=0)
        for m in range(M)
    ], dim=0).float() / 255.0

    # Variance across ensemble: (N, H, W, 3)
    var_rgb = imgs.var(dim=0, unbiased=False)

    # Reduce channels -> (N, H, W)
    var_map = var_rgb.mean(dim=-1)

    # Normalize per-image so alpha behaves consistently across scenes
    # (divide by per-image median variance)
    denom = var_map.flatten(1).median(dim=1).values.view(N, 1, 1) + eps
    var_norm = var_map / denom

    # Confidence map in [0,1]: high variance -> low confidence
    conf_map = torch.exp(-alpha * var_norm)  # (N,H,W)

    # Image-level stats
    mean_conf = conf_map.mean(dim=(1, 2))                          # (N,)
    frac_conf = (conf_map > tau_pix).float().mean(dim=(1, 2))      # (N,)
    var_mean = var_map.mean(dim=(1, 2))                            # (N,)

    # Gate
    keep_mask = (mean_conf >= tau_mean) & (frac_conf >= kappa)

    new_c2w_kept = new_c2w[keep_mask]

    stats = {
        "mean_conf": mean_conf,
        "frac_conf": frac_conf,
        "var_mean": var_mean,
    }
    return new_c2w_kept, keep_mask, stats


def filter_candidates_from_var_map(var_map, new_c2w, alpha=10.0, tau_mean=0.18, tau_pix=0.20, kappa=0.20, eps=1e-8):
    N = var_map.shape[0]
    if N == 0:
        keep_mask = torch.zeros((0,), dtype=torch.bool, device=var_map.device)
        return new_c2w[keep_mask], keep_mask, {
            "mean_conf": torch.empty((0,), device=var_map.device),
            "frac_conf": torch.empty((0,), device=var_map.device),
        }

    denom = var_map.flatten(1).median(dim=1).values
    denom = torch.clamp(denom, min=1e-6).view(N,1,1)
    var_norm = var_map / denom
    conf_map = torch.exp(-alpha * var_norm)

    mean_conf = conf_map.mean(dim=(1,2))
    frac_conf = (conf_map > tau_pix).float().mean(dim=(1,2))

    scores = mean_conf.clone()

    #K = min(300, int(0.35 * scores.numel()))
    K = min(500, scores.numel())
    if K <= 0:
        keep_mask = torch.zeros_like(scores, dtype=torch.bool)
        return new_c2w[keep_mask], keep_mask, {"mean_conf": mean_conf, "frac_conf": frac_conf}
    topk = torch.topk(scores, k=K).indices
    keep_mask = torch.zeros_like(scores, dtype=torch.bool)
    keep_mask[topk] = True

    new_c2w_kept = new_c2w[keep_mask]
    return new_c2w_kept, keep_mask, {"mean_conf": mean_conf, "frac_conf": frac_conf}

def create_new_images_low_res_torch(model, c2w, selection_scale=0.125):
    cams_all = model.pipeline.datamanager.train_dataset.cameras.to(model.device)
    cams0 = cams_all[0:1]

    cams0_low = Cameras(
        camera_to_worlds=cams0.camera_to_worlds.clone(),
        fx=cams0.fx.clone(),
        fy=cams0.fy.clone(),
        cx=cams0.cx.clone(),
        cy=cams0.cy.clone(),
        width=cams0.width.clone(),
        height=cams0.height.clone(),
        distortion_params=(cams0.distortion_params.clone() if cams0.distortion_params is not None else None),
        camera_type=cams0.camera_type.clone(),
        times=(cams0.times.clone() if cams0.times is not None else None),
        metadata=cams0.metadata,
    ).to(model.device)

    # Low-res renders only for candidate scoring/selection.
    cams0_low.rescale_output_resolution(selection_scale)

    imgs = []

    model.pipeline.model.eval()
    with torch.no_grad():
        total = c2w.shape[0]
        for i in range(total):
            if i == 0 or (i + 1) % 50 == 0 or i + 1 == total:
                print(f"[selection-render] candidate view {i + 1}/{total}")
            c2w_new_batched = c2w[i].unsqueeze(0).to(model.device)  # (1,3,4)

            cam_i = Cameras(
                camera_to_worlds=c2w_new_batched,
                fx=cams0_low.fx,
                fy=cams0_low.fy,
                cx=cams0_low.cx,
                cy=cams0_low.cy,
                width=cams0_low.width,
                height=cams0_low.height,
                distortion_params=cams0_low.distortion_params,
                camera_type=cams0_low.camera_type,
                times=cams0_low.times,
                metadata=cams0_low.metadata,
            ).to(model.device)

            outputs = model.pipeline.model.get_outputs_for_camera(cam_i)
            rgb = outputs["rgb"]          # (1,H,W,3) float [0,1]
            imgs.append(rgb[0].detach().cpu())

    return torch.stack(imgs, dim=0)


def score_candidates_from_var_map(
    var_map,                         # (N,H,W) nonnegative
    alpha=10.0,
    eps=1e-8,
):
    N = var_map.shape[0]
    var_map = torch.clamp(var_map, min=0.0)

    # per-image normalization for stability
    denom = var_map.flatten(1).median(dim=1).values
    denom = torch.clamp(denom, min=1e-6).view(N, 1, 1)
    var_norm = var_map / denom

    # reliability (0..1)
    conf_map = torch.exp(-alpha * var_norm)

    # informativeness (bounded-ish)
    info_map = torch.log1p(var_norm)   # log(1 + var_norm)

    # score: sum over pixels
    score = (info_map * conf_map).mean(dim=(1, 2))  # (N,)

    # extra stats you might want
    mean_conf = conf_map.mean(dim=(1, 2))
    frac_conf = (conf_map > 0.2).float().mean(dim=(1, 2))

    return score, mean_conf, frac_conf

def median_and_weights_from_ensemble(imgs_uint8, tau=1.5, blur_ksize=9, w_min=0.1):
    """
    imgs_uint8: list length M, each (H,W,3) uint8
    returns:
      median_uint8: (H,W,3) uint8
      weights_f32:  (H,W) float32
      var_map_f32:  (H,W) float32  (optional debug)
    """
    imgs = np.stack(imgs_uint8, axis=0).astype(np.float32)  # (M,H,W,3)

    median = np.median(imgs, axis=0)                        # (H,W,3) float32
    median_uint8 = np.clip(median, 0, 255).astype(np.uint8)

    mean = imgs.mean(axis=0)                                # (H,W,3)
    var_rgb = ((imgs - mean) ** 2).mean(axis=0)             # (H,W,3)
    var_map = var_rgb.mean(axis=-1).astype(np.float32)      # (H,W)

    # same normalization + blur as before
    if blur_ksize is not None and blur_ksize > 1:
        var_map = cv2.GaussianBlur(var_map, (blur_ksize, blur_ksize), 0)

    lo, hi = np.percentile(var_map, [5, 95])
    var_norm = np.clip((var_map - lo) / (hi - lo + 1e-8), 0, 1)

    w = (1.0 - var_norm) ** tau
    w = w_min + (1.0 - w_min) * w
    w = w.astype(np.float32)

    return median_uint8, w, var_map

def prepare_output_dataset(src_run1_dir, dst_dir):
    src = Path(src_run1_dir)
    dst = Path(dst_dir)

    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        raise RuntimeError(f"Destination already exists: {dst}")

    shutil.copytree(src, dst)

    image_dirs = {
        "images": dst / "images",
        "images_2": dst / "images_2",
        "images_4": dst / "images_4",
        "images_8": dst / "images_8",
    }
    for name, img_dir in image_dirs.items():
        if not img_dir.exists():
            raise RuntimeError(f"Missing {name} in {dst}")
        for p in sorted(img_dir.glob("nerf_sample_*.png")):
            p.unlink()

    expected_sizes = {}
    for name, img_dir in image_dirs.items():
        ref = None
        for p in sorted(img_dir.rglob("*")):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS and not p.name.startswith("nerf_sample_"):
                ref = p
                break
        if ref is None:
            raise RuntimeError(f"Could not find a reference image in {img_dir} to infer expected resolution.")
        with Image.open(ref) as im:
            expected_sizes[name] = (im.width, im.height)

    weight_dirs = {
        "weights_nerf_samples": dst / "weights_nerf_samples",
        "weights_nerf_samples_2": dst / "weights_nerf_samples_2",
        "weights_nerf_samples_4": dst / "weights_nerf_samples_4",
        "weights_nerf_samples_8": dst / "weights_nerf_samples_8",
    }
    for wdir in weight_dirs.values():
        wdir.mkdir(parents=True, exist_ok=True)
        for p in sorted(wdir.glob("nerf_sample_*.npy")):
            p.unlink()

    print(
        "[prepare_output_dataset] Expected sizes:",
        {k: {"W": v[0], "H": v[1]} for k, v in expected_sizes.items()},
    )
    return image_dirs, weight_dirs, expected_sizes

def save_multiscale_nerf_sample(median_uint8, image_dirs, expected_sizes, index):
    img = Image.fromarray(median_uint8)
    fname = f"nerf_sample_{index:03d}.png"

    # Save to all target scales by matching exact expected dimensions.
    for key in ["images", "images_2", "images_4", "images_8"]:
        exp_w, exp_h = expected_sizes[key]
        img_scale = img.resize((exp_w, exp_h), resample=Image.Resampling.LANCZOS)
        if img_scale.width != exp_w or img_scale.height != exp_h:
            raise RuntimeError(
                f"Resolution mismatch for {fname} in {key}/: got (W={img_scale.width}, H={img_scale.height}), "
                f"expected (W={exp_w}, H={exp_h})."
            )
        if index == 0:
            print(
                f"[resolution-check] {key} -> generated (W={img_scale.width}, H={img_scale.height}), "
                f"expected (W={exp_w}, H={exp_h})"
            )
        img_scale.save(image_dirs[key] / fname)

def save_multiscale_weights(weight_map, weight_dirs, expected_sizes, index):
    fname = f"nerf_sample_{index:03d}.npy"
    weight_map = weight_map.astype(np.float32)
    for key in ["weights_nerf_samples", "weights_nerf_samples_2", "weights_nerf_samples_4", "weights_nerf_samples_8"]:
        img_key = key.replace("weights_nerf_samples", "images")
        exp_w, exp_h = expected_sizes[img_key]
        weight_scale = cv2.resize(weight_map, (exp_w, exp_h), interpolation=cv2.INTER_AREA)
        if weight_scale.shape != (exp_h, exp_w):
            raise RuntimeError(
                f"Weight resolution mismatch for {fname} in {key}/: got (W={weight_scale.shape[1]}, H={weight_scale.shape[0]}), "
                f"expected (W={exp_w}, H={exp_h})."
            )
        np.save(weight_dirs[key] / fname, weight_scale.astype(np.float32))


def save_weight_visualization(weight_map, out_dir, index):
    """
    Save a grayscale visualization where:
      - white = low confidence (low weight)
      - black = high confidence (high weight)
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    w = weight_map.astype(np.float32)
    w_min = float(np.min(w))
    w_max = float(np.max(w))
    if w_max - w_min < 1e-8:
        vis = np.full_like(w, 127, dtype=np.uint8)
    else:
        # Normalize to [0,1], then invert so low weights are white.
        wn = (w - w_min) / (w_max - w_min)
        vis = np.clip((1.0 - wn) * 255.0, 0, 255).astype(np.uint8)

    fname = out_dir / f"nerf_sample_{index:03d}.png"
    Image.fromarray(vis, mode="L").save(fname)

def render_kept_views_to_tmp_pngs(
    folders, exp_dirs, new_c2w_kept,
    tmp_root, render_scale=0.5, num_models_to_render=None
):
    """
    Creates:
      tmp_root/model_000/nerf_sample_000.png
      tmp_root/model_001/nerf_sample_000.png
      ...
    """
    tmp_root = Path(tmp_root)
    tmp_root.mkdir(parents=True, exist_ok=True)

    K = new_c2w_kept.shape[0]

    if num_models_to_render is None:
        num_models_to_render = len(folders)
    num_models_to_render = max(1, min(num_models_to_render, len(folders)))

    print("############### Starting tmp rendering for kept views ###############")
    print(
        f"[tmp-render] models={num_models_to_render}/{len(folders)}, "
        f"kept_views={K}, render_scale={render_scale}"
    )

    for m, (cfg, exp_dir) in enumerate(
        zip(folders[:num_models_to_render], exp_dirs[:num_models_to_render]), start=1
    ):
        print(f"[tmp-render] loading model {m}/{num_models_to_render}: {cfg}")
        model_dir = tmp_root / f"model_{m:03d}"
        model_dir.mkdir(parents=True, exist_ok=True)

        with pushd(exp_dir):
            model = Nerfacto(cfg)

        for i in range(K):
            if i == 0 or (i + 1) % 10 == 0 or i + 1 == K:
                print(f"[tmp-render] model {m}/{num_models_to_render} view {i + 1}/{K}")
            img_uint8 = render_one_view_uint8(model, new_c2w_kept[i], scale=render_scale)
            Image.fromarray(img_uint8).save(model_dir / f"nerf_sample_{i:03d}.png")

        del model
        torch.cuda.empty_cache()

        print(f"[tmp-render] finished model {m}/{num_models_to_render}")


    return tmp_root

def build_final_dataset_from_tmp(
    src_base_dataset_dir, dst_final_dataset_dir,
    tmp_root, K, tau=1.5, blur_ksize=9, w_min=0.1, cleanup_tmp=False, debug_weights_dir=None
):
    image_dirs, weight_dirs, expected_sizes = prepare_output_dataset(src_base_dataset_dir, dst_final_dataset_dir)

    tmp_root = Path(tmp_root)

    print("############# Building final dataset from tmp renders #############")
    print(f"[final-build] total kept views to write: {K}")

    for i in range(K):
        if i == 0 or (i + 1) % 10 == 0 or i + 1 == K:
            print(f"[final-build] writing sample {i + 1}/{K}")
        imgs_i = []
        model_dirs = sorted(tmp_root.glob("model_*"))
        for md in model_dirs:
            p = md / f"nerf_sample_{i:03d}.png"
            imgs_i.append(np.array(Image.open(p).convert("RGB"), dtype=np.uint8))

        # Original behavior: write ensemble median image and variance-based weights.
        median_uint8, w_f32, _ = median_and_weights_from_ensemble(
            imgs_i, tau=tau, blur_ksize=blur_ksize, w_min=w_min
        )
        if i == 0:
            print(f"[image-selection] using ensemble median over {len(model_dirs)} models")

        save_multiscale_nerf_sample(median_uint8, image_dirs, expected_sizes, i)
        save_multiscale_weights(w_f32, weight_dirs, expected_sizes, i)
        if debug_weights_dir is not None:
            save_weight_visualization(w_f32, debug_weights_dir, i)

    # optional packed file
    # weights_all = np.stack([np.load(weights_dir / f"nerf_sample_{i:03d}.npy") for i in range(K)], axis=0)
    # np.save(weights_dir / "weights_all.npy", weights_all)

    if cleanup_tmp:
        shutil.rmtree(tmp_root)

    return Path(dst_final_dataset_dir)


def plot_new_samples(cam_positions, new_positions, name, output_dir):
    points_np = cam_positions.detach().numpy()
    new_points_np = new_positions.detach().cpu().numpy()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(points_np[:, 0], points_np[:, 1], points_np[:, 2],
               s=5, label="original cams", color="blue")

    # new sampled positions: new_points_np is (n_samples, 3)
    ax.scatter(new_points_np[:, 0], new_points_np[:, 1], new_points_np[:, 2],
               s=40, label="new samples", color="red")

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Camera Positions with New Samples')
    ax.set_box_aspect([1, 1, 1])
    ax.legend()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{name}.png"
    plt.savefig(str(out_path), dpi=300, bbox_inches="tight")
    plt.close(fig)

def c2w_to_colmap_qvec_tvec(c2w: torch.Tensor):
    R_c2w = c2w[:, :3]
    t_c2w = c2w[:, 3]

    R_w2c = R_c2w.T
    t_w2c = -R_w2c @ t_c2w

    # rotmat2qvec is from colmap_parsing_utils
    qvec = rotmat2qvec(R_w2c.cpu().numpy())
    tvec = t_w2c.cpu().numpy()
    return qvec, tvec

def load_colmap_model(colmap_dir):
    cameras, images, points3D = colmap.read_model(colmap_dir, ext=".bin")
    return cameras, images, points3D

def append_nerf_images_to_colmap(
    colmap_dir,
    new_c2w,
):
    cameras, images, points3D = load_colmap_model(colmap_dir)

    base_camera_id = list(cameras.keys())[0]
    next_image_id = max(images.keys()) + 1

    for i in range(new_c2w.shape[0]):
        c2w = new_c2w[i]
        qvec, tvec = c2w_to_colmap_qvec_tvec(c2w)
        image_name = f"nerf_sample_{i:03d}.png"

        new_image = ColmapImage(
            id=next_image_id,
            qvec=qvec,
            tvec=tvec,
            camera_id=base_camera_id,
            name=image_name,
            xys=np.empty((0, 2)),
            point3D_ids=np.empty((0,), dtype=np.int64),
        )
        images[next_image_id] = new_image
        next_image_id += 1

    # ✅ Use colmap.write_model from the same module
    colmap.write_model(cameras, images, points3D, colmap_dir, ext=".bin")
    print("✅ images.bin successfully overwritten with NeRF views")



def main(args):
    torch.manual_seed(0)
    np.random.seed(0)
    folders = args.nerf_folders
    exp_dirs = args.exp_dirs

    sum_rgb = None
    sum_sq_rgb = None
    M = len(args.nerf_folders)

    # First load the cameras in the dataset
    with pushd(exp_dirs[0]):
        model = Nerfacto(folders[0])

    cams = model.pipeline.datamanager.train_dataset.cameras.to('cpu')
    dpo = model.pipeline.datamanager.train_dataparser_outputs
    c2w = cams.camera_to_worlds.cpu()
    cam_positions = c2w[..., :3, 3]

    del model
    torch.cuda.empty_cache()

    n_samples_jitter = 100
    n_samples_interpolation = 100
    print(f"[phase] sampling candidate poses: jitter={n_samples_jitter}, interpolation={n_samples_interpolation}")

    new_c2w_jitter = sample_new_cameras_jitter(cam_positions, c2w, n_samples_jitter)
    new_c2w_interpolation = sample_new_cameras_interpolation(cam_positions, c2w, n_samples_interpolation)

    new_c2w = torch.cat(
        [new_c2w_jitter, new_c2w_interpolation],
        dim=0
    )
    print(f"[phase] total candidate poses: {new_c2w.shape[0]}")

    plot_new_samples(cam_positions, new_c2w[..., :3, 3], "all_samples", args.debug_plot_dir)


    print(f"[phase] candidate scoring over ensemble: {len(folders)} models")
    for mi, (cfg, exp_dir) in enumerate(zip(folders, exp_dirs), start=1):
        print(f"[phase] scoring model {mi}/{len(folders)}")
        with pushd(exp_dir):
            model = Nerfacto(cfg)        

        # render low-res for ALL candidate poses (downsample by 8 for selection)
        rgb = create_new_images_low_res_torch(
            model, new_c2w, selection_scale=0.125
        )  # (N,H,W,3) float [0,1] on CPU

        if sum_rgb is None:
            sum_rgb = rgb
            sum_sq_rgb = rgb * rgb
        else:
            sum_rgb += rgb
            sum_sq_rgb += rgb * rgb

        # free GPU memory
        del model
        torch.cuda.empty_cache()
        print(f"[phase] completed scoring model {mi}/{len(folders)}")

    mean = sum_rgb / M
    var_rgb = (sum_sq_rgb / M) - mean * mean
    var_map = var_rgb.mean(dim=-1)  # (N,H,W)
    var_map = torch.clamp(var_map, min=0.0)

    new_c2w_kept1, keep_mask1, stats1 = filter_candidates_from_var_map(var_map, new_c2w)
    var_map_kept1 = var_map[keep_mask1]

    plot_new_samples(
        cam_positions,
        new_c2w_kept1[..., :3, 3],
        "all_samples_first_selection",
        args.debug_plot_dir,
    )

    print("initial number interpolation", new_c2w_interpolation.shape[0])
    print("initial number jitter", new_c2w_jitter.shape[0])
    print("number of kept", new_c2w_kept1.shape[0])

    # Second stage: random subset from first-stage kept candidates.
    K_keep = min(args.num_final_samples, new_c2w_kept1.shape[0])
    print(
        f"[phase] final selection keep={K_keep} "
        f"(requested={args.num_final_samples}, available={new_c2w_kept1.shape[0]}) [random]"
    )
    rand_idx = torch.randperm(new_c2w_kept1.shape[0])[:K_keep]
    new_c2w_kept2 = new_c2w_kept1[rand_idx]

    plot_new_samples(
        cam_positions,
        new_c2w_kept2[..., :3, 3],
        "all_samples_second_selection",
        args.debug_plot_dir,
    )

    print("number of kept after second selection", new_c2w_kept2.shape[0])

    K = new_c2w_kept2.shape[0]
    tmp_root = os.path.join(args.tmp_root, "tmp_ensemble_renders")

    render_kept_views_to_tmp_pngs(
        folders=folders,
        exp_dirs=exp_dirs,
        new_c2w_kept=new_c2w_kept2,
        tmp_root=tmp_root,
        render_scale=args.final_render_scale,
        num_models_to_render=None,  # render all models for ensemble median
    )

    final_dir = build_final_dataset_from_tmp(
        src_base_dataset_dir=args.input_dataset,
        dst_final_dataset_dir=args.output_dataset,
        tmp_root=tmp_root,
        K=K,
        tau=args.tau,
        blur_ksize=9,
        w_min=0.1,
        cleanup_tmp=True,
        debug_weights_dir=Path(args.debug_plot_dir) / "weight_maps",
    )

    colmap_dir = Path(final_dir) / "sparse" / "0"
    # NeRF cameras live in Nerfstudio's transformed frame (auto-orient / center / scale).
    # Convert sampled poses back to original COLMAP world frame before writing images.bin.
    new_c2w_colmap = dpo.transform_poses_to_original_space(
        new_c2w_kept2.cpu(),
        camera_convention="opencv",
    )
    append_nerf_images_to_colmap(
        colmap_dir=str(colmap_dir),
        new_c2w=new_c2w_colmap.cpu(),   # (K,3,4) in original COLMAP frame
    )
    print(f"[done] final augmented dataset written to: {final_dir}")

    
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--nerf-folders", nargs="+", required=True)
    p.add_argument("--exp-dirs", nargs="+", required=True)
    p.add_argument("--input-dataset", required=True)
    p.add_argument("--output-dataset", required=True)
    p.add_argument("--tmp-root", default="/tmp")          # or your scratch path
    p.add_argument("--tau", type=float, default=1.5)
    p.add_argument(
        "--final-render-scale",
        type=float,
        default=1.0,
        help="Scale used to render final kept NeRF samples before writing multiscale folders. "
             "Use 0.125 to render at roughly images_8 resolution then upsample.",
    )
    p.add_argument("--debug-plot-dir", default="image_supervision")
    p.add_argument(
        "--num-final-samples",
        type=int,
        default=50,
        help="Number of final kept synthetic views to add to the dataset (default: 50).",
    )
    args = p.parse_args()
    main(args)
