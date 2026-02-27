import math
from typing import Dict, Optional, Tuple
import os

import numpy as np
import torch
from submodules.gsplat.examples.datasets.colmap import Parser
from submodules.gsplat.examples.utils import knn, rgb_to_sh
from submodules.gsplat.gsplat.optimizers import SelectiveAdam

from gsplat_dir.align import umeyama_align

def _norm_name(p: str) -> str:
    # normalize to compare across jpg/png, case, and nested dirs
    b = os.path.basename(p).lower()
    b, _ = os.path.splitext(b)
    return b


def create_splats_with_optimizers(
    parser: Parser,
    cfg,
    nerf_init,
    init_type: str = "sfm",
    init_num_pts: int = 100_000,
    init_extent: float = 3.0,
    init_opacity: float = 0.1,
    init_scale: float = 1.0,
    means_lr: float = 1.6e-4,
    scales_lr: float = 5e-3,
    opacities_lr: float = 5e-2,
    quats_lr: float = 1e-3,
    sh0_lr: float = 2.5e-3,
    shN_lr: float = 2.5e-3 / 20,
    scene_scale: float = 1.0,
    sh_degree: int = 3,
    sparse_grad: bool = False,
    visible_adam: bool = False,
    batch_size: int = 1,
    feature_dim: Optional[int] = None,
    device: str = "cuda",
    world_rank: int = 0,
    world_size: int = 1,
) -> Tuple[torch.nn.ParameterDict, Dict[str, torch.optim.Optimizer]]:
    
    if nerf_init:
        # num_points = len(torch.from_numpy(parser.points).float()
        payload = torch.load(cfg.pt_path, map_location="cuda")

        xyzrgb = payload["xyzrgb"].detach().cpu().numpy()
        num_points = len(xyzrgb)
        C_nerf_all = payload["camera_to_worlds"][:, :3, 3].detach().cpu().numpy()

        # --- build name lists from what you actually saved ---
        # prefer relative list if present, else absolute
        nerf_paths = payload.get("image_filenames_rel") or payload.get("image_filenames_abs")
        if nerf_paths is None:
            raise KeyError("payload is missing 'image_filenames_rel'/'image_filenames_abs'")

        # normalize both sides
        nerf_keys = [_norm_name(p) for p in nerf_paths]
        gs_names  = parser.image_names                       # these are COLMAP image names (relative)
        gs_keys   = [_norm_name(p) for p in gs_names]
        name_to_gs = {k: i for i, k in enumerate(gs_keys)}

        # intersection in NeRF-train order
        pairs = [(i, name_to_gs[k]) for i, k in enumerate(nerf_keys) if k in name_to_gs]
        if len(pairs) < 3:
            # help debugging: show a few sample unmatched names
            ex = set(nerf_keys[:10]) - set(name_to_gs.keys())
            raise RuntimeError(f"Not enough overlapping frames (found {len(pairs)}). "
                            f"Example unmatched (first few): {list(ex)[:5]}")

        i_nerf = np.array([p[0] for p in pairs], dtype=np.int64)
        i_gs   = np.array([p[1] for p in pairs], dtype=np.int64)

        # subselect cameras in the same order
        C_nerf = C_nerf_all[i_nerf]                # (M, 3)
        C_gs   = parser.camtoworlds[i_gs, :3, 3]   # (M, 3)

        # Umeyama alignment
        s, R, t = umeyama_align(C_nerf, C_gs, with_scale=True)
        Cn_aligned = (s * (R @ C_nerf.T).T + t)
        print("overlap frames:", len(pairs), "mean |ΔC| =", np.linalg.norm(Cn_aligned - C_gs, axis=1).mean())

        # transform all NeRF points
        xyz = xyzrgb[:, :3]
        xyz_aligned = (s * (R @ xyz.T).T + t)
        xyzrgb_aligned = np.concatenate([xyz_aligned, xyzrgb[:, 3:6]], axis=1)

        points = torch.from_numpy(xyzrgb_aligned[:num_points, :3]).float().to(device)
        rgbs   = torch.from_numpy(xyzrgb_aligned[:num_points, 3:6]).float().to(device)
        #if rgbs.max() > 1.0:
        #    rgbs = rgbs / 255.0

        # Initialize the GS size to be the average dist of the 3 nearest neighbors
        dist2_avg = (knn(points, 2)[:, 1:] ** 2).mean(dim=-1)  # [N,]
    else:
        if init_type == "sfm":
            points = torch.from_numpy(parser.points).float()
            rgbs = torch.from_numpy(parser.points_rgb / 255.0).float()
        elif init_type == "random":
            points = init_extent * scene_scale * (torch.rand((init_num_pts, 3)) * 2 - 1)
            rgbs = torch.rand((init_num_pts, 3))
        else:
            raise ValueError("Please specify a correct init_type: sfm or random")
        # Initialize the GS size to be the average dist of the 3 nearest neighbors
        dist2_avg = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1)  # [N,]

    dist_avg = torch.sqrt(dist2_avg)
    scales = torch.log(dist_avg * init_scale).unsqueeze(-1).repeat(1, 3)  # [N, 3]

    # Distribute the GSs to different ranks (also works for single rank)
    points = points[world_rank::world_size]
    rgbs = rgbs[world_rank::world_size]
    scales = scales[world_rank::world_size]

    N = points.shape[0]
    quats = torch.rand((N, 4))  # [N, 4]
    opacities = torch.logit(torch.full((N,), init_opacity))  # [N,]

    params = [
        # name, value, lr
        ("means", torch.nn.Parameter(points), means_lr * scene_scale),
        ("scales", torch.nn.Parameter(scales), scales_lr),
        ("quats", torch.nn.Parameter(quats), quats_lr),
        ("opacities", torch.nn.Parameter(opacities), opacities_lr),
    ]

    if feature_dim is None:
        # color is SH coefficients.
        colors = torch.zeros((N, (sh_degree + 1) ** 2, 3))  # [N, K, 3]
        colors[:, 0, :] = rgb_to_sh(rgbs)
        params.append(("sh0", torch.nn.Parameter(colors[:, :1, :]), sh0_lr))
        params.append(("shN", torch.nn.Parameter(colors[:, 1:, :]), shN_lr))
    else:
        # features will be used for appearance and view-dependent shading
        features = torch.rand(N, feature_dim)  # [N, feature_dim]
        params.append(("features", torch.nn.Parameter(features), sh0_lr))
        colors = torch.logit(rgbs)  # [N, 3]
        params.append(("colors", torch.nn.Parameter(colors), sh0_lr))

    splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to(device)
    # Scale learning rate based on batch size, reference:
    # https://www.cs.princeton.edu/~smalladi/blog/2024/01/22/SDEs-ScalingRules/
    # Note that this would not make the training exactly equivalent, see
    # https://arxiv.org/pdf/2402.18824v1
    BS = batch_size * world_size
    optimizer_class = None
    if sparse_grad:
        optimizer_class = torch.optim.SparseAdam
    elif visible_adam:
        optimizer_class = SelectiveAdam
    else:
        optimizer_class = torch.optim.Adam
    optimizers = {
        name: optimizer_class(
            [{"params": splats[name], "lr": lr * math.sqrt(BS), "name": name}],
            eps=1e-15 / math.sqrt(BS),
            # TODO: check betas logic when BS is larger than 10 betas[0] will be zero.
            betas=(1 - BS * (1 - 0.9), 1 - BS * (1 - 0.999)),
        )
        for name, _, lr in params
    }
    return splats, optimizers