# Modified version of nerf_step, which queries uniformly and outputs the points categorized as maxima

import warnings
import math
import torch
import logging

from nerf_models import Nerfacto
from nerfstudio.field_components.field_heads import FieldHeadNames
from point_samplers import sobel_edge_detector_sampler, canny_edge_detector_sampler, random_sampler, mixed_sampler
from gs_initializer import Initializer
from pathlib import Path
import argparse

warnings.filterwarnings(
    "ignore",
    message="Using a non-tuple sequence for multidimensional indexing is deprecated",
)

def create_parser():
    parser = argparse.ArgumentParser(
        description="parser"
    )

    parser.add_argument(
        "--nerf-folder",
        "-nf",
        type=str,
        required=True,
        help="Folder containing the config.yml file to set up the nerf model"
    )

    parser.add_argument(
        "--output-name",
        "-o",
        type=str,
        required=True,
        help="name of the output file"
    )

    parser.add_argument(
        "--sampling-size",
        "-n",
        type=int,
        required=True,
        help="number of rays to sample"
    )

    parser.add_argument(
        "--ray-sampling-strategy",
        "-s",
        type=str,
        required=False,
        help="name odf the filter to use: canny | sobel | mixed-sobel | mixed-canny"
    )

    parser.add_argument(
        "--percentage-random",
        "-pr",
        type=float,
        required=False,
        help="name odf the filter to use: canny | sobel | mixed-sobel | mixed-canny"
    )

    return parser

def rel_to_images_root(p: str) -> str:
    # find 'images' in the path and take from there (COLMAP parser returns names relative to images/)
    parts = Path(p).parts
    if "images" in parts:
        i = parts.index("images")
        return str(Path(*parts[i:]))  # e.g. 'images/seq/frame0001.png'
    return Path(p).name


def prune_peaks_for_ray(density_ray, peak_indices, max_peaks=4):
    # density_ray: (M,)
    # peak_indices: 1D tensor of ints

    # if no peaks, nothing to do
    if peak_indices.numel() == 0:
        return peak_indices

    # ensure sorted
    peak_indices, _ = torch.sort(peak_indices)

    while peak_indices.numel() > max_peaks:
        # differences between consecutive peaks
        diffs = peak_indices[1:] - peak_indices[:-1]  # [num_peaks-1]

        k = torch.argmin(diffs)  # index in 'diffs', scalar

        i1 = peak_indices[k]
        i2 = peak_indices[k + 1]

        # remove the one with smaller density
        if density_ray[i1] < density_ray[i2]:
            remove_idx = i1
        else:
            remove_idx = i2

        keep_mask = peak_indices != remove_idx
        peak_indices = peak_indices[keep_mask]

    return peak_indices


def find_colors(density_ray, color_ray, index, radius=2):
    """
    density_ray: (M,)    1D tensor of densities
    color_ray:   (M, 3)  2D tensor of colors
    index:       int or 0D tensor
    radius:      int, number of neighbours on each side
    """
    if isinstance(index, torch.Tensor):
        index = index.item()
    M = density_ray.shape[0]

    start = max(0, index - radius)
    end = min(M, index + radius + 1)  # +1 because slicing is exclusive

    window_indices = torch.arange(start, end, device=density_ray.device)

    dens_window = density_ray[window_indices]   # (K,)
    col_window = color_ray[window_indices]      # (K, 3)

    weights = dens_window.clone()
    weight_sum = weights.sum()

    if weight_sum > 0:
        weights = weights / weight_sum
    else:
        # fallback: uniform average if all weights are zero
        weights = torch.ones_like(weights) / weights.numel()

    weighted_color = (col_window * weights.unsqueeze(-1)).sum(dim=0)  # (3,)

    return weighted_color


def compute_initial_positions(positions, densities, colors, transmittance, max_peaks=3, radius=2):
    # densities: (N, M, 1)
    # positions: (N, M, 3)
    # colors:    (N, M, 3)
    N, M = densities.shape[0], densities.shape[1]

    densities_1d = densities.squeeze(-1)  # (N, M)
    transmittances_1d = transmittance.squeeze(-1) # (N, M)

    # Find maxima per ray
    max_vals, _ = densities_1d.max(dim=1, keepdim=True)  # (N, 1)
    threshold = max_vals / 4.0

    # First find the peaks (local maxima above threshold)
    left = densities_1d[:, 1:-1] > densities_1d[:, :-2]
    right = densities_1d[:, 1:-1] > densities_1d[:, 2:]
    high = densities_1d[:, 1:-1] > threshold
    high_2 = transmittances_1d[:, 1:-1] <= 0.5

    peak_mask_inner = left & right & high & high_2  # (N, M-2)

    full_peak_mask = torch.zeros_like(densities_1d, dtype=torch.bool)  # (N, M)
    full_peak_mask[:, 1:-1] = peak_mask_inner

    result_positions = []
    result_colors = []

    for i in range(N):
        ray_mask = full_peak_mask[i, :]                 # (M,)
        peak_indices = torch.nonzero(ray_mask, as_tuple=False).flatten()  # 1D

        dens_i = densities_1d[i]                        # (M,)
        pos_i = positions[i]                            # (M, 3)
        col_i = colors[i]                               # (M, 3)

        # prune peaks for this ray
        pruned_indices = prune_peaks_for_ray(dens_i, peak_indices, max_peaks=max_peaks)

        # for each kept peak, append a position and a color
        for idx in pruned_indices:
            idx_int = idx.item()

            # position at this peak
            result_positions.append(pos_i[idx_int])  # (3,)

            # density-weighted color around this peak
            weighted_color = find_colors(dens_i, col_i, idx_int, radius=radius)
            result_colors.append(weighted_color)     # (3,)

    if len(result_positions) == 0:
        # no peaks at all — define some fallback (here just zeros)
        peak_positions = torch.zeros((0, 3), device=positions.device, dtype=positions.dtype)
        peak_colors = torch.zeros((0, 3), device=colors.device, dtype=colors.dtype)
    else:
        peak_positions = torch.stack(result_positions, dim=0)  # (K, 3)
        peak_colors = torch.stack(result_colors, dim=0)        # (K, 3)

    return peak_positions, peak_colors




if __name__ == "__main__":

    parser = create_parser()
    args = parser.parse_args()

    folder = args.nerf_folder
    N_RAYS = args.sampling_size
    BATCH_SIZE = 5_000
    RAYS_BATCH_NAME = args.output_name
    n_batches = math.ceil(N_RAYS / BATCH_SIZE)

    
    print('########### loading model cacca pupù')
    model = Nerfacto(folder)
    print('######## done loading the model first try')
    cams = model.pipeline.datamanager.train_dataset.cameras.to('cpu')
    dpo = model.pipeline.datamanager.train_dataparser_outputs

    # saple points
    print("######## done loading the model, now sample the points")

    if args.ray_sampling_strategy == "canny":
        coords = canny_edge_detector_sampler(model.pipeline.datamanager, N_RAYS, model.device)
    elif args.ray_sampling_strategy == "sobel":
        coords = sobel_edge_detector_sampler(model.pipeline.datamanager, N_RAYS, model.device)
    elif args.ray_sampling_strategy == "mixed-sobel":
        coords = mixed_sampler(model.pipeline.datamanager, N_RAYS, share_rnd = args.percentage_random, edge_detector = "sobel", device = model.device)
    elif args.ray_sampling_strategy == "mixed-canny":
        coords = mixed_sampler(model.pipeline.datamanager, N_RAYS, share_rnd = args.percentage_random, edge_detector = "canny", device = model.device)
    else:
        coords = random_sampler(model.pipeline.datamanager, N_RAYS, model.device)


    print("####### Reached the start of the loop")
    xyzrgb_chunks = []
    for b in range(n_batches):

        # get initial and final index of the index rays to query
        s = b * BATCH_SIZE
        e = min((b + 1) * BATCH_SIZE, N_RAYS)
        if s >= e:
            break
        batch_rays_indexes = coords[s:e, :]
        B = batch_rays_indexes.shape[0]

        logging.info('sampling rays')
        rays = model.create_rays(batch_rays_indexes)

        logging.info('sampling points')

        # Query the model
        sampled = model.sample_points_uniform(rays)
        outputs, field_outputs, weights  = model.evaluate_points(sampled)

        # Obtain positions, densities and colors to find the points
        positions = sampled.frustums.get_positions()
        densities = field_outputs[FieldHeadNames.DENSITY]
        colors = field_outputs[FieldHeadNames.RGB]

        # Find the transmittance
        gs_initializer = Initializer(weights, sampled)
        transmittance = gs_initializer.compute_transmittance()

        # Call the function for finding the peaks
        positions_points, color_points = compute_initial_positions(positions, densities, colors, transmittance)

        color_points = color_points.clamp(0.0, 1.0)

        xyzrgb_batch = torch.cat([positions_points, color_points], dim=-1)

        xyzrgb_chunks.append(xyzrgb_batch.detach().cpu())

        print("####### Reached the", b, "batch")
        print(xyzrgb_batch.shape)


    xyzrgb = torch.cat(xyzrgb_chunks, dim=0)
    xyzrgb = xyzrgb[:1_000_000]
    print(xyzrgb.shape)

    image_filenames_abs = [str(p) for p in dpo.image_filenames]
    image_filenames_rel = [rel_to_images_root(p) for p in image_filenames_abs]

    payload = {
        "xyzrgb": xyzrgb.cpu(),
        "camera_to_worlds": cams.camera_to_worlds.cpu(),
        "K": cams.get_intrinsics_matrices().cpu(),
        "image_filenames_abs": image_filenames_abs,
        "image_filenames_rel": image_filenames_rel,
    }

    logging.info('saving initial positions')
    torch.save(payload, RAYS_BATCH_NAME)
