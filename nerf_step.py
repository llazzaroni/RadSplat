import warnings
import math
import torch
import logging

from nerf_models import Nerfacto
from point_samplers import sobel_edge_detector_sampler, canny_edge_detector_sampler, random_sampler
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
        "--ray-sampling-strategy",
        "-s",
        type=str,
        required=False,
        help="name odf the filter to use: canny | sobel"
    )

    return parser

def rel_to_images_root(p: str) -> str:
    # find 'images' in the path and take from there (COLMAP parser returns names relative to images/)
    parts = Path(p).parts
    if "images" in parts:
        i = parts.index("images")
        return str(Path(*parts[i:]))  # e.g. 'images/seq/frame0001.png'
    return Path(p).name

if __name__ == "__main__":

    parser = create_parser()
    args = parser.parse_args()

    folder = args.nerf_folder
    N_RAYS = 500_000
    BATCH_SIZE = 5_000
    RAYS_BATCH_NAME = args.output_name
    n_batches = math.ceil(N_RAYS / BATCH_SIZE)

    
    print('########### loading model')
    model = Nerfacto(folder)
    cams = model.pipeline.datamanager.train_dataset.cameras.to('cpu')
    dpo = model.pipeline.datamanager.train_dataparser_outputs

    # saple points

    if args.ray_sampling_strategy == "canny":
        coords = canny_edge_detector_sampler(model.pipeline.datamanager, N_RAYS, model.device)
    elif args.ray_sampling_strategy == "sobel":
        coords = sobel_edge_detector_sampler(model.pipeline.datamanager, N_RAYS, model.device)
    else:
        coords = random_sampler(model.pipeline.datamanager, N_RAYS, model.device)



    xyzrgb_chunks = []
    for b in range(n_batches):
        print(f"worinng of chink {b}/{n_batches}")

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

        sampled = model.sample_points(rays)
        outputs, field_outputs, weights  = model.evaluate_points(sampled)

        logging.info('init gs')
        gs_initializer = Initializer(weights, sampled)

        logging.info('compute trasmittance')
        trasmittance = gs_initializer.compute_transmittance()

        logging.info('computer initial_position')
        initial_position = gs_initializer.compute_inital_positions(trasmittance, 0.5)

        xyzrgb_batch = torch.cat([initial_position, outputs['rgb']], dim=-1)

        xyzrgb_chunks.append(xyzrgb_batch.detach().cpu())


    xyzrgb = torch.cat(xyzrgb_chunks, dim=0)

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
