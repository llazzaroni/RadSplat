import numpy as np
import logging

from submodules.nerf.run_nerf_helpers import get_rays_np, init_nerf_model
from submodules.nerf.load_llff import load_llff_data

def sample_rays(datadir: str, factor : int = 8, n : int = 1_000_000) -> np.ndarray:

    logging.info("Loading data from: " + datadir)
    images, poses, _, _, _ = load_llff_data(datadir, factor, recenter=True, bd_factor=.75, spherify=False)

    y_size, x_size, _ = images[0].shape

    hwf = poses[0, :3, -1]
    H, W, focal = hwf

    # get_rays_np() returns rays_origin=[H, W, 3], rays_direction=[H, W, 3]
    # for each pixel in the image. This stack() adds a new dimension.
    rays = [get_rays_np(H, W, focal, p) for p in poses[:, :3, :4]]
    rays = np.stack(rays, axis=0)  #(194, 2, 411, 618, 3)

    img_idx = np.random.randint(0, len(images), size=n)
    y_idx = np.random.randint(0, y_size, size=n)
    x_idx = np.random.randint(0, x_size, size=n)


    origins = rays[img_idx, 0, y_idx, x_idx]
    dirs = rays[img_idx, 1, y_idx, x_idx]

    rays_sampled = np.stack([origins, dirs], axis=1)

    return rays_sampled

if __name__ == "__main__":
    
    datadir = '/home/bolla/Documents/course_materials/ds_lab/data/bicycle'
    modeldir = '/home/bolla/Documents/course_materials/ds_lab/models/test_run_1'

    rays = sample_rays(datadir, 8)
    print(rays[0])

    # TODO initialize model

    # TODO run nerf for each ray

    # get z median

    # frmat results

    pass
