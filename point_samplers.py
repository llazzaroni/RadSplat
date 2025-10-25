from os import replace
from pathlib import Path
import numpy as np
import torch
import cv2

import sys

NS_ROOT = Path(__file__).parent / "submodules" / "nerfstudio"
sys.path.insert(0, str(NS_ROOT))

from nerfstudio.data.datamanagers.base_datamanager import DataManager


def random_sampler(data_manager: DataManager, n: int, device) -> torch.Tensor:
    """
    Sample n rays from a set of given cameras

    :Param data_manager -> (DataManager) the data manager containing the target cameras
    :Param n -> (int) number of rays to samplePipeline

    :Retunr (torch.Tensor) a tesor of shape (n, 3) containing (camera idx, point y, point x)
    """

    # get cameras
    cameras = data_manager.train_dataset.cameras.to(device)
    num_imgs = len(cameras)

    # create a tensor of size n, 1 which assign to each ray to sample the camera to use
    img_idx = torch.randint(low=0, high=num_imgs, size=(n,), device=device)

    # for each camera we get the image height and width
    H = cameras.height.squeeze(-1).to(device)
    W = cameras.width.squeeze(-1).to(device)
    # assign that size to each entry sampled
    sel_H = H[img_idx]
    sel_W = W[img_idx]

    # sample all the x, y points
    # generate a random number from 0 to 1 and multiply for width/height then floor
    y = torch.floor(torch.rand(n, device=device) * sel_H).long()
    x = torch.floor(torch.rand(n, device=device) * sel_W).long()

    # compact everything in one tensor
    ray_indices = torch.stack([img_idx, y, x], dim=-1).long()

    return ray_indices


# def sobel_edge_detector_sampler(data_manager : DataManager, n : int,  device) -> torch.Tensor:
def sobel_edge_detector_sampler(
    data_manager: DataManager, n: int, device
) -> torch.Tensor:

    camera_shapes = {}

    cameras = data_manager.train_dataset.cameras.to(device)

    # get images sizes
    H = cameras.height.squeeze(-1).to(device)
    W = cameras.width.squeeze(-1).to(device)

    # get image paths
    img_paths: list[Path] = data_manager.train_dataparser_outputs.image_filenames
    # determine how many points to sample from each image
    img_weights = np.zeros(len(img_paths))
    pixels_weights = []

    for idx, img_path in enumerate(img_paths):

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # Apply Sobel operator
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)  # Horizontal edges
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)  # Vertical edges

        # Compute gradient magnitude
        gradient_magnitude = cv2.magnitude(sobelx, sobely)

        # Convert to uint8
        gradient_magnitude = cv2.convertScaleAbs(gradient_magnitude)

        # Normalize
        gradient_magnitude = gradient_magnitude / 255

        # smooth results
        smoothed = cv2.GaussianBlur(gradient_magnitude, (5, 5), 0).ravel()

        img_weights[idx] = smoothed.sum()
        smoothed = smoothed / smoothed.sum()
        pixels_weights.append(smoothed)

    x = []
    y = []
    camera_idx = []

    img_weights = img_weights / img_weights.sum()
    samples_per_image = np.floor(img_weights * n)

    indeces = {(H[0], W[0]): [(y, x) for y in range(H[0]) for x in range(W[0])]}

    for ind, weights in enumerate(pixels_weights):
        
        img_shape = (cameras[ind].height, cameras[ind].width)
        if img_shape in indeces:
            img_indeces = indeces[img_shape]
        else:
            img_indeces = [(y, x) for y in range(img_shape[0]) for x in range(img_shape[1])]
            indeces[img_shape] = img_indeces


        sampled_indeces = np.random.choice(
            len(img_indeces),
            size = int(samples_per_image[ind]),
            p = weights,
            replace = False
        )

        print(sampled_indeces)

        for s_ind in sampled_indeces:
            x.append(img_indeces[s_ind][1])
            y.append(img_indeces[s_ind][0])
            camera_idx.append(ind)

    ray_indices = torch.stack([torch.tensor(camera_idx), torch.tensor(y), torch.tensor(x)], dim=-1).long()

    return ray_indices
