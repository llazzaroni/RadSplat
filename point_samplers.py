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

    dataset = data_manager.train_dataset

    # determine how many points to sample from each image
    img_weights = np.zeros(len(dataset))
    imgs_shapes = []
    pixels_weights = []

    for idx in range(len(dataset)):

        img = dataset.get_numpy_image(idx)
        H, W, _ = img.shape
        imgs_shapes.append((H,W))

        # convert to gray scale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply Sobel operator
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)  # Horizontal edges
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)  # Vertical edges

        # Compute gradient magnitude
        gradient_magnitude = cv2.magnitude(sobelx, sobely)

        # Convert to uint8
        gradient_magnitude = cv2.convertScaleAbs(gradient_magnitude).ravel()

        # Normalize
        gradient_magnitude = gradient_magnitude / 255

        # smooth results
        # smoothed = cv2.GaussianBlur(gradient_magnitude, (5, 5), 0).ravel()

        img_weights[idx] = gradient_magnitude.sum()
        normalized = gradient_magnitude / gradient_magnitude.sum()
        pixels_weights.append(normalized)

    x = []
    y = []
    camera_idx = []

    img_weights = img_weights / img_weights.sum()
    samples_per_image = np.floor(img_weights * n)

    for ind, weights in enumerate(pixels_weights):

        H,W = imgs_shapes[ind]

        print(H)
        print(W)
        img_indeces = [(x,y) for y in range(H-1, -1, -1) for x in range(W)]
        print(len(img_indeces))
        print(len(weights))

        sampled_indeces = np.random.choice(
            len(img_indeces),
            size = int(samples_per_image[ind]),
            p = weights,
            replace = False
        )


        for s_ind in sampled_indeces:
            x.append(img_indeces[s_ind][0])
            y.append(img_indeces[s_ind][1])
        camera_idx += [ind] * len(sampled_indeces)

    ray_indices = torch.stack([torch.tensor(camera_idx), torch.tensor(y), torch.tensor(x)], dim=-1).long()

    return ray_indices
