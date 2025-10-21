from pathlib import Path
import torch

import sys
NS_ROOT = Path(__file__).parent / "submodules" / "nerfstudio"
sys.path.insert(0, str(NS_ROOT))

from nerfstudio.data.datamanagers.base_datamanager import DataManager

def random_sampler(data_manager : DataManager, n : int,  device) -> torch.Tensor:
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

