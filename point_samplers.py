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
    ray_indices = torch.stack([img_idx, y, x], dim=-1).long().to(torch.device(device))

    return ray_indices


def sobel_edge_detector_sampler(
    data_manager: DataManager, n: int, device
) -> torch.Tensor:

    dataset = data_manager.train_dataset

    # determine how many points to sample from each image
    img_weights = np.zeros(len(dataset))
    imgs_shapes = []
    pixels_weights = []

    cameras = data_manager.train_dataset.cameras.to(device)
    H_all = cameras.height.squeeze(-1).cpu().numpy()
    W_all = cameras.width.squeeze(-1).cpu().numpy()

    for idx in range(len(dataset)):

        H, W = int(H_all[idx]), int(W_all[idx])
        imgs_shapes.append((H,W))
        img = dataset.get_numpy_image(idx)

        # convert to gray scale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply Sobel operator
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)  # Horizontal edges
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)  # Vertical edges

        # Compute gradient magnitude
        gradient_magnitude = cv2.magnitude(sobelx, sobely)

        if gradient_magnitude.shape != (H, W):
            gradient_magnitude = cv2.resize(gradient_magnitude, (W, H))


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

        img_indeces = [(x,y) for y in range(H-1, -1, -1) for x in range(W)]

        sample_size = min(len(img_indeces), int(samples_per_image[ind]))

        sampled_indeces = np.random.choice(
            len(img_indeces),
            size = sample_size,
            p = weights,
            replace = False
        )

        for s_ind in sampled_indeces:
            x.append(img_indeces[s_ind][0])
            y.append(img_indeces[s_ind][1])
        camera_idx += [ind] * len(sampled_indeces)

    ray_indices = torch.stack([torch.tensor(camera_idx), torch.tensor(y), torch.tensor(x)], dim=-1).long().to(torch.device(device))



    return ray_indices

def canny_edge_detector_sampler(
    data_manager: DataManager, n: int, device
) -> torch.Tensor:

    dataset = data_manager.train_dataset

    # determine how many points to sample from each image
    img_weights = np.zeros(len(dataset))
    imgs_shapes = []
    pixels_weights = []

    cameras = data_manager.train_dataset.cameras.to(device)
    H_all = cameras.height.squeeze(-1).cpu().numpy()
    W_all = cameras.width.squeeze(-1).cpu().numpy()

    for idx in range(len(dataset)):

        H, W = int(H_all[idx]), int(W_all[idx])
        imgs_shapes.append((H,W))
        img = dataset.get_numpy_image(idx)

        # convert to gray scale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply canny operator
        edges = cv2.Canny(img, threshold1=25, threshold2=60)

        # Convert to uint8
        gradient_magnitude = cv2.convertScaleAbs(edges)

        if gradient_magnitude.shape != (H, W):
            gradient_magnitude = cv2.resize(gradient_magnitude, (W, H))

        gradient_magnitude = gradient_magnitude.ravel()

        # Normalize
        gradient_magnitude = gradient_magnitude / 255

        # smooth results
        # smoothed = cv2.GaussianBlur(gradient_magnitude, (5, 5), 0).ravel()

        img_weights[idx] = gradient_magnitude.sum()
        # TODO hanhandle case sum is 0
        normalized = gradient_magnitude / gradient_magnitude.sum()
        pixels_weights.append(normalized)

    x = []
    y = []
    camera_idx = []

    img_weights = img_weights / img_weights.sum()
    samples_per_image = np.floor(img_weights * n)

    for ind, weights in enumerate(pixels_weights):

        H,W = imgs_shapes[ind]

        img_indeces = [(x,y) for y in range(H-1, -1, -1) for x in range(W)]

        sample_size = min(len(img_indeces), int(samples_per_image[ind]))

        sampled_indeces = np.random.choice(
            len(img_indeces),
            size = sample_size,
            p = weights,
            replace = False
        )

        for s_ind in sampled_indeces:
            x.append(img_indeces[s_ind][0])
            y.append(img_indeces[s_ind][1])
        camera_idx += [ind] * len(sampled_indeces)

    ray_indices = torch.stack([torch.tensor(camera_idx), torch.tensor(y), torch.tensor(x)], dim=-1).long().to(torch.device(device))


    return ray_indices

def mixed_sampler(data_manager: DataManager, n : int,  share_rnd : float, edge_detector : str, device : str) -> torch.Tensor:

    random_saple_size = int(n * share_rnd)
    edge_sample_size = n - random_saple_size

    rnd_sample = random_sampler(data_manager = data_manager, n =random_saple_size, device=device)

    if edge_detector == "canny":
        edge_sample = canny_edge_detector_sampler(data_manager, edge_sample_size, device)
    else:
        edge_sample = sobel_edge_detector_sampler(data_manager, edge_sample_size, device)

    concatenated = torch.cat([rnd_sample, edge_sample], dim=0)

    return concatenated

def patched_sampler(data_manager, n, device, n_horizontal, n_vertical):
    dataset = data_manager.train_dataset

    img_weights = np.zeros(len(dataset), dtype=np.float64)
    imgs_shapes = []
    sub_images_weights = []

    cameras = data_manager.train_dataset.cameras.to(device)
    H_all = cameras.height.squeeze(-1).cpu().numpy()
    W_all = cameras.width.squeeze(-1).cpu().numpy()

    # ---------- 1) Compute per-image and per-subimage weights ----------
    for img_idx in range(len(dataset)):
        # (1) extract image and info
        H, W = int(H_all[img_idx]), int(W_all[img_idx])
        imgs_shapes.append((H, W))
        img = dataset.get_numpy_image(img_idx)   # H x W x 3 (BGR)

        # (2) convert to gray scale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # (3) total edge energy per image
        edges = cv2.Canny(gray, threshold1=25, threshold2=60)
        gradient_magnitude = cv2.convertScaleAbs(edges)
        if gradient_magnitude.shape != (H, W):
            gradient_magnitude = cv2.resize(gradient_magnitude, (W, H))
        gradient_magnitude = gradient_magnitude.astype(np.float32) / 255.0
        img_weights[img_idx] = gradient_magnitude.sum()

        # (4) split in subpatches, compute edge energy per subpatch
        w_sub = max(1, W // n_horizontal)
        h_sub = max(1, H // n_vertical)

        sub_image_weights = np.zeros(n_horizontal * n_vertical, dtype=np.float64)
        patch_idx = 0

        for i in range(n_vertical):
            for j in range(n_horizontal):
                # y: rows, x: cols
                if (i != n_vertical - 1) and (j != n_horizontal - 1):
                    y0 = i * h_sub
                    y1 = (i + 1) * h_sub
                    x0 = j * w_sub
                    x1 = (j + 1) * w_sub
                elif (i == n_vertical - 1) and (j != n_horizontal - 1):
                    y0 = i * h_sub
                    y1 = H
                    x0 = j * w_sub
                    x1 = (j + 1) * w_sub
                elif (i != n_vertical - 1) and (j == n_horizontal - 1):
                    y0 = i * h_sub
                    y1 = (i + 1) * h_sub
                    x0 = j * w_sub
                    x1 = W
                else:
                    y0 = i * h_sub
                    y1 = H
                    x0 = j * w_sub
                    x1 = W

                sub_image = gray[y0:y1, x0:x1]
                if sub_image.size == 0:
                    sub_image_weights[patch_idx] = 0.0
                    patch_idx += 1
                    continue

                edges_sub = cv2.Canny(sub_image, threshold1=25, threshold2=60)
                grad_sub = cv2.convertScaleAbs(edges_sub).astype(np.float32) / 255.0
                sub_image_weights[patch_idx] = grad_sub.sum()
                patch_idx += 1

        # normalize subimage weights safely
        total_sub = sub_image_weights.sum()
        if total_sub > 0:
            sub_image_weights /= total_sub
        else:
            sub_image_weights[:] = 1.0 / sub_image_weights.size

        sub_images_weights.append(sub_image_weights)

    # ---------- 2) Decide how many samples per image (sum = n exactly) ----------
    total_img = img_weights.sum()
    if total_img > 0:
        img_probs = img_weights / total_img
    else:
        img_probs = np.ones_like(img_weights) / len(img_weights)

    # base allocation by floor
    ideal_per_image = img_probs * n
    samples_per_image = np.floor(ideal_per_image).astype(int)
    deficit = n - samples_per_image.sum()

    if deficit > 0:
        # give leftover samples to images with largest fractional part
        frac = ideal_per_image - samples_per_image
        order = np.argsort(-frac)  # descending
        for k in range(deficit):
            samples_per_image[order[k]] += 1
    elif deficit < 0:
        # remove extra from images with smallest fractional part
        frac = ideal_per_image - samples_per_image
        order = np.argsort(frac)  # ascending
        for k in range(-deficit):
            # only remove where we have something
            for idx in order:
                if samples_per_image[idx] > 0:
                    samples_per_image[idx] -= 1
                    break

    assert samples_per_image.sum() == n, "Internal error: total samples_per_image != n"

    # ---------- 3) Sample rays per image and per subimage ----------
    coords_list = []

    for img_idx, n_samples_img in enumerate(samples_per_image):
        if n_samples_img <= 0:
            continue

        H, W = imgs_shapes[img_idx]
        w_sub = max(1, W // n_horizontal)
        h_sub = max(1, H // n_vertical)

        sub_weights = sub_images_weights[img_idx]

        # distribute n_samples_img across patches
        ideal_per_patch = sub_weights * n_samples_img
        samples_per_patch = np.floor(ideal_per_patch).astype(int)
        deficit_img = n_samples_img - samples_per_patch.sum()

        if deficit_img > 0:
            frac = ideal_per_patch - samples_per_patch
            order = np.argsort(-frac)
            for k in range(deficit_img):
                samples_per_patch[order[k]] += 1
        elif deficit_img < 0:
            frac = ideal_per_patch - samples_per_patch
            order = np.argsort(frac)
            for k in range(-deficit_img):
                for p in order:
                    if samples_per_patch[p] > 0:
                        samples_per_patch[p] -= 1
                        break

        assert samples_per_patch.sum() == n_samples_img, "Internal error: per-image patches != n_samples_img"

        patch_idx = 0
        for i in range(n_vertical):
            for j in range(n_horizontal):
                n_samples_sub = int(samples_per_patch[patch_idx])
                patch_idx += 1

                if n_samples_sub <= 0:
                    continue

                # y: rows, x: cols
                if (i != n_vertical - 1) and (j != n_horizontal - 1):
                    y0 = i * h_sub
                    y1 = (i + 1) * h_sub
                    x0 = j * w_sub
                    x1 = (j + 1) * w_sub
                elif (i == n_vertical - 1) and (j != n_horizontal - 1):
                    y0 = i * h_sub
                    y1 = H
                    x0 = j * w_sub
                    x1 = (j + 1) * w_sub
                elif (i != n_vertical - 1) and (j == n_horizontal - 1):
                    y0 = i * h_sub
                    y1 = (i + 1) * h_sub
                    x0 = j * w_sub
                    x1 = W
                else:
                    y0 = i * h_sub
                    y1 = H
                    x0 = j * w_sub
                    x1 = W

                # safety: ensure ranges are valid
                if y1 <= y0 or x1 <= x0:
                    continue

                ys = torch.randint(low=y0, high=y1, size=(n_samples_sub,), device=device)
                xs = torch.randint(low=x0, high=x1, size=(n_samples_sub,), device=device)
                cam_ids = torch.full((n_samples_sub,), img_idx, dtype=torch.long, device=device)

                coords_list.append(torch.stack([cam_ids, ys, xs], dim=-1))

    if not coords_list:
        raise RuntimeError("patched_sampler produced no samples (check parameters).")

    coords = torch.cat(coords_list, dim=0)

    # final safety: trim or assert
    if coords.shape[0] > n:
        coords = coords[:n]
    elif coords.shape[0] < n:
        # shouldn't happen, but just in case of weird rounding
        print(f"Warning: patched_sampler produced {coords.shape[0]} < {n} samples.")

    return coords