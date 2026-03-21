import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image
from pycolmap import SceneManager
from tqdm import tqdm
from typing_extensions import assert_never

from submodules.gsplat.examples.datasets.normalize import (
    align_principal_axes,
    similarity_from_cameras,
    transform_cameras,
    transform_points,
)
from submodules.gsplat.examples.datasets.colmap import _get_rel_paths


def _norm_name(name: str) -> str:
    base = os.path.basename(name).lower()
    stem, _ = os.path.splitext(base)
    return stem


class Parser:
    """COLMAP parser."""

    def __init__(
        self,
        data_dir: str,
        factor: int = 1,
        nerf_samples_factor: Optional[int] = None,
        normalize: bool = False,
        test_every: int = 8,
        split_payload_path: str = "",
    ):
        self.data_dir = data_dir
        self.factor = factor
        self.nerf_samples_factor = factor if nerf_samples_factor is None else nerf_samples_factor
        self.normalize = normalize
        self.test_every = test_every
        self.split_payload_path = split_payload_path
        self.weight_dir = None
        self.nerf_weight_dir = None

        colmap_dir = os.path.join(data_dir, "sparse/0/")
        if not os.path.exists(colmap_dir):
            colmap_dir = os.path.join(data_dir, "sparse")
        assert os.path.exists(
            colmap_dir
        ), f"COLMAP directory {colmap_dir} does not exist."

        manager = SceneManager(colmap_dir)
        manager.load_cameras()
        manager.load_images()
        manager.load_points3D()

        # Extract extrinsic matrices in world-to-camera format.
        imdata = manager.images
        w2c_mats = []
        camera_ids = []
        Ks_dict = dict()
        params_dict = dict()
        imsize_dict = dict()  # width, height
        mask_dict = dict()
        bottom = np.array([0, 0, 0, 1]).reshape(1, 4)
        for k in imdata:
            im = imdata[k]
            rot = im.R()
            trans = im.tvec.reshape(3, 1)
            w2c = np.concatenate([np.concatenate([rot, trans], 1), bottom], axis=0)
            w2c_mats.append(w2c)

            # support different camera intrinsics
            camera_id = im.camera_id
            camera_ids.append(camera_id)

            # camera intrinsics
            cam = manager.cameras[camera_id]
            fx, fy, cx, cy = cam.fx, cam.fy, cam.cx, cam.cy
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            K[:2, :] /= factor
            Ks_dict[camera_id] = K

            # Get distortion parameters.
            type_ = cam.camera_type
            if type_ == 0 or type_ == "SIMPLE_PINHOLE":
                params = np.empty(0, dtype=np.float32)
                camtype = "perspective"
            elif type_ == 1 or type_ == "PINHOLE":
                params = np.empty(0, dtype=np.float32)
                camtype = "perspective"
            if type_ == 2 or type_ == "SIMPLE_RADIAL":
                params = np.array([cam.k1, 0.0, 0.0, 0.0], dtype=np.float32)
                camtype = "perspective"
            elif type_ == 3 or type_ == "RADIAL":
                params = np.array([cam.k1, cam.k2, 0.0, 0.0], dtype=np.float32)
                camtype = "perspective"
            elif type_ == 4 or type_ == "OPENCV":
                params = np.array([cam.k1, cam.k2, cam.p1, cam.p2], dtype=np.float32)
                camtype = "perspective"
            elif type_ == 5 or type_ == "OPENCV_FISHEYE":
                params = np.array([cam.k1, cam.k2, cam.k3, cam.k4], dtype=np.float32)
                camtype = "fisheye"
            assert (
                camtype == "perspective" or camtype == "fisheye"
            ), f"Only perspective and fisheye cameras are supported, got {type_}"

            params_dict[camera_id] = params
            imsize_dict[camera_id] = (cam.width // factor, cam.height // factor)
            mask_dict[camera_id] = None
        print(
            f"[Parser] {len(imdata)} images, taken by {len(set(camera_ids))} cameras."
        )

        if len(imdata) == 0:
            raise ValueError("No images found in COLMAP.")
        if not (type_ == 0 or type_ == 1):
            print("Warning: COLMAP Camera is not PINHOLE. Images have distortion.")

        w2c_mats = np.stack(w2c_mats, axis=0)

        # Convert extrinsics to camera-to-world.
        camtoworlds = np.linalg.inv(w2c_mats)

        # Image names from COLMAP. No need for permuting the poses according to
        # image names anymore.
        image_names = [imdata[k].name for k in imdata]

        # Previous Nerf results were generated with images sorted by filename,
        # ensure metrics are reported on the same test set.
        inds = np.argsort(image_names)
        image_names = [image_names[i] for i in inds]
        camtoworlds = camtoworlds[inds]
        camera_ids = [camera_ids[i] for i in inds]

        # Load extended metadata. Used by Bilarf dataset.
        self.extconf = {
            "spiral_radius_scale": 1.0,
            "no_factor_suffix": False,
        }
        extconf_file = os.path.join(data_dir, "ext_metadata.json")
        if os.path.exists(extconf_file):
            with open(extconf_file) as f:
                self.extconf.update(json.load(f))

        # Load bounds if possible (only used in forward facing scenes).
        self.bounds = np.array([0.01, 1.0])
        posefile = os.path.join(data_dir, "poses_bounds.npy")
        if os.path.exists(posefile):
            self.bounds = np.load(posefile)[:, -2:]

        # Load images.
        colmap_image_dir = os.path.join(data_dir, "images")
        for d in [colmap_image_dir]:
            if not os.path.exists(d):
                raise ValueError(f"Image folder {d} does not exist.")

        def _build_image_paths_for_factor(factor_value: int):
            if factor_value > 1 and not self.extconf["no_factor_suffix"]:
                image_dir_suffix = f"_{factor_value}"
            else:
                image_dir_suffix = ""
            image_dir = os.path.join(data_dir, "images" + image_dir_suffix)
            if not os.path.exists(image_dir):
                raise ValueError(f"Image folder {image_dir} does not exist.")

            # Downsampled images may have different names vs images used for COLMAP.
            colmap_files = sorted(_get_rel_paths(colmap_image_dir))
            image_files = sorted(_get_rel_paths(image_dir))
            # Keep using the existing images_<factor> directory directly.
            # Do not auto-generate images_<factor>_png caches, because we may have
            # auxiliary factor-specific assets (e.g. weight maps) aligned to the
            # existing folder naming/shape conventions.
            colmap_to_image = dict(zip(colmap_files, image_files))
            image_paths = [os.path.join(image_dir, colmap_to_image[f]) for f in image_names]
            return image_dir, image_paths

        image_dir, image_paths = _build_image_paths_for_factor(self.factor)
        if self.nerf_samples_factor == self.factor:
            nerf_image_paths = image_paths
        else:
            _, nerf_image_paths = _build_image_paths_for_factor(self.nerf_samples_factor)
        self.weight_dir = self._resolve_weight_dir(self.factor)
        self.nerf_weight_dir = self._resolve_weight_dir(self.nerf_samples_factor)

        # 3D points and {image_name -> [point_idx]}
        points = manager.points3D.astype(np.float32)
        points_err = manager.point3D_errors.astype(np.float32)
        points_rgb = manager.point3D_colors.astype(np.uint8)
        point_indices = dict()

        image_id_to_name = {v: k for k, v in manager.name_to_image_id.items()}
        for point_id, data in manager.point3D_id_to_images.items():
            for image_id, _ in data:
                image_name = image_id_to_name[image_id]
                point_idx = manager.point3D_id_to_point3D_idx[point_id]
                point_indices.setdefault(image_name, []).append(point_idx)
        point_indices = {
            k: np.array(v).astype(np.int32) for k, v in point_indices.items()
        }

        # Normalize the world space.
        if normalize:
            T1 = similarity_from_cameras(camtoworlds)
            camtoworlds = transform_cameras(T1, camtoworlds)
            points = transform_points(T1, points)

            T2 = align_principal_axes(points)
            camtoworlds = transform_cameras(T2, camtoworlds)
            points = transform_points(T2, points)

            transform = T2 @ T1

            # Fix for up side down. We assume more points towards
            # the bottom of the scene which is true when ground floor is
            # present in the images.
            if np.median(points[:, 2]) > np.mean(points[:, 2]):
                # rotate 180 degrees around x axis such that z is flipped
                T3 = np.array(
                    [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, -1.0, 0.0, 0.0],
                        [0.0, 0.0, -1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                )
                camtoworlds = transform_cameras(T3, camtoworlds)
                points = transform_points(T3, points)
                transform = T3 @ transform
        else:
            transform = np.eye(4)

        self.image_names = image_names  # List[str], (num_images,)
        self.image_paths = image_paths  # List[str], (num_images,)
        self.image_paths_real = image_paths  # List[str], (num_images,)
        self.image_paths_nerf = nerf_image_paths  # List[str], (num_images,)
        self.camtoworlds = camtoworlds  # np.ndarray, (num_images, 4, 4)
        self.camera_ids = camera_ids  # List[int], (num_images,)
        self.Ks_dict = Ks_dict  # Dict of camera_id -> K
        self.params_dict = params_dict  # Dict of camera_id -> params
        self.imsize_dict = imsize_dict  # Dict of camera_id -> (width, height)
        self.mask_dict = mask_dict  # Dict of camera_id -> mask
        self.points = points  # np.ndarray, (num_points, 3)
        self.points_err = points_err  # np.ndarray, (num_points,)
        self.points_rgb = points_rgb  # np.ndarray, (num_points, 3)
        self.point_indices = point_indices  # Dict[str, np.ndarray], image_name -> [M,]
        self.transform = transform  # np.ndarray, (4, 4)

        # load one image to check the size. In the case of tanksandtemples dataset, the
        # intrinsics stored in COLMAP corresponds to 2x upsampled images.
        actual_image = imageio.imread(self.image_paths[0])[..., :3]
        actual_height, actual_width = actual_image.shape[:2]
        colmap_width, colmap_height = self.imsize_dict[self.camera_ids[0]]
        s_height, s_width = actual_height / colmap_height, actual_width / colmap_width
        for camera_id, K in self.Ks_dict.items():
            K[0, :] *= s_width
            K[1, :] *= s_height
            self.Ks_dict[camera_id] = K
            width, height = self.imsize_dict[camera_id]
            self.imsize_dict[camera_id] = (int(width * s_width), int(height * s_height))

        # undistortion
        self.mapx_dict = dict()
        self.mapy_dict = dict()
        self.roi_undist_dict = dict()
        for camera_id in self.params_dict.keys():
            params = self.params_dict[camera_id]
            if len(params) == 0:
                continue  # no distortion
            assert camera_id in self.Ks_dict, f"Missing K for camera {camera_id}"
            assert (
                camera_id in self.params_dict
            ), f"Missing params for camera {camera_id}"
            K = self.Ks_dict[camera_id]
            width, height = self.imsize_dict[camera_id]

            if camtype == "perspective":
                K_undist, roi_undist = cv2.getOptimalNewCameraMatrix(
                    K, params, (width, height), 0
                )
                mapx, mapy = cv2.initUndistortRectifyMap(
                    K, params, None, K_undist, (width, height), cv2.CV_32FC1
                )
                mask = None
            elif camtype == "fisheye":
                fx = K[0, 0]
                fy = K[1, 1]
                cx = K[0, 2]
                cy = K[1, 2]
                grid_x, grid_y = np.meshgrid(
                    np.arange(width, dtype=np.float32),
                    np.arange(height, dtype=np.float32),
                    indexing="xy",
                )
                x1 = (grid_x - cx) / fx
                y1 = (grid_y - cy) / fy
                theta = np.sqrt(x1**2 + y1**2)
                r = (
                    1.0
                    + params[0] * theta**2
                    + params[1] * theta**4
                    + params[2] * theta**6
                    + params[3] * theta**8
                )
                mapx = (fx * x1 * r + width // 2).astype(np.float32)
                mapy = (fy * y1 * r + height // 2).astype(np.float32)

                # Use mask to define ROI
                mask = np.logical_and(
                    np.logical_and(mapx > 0, mapy > 0),
                    np.logical_and(mapx < width - 1, mapy < height - 1),
                )
                y_indices, x_indices = np.nonzero(mask)
                y_min, y_max = y_indices.min(), y_indices.max() + 1
                x_min, x_max = x_indices.min(), x_indices.max() + 1
                mask = mask[y_min:y_max, x_min:x_max]
                K_undist = K.copy()
                K_undist[0, 2] -= x_min
                K_undist[1, 2] -= y_min
                roi_undist = [x_min, y_min, x_max - x_min, y_max - y_min]
            else:
                assert_never(camtype)

            self.mapx_dict[camera_id] = mapx
            self.mapy_dict[camera_id] = mapy
            self.Ks_dict[camera_id] = K_undist
            self.roi_undist_dict[camera_id] = roi_undist
            self.imsize_dict[camera_id] = (roi_undist[2], roi_undist[3])
            self.mask_dict[camera_id] = mask

        # size of the scene measured by cameras
        camera_locations = camtoworlds[:, :3, 3]
        scene_center = np.mean(camera_locations, axis=0)
        dists = np.linalg.norm(camera_locations - scene_center, axis=1)
        self.scene_scale = np.max(dists)

        # Optional split override from nerf_step payload (.pt).
        self.train_indices_override = None
        self.val_indices_override = None
        self._load_split_overrides()

    def _resolve_weight_dir(self, factor_value: int) -> Optional[str]:
        if factor_value > 1 and not self.extconf["no_factor_suffix"]:
            suffix = f"_{factor_value}"
        else:
            suffix = ""
        weight_dir = os.path.join(self.data_dir, "weights_nerf_samples" + suffix)
        if os.path.isdir(weight_dir):
            return weight_dir
        return None

    def resolve_depth_dir(self, prefix: str, factor_value: int) -> Optional[str]:
        if factor_value > 1 and not self.extconf["no_factor_suffix"]:
            suffix = f"_{factor_value}"
        else:
            suffix = ""
        depth_dir = os.path.join(self.data_dir, prefix + suffix)
        if os.path.isdir(depth_dir):
            return depth_dir
        return None

    def _load_split_overrides(self) -> None:
        if not self.split_payload_path:
            return
        if not os.path.isfile(self.split_payload_path):
            print(
                f"[Parser] split payload not found at {self.split_payload_path}. "
                "Falling back to test_every split."
            )
            return

        payload = torch.load(self.split_payload_path, map_location="cpu")

        # Preferred: explicit mapping from relative filename to split label.
        split_map = payload.get("split_by_image_rel", None)

        # Backward-compatible fallback: train/val filename lists.
        if split_map is None:
            split_map = {}
            train_rel = payload.get("train_image_filenames_rel", payload.get("image_filenames_rel", []))
            val_rel = payload.get("val_image_filenames_rel", [])
            for n in train_rel:
                split_map[str(n)] = "train"
            val_name = payload.get("val_split_name", "val")
            for n in val_rel:
                split_map.setdefault(str(n), val_name)

        if not split_map:
            print("[Parser] split payload has no split info. Falling back to test_every split.")
            return

        # Build robust lookup: exact + basename + stem.
        split_lookup = {}
        for key, value in split_map.items():
            key_str = str(key)
            split_lookup[key_str] = value
            split_lookup[os.path.basename(key_str)] = value
            split_lookup[_norm_name(key_str)] = value

        train_idx = []
        val_idx = []
        unmatched = []
        unmatched_nerf_to_train = 0
        for i, name in enumerate(self.image_names):
            split_value = (
                split_lookup.get(name)
                or split_lookup.get(os.path.basename(name))
                or split_lookup.get(_norm_name(name))
            )
            if split_value is None:
                # Synthetic NeRF samples are generated after nerf_step payload creation,
                # so they are expected to be unmatched here. Put them in train by default.
                if os.path.basename(name).startswith("nerf_sample_"):
                    train_idx.append(i)
                    unmatched_nerf_to_train += 1
                    continue
                unmatched.append(name)
                continue
            if str(split_value).lower() == "train":
                train_idx.append(i)
            else:
                val_idx.append(i)

        if len(train_idx) == 0:
            print("[Parser] split payload did not match any train images. Falling back to test_every split.")
            return

        self.train_indices_override = np.array(sorted(set(train_idx)), dtype=np.int64)
        self.val_indices_override = np.array(sorted(set(val_idx)), dtype=np.int64)
        print(
            f"[Parser] using split payload '{self.split_payload_path}': "
            f"train={len(self.train_indices_override)}, val={len(self.val_indices_override)}, "
            f"unmatched={len(unmatched)}, unmatched_nerf_to_train={unmatched_nerf_to_train}"
        )


class Dataset:
    """A simple dataset class."""

    def __init__(
        self,
        parser: Parser,
        split: str = "train",
        patch_size: Optional[int] = None,
        load_depths: bool = False,
        use_nerf_factor_for_real: bool = False,
        load_nerf_depths: bool = False,
        nerf_depth_prefix: str = "depths_nerf",
        nerf_depth_factor: Optional[int] = None,
        nerf_depth_include_real: bool = True,
        nerf_depth_include_nerf_samples: bool = True,
    ):
        self.parser = parser
        self.split = split
        self.patch_size = patch_size
        self.load_depths = load_depths
        self.use_nerf_factor_for_real = use_nerf_factor_for_real
        self.load_nerf_depths = load_nerf_depths
        self.nerf_depth_prefix = nerf_depth_prefix
        self.nerf_depth_factor = nerf_depth_factor
        self.nerf_depth_include_real = nerf_depth_include_real
        self.nerf_depth_include_nerf_samples = nerf_depth_include_nerf_samples
        if self.parser.train_indices_override is not None:
            if split == "train":
                self.indices = self.parser.train_indices_override
            else:
                self.indices = self.parser.val_indices_override
        else:
            indices = np.arange(len(self.parser.image_names))
            if split == "train":
                self.indices = indices[indices % self.parser.test_every != 0]
            else:
                self.indices = indices[indices % self.parser.test_every == 0]
        self._warned_missing_weight = set()
        self._warned_mixed_undistort = set()
        self._warned_missing_nerf_depth = set()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        index = self.indices[item]
        image_name = self.parser.image_names[index]
        is_nerf_sample = os.path.basename(image_name).startswith("nerf_sample_")
        image_path = (
            self.parser.image_paths_nerf[index]
            if (is_nerf_sample or self.use_nerf_factor_for_real) and hasattr(self.parser, "image_paths_nerf")
            else self.parser.image_paths[index]
        )
        image = imageio.imread(image_path)[..., :3]
        camera_id = self.parser.camera_ids[index]
        K = self.parser.Ks_dict[camera_id].copy()  # undistorted K
        params = self.parser.params_dict[camera_id]
        camtoworlds = self.parser.camtoworlds[index]
        mask = self.parser.mask_dict[camera_id]
        weight_map = np.ones(image.shape[:2], dtype=np.float32)
        nerf_depth = np.zeros(image.shape[:2], dtype=np.float32)
        has_nerf_depth = False

        # Reference size for Ks_dict[camera_id] (often undistorted/cropped size).
        base_w, base_h = self.parser.imsize_dict[camera_id]

        sample_weight_dir = self.parser.nerf_weight_dir if is_nerf_sample else self.parser.weight_dir
        if is_nerf_sample and sample_weight_dir is not None:
            stem = os.path.splitext(os.path.basename(image_name))[0]
            weight_path = os.path.join(sample_weight_dir, f"{stem}.npy")
            if os.path.isfile(weight_path):
                weight_map = np.load(weight_path).astype(np.float32)
                if weight_map.shape != image.shape[:2]:
                    raise RuntimeError(
                        f"Weight map shape mismatch for {image_name}: "
                        f"weights={weight_map.shape}, image={image.shape[:2]}"
                    )
            elif weight_path not in self._warned_missing_weight:
                print(f"[Dataset] missing weight map for {image_name}: {weight_path}. Using uniform weights.")
                self._warned_missing_weight.add(weight_path)

        if self.load_nerf_depths:
            use_sample = (is_nerf_sample and self.nerf_depth_include_nerf_samples) or (
                (not is_nerf_sample) and self.nerf_depth_include_real
            )
            if use_sample:
                if self.nerf_depth_factor is not None:
                    depth_factor = int(self.nerf_depth_factor)
                else:
                    depth_factor = (
                        self.parser.nerf_samples_factor
                        if (is_nerf_sample or self.use_nerf_factor_for_real)
                        else self.parser.factor
                    )
                depth_dir = self.parser.resolve_depth_dir(self.nerf_depth_prefix, depth_factor)
                if depth_dir is not None:
                    depth_path = os.path.join(
                        depth_dir, str(Path(image_name).with_suffix(".npy"))
                    )
                    if os.path.isfile(depth_path):
                        nerf_depth = np.load(depth_path).astype(np.float32)
                        if nerf_depth.shape != image.shape[:2]:
                            nerf_depth = cv2.resize(
                                nerf_depth,
                                (image.shape[1], image.shape[0]),
                                interpolation=cv2.INTER_LINEAR,
                            )
                        has_nerf_depth = True
                    elif depth_path not in self._warned_missing_nerf_depth:
                        print(
                            f"[Dataset] missing nerf depth for {image_name}: {depth_path}. "
                            "Depth supervision disabled for this sample."
                        )
                        self._warned_missing_nerf_depth.add(depth_path)

        if len(params) > 0:
            # Images are distorted. Undistort them.
            mapx, mapy = (
                self.parser.mapx_dict[camera_id],
                self.parser.mapy_dict[camera_id],
            )
            if mapx.shape[:2] != image.shape[:2]:
                warn_key = f"{camera_id}:{image.shape[1]}x{image.shape[0]}"
                if warn_key not in self._warned_mixed_undistort:
                    print(
                        f"[Dataset] skipping undistort for mixed-resolution sample '{image_name}' "
                        f"(image={image.shape[:2]}, map={mapx.shape[:2]})."
                    )
                    self._warned_mixed_undistort.add(warn_key)
            else:
                image = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)
                weight_map = cv2.remap(weight_map, mapx, mapy, cv2.INTER_LINEAR)
                if self.load_nerf_depths and has_nerf_depth:
                    nerf_depth = cv2.remap(nerf_depth, mapx, mapy, cv2.INTER_LINEAR)
                x, y, w, h = self.parser.roi_undist_dict[camera_id]
                image = image[y : y + h, x : x + w]
                weight_map = weight_map[y : y + h, x : x + w]
                if self.load_nerf_depths and has_nerf_depth:
                    nerf_depth = nerf_depth[y : y + h, x : x + w]

        # Final intrinsics scaling to actual sample resolution.
        # IMPORTANT: do this after optional undistort/crop, because Ks_dict already
        # corresponds to the undistorted ROI size for distorted cameras.
        img_h, img_w = image.shape[:2]
        if img_w != base_w or img_h != base_h:
            sx = img_w / float(base_w)
            sy = img_h / float(base_h)
            K[0, :] *= sx
            K[1, :] *= sy

        if self.patch_size is not None:
            # Random crop.
            h, w = image.shape[:2]
            x = np.random.randint(0, max(w - self.patch_size, 1))
            y = np.random.randint(0, max(h - self.patch_size, 1))
            image = image[y : y + self.patch_size, x : x + self.patch_size]
            weight_map = weight_map[y : y + self.patch_size, x : x + self.patch_size]
            if self.load_nerf_depths and has_nerf_depth:
                nerf_depth = nerf_depth[y : y + self.patch_size, x : x + self.patch_size]
            K[0, 2] -= x
            K[1, 2] -= y

        data = {
            "K": torch.from_numpy(K).float(),
            "camtoworld": torch.from_numpy(camtoworlds).float(),
            "image": torch.from_numpy(image).float(),
            "image_id": item,  # the index of the image in the dataset
            "is_nerf_sample": torch.tensor(is_nerf_sample, dtype=torch.bool),
            "weight_map": torch.from_numpy(weight_map).float(),
            "nerf_depth": torch.from_numpy(nerf_depth).float(),
            "has_nerf_depth": torch.tensor(has_nerf_depth, dtype=torch.bool),
        }
        if mask is not None:
            data["mask"] = torch.from_numpy(mask).bool()

        if self.load_depths:
            # projected points to image plane to get depths
            worldtocams = np.linalg.inv(camtoworlds)
            point_indices = self.parser.point_indices.get(image_name, None)
            if point_indices is None:
                data["points"] = torch.empty((0, 2), dtype=torch.float32)
                data["depths"] = torch.empty((0,), dtype=torch.float32)
                return data
            points_world = self.parser.points[point_indices]
            points_cam = (worldtocams[:3, :3] @ points_world.T + worldtocams[:3, 3:4]).T
            points_proj = (K @ points_cam.T).T
            points = points_proj[:, :2] / points_proj[:, 2:3]  # (M, 2)
            depths = points_cam[:, 2]  # (M,)
            # filter out points outside the image
            selector = (
                (points[:, 0] >= 0)
                & (points[:, 0] < image.shape[1])
                & (points[:, 1] >= 0)
                & (points[:, 1] < image.shape[0])
                & (depths > 0)
            )
            points = points[selector]
            depths = depths[selector]
            data["points"] = torch.from_numpy(points).float()
            data["depths"] = torch.from_numpy(depths).float()

        return data
