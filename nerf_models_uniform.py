import torchvision
import warnings
import logging
import sys
import numpy as np
import torch

_orig_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_load(*args, **kwargs)
torch.load = _patched_load

from pathlib import Path, PurePath
NS_ROOT = Path(__file__).parent / "submodules" / "nerfstudio"
sys.path.insert(0, str(NS_ROOT))

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.model_components.ray_samplers import SpacedSampler, UniformSampler
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.model_components.ray_generators import RayGenerator
from nerfstudio.model_components.ray_samplers import ProposalNetworkSampler, UniformSampler

warnings.filterwarnings(
    "ignore",
    message="Using a non-tuple sequence for multidimensional indexing is deprecated",
)

class Nerfacto_uniform:

    def __init__(self, config_file_path: str):
        """
        This class is an interface to the nerfacto model from nerfstudio
        
        :Param config_file_path -> (str) path to the config file
        """

        logging.info("Model interface initialization")

        self.config_file_path = Path(config_file_path)

        # load model and pipeline
        self.pipeline, self.model = self._load_model()
        # set device based on configs (GPU / CPU)
        self.device = next(self.model.parameters()).device

        # Temporaneo
        self.sampler = UniformSampler(
            num_samples=100,
            train_stratified=False,   # optional, to avoid randomness in eval
            single_jitter=False,
        )

        logging.info(f"Model interface initialized succesfully - device: {self.device}")

        pass

    def _load_model(self):
        """
        This function load the model from the checkpoint and load it into
        the class instance
        """
        _, pipeline, _, _ = eval_setup(self.config_file_path, test_mode="test")
        return pipeline, pipeline.model

    def render_camera(self, camera_index: int):
        """
        Render an image from the traini dataset

        :Param camera_index -> (int) the index of the camera from the train set to render
        """

        # Check train_set is not none 
        assert self.pipeline.datamanager.train_dataset

        # Get camera from train set data
        camera = self.pipeline.datamanager.train_dataset.cameras[camera_index]

        print(camera)

        # Run model on specific camera
        logging.info(f"Generating output for camera {camera_index}")
        outputs = self.model.get_outputs_for_camera(camera)
        image = outputs["rgb"]

        logging.info(f"Image generated")

        return image

    def create_rays(self, ray_indices : torch.Tensor):
        """
        Generate rays for specific pixel coordinates and camera index

        :Param camera_index -> (int) index of the camera to use from the train set
        :Param coords -> (torch.Tensor) a tensor shape (n, 2) containing the coordinates of the n pixels to render

        :Return RayBundle
        """

        assert self.pipeline.datamanager.train_dataset

        # get cameras from training set
        cameras: Cameras = self.pipeline.datamanager.train_dataset.cameras
        # make sure the cameras are on the same device
        cameras = cameras.to(device=self.device)


        # generate rays
        ray_generator = RayGenerator(cameras).to(self.device)
        ray_bundle = ray_generator(ray_indices)
        # use the model collider to assign near / fars values
        rays = self.model.collider.set_nears_and_fars(ray_bundle)

        return rays.to(self.device)
    
    def sample_points_uniform(self, rays: RayBundle, num_samples: int = 100, max_dist: float = 8.0):
        rays = rays.to(self.device)

        # linspace in *euclidean* depth
        bins = torch.linspace(0.0, max_dist, num_samples + 1, device=self.device)[None, :]  # [1, N+1]
        euclidean_bins = bins.expand(rays.origins.shape[0], -1)  # [num_rays, N+1]

        ray_samples = rays.get_ray_samples(
            bin_starts=euclidean_bins[..., :-1, None],
            bin_ends=euclidean_bins[..., 1:, None],
        )

        return ray_samples.to(self.device)

    def evaluate_points(self, ray_samples : RaySamples):
        """
        Evaluate a RaySamples and returns the value of the pixel and the single points
        evaluation

        :Param ray_samples -> (RaySamples) samples along the ray to evaluate

        :Return field_outputs, outputs
        """

        ray_samples = ray_samples.to(self.device)

        # Additional step to make sure tensors stay on self.device
        with torch.no_grad():

            field_outputs = self.model.field.forward(ray_samples, compute_normals=self.model.config.predict_normals)

            weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])

            rgb = self.model.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
            depth = self.model.renderer_depth(weights=weights, ray_samples=ray_samples)
            expected_depth = self.model.renderer_expected_depth(weights=weights, ray_samples=ray_samples)
            accumulation = self.model.renderer_accumulation(weights=weights)

        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
            "expected_depth": expected_depth,
        }

        return outputs, field_outputs, weights

