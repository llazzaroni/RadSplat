from submodules.nerfstudio.nerfstudio.cameras.cameras import Cameras
from submodules.nerfstudio.nerfstudio.cameras.rays import RayBundle, RaySamples
from submodules.nerfstudio.nerfstudio.model_components.ray_samplers import SpacedSampler, UniformSampler
from submodules.nerfstudio.nerfstudio.field_components.field_heads import FieldHeadNames

# from submodules.nerfstudio.nerfstudio.field_components.field_heads import FieldHeadNames
import torchvision
import warnings

warnings.filterwarnings(
    "ignore",
    message="Using a non-tuple sequence for multidimensional indexing is deprecated",
)
from pathlib import Path
import numpy as np
import torch

_orig_load = torch.load


def _patched_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_load(*args, **kwargs)


torch.load = _patched_load
from submodules.nerfstudio.nerfstudio.utils.eval_utils import eval_setup


class Model:

    def __init__(self, checkpoint_folder: str):

        print("###################################### initi model")

        self.checkpoint_folder = Path(checkpoint_folder)
        self.pipeline, self.model = self._load_model()
        self._load_model()
        self.device = next(self.model.parameters()).device

        print("###################################### model loaded")

        pass

    def _load_model(self):
        _, pipeline, _, _ = eval_setup(self.checkpoint_folder, test_mode="test")
        return pipeline, pipeline.model

    def render_camera(self, camera_index: int):
        """
        This function is used to render an entire image from the traini dataset

        :Param camera_index -> (int) the index of the camera from the train set to render
        """

        print("###################################### rendering image")
        if self.pipeline.datamanager.train_dataset:
            camera = self.pipeline.datamanager.train_dataset.cameras[0]

            print("###################################### camera")
            print(camera)
            print("###################################### run model")

            outputs = self.model.get_outputs_for_camera(camera)
            image = outputs["rgb"]

            return image

        return None

    def create_rays(self):

        if self.pipeline.datamanager.train_dataset:

            cameras: Cameras = self.pipeline.datamanager.train_dataset.cameras
            cameras = cameras.to(device=self.device)

            i = 0

            coords = torch.tensor(
                [
                    [10, 20],  # x, y
                    [100, 50],
                    [200, 120],
                ],
                dtype=torch.float32,
                device=self.device
            )

            rays = cameras.generate_rays(camera_indices=i, coords = coords)
            rays = self.model.collider.set_nears_and_fars(rays)

            return rays.to(self.device)

    def sample_rays(self, rays : RayBundle):

        rays = rays.to(self.device)
        sampler = SpacedSampler(
                    spacing_fn= lambda x : x,
                    spacing_fn_inv= lambda x : x,
                    num_samples= 10
                ) 
        
        sampled = sampler.generate_ray_samples(rays, 10)
        print(sampled)
        return sampled.to(self.device)

    def evaluate_points(self, ray_samples : RaySamples):

        ray_samples = ray_samples.to(self.device)

        with torch.no_grad():

            field_outputs = self.model.field.forward(ray_samples, compute_normals=False)

            print(field_outputs)

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

        return outputs




if __name__ == "__main__":
    folder = "/work/courses/dslab/team20/rbollati/running_env/outputs/poster/nerfacto/2025-10-18_013814/config.yml"

    pipeline = Model(folder)
    rays = pipeline.create_rays()
    sampled = pipeline.sample_rays(rays)
    evaluated = pipeline.evaluate_points(sampled)
    # out_path = "render.png"
    # torchvision.utils.save_image(image.permute(2, 0, 1), out_path)
    print(evaluated)
