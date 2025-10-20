from pathlib import Path
import torch
import numpy as np

import sys
NS_ROOT = Path(__file__).parent / "submodules" / "nerfstudio"
sys.path.insert(0, str(NS_ROOT))

from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.model_components.ray_generators import RayGenerator
from nerfstudio.field_components.field_heads import FieldHeadNames


# Rbollati for eval_setup fix
_orig_load = torch.load


def _patched_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_load(*args, **kwargs)


torch.load = _patched_load



def main() -> None:

    print("Started job")

    # Folder with the checkpoints
    folder = "/work/courses/dslab/team20/rbollati/running_env/outputs/poster/nerfacto/2025-10-18_013814/config.yml"
    checkpoint_folder = Path(folder)

    # Initialize the pipeline, which contains the datamanager and the model
    config, pipeline, _, _ = eval_setup(checkpoint_folder, test_mode="test")
    dm = pipeline.datamanager
    model = pipeline.model
    device = pipeline.device

    print("Loaded datamanager and model")

    # Randomly sample
    cams = dm.train_dataset.cameras.to(device)
    num_imgs = len(cams)
    num_rays = 100

    img_idx = torch.randint(low=0, high=num_imgs, size=(num_rays,), device=device)

    H = cams.height.squeeze(-1).to(device)
    W = cams.width.squeeze(-1).to(device)
    sel_H = H[img_idx]
    sel_W = W[img_idx]

    y = torch.floor(torch.rand(num_rays, device=device) * sel_H).long()
    x = torch.floor(torch.rand(num_rays, device=device) * sel_W).long()

    ray_indices = torch.stack([img_idx, y, x], dim=-1).long()


    print("Sampled the rays")
    #print(ray_indices)

    ray_generator = RayGenerator(cams).to(device)
    ray_bundle = ray_generator(ray_indices)

    print("Ray bundle produced")
    #print(ray_bundle)

    # Chato fix
    ray_bundle = model.collider(ray_bundle)

    # Pass the ray bundle through the model
    ray_samples, weights_list, ray_samples_list = model.proposal_sampler(ray_bundle, density_fns=model.density_fns)
    field_outputs = model.field.forward(ray_samples, compute_normals=model.config.predict_normals)
    weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])

    print("Passed the ray bundle through the model")

    # Compute the final positions according to the paper
    # First compute the transmittances
    cum_inc = torch.cumsum(weights, dim=1)
    cum_exc = cum_inc - weights
    T_per_sample = 1.0 - cum_exc

    # Then find the positions where the accumulated transmittance is >0.5
    positions = ray_samples.frustums.get_positions()
    T = T_per_sample.squeeze(-1)
    mask = (T > 0.5)
    count = mask.sum(dim=1)
    last_idx = (count - 1).clamp(min=0)

    radsplat_positions = positions[torch.arange(num_rays, device=positions.device), last_idx, :]

    print(radsplat_positions)

if __name__ == "__main__":
    main()
