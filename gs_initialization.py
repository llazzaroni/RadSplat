import warnings
import torch

from nerf_models import Nerfacto
from point_samplers import random_sampler
from gs_initializer import Initializer

warnings.filterwarnings(
    "ignore",
    message="Using a non-tuple sequence for multidimensional indexing is deprecated",
)

if __name__ == "__main__":
    folder = "/work/courses/dslab/team20/rbollati/running_env/outputs/poster/nerfacto/2025-10-18_013814/config.yml"

    
    print('########### loading model')
    model = Nerfacto(folder)

    print('########### sampling rays')
    coords = random_sampler(model.pipeline.datamanager, 1000, model.device)
    rays = model.create_rays(coords)

    print('########### sampling points')

    sampled = model.sample_points(rays)
    outputs, field_outputs, weights  = model.evaluate_points(sampled)

    print('########### init gs')
    gs_initializer = Initializer(weights, sampled)

    print('########### compute trasmittance')
    trasmittance = gs_initializer.compute_transmittance()

    print('########### computer initial_position')
    initial_position = gs_initializer.compute_inital_positions(trasmittance, 0.5)

    # Concatenate rgb to the positions
    xyzrgb_batch = torch.cat([initial_position, outputs['rgb']], dim=-1)

    print('########### initial positions')
    torch.save(xyzrgb_batch, 'test_from_refactor.pt')

    print(initial_position)
