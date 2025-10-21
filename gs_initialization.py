import warnings

from nerf_models import Nerfacto
from point_samplers import random_sampler
from gs_initializer import Initializer

warnings.filterwarnings(
    "ignore",
    message="Using a non-tuple sequence for multidimensional indexing is deprecated",
)

if __name__ == "__main__":
    folder = "/work/courses/dslab/team20/rbollati/running_env/outputs/poster/nerfacto/2025-10-18_013814/config.yml"

    
    model = Nerfacto(folder)

    coords = random_sampler(model.pipeline.datamanager, 1000, model.device)
    rays = model.create_rays(coords)

    sampled = model.sample_points(rays, 100)
    pixels_prediction, field_outputs, weights  = model.evaluate_points(sampled)
    gs_initializer = Initializer(weights, field_outputs)

    trasmittance = gs_initializer.compute_transmittance()
    initial_position = gs_initializer.compute_inital_positions(trasmittance, 0.5)

    print(initial_position)
