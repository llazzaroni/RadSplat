import warnings
import math
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

    N_RAYS = 1_000_000
    BATCH_SIZE = 5_000

    n_batches = math.ceil(N_RAYS / BATCH_SIZE)

    
    print('########### loading model')
    model = Nerfacto(folder)

    # saple points
    coords = random_sampler(model.pipeline.datamanager, N_RAYS, model.device)

    xyzrgb_chunks = []
    for b in range(n_batches):
        print(f"worinng of chink {b}/{n_batches}")

        # get initial and final index of the index rays to query
        s = b * BATCH_SIZE
        e = min((b + 1) * BATCH_SIZE, N_RAYS)
        if s >= e:
            break
        batch_rays_indexes = coords[s:e, :]
        B = batch_rays_indexes.shape[0]

        print('########### sampling rays')
        rays = model.create_rays(batch_rays_indexes)

        print('########### sampling points')

        sampled = model.sample_points(rays)
        outputs, field_outputs, weights  = model.evaluate_points(sampled)

        print('########### init gs')
        gs_initializer = Initializer(weights, sampled)

        print('########### compute trasmittance')
        trasmittance = gs_initializer.compute_transmittance()

        print('########### computer initial_position')
        initial_position = gs_initializer.compute_inital_positions(trasmittance, 0.5)

        xyzrgb_batch = torch.cat([initial_position, outputs['rgb']], dim=-1)

        xyzrgb_chunks.append(xyzrgb_batch.detach().cpu())


    xyzrgb = torch.cat(xyzrgb_chunks, dim=0)

    print('########### saving initial positions')
    torch.save(xyzrgb, 'big_sample.pt')
