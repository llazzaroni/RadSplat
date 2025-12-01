import warnings
import math
import torch
import logging

from nerf_models import Nerfacto
from nerfstudio.field_components.field_heads import FieldHeadNames
from point_samplers import sobel_edge_detector_sampler, canny_edge_detector_sampler, random_sampler, mixed_sampler
from gs_initializer import Initializer
from pathlib import Path
import argparse
import numpy as np

folder = "/work/courses/dslab/team20/rbollati/running_env/experiments/20251121_183303_counter/outputs/20251121_183303_counter/nerfacto/20251121_183303_counter/config.yml"
N_RAYS = 200
model = Nerfacto(folder)
cams = model.pipeline.datamanager.train_dataset.cameras.to('cpu')
dpo = model.pipeline.datamanager.train_dataparser_outputs
print("initialized the model")
coords = random_sampler(model.pipeline.datamanager, N_RAYS, model.device)
rays = model.create_rays(coords)
print("sampled the rays")
sampled = model.sample_points(rays)
outputs, field_outputs, weights  = model.evaluate_points(sampled)
print("evaluated points")
densities = field_outputs[FieldHeadNames.DENSITY]
gs_initializer = Initializer(weights, sampled)
transmittance = gs_initializer.compute_transmittance()
positions = sampled.frustums.get_positions()
densities_np = densities.detach().cpu().numpy()
transmittance_np = transmittance.detach().cpu().numpy()
positions_np = positions.detach().cpu().numpy()

sampled_uniform = model.sample_points_uniform(rays)
outputs_uniform, field_outputs_uniform, weights_uniform  = model.evaluate_points(sampled_uniform)
densities_uniform = field_outputs_uniform[FieldHeadNames.DENSITY]
gs_initializer_uniform = Initializer(weights_uniform, sampled_uniform)
transmittance_uniform = gs_initializer_uniform.compute_transmittance()
positions_uniform = sampled_uniform.frustums.get_positions()
densities_np_uniform = densities_uniform.detach().cpu().numpy()
transmittance_np_uniform = transmittance_uniform.detach().cpu().numpy()
positions_np_uniform = positions_uniform.detach().cpu().numpy()


np.save("/home/llazzaroni/ds-lab/RadSplat/density_rays/data/densities.npy", densities_np)
np.save("/home/llazzaroni/ds-lab/RadSplat/density_rays/data/transmittances.npy", transmittance_np)
np.save("/home/llazzaroni/ds-lab/RadSplat/density_rays/data/positions.npy", positions_np)

np.save("/home/llazzaroni/ds-lab/RadSplat/density_rays/data/densities_uniform.npy", densities_np_uniform)
np.save("/home/llazzaroni/ds-lab/RadSplat/density_rays/data/transmittances_uniform.npy", transmittance_np_uniform)
np.save("/home/llazzaroni/ds-lab/RadSplat/density_rays/data/positions_uniform.npy", positions_np_uniform)
print("saved to npy")