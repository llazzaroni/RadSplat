import torch

class Initializer:

    def __init__(self, weights, ray_samples) -> None:
        
        self.weights = weights
        self.ray_samples = ray_samples

    def compute_transmittance(self):

        cum_inc = torch.cumsum(self.weights, dim=1)
        cum_exc = cum_inc - self.weights
        T_per_sample = 1.0 - cum_exc

        return T_per_sample

    def compute_inital_positions(self, T_per_sample, treshold):

        positions = self.ray_samples.frustums.get_positions()
        T = T_per_sample.squeeze(-1)
        mask = (T > treshold)
        count = mask.sum(dim=1)
        last_idx = (count - 1).clamp(min=0)

        initial_gaussians_position = positions[torch.arange(len(positions), device=positions.device), last_idx, :]

        return initial_gaussians_position


