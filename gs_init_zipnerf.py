import numpy as np
import logging
from jax import random
import jax
import jax.numpy as jnp
from flax.training import checkpoints
import json

from submodules.camp_zipnerf.camp_zipnerf.internal import models  
from submodules.camp_zipnerf.camp_zipnerf.internal import datasets
from submodules.camp_zipnerf.camp_zipnerf.internal import configs
from submodules.camp_zipnerf.camp_zipnerf.internal import utils 


def sample_rays(config, n : int = 1_000_000) -> utils.Rays:

    logging.info("Loading data from: " + config.data_dir)

    dataset = datasets.load_dataset('test', config.data_dir, config)

    logging.info("Dataset loaded")

    sampling_distribution = distribute_values(n, dataset.size)

    rays =  dataset.generate_ray_batch(1).rays

    # for i in range(sampling_distribution.shape[0]):
    #     ray_batch = dataset.generate_ray_batch(i).rays

    return rays

def distribute_values(total : int, n_groups : int) -> np.ndarray:
    """
    This function is used to distribut an 'amount' total among n different groups

    :Param total -> (int) the amount to distribute
    :Param n_groups -> (int) the number of groups to distribute the amount among

    :Return (np.ndarray) the array of the values distributed
    """

    random_values = np.random.rand(n_groups)
    random_values /= random_values.sum()

    return random_values * total

class Model:

    _rng = random.PRNGKey(0)

    def __init__(self, rays, config) -> None:

        self.model , self.init_param = models.construct_model(self._rng, rays, config)
        if config.checkpoint_dir:
            self.state = checkpoints.restore_checkpoint(config.checkpoint_dir, target=None)

    def render_ray(self, rays):

        renderings, ray_history = self.model.apply(
            {'params': self.state},
            rng=None,
            rays=rays,
            train_frac=1.0,
            compute_extras=False,
            zero_glo=True,
            train=False
        )

        return renderings


if __name__ == "__main__":
    

    logging.info("Start...")
    config = configs.Config()

    config.data_dir = '/home/rbollati/ds-lab/data/nyc'
    config.checkpoint_dir = '/home/rbollati/ds-lab/checkpoints/camp_official/nyc'


    logging.info("Sampling Rays")
    rays = sample_rays(config, 10)
    single_ray = jax.tree_util.tree_map(lambda x: x[:1], rays)

    model = Model(rays, config)

    renderings = model.render_ray(single_ray)

    rgb = renderings[-1]['rgb']  # shape (1, 3)
    depth = renderings[-1]['distance']  # if available
    acc = renderings[-1]['acc']


    result = {
            'rgb' : rgb,
            'depth' : depth
            'acc' : acc
         } 


    with open('/home/rbollati/ds-lab/result.json' , 'w') as f:
        json.dump(result, f, indent =2)


    # pred_color = jnp.squeeze(rgb)  # shape (3,)



    # TODO initialize model

    # TODO run nerf for each ray

    # get z median

    # frmat resuls
