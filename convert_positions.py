import torch
from submodules.gsplat.examples.utils import knn

data  = torch.load('example_data/ckpt_6999_rank0_.pt', map_location=torch.device('cpu'))
data2  = torch.load('example_data/positions.pt', map_location=torch.device('cpu'))
data3 = torch.load('example_data/complete_output.pt', map_location=torch.device('cpu'))

rgb = data3['rgb'].unsqueeze(1)

num_gs = data3['position'].shape[0]

dist2_avg = (knn(data3['position'], 4)[:, 1:] ** 2).mean(dim=-1)
dist_avg = torch.sqrt(dist2_avg)
scales = torch.log(dist_avg).unsqueeze(-1).repeat(1, 3)

sintetic_chkp = {
        'step' : 0,
        'splats': {
            'means' : data3['position'],
            'scales' : scales,
            'opacities' : torch.full((num_gs,), 0.1),
            'quats' : torch.ones(num_gs, 4),
            'sh0' : rgb,
            'shN' : torch.zeros(num_gs, 1, 3)
            }
            
        }

print(sintetic_chkp)

