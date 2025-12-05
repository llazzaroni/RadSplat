# File for making the plots; highlights in red the points that are selected as the peaks

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from density_rays.find_points_gsplat import find_peaks

def euclid_dist(x1, x2, y1, y2, z1, z2):
    return np.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)

def get_positions_rel(positions):
    positions_rel = []
    for i in range(positions.shape[0]):
        positions_rel_ray = []
        for j in range(positions.shape[1]):
            positions_rel_ray.append(euclid_dist(positions[i,j,0], positions[i,0,0], positions[i,j,1], positions[i,0,1], positions[i,j,2], positions[i,0,2]))
        positions_rel.append(np.array(positions_rel_ray))
    return positions_rel


densities = np.load("density_rays/data/densities.npy")
transmittances = np.load("density_rays/data/transmittances.npy")
positions = np.load("density_rays/data/positions.npy")

densities_uniform = np.load("density_rays/data/densities_uniform.npy")
transmittances_uniform = np.load("density_rays/data/transmittances_uniform.npy")
positions_uniform = np.load("density_rays/data/positions_uniform.npy")

positions_rel = get_positions_rel(positions)
positions_rel_uniform = get_positions_rel(positions_uniform)

for i in range(densities.shape[0]):
    density = densities[i,:,:].squeeze()
    transmittance = transmittances[i,:,:].squeeze()
    position = positions_rel[i]

    density_uniform = densities_uniform[i,:,:].squeeze()
    indices_peaks = find_peaks(density_uniform)
    transmittance_uniform = transmittances_uniform[i,:,:].squeeze()
    position_uniform = positions_rel_uniform[i]


    plt.figure(figsize=(8, 4.5))
    plt.plot(position, density, marker="o")
    plt.title("density over records")
    plt.xlabel("Distance from origin")
    plt.ylabel("density")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path(f"/home/llazzaroni/ds-lab/RadSplat/density_rays/pictures/img{i}"), dpi=150)
    plt.close()
    plt.figure(figsize=(8, 4.5))
    plt.plot(position, transmittance, marker="o")
    plt.title("transmittance over distance")
    plt.xlabel("Distance from origin")
    plt.ylabel("transmittance")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path(f"/home/llazzaroni/ds-lab/RadSplat/density_rays/pictures/img{i}_t"), dpi=150)
    plt.close()

    
    plt.figure(figsize=(8, 4.5))
    plt.plot(position_uniform, transmittance_uniform, marker="o")
    plt.title("transmittance over distance")
    plt.xlabel("Distance from origin")
    plt.ylabel("transmittance")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path(f"/home/llazzaroni/ds-lab/RadSplat/density_rays/pictures/img{i}_t_uniform"), dpi=150)
    plt.close()

    plt.figure(figsize=(8, 4.5))
    plt.plot(position_uniform, density_uniform, marker="o", label="density")
    plt.scatter(
        np.array(position_uniform)[indices_peaks],
        density_uniform[indices_peaks],
        c="red",
        s=40,
        zorder=3,
        label="peaks",
    )
    plt.title("density over records")
    plt.xlabel("Distance from origin")
    plt.ylabel("density")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path(f"/home/llazzaroni/ds-lab/RadSplat/density_rays/pictures/img{i}_uniform"), dpi=150)
    plt.close()