from pathlib import Path
import torch
import math
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

def rel_to_images_root(p: str) -> str:
    # find 'images' in the path and take from there (COLMAP parser returns names relative to images/)
    parts = Path(p).parts
    if "images" in parts:
        i = parts.index("images")
        return str(Path(*parts[i:]))  # e.g. 'images/seq/frame0001.png'
    return Path(p).name

offset = 5856
'''
def frame_to_dscf(name: str) -> str:
    import re
    m = re.search(r"frame_(\d+)", name)
    if not m:
        return name
    n = int(m.group(1))
    return f"images/DSCF{offset + n:04d}.JPG"

'''



def main() -> None:

    print("Started job")

    # Folder with the checkpoints
    #folder = "/work/courses/dslab/team20/rbollati/running_env/outputs/poster/nerfacto/2025-10-18_013814/config.yml"
    #folder = "/work/courses/dslab/team20/rbollati/running_env/outputs/instant-ngp-3k-steps/instant-ngp/2025-11-08_113850/config.yml"
    folder = "/work/courses/dslab/team20/rbollati/running_env/outputs/poster/nerfacto/2025-11-08_124615/config.yml"
    #folder = "/work/courses/dslab/team20/rbollati/running_env/outputs/counter_ns/nerfacto/2025-11-03_173612/config.yml"
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
    num_rays = 1_000_000
    batch_size = 8_000
    n_batches = math.ceil(num_rays / batch_size)

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

    xyzrgb_chunks = []

    with torch.no_grad():
        for b in range(n_batches):

            s = b * batch_size
            e = min((b + 1) * batch_size, num_rays)
            if s >= e:
                break
            ray_indices_batch = ray_indices[s:e, :]
            B = ray_indices_batch.shape[0]

            ray_bundle = ray_generator(ray_indices_batch)

            #print("Ray bundle produced")
            #print(ray_bundle)

            # Chato fix
            collider = getattr(model, "collider", None)
            if collider is not None:
                ray_bundle = collider(ray_bundle)

            # Pass the ray bundle through the model
            ray_samples, weights_list, ray_samples_list = model.proposal_sampler(ray_bundle, density_fns=model.density_fns)
            field_outputs = model.field.forward(ray_samples, compute_normals=model.config.predict_normals)
            weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])

            #print("Passed the ray bundle through the model")

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

            radsplat_positions = positions[torch.arange(B, device=positions.device), last_idx, :]

            #print("Computed the positions")
            #print(radsplat_positions)

            # Compute the rgb values
            rgb_samples = field_outputs[FieldHeadNames.RGB]
            rgb = model.renderer_rgb(rgb=rgb_samples, weights=weights)

            # Concatenate rgb to the positions
            xyzrgb_batch = torch.cat([radsplat_positions, rgb], dim=-1)

            xyzrgb_chunks.append(xyzrgb_batch.detach().cpu())

            print(f"Processed batch {b+1}/{n_batches}")

    # Positions and colors have been saved
    
    xyzrgb = torch.cat(xyzrgb_chunks, dim=0)

    # Build a mapping between cameras positions and cameras ids

    dpo = dm.train_dataparser_outputs
    image_filenames_abs = [str(p) for p in dpo.image_filenames]
    image_filenames_rel = [rel_to_images_root(p) for p in image_filenames_abs]
    #image_filenames_rel = [frame_to_dscf(p) for p in image_filenames_rel]
    print(image_filenames_rel)


    # Put everything in the payload
    payload = {
        "xyzrgb": xyzrgb.cpu(),
        "camera_to_worlds": cams.camera_to_worlds.cpu(),
        "K": cams.get_intrinsics_matrices().cpu(),
        "image_filenames_abs": image_filenames_abs,
        "image_filenames_rel": image_filenames_rel,
    }

    out_dir = Path("/work/courses/dslab/team20/positions_nerf")
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_dir / "radsplat_poster_less_iter.pt")

    print(xyzrgb.shape)
    print("Saved the positions")

    

if __name__ == "__main__":
    main()
