import torchvision
import warnings
warnings.filterwarnings(
    "ignore",
    message="Using a non-tuple sequence for multidimensional indexing is deprecated",
)
from pathlib import Path
import numpy as np
import torch
_orig_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_load(*args, **kwargs)
torch.load = _patched_load
from submodules.nerfstudio.nerfstudio.utils.eval_utils import eval_setup

class InitializationPipeline:

    def __init__(self, checkpoint_folder: str):

        print('###################################### initi model')

        self.checkpoint_folder = Path(checkpoint_folder)
        self.pipeline, self.model = self._load_model() 
        self._load_model()

        print('###################################### model loaded')


        pass

    def _load_model(self):
        _, pipeline, _, _ = eval_setup(self.checkpoint_folder, test_mode="test")
        return pipeline, pipeline.model

    def render_camera(self):

        print('###################################### rendering image')
        if self.pipeline.datamanager.train_dataset:
            camera = self.pipeline.datamanager.train_dataset.cameras[0]

            print('###################################### camera')
            print(camera)
            print('###################################### run model')

            outputs = self.model.get_outputs_for_camera(camera)
            image = outputs["rgb"]

            return image
        
        return None

if __name__ == "__main__": 
    folder = "/work/courses/dslab/team20/rbollati/running_env/outputs/poster/nerfacto/2025-10-18_013814/config.yml"

    pipeline = InitializationPipeline(folder)  
    image = pipeline.render_camera()
    out_path = "render.png"
    torchvision.utils.save_image(image.permute(2, 0, 1), out_path)
    print(image)

