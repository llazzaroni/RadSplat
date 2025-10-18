from pathlib import Path

from submodules.nerfstudio.nerfstudio.utils.eval_utils import eval_setup

class InitializationPipeline:

    def __init__(self, checkpoint_folder: str):

        self.checkpoint_folder = Path(checkpoint_folder)
        self.pipeline, self.model = self._load_model() 
        self._load_model()

        pass

    def _load_model(self):
        _, pipeline, _, _ = eval_setup(self.checkpoint_folder, test_mode="test")

        return pipeline, pipeline.model

    def render_camera(self):
        if self.pipeline.datamanager.train_dataset:
            camera = self.pipeline.datamanager.train_dataset[0]["image_idx"]  # or a custom camera
            outputs = self.model.get_outputs_for_camera(camera)
            image = outputs["rgb"]

            return image
        
        return None


