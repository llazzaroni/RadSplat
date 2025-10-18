from nerfstudio.utils.eval_utils import eval_setup

class InitializationPipeline:

    def __init__(self, checkpoint_folder: str):

        self.model_folder = checkpoint_folder


        pass

    def _load_model(self):
        pipeline, _, _ = eval_setup(self.model_folder, test_mode="test")


