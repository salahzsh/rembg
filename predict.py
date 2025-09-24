from rembg import remove, new_session
from PIL import Image

from cog import BasePredictor, Input, Path

class Predictor(BasePredictor):
    def setup(self):
        pass

    # The arguments and types the model takes as input
    def predict(self,
          image: Path = Input(
            description="Input image"),
          model: str = Input(
            description="Model",
            choices=[
                "birefnet-general",
                "birefnet-general-lite",
                "birefnet-portrait",
                "birefnet-dis",
                "birefnet-hrsod",
                "birefnet-cod",
                "birefnet-massive",
                "isnet-anime",
                "dis_custom",
                "isnet-general-use",
                "sam",
                "silueta",
                "u2net_cloth_seg",
                "u2net_custom",
                "u2net_human_seg",
                "u2net",
                "u2netp",
                "bria-rmbg",
                "ben_custom"
            ],
            default="birefnet-general"
          )
    ) -> Path:
        image = Image.open(str(image))
        session = new_session(model)
        output = remove(image, session=session)
        output_path = f"/tmp/out.png"
        output.save(output_path)

        return Path(output_path)