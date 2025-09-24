from rembg import remove, new_session
from PIL import Image
from typing import Optional, Tuple

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
          ),
          alpha_matting: bool = Input(
            description="Use alpha matting for smoother edges",
            default=False
          ),
          alpha_matting_foreground_threshold: int = Input(
            description="Foreground threshold for alpha matting",
            default=240,
            ge=0,
            le=255
          ),
          alpha_matting_background_threshold: int = Input(
            description="Background threshold for alpha matting",
            default=10,
            ge=0,
            le=255
          ),
          alpha_matting_erode_size: int = Input(
            description="Erosion size for alpha matting",
            default=10,
            ge=0
          ),
          only_mask: bool = Input(
            description="Return only the binary mask instead of the cutout",
            default=False
          ),
          post_process_mask: bool = Input(
            description="Apply post-processing to smooth mask boundaries",
            default=False
          ),
          bgcolor: Optional[str] = Input(
            description="Background color as hex string or leave empty for transparent",
            default=None
          )
    ) -> Path:
        image = Image.open(str(image))
        session = new_session(model)

        # Convert hex color string to RGBA tuple if provided
        bgcolor_tuple = None
        if bgcolor is not None:
            # Remove '#' if present and convert hex to RGB
            hex_color = bgcolor.lstrip('#')
            if len(hex_color) == 6:  # RGB format
                r = int(hex_color[0:2], 16)
                g = int(hex_color[2:4], 16)
                b = int(hex_color[4:6], 16)
                bgcolor_tuple = (r, g, b, 255)  # Add full alpha
            elif len(hex_color) == 8:  # RGBA format
                r = int(hex_color[0:2], 16)
                g = int(hex_color[2:4], 16)
                b = int(hex_color[4:6], 16)
                a = int(hex_color[6:8], 16)
                bgcolor_tuple = (r, g, b, a)

        output = remove(
            image,
            session=session,
            alpha_matting=alpha_matting,
            alpha_matting_foreground_threshold=alpha_matting_foreground_threshold,
            alpha_matting_background_threshold=alpha_matting_background_threshold,
            alpha_matting_erode_size=alpha_matting_erode_size,
            only_mask=only_mask,
            post_process_mask=post_process_mask,
            bgcolor=bgcolor_tuple
        )

        output_path = f"/tmp/out.png"
        output.save(output_path)

        return Path(output_path)