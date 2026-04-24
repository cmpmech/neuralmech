from pathlib import Path

import numpy as np
from PIL import Image

from postprocessing import show_image

BASE_DIR = Path(__file__).parent

img = Image.open(BASE_DIR / "duckling.jpg")
img = img.resize((800, 800), resample=Image.Resampling.LANCZOS)

show_image(np.asarray(img))

img.save(BASE_DIR / "output.jpg", quality=100)
