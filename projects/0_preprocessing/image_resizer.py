import numpy as np
from PIL import Image
from postprocessing import show_image

img = Image.open("duckling.jpg")
img = img.resize((800, 800), resample=Image.Resampling.LANCZOS)

show_image(np.asarray(img))

img.save("output.jpg", quality=100)