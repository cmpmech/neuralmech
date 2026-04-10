from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from postprocessing import show_image

BASE_DIR = Path(__file__).parent
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

img = np.asarray(Image.open(BASE_DIR / '../../data/images/onions.jpg').convert('L')).copy()
x = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(dtype=torch.float32)
# x = torch.randn((1, 1, 5, 5))

model = nn.ConvTranspose2d(1, 1, 3, stride=2, padding=0, bias=False)
with torch.no_grad():
    model.weight.fill_(1.0)
y = model(x)

# ---------------------------- postprocessing ----------------------------
show_image(x[0, 0].detach().cpu().numpy(), grayscale=True, path=BASE_DIR / '../../results/checkerboarding1.jpg')
show_image(y[0, 0].detach().cpu().numpy(), grayscale=True, path=BASE_DIR / '../../results/checkerboarding2.jpg')