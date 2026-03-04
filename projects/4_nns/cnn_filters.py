import numpy as np
import torch
from torch import nn
from PIL import Image
from postprocessing import show_image

torch.manual_seed(0)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# -------------------------- training settings ---------------------------
resolution = 256

epochs = 2000
lr = 2e-2

# define loss
cost_fun = nn.MSELoss()

# ---------------------------- model settings ----------------------------
channels = [1, 1]
activations = []
kernel_size = 3
stride = 1
padding = 0

model = nn.Conv2d(channels[0], channels[1], kernel_size, stride, padding, bias=False)

for filter in range(7):
    # ----------------------------- set filters ------------------------------
    with torch.no_grad():
        if filter == 0: # identity
            model.weight[:] = torch.tensor([[0,0,0],
                                            [0,1,0],
                                            [0,0,0]])
        elif filter == 1: # shift and subtract
            model.weight[:] = torch.tensor([[0,0,0],
                                            [0,1,0],
                                            [0,0,-1]])
        elif filter == 2: # edge detection
            model.weight[:] = torch.tensor([[-1/8,-1/8,-1/8],
                                            [-1/8,1,-1/8],
                                            [-1/8,-1/8,-1/8]])
        elif filter == 3: # embossing
            model.weight[:] = torch.tensor([[-2,-1,0],
                                            [-1,1,1],
                                            [0,1,2]])
        elif filter == 4: # gaussian
            model.weight[:] = torch.tensor([[1,2,1],
                                            [2,4,2],
                                            [1,2,1]]) / 16
        elif filter == 5: # averaging
            model.weight[:] = torch.tensor([[1,1,1],
                                            [1,1,1],
                                            [1,1,1]]) / 9
        elif filter == 6: # random
            torch.manual_seed(1) # 1
            model.weight[:] = torch.randn(model.weight.size())
            print(model.weight)

    # ------------------------------ input data ------------------------------
    img = Image.open('../../data/images/water.jpg').convert('L')

    x = torch.from_numpy(np.asarray(img)).unsqueeze(0).unsqueeze(0).to(dtype=torch.float32)

    # ------------------------------ prediction ------------------------------
    with torch.no_grad():
        y = model(x)

# ---------------------------- postprocessing ----------------------------
    show_image(y[0, 0].numpy(), grayscale=True, path=f'../../results/filter_example_{filter}.jpg')