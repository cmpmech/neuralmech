import torch
from torch import nn
import time
import matplotlib.pyplot as plt
from NN import MLP
from postprocessing import show_image
from PIL import Image
import numpy as np
from tqdm import tqdm

torch.manual_seed(1)
torch.backends.cudnn.deterministic = True
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
activation_id = None

initialization = 0 #0 # 1 # 2

# -------------------------------- helper --------------------------------
def init_weights(m):
    if type(m) == torch.nn.Linear:
        if initialization == 0:
            torch.nn.init.uniform_(m.weight, a=-10, b=10)
            torch.nn.init.uniform_(m.bias, a=-10, b=10)
        elif initialization == 1:
            torch.nn.init.uniform_(m.weight, a=-1, b=1)
            torch.nn.init.uniform_(m.bias, a=-1, b=1)
        elif initialization == 2:
            gain = nn.init.calculate_gain('tanh')
            nn.init.xavier_uniform_(m.weight, gain=gain)
            nn.init.zeros_(m.bias)

# --------------------------- hyperparameters ----------------------------
# depth study
hidden_layers = 4
neurons = 128

parameters = neurons * 8 + neurons * neurons * (hidden_layers - 1) + neurons * hidden_layers
print(f'parameters: {parameters}')

samples = 800
activation = torch.nn.Tanh()

# ---------------------------- preprocessing -----------------------------
layers = [2] + [neurons] * hidden_layers + [3]
activations = [activation] * hidden_layers

x = torch.linspace(-1,1, samples)
y = torch.linspace(-1,1, samples)
x, y = torch.meshgrid(x, y, indexing='ij')
mlp_input = torch.cat((x.flatten().unsqueeze(1), y.flatten().unsqueeze(1)), 1).to(device)

# --------------------------- model prediction ---------------------------
model = MLP(layers, activations)
model.apply(init_weights)
model.to(device)

def normalize_output(y, eps=1e-8):
    y_min = y.amin(dim=(0, 1), keepdim=True)
    y_max = y.amax(dim=(0, 1), keepdim=True)
    return (y - y_min) / (y_max - y_min + eps)

if initialization == 1:
    lr = 1e-2
    epochs = 10001
elif initialization == 0:
    lr = 2e-2
    epochs = 10001
elif initialization == 2:
    lr = 2e-3
    epochs = 10001
cost_fun = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr)

# -------------------------------- target --------------------------------
img = Image.open('../../data/images/memphis.jpg').convert('RGB')
target = torch.from_numpy(np.asarray(img)).to(torch.float32).to(device)

# ---------------------------- training loop -----------------------------
train_cost = [0] * epochs
tic = time.time()
print_every = 1
pbar = tqdm(range(epochs))
model.train()
for epoch in pbar:
    optimizer.zero_grad()
    y_pred = model(mlp_input).reshape(samples, samples, 3)
    cost = cost_fun(y_pred, target)
    cost.backward()
    optimizer.step()
    train_cost[epoch] = cost.item()

    if epoch % print_every == 0:
        pbar.set_postfix({'train': f'{train_cost[epoch]:.2e}'})

    # postprocessing
    if epoch == 0 or epoch == 10 or epoch == 100 or epoch == 200 or epoch == 400 or epoch == 1000 or epoch == 2000 or epoch == 10000:
        show_image(normalize_output(y_pred.detach().cpu()).numpy(), path=f'../../results/learning_image_{initialization}_{epoch}.pdf')

toc = time.time()
print(f'elapsed time {toc - tic:.2f} s')

with torch.no_grad():
    z_pred = model(mlp_input).reshape(samples, samples, 3).cpu()
    z_pred = normalize_output(z_pred)

# ---------------------------- postprocessing ----------------------------

fig, ax = plt.subplots()
ax.set_yscale('log')
ax.plot(train_cost, 'k')
plt.show()

show_image(normalize_output(target.detach().cpu()).numpy(), path=f'../../results/memphis_target.pdf')