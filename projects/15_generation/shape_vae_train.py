from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchinfo import summary
from tqdm import tqdm

from DL import Standardizer, build_ae_cnn_config, init_weights
from NN import DCN, MLP, VAE

BASE_DIR = Path(__file__).parent
DATA_DIR = (BASE_DIR / "../../data").resolve()
MODEL_DIR = (BASE_DIR / "../../models").resolve()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True

# -------------------------------------- settings -------------------------------------
# hyperparameters
epochs = 600
lr = 1e-2  # change over ae
weight_decay = 1e-2
batch_size = 64

# beta = 0. # good reconstruction
beta = 0.05  # goodish reconstruction & latent
# beta = 0.2 # good latent

# define loss
recon_loss = nn.MSELoss(reduction="mean")


def kl_div(mean_pred, logvar_pred):
    var_pred = torch.exp(logvar_pred)
    kl = 0.5 * torch.mean(var_pred + mean_pred**2 - logvar_pred - 1)
    return kl


def cost_fun(x_pred, mean_pred, logvar_pred, x, beta=1.0):
    mse = recon_loss(x_pred, x)
    kl = kl_div(mean_pred, logvar_pred)
    return mse + beta * kl


# model settings
base, depth = 2, 5
latent_dim = 512  # 2 8 32 128 512 # 2 as latent_dim for visualization
conv_layers = 1
channel_dim = 1
kernel_size = 3
act = partial(nn.PReLU, init=0.2)

# ------------------------------------ prepare data -----------------------------------
domain_size = 128

labels = ["circle", "ellipse", "square", "triangle", "cross", "star"]
data = []
for label in labels:
    data.append(
        torch.from_numpy(
            np.load(DATA_DIR / f"shapes_{label}_{domain_size}.npy")
        )
    )
data = torch.from_numpy(np.concatenate(data, axis=0))
data = data.to(torch.float32).unsqueeze(1)

dataset = TensorDataset(data)
train_data, val_data = random_split(dataset, [0.9, 0.1])
train_loader = DataLoader(
    train_data, batch_size=batch_size, shuffle=True, drop_last=True
)
val_loader = DataLoader(val_data, batch_size=len(val_data), shuffle=True)

X_train = train_data.dataset.tensors[0][train_data.indices]
standardizex = Standardizer(X_train, dim=(0, 2, 3))

# --------------------------- instantiate model & optimizer ---------------------------
channels, strides = build_ae_cnn_config(depth, conv_layers, channel_dim, base)
channels[0] = 1  # true input size (in case channel_dim != 1)
red_domain_size = domain_size // 2**depth
layers = [red_domain_size**2 * channels[-1], latent_dim]

Encoder = nn.Sequential()
Encoder.append(
    DCN(
        channels,
        [[nn.GroupNorm(1, channel), act()] for channel in channels[1:]],
        kernel_size,
        stride=strides,
        padding=kernel_size // 2,
        dim=2,
    )
)
Encoder.append(nn.Flatten())
Encoder.append(MLP(layers[:-1] + [layers[-1] * 2], post_modules=[act()]))

upsamplings = [
    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
    if s == 2
    else None
    for s in strides[::-1]
]


Decoder = nn.Sequential()
Decoder.append(MLP(layers[::-1], post_modules=[act() for _ in range(len(layers) - 1)]))
Decoder.append(nn.Unflatten(1, (channels[-1], red_domain_size, red_domain_size)))
Decoder.append(
    DCN(
        channels[::-1],
        [[nn.GroupNorm(1, channel), act()] for channel in channels[-2:0:-1]],
        kernel_size,
        stride=1,
        padding=kernel_size // 2,
        dim=2,
        pre_modules=upsamplings,
    )
)

model = VAE(Encoder, Decoder).to(device)
init_weights(model, act())
summary(model, (1, 1, domain_size, domain_size), depth=4)

optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

# -------------------------------------- training -------------------------------------
print_every = 10
train_cost = [0] * epochs
val_cost = [0] * epochs
pbar = tqdm(range(epochs), desc="Training: ", ncols=90)
for epoch in pbar:
    model.train()
    for x in train_loader:
        x = standardizex(x[0]).to(device)
        optimizer.zero_grad()
        x_pred, mean_pred, logvar_pred = model(x)
        cost = cost_fun(x_pred, mean_pred, logvar_pred, x, beta)
        cost.backward()
        optimizer.step()
        train_cost[epoch] += cost.item()
    train_cost[epoch] /= len(train_loader)

    model.eval()
    with torch.no_grad():
        for x in val_loader:
            x = standardizex(x[0]).to(device)
            x_pred, mean_pred, logvar_pred = model(x)
            cost = cost_fun(x_pred, mean_pred, logvar_pred, x, beta)
            val_cost[epoch] += cost.item()
        val_cost[epoch] /= len(val_loader)

    if epoch % print_every == 0:
        pbar.set_postfix(
            {"train": f"{train_cost[epoch]:.2e}", "val": f"{val_cost[epoch]:.2e}"}
        )

# --------------------------------------- export --------------------------------------
model.standardizer = standardizex
torch.save(
    model, MODEL_DIR / f"shape_vae_{latent_dim}_{beta}_{domain_size}.pt2"
)

# ----------------------------------- postprocessing ----------------------------------
fig, ax = plt.subplots()
ax.plot(train_cost, "k")
ax.plot(val_cost, "r")
ax.set_yscale("log")
plt.show()
