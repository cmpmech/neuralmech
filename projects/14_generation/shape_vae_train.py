from functools import partial
from pathlib import Path

import cmasher as cmr
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchinfo import summary
from tqdm import tqdm

from DL import Standardizer, build_ae_cnn_config, init_weights
from NN import AE, DCN, MLP, VAE

BASE_DIR = Path(__file__).parent
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
# -------------------------- training settings ---------------------------
epochs = 600  # 400  # 1000
lr = 2e-3  # 1e-2  # 4e-3  # 2e-3  # 4e-3  # 1e-3  # 1e-3  # 2e-3
weight_decay = 1e-2  # 5e-2  # 1e-2  # 1e-2  # 1e-10
batch_size = 32  # 128  # 256  # 128
# 64  # 32  # 32  # 64  # 32  # 64  # 32  # 16  # 16  # 64  # 32  # 64  # 16

# define loss
reconstruction_loss = nn.MSELoss(reduction="mean")  # sum for vae?


cost_fun = reconstruction_loss

# ---------------------------- model settings ----------------------------
base, depth, latent_dim = 2, 5, 2  # 2  # 32  # 2
conv_layers = 1  # 1  # 1  # 0
channel_dim = 1
# 4  # 8  # 8  # 8  # 16  # 32  # 16  # 16  # 1  # 16  # "fake" input dim
kernel_size = 3
# act = nn.GELU  # TODO change to PReLU?
# act = nn.PReLU
act = partial(nn.PReLU, init=0.2)

domain_size = 128  # 256  # 256

# ----------------------------- prepare data -----------------------------
data = torch.from_numpy(
    np.concatenate(
        [
            np.load(BASE_DIR / f"../../data/shapes_{'circle'}_{domain_size}.npy"),
            np.load(BASE_DIR / f"../../data/shapes_{'square'}_{domain_size}.npy"),
            np.load(BASE_DIR / f"../../data/shapes_{'triangle'}_{domain_size}.npy"),
            np.load(BASE_DIR / f"../../data/shapes_{'star'}_{domain_size}.npy"),
            np.load(BASE_DIR / f"../../data/shapes_{'ellipse'}_{domain_size}.npy"),
            np.load(BASE_DIR / f"../../data/shapes_{'cross'}_{domain_size}.npy"),
        ],
        axis=0,
    )
)

# data = torch.cat(
#     [
#         torch.from_numpy(
#             np.load(BASE_DIR / f"../../data/shapes_{'circle'}_{domain_size}.npy")
#         ),
#         torch.from_numpy(
#             np.load(BASE_DIR / f"../../data/shapes_{'square'}_{domain_size}.npy")
#         ),
#     ],
#     dim=0,
# )
data = data.to(torch.float32).unsqueeze(1)

# clip = 256  # 180  # 140 is good 90  # 70ish works # 40  # 40  # 71  # 40 good
# # 160  # with 20 good, 40 okayish, 80 not really, 160 seems good and generalizes
# data = data[:clip]  # TODO

dataset = TensorDataset(data)
train_data, val_data = random_split(dataset, [0.9, 0.1])  # TODO 0.9, 0.1
train_loader = DataLoader(
    train_data, batch_size=batch_size, shuffle=True, drop_last=True
)  # TODO check
val_loader = DataLoader(val_data, batch_size=len(val_data), shuffle=True)  # full batch

X_train = train_data.dataset.tensors[0][train_data.indices]
# standardizex = Standardizer(X_train, dim=(0, 3))
standardizex = Standardizer(X_train, dim=(0, 2, 3))
# TRIAL DEACTIVATING STANDARDIZER
# standardizex.x_mean = 0.0
# standardizex.x_std = 1.0

# -------------------------- instantiate model ---------------------------
#################
channels, strides = build_ae_cnn_config(depth, conv_layers, channel_dim, base)
channels[0] = 1  # true input size
red_domain_size = domain_size // 2**depth
layers = [red_domain_size**2 * channels[-1], latent_dim]

Encoder = nn.Sequential()
Encoder.append(
    DCN(
        channels,
        [act() for _ in range(len(channels) - 1)],
        kernel_size,
        stride=strides,
        padding=kernel_size // 2,
        dim=2,
        normalizations=[nn.GroupNorm(1, channel) for channel in channels[1:]],
        # normalizations=[nn.BatchNorm2d(channel) for channel in channels[1:]],
    )
)
Encoder.append(nn.Flatten())
Encoder.append(MLP(layers, [act()]))  # TODO no activation?

# upsamplings = [
#     nn.Upsample(scale_factor=2, mode="nearest") if s == 2 else None
#     for s in strides[::-1]
# ]
upsamplings = [
    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
    if s == 2
    else None
    for s in strides[::-1]
]


Decoder = nn.Sequential()
Decoder.append(MLP([latent_dim, red_domain_size**2 * channels[-1]], [act()]))
Decoder.append(nn.Unflatten(1, (channels[-1], red_domain_size, red_domain_size)))
Decoder.append(
    DCN(
        channels[::-1],
        [act() for _ in range(len(channels) - 2)],
        kernel_size,
        stride=1,
        padding=kernel_size // 2,
        dim=2,
        resamplings=upsamplings,
        normalizations=[nn.GroupNorm(1, channel) for channel in channels[-2:0:-1]],
        # normalizations=[nn.BatchNorm2d(channel) for channel in channels[-2:0:-1]],
    )
)
# TODO sigmoid could be added
# Decoder.append(nn.Sigmoid())
# TODO try batchnorm

model = AE(Encoder, Decoder).to(device)
init_weights(model, act())

# print(channels)
# print(channels[-2:1:-1])

print(summary(model, (1, 1, domain_size, domain_size), depth=4))

# ------------------------ instantiate optimizer -------------------------
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = None
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=epochs, eta_min=lr * 1e-2
)

# ------------------------------- training -------------------------------
print_every = 10
train_cost = [0] * epochs
val_cost = [0] * epochs
pbar = tqdm(range(epochs), desc="Training: ", ncols=90)
for epoch in pbar:
    model.train()
    for x in train_loader:
        x = standardizex(x[0]).to(device)  # unwrap & standardize
        optimizer.zero_grad()
        x_pred = model(x)
        cost = cost_fun(x_pred, x)
        cost.backward()
        optimizer.step()
        train_cost[epoch] += cost.item()
    train_cost[epoch] /= len(train_loader)  # avg per batch
    if scheduler is not None:
        scheduler.step()

    model.eval()
    with torch.no_grad():
        for x in val_loader:
            x = standardizex(x[0]).to(device)  # unwrap & standardize
            x_pred = model(x)
            cost = cost_fun(x_pred, x)
            val_cost[epoch] += cost.item()
        val_cost[epoch] /= len(val_loader)  # avg per batch

    if epoch % print_every == 0:
        pbar.set_postfix(
            {"train": f"{train_cost[epoch]:.2e}", "val": f"{val_cost[epoch]:.2e}"}
        )

# TESTING ##################################################
from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

fig, ax = plt.subplots()
ax.plot(train_cost, "k")
ax.plot(val_cost, "r")
ax.set_yscale("log")
plt.savefig(BASE_DIR / f"../../tmp/history_{timestamp}.png")
plt.show()

# TRAINING DATA
model.eval()

fig, ax = plt.subplots(10, 3, figsize=(6, 20), dpi=domain_size)

# x = next(iter(val_loader))
x = next(iter(train_loader))
x = standardizex(x[0]).to(device)  # unwrap & standardize
x_pred = model(x)
for i in range(10):
    ax[i, 0].imshow(standardizex.inverse(x.cpu())[i, 0], cmap="binary", vmin=0, vmax=1)
    ax[i, 1].imshow(
        standardizex.inverse(x_pred.detach().cpu())[i, 0], cmap="binary", vmin=0, vmax=1
    )
    ax[i, 2].imshow(((x_pred.detach() - x).cpu() ** 2)[i, 0], cmap="hot_r")

    for j in range(3):
        ax[i, j].set_aspect("equal")
        ax[i, j].axis("off")
        ax[i, j].set_rasterized(True)
fig.tight_layout(pad=0)
plt.savefig(BASE_DIR / f"../../tmp/prediction_{timestamp}.png")
plt.show()


# x = next(iter(train_loader))[0]
# x = x.to(device)
# print(x.shape)
# print(model(x).shape)
