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
from NN import AE, DCN, MLP

BASE_DIR = Path(__file__).parent
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
# -------------------------- training settings ---------------------------
epochs = 300 #600
lr = 4e-3 #2e-3 #2e-3
weight_decay = 1e-2
batch_size = 32

# define loss
cost_fun = nn.MSELoss(reduction="mean")

# ---------------------------- model settings ----------------------------
base, depth = 2, 5
conv_layers = 1
channel_dim = 1
kernel_size = 3
act = partial(nn.PReLU, init=0.2)

bottleneck_layers = 2 # run with 0, 1, 2, 3, (4)?
# act = nn.GELU
compression = 2**(-depth - bottleneck_layers)
print(f'compression ratio {compression * 100:.2f} %')
# ----------------------------- prepare data -----------------------------
domain_size = 256

data = torch.from_numpy(np.load(BASE_DIR / f"../../data/fibers_{domain_size}.npy"))
data = data.to(torch.float32).unsqueeze(1)

# clip = 380 #160 #80 # 40 worse than 20?     #160 #80 #80  # 40 # TODO
# data = data[:clip]

dataset = TensorDataset(data)
train_data, val_data = random_split(dataset, [0.9, 0.1])
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(val_data, batch_size=len(val_data), shuffle=True)  # full batch

X_train = train_data.dataset.tensors[0][train_data.indices]
standardizex = Standardizer(X_train, dim=(0, 2, 3))

# -------------------------- instantiate model ---------------------------
channels, strides = build_ae_cnn_config(depth, conv_layers, channel_dim, base)
channels[0] = 1  # true input size (in case channel_dim != 1)

bottleneck_channels = [int(channels[-1] / 2**(i + 1)) for i in range(bottleneck_layers)]
bottleneck_strides = [1] * bottleneck_layers

channels += bottleneck_channels
strides += bottleneck_strides


# red_domain_size = domain_size // 2**depth
# layers = [red_domain_size**2 * channels[-1], latent_dim]

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
    )
)

# Encoder.append(nn.Flatten())
# Encoder.append(MLP(layers, [act()]))

upsamplings = [
    # nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
    nn.Upsample(scale_factor=2, mode="nearest")
    if s == 2
    else None
    for s in strides[::-1]
]


Decoder = nn.Sequential()
# Decoder.append(MLP([latent_dim, red_domain_size**2 * channels[-1]], [act()]))
# Decoder.append(nn.Unflatten(1, (channels[-1], red_domain_size, red_domain_size)))
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
    )
)

model = AE(Encoder, Decoder).to(device)
init_weights(model, act())
summary(model, (1, 1, domain_size, domain_size), depth=4)

# ------------------------ instantiate optimizer -------------------------
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
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

# ----------------------------- export model -----------------------------
model.standardizer = standardizex # just for saving
torch.save(model, BASE_DIR / f'../../models/fiber_ae_{bottleneck_layers}_{domain_size}.pt2')

# ---------------------------- postprocessing ----------------------------
fig, ax = plt.subplots()
ax.plot(train_cost, "k")
ax.plot(val_cost, "r")
ax.set_yscale("log")
# plt.savefig(BASE_DIR / '../../tmp/history.png')
plt.show()




###### TESTING
model.eval()

# x = next(iter(train_loader))
x = next(iter(val_loader))
x = standardizex(x[0]).to(device)  # unwrap & standardize
x_pred = model(x)

fig, ax = plt.subplots(1, 2, figsize=(4, 2), dpi=domain_size)
ax[0].imshow(standardizex.inverse(x.cpu())[0, 0], cmap="binary", vmin=0, vmax=1)
ax[1].imshow(
    standardizex.inverse(x_pred.detach().cpu())[0, 0], cmap="binary", vmin=0, vmax=1
)
for i in range(2):
    ax[i].set_aspect("equal")
    ax[i].axis("off")
    ax[i].set_rasterized(True)
fig.tight_layout(pad=0)
# plt.savefig(BASE_DIR / '../../tmp/prediction.png')
plt.show()
