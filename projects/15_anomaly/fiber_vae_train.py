# based off fiber_ae_train
# 
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchinfo import summary
from tqdm import tqdm

from DL import Standardizer, build_ae_cnn_config, init_weights
from NN import DCN, MLP, VAE

BASE_DIR = Path(__file__).parent
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True

# -------------------------- training settings ---------------------------
epochs = 66
lr = 2e-2  # change over ae
weight_decay = 1e-4    ### ??
batch_size = 64

# beta = 0. # good reconstruction
beta = 0.2
# beta = 0.13
# beta = 0.2 # good latent

# define loss
recon_loss = nn.MSELoss(reduction="mean")


# prevent posterior collapsing for any latent dimension by enforcing a minimum KL divergence
free_bits = 5.0  # nats per latent dimension
def kl_div(mean_pred, logvar_pred):
    var_pred = torch.exp(logvar_pred)
    kl_per_dim = 0.5 * (var_pred + mean_pred**2 - logvar_pred - 1)  # shape: (B, latent_dim)
    kl_per_dim = torch.clamp(kl_per_dim, min=free_bits / latent_dim)
    return kl_per_dim.mean()
# def kl_div(mean_pred, logvar_pred):
#     var_pred = torch.exp(logvar_pred)
#     kl = 0.5 * torch.mean(var_pred + mean_pred**2 - logvar_pred - 1)
#     return kl


def cost_fun(x_pred, mean_pred, logvar_pred, x, beta=1.0):
    mse = recon_loss(x_pred, x)
    kl = kl_div(mean_pred, logvar_pred)
    return mse + beta * kl


# ---------------------------- model settings ----------------------------
base, depth = 2, 5
latent_dim = 64  # 2 8 32 128 512 # 2 as latent_dim for visualization
conv_layers = 1
channel_dim = 1
kernel_size = 3
act = partial(nn.PReLU, init=0.2)

encoder_lr_factor = 0.1  # smaller lr for encoder to prevent posterior collapse

# bottleneck_layers = 1  # controls compression ratio
# compression = 2 ** (-depth - bottleneck_layers)
# print(f"compression ratio {compression * 100:.2f} %")

# ----------------------------- prepare data -----------------------------
domain_size = 256
data = []
data.append(torch.from_numpy(np.load(BASE_DIR / f"../../data/fibers_{domain_size}.npy") ) )

## Add squares to dataset
num_circles = 10
# for num_squares in range(num_circles + 1):
#         data.append(
#         torch.from_numpy(
#             np.load(BASE_DIR / f"../../data/fibers_anomaly_{num_circles}_{domain_size}.npy")
#         )
#     )
data = torch.from_numpy(np.concatenate(data, axis=0))
data = data.to(torch.float32).unsqueeze(1)

dataset = TensorDataset(data)
train_data, val_data = random_split(dataset, [0.9, 0.1])
train_loader = DataLoader(
    train_data, batch_size=batch_size, shuffle=True, drop_last=True
)
val_loader = DataLoader(val_data, batch_size=len(val_data), shuffle=True)  # full batch

X_train = train_data.dataset.tensors[0][train_data.indices]
standardizex = Standardizer(X_train, dim=(0, 2, 3))

# -------------------------- instantiate model ---------------------------
## details on this??
channels, strides = build_ae_cnn_config(depth, conv_layers, channel_dim, base)
channels[0] = 1  # true input size (in case channel_dim != 1)

## vae
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
    )
)
Encoder.append(nn.Flatten())             # flatten output of CNN
Encoder.append(MLP(layers[:-1] + [layers[-1] * 2], [act()]))     # *2 for mean and SD in AE

upsamplings = [
    # nn.Upsample(scale_factor=2, mode="nearest") if s == 2 else None
    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
    if s == 2
    else None
    for s in strides[::-1]
]


Decoder = nn.Sequential()
Decoder.append(MLP(layers[::-1], [act()] * (len(layers) - 1)))      # not times 2
Decoder.append(nn.Unflatten(1, (channels[-1], red_domain_size, red_domain_size)))  

# what is this??
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

model = VAE(Encoder, Decoder).to(device)
init_weights(model, act())
summary(model, (1, 1, domain_size, domain_size), depth=4)


# ------------------------ instantiate optimizer -------------------------

optimizer = torch.optim.AdamW([
    {"params": model.encode.parameters(), "lr": lr * encoder_lr_factor},
    {"params": model.decode.parameters(), "lr": lr},
], weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=epochs, eta_min=lr * 1e-2
)
# optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
# scheduler = None

# ------------------------------- training -------------------------------
print_every = 5
train_cost = [0] * epochs
val_cost = [0] * epochs
pbar = tqdm(range(epochs), desc="Training: ", ncols=90)
for epoch in pbar:
    model.train()
    for x in train_loader:
        x = standardizex(x[0]).to(device)  # unwrap & standardize
        optimizer.zero_grad()
        x_pred, mean_pred, logvar_pred = model(x)
        cost = cost_fun(x_pred, mean_pred, logvar_pred, x, beta)
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
            x_pred, mean_pred, logvar_pred = model(x)
            cost = cost_fun(x_pred, mean_pred, logvar_pred, x, beta)
            val_cost[epoch] += cost.item()
        val_cost[epoch] /= len(val_loader)  # avg per batch

    if epoch % print_every == 0:
        pbar.set_postfix(
            {"train": f"{train_cost[epoch]:.2e}", "val": f"{val_cost[epoch]:.2e}"}
        )

# ----------------------------- export model -----------------------------
model.standardizer = standardizex  # just for saving
torch.save(
    model, BASE_DIR / f"../../models/fiber_vae_depth{depth}_latent{latent_dim}_beta{beta}_{domain_size}.pt2"
)

# ---------------------------- postprocessing ----------------------------
fig, ax = plt.subplots()
ax.plot(train_cost, "k")
ax.plot(val_cost, "r")
ax.set_yscale("log")
ax.set_title(f"training history beta={beta}, latent_dim={latent_dim}")
plt.savefig(BASE_DIR / f"../../tmp/b{beta}_l{latent_dim}_history.png")
plt.show()


## compare prediction and input
sample_ind = 0
model.eval()
with torch.no_grad():
    x = next(iter(val_loader))
    x = standardizex(x[0]).to(device)  # unwrap & standardize
    x_pred, mean_pred, logvar_pred = model(x)
    x_orig = standardizex.inverse(x.cpu())[sample_ind, 0]
    x_recon = standardizex.inverse(x_pred.detach().cpu())[sample_ind, 0]

fig2, ax2 = plt.subplots(2, 2, figsize=(6, 6), dpi=domain_size)
ax2[0, 0].imshow(x_orig, cmap="binary", vmin=0, vmax=1)
ax2[0, 0].set_title("original")
ax2[0, 1].imshow(x_recon, cmap="binary", vmin=0, vmax=1)
ax2[0, 1].set_title(f"reconstruction (latent {latent_dim})")

# repeat for training data
with torch.no_grad():
    x = next(iter(train_loader))
    x = standardizex(x[0]).to(device)  # unwrap & standardize
    x_pred, mean_pred, logvar_pred = model(x)
    x_orig = standardizex.inverse(x.cpu())[sample_ind, 0]
    x_recon = standardizex.inverse(x_pred.detach().cpu())[sample_ind, 0]

ax2[1, 0].imshow(x_orig, cmap="binary", vmin=0, vmax=1)
ax2[1, 0].set_title("original (train)")
ax2[1, 1].imshow(x_recon, cmap="binary", vmin=0, vmax=1)
ax2[1, 1].set_title(f"reconstruction (latent {latent_dim})")

for a in ax2.flatten():
    a.set_aspect("equal")
    a.axis("off")
    a.set_rasterized(True)

fig2.tight_layout(pad=1)
plt.savefig(BASE_DIR / f"../../tmp/b{beta}_l{latent_dim}_vae_pred.png")

plt.show()
# print("Mean:\n", mean_pred[sample_ind])
# print("LogVar:\n", logvar_pred[sample_ind])

plt.plot( mean_pred[sample_ind].cpu(), 'o', label="mean")
plt.plot( logvar_pred[sample_ind].cpu(), 'o', label="logvar")
plt.legend()
plt.title(f"VAE latent_dim = {latent_dim}")