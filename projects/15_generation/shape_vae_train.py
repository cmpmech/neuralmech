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
epochs = 600 #500 #500 #500 #300 #600
lr = 1e-2 #4e-3 #1e-2 #4e-3 #2e-3
weight_decay = 1e-2 #0 #1e-2
batch_size = 64 #32

# beta = 0. # GOOD RECONSTRUCTION
beta = 0.05 # latents goodish/reconstruction okayish
# beta = 0.2 # GOOD LATENT





# beta = lambda epoch : 0.01 # VERY GOOD RECONSTRUCTION, but latent not so good

# beta = lambda epoch : 0.04 / epochs * epoch + 0.01 # VERY GOOD LATENT SPACE -> maybe increase iterations? -> didn't help
# beta = lambda epoch : 0.02 / epochs * epoch + 0.01 # COULD BE A COMPROMISE



# beta = lambda epoch : 100


# TODO next step: testing -> interpolation plot as

# define loss TODO: could also both be with mean/sum
recon_loss = nn.MSELoss(reduction="mean")


def kl_div(mean_pred, logvar_pred):
    var_pred = torch.exp(logvar_pred)
    kl = 0.5 * torch.mean(var_pred + mean_pred**2 - logvar_pred - 1)
    return kl
# def kl_div(mean_pred, logvar_pred):
#     var_pred = torch.exp(logvar_pred)
#     kl = 0.5 * torch.sum(var_pred + mean_pred**2 - logvar_pred - 1, dim=1)
#     return kl.mean()  # mean over batch, sum over latent dims



def cost_fun(x_pred, mean_pred, logvar_pred, x, beta=1.0):
    mse = recon_loss(x_pred, x)
    kl = kl_div(mean_pred, logvar_pred)
    return mse + beta * kl


# ---------------------------- model settings ----------------------------
base, depth, latent_dim = 2, 5, 2 #2  # 2 as latent_dim for visualization
conv_layers = 1
channel_dim = 1
kernel_size = 3
act = partial(nn.PReLU, init=0.2)

# ----------------------------- prepare data -----------------------------
domain_size = 128

labels = ["circle", "ellipse", "square", "triangle", "cross", "star"]
data = []
for label in labels:
    data.append(
        torch.from_numpy(
            np.load(BASE_DIR / f"../../data/shapes_{label}_{domain_size}.npy")
        )
    )
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
channels, strides = build_ae_cnn_config(depth, conv_layers, channel_dim, base)
channels[0] = 1  # true input size (in case channel_dim != 1)
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
Encoder.append(nn.Flatten())
Encoder.append(MLP(layers[:-1] + [layers[-1] * 2], [act()]))

upsamplings = [
    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
    if s == 2
    else None
    for s in strides[::-1]
]


Decoder = nn.Sequential()
Decoder.append(MLP(layers[::-1], [act()] * (len(layers) - 1)))
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
    )
)

model = VAE(Encoder, Decoder).to(device)
init_weights(model, act())
summary(model, (1, 1, domain_size, domain_size), depth=4)


# ------------------------ instantiate optimizer -------------------------
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=epochs, eta_min=lr * 1e-2
)
scheduler = None

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
        x_pred, mean_pred, logvar_pred = model(x)
        cost = cost_fun(x_pred, mean_pred, logvar_pred, x, beta)
        # x_pred = model(x)
        # cost = cost_fun(x_pred, x)
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
            # x_pred = model(x)
            # cost = cost_fun(x_pred, x)
            val_cost[epoch] += cost.item()
        val_cost[epoch] /= len(val_loader)  # avg per batch

    if epoch % print_every == 0:
        pbar.set_postfix(
            {"train": f"{train_cost[epoch]:.2e}", "val": f"{val_cost[epoch]:.2e}"}
        )

# ----------------------------- export model -----------------------------
model.standardizer = standardizex  # just for saving
torch.save(model, BASE_DIR / f"../../models/shape_vae_{beta}_{domain_size}.pt2")

# ---------------------------- postprocessing ----------------------------
# fig, ax = plt.subplots()
# ax.plot(train_cost, "k")
# ax.plot(val_cost, "r")
# ax.set_yscale("log")
# plt.show()


##########################################################################
# TRAINING DATA
from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

fig, ax = plt.subplots()
ax.plot(train_cost, "k")
ax.plot(val_cost, "r")
ax.set_yscale("log")
plt.savefig(BASE_DIR / f"../../tmp/history_{timestamp}.png")
plt.show()


model.eval()
samples = 4
z = torch.randn(samples, latent_dim, device=device)
with torch.no_grad():
    x_pred = standardizex.inverse(model.decode(z).cpu())[:,0,:,:]

fig, ax = plt.subplots(1, 4, figsize=(8, 1), dpi=domain_size)
for i in range(4):
    ax[i].imshow(x_pred[i], cmap="binary", vmin=0, vmax=1)
    ax[i].set_aspect("equal")
    ax[i].axis("off")
    ax[i].set_rasterized(True)
fig.tight_layout(pad=0)
plt.savefig(BASE_DIR / f"../../tmp/prediction_{timestamp}.png")
plt.show()


# latent space
sampling_steps = 4

if latent_dim == 2:
    fig, ax = plt.subplots()

    with torch.no_grad():
        for _ in range(sampling_steps):
            for x in train_loader:
                distributions = model.encode(standardizex(x[0]).to(device))
                mean, logvar = torch.chunk(distributions, chunks=2, dim=1)
                std = torch.exp(0.5 * logvar)
                z = mean + torch.randn_like(std) * std
                z = z.cpu()
                ax.plot(z[:,0], z[:,1], 'ko', markersize=2)

            for x in val_loader:
                distributions = model.encode(standardizex(x[0]).to(device))
                mean, logvar = torch.chunk(distributions, chunks=2, dim=1)
                std = torch.exp(0.5 * logvar)
                z = mean + torch.randn_like(std) * std
                z = z.cpu()
                ax.plot(z[:,0], z[:,1], 'ro', markersize=2)

        # for x in train_loader:
        #     _, z, _ = model(standardizex(x[0]).to(device))
        #     z = z.cpu()
        #     ax.plot(z[:,0], z[:,1], 'ko')
        # for x in val_loader:
        #     _, z, logvar = model(standardizex(x[0]).to(device)) # TODO remove logvar
        #     z = z.cpu()
        #     ax.plot(z[:,0], z[:,1], 'ro')
        #     print(logvar.mean())
        #     print(z.mean())
    plt.savefig(BASE_DIR / f"../../tmp/latents_{timestamp}.png")
    plt.show()

# TODO DEBUGGING -> ADD MORE SAMPLES + CLEAN UP
# DO SAME LATENT SPACE SAMPLING WITH BETA NOT AS HIGH
# INCREASE SAMPLES BACK TO 500






# fig, ax = plt.subplots(10, 3, figsize=(6, 20), dpi=domain_size)




# fig, ax = plt.subplots(10, 3, figsize=(6, 20), dpi=domain_size)
# x = next(iter(train_loader))
# # x = next(iter(val_loader))
# x = standardizex(x[0]).to(device)  # unwrap & standardize
# x_pred = model(x)
# for i in range(10):
#     ax[i, 0].imshow(standardizex.inverse(x.cpu())[i, 0], cmap="binary", vmin=0, vmax=1)
#     ax[i, 1].imshow(
#         standardizex.inverse(x_pred.detach().cpu())[i, 0], cmap="binary", vmin=0, vmax=1
#     )
#     ax[i, 2].imshow(((x_pred.detach() - x).cpu() ** 2)[i, 0], cmap="hot_r")

#     for j in range(3):
#         ax[i, j].set_aspect("equal")
#         ax[i, j].axis("off")
#         ax[i, j].set_rasterized(True)
# fig.tight_layout(pad=0)
# plt.savefig(BASE_DIR / f"../../tmp/prediction_{timestamp}.png")
# plt.show()
