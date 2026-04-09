from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchinfo import summary
from tqdm import tqdm

from DL import Standardizer, init_weights
from NN import AE, DCN, MLP, VAE

BASE_DIR = Path(__file__).parent
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
# -------------------------- training settings ---------------------------
epochs = 500
lr = 1e-2  # 1e-2 #5e-3 #5e-3 #1e-2
weight_decay = 1e-10
batch_size = 32

# define loss
reconstruction_loss = nn.MSELoss(reduction="sum")
kl_anneal_epochs = 400
kl_weight_final = 1e-1


def cost_fun(x_pred, mean_pred, logvar_pred, x, epoch):
    mse = reconstruction_loss(x_pred, x)
    var_pred = torch.exp(logvar_pred)
    kl = -0.5 * torch.sum(1 + logvar_pred - mean_pred.pow(2) - var_pred)
    beta = min(epoch / kl_anneal_epochs, 1.0) * kl_weight_final
    return mse + beta * kl, mse.detach(), kl.detach()


# ---------------------------- model settings ----------------------------
domain_size, channel_dim = 128, 1

# base, depth, latent_dim = 2, 5, 32
base, depth, latent_dim = 2, 4, 64  # 32 #64 #32
conv_layers = 2
kernel_size = 3
act = nn.GELU(approximate="tanh")

# ----------------------------- prepare data -----------------------------
data = torch.from_numpy(np.load(BASE_DIR / f"../../data/fibers_{domain_size}.npy"))
data = data.to(torch.float32).unsqueeze(1)

clip = 80  # 40 # TODO
data = data[:clip]

dataset = TensorDataset(data)
train_data, val_data = random_split(dataset, [0.9, 0.1])  # TODO 0.9, 0.1
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=len(val_data), shuffle=True)  # full batch

X_train = train_data.dataset.tensors[0][train_data.indices]
standardizex = Standardizer(X_train, dim=(0, 3))

# INSTANTIATE MODEL V2
# channels = [channel_dim * (base ** (i)) for i in range(depth + 1)]
# Encoder = nn.Sequential()
#
# # V1
# Encoder.append(DCN(channels,
#                    [act for _ in range(len(channels) - 1)],
#                    kernel_size, stride=2, padding=1, dim=2))


# V2
def build_encoder_config(depth, conv_layers, channel_dim, base):
    channels, strides = [], []

    for i in range(depth + 1):
        # starting channel at depth i
        channels.append(channel_dim * (base**i))
        strides.append(1)

        # intermediate conv layers (except last depth)
        if i < depth:
            for _ in range(conv_layers):
                channels.append(channel_dim * (base ** (i + 1)))
                strides.append(1)
            strides[-1] = 2  # last stride in each group downsamples

    return channels, strides[:-1]


# ENCODER
channels, strides = build_encoder_config(depth, conv_layers, channel_dim, base)
red_domain_size = domain_size // 2**depth
layers = [red_domain_size**2 * channel_dim * base**depth, latent_dim]

channels_encoder = channels.copy()
Encoder = nn.Sequential()
Encoder.append(
    DCN(
        channels_encoder,
        [act for _ in range(len(channels) - 1)],
        kernel_size,
        stride=strides,
        padding=kernel_size // 2,
        dim=2,
        normalizations=[nn.GroupNorm(1, channel) for channel in channels_encoder[1:]],
    )
)
Encoder.append(nn.Flatten())
Encoder.append(MLP([red_domain_size**2 * channels[-1], latent_dim * 2], [None]))

# DECODER
upsamplings = [
    nn.Upsample(scale_factor=2, mode="nearest") if s == 2 else None
    for s in strides[::-1]
]

# print(strides)
# print(channels)
# summary(Encoder, (1, 1, domain_size, domain_size))
Decoder = nn.Sequential()
Decoder.append(MLP([latent_dim, red_domain_size**2 * channels[-1]], [act]))
Decoder.append(nn.Unflatten(1, (channels[-1], red_domain_size, red_domain_size)))
Decoder.append(
    DCN(
        channels[::-1],
        [act for _ in range(len(channels) - 2)],
        kernel_size,
        stride=1,
        padding=kernel_size // 2,
        dim=2,
        resamplings=upsamplings,
        normalizations=[nn.GroupNorm(1, channel) for channel in channels[:-1:-1]],
    )
)
# normalizations=[nn.BatchNorm2d(channels[i]) for i in range(len(channels) - 2, 0, -1)]))

# AUTOENCODER
model = VAE(Encoder, Decoder).to(device)
init_weights(model, act)


print(layers)

# summary(model, (1, channel_dim, domain_size, domain_size))
#
# # -------------------------- instantiate model ---------------------------
# channels = [channel_dim * (base ** (i)) for i in range(depth + 1)]
# red_domain_size = domain_size // 2**(len(channels) - 1)
# layers = [red_domain_size**2 * base**(len(channels) - 1), latent_dim] # assuming compression by 2
# upsamplings = [nn.Upsample(scale_factor=2, # alternative 'nearest'
#                            mode='nearest')] * (len(channels) - 1)
#
# Encoder = nn.Sequential()
# Encoder.append(DCN(channels,
#                    [act for _ in range(len(channels) - 1)],
#                    kernel_size, stride=2, padding=1, dim=2))
# # Encoder.append(nn.Flatten())
# # Encoder.append(MLP(layers, [act for _ in range(len(layers) - 1)]))
#
# Decoder = nn.Sequential()
# # Decoder.append(MLP(layers[::-1],
# #                    [act for _ in range(len(layers) - 1)]))
# # Decoder.append(nn.Unflatten(1, (channels[-1],
# #                                 red_domain_size, red_domain_size)))
#
# Decoder.append(DCN(channels[::-1],
#                    [act for _ in range(len(channels) - 2)],
#                    kernel_size, stride=1, padding=1, dim=2,
#                    resamplings=upsamplings))
#
# model = AE(Encoder, Decoder).to(device)
# init_weights(model, act)
#
# ------------------------ instantiate optimizer -------------------------
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
# scheduler = None
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=epochs, eta_min=lr * 1e-2
)

# ------------------------------- training -------------------------------
print_every = 10
train_cost, train_mse, train_kl = [0] * epochs, [0] * epochs, [0] * epochs
val_cost, val_mse, val_kl = [0] * epochs, [0] * epochs, [0] * epochs
pbar = tqdm(range(epochs), desc="Training: ", ncols=90)
for epoch in pbar:
    model.train()
    for x in train_loader:
        x = x[0]
        if torch.rand(1) > 0.5: x = torch.flip(x, dims=[-1])   # random H flip
        if torch.rand(1) > 0.5: x = torch.flip(x, dims=[-2])   # random V flip
        x = standardizex(x).to(device)  # standardize
        optimizer.zero_grad()
        x_pred, mean_pred, logvar_pred = model(x)
        cost, mse, kl = cost_fun(x_pred, mean_pred, logvar_pred, x, epoch)
        cost.backward()
        optimizer.step()
        train_cost[epoch] += cost.item()
        train_mse[epoch] += mse.item()
        train_kl[epoch] += kl.item()
    train_cost[epoch] /= len(train_data)  # avg over data
    train_mse[epoch] /= len(train_data)
    train_kl[epoch] /= len(train_data)
    if scheduler is not None:
        scheduler.step()

    model.eval()
    with torch.no_grad():
        for x in val_loader:
            x = standardizex(x[0]).to(device)  # unwrap & standardize
            x_pred, mean_pred, logvar_pred = model(x)
            cost, mse, kl = cost_fun(x_pred, mean_pred, logvar_pred, x, epoch)
            val_cost[epoch] += cost.item()
            val_mse[epoch] += mse.item()
            val_kl[epoch] += kl.item()
        val_cost[epoch] /= len(val_data)  # avg over data
        val_mse[epoch] /= len(val_data)
        val_kl[epoch] /= len(val_data)
    if epoch % print_every == 0:
        pbar.set_postfix(
            {"train": f"{train_cost[epoch]:.2e}", "val": f"{val_cost[epoch]:.2e}"}
        )


# TESTING
fig, ax = plt.subplots()
ax.plot(train_cost, "k")
ax.plot(val_cost, "r")
ax.set_yscale("log")
plt.show()

# TRAINING DATA
model.eval()

# x = next(iter(train_loader))
x = next(iter(val_loader))
x = standardizex(x[0]).to(device)  # unwrap & standardize
x_pred, _, _ = model(x)

fig, ax = plt.subplots(1, 2, figsize=(4, 2), dpi=domain_size)
ax[0].imshow(standardizex.inverse(x.cpu())[0, 0], cmap="binary_r", vmin=-1, vmax=1)
ax[1].imshow(
    standardizex.inverse(x_pred.detach().cpu())[0, 0], cmap="binary_r", vmin=-1, vmax=1
)
for i in range(2):
    ax[i].set_aspect("equal")
    ax[i].axis("off")
    ax[i].set_rasterized(True)
fig.tight_layout(pad=0)
plt.show()


# test = X_train[0:1]
# print(test.shape)
# print(model(test.to(device)).shape)

# SAMPLING
model.eval()
with torch.no_grad():
    z = torch.randn(1, latent_dim, device=device)
    x_sample = model.decode(z)
    x_sample = standardizex.inverse(x_sample.cpu())

fig, ax = plt.subplots(figsize=(2, 2), dpi=domain_size)
ax.imshow(x_sample[0, 0], cmap="binary_r", vmin=-1, vmax=1)
ax.set_aspect("equal")
ax.axis("off")
ax.set_rasterized(True)
fig.tight_layout(pad=0)
plt.show()
