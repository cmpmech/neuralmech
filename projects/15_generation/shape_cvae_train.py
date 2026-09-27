from datetime import datetime
from functools import partial
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchinfo import summary
from tqdm import tqdm

from DL import Standardizer, build_ae_cnn_config, init_weights
from NN import DCN, MLP

BASE_DIR = Path(__file__).parent
DATA_DIR = (BASE_DIR / "../../data").resolve()
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True

# -------------------------------------- settings -------------------------------------
# hyperparameters
epochs = 600
lr = 1e-2
weight_decay = 1e-2
batch_size = 64
beta = 0.15  # tighter than shape_vae_train.py: prior draws for the widely translated
# ellipse class land off-manifold at lower beta

# define loss
recon_loss = nn.BCEWithLogitsLoss(reduction="mean")


def kl_div(mean_pred, logvar_pred):
    var_pred = torch.exp(logvar_pred)
    return 0.5 * torch.mean(var_pred + mean_pred**2 - logvar_pred - 1)


def cost_fun(x_pred, mean_pred, logvar_pred, x, beta=1.0):
    bce = recon_loss(x_pred, x)
    kl = kl_div(mean_pred, logvar_pred)
    return bce + beta * kl


# model settings
base, depth = 2, 5
latent_dim = 16
conv_layers = 1
channel_dim = 1
kernel_size = 3
act = partial(nn.PReLU, init=0.2)

# conditioning
domain_size = 128
labels = ["circle", "ellipse", "square", "triangle", "cross", "star"]
classes = len(labels)
samples_per_class = 8  # decoded per class in the conditional sampling grid
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")


# --------------------------------------- helper --------------------------------------
class ConditionalVAE(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encode = encoder
        self.decode = decoder

    def reparameterize(self, mean, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mean + eps * std
        return mean

    def forward(self, x, c):
        # the label enters the encoder as constant feature maps so the convolutions see
        # it everywhere, and the decoder as plain features appended to the latent vector
        c_maps = c[:, :, None, None].expand(-1, -1, x.shape[2], x.shape[3])
        distributions = self.encode(torch.cat([x, c_maps], dim=1))
        mean, logvar = torch.chunk(distributions, chunks=2, dim=1)
        z = self.reparameterize(mean, logvar)
        y = self.decode(torch.cat([z, c], dim=1))
        return y, mean, logvar


# ----------------------------------- prepare data ------------------------------------
# the six classes are stored one file per shape with no labels, so the class index is
# reconstructed here from the load order
X = []
targets = []
for idx, label in enumerate(labels):
    shapes = np.load(DATA_DIR / f"shapes_{label}_{domain_size}.npy")
    X.append(shapes)
    targets.append(np.full(len(shapes), idx))

X = torch.from_numpy(np.concatenate(X, axis=0)).to(torch.float32).unsqueeze(1)
targets = torch.from_numpy(np.concatenate(targets)).to(torch.long)
C = nn.functional.one_hot(targets, classes).to(torch.float32)

dataset = TensorDataset(X, C)
train_data, val_data = random_split(dataset, [0.9, 0.1])
train_loader = DataLoader(
    train_data, batch_size=batch_size, shuffle=True, drop_last=True
)
val_loader = DataLoader(val_data, batch_size=len(val_data), shuffle=True)

X_train = train_data.dataset.tensors[0][train_data.indices]
standardizex = Standardizer(X_train, dim=(0, 2, 3))

# --------------------------- instantiate model & optimizer ---------------------------
channels, strides = build_ae_cnn_config(depth, conv_layers, channel_dim, base)
channels[0] = 1 + classes  # image plus one constant map per class
red_domain_size = domain_size // 2**depth
flat_dim = red_domain_size**2 * channels[-1]

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
Encoder.append(MLP([flat_dim, latent_dim * 2], post_modules=[act()]))

upsamplings = [
    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
    if s == 2
    else None
    for s in strides[::-1]
]

decoder_channels = channels[::-1]
decoder_channels[-1] = 1  # the reconstruction has no condition maps to predict

Decoder = nn.Sequential()
Decoder.append(MLP([latent_dim + classes, flat_dim], post_modules=[act()]))
Decoder.append(nn.Unflatten(1, (channels[-1], red_domain_size, red_domain_size)))
Decoder.append(
    DCN(
        decoder_channels,
        [
            [nn.GroupNorm(1, channel), act()]
            for channel in decoder_channels[1:-1]
        ],
        kernel_size,
        stride=1,
        padding=kernel_size // 2,
        dim=2,
        pre_modules=upsamplings,
    )
)

model = ConditionalVAE(Encoder, Decoder).to(device)
init_weights(model, act())
summary(model, [(1, 1, domain_size, domain_size), (1, classes)], depth=4)

optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=epochs, eta_min=lr * 1e-2
)

# -------------------------------------- training -------------------------------------
print_every = 10
train_cost = [0] * epochs
val_cost = [0] * epochs

tic = time.time()
pbar = tqdm(range(epochs), desc="Training: ", ncols=90)
for epoch in pbar:
    model.train()
    for x, c in train_loader:
        x_target = x.to(device)
        c = c.to(device)
        x = standardizex(x).to(device)  # standardize on cpu, as in shape_vae_train.py
        optimizer.zero_grad()
        x_pred, mean_pred, logvar_pred = model(x, c)
        cost = cost_fun(x_pred, mean_pred, logvar_pred, x_target, beta)
        cost.backward()
        optimizer.step()
        train_cost[epoch] += cost.item()
    train_cost[epoch] /= len(train_loader)
    if scheduler is not None:
        scheduler.step()

    model.eval()
    with torch.no_grad():
        for x, c in val_loader:
            x_target = x.to(device)
            c = c.to(device)
            x = standardizex(x).to(device)
            x_pred, mean_pred, logvar_pred = model(x, c)
            cost = cost_fun(x_pred, mean_pred, logvar_pred, x_target, beta)
            val_cost[epoch] += cost.item()
        val_cost[epoch] /= len(val_loader)

    if epoch % print_every == 0:
        pbar.set_postfix(
            {"train": f"{train_cost[epoch]:.2e}", "val": f"{val_cost[epoch]:.2e}"}
        )
toc = time.time()
print(f"elapsed time {toc - tic:.2f} s")

# --------------------------------------- export --------------------------------------
model.standardizer = standardizex
torch.save(
    model,
    BASE_DIR / f"../../models/shape_cvae_{latent_dim}_{beta}_{domain_size}.pt2",
)

# ----------------------------------- postprocessing ----------------------------------
fig, ax = plt.subplots()
ax.plot(train_cost, "k")
ax.plot(val_cost, "r")
ax.set_yscale("log")
ax.set_xlabel("epoch")
ax.set_ylabel("loss")
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.savefig(RESULTS_DIR / f"shape_cvae_loss_{latent_dim}_{beta}_{run_id}.png")
plt.show()

# the point of conditioning: one latent draw per column, decoded once per class, so the
# same z produces the requested shape in every row
model.eval()
with torch.no_grad():
    z = torch.randn(samples_per_class, latent_dim, device=device)
    grid = torch.zeros(classes, samples_per_class, domain_size, domain_size)
    for idx in range(classes):
        c = nn.functional.one_hot(
            torch.full((samples_per_class,), idx, device=device), classes
        ).to(torch.float32)
        x_pred = model.decode(torch.cat([z, c], dim=1))
        grid[idx] = torch.sigmoid(x_pred)[:, 0].cpu()

fig, ax = plt.subplots(
    classes, samples_per_class, figsize=(samples_per_class, classes), dpi=domain_size
)
for i in range(classes):
    for j in range(samples_per_class):
        ax[i, j].imshow(grid[i, j].T, cmap="binary", origin="lower", vmin=0, vmax=1)
        ax[i, j].set_aspect("equal")
        ax[i, j].axis("off")
        ax[i, j].set_rasterized(True)
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.savefig(RESULTS_DIR / f"shape_cvae_samples_{latent_dim}_{beta}_{run_id}.png")
plt.show()

summary_path = RESULTS_DIR / f"shape_cvae_summary_{latent_dim}_{beta}_{run_id}.txt"
with open(summary_path, "w", encoding="utf-8") as f:
    f.write("shape conditional VAE training summary\n")
    f.write(f"run_id: {run_id}\n")
    f.write(f"device: {device}\n")
    f.write(f"conditioning: one-hot class label, {classes} classes\n")
    f.write(f"labels: {','.join(labels)}\n")
    f.write(f"domain_size: {domain_size}\n")
    f.write(f"samples: {len(X)}\n")
    f.write(f"epochs: {epochs}\n")
    f.write(f"batch_size: {batch_size}\n")
    f.write(f"learning_rate: {lr}\n")
    f.write(f"weight_decay: {weight_decay}\n")
    f.write(f"beta: {beta}\n")
    f.write(f"latent_dim: {latent_dim}\n")
    f.write(f"depth: {depth}\n")
    f.write("reconstruction_loss: BCEWithLogitsLoss\n")
    f.write(f"training_time_seconds: {toc - tic:.2f}\n")
    f.write(f"final_train_loss: {train_cost[-1]:.8e}\n")
    f.write(f"final_val_loss: {val_cost[-1]:.8e}\n")
print(f"saved {summary_path}")
