import time
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
EPOCHS = 800
LR = 1e-3
REGULARIZATION = 1e-2
BATCH_SIZE = 32
BETA = 16.0  # kl weight; 1 is the plain evidence lower bound
PRIOR_STEPS = 20000
PRIOR_REGULARIZATION = 1e-4
PRIOR_LATENT = 64
PRIOR_WIDTH = 512
PRIOR_BETA = 0.1

# define loss
recon_loss = nn.MSELoss(reduction="sum")


def kl_div(mean_pred, logvar_pred):
    logvar_pred = logvar_pred.clamp(-10, 10)  # prevent exp overflow
    var_pred = torch.exp(logvar_pred)
    kl = 0.5 * (var_pred + mean_pred**2 - logvar_pred - 1)
    return kl.mean(dim=0)


def cost_fun(x_pred, mean_pred, logvar_pred, x):
    recon = recon_loss(x_pred, x) / x.shape[0]
    kl = kl_div(mean_pred, logvar_pred).sum()
    return recon + BETA * kl, recon.detach(), kl.detach()


# model settings
RESOLUTION = 256
DEPTH = 5
CONV_LAYERS = 1
CHANNEL_DIM = 2
KERNEL_SIZE = 3
LATENT_CHANNELS = 4  # per cell of the coarse grid
THRESHOLD = 0.5
act = partial(nn.PReLU, init=0.2)

# ------------------------------------ prepare data -----------------------------------
data = torch.from_numpy(np.load(DATA_DIR / f"fibers_{RESOLUTION}.npy"))
data = data.to(torch.float32).unsqueeze(1)

dataset = TensorDataset(data)
split = torch.Generator().manual_seed(0)
train_data, val_data = random_split(dataset, [0.9, 0.1], generator=split)
train_loader = DataLoader(
    train_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True
)

X_train = train_data.dataset.tensors[0][train_data.indices]
X_val = train_data.dataset.tensors[0][val_data.indices]
standardizex = Standardizer(X_train, dim=(0, 2, 3))


# data augmentation
def transform(x, k):
    return torch.rot90(x.flip(-1) if k >= 4 else x, k % 4, [-2, -1])


def augment(x):
    return torch.cat([transform(part, k) for k, part in enumerate(x.chunk(8))])


# --------------------------- instantiate model & optimizer ---------------------------
channels, strides = build_ae_cnn_config(DEPTH, CONV_LAYERS, CHANNEL_DIM, 2)
channels[0] = 1  # true input size (in case CHANNEL_DIM != 1)
red_resolution = RESOLUTION // 2**DEPTH
LATENT = red_resolution**2 * LATENT_CHANNELS

Encoder = nn.Sequential(
    DCN(
        channels + [2 * LATENT_CHANNELS],
        [[nn.GroupNorm(1, channel), act()] for channel in channels[1:]],
        [KERNEL_SIZE] * (len(channels) - 1) + [1],
        stride=strides + [1],
        padding=[KERNEL_SIZE // 2] * (len(channels) - 1) + [0],
        bias=[False] * (len(channels) - 1) + [True],
        dim=2,
    ),
    nn.Flatten(),
)

upsamplings = [None] + [
    nn.Upsample(scale_factor=2, mode="nearest") if s == 2 else None
    for s in strides[::-1]
]

Decoder = nn.Sequential(
    nn.Unflatten(1, (LATENT_CHANNELS, red_resolution, red_resolution)),
    DCN(
        [LATENT_CHANNELS] + channels[::-1],
        [[nn.GroupNorm(1, channel), act()] for channel in channels[::-1][:-1]],
        [1] + [KERNEL_SIZE] * (len(channels) - 1),
        stride=1,
        padding=[0] + [KERNEL_SIZE // 2] * (len(channels) - 1),
        pre_modules=upsamplings,
        dim=2,
    ),
)

model = VAE(Encoder, Decoder).to(device)
init_weights(model, act())
summary(model, (1, 1, RESOLUTION, RESOLUTION), depth=4)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=REGULARIZATION)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=EPOCHS, eta_min=LR * 1e-2
)

# -------------------------------------- training -------------------------------------
print_every = 50
train_cost = [0] * EPOCHS
val_cost = [0] * EPOCHS
val_rate = [0] * EPOCHS
best_cost = float("inf")
best_state = None
tic = time.time()
pbar = tqdm(range(EPOCHS), desc="Training: ", ncols=90)
for epoch in pbar:
    model.train()
    for x in train_loader:
        x = standardizex(augment(x[0])).to(device)
        optimizer.zero_grad()
        x_pred, mean_pred, logvar_pred = model(x)
        cost, recon, kl = cost_fun(x_pred, mean_pred, logvar_pred, x)
        cost.backward()
        optimizer.step()
        train_cost[epoch] += recon.item()
    train_cost[epoch] /= len(train_loader)
    scheduler.step()

    model.eval()
    with torch.no_grad():
        x = standardizex(X_val).to(device)
        x_pred, mean_pred, logvar_pred = model(x)
        cost, recon, kl = cost_fun(x_pred, mean_pred, logvar_pred, x)
    val_cost[epoch] = recon.item()
    val_rate[epoch] = kl.item()
    if cost.item() < best_cost:
        best_cost = cost.item()
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if epoch % print_every == 0:
        pbar.set_postfix(
            {"train": f"{train_cost[epoch]:.2e}", "val": f"{val_cost[epoch]:.2e}"}
        )
toc = time.time()
print(f"elapsed time {toc - tic:.2f} s")

model.load_state_dict(best_state)
model.eval()

# ----------------------------------- learned prior -----------------------------------
with torch.no_grad():
    encoded = [
        model.encode(standardizex(transform(X_train, k)).to(device)) for k in range(8)
    ]
    mean_train, _ = torch.chunk(torch.cat(encoded), chunks=2, dim=1)
standardizez = Standardizer(mean_train)

prior = VAE(
    MLP([LATENT, PRIOR_WIDTH, PRIOR_WIDTH, 2 * PRIOR_LATENT], [act(), act()]),
    MLP([PRIOR_LATENT, PRIOR_WIDTH, PRIOR_WIDTH, LATENT], [act(), act()]),
).to(device)
init_weights(prior, act())
prior_optimizer = torch.optim.AdamW(
    prior.parameters(), lr=LR, weight_decay=PRIOR_REGULARIZATION
)
prior_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    prior_optimizer, T_max=PRIOR_STEPS, eta_min=LR * 1e-2
)

tic = time.time()
pbar = tqdm(range(PRIOR_STEPS), desc="Prior:    ", ncols=90)
for step in pbar:
    ids = torch.randint(0, mean_train.shape[0], (4 * BATCH_SIZE,), device=device)
    z = standardizez(mean_train[ids])
    prior_optimizer.zero_grad()
    z_pred, mean_pred, logvar_pred = prior(z)
    recon = recon_loss(z_pred, z) / z.shape[0]
    cost = recon + PRIOR_BETA * kl_div(mean_pred, logvar_pred).sum()
    cost.backward()
    prior_optimizer.step()
    prior_scheduler.step()
    if step % print_every == 0:
        pbar.set_postfix({"cost": f"{cost.item():.2e}"})
prior.eval()
toc = time.time()
print(f"elapsed time {toc - tic:.2f} s")

# ------------------------------------- validation ------------------------------------
with torch.no_grad():
    x = standardizex(X_val).to(device)
    x_pred, _, _ = model(x)
    x_normal = model.decode(torch.randn(X_val.shape[0], LATENT, device=device))
    z = prior.decode(torch.randn(X_val.shape[0], PRIOR_LATENT, device=device))
    x_learned = model.decode(standardizez.inverse(z))
x_recon = standardizex.inverse(x_pred.cpu()) >= THRESHOLD
x_normal = standardizex.inverse(x_normal.cpu()).clamp(0, 1)
x_learned = standardizex.inverse(x_learned.cpu()).clamp(0, 1)

intersection = (x_recon * X_val).sum(dim=(1, 2, 3))
union = ((x_recon + X_val) >= 1).sum(dim=(1, 2, 3))
iou = (intersection / union).mean()
print(f"iou {iou:.3f}")

# --------------------------------------- export --------------------------------------
model.standardizer = standardizex
model.prior = prior
model.standardizez = standardizez
torch.save(model, MODEL_DIR / f"fiber_vae_{LATENT}_{BETA}_{RESOLUTION}.pt2")

# ----------------------------------- postprocessing ----------------------------------
fig, ax = plt.subplots()
ax.plot(train_cost, "k")
ax.plot(val_cost, "r")
ax.plot(val_rate, "b")
ax.set_yscale("log")
ax.set_xlabel("epoch")
ax.set_ylabel("cost")
plt.show()

fig, ax = plt.subplots(4, 4, figsize=(8, 8), dpi=RESOLUTION // 2)
for i in range(4):
    ax[0, i].imshow(X_val[i, 0], cmap="binary", vmin=0, vmax=1)
    ax[1, i].imshow(x_recon[i, 0], cmap="binary", vmin=0, vmax=1)
    ax[2, i].imshow(x_normal[i, 0], cmap="binary", vmin=0, vmax=1)
    ax[3, i].imshow(x_learned[i, 0], cmap="binary", vmin=0, vmax=1)
for axis in ax.ravel():
    axis.set_aspect("equal")
    axis.axis("off")
    axis.set_rasterized(True)
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.show()
