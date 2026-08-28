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
EPOCHS = 150
LR = 1e-3
REGULARIZATION = 1e-2
BATCH_SIZE = 32
BETA = 3.0  # kl weight; 1 is the plain evidence lower bound

# define loss
# both terms are summed per image, so BETA weighs nats against nats and BETA = 1 is the
# evidence lower bound. with a mean reduction the weight would instead have to absorb
# the ratio of pixel count to latent size
recon_loss = nn.BCEWithLogitsLoss(reduction="sum")


def kl_div(mean_pred, logvar_pred):
    var_pred = torch.exp(logvar_pred)
    kl = 0.5 * torch.sum(var_pred + mean_pred**2 - logvar_pred - 1)
    return kl / mean_pred.shape[0]


def cost_fun(x_pred, mean_pred, logvar_pred, x):
    recon = recon_loss(x_pred, x) / x.shape[0]
    return recon + BETA * kl_div(mean_pred, logvar_pred)


# model settings
RESOLUTION = 256
DEPTH = 5
CONV_LAYERS = 1
CHANNEL_DIM = 2
KERNEL_SIZE = 3
LATENT = 64
THRESHOLD = 0.5
act = partial(nn.PReLU, init=0.2)

# ------------------------------------ prepare data -----------------------------------
data = torch.from_numpy(np.load(DATA_DIR / f"fibers_{RESOLUTION}.npy"))
data = data.to(torch.float32).unsqueeze(1)

dataset = TensorDataset(data)
train_data, val_data = random_split(dataset, [0.9, 0.1])
train_loader = DataLoader(
    train_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True
)
val_loader = DataLoader(val_data, batch_size=len(val_data), shuffle=True)  # full batch

X_train = train_data.dataset.tensors[0][train_data.indices]
standardizex = Standardizer(X_train, dim=(0, 2, 3))


def augment(x):  # the microstructures are invariant under flips and quarter turns
    if torch.rand(1) < 0.5:
        x = torch.flip(x, [-1])
    return torch.rot90(x, int(torch.randint(4, (1,))), [-2, -1])


# --------------------------- instantiate model & optimizer ---------------------------
channels, strides = build_ae_cnn_config(DEPTH, CONV_LAYERS, CHANNEL_DIM, 2)
channels[0] = 1  # true input size (in case CHANNEL_DIM != 1)
red_resolution = RESOLUTION // 2**DEPTH
layers = [red_resolution**2 * channels[-1], LATENT]

# the code is a flat vector: a spatial latent samples every position independently and
# decodes into speckle rather than into whole fibers
Encoder = nn.Sequential(
    DCN(
        channels,
        [[nn.GroupNorm(1, channel), act()] for channel in channels[1:]],
        KERNEL_SIZE,
        stride=strides,
        padding=KERNEL_SIZE // 2,
        dim=2,
    ),
    nn.Flatten(),
    MLP([layers[0], 2 * layers[1]]),  # mean and logvar stacked
)

upsamplings = [
    nn.Upsample(scale_factor=2, mode="nearest") if s == 2 else None
    for s in strides[::-1]
]

Decoder = nn.Sequential(
    MLP(layers[::-1], post_modules=[act()]),
    nn.Unflatten(1, (channels[-1], red_resolution, red_resolution)),
    DCN(
        channels[::-1],
        [[nn.GroupNorm(1, channel), act()] for channel in channels[-2:0:-1]],
        KERNEL_SIZE,
        stride=1,
        padding=KERNEL_SIZE // 2,
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
print_every = 10
train_cost = [0] * EPOCHS
val_cost = [0] * EPOCHS
tic = time.time()
pbar = tqdm(range(EPOCHS), desc="Training: ", ncols=90)
for epoch in pbar:
    model.train()
    for x in train_loader:
        x_target = augment(x[0])  # binary image, the target of the logits
        x = standardizex(x_target).to(device)  # standardized encoder input
        x_target = x_target.to(device)
        optimizer.zero_grad()
        x_pred, mean_pred, logvar_pred = model(x)
        cost = cost_fun(x_pred, mean_pred, logvar_pred, x_target)
        cost.backward()
        optimizer.step()
        train_cost[epoch] += cost.item()
    train_cost[epoch] /= len(train_loader)  # avg per batch
    scheduler.step()

    model.eval()
    with torch.no_grad():
        for x in val_loader:
            x_target = x[0].to(device)
            x = standardizex(x[0]).to(device)
            x_pred, mean_pred, logvar_pred = model(x)
            cost = cost_fun(x_pred, mean_pred, logvar_pred, x_target)
            val_cost[epoch] += cost.item()
        val_cost[epoch] /= len(val_loader)  # avg per batch

    if epoch % print_every == 0:
        pbar.set_postfix(
            {"train": f"{train_cost[epoch]:.2e}", "val": f"{val_cost[epoch]:.2e}"}
        )
toc = time.time()
print(f"elapsed time {toc - tic:.2f} s")

# ------------------------------------ export model -----------------------------------
model.standardizer = standardizex  # just for saving
torch.save(model, MODEL_DIR / f"fiber_vae_{LATENT}_{BETA}_{RESOLUTION}.pt2")

# ----------------------------------- postprocessing ----------------------------------
fig, ax = plt.subplots()
ax.plot(train_cost, "k")
ax.plot(val_cost, "r")
ax.set_yscale("log")
ax.set_xlabel("epoch")
ax.set_ylabel("cost")
plt.show()

model.eval()
x = next(iter(val_loader))
x_target = x[0].to(device)
x = standardizex(x[0]).to(device)
with torch.no_grad():
    x_pred, mean_pred, logvar_pred = model(x)
    x_gen = model.decode(torch.randn(x_target.shape[0], LATENT, device=device))
x_recon = torch.sigmoid(x_pred) >= THRESHOLD
x_gen = torch.sigmoid(x_gen) >= THRESHOLD

# a code drawn from the prior has to decode into a microstructure of its own, which is
# the test the reconstruction alone does not make
fig, ax = plt.subplots(3, 4, figsize=(8, 6), dpi=RESOLUTION // 2)
for i in range(4):
    ax[0, i].imshow(x_target.cpu()[i, 0], cmap="binary", vmin=0, vmax=1)
    ax[1, i].imshow(x_recon.cpu()[i, 0], cmap="binary", vmin=0, vmax=1)
    ax[2, i].imshow(x_gen.cpu()[i, 0], cmap="binary", vmin=0, vmax=1)
for axis in ax.ravel():
    axis.set_aspect("equal")
    axis.axis("off")
    axis.set_rasterized(True)
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.show()
