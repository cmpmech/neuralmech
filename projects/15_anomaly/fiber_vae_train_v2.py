import math
import time
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import ndimage
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchinfo import summary
from tqdm import tqdm

from DL import Standardizer, build_ae_cnn_config, init_weights
from NN import DCN, VAE, SlotDecoder, SlotEncoder

BASE_DIR = Path(__file__).parent
DATA_DIR = (BASE_DIR / "../../data").resolve()
MODEL_DIR = (BASE_DIR / "../../models").resolve()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True

# -------------------------------------- settings -------------------------------------
# hyperparameters
EPOCHS = 4000
LR = 1e-3
REGULARIZATION = 1e-2
BATCH_SIZE = 32
BETA = 32.0  # kl weight; 1 is the plain evidence lower bound

# define loss
# both terms are summed per image, so BETA weighs nats against nats and BETA = 1 is the
# evidence lower bound. with a mean reduction the weight would instead have to absorb
# the ratio of pixel count to latent size.
# BETA has to be read against the pixel count: the reconstruction is summed over
# RESOLUTION**2 pixels while the rate is not, so a given BETA throttles four times as
# hard on a 128 grid as it does here.
# unlike the flat code of v1, this latent really does trade. measured on a 128 grid, the
# aggregate posterior sits 8.0 nats from the prior at BETA = 1 and 1.6 nats at BETA = 8,
# while the reconstruction falls from 0.97 to 0.82 intersection over union. 8 there is
# the knee, and it is where the prior starts producing the right number of fibers, so
# BETA = 32 here. drop it towards 4 for reconstruction, raise it for the latent
recon_loss = nn.BCEWithLogitsLoss(reduction="sum")


def kl_div(mean_pred, logvar_pred):
    logvar_pred = logvar_pred.clamp(-10, 10)  # exp overflows in the early transient
    var_pred = torch.exp(logvar_pred)
    kl = 0.5 * (var_pred + mean_pred**2 - logvar_pred - 1)
    return kl.mean(dim=0)  # nats per dimension, so a single dimension stays visible


def cost_fun(x_pred, mean_pred, logvar_pred, x):
    recon = recon_loss(x_pred, x) / x.shape[0]
    kl = kl_div(mean_pred, logvar_pred).sum()
    return recon + BETA * kl, recon.detach(), kl.detach()


# model settings
# the code is a grid of GRID x GRID slots of CELL dimensions each, plus one GLOBAL
# vector shared by the whole image. a slot is anchored to a place, so the encoder never
# has to invent an order for the fibers, which is what forces the flat code of v1 to
# spend 64 dimensions on a microstructure that has about 21 degrees of freedom.
# DEPTH follows from how far it is from the slot grid up to the image
RESOLUTION = 256
GRID = 8
CELL = 4
GLOBAL = 4  # carries the factors that belong to the image, above all the fiber radius
LATENT = GRID**2 * CELL + GLOBAL
DEPTH = round(math.log2(RESOLUTION // GRID))
CONV_LAYERS = 1
CHANNEL_DIM = 2
KERNEL_SIZE = 3
THRESHOLD = 0.5
act = partial(nn.PReLU, init=0.2)

# ------------------------------------ prepare data -----------------------------------
data = torch.from_numpy(np.load(DATA_DIR / f"fibers_{RESOLUTION}.npy"))
data = data.to(torch.float32).unsqueeze(1)

dataset = TensorDataset(data)
split = torch.Generator().manual_seed(0)  # pinned, so the split survives edits above
train_data, val_data = random_split(dataset, [0.9, 0.1], generator=split)
train_loader = DataLoader(
    train_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True
)

X_train = train_data.dataset.tensors[0][train_data.indices]
X_val = train_data.dataset.tensors[0][val_data.indices]
standardizex = Standardizer(X_train, dim=(0, 2, 3))


# the microstructures are invariant under flips and quarter turns, so the eight
# transforms are free data. one per batch slice rather than one per batch, so a single
# gradient spans the whole symmetry group instead of being one correlated transform
def augment(x):
    slices = x.chunk(8)
    return torch.cat([
        torch.rot90(part.flip(-1) if k >= 4 else part, k % 4, [-2, -1])
        for k, part in enumerate(slices)
    ])


# --------------------------- instantiate model & optimizer ---------------------------
channels, strides = build_ae_cnn_config(DEPTH, CONV_LAYERS, CHANNEL_DIM, 2)
channels[0] = 1  # true input size (in case CHANNEL_DIM != 1)

Encoder = SlotEncoder(
    DCN(
        channels,
        [[nn.GroupNorm(1, channel), act()] for channel in channels[1:]],
        KERNEL_SIZE,
        stride=strides,
        padding=KERNEL_SIZE // 2,
        dim=2,
    ),
    channels[-1],
    CELL,
    GLOBAL,
)

upsamplings = [
    nn.Upsample(scale_factor=2, mode="nearest") if s == 2 else None
    for s in strides[::-1]
]

decoder_channels = channels[::-1]
decoder_channels[0] = CELL + GLOBAL  # the code enters as channels, not as a flat vector

Decoder = SlotDecoder(
    DCN(
        decoder_channels,
        # one entry short of the conv count on purpose: the last conv emits raw logits
        [[nn.GroupNorm(1, channel), act()] for channel in decoder_channels[1:-1]],
        KERNEL_SIZE,
        stride=1,
        padding=KERNEL_SIZE // 2,
        pre_modules=upsamplings,
        dim=2,
    ),
    GRID,
    CELL,
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
        x_target = augment(x[0])  # binary image, the target of the logits
        x = standardizex(x_target).to(device)  # standardized encoder input
        x_target = x_target.to(device)
        optimizer.zero_grad()
        x_pred, mean_pred, logvar_pred = model(x)
        cost, recon, kl = cost_fun(x_pred, mean_pred, logvar_pred, x_target)
        cost.backward()
        optimizer.step()
        train_cost[epoch] += recon.item()  # the rate is tracked on its own below
    train_cost[epoch] /= len(train_loader)  # avg per batch
    scheduler.step()

    # the validation pass decodes the latent mean, so it measures reconstruction alone
    # and is not comparable to the sampled, augmented training pass
    model.eval()
    with torch.no_grad():
        x_pred, mean_pred, logvar_pred = model(standardizex(X_val).to(device))
        cost, recon, kl = cost_fun(x_pred, mean_pred, logvar_pred, X_val.to(device))
    val_cost[epoch] = recon.item()
    val_rate[epoch] = kl.item()
    if cost.item() < best_cost:  # the exported model is the best one, not the last one
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

# --------------------------------- latent diagnostics --------------------------------
# a code drawn from the prior only decodes into a microstructure of its own if the
# aggregate posterior really is the prior, so the latent is measured rather than assumed
with torch.no_grad():
    x_pred, mean_pred, logvar_pred = model(standardizex(X_val).to(device))
    latent = model.encode(standardizex(X_train).to(device))
    mean_train, logvar_train = torch.chunk(latent, chunks=2, dim=1)
    x_gen = model.decode(torch.randn(X_val.shape[0], LATENT, device=device))
p_recon = torch.sigmoid(x_pred)
p_gen = torch.sigmoid(x_gen)
x_recon = p_recon >= THRESHOLD
x_gen = p_gen >= THRESHOLD

# the intersection over union is read on the fibers alone: they cover an eighth of the
# image, so counting the matching background would inflate it
X_val = X_val.to(device)
intersection = (x_recon * X_val).sum(dim=(1, 2, 3))
union = ((x_recon + X_val) >= 1).sum(dim=(1, 2, 3))
iou = (intersection / union).mean()
bce = recon_loss(x_pred, X_val) / X_val.shape[0]  # of the exported model, not the last

kl_dim = kl_div(mean_train, logvar_train)
active = kl_dim > 0.05  # a collapsed dimension decodes as pure prior noise, harmlessly
# the marginal a prior sample is drawn from is the posterior spread plus the mean spread
variance = torch.exp(logvar_train).mean(dim=0) + mean_train.var(dim=0, unbiased=False)
# a prior sample is a microstructure only if its components are round disks, so the
# latent is scored on the samples themselves, not just on its own summary statistics
counts, roundness = [], []
for mask in x_gen[:64, 0].cpu().numpy():
    labels, count = ndimage.label(mask)
    counts.append(count)
    for k in range(1, count + 1):
        pixels = np.argwhere(labels == k)
        if len(pixels) < 8:  # speckle, not a fiber
            roundness.append(0.0)
            continue
        radius = np.sqrt(((pixels - pixels.mean(axis=0)) ** 2).sum(axis=1).max())
        roundness.append(len(pixels) / (np.pi * radius**2))

print(f"iou {iou:.3f} bce {bce:.0f} kl {kl_dim.sum():.1f} au {active.sum()}")
print(
    f"prior samples {np.mean(counts):.1f} blobs, roundness {np.mean(roundness):.3f} "
    f"(the data has 7.0 blobs at 0.99)"
)
print(
    f"latent mean {mean_train.mean(dim=0).abs().max():.3f} max, "
    f"std {variance.sqrt().min():.3f} to {variance.sqrt().max():.3f}"
)

# ------------------------------------ export model -----------------------------------
model.standardizer = standardizex  # just for saving
torch.save(model, MODEL_DIR / f"fiber_vae_v2_{GRID}_{CELL}_{BETA}_{RESOLUTION}.pt2")

# ----------------------------------- postprocessing ----------------------------------
fig, ax = plt.subplots()
ax.plot(train_cost, "k")
ax.plot(val_cost, "r")
ax.plot(val_rate, "b")
ax.set_yscale("log")
ax.set_xlabel("epoch")
ax.set_ylabel("cost")
plt.show()

# the probabilities sit next to the thresholded masks because thresholding hides an
# unsaturated decoder, and grey mush is the failure mode that matters here
fig, ax = plt.subplots(4, 4, figsize=(8, 8), dpi=RESOLUTION // 2)
for i in range(4):
    ax[0, i].imshow(X_val.cpu()[i, 0], cmap="binary", vmin=0, vmax=1)
    ax[1, i].imshow(p_recon.cpu()[i, 0], cmap="binary", vmin=0, vmax=1)
    ax[2, i].imshow(x_recon.cpu()[i, 0], cmap="binary", vmin=0, vmax=1)
    ax[3, i].imshow(x_gen.cpu()[i, 0], cmap="binary", vmin=0, vmax=1)
for axis in ax.ravel():
    axis.set_aspect("equal")
    axis.axis("off")
    axis.set_rasterized(True)
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.show()

# the pooled active dimensions against the standard normal they are trained to follow
z = mean_train[:, active].flatten().cpu()
grid = torch.linspace(-4, 4, 200)
fig, ax = plt.subplots()
ax.hist(z.numpy(), bins=60, range=(-4, 4), density=True, color="k")
ax.plot(grid, torch.exp(-0.5 * grid**2) / np.sqrt(2 * np.pi), "r")
ax.set_xlabel("latent")
ax.set_ylabel("density")
plt.show()
