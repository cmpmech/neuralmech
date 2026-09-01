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
from NN import DCN, VAE, SlotDecoder, SlotEncoder, TwoStagePrior

BASE_DIR = Path(__file__).parent
DATA_DIR = (BASE_DIR / "../../data").resolve()
MODEL_DIR = (BASE_DIR / "../../models").resolve()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True

# -------------------------------------- settings -------------------------------------
# hyperparameters
EPOCHS = 3000
LR = 1e-3
REGULARIZATION = 1e-2
BATCH_SIZE = 32
BETA = 1.0  # kl weight; 1 is the plain evidence lower bound
PRIOR_STEPS = 20000
PRIOR_REGULARIZATION = 1e-4
PRIOR_LATENT = 64  # code size of the second stage, which is the prior

# define loss
# both terms are summed per image, so BETA weighs nats against nats and BETA = 1 is the
# evidence lower bound. with a mean reduction the weight would instead have to absorb
# the ratio of pixel count to latent size.
# BETA stays at 1 because raising it is not how this latent is fixed. a larger BETA does
# pull the aggregate posterior towards a standard normal, but only by spending
# reconstruction: on a 128 grid the intersection over union falls from 0.97 to 0.70 as
# BETA goes from 1 to 32. that trades along the rate-distortion curve instead of moving
# it. the prior fitted below moves it
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
# same convolutional autoencoder as fiber_vae_train.py, but on a coarser grid of fatter
# slots. that is what the simpler prior below needs: it models the whole code at once
# rather than one slot at a time, and 132 dimensions are within its reach where the 260
# of the 8 x 8 grid are not. the cost is reconstruction, since a 4 x 4 grid resolves
# fiber positions less finely. DEPTH follows from the slot grid up to the image
RESOLUTION = 256
GRID = 4
CELL = 8
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
# transforms are free data
def transform(x, k):
    return torch.rot90(x.flip(-1) if k >= 4 else x, k % 4, [-2, -1])


# one transform per batch slice rather than one per batch, so a single gradient spans
# the whole symmetry group instead of being one correlated transform
def augment(x):
    return torch.cat([transform(part, k) for k, part in enumerate(x.chunk(8))])


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

# ----------------------------------- learned prior -----------------------------------
# a standard normal treats every slot as independent, and no product of independent
# slots can say that two neighbouring places are never both filled. that is the whole
# content of the fibers not overlapping, so a code drawn from a standard normal decodes
# into merged worms no matter how the rate is throttled. the fix is not to force the
# codes onto a fixed prior but to fit a prior to the codes.
# here that prior is simply a second, much smaller variational autoencoder over the
# codes, the two-stage construction of dai and wipf. it holds no attention and reads no
# structure in the code, which is what keeps it plain; the whole 132 dimensional
# distribution has to be learned in one piece, which is why the slot grid above is
# coarse. the autoregressive prior of fiber_vae_train.py buys about 0.06 more roundness
# on the samples and 0.02 more reconstruction, at the price of being far less simple
# the symmetries are free codes just as they are free images
with torch.no_grad():
    encoded = [
        model.encode(standardizex(transform(X_train, k)).to(device)) for k in range(8)
    ]
    codes = torch.cat([torch.chunk(e, chunks=2, dim=1)[0] for e in encoded])

prior = TwoStagePrior(codes, PRIOR_LATENT).to(device)
prior_optimizer = torch.optim.AdamW(
    prior.parameters(), lr=LR, weight_decay=PRIOR_REGULARIZATION
)
prior_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    prior_optimizer, T_max=PRIOR_STEPS, eta_min=LR * 1e-2
)

tic = time.time()
pbar = tqdm(range(PRIOR_STEPS), desc="Prior:    ", ncols=90)
for step in pbar:
    batch = codes[torch.randint(0, codes.shape[0], (BATCH_SIZE * 4,), device=device)]
    cost = -prior.log_prob(batch).mean()  # the evidence lower bound of the second stage
    prior_optimizer.zero_grad()
    cost.backward()
    prior_optimizer.step()
    prior_scheduler.step()
    if step % print_every == 0:
        pbar.set_postfix({"nll": f"{cost.item():.2e}"})
prior.eval()
print(f"elapsed time {time.time() - tic:.2f} s")

# --------------------------------- latent diagnostics --------------------------------
# a code drawn from the prior only decodes into a microstructure of its own if the prior
# really is the distribution of the codes, so both priors are measured rather than
# assumed, side by side against the data they are supposed to reproduce
with torch.no_grad():
    x_pred, mean_pred, logvar_pred = model(standardizex(X_val).to(device))
    mean_train, logvar_train = torch.chunk(
        model.encode(standardizex(X_train).to(device)), chunks=2, dim=1
    )
    x_normal = model.decode(torch.randn(X_val.shape[0], LATENT, device=device))
    x_learned = model.decode(prior.sample(X_val.shape[0]))
p_recon = torch.sigmoid(x_pred)
x_recon = p_recon >= THRESHOLD

# the intersection over union is read on the fibers alone: they cover an eighth of the
# image, so counting the matching background would inflate it
X_val = X_val.to(device)
intersection = (x_recon * X_val).sum(dim=(1, 2, 3))
union = ((x_recon + X_val) >= 1).sum(dim=(1, 2, 3))
iou = (intersection / union).mean()
bce = recon_loss(x_pred, X_val) / X_val.shape[0]  # of the exported model, not the last

kl_dim = kl_div(mean_train, logvar_train)
active = kl_dim > 0.05  # a collapsed dimension decodes as pure prior noise, harmlessly


# a sample is a microstructure only if its components are round disks covering the right
# area, so a prior is scored on the samples themselves, not on summary statistics of the
# codes: a latent can match the standard normal in every marginal and still be empty
# between the codes it was trained on
def sample_stats(masks):
    counts, roundness = [], []
    for mask in masks:
        labels, count = ndimage.label(mask)
        counts.append(count)
        for k in range(1, count + 1):
            pixels = np.argwhere(labels == k)
            if len(pixels) < 8:  # speckle, not a fiber
                roundness.append(0.0)
                continue
            radius = np.sqrt(((pixels - pixels.mean(axis=0)) ** 2).sum(axis=1).max())
            roundness.append(len(pixels) / (np.pi * radius**2))
    return np.mean(counts), np.mean(roundness), masks.mean()


print(f"iou {iou:.3f} bce {bce:.0f} kl {kl_dim.sum():.1f} au {active.sum()}")
for name, masks in [
    ("data", X_val[:, 0].cpu().numpy() >= THRESHOLD),
    ("reconstruction", x_recon[:, 0].cpu().numpy()),
    ("normal prior", (torch.sigmoid(x_normal) >= THRESHOLD)[:, 0].cpu().numpy()),
    ("learned prior", (torch.sigmoid(x_learned) >= THRESHOLD)[:, 0].cpu().numpy()),
]:
    blobs, roundness, area = sample_stats(masks)
    print(f"{name:<15} {blobs:.2f} fibers, roundness {roundness:.3f}, area {area:.3f}")

# --------------------------------------- export --------------------------------------
model.standardizer = standardizex  # just for saving
model.prior = prior  # so a sample is one call on the loaded model
torch.save(model, MODEL_DIR / f"fiber_vae_v2_{LATENT}_{BETA}_{RESOLUTION}.pt2")

# ----------------------------------- postprocessing ----------------------------------
fig, ax = plt.subplots()
ax.plot(train_cost, "k")
ax.plot(val_cost, "r")
ax.plot(val_rate, "b")
ax.set_yscale("log")
ax.set_xlabel("epoch")
ax.set_ylabel("cost")
plt.show()

# the two priors sit under the reconstruction, because the failure they cause is
# invisible in the reconstruction and only shows up in what they decode into
fig, ax = plt.subplots(4, 4, figsize=(8, 8), dpi=RESOLUTION // 2)
for i in range(4):
    ax[0, i].imshow(X_val.cpu()[i, 0], cmap="binary", vmin=0, vmax=1)
    ax[1, i].imshow(x_recon.cpu()[i, 0], cmap="binary", vmin=0, vmax=1)
    ax[2, i].imshow(torch.sigmoid(x_normal).cpu()[i, 0], cmap="binary", vmin=0, vmax=1)
    ax[3, i].imshow(torch.sigmoid(x_learned).cpu()[i, 0], cmap="binary", vmin=0, vmax=1)
for axis in ax.ravel():
    axis.set_aspect("equal")
    axis.axis("off")
    axis.set_rasterized(True)
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.show()

# the pooled active dimensions against the standard normal the rate term pulls them
# towards. they follow it closely, which is exactly why the marginals are no evidence:
# the codes are still far too sparse in 260 dimensions for a draw to land among them
z = mean_train[:, active].flatten().cpu()
grid = torch.linspace(-4, 4, 200)
fig, ax = plt.subplots()
ax.hist(z.numpy(), bins=60, range=(-4, 4), density=True, color="k")
ax.plot(grid, torch.exp(-0.5 * grid**2) / np.sqrt(2 * np.pi), "r")
ax.set_xlabel("latent")
ax.set_ylabel("density")
plt.show()
