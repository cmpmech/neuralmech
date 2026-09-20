import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchinfo import summary
from tqdm import tqdm

from DL import sinusoidal_embedding
from NN import DCN, MLP, ConditionedResidualBlock, UNet

BASE_DIR = Path(__file__).parent
DATA_DIR = (BASE_DIR / "../../data").resolve()
MODEL_DIR = (BASE_DIR / "../../models").resolve()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True

# -------------------------------------- settings -------------------------------------
# hyperparameters
EPOCHS = 1500
LR = 2e-4
REGULARIZATION = 1e-2
BATCH_SIZE = 16
GRAD_CLIP = 1.0

# define loss
cost_fun = nn.MSELoss(reduction="mean")

# diffusion settings
T = 200

# model settings
RESOLUTION = 256
CHANNELS = [32, 64, 128, 256]
EMBEDDING = 128
SAMPLES = 50
DATA_SAMPLES = 500  # the dataset holds more; the first 500 are the original training set


# ----------------------------------- noise schedule ----------------------------------
def cosine_schedule(timesteps, s=0.008):
    steps = torch.linspace(0, 1, timesteps + 1, device=device)
    alpha_bars = torch.cos((steps + s) / (1 + s) * torch.pi / 2) ** 2
    alpha_bars = alpha_bars / alpha_bars[0]
    betas = 1 - alpha_bars[1:] / alpha_bars[:-1]
    return betas.clamp(1e-4, 0.999)


betas = cosine_schedule(T)
alphas = 1 - betas
alpha_bars = torch.cumprod(alphas, dim=0)
alpha_bars_prev = torch.cat([torch.ones(1, device=device), alpha_bars[:-1]])
posterior_variance = betas * (1 - alpha_bars_prev) / (1 - alpha_bars)


def noise(x0, t, eps=None):
    if eps is None:
        eps = torch.randn_like(x0)
    a_bar = alpha_bars[t].view(-1, 1, 1, 1)
    return torch.sqrt(a_bar) * x0 + torch.sqrt(1 - a_bar) * eps, eps


# ------------------------------------ prepare data -----------------------------------
# binary images scaled to [-1, 1]
data = torch.from_numpy(np.load(DATA_DIR / f"fibers_{RESOLUTION}.npy")[:DATA_SAMPLES])
data = 2 * data.to(torch.float32).unsqueeze(1) - 1

dataset = TensorDataset(data)
split = torch.Generator().manual_seed(0)
train_data, val_data = random_split(dataset, [0.9, 0.1], generator=split)
train_loader = DataLoader(
    train_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True
)

X_val = train_data.dataset.tensors[0][val_data.indices]


# data augmentation
def transform(x, k):
    return torch.rot90(x.flip(-1) if k >= 4 else x, k % 4, [-2, -1])


def augment(x):
    return torch.cat([transform(part, k) for k, part in enumerate(x.chunk(8))])


# --------------------------- instantiate model & optimizer ---------------------------
def block(channels_in, channels_out):
    return ConditionedResidualBlock(
        DCN(
            [channels_in, channels_out],
            kernel_size=3,
            stride=1,
            padding=1,
            pre_modules=[[nn.GroupNorm(min(8, channels_in), channels_in), nn.SiLU()]],
        ),
        DCN(
            [channels_out, channels_out],
            kernel_size=3,
            stride=1,
            padding=1,
            pre_modules=[[nn.GroupNorm(min(8, channels_out), channels_out), nn.SiLU()]],
        ),
        MLP([EMBEDDING, channels_out], pre_modules=[nn.SiLU()]),
        None
        if channels_in == channels_out
        else nn.Conv2d(channels_in, channels_out, 1),
    )


levels = [1] + CHANNELS
unet = UNet(
    downs=[block(levels[i], levels[i + 1]) for i in range(len(CHANNELS))],
    ups=[
        block(2 * levels[i + 1], levels[max(i, 1)])
        for i in reversed(range(len(CHANNELS)))
    ],
    bottleneck=block(CHANNELS[-1], CHANNELS[-1]),
    downsamplers=[nn.Conv2d(c, c, 4, 2, 1) for c in CHANNELS],
    upsamplers=[nn.ConvTranspose2d(c, c, 4, 2, 1) for c in reversed(CHANNELS)],
)
model = nn.ModuleDict(
    {
        "time": MLP([CHANNELS[0], EMBEDDING, EMBEDDING], [nn.SiLU(), None]),
        "unet": unet,
        "head": DCN(
            [CHANNELS[0], 1],
            kernel_size=1,
            stride=1,
            padding=0,
            pre_modules=[[nn.GroupNorm(8, CHANNELS[0]), nn.SiLU()]],
        ),
    }
).to(device)


def denoise(x, t):
    embedding = model["time"](sinusoidal_embedding(t, CHANNELS[0]))
    return model["head"](model["unet"](x, embedding))


summary(
    model["unet"],
    input_data=(torch.zeros(1, 1, RESOLUTION, RESOLUTION), torch.zeros(1, EMBEDDING)),
    device=device,
    depth=2,
)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=REGULARIZATION)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=EPOCHS, eta_min=LR * 1e-2
)

# -------------------------------------- training -------------------------------------
print_every = 50
train_cost = [0] * EPOCHS
val_cost = [0] * EPOCHS
best_cost = float("inf")
best_state = None
val_seed = torch.Generator(device=device).manual_seed(0)
tic = time.time()
pbar = tqdm(range(EPOCHS), desc="Training: ", ncols=90)
for epoch in pbar:
    model.train()
    for x in train_loader:
        x0 = augment(x[0]).to(device)
        t = torch.randint(0, T, (x0.shape[0],), device=device)
        xt, eps = noise(x0, t)
        optimizer.zero_grad()
        cost = cost_fun(denoise(xt, t), eps)
        cost.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        train_cost[epoch] += cost.item()
    train_cost[epoch] /= len(train_loader)  # avg per batch
    scheduler.step()

    model.eval()
    with torch.no_grad():
        x0 = X_val.to(device)
        t = torch.randint(0, T, (x0.shape[0],), device=device, generator=val_seed)
        eps = torch.randn(x0.shape, device=device, generator=val_seed)
        xt, eps = noise(x0, t, eps)
        val_cost[epoch] = cost_fun(denoise(xt, t), eps).item()
    if val_cost[epoch] < best_cost:
        best_cost = val_cost[epoch]
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if epoch % print_every == 0:
        pbar.set_postfix(
            {"train": f"{train_cost[epoch]:.2e}", "val": f"{val_cost[epoch]:.2e}"}
        )
toc = time.time()
print(f"elapsed time {toc - tic:.2f} s")

model.load_state_dict(best_state)
model.eval()


# -------------------------------------- sampling -------------------------------------
@torch.no_grad()
def sample(n):
    x = torch.randn(n, 1, RESOLUTION, RESOLUTION, device=device)
    for step in reversed(range(T)):
        t = torch.full((n,), step, device=device)
        eps_pred = denoise(x, t)
        x0_pred = (x - torch.sqrt(1 - alpha_bars[step]) * eps_pred) / torch.sqrt(
            alpha_bars[step]
        )
        x0_pred = x0_pred.clamp(-1, 1)
        x = (
            torch.sqrt(alpha_bars_prev[step]) * betas[step] * x0_pred
            + torch.sqrt(alphas[step]) * (1 - alpha_bars_prev[step]) * x
        ) / (1 - alpha_bars[step])
        if step > 0:
            x = x + torch.sqrt(posterior_variance[step]) * torch.randn_like(x)
    return ((x.clamp(-1, 1) + 1) / 2).cpu()


tic = time.time()
x_sample = sample(SAMPLES)
print(f"elapsed time {time.time() - tic:.2f} s")


X_val = (X_val + 1) / 2

# --------------------------------------- export --------------------------------------
model.T = T
model.betas = betas
model.alphas = alphas
model.alpha_bars = alpha_bars
model.posterior_variance = posterior_variance
torch.save(model, MODEL_DIR / f"fiber_diffusion_{T}_{RESOLUTION}.pt2")

# ----------------------------------- postprocessing ----------------------------------
fig, ax = plt.subplots()
ax.plot(train_cost, "k")
ax.plot(val_cost, "r")
ax.set_yscale("log")
ax.set_xlabel("epoch")
ax.set_ylabel("cost")
plt.show()

fig, ax = plt.subplots(2, 4, figsize=(8, 4), dpi=RESOLUTION // 2)
for i in range(4):
    ax[0, i].imshow(X_val[i, 0], cmap="binary", vmin=0, vmax=1)
    ax[1, i].imshow(x_sample[i, 0], cmap="binary", vmin=0, vmax=1)
for axis in ax.ravel():
    axis.set_aspect("equal")
    axis.axis("off")
    axis.set_rasterized(True)
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.show()
