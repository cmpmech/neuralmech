import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
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
BATCH_SIZE = 64
GRAD_CLIP = 1.0

# define loss
cost_fun = nn.MSELoss(reduction="mean")

# diffusion settings
T = 200

# model settings
RESOLUTION = 128
CHANNELS = [64, 128, 256, 512]
EMBEDDING = 256
SAMPLES = 16
labels = ["circle", "ellipse", "square", "triangle", "cross", "star"]


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


def noise(x0, t):
    eps = torch.randn_like(x0)
    a_bar = alpha_bars[t].view(-1, 1, 1, 1)
    return torch.sqrt(a_bar) * x0 + torch.sqrt(1 - a_bar) * eps, eps


# ------------------------------------ prepare data -----------------------------------
# all classes pooled into one unlabelled set, scaled to [-1, 1]
X = [np.load(DATA_DIR / f"shapes_{label}_{RESOLUTION}.npy") for label in labels]
X = torch.from_numpy(np.concatenate(X, axis=0)).to(torch.float32).unsqueeze(1)
X = 2 * X - 1

dataset = TensorDataset(X)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)


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
print_every = 10
train_cost = [0] * EPOCHS
tic = time.time()
pbar = tqdm(range(EPOCHS), desc="Training: ", ncols=90)
for epoch in pbar:
    model.train()
    for x in train_loader:
        x0 = x[0].to(device)
        t = torch.randint(0, T, (x0.shape[0],), device=device)
        xt, eps = noise(x0, t)
        optimizer.zero_grad()
        cost = cost_fun(denoise(xt, t), eps)
        cost.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        train_cost[epoch] += cost.item()
    train_cost[epoch] /= len(train_loader)
    scheduler.step()

    if epoch % print_every == 0:
        pbar.set_postfix({"train": f"{train_cost[epoch]:.2e}"})
toc = time.time()
print(f"elapsed time {toc - tic:.2f} s")


# -------------------------------------- sampling -------------------------------------
@torch.no_grad()
def sample(n):
    model.eval()
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


gen_shapes = sample(SAMPLES)

# --------------------------------------- export --------------------------------------
model.T = T
model.betas = betas
model.alphas = alphas
model.alpha_bars = alpha_bars
model.posterior_variance = posterior_variance
torch.save(model, MODEL_DIR / f"shape_diffusion_{T}_{RESOLUTION}.pt2")

# ----------------------------------- postprocessing ----------------------------------
fig, ax = plt.subplots()
ax.plot(train_cost, "k")
ax.set_yscale("log")
ax.set_xlabel("epoch")
ax.set_ylabel("cost")
plt.show()

fig, ax = plt.subplots(4, 4, figsize=(4, 4), dpi=RESOLUTION)
for axis, image in zip(ax.flat, gen_shapes):
    axis.imshow(image[0].T, cmap="binary", origin="lower", vmin=0, vmax=1)
    axis.set_aspect("equal")
    axis.axis("off")
    axis.set_rasterized(True)
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.show()
