import argparse
from datetime import datetime
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from NN import DCN, MLP, UNet

BASE_DIR = Path(__file__).parent
DATA_DIR = (BASE_DIR / "../../data").resolve()
RESULTS_DIR = (BASE_DIR / "../../results/shape_DDPM").resolve()
MODELS_DIR = (BASE_DIR / "../../models").resolve()
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
parser.add_argument("--animate", action="store_true")
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
# unconditional DDPM: the baseline the conditional variants are measured against. it
# learns the shape distribution as a whole, with no way to ask for one class
# hyperparameters
epochs = 1500
batch_size = 64
lr = 2e-4
weight_decay = 1e-2
grad_clip = 1.0

# define loss
cost_fun = nn.MSELoss()

# diffusion settings
T = 200
noise_schedule = "cosine"

# model settings
domain_size = 128
base_channels = 64
time_emb_dim = 256
labels = ["circle", "ellipse", "square", "triangle", "cross", "star"]

# sampling
n_samples = 16
sample_every = 50
sample_seed = 7
print_every = 10
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")


# ----------------------------------- noise schedule ----------------------------------
def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alpha_bars = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alpha_bars = alpha_bars / alpha_bars[0]
    betas = 1 - (alpha_bars[1:] / alpha_bars[:-1])
    return torch.clamp(betas, 1e-4, 0.9999)


betas = cosine_beta_schedule(T).to(device)
alphas = 1.0 - betas
alpha_bars = torch.cumprod(alphas, dim=0)
alpha_bars_prev = torch.cat([torch.ones(1, device=device), alpha_bars[:-1]])
posterior_variance = betas * (1.0 - alpha_bars_prev) / (1.0 - alpha_bars)


def sinusoidal_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(
        -torch.arange(half, device=t.device) * (np.log(10000) / (half - 1))
    )
    angles = t[:, None].float() * freqs[None]
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


def q_sample(x_start, t, noise):
    a = alpha_bars[t].view(-1, 1, 1, 1)
    return torch.sqrt(a) * x_start + torch.sqrt(1 - a) * noise


# ----------------------------------- prepare data ------------------------------------
# all six classes pooled into one unlabelled set -- the model never sees a class index
X = [np.load(DATA_DIR / f"shapes_{label}_{domain_size}.npy") for label in labels]
X = np.concatenate(X, axis=0).astype(np.float32)
X = torch.from_numpy(X).unsqueeze(1) * 2.0 - 1.0

dataset = TensorDataset(X)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)


# --------------------------------------- model ---------------------------------------
# both convolution stacks and the embedding projection come from the library. DCN's
# pre_modules slot carries the GroupNorm and SiLU that precede each convolution, the
# pre-activation ordering its docstring describes, and MLP's carries the SiLU before the
# embedding's linear map. only forward is written here, because the timestep embedding
# enters between the two stacks and no single-argument module can express that
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, emb_dim):
        super().__init__()
        self.block1 = DCN(
            [in_ch, out_ch],
            kernel_size=3,
            stride=1,
            padding=1,
            dim=2,
            pre_modules=[[nn.GroupNorm(min(8, in_ch), in_ch), nn.SiLU()]],
        )
        self.block2 = DCN(
            [out_ch, out_ch],
            kernel_size=3,
            stride=1,
            padding=1,
            dim=2,
            pre_modules=[[nn.GroupNorm(min(8, out_ch), out_ch), nn.SiLU()]],
        )
        self.emb_proj = MLP([emb_dim, out_ch], pre_modules=[nn.SiLU()])
        self.res_conv = (
            nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x, emb):
        h = self.block1(x)
        h = h + self.emb_proj(emb)[:, :, None, None]
        h = self.block2(h)
        return h + self.res_conv(x)


# NN.UNet holds the encoder/bottleneck/decoder lists and the skip algebra already; its
# forward takes only x and stores each skip after downsampling. subclassing keeps the
# container and replaces the traversal: the timestep reaches every block, and skips are
# taken before the downsample so they keep full resolution
class TimeConditionedUNet(UNet):
    def __init__(self, base_channels=64, emb_dim=256):
        c = base_channels
        super().__init__(
            downs=[
                ResBlock(1, c, emb_dim),
                ResBlock(c, c * 2, emb_dim),
                ResBlock(c * 2, c * 4, emb_dim),
                ResBlock(c * 4, c * 8, emb_dim),
            ],
            ups=[
                ResBlock(c * 16, c * 4, emb_dim),
                ResBlock(c * 8, c * 2, emb_dim),
                ResBlock(c * 4, c, emb_dim),
                ResBlock(c * 2, c, emb_dim),
            ],
            bottleneck=ResBlock(c * 8, c * 8, emb_dim),
        )
        self.base_channels = base_channels
        self.time_mlp = MLP([base_channels, emb_dim, emb_dim], [nn.SiLU(), None])
        self.downsamplers = nn.ModuleList(
            [nn.Conv2d(ch, ch, 4, 2, 1) for ch in (c, c * 2, c * 4, c * 8)]
        )
        self.upsamplers = nn.ModuleList(
            [nn.ConvTranspose2d(ch, ch, 4, 2, 1) for ch in (c * 8, c * 4, c * 2, c)]
        )
        self.out = DCN(
            [c, 1],
            kernel_size=1,
            stride=1,
            padding=0,
            dim=2,
            pre_modules=[[nn.GroupNorm(min(8, c), c), nn.SiLU()]],
        )

    def forward(self, x, t):
        emb = self.time_mlp(sinusoidal_embedding(t, self.base_channels))

        skips = []
        for block, downsample in zip(self.downs, self.downsamplers):
            x = block(x, emb)
            skips.append(x)  # before the downsample, so the skip keeps its resolution
            x = downsample(x)

        x = self.bottleneck(x, emb)

        for block, upsample, skip in zip(self.ups, self.upsamplers, reversed(skips)):
            x = block(torch.cat([upsample(x), skip], dim=1), emb)

        return self.out(x)


# --------------------------- instantiate model & optimizer ---------------------------
model = TimeConditionedUNet(base_channels, time_emb_dim).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=epochs, eta_min=lr * 1e-2
)


# -------------------------------------- sampling -------------------------------------
@torch.no_grad()
def sample(n=n_samples):
    # the seed is fixed, so successive grids differ because the model improved and not
    # because the starting noise changed
    torch.manual_seed(sample_seed)
    model.eval()
    x = torch.randn(n, 1, domain_size, domain_size, device=device)
    for step in reversed(range(T)):
        t = torch.full((n,), step, device=device, dtype=torch.long)
        eps_pred = model(x, t)
        x = x - (1 - alphas[step]) / torch.sqrt(1 - alpha_bars[step]) * eps_pred
        x = x / torch.sqrt(alphas[step])
        if step > 0:
            x = x + torch.sqrt(posterior_variance[step]) * torch.randn_like(x)
    model.train()
    return ((x.clamp(-1, 1) + 1.0) / 2.0).cpu()


def save_samples(epoch):
    grid = sample()
    fig, ax = plt.subplots(4, 4, figsize=(4, 4), dpi=domain_size)
    for axis, image in zip(ax.flat, grid):
        axis.imshow(image[0].T, cmap="binary", origin="lower", vmin=0, vmax=1)
        axis.set_aspect("equal")
        axis.axis("off")
        axis.set_rasterized(True)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(RESULTS_DIR / f"samples_{run_id}_{epoch:04d}.png")
    plt.close()


# -------------------------------------- training -------------------------------------
train_cost = [0] * epochs
best_train_cost = float("inf")

start_time = time.perf_counter()
pbar = tqdm(range(epochs), desc="Training: ", ncols=90)
for epoch in pbar:
    model.train()
    for x in train_loader:
        x = x[0].to(device)
        t = torch.randint(0, T, (x.shape[0],), device=device)
        noise = torch.randn_like(x)

        optimizer.zero_grad()
        cost = cost_fun(model(q_sample(x, t, noise), t), noise)
        cost.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        train_cost[epoch] += cost.item()
    train_cost[epoch] /= len(train_loader)
    scheduler.step()
    best_train_cost = min(best_train_cost, train_cost[epoch])

    if epoch % sample_every == 0 or epoch == epochs - 1:
        save_samples(epoch)

    if epoch % print_every == 0:
        pbar.set_postfix({"train": f"{train_cost[epoch]:.2e}"})
training_time = time.perf_counter() - start_time

# ------------------------------------ export model -----------------------------------
torch.save(
    {
        "model": model.state_dict(),
        "train_cost": train_cost,
        "T": T,
        "domain_size": domain_size,
        "base_channels": base_channels,
        "time_emb_dim": time_emb_dim,
        "labels": labels,
        "run_id": run_id,
    },
    MODELS_DIR / f"shape_DDPM_{T}_{domain_size}.pt2",
)

# ----------------------------------- postprocessing ----------------------------------
fig, ax = plt.subplots()
ax.plot(train_cost, "k")
ax.set_yscale("log")
ax.set_xlabel("epoch")
ax.set_ylabel("loss")
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.savefig(RESULTS_DIR / f"loss_{run_id}.png")
if not (args.book or args.animate):
    plt.show()
plt.close()

np.savetxt(
    RESULTS_DIR / f"history_{run_id}.csv",
    np.column_stack([np.arange(1, epochs + 1), np.asarray(train_cost)]),
    delimiter=",",
    header="epoch,train_loss",
    comments="",
)

summary_path = RESULTS_DIR / f"summary_{run_id}.txt"
with open(summary_path, "w", encoding="utf-8") as f:
    f.write("shape unconditional DDPM training summary\n")
    f.write(f"run_id: {run_id}\n")
    f.write(f"device: {device}\n")
    f.write("architecture: NN.UNet subclass, ResBlock on DCN and MLP\n")
    f.write("conditioning: none, unconditional baseline\n")
    f.write(f"labels: {','.join(labels)}\n")
    f.write(f"domain_size: {domain_size}\n")
    f.write(f"samples: {len(X)}\n")
    f.write(f"epochs: {epochs}\n")
    f.write(f"batch_size: {batch_size}\n")
    f.write(f"learning_rate: {lr}\n")
    f.write(f"weight_decay: {weight_decay}\n")
    f.write(f"grad_clip: {grad_clip}\n")
    f.write(f"timesteps: {T}\n")
    f.write(f"noise_schedule: {noise_schedule}\n")
    f.write(f"base_channels: {base_channels}\n")
    f.write(f"time_emb_dim: {time_emb_dim}\n")
    f.write(f"parameters: {sum(p.numel() for p in model.parameters())}\n")
    f.write("reconstruction_loss: MSELoss on predicted noise\n")
    f.write(f"training_time_seconds: {training_time:.2f}\n")
    f.write(f"final_train_loss: {train_cost[-1]:.8e}\n")
    f.write(f"best_train_loss: {best_train_cost:.8e}\n")
    f.write(f"sample_seed: {sample_seed}\n")
print(f"saved {summary_path}")
