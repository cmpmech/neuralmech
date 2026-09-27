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

from DL import init_weights
from NN import DCN, MLP, UNet

BASE_DIR = Path(__file__).parent
DATA_DIR = (BASE_DIR / "../../data").resolve()
RESULTS_DIR = (BASE_DIR / "../../results/shape_cddpm_v2").resolve()
MODELS_DIR = (BASE_DIR / "../../models").resolve()
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
parser.add_argument("--sweep-only", action="store_true")
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
# merges shape_cddpm_v2_train.py and shape_cddpm_v2_guidance.py. guidance is a
# sampling-time choice, so training and the sweep belong in one process: the sweep then
# runs against the model already in memory and no architecture can drift between the
# two. --sweep-only skips training and reloads the checkpoint, which is what made the
# separate sweep script worth having -- minutes instead of a 45 minute retrain
# hyperparameters
epochs = 1000
batch_size = 64
lr = 2e-4
weight_decay = 1e-2

# define loss
cost_fun = nn.MSELoss()

# diffusion settings
T = 200
noise_schedule = "cosine"

# conditioning. the label embedding is added to the time embedding and injected into
# every ResBlock, so neither the timestep nor the class has to survive four
# downsamplings. label_dropout trains the unconditional branch that guidance
# extrapolates from; the embedding is held out of weight decay because decaying it
# shrinks exactly the signal the conditioning depends on
label_dropout = 0.2
label_emb_std = 0.5
label_emb_weight_decay = 0.0
sensitivity_step = 190  # near pure noise, where the image cannot reveal the class

# model settings
domain_size = 128
base_channels = 64
time_emb_dim = 256
labels = ["circle", "ellipse", "square", "triangle", "cross", "star"]
classes = len(labels)

# guidance sweep. the training grids are drawn at sample_guidance; the sweep afterwards
# covers the range. 1.0 is plain conditional sampling and measured 46/48 correct by eye,
# so it is the default rather than the 3.0 the earlier scripts hard-coded
guidances = (0.0, 1.0, 1.5, 2.0, 3.0, 5.0)
sample_guidance = 1.0
samples_per_class = 8
# validity thresholds: the thinnest real class covers 0.124 of the frame, so anything
# under 0.05 is blank, and a coherent region keeps most of its ink under a 3x3 erosion
min_ink = 0.05
min_kept = 0.5
sample_seed = 7
sample_epochs = (0, 25, 50, 100, 250, 500, 750)
print_every = 10
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
model_path = MODELS_DIR / f"shape_cddpm_v2_{T}_{domain_size}.pt2"

# scoring classifier. a small supervised CNN trained on the same six classes turns the
# "46 of 48 look right" eyeball count into a recorded number
classifier_epochs = 30
classifier_lr = 1e-3
classifier_channels = [1, 16, 32, 64]

# ----------------------------------- prepare data ------------------------------------
# one file per shape with no labels, so the class index comes from the load order
X = []
targets = []
for idx, label in enumerate(labels):
    shapes = np.load(DATA_DIR / f"shapes_{label}_{domain_size}.npy")
    X.append(shapes)
    targets.append(np.full(len(shapes), idx))

X = np.concatenate(X, axis=0).astype(np.float32)
X = torch.from_numpy(X).unsqueeze(1) * 2.0 - 1.0  # the model predicts noise in [-1, 1]
targets = torch.from_numpy(np.concatenate(targets)).to(torch.long)

dataset = TensorDataset(X, targets)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)


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


# --------------------------------------- model ---------------------------------------
# the two convolution stacks and the embedding projection are built by the library.
# DCN's pre_modules slot takes the GroupNorm and SiLU that precede each convolution,
# which is the pre-activation ordering its docstring describes; MLP's does the same for
# the SiLU before the embedding's linear map. only the three lines of forward are local,
# because the embedding has to enter between the two stacks and no single-argument
# module can express that
class ConditionalResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, emb_dim):
        super().__init__()
        self.block1 = DCN(
            [in_ch, out_ch],
            kernel_size=3,
            stride=1,
            padding=1,
            dim=2,
            pre_modules=[[nn.GroupNorm(min(8, in_ch), in_ch), nn.SiLU()]],
            bias=True,
        )
        self.block2 = DCN(
            [out_ch, out_ch],
            kernel_size=3,
            stride=1,
            padding=1,
            dim=2,
            pre_modules=[[nn.GroupNorm(min(8, out_ch), out_ch), nn.SiLU()]],
            bias=True,
        )
        self.emb_proj = MLP([emb_dim, out_ch], pre_modules=[nn.SiLU()])
        self.res_conv = (
            nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x, emb):
        h = self.block1(x)
        h = h + self.emb_proj(emb)[:, :, None, None]  # per-channel conditional bias
        h = self.block2(h)
        return h + self.res_conv(x)


# NN.UNet already holds the encoder/bottleneck/decoder lists and the skip algebra, but
# its forward takes only x and appends each skip after downsampling. subclassing keeps
# the container and replaces just the traversal: the embedding is threaded into every
# block, and skips are taken before the downsample so they carry full resolution
class ConditionalUNet(UNet):
    def __init__(self, base_channels=64, emb_dim=256, classes=6, label_std=0.5):
        c = base_channels
        super().__init__(
            downs=[
                ConditionalResBlock(1, c, emb_dim),
                ConditionalResBlock(c, c * 2, emb_dim),
                ConditionalResBlock(c * 2, c * 4, emb_dim),
                ConditionalResBlock(c * 4, c * 8, emb_dim),
            ],
            ups=[
                ConditionalResBlock(c * 16, c * 4, emb_dim),
                ConditionalResBlock(c * 8, c * 2, emb_dim),
                ConditionalResBlock(c * 4, c, emb_dim),
                ConditionalResBlock(c * 2, c, emb_dim),
            ],
            bottleneck=ConditionalResBlock(c * 8, c * 8, emb_dim),
        )
        self.base_channels = base_channels
        self.time_mlp = MLP([base_channels, emb_dim, emb_dim], [nn.SiLU(), None])
        # one row per class plus a null row for the dropped-label case
        self.label_emb = nn.Embedding(classes + 1, emb_dim)
        nn.init.normal_(self.label_emb.weight, std=label_std)

        self.downsamplers = nn.ModuleList(
            [nn.Conv2d(ch, ch, 4, 2, 1) for ch in (c, c * 2, c * 4, c * 8)]
        )
        self.upsamplers = nn.ModuleList(
            [
                nn.ConvTranspose2d(ch, ch, 4, 2, 1)
                for ch in (c * 8, c * 4, c * 2, c)
            ]
        )
        self.out = DCN(
            [c, 1],
            kernel_size=1,
            stride=1,
            padding=0,
            dim=2,
            pre_modules=[[nn.GroupNorm(min(8, c), c), nn.SiLU()]],
            bias=True,
        )

    def forward(self, x, t, y):
        emb = self.time_mlp(sinusoidal_embedding(t, self.base_channels))
        emb = emb + self.label_emb(y)  # class rides the timestep's path to every block

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
model = ConditionalUNet(base_channels, time_emb_dim, classes, label_emb_std).to(device)

# the label embedding is split into its own parameter group so weight decay never
# shrinks it. with it decayed, the embedding norm collapsed and the conditional and
# unconditional predictions became indistinguishable
decay_params = [p for n, p in model.named_parameters() if "label_emb" not in n]
optimizer = torch.optim.AdamW(
    [
        {"params": decay_params, "weight_decay": weight_decay},
        {
            "params": model.label_emb.parameters(),
            "weight_decay": label_emb_weight_decay,
        },
    ],
    lr=lr,
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=epochs, eta_min=lr * 1e-2
)


# ------------------------------------- classifier ------------------------------------
# scores generated samples so class adherence is a number rather than an eyeball count
classifier = nn.Sequential(
    DCN(
        classifier_channels,
        [[nn.BatchNorm2d(ch), nn.ReLU()] for ch in classifier_channels[1:]],
        kernel_size=3,
        stride=2,
        padding=1,
        dim=2,
    ),
    nn.AdaptiveAvgPool2d(1),
    nn.Flatten(),
    MLP([classifier_channels[-1], classes]),
).to(device)
init_weights(classifier, nn.ReLU())

classifier_optimizer = torch.optim.Adam(classifier.parameters(), lr=classifier_lr)
classifier_cost_fun = nn.CrossEntropyLoss()

classifier.train()
for epoch in tqdm(range(classifier_epochs), desc="Classifier: ", ncols=90):
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        classifier_optimizer.zero_grad()
        cost = classifier_cost_fun(classifier(x), y)
        cost.backward()
        classifier_optimizer.step()

classifier.eval()
with torch.no_grad():
    correct = 0
    for x, y in train_loader:
        prediction = classifier(x.to(device)).argmax(dim=1).cpu()
        correct += (prediction == y).sum().item()
    classifier_accuracy = correct / (len(train_loader) * batch_size)
print(f"classifier accuracy on real shapes {classifier_accuracy:.4f}", flush=True)


# -------------------------------------- sampling -------------------------------------
@torch.no_grad()
def sample_class(idx, n, guidance):
    # the seed is reset per class and per guidance, so a cell of the grid differs only
    # by the label and the guidance -- never by the noise it started from
    torch.manual_seed(sample_seed)
    y = torch.full((n,), idx, device=device, dtype=torch.long)
    y_null = torch.full((n,), classes, device=device, dtype=torch.long)
    x = torch.randn(n, 1, domain_size, domain_size, device=device)
    for step in reversed(range(T)):
        t = torch.full((n,), step, device=device, dtype=torch.long)
        eps_cond = model(x, t, y)
        if guidance == 1.0:  # collapses to plain conditional sampling, one pass
            eps_pred = eps_cond
        else:
            eps_uncond = model(x, t, y_null)
            eps_pred = eps_uncond + guidance * (eps_cond - eps_uncond)
        x = x - (1 - alphas[step]) / torch.sqrt(1 - alpha_bars[step]) * eps_pred
        x = x / torch.sqrt(alphas[step])
        if step > 0:
            x = x + torch.sqrt(posterior_variance[step]) * torch.randn_like(x)
    return ((x.clamp(-1, 1) + 1.0) / 2.0).cpu()


def sample_grid(guidance):
    model.eval()
    grid = torch.stack(
        [
            sample_class(idx, samples_per_class, guidance)[:, 0]
            for idx in range(classes)
        ]
    )
    model.train()
    return grid  # (classes, samples_per_class, domain_size, domain_size)


def save_grid(grid, path):
    fig, ax = plt.subplots(
        classes,
        samples_per_class,
        figsize=(samples_per_class, classes),
        dpi=domain_size,
    )
    for i in range(classes):
        for j in range(samples_per_class):
            ax[i, j].imshow(grid[i, j].T, cmap="binary", origin="lower", vmin=0, vmax=1)
            ax[i, j].set_aspect("equal")
            ax[i, j].axis("off")
            ax[i, j].set_rasterized(True)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(path)
    plt.close()


# --------------------------------------- metrics -------------------------------------
# two metrics have already failed here. blank_fraction counted pixels near 0 or 1, which
# a crisp shape is made entirely of, so it scored the best grids worst. uniform_fraction
# replaced it with a variance test, which a blank cell fails but speckle passes -- and
# speckle is exactly what the ellipse row degrades into. this tests both: a real shape
# covers at least 12 percent of the frame (the ellipse, thinnest of the six, sits at
# 0.124) and survives erosion, while scattered dots cover almost nothing and vanish
def sample_validity(grid):
    binary = (grid.reshape(-1, 1, domain_size, domain_size) >= 0.5).float()
    eroded = -nn.functional.max_pool2d(-binary, 3, stride=1, padding=1)
    ink = binary.flatten(1).mean(dim=1)
    kept = eroded.flatten(1).sum(dim=1) / binary.flatten(1).sum(dim=1).clamp_min(1.0)
    return (ink > min_ink) & (kept > min_kept)


def degenerate_fraction(grid):
    return 1.0 - sample_validity(grid).float().mean().item()


@torch.no_grad()
def class_accuracy(grid):
    # the classifier has no "none of these" class, so it labels noise confidently -- and
    # because the ellipse carries the least ink, a blank frame lands there. a cell
    # only counts if it is a real shape AND carries the right label. the per-class
    # breakdown is returned too, so one dead class cannot hide inside the average
    flat = grid.reshape(classes * samples_per_class, 1, domain_size, domain_size)
    prediction = classifier((flat * 2.0 - 1.0).to(device)).argmax(dim=1).cpu()
    expected = torch.arange(classes).repeat_interleave(samples_per_class)
    correct = (prediction == expected) & sample_validity(grid)
    per_class = correct.reshape(classes, samples_per_class).float().mean(dim=1)
    return correct.float().mean().item(), per_class


@torch.no_grad()
def label_sensitivity(step):
    # how far the predicted noise moves when only the label changes, measured near pure
    # noise where the image itself cannot reveal the class. at mid-range timesteps the
    # partly denoised image already gives the answer, so the number looks fine even
    # when the conditioning is dead
    model.eval()
    x = torch.randn(classes, 1, domain_size, domain_size, device=device)
    t = torch.full((classes,), step, device=device, dtype=torch.long)
    y = torch.arange(classes, device=device)
    y_null = torch.full((classes,), classes, device=device, dtype=torch.long)
    difference = model(x, t, y) - model(x, t, y_null)
    model.train()
    return difference.abs().mean().item()


# -------------------------------------- training -------------------------------------
train_cost = [0] * epochs
best_train_cost = float("inf")
training_time = 0.0

if args.sweep_only:
    checkpoint = torch.load(model_path, weights_only=False, map_location=device)
    model.load_state_dict(checkpoint["model"])
    classifier.load_state_dict(checkpoint["classifier"])
    train_cost = checkpoint["train_cost"]
    best_train_cost = min(train_cost)
    print(f"loaded checkpoint trained for {len(train_cost)} epochs", flush=True)
else:
    start_time = time.perf_counter()
    pbar = tqdm(range(epochs), desc="Training: ", ncols=90)
    for epoch in pbar:
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            t = torch.randint(0, T, (x.shape[0],), device=device)
            noise = torch.randn_like(x)
            # a fraction of labels is replaced by the null token, so the same network
            # also learns the unconditional prediction that guidance extrapolates from
            dropped = torch.rand(x.shape[0], device=device) < label_dropout
            y = torch.where(dropped, torch.full_like(y, classes), y)

            optimizer.zero_grad()
            cost = cost_fun(model(q_sample(x, t, noise), t, y), noise)
            cost.backward()
            optimizer.step()
            train_cost[epoch] += cost.item()
        train_cost[epoch] /= len(train_loader)
        scheduler.step()
        best_train_cost = min(best_train_cost, train_cost[epoch])

        if epoch in sample_epochs or epoch == epochs - 1:
            grid = sample_grid(sample_guidance)
            save_grid(grid, RESULTS_DIR / f"samples_{run_id}_{epoch:04d}.png")

        if epoch % print_every == 0:
            pbar.set_postfix({"train": f"{train_cost[epoch]:.2e}"})
    training_time = time.perf_counter() - start_time

# ------------------------------------ export model -----------------------------------
if not args.sweep_only:
    torch.save(
        {
            "model": model.state_dict(),
            "classifier": classifier.state_dict(),
            "train_cost": train_cost,
            "T": T,
            "domain_size": domain_size,
            "base_channels": base_channels,
            "time_emb_dim": time_emb_dim,
            "labels": labels,
            "run_id": run_id,
        },
        model_path,
    )

# --------------------------------------- sweep ---------------------------------------
# guidance never touches a weight, so the whole range runs against the one model already
# in memory -- no reload, and no way for the architecture to drift from the checkpoint
records = []
start_time = time.perf_counter()
for guidance in tqdm(guidances, desc="Guidance: ", ncols=90):
    grid = sample_grid(guidance)
    name = f"guidance_{guidance:.2f}_{run_id}.png".replace(".", "p", 1)
    save_grid(grid, RESULTS_DIR / name)
    accuracy, per_class = class_accuracy(grid)
    records.append(
        {
            "guidance": guidance,
            "accuracy": accuracy,
            "per_class": per_class,
            "degenerate": degenerate_fraction(grid),
            "mean": grid.mean().item(),
            "spread": grid.std(dim=1).mean().item(),  # variation within a class
        }
    )
sweep_time = time.perf_counter() - start_time

sensitivity = label_sensitivity(sensitivity_step)
emb_norm = model.label_emb.weight.norm().item()
best = max(records, key=lambda r: r["accuracy"])

# ----------------------------------- postprocessing ----------------------------------
fig, ax = plt.subplots()
ax.plot(train_cost, "k")
ax.set_yscale("log")
ax.set_xlabel("epoch")
ax.set_ylabel("loss")
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.savefig(RESULTS_DIR / f"loss_{run_id}.png")
if not args.book:
    plt.show()
plt.close()

fig, ax = plt.subplots()
ax.plot([r["guidance"] for r in records], [r["accuracy"] for r in records], "k-o")
ax.plot([r["guidance"] for r in records], [r["degenerate"] for r in records], "r-o")
ax.set_xlabel("guidance")
ax.set_ylabel("class accuracy (black), degenerate fraction (red)")
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.savefig(RESULTS_DIR / f"sweep_{run_id}.png")
if not args.book:
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
    f.write("shape conditional DDPM v2, merged training and guidance sweep\n")
    f.write(f"run_id: {run_id}\n")
    f.write(f"device: {device}\n")
    f.write(f"sweep_only: {args.sweep_only}\n")
    f.write(f"checkpoint: {model_path.resolve()}\n")
    f.write(f"checkpoint_epochs: {len(train_cost)}\n")
    f.write("architecture: NN.UNet subclass, ConditionalResBlock on DCN and MLP\n")
    f.write("conditioning: class embedding added to the time embedding, per ResBlock\n")
    f.write(f"labels: {','.join(labels)}\n")
    f.write(f"domain_size: {domain_size}\n")
    f.write(f"samples: {len(X)}\n")
    f.write(f"epochs: {epochs}\n")
    f.write(f"batch_size: {batch_size}\n")
    f.write(f"learning_rate: {lr}\n")
    f.write(f"weight_decay: {weight_decay}\n")
    f.write(f"label_dropout: {label_dropout}\n")
    f.write(f"label_emb_std: {label_emb_std}\n")
    f.write(f"label_emb_weight_decay: {label_emb_weight_decay}\n")
    f.write(f"label_emb_norm: {emb_norm:.6f}\n")
    f.write(f"label_sensitivity_at_t{sensitivity_step}: {sensitivity:.6f}\n")
    f.write(f"timesteps: {T}\n")
    f.write(f"noise_schedule: {noise_schedule}\n")
    f.write(f"base_channels: {base_channels}\n")
    f.write(f"time_emb_dim: {time_emb_dim}\n")
    f.write(f"parameters: {sum(p.numel() for p in model.parameters())}\n")
    f.write("reconstruction_loss: MSELoss on predicted noise\n")
    f.write(f"training_time_seconds: {training_time:.2f}\n")
    f.write(f"sweep_time_seconds: {sweep_time:.2f}\n")
    f.write(f"final_train_loss: {train_cost[-1]:.8e}\n")
    f.write(f"best_train_loss: {best_train_cost:.8e}\n")
    f.write(f"classifier_accuracy_on_real: {classifier_accuracy:.6f}\n")
    f.write(f"sample_seed: {sample_seed}\n")
    f.write(f"samples_per_class: {samples_per_class}\n")
    f.write(f"best_guidance: {best['guidance']:.2f}\n")
    f.write(f"best_class_accuracy: {best['accuracy']:.6f}\n")
    f.write("guidance,class_accuracy,degenerate_fraction,mean_intensity,spread\n")
    for r in records:
        f.write(
            f"{r['guidance']:.2f},{r['accuracy']:.6f},{r['degenerate']:.6f},"
            f"{r['mean']:.6f},{r['spread']:.6f}\n"
        )
    # per class, so a row that generates nothing is visible instead of averaged away
    f.write(f"guidance,{','.join(labels)}\n")
    for r in records:
        row = ",".join(f"{value:.4f}" for value in r["per_class"].tolist())
        f.write(f"{r['guidance']:.2f},{row}\n")
print(f"saved {summary_path}")
print(f"best guidance {best['guidance']:.2f} at accuracy {best['accuracy']:.4f}")
