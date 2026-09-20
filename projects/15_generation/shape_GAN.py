import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from DL import init_weights
from NN import DCN, MLP
from postprocessing import save_csv

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
RGB_PDF_DIR = (RESULTS_DIR / "rgb_pdf").resolve()
CSV_DIR = (RESULTS_DIR / "data").resolve()
ANIMATION_DIR = (RESULTS_DIR / "animations/shape_gan").resolve()
DATA_DIR = (BASE_DIR / "../../data").resolve()
MODEL_DIR = (BASE_DIR / "../../models").resolve()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
parser.add_argument("--animate", action="store_true")
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
# hyperparameters
epochs = 1000
batch_size = 64
lr = 1e-4
betas = (0.5, 0.999)
real_target = (0.8, 1.0)  # one-sided label smoothing, a brake on the discriminator
fake_target = (0.0, 0.2)

# define loss
cost_fun = nn.BCEWithLogitsLoss()

# model settings
domain_size = 128
latent_dim = 64
base_channels = 64
start_size = 4  # spatial grid the latent is projected onto before upsampling
labels = ["circle", "ellipse", "square", "triangle", "cross", "star"]

# sampling
samples = 16
sample_every = 10
sample_seed = 7
print_every = 10

# ------------------------------------ prepare data -----------------------------------
# all six classes pooled into one unlabelled set
X = [np.load(DATA_DIR / f"shapes_{label}_{domain_size}.npy") for label in labels]
X = np.concatenate(X, axis=0).astype(np.float32)
X = torch.from_numpy(X).unsqueeze(1) * 2.0 - 1.0  # the generator ends in tanh

dataset = TensorDataset(X)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)


# --------------------------------------- model ---------------------------------------
class Generator(nn.Module):
    def __init__(self, z_dim, base):
        super().__init__()
        channels = [base * 16, base * 8, base * 4, base * 2, base, 1]
        self.project = MLP(
            [z_dim, channels[0] * start_size**2], post_modules=[nn.ReLU(True)]
        )
        self.unflatten = nn.Unflatten(1, (channels[0], start_size, start_size))
        post = [
            [nn.BatchNorm2d(channels[i + 1]), nn.ReLU(True)]
            for i in range(len(channels) - 2)
        ]
        self.net = DCN(
            channels,
            post + [nn.Tanh()],
            kernel_size=3,
            stride=1,
            padding=1,
            dim=2,
            pre_modules=[nn.ConvTranspose2d(ch, ch, 2, 2) for ch in channels[:-1]],
        )

    def forward(self, z):
        return self.net(self.unflatten(self.project(z)))


class Discriminator(nn.Module):
    def __init__(self, base):
        super().__init__()
        channels = [1, base, base * 2, base * 4, base * 8, base * 16]
        post = [nn.LeakyReLU(0.2, inplace=True)]
        post += [
            [nn.BatchNorm2d(channels[i + 1]), nn.LeakyReLU(0.2, inplace=True)]
            for i in range(1, len(channels) - 1)
        ]
        self.trunk = DCN(channels, post, kernel_size=4, stride=2, padding=1, dim=2)
        self.score = MLP([base * 16, 1])

    def forward(self, x):
        features = self.trunk(x).mean(dim=(2, 3))
        return self.score(features).view(-1)  # logits, the loss applies the sigmoid


# --------------------------- instantiate model & optimizer ---------------------------
generator = Generator(latent_dim, base_channels).to(device)
discriminator = Discriminator(base_channels).to(device)
init_weights(generator, nn.ReLU())
init_weights(discriminator, nn.LeakyReLU(0.2))

optimizer_g = torch.optim.Adam(generator.parameters(), lr=lr, betas=betas)
optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=betas)


# --------------------------------------- helper --------------------------------------
@torch.no_grad()
def sample_figure():
    generator.eval()
    torch.manual_seed(sample_seed)
    z = torch.randn(samples, latent_dim, device=device)
    grid = ((generator(z) + 1.0) / 2.0)[:, 0].cpu()
    generator.train()

    fig, ax = plt.subplots(4, 4, figsize=(4, 4), dpi=domain_size)
    for axis, image in zip(ax.flat, grid):
        axis.imshow(image.T, cmap="binary", origin="lower", vmin=0, vmax=1)
        axis.set_aspect("equal")
        axis.axis("off")
        axis.set_rasterized(True)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    return fig


@torch.no_grad()
def mode_spread(n=32):
    # a collapsed generator returns nearly the same picture whatever the latent
    generator.eval()
    z = torch.randn(n, latent_dim, device=device)
    generated = generator(z).reshape(n, -1)
    distances = torch.cdist(generated, generated)
    generator.train()
    return (distances.sum() / (n * (n - 1))).item() / generated.shape[1] ** 0.5


# -------------------------------------- training -------------------------------------
if args.animate:
    ANIMATION_DIR.mkdir(parents=True, exist_ok=True)

generator_cost = [0] * epochs
discriminator_cost = [0] * epochs
spread = [0] * epochs

tic = time.time()
pbar = tqdm(range(epochs), desc="Training: ", ncols=90)
for epoch in pbar:
    for x in train_loader:
        real = x[0].to(device)
        batch = real.shape[0]
        real_labels = torch.empty(batch, device=device).uniform_(*real_target)
        fake_labels = torch.empty(batch, device=device).uniform_(*fake_target)

        z = torch.randn(batch, latent_dim, device=device)
        fake = generator(z)
        cost_d = cost_fun(discriminator(real), real_labels)
        cost_d = cost_d + cost_fun(discriminator(fake.detach()), fake_labels)

        optimizer_d.zero_grad()
        cost_d.backward()
        optimizer_d.step()

        z = torch.randn(batch, latent_dim, device=device)
        cost_g = cost_fun(discriminator(generator(z)), real_labels)

        optimizer_g.zero_grad()
        cost_g.backward()
        optimizer_g.step()

        discriminator_cost[epoch] += cost_d.item()
        generator_cost[epoch] += cost_g.item()

    discriminator_cost[epoch] /= len(train_loader)
    generator_cost[epoch] /= len(train_loader)
    spread[epoch] = mode_spread()

    if args.animate and epoch % sample_every == 0:
        fig = sample_figure()
        fig.savefig(ANIMATION_DIR / f"frame_{epoch // sample_every}.jpg")
        plt.close(fig)

    if epoch % print_every == 0:
        pbar.set_postfix(
            {
                "d": f"{discriminator_cost[epoch]:.2e}",
                "g": f"{generator_cost[epoch]:.2e}",
                "spread": f"{spread[epoch]:.4f}",
            }
        )
toc = time.time()
print(f"elapsed time {toc - tic:.2f} s")

# --------------------------------------- export --------------------------------------
torch.save(generator, MODEL_DIR / f"shape_GAN_{domain_size}.pt2")

# ----------------------------------- postprocessing ----------------------------------
fig_samples = sample_figure()

fig_cost, ax = plt.subplots()
ax.plot(discriminator_cost, "k")
ax.plot(generator_cost, "r")
ax.set_xlabel("epoch")
ax.set_ylabel("cost")

fig_spread, ax = plt.subplots()
ax.plot(spread, "k")
ax.set_xlabel("epoch")
ax.set_ylabel("mode spread")

if not args.book:
    plt.show()
# -------------------------------- book postprocessing --------------------------------
else:
    fig_samples.savefig(RGB_PDF_DIR / "shape_gan_samples.pdf")
    fig_cost.savefig(RGB_PDF_DIR / "shape_gan_cost.pdf")
    fig_spread.savefig(RGB_PDF_DIR / "shape_gan_spread.pdf")
    plt.close("all")
    save_csv(
        CSV_DIR / "shape_gan_history.csv",
        x=np.arange(1, epochs + 1),
        y1=discriminator_cost,
        y2=generator_cost,
        y3=spread,
    )
