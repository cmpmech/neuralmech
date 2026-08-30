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
from NN import DCN, MLP

BASE_DIR = Path(__file__).parent
DATA_DIR = (BASE_DIR / "../../data").resolve()
RESULTS_DIR = (BASE_DIR / "../../results/shape_GAN").resolve()
MODELS_DIR = (BASE_DIR / "../../models").resolve()
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
# unconditional GAN: the discriminator outputs a probability and both networks are
# trained with binary cross-entropy. the unlabelled baseline for shape_cgan_train.py
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
n_samples = 16
sample_every = 10
sample_seed = 7
print_every = 10
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

# ----------------------------------- prepare data ------------------------------------
# all six classes pooled into one unlabelled set
X = [np.load(DATA_DIR / f"shapes_{label}_{domain_size}.npy") for label in labels]
X = np.concatenate(X, axis=0).astype(np.float32)
X = torch.from_numpy(X).unsqueeze(1) * 2.0 - 1.0  # the generator ends in tanh

dataset = TensorDataset(X)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)


# --------------------------------------- model ---------------------------------------
# the upsampling lives in the pre_modules slot DCN's docstring reserves for resampling,
# and a learned ConvTranspose2d goes there rather than a fixed nn.Upsample. nearest
# neighbour interpolation followed by 3x3 convolutions is a low-pass pipeline: it
# rounded off exactly the corners that separate a square from a blob. kernel 2 with
# stride 2 divides evenly, so the transposed convolution never overlaps itself and the
# checkerboard artefact it is usually blamed for cannot arise
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
    # strided convolutions down to a single score. batch normalization is kept here,
    # unlike the WGAN critic where it would break the gradient penalty
    def __init__(self, base):
        super().__init__()
        channels = [1, base, base * 2, base * 4, base * 8, base * 16]
        post = [nn.LeakyReLU(0.2, inplace=True)]
        post += [
            [nn.BatchNorm2d(channels[i + 1]), nn.LeakyReLU(0.2, inplace=True)]
            for i in range(1, len(channels) - 1)
        ]
        self.trunk = DCN(
            channels, post, kernel_size=4, stride=2, padding=1, dim=2
        )
        self.score = MLP([base * 16, 1])

    def forward(self, x):
        features = self.trunk(x).mean(dim=(2, 3))  # global average pool
        return self.score(features).view(-1)  # logits; the loss applies the sigmoid


# --------------------------- instantiate model & optimizer ---------------------------
generator = Generator(latent_dim, base_channels).to(device)
discriminator = Discriminator(base_channels).to(device)
init_weights(generator, nn.ReLU())
init_weights(discriminator, nn.LeakyReLU(0.2))

optimizer_g = torch.optim.Adam(generator.parameters(), lr=lr, betas=betas)
optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=betas)


# --------------------------------------- helper --------------------------------------
@torch.no_grad()
def save_samples(epoch):
    # the same fixed latents every time, so a grid changes only because the generator
    # changed and not because the noise did
    generator.eval()
    torch.manual_seed(sample_seed)
    z = torch.randn(n_samples, latent_dim, device=device)
    grid = ((generator(z) + 1.0) / 2.0)[:, 0].cpu()
    generator.train()

    fig, ax = plt.subplots(4, 4, figsize=(4, 4), dpi=domain_size)
    for axis, image in zip(ax.flat, grid):
        axis.imshow(image.T, cmap="binary", origin="lower", vmin=0, vmax=1)
        axis.set_aspect("equal")
        axis.axis("off")
        axis.set_rasterized(True)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(RESULTS_DIR / f"samples_{run_id}_{epoch:04d}.png")
    plt.close()


@torch.no_grad()
def mode_spread(n=32):
    # mean pairwise distance between generated images. a collapsed generator returns
    # nearly the same picture whatever the latent, driving this towards zero
    generator.eval()
    z = torch.randn(n, latent_dim, device=device)
    samples = generator(z).reshape(n, -1)
    distances = torch.cdist(samples, samples)
    generator.train()
    return (distances.sum() / (n * (n - 1))).item() / samples.shape[1] ** 0.5


# -------------------------------------- training -------------------------------------
generator_cost = [0] * epochs
discriminator_cost = [0] * epochs
spread = [0] * epochs

start_time = time.perf_counter()
pbar = tqdm(range(epochs), desc="Training: ", ncols=90)
for epoch in pbar:
    for x in train_loader:
        real = x[0].to(device)
        batch = real.shape[0]
        real_labels = torch.empty(batch, device=device).uniform_(*real_target)
        fake_labels = torch.empty(batch, device=device).uniform_(*fake_target)

        # the discriminator learns to separate real from generated
        z = torch.randn(batch, latent_dim, device=device)
        fake = generator(z)
        cost_d = cost_fun(discriminator(real), real_labels)
        cost_d = cost_d + cost_fun(discriminator(fake.detach()), fake_labels)

        optimizer_d.zero_grad()
        cost_d.backward()
        optimizer_d.step()

        # the generator learns to make the discriminator call its output real
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

    if epoch % sample_every == 0 or epoch == epochs - 1:
        save_samples(epoch)

    if epoch % print_every == 0:
        pbar.set_postfix(
            {
                "d": f"{discriminator_cost[epoch]:.2e}",
                "g": f"{generator_cost[epoch]:.2e}",
                "spread": f"{spread[epoch]:.4f}",
            }
        )
training_time = time.perf_counter() - start_time

# ------------------------------------ export model -----------------------------------
torch.save(
    {
        "generator": generator.state_dict(),
        "discriminator": discriminator.state_dict(),
        "generator_cost": generator_cost,
        "discriminator_cost": discriminator_cost,
        "spread": spread,
        "latent_dim": latent_dim,
        "domain_size": domain_size,
        "labels": labels,
        "run_id": run_id,
    },
    MODELS_DIR / f"shape_GAN_{domain_size}.pt2",
)

# ----------------------------------- postprocessing ----------------------------------
fig, ax = plt.subplots()
ax.plot(discriminator_cost, "k")
ax.plot(generator_cost, "r")
ax.set_xlabel("epoch")
ax.set_ylabel("loss")
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.savefig(RESULTS_DIR / f"loss_{run_id}.png")
if not args.book:
    plt.show()
plt.close()

fig, ax = plt.subplots()
ax.plot(spread, "k")
ax.set_xlabel("epoch")
ax.set_ylabel("mode spread")
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.savefig(RESULTS_DIR / f"spread_{run_id}.png")
if not args.book:
    plt.show()
plt.close()

np.savetxt(
    RESULTS_DIR / f"history_{run_id}.csv",
    np.column_stack(
        [
            np.arange(1, epochs + 1),
            np.asarray(discriminator_cost),
            np.asarray(generator_cost),
            np.asarray(spread),
        ]
    ),
    delimiter=",",
    header="epoch,discriminator_loss,generator_loss,mode_spread",
    comments="",
)

summary_path = RESULTS_DIR / f"summary_{run_id}.txt"
with open(summary_path, "w", encoding="utf-8") as f:
    f.write("shape unconditional GAN training summary\n")
    f.write(f"run_id: {run_id}\n")
    f.write(f"device: {device}\n")
    f.write("architecture: NN.DCN and NN.MLP, ConvTranspose resampler in pre_modules\n")
    f.write("conditioning: none, unconditional baseline\n")
    f.write(f"labels: {','.join(labels)}\n")
    f.write(f"domain_size: {domain_size}\n")
    f.write(f"samples: {len(X)}\n")
    f.write(f"epochs: {epochs}\n")
    f.write(f"batch_size: {batch_size}\n")
    f.write(f"learning_rate: {lr}\n")
    f.write(f"betas: {betas}\n")
    f.write(f"latent_dim: {latent_dim}\n")
    f.write(f"base_channels: {base_channels}\n")
    f.write(f"label_smoothing_real: {real_target}\n")
    f.write("loss: BCEWithLogitsLoss\n")
    g_parameters = sum(p.numel() for p in generator.parameters())
    d_parameters = sum(p.numel() for p in discriminator.parameters())
    f.write(f"generator_parameters: {g_parameters}\n")
    f.write(f"discriminator_parameters: {d_parameters}\n")
    f.write(f"training_time_seconds: {training_time:.2f}\n")
    f.write(f"final_discriminator_loss: {discriminator_cost[-1]:.8e}\n")
    f.write(f"final_generator_loss: {generator_cost[-1]:.8e}\n")
    f.write(f"final_mode_spread: {spread[-1]:.8e}\n")
    f.write(f"sample_seed: {sample_seed}\n")
print(f"saved {summary_path}")
