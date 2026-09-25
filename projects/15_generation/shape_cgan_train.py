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
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
MODELS_DIR = (BASE_DIR / "../../models").resolve()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
# conditional GAN: the discriminator outputs a probability and is trained with binary
# cross-entropy, as in shape_GAN.py. no timestep here, so the label needs no per-block
# injection -- it enters once at each network's input, the standard construction, which
# suits the single-argument forward of NN.DCN
# hyperparameters
epochs = 1000
batch_size = 64
lr = 1e-4
betas = (0.5, 0.999)
label_smoothing = 0.9  # real target below 1.0, a brake on discriminator overconfidence

# define loss
cost_fun = nn.BCEWithLogitsLoss()

# model settings
latent_dim = 64
base_channels = 64

# conditioning -- same six classes in the same load order as shape_cvae_train.py. the
# generator takes the label appended to its latent; the discriminator judges an image
# against a label through a projection term, so it asks "is this a real star" rather
# than only "is this a real shape". that pressure is why the label cannot be ignored
# here the way it was in the diffusion model, where the objective never needed it
domain_size = 128
labels = ["circle", "ellipse", "square", "triangle", "cross", "star"]
classes = len(labels)
label_dim = 32  # width of the learned class embedding fed to the generator

# sampling
samples_per_class = 8
sample_epochs = (0, 10, 25, 50, 100, 250, 500, 750)
sample_seed = 7
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

# ----------------------------------- prepare data ------------------------------------
# the six classes are stored one file per shape with no labels, so the class index is
# reconstructed here from the load order
X = []
targets = []
for idx, label in enumerate(labels):
    shapes = np.load(DATA_DIR / f"shapes_{label}_{domain_size}.npy")
    X.append(shapes)
    targets.append(np.full(len(shapes), idx))

X = np.concatenate(X, axis=0).astype(np.float32)
X = torch.from_numpy(X).unsqueeze(1) * 2.0 - 1.0  # the generator ends in tanh
targets = torch.from_numpy(np.concatenate(targets)).to(torch.long)

dataset = TensorDataset(X, targets)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)


# --------------------------------------- model ---------------------------------------
START_SIZE = 4  # spatial grid the latent is projected onto before upsampling


class Generator(nn.Module):
    # latent and label are concatenated into one vector, projected by an MLP onto a 4x4
    # grid, then expanded 4 -> 8 -> 16 -> 32 -> 64 -> 128 by a DCN whose pre_modules
    # carry the upsampling. DCN builds only ordinary convolutions, so this is the same
    # upsample-then-convolve pattern shape_vae_train.py uses for its decoder, and it
    # avoids the checkerboard artefacts transposed convolutions are prone to
    def __init__(self, z_dim, label_dim, classes, base):
        super().__init__()
        self.label_emb = nn.Embedding(classes, label_dim)
        channels = [base * 8, base * 4, base * 2, base, base // 2, 1]
        self.project = MLP(
            [z_dim + label_dim, channels[0] * START_SIZE**2],
            post_modules=[nn.ReLU(True)],
        )
        self.unflatten = nn.Unflatten(1, (channels[0], START_SIZE, START_SIZE))
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
            pre_modules=[
                nn.Upsample(scale_factor=2, mode="nearest")
                for _ in range(len(channels) - 1)
            ],
        )

    def forward(self, z, y):
        h = torch.cat([z, self.label_emb(y)], dim=1)
        return self.net(self.unflatten(self.project(h)))


class Discriminator(nn.Module):
    # a projection discriminator: the trunk scores the image unconditionally and an
    # inner product between the label embedding and the pooled features supplies the
    # class-dependent part. the label never passes through the downsampling as extra
    # channels -- the route that measured a sensitivity of 0.0072 in the diffusion model
    def __init__(self, classes, base):
        super().__init__()
        channels = [1, base, base * 2, base * 4, base * 8, base * 16]
        activations = [
            nn.LeakyReLU(0.2, inplace=True) for _ in range(len(channels) - 1)
        ]
        self.trunk = DCN(
            channels,
            activations,
            kernel_size=4,
            stride=2,
            padding=1,
            dim=2,
        )
        self.score = nn.Linear(base * 16, 1)
        self.label_emb = nn.Embedding(classes, base * 16)

    def forward(self, x, y):
        features = self.trunk(x).mean(dim=(2, 3))  # global average pool
        unconditional = self.score(features).view(-1)
        projection = (self.label_emb(y) * features).sum(dim=1)
        return unconditional + projection  # logits; the loss applies the sigmoid


# --------------------------- instantiate model & optimizer ---------------------------
generator = Generator(latent_dim, label_dim, classes, base_channels).to(device)
discriminator = Discriminator(classes, base_channels).to(device)
init_weights(generator, nn.ReLU())
init_weights(discriminator, nn.LeakyReLU(0.2))

optimizer_g = torch.optim.Adam(generator.parameters(), lr=lr, betas=betas)
optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=betas)


# --------------------------------------- helper --------------------------------------
@torch.no_grad()
def save_samples(epoch):
    # rows are classes, columns are latent draws, and the same draws are reused for
    # every row -- so a column isolates the effect of the label, exactly as in the
    # conditional VAE and diffusion grids
    generator.eval()
    torch.manual_seed(sample_seed)
    z = torch.randn(samples_per_class, latent_dim, device=device)
    grid = []
    for idx in range(classes):
        y = torch.full((samples_per_class,), idx, device=device, dtype=torch.long)
        grid.append(((generator(z, y) + 1.0) / 2.0)[:, 0].cpu())
    generator.train()

    fig, ax = plt.subplots(
        classes,
        samples_per_class,
        figsize=(samples_per_class, classes),
        dpi=domain_size,
    )
    for i in range(classes):
        for j in range(samples_per_class):
            ax[i, j].imshow(grid[i][j].T, cmap="binary", origin="lower", vmin=0, vmax=1)
            ax[i, j].set_aspect("equal")
            ax[i, j].axis("off")
            ax[i, j].set_rasterized(True)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(RESULTS_DIR / f"shape_cgan_samples_{run_id}_{epoch:04d}.png")
    plt.close()


@torch.no_grad()
def mode_spread(n=32):
    # a GAN can satisfy the discriminator by producing one convincing example per class.
    # the spread across latent draws at a fixed label detects that: near zero means
    # every sample of a class is the same picture
    generator.eval()
    z = torch.randn(n, latent_dim, device=device)
    spreads = []
    for idx in range(classes):
        y = torch.full((n,), idx, device=device, dtype=torch.long)
        spreads.append(generator(z, y).std(dim=0).mean().item())
    generator.train()
    return sum(spreads) / len(spreads)


@torch.no_grad()
def label_obedience(n=16):
    # does the label change what is drawn? decode the same latents under every class and
    # measure how far the images move. near zero means the generator ignores the label,
    # which is the failure the conditional VAE hit at high latent capacity
    generator.eval()
    z = torch.randn(n, latent_dim, device=device)
    images = []
    for idx in range(classes):
        y = torch.full((n,), idx, device=device, dtype=torch.long)
        images.append(generator(z, y))
    generator.train()
    images = torch.stack(images)
    return images.std(dim=0).mean().item()


# -------------------------------------- training -------------------------------------
generator_losses = [0] * epochs
discriminator_losses = [0] * epochs

tic = time.time()
pbar = tqdm(range(epochs), desc="Training: ", ncols=90)
for epoch in pbar:
    epoch_g, epoch_d = 0.0, 0.0

    for real, y in train_loader:
        real = real.to(device)
        y = y.to(device)
        batch = real.shape[0]

        # discriminator: real images at their own label are true, generated images at
        # the label they were asked for are false
        z = torch.randn(batch, latent_dim, device=device)
        fake = generator(z, y)
        real_logits = discriminator(real, y)
        fake_logits = discriminator(fake.detach(), y)
        loss_real = cost_fun(real_logits, torch.full_like(real_logits, label_smoothing))
        loss_fake = cost_fun(fake_logits, torch.zeros_like(fake_logits))
        discriminator_loss = loss_real + loss_fake

        optimizer_d.zero_grad()
        discriminator_loss.backward()
        optimizer_d.step()
        epoch_d += discriminator_loss.item()

        # generator: persuade the discriminator that its samples are real at that label
        z = torch.randn(batch, latent_dim, device=device)
        logits = discriminator(generator(z, y), y)
        generator_loss = cost_fun(logits, torch.ones_like(logits))

        optimizer_g.zero_grad()
        generator_loss.backward()
        optimizer_g.step()
        epoch_g += generator_loss.item()

    discriminator_losses[epoch] = epoch_d / len(train_loader)
    generator_losses[epoch] = epoch_g / len(train_loader)
    pbar.set_postfix(
        {
            "d": f"{discriminator_losses[epoch]:.2e}",
            "g": f"{generator_losses[epoch]:.2e}",
        }
    )

    # flush explicitly: stdout is block buffered under nohup, so without this the
    # diagnostics only reach the log once the process exits
    if epoch in sample_epochs or epoch == epochs - 1:
        save_samples(epoch)
        print(
            f"epoch {epoch} spread {mode_spread():.4f}"
            f" obedience {label_obedience():.4f}",
            flush=True,
        )

toc = time.time()
print(f"elapsed time {toc - tic:.2f} s")

# ------------------------------------ export model -----------------------------------
torch.save(
    {
        "generator": generator.state_dict(),
        "discriminator": discriminator.state_dict(),
        "latent_dim": latent_dim,
        "label_dim": label_dim,
        "base_channels": base_channels,
        "domain_size": domain_size,
        "labels": labels,
        "generator_losses": generator_losses,
        "discriminator_losses": discriminator_losses,
    },
    MODELS_DIR / f"shape_cgan_{domain_size}.pt",
)

# ----------------------------------- postprocessing ----------------------------------
fig, ax = plt.subplots()
ax.plot(discriminator_losses, "k")
ax.plot(generator_losses, "r")
ax.set_xlabel("epoch")
ax.set_ylabel("loss")
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.savefig(RESULTS_DIR / f"shape_cgan_loss_{run_id}.png")
if not args.book:
    plt.show()
plt.close()

np.savetxt(
    RESULTS_DIR / f"shape_cgan_history_{run_id}.csv",
    np.column_stack(
        [
            np.arange(1, epochs + 1),
            np.asarray(discriminator_losses),
            np.asarray(generator_losses),
        ]
    ),
    delimiter=",",
    header="epoch,discriminator_loss,generator_loss",
    comments="",
)

summary_path = RESULTS_DIR / f"shape_cgan_summary_{run_id}.txt"
with open(summary_path, "w", encoding="utf-8") as f:
    f.write("shape conditional GAN training summary\n")
    f.write(f"run_id: {run_id}\n")
    f.write(f"device: {device}\n")
    f.write("conditioning: label into the latent, projection discriminator\n")
    f.write(f"labels: {','.join(labels)}\n")
    f.write(f"domain_size: {domain_size}\n")
    f.write(f"samples: {len(X)}\n")
    f.write(f"epochs: {epochs}\n")
    f.write(f"batch_size: {batch_size}\n")
    f.write(f"learning_rate: {lr}\n")
    f.write(f"betas: {betas}\n")
    f.write(f"label_smoothing: {label_smoothing}\n")
    f.write("adversarial_loss: BCEWithLogitsLoss\n")
    f.write(f"latent_dim: {latent_dim}\n")
    f.write(f"label_dim: {label_dim}\n")
    f.write(f"base_channels: {base_channels}\n")
    f.write(f"generator_parameters: {sum(p.numel() for p in generator.parameters())}\n")
    f.write(
        "discriminator_parameters: "
        f"{sum(p.numel() for p in discriminator.parameters())}\n"
    )
    f.write(f"mode_spread: {mode_spread():.6f}\n")
    f.write(f"label_obedience: {label_obedience():.6f}\n")
    f.write(f"training_time_seconds: {toc - tic:.2f}\n")
    f.write(f"final_discriminator_loss: {discriminator_losses[-1]:.8e}\n")
    f.write(f"final_generator_loss: {generator_losses[-1]:.8e}\n")
print(f"saved {summary_path}")
