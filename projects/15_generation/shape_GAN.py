from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "../../data"
RESULTS_DIR = BASE_DIR / "../../results"
MODELS_DIR = BASE_DIR / "../../models"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)


# -------------------------- training settings ---------------------------
domain_size = 128
epochs = 1000
batch_size = 64
latent_dim = 64
lr = 1e-4
betas = (0.5, 0.999)
sample_every = 10


# ----------------------------- prepare data -----------------------------
labels = ["circle", "ellipse", "square", "triangle", "cross", "star"]

data = []
for label in labels:
    path = DATA_DIR / f"shapes_{label}_{domain_size}.npy"
    data.append(np.load(path))

data = np.concatenate(data, axis=0).astype(np.float32)
data = torch.from_numpy(data).unsqueeze(1)
data = data * 2.0 - 1.0  # Generator uses tanh, so images live in [-1, 1].

dataset = TensorDataset(data)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)


# ------------------------------- models ---------------------------------
class Generator(nn.Module):
    def __init__(self, z_dim=64, base_channels=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(z_dim, base_channels * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(base_channels * 16),
            nn.ReLU(True),
            nn.ConvTranspose2d(base_channels * 16, base_channels * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(base_channels * 2, base_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(True),
            nn.ConvTranspose2d(base_channels, 1, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self, base_channels=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, base_channels, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 4, base_channels * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 8, base_channels * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 16, 1, 4, 1, 0),
        )

    def forward(self, x):
        return self.net(x).view(-1)


def init_weights(module):
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
        nn.init.normal_(module.weight, 0.0, 0.02)
        if getattr(module, "bias", None) is not None:
            nn.init.zeros_(module.bias)


def save_samples(generator, fixed_noise, epoch):
    generator.eval()
    with torch.no_grad():
        samples = generator(fixed_noise).cpu()
    samples = (samples + 1.0) / 2.0

    fig, axes = plt.subplots(4, 4, figsize=(4, 4), dpi=domain_size)
    for ax, sample in zip(axes.flat, samples[:16]):
        ax.imshow(sample[0].numpy().T, cmap="binary", origin="lower", vmin=0, vmax=1)
        ax.set_aspect("equal")
        ax.axis("off")
    fig.tight_layout(pad=0)
    fig.savefig(RESULTS_DIR / f"shape_GAN_epoch_{epoch:04d}.png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    generator.train()


# -------------------------- instantiate model ---------------------------
generator = Generator(latent_dim).to(device)
discriminator = Discriminator().to(device)
generator.apply(init_weights)
discriminator.apply(init_weights)

criterion = nn.BCEWithLogitsLoss()
optimizer_g = torch.optim.Adam(generator.parameters(), lr=lr, betas=betas)
optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=betas)

fixed_noise = torch.randn(16, latent_dim, 1, 1, device=device)


# ------------------------------- training -------------------------------
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

generator_losses = []
discriminator_losses = []

pbar = tqdm(range(epochs), desc="Training GAN", ncols=90)
for epoch in pbar:
    epoch_g_loss = 0.0
    epoch_d_loss = 0.0

    for real, in train_loader:
        real = real.to(device)
        batch = real.shape[0]

        real_targets = torch.empty(batch, device=device).uniform_(0.8, 1.0)
        fake_targets = torch.empty(batch, device=device).uniform_(0.0, 0.2)

        # Train discriminator to classify real images as real and generated images as fake.
        z = torch.randn(batch, latent_dim, 1, 1, device=device)
        fake = generator(z)

        real_logits = discriminator(real)
        fake_logits = discriminator(fake.detach())
        real_loss = criterion(real_logits, real_targets)
        fake_loss = criterion(fake_logits, fake_targets)
        discriminator_loss = real_loss + fake_loss

        optimizer_d.zero_grad()
        discriminator_loss.backward()
        optimizer_d.step()

        # Train generator to make generated images classify as real.
        z = torch.randn(batch, latent_dim, 1, 1, device=device)
        fake = generator(z)
        fake_logits = discriminator(fake)
        generator_loss = criterion(fake_logits, real_targets)

        optimizer_g.zero_grad()
        generator_loss.backward()
        optimizer_g.step()

        epoch_d_loss += discriminator_loss.item()
        epoch_g_loss += generator_loss.item()

    avg_d_loss = epoch_d_loss / len(train_loader)
    avg_g_loss = epoch_g_loss / len(train_loader)
    discriminator_losses.append(avg_d_loss)
    generator_losses.append(avg_g_loss)

    if epoch % sample_every == 0 or epoch == epochs - 1:
        save_samples(generator, fixed_noise, epoch)

    pbar.set_postfix({"discriminator": f"{avg_d_loss:.2e}", "generator": f"{avg_g_loss:.2e}"})


# ----------------------------- export model -----------------------------
torch.save(
    {
        "generator": generator.state_dict(),
        "discriminator": discriminator.state_dict(),
        "latent_dim": latent_dim,
        "domain_size": domain_size,
        "labels": labels,
        "generator_losses": generator_losses,
        "discriminator_losses": discriminator_losses,
    },
    MODELS_DIR / f"shape_GAN_{domain_size}.pt",
)

fig, ax = plt.subplots()
ax.plot(discriminator_losses, label="discriminator")
ax.plot(generator_losses, label="generator")
ax.set_xlabel("epoch")
ax.set_ylabel("loss")
ax.legend()
fig.tight_layout()
fig.savefig(RESULTS_DIR / "shape_GAN_losses.png", dpi=200)
plt.close(fig)
