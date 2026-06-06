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


domain_size = 128
epochs = 1000
batch_size = 64
latent_dim = 64
critic_steps = 5
lambda_gp = 10.0
lr = 1e-4
betas = (0.0, 0.9)
sample_every = 25


labels = ["circle", "ellipse", "square", "triangle", "cross", "star"]

data = []
for label in labels:
    path = DATA_DIR / f"shapes_{label}_{domain_size}.npy"
    data.append(np.load(path))

data = np.concatenate(data, axis=0).astype(np.float32)
data = torch.from_numpy(data).unsqueeze(1)
data = data * 2.0 - 1.0

dataset = TensorDataset(data)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)


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


class Critic(nn.Module):
    def __init__(self, base_channels=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, base_channels, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 4, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 4, base_channels * 8, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 8, base_channels * 16, 4, 2, 1),
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


def gradient_penalty(critic, real, fake):
    batch = real.shape[0]
    alpha = torch.rand(batch, 1, 1, 1, device=device)
    interpolated = alpha * real + (1 - alpha) * fake
    interpolated.requires_grad_(True)

    scores = critic(interpolated)
    gradients = torch.autograd.grad(
        outputs=scores,
        inputs=interpolated,
        grad_outputs=torch.ones_like(scores),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(batch, -1)
    return ((gradients.norm(2, dim=1) - 1) ** 2).mean()


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
    fig.savefig(RESULTS_DIR / f"shape_WGAN_epoch_{epoch:04d}.png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    generator.train()


generator = Generator(latent_dim).to(device)
critic = Critic().to(device)
generator.apply(init_weights)
critic.apply(init_weights)

optimizer_g = torch.optim.Adam(generator.parameters(), lr=lr, betas=betas)
optimizer_c = torch.optim.Adam(critic.parameters(), lr=lr, betas=betas)

fixed_noise = torch.randn(16, latent_dim, 1, 1, device=device)


RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

generator_losses = []
critic_losses = []

pbar = tqdm(range(epochs), desc="Training WGAN-GP", ncols=90)
for epoch in pbar:
    epoch_g_loss = 0.0
    epoch_c_loss = 0.0
    g_updates = 0

    for batch_idx, (real,) in enumerate(train_loader):
        real = real.to(device)
        batch = real.shape[0]

        z = torch.randn(batch, latent_dim, 1, 1, device=device)
        fake = generator(z)
        real_score = critic(real)
        fake_score = critic(fake.detach())
        gp = gradient_penalty(critic, real, fake.detach())
        critic_loss = fake_score.mean() - real_score.mean() + lambda_gp * gp

        optimizer_c.zero_grad()
        critic_loss.backward()
        optimizer_c.step()

        epoch_c_loss += critic_loss.item()

        if batch_idx % critic_steps == 0:
            z = torch.randn(batch, latent_dim, 1, 1, device=device)
            fake = generator(z)
            generator_loss = -critic(fake).mean()

            optimizer_g.zero_grad()
            generator_loss.backward()
            optimizer_g.step()

            epoch_g_loss += generator_loss.item()
            g_updates += 1

    avg_c_loss = epoch_c_loss / len(train_loader)
    avg_g_loss = epoch_g_loss / max(g_updates, 1)
    critic_losses.append(avg_c_loss)
    generator_losses.append(avg_g_loss)

    if epoch % sample_every == 0 or epoch == epochs - 1:
        save_samples(generator, fixed_noise, epoch)

    pbar.set_postfix({"critic": f"{avg_c_loss:.2e}", "generator": f"{avg_g_loss:.2e}"})


torch.save(
    {
        "generator": generator.state_dict(),
        "critic": critic.state_dict(),
        "latent_dim": latent_dim,
        "domain_size": domain_size,
        "labels": labels,
        "generator_losses": generator_losses,
        "critic_losses": critic_losses,
    },
    MODELS_DIR / f"shape_WGAN_{domain_size}.pt",
)

fig, ax = plt.subplots()
ax.plot(critic_losses, label="critic")
ax.plot(generator_losses, label="generator")
ax.set_xlabel("epoch")
ax.set_ylabel("loss")
ax.legend()
fig.tight_layout()
fig.savefig(RESULTS_DIR / "shape_WGAN_losses.png", dpi=200)
plt.close(fig)
