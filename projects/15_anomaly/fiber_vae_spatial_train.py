import copy
from datetime import datetime
from functools import partial
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchinfo import summary
from tqdm import tqdm

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True


# ------------------------------- helpers --------------------------------
def init_weights(model, activation=None):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            if isinstance(activation, nn.PReLU):
                nn.init.kaiming_uniform_(
                    m.weight, a=activation.init, nonlinearity="leaky_relu"
                )
            else:
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)


class Standardizer(nn.Module):
    def __init__(self, X, dim=0):
        super().__init__()
        self.register_buffer("x_mean", X.mean(dim=dim, keepdim=True))
        self.register_buffer("x_std", X.std(dim=dim, keepdim=True).clamp_min(1e-8))

    def __call__(self, x):
        return (x - self.x_mean) / self.x_std


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, activation=None):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(1, out_channels),
            activation if activation is not None else nn.PReLU(init=0.2),
        )

    def forward(self, x):
        return self.block(x)


class SpatialVAE(nn.Module):
    def __init__(self, encoder, mean_layer, logvar_layer, decoder):
        super().__init__()
        self.encoder = encoder
        self.mean_layer = mean_layer
        self.logvar_layer = logvar_layer
        self.decoder = decoder

    def encode(self, x):
        features = self.encoder(x)
        return self.mean_layer(features), self.logvar_layer(features)

    def reparameterize(self, mean, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mean + eps * std
        return mean

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        y = self.decode(z)
        return y, mean, logvar


# -------------------------- training settings ---------------------------
epochs = 50
lr = 1e-3
weight_decay = 1e-2
batch_size = 32
beta = 1e-5

recon_loss = nn.BCEWithLogitsLoss(reduction="mean")


def kl_div(mean_pred, logvar_pred):
    var_pred = torch.exp(logvar_pred)
    return 0.5 * torch.mean(var_pred + mean_pred**2 - logvar_pred - 1)


def cost_fun(x_pred, mean_pred, logvar_pred, x, beta=1.0):
    bce = recon_loss(x_pred, x)
    kl = kl_div(mean_pred, logvar_pred)
    return bce + beta * kl


def reconstruction_metrics(x_prob, x_target, threshold=0.5):
    x_binary = (x_prob >= threshold).to(x_target.dtype)
    intersection = torch.sum(x_binary * x_target, dim=(1, 2, 3))
    union = torch.sum((x_binary + x_target) > 0, dim=(1, 2, 3)).clamp_min(1)
    pred_sum = torch.sum(x_binary, dim=(1, 2, 3))
    target_sum = torch.sum(x_target, dim=(1, 2, 3))
    dice_den = (pred_sum + target_sum).clamp_min(1)

    return {
        "mse": torch.mean((x_prob - x_target) ** 2).item(),
        "mae": torch.mean(torch.abs(x_prob - x_target)).item(),
        "binary_accuracy": torch.mean((x_binary == x_target).to(torch.float32)).item(),
        "iou": torch.mean(intersection / union).item(),
        "dice": torch.mean(2 * intersection / dice_den).item(),
    }


# ---------------------------- model settings ----------------------------
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

domain_size = 256
base_channels = 16
latent_channels = 16
depth = 3
act = partial(nn.PReLU, init=0.2)
architecture = "spatial"

# ----------------------------- prepare data -----------------------------
data = torch.from_numpy(np.load(BASE_DIR / f"../../data/fibers_{domain_size}.npy"))
data = data.to(torch.float32).unsqueeze(1)

dataset = TensorDataset(data)
train_data, val_data = random_split(dataset, [0.9, 0.1])
train_loader = DataLoader(
    train_data, batch_size=batch_size, shuffle=True, drop_last=True
)
val_loader = DataLoader(val_data, batch_size=len(val_data), shuffle=True)

X_train = train_data.dataset.tensors[0][train_data.indices]
standardizex = Standardizer(X_train, dim=(0, 2, 3)).to(device)

# -------------------------- instantiate model ---------------------------
encoder_modules = []
in_channels = 1
encoder_channels = []
for i in range(depth):
    out_channels = base_channels * 2**i
    encoder_channels.append(out_channels)
    encoder_modules.append(ConvBlock(in_channels, out_channels, stride=2, activation=act()))
    in_channels = out_channels

Encoder = nn.Sequential(*encoder_modules)
mean_layer = nn.Conv2d(encoder_channels[-1], latent_channels, kernel_size=1)
logvar_layer = nn.Conv2d(encoder_channels[-1], latent_channels, kernel_size=1)

decoder_modules = [
    nn.Conv2d(latent_channels, encoder_channels[-1], kernel_size=3, padding=1),
    nn.GroupNorm(1, encoder_channels[-1]),
    act(),
]
decoder_channels = encoder_channels[::-1]
in_channels = encoder_channels[-1]
for out_channels in decoder_channels[1:]:
    decoder_modules.extend(
        [
            nn.Upsample(scale_factor=2, mode="nearest"),
            ConvBlock(in_channels, out_channels, stride=1, activation=act()),
        ]
    )
    in_channels = out_channels
decoder_modules.extend(
    [
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(in_channels, 1, kernel_size=3, padding=1),
    ]
)
Decoder = nn.Sequential(*decoder_modules)

model = SpatialVAE(Encoder, mean_layer, logvar_layer, Decoder).to(device)
init_weights(model, act())
summary(model, (1, 1, domain_size, domain_size), depth=4)

# ------------------------ instantiate optimizer -------------------------
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=epochs, eta_min=lr * 1e-2
)

# ------------------------------- training -------------------------------
print_every = 10
train_cost = [0] * epochs
val_cost = [0] * epochs
best_val_cost = float("inf")
best_epoch = 0
best_model_state = None
best_model_path = (
    BASE_DIR
    / f"../../models/fiber_vae_spatial_best_d{depth}_{latent_channels}_{beta}_{domain_size}.pt2"
)
final_model_path = (
    BASE_DIR
    / f"../../models/fiber_vae_spatial_final_d{depth}_{latent_channels}_{beta}_{domain_size}.pt2"
)

start_time = time.perf_counter()
pbar = tqdm(range(epochs), desc="Training: ", ncols=90)
for epoch in pbar:
    model.train()
    for x in train_loader:
        x_target = x[0].to(device)
        x = standardizex(x_target)
        optimizer.zero_grad()
        x_pred, mean_pred, logvar_pred = model(x)
        cost = cost_fun(x_pred, mean_pred, logvar_pred, x_target, beta)
        cost.backward()
        optimizer.step()
        train_cost[epoch] += cost.item()
    train_cost[epoch] /= len(train_loader)
    if scheduler is not None:
        scheduler.step()

    model.eval()
    with torch.no_grad():
        for x in val_loader:
            x_target = x[0].to(device)
            x = standardizex(x_target)
            x_pred, mean_pred, logvar_pred = model(x)
            cost = cost_fun(x_pred, mean_pred, logvar_pred, x_target, beta)
            val_cost[epoch] += cost.item()
        val_cost[epoch] /= len(val_loader)
        if val_cost[epoch] < best_val_cost:
            best_val_cost = val_cost[epoch]
            best_epoch = epoch + 1
            best_model_state = copy.deepcopy(model.state_dict())
            model.standardizer = standardizex
            torch.save(model, best_model_path)

    if epoch % print_every == 0:
        pbar.set_postfix(
            {"train": f"{train_cost[epoch]:.2e}", "val": f"{val_cost[epoch]:.2e}"}
        )
training_time = time.perf_counter() - start_time

# ----------------------------- export model -----------------------------
model.standardizer = standardizex
torch.save(model, final_model_path)

# ---------------------------- postprocessing ----------------------------
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots()
ax.plot(train_cost, "k")
ax.plot(val_cost, "r")
ax.set_yscale("log")
ax.set_xlabel("epoch")
ax.set_ylabel("loss")
ax.legend(["train", "validation"])
fig.tight_layout()
plt.savefig(
    RESULTS_DIR
    / f"fiber_vae_spatial_loss_{latent_channels}_{beta}_{domain_size}_{run_id}.png",
    bbox_inches="tight",
)
plt.show()

model.eval()
if best_model_state is not None:
    model.load_state_dict(best_model_state)
    model.eval()

with torch.no_grad():
    x_target = next(iter(val_loader))[0].to(device)
    x = standardizex(x_target)
    x_pred, mean_pred, logvar_pred = model(x)
    x_prob = torch.sigmoid(x_pred)
    x_binary = (x_prob >= 0.5).to(x_target.dtype)
    val_metrics = reconstruction_metrics(x_prob, x_target)
    val_kl = kl_div(mean_pred, logvar_pred).item()

fig, ax = plt.subplots(1, 3, figsize=(6, 2), dpi=domain_size)
ax[0].imshow(x_target.cpu()[0, 0], cmap="binary", vmin=0, vmax=1)
ax[1].imshow(x_prob.detach().cpu()[0, 0], cmap="binary", vmin=0, vmax=1)
ax[2].imshow(x_binary.detach().cpu()[0, 0], cmap="binary", vmin=0, vmax=1)
for i in range(3):
    ax[i].set_aspect("equal")
    ax[i].axis("off")
    ax[i].set_rasterized(True)
fig.tight_layout(pad=0)
plt.savefig(
    RESULTS_DIR
    / (
        f"fiber_vae_spatial_reconstruction_{latent_channels}_{beta}_"
        f"{domain_size}_{run_id}.png"
    ),
    bbox_inches="tight",
    pad_inches=0,
)
plt.show()

history = np.column_stack(
    [np.arange(1, epochs + 1), np.asarray(train_cost), np.asarray(val_cost)]
)
np.savetxt(
    RESULTS_DIR
    / f"fiber_vae_spatial_history_{latent_channels}_{beta}_{domain_size}_{run_id}.csv",
    history,
    delimiter=",",
    header="epoch,train_loss,val_loss",
    comments="",
)

summary_path = (
    RESULTS_DIR
    / f"fiber_vae_spatial_summary_{latent_channels}_{beta}_{domain_size}_{run_id}.txt"
)
with open(summary_path, "w", encoding="utf-8") as f:
    f.write("fiber spatial VAE training summary\n")
    f.write(f"run_id: {run_id}\n")
    f.write(f"architecture: {architecture}\n")
    f.write(f"training_time_seconds: {training_time:.2f}\n")
    f.write(f"device: {device}\n")
    if torch.cuda.is_available():
        f.write(f"gpu: {torch.cuda.get_device_name(0)}\n")
    f.write(f"epochs: {epochs}\n")
    f.write(f"batch_size: {batch_size}\n")
    f.write(f"learning_rate: {lr}\n")
    f.write(f"weight_decay: {weight_decay}\n")
    f.write(f"beta: {beta}\n")
    f.write(f"base_channels: {base_channels}\n")
    f.write(f"latent_channels: {latent_channels}\n")
    f.write(f"depth: {depth}\n")
    f.write(f"latent_spatial_size: {domain_size // 2**depth}\n")
    f.write("upsampling: nearest\n")
    f.write("reconstruction_loss: BCEWithLogitsLoss\n")
    f.write(f"final_train_loss: {train_cost[-1]:.8e}\n")
    f.write(f"final_val_loss: {val_cost[-1]:.8e}\n")
    f.write(f"best_val_loss: {best_val_cost:.8e}\n")
    f.write(f"best_val_epoch: {best_epoch}\n")
    f.write(f"best_model_path: {best_model_path.resolve()}\n")
    f.write(f"final_model_path: {final_model_path.resolve()}\n")
    f.write(f"validation_kl: {val_kl:.8e}\n")
    for name, value in val_metrics.items():
        f.write(f"validation_{name}: {value:.8e}\n")
