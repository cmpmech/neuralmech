import copy
from functools import partial
from pathlib import Path
from datetime import datetime
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
        if isinstance(m, (nn.Linear, nn.Conv2d)):
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

    def inverse(self, x):
        return x * self.x_std + self.x_mean


def build_ae_cnn_config(depth, conv_layers, channel_dim, base):
    channels, strides = [], []
    for i in range(depth + 1):
        channels.append(channel_dim * (base**i))
        strides.append(1)
        if i < depth:
            for _ in range(conv_layers):
                channels.append(channel_dim * (base ** (i + 1)))
                strides.append(1)
            strides[-1] = 2
    return channels, strides[:-1]


def get_layer_param(param, i):
    return param[i] if isinstance(param, list) else param


class MLP(nn.Module):
    def __init__(self, layers, activations=None):
        super().__init__()
        modules = []
        for i in range(len(layers) - 1):
            modules.append(nn.Linear(layers[i], layers[i + 1]))
            if activations and i < len(activations) and activations[i]:
                modules.append(activations[i])
        self.model = nn.Sequential(*modules)

    def forward(self, x):
        return self.model(x)


class DCN(nn.Module):
    def __init__(
        self,
        channels,
        activations,
        kernel_size,
        stride,
        padding,
        normalizations=None,
        resamplings=None,
    ):
        super().__init__()
        normalizations = normalizations or []
        resamplings = resamplings or []
        modules = []
        for i in range(len(channels) - 1):
            if resamplings and i < len(resamplings) and resamplings[i]:
                modules.append(resamplings[i])
            modules.append(
                nn.Conv2d(
                    channels[i],
                    channels[i + 1],
                    get_layer_param(kernel_size, i),
                    get_layer_param(stride, i),
                    get_layer_param(padding, i),
                    bias=False,
                )
            )
            if normalizations and i < len(normalizations) and normalizations[i]:
                modules.append(normalizations[i])
            if activations and i < len(activations) and activations[i]:
                modules.append(activations[i])
        self.model = nn.Sequential(*modules)

    def forward(self, x):
        return self.model(x)


class VAE(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encode = encoder
        self.decode = decoder

    def reparameterize(self, mean, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mean + eps * std
        return mean

    def forward(self, x):
        distributions = self.encode(x)
        mean, logvar = torch.chunk(distributions, chunks=2, dim=1)
        z = self.reparameterize(mean, logvar)
        y = self.decode(z)
        return y, mean, logvar

# -------------------------- training settings ---------------------------
epochs = 25
lr = 1e-3
weight_decay = 1e-2
batch_size = 32
beta = 0.0

# define loss
recon_loss = nn.BCEWithLogitsLoss(reduction="mean")


def kl_div(mean_pred, logvar_pred):
    var_pred = torch.exp(logvar_pred)
    kl = 0.5 * torch.mean(var_pred + mean_pred**2 - logvar_pred - 1)
    return kl


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

base, depth = 2, 4
latent_dim = 512
conv_layers = 1
encoder_channel_dim = 2
decoder_channel_dim = 2
kernel_size = 3
act = partial(nn.PReLU, init=0.2)

# ----------------------------- prepare data -----------------------------
domain_size = 256

data = torch.from_numpy(np.load(BASE_DIR / f"../../data/fibers_{domain_size}.npy"))
data = data.to(torch.float32).unsqueeze(1)

dataset = TensorDataset(data)
train_data, val_data = random_split(dataset, [0.9, 0.1])
train_loader = DataLoader(
    train_data, batch_size=batch_size, shuffle=True, drop_last=True
)
val_loader = DataLoader(val_data, batch_size=len(val_data), shuffle=True)  # full batch

X_train = train_data.dataset.tensors[0][train_data.indices]
standardizex = Standardizer(X_train, dim=(0, 2, 3)).to(device)

# -------------------------- instantiate model ---------------------------
encoder_channels, strides = build_ae_cnn_config(
    depth, conv_layers, encoder_channel_dim, base
)
encoder_channels[0] = 1  # true input size

decoder_channels, _ = build_ae_cnn_config(depth, conv_layers, decoder_channel_dim, base)
decoder_channels[0] = 1  # true output size

red_domain_size = domain_size // 2**depth
encoder_layers = [red_domain_size**2 * encoder_channels[-1], latent_dim]
decoder_layers = [latent_dim, red_domain_size**2 * decoder_channels[-1]]

Encoder = nn.Sequential()
Encoder.append(
    DCN(
        encoder_channels,
        [act() for _ in range(len(encoder_channels) - 1)],
        kernel_size,
        stride=strides,
        padding=kernel_size // 2,
        normalizations=[nn.GroupNorm(1, channel) for channel in encoder_channels[1:]],
    )
)
Encoder.append(nn.Flatten())
Encoder.append(MLP(encoder_layers[:-1] + [encoder_layers[-1] * 2], [act()]))

upsamplings = [
    nn.Upsample(scale_factor=2, mode="nearest") if s == 2 else None
    for s in strides[::-1]
]

Decoder = nn.Sequential()
Decoder.append(MLP(decoder_layers, [act()] * (len(decoder_layers) - 1)))
Decoder.append(
    nn.Unflatten(1, (decoder_channels[-1], red_domain_size, red_domain_size))
)
Decoder.append(
    DCN(
        decoder_channels[::-1],
        [act() for _ in range(len(decoder_channels) - 2)],
        kernel_size,
        stride=1,
        padding=kernel_size // 2,
        resamplings=upsamplings,
        normalizations=[
            nn.GroupNorm(1, channel) for channel in decoder_channels[-2:0:-1]
        ],
    )
)

model = VAE(Encoder, Decoder).to(device)
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
    BASE_DIR / f"../../models/fiber_vae_best_{latent_dim}_{beta}_{domain_size}.pt2"
)
start_time = time.perf_counter()
pbar = tqdm(range(epochs), desc="Training: ", ncols=90)
for epoch in pbar:
    model.train()
    for x in train_loader:
        x_target = x[0].to(device)  # unwrap original binary image
        x = standardizex(x_target)  # standardize encoder input
        optimizer.zero_grad()
        x_pred, mean_pred, logvar_pred = model(x)
        cost = cost_fun(x_pred, mean_pred, logvar_pred, x_target, beta)
        cost.backward()
        optimizer.step()
        train_cost[epoch] += cost.item()
    train_cost[epoch] /= len(train_loader)  # avg per batch
    if scheduler is not None:
        scheduler.step()

    model.eval()
    with torch.no_grad():
        for x in val_loader:
            x_target = x[0].to(device)  # unwrap original binary image
            x = standardizex(x_target)  # standardize encoder input
            x_pred, mean_pred, logvar_pred = model(x)
            cost = cost_fun(x_pred, mean_pred, logvar_pred, x_target, beta)
            val_cost[epoch] += cost.item()
        val_cost[epoch] /= len(val_loader)  # avg per batch
        if val_cost[epoch] < best_val_cost:
            best_val_cost = val_cost[epoch]
            best_epoch = epoch + 1
            best_model_state = copy.deepcopy(model.state_dict())
            model.standardizer = standardizex  # just for saving
            torch.save(model, best_model_path)

    if epoch % print_every == 0:
        pbar.set_postfix(
            {"train": f"{train_cost[epoch]:.2e}", "val": f"{val_cost[epoch]:.2e}"}
        )
training_time = time.perf_counter() - start_time

# ----------------------------- export model -----------------------------
model.standardizer = standardizex  # just for saving
torch.save(
    model, BASE_DIR / f"../../models/fiber_vae_final_{latent_dim}_{beta}_{domain_size}.pt2"
)

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
    RESULTS_DIR / f"fiber_vae_loss_{latent_dim}_{beta}_{domain_size}_{run_id}.png",
    bbox_inches="tight",
)
plt.show()

# testing
model.eval()
if best_model_state is not None:
    model.load_state_dict(best_model_state)
    model.eval()

with torch.no_grad():
    x_target = next(iter(val_loader))[0].to(device)
    x = standardizex(x_target)  # standardize encoder input
    x_pred, mean_pred, logvar_pred = model(x)
    x_prob = torch.sigmoid(x_pred)
    val_metrics = reconstruction_metrics(x_prob, x_target)
    val_kl = kl_div(mean_pred, logvar_pred).item()

fig, ax = plt.subplots(1, 2, figsize=(4, 2), dpi=domain_size)
ax[0].imshow(x_target.cpu()[0, 0], cmap="binary", vmin=0, vmax=1)
ax[1].imshow(
    x_prob.detach().cpu()[0, 0], cmap="binary", vmin=0, vmax=1
)
for i in range(2):
    ax[i].set_aspect("equal")
    ax[i].axis("off")
    ax[i].set_rasterized(True)
fig.tight_layout(pad=0)
plt.savefig(
    RESULTS_DIR
    / f"fiber_vae_reconstruction_{latent_dim}_{beta}_{domain_size}_{run_id}.png",
    bbox_inches="tight",
    pad_inches=0,
)
plt.show()

history = np.column_stack(
    [np.arange(1, epochs + 1), np.asarray(train_cost), np.asarray(val_cost)]
)
np.savetxt(
    RESULTS_DIR / f"fiber_vae_history_{latent_dim}_{beta}_{domain_size}_{run_id}.csv",
    history,
    delimiter=",",
    header="epoch,train_loss,val_loss",
    comments="",
)

summary_path = (
    RESULTS_DIR / f"fiber_vae_summary_{latent_dim}_{beta}_{domain_size}_{run_id}.txt"
)
with open(summary_path, "w", encoding="utf-8") as f:
    f.write("fiber VAE training summary\n")
    f.write(f"run_id: {run_id}\n")
    f.write(f"training_time_seconds: {training_time:.2f}\n")
    f.write(f"device: {device}\n")
    if torch.cuda.is_available():
        f.write(f"gpu: {torch.cuda.get_device_name(0)}\n")
    f.write(f"epochs: {epochs}\n")
    f.write(f"batch_size: {batch_size}\n")
    f.write(f"learning_rate: {lr}\n")
    f.write(f"weight_decay: {weight_decay}\n")
    f.write(f"beta: {beta}\n")
    f.write(f"latent_dim: {latent_dim}\n")
    f.write(f"encoder_channel_dim: {encoder_channel_dim}\n")
    f.write(f"decoder_channel_dim: {decoder_channel_dim}\n")
    f.write(f"base: {base}\n")
    f.write(f"depth: {depth}\n")
    f.write(f"conv_layers: {conv_layers}\n")
    f.write(f"kernel_size: {kernel_size}\n")
    f.write("upsampling: nearest\n")
    f.write("reconstruction_loss: BCEWithLogitsLoss\n")
    f.write(f"final_train_loss: {train_cost[-1]:.8e}\n")
    f.write(f"final_val_loss: {val_cost[-1]:.8e}\n")
    f.write(f"best_val_loss: {best_val_cost:.8e}\n")
    f.write(f"best_val_epoch: {best_epoch}\n")
    f.write(f"best_model_path: {best_model_path.resolve()}\n")
    f.write(
        "final_model_path: "
        f"{(BASE_DIR / f'../../models/fiber_vae_final_{latent_dim}_{beta}_{domain_size}.pt2').resolve()}\n"
    )
    f.write(f"validation_kl: {val_kl:.8e}\n")
    for name, value in val_metrics.items():
        f.write(f"validation_{name}: {value:.8e}\n")
