from pathlib import Path

from NN import DCN, MLP, VAE
from torch.utils.data import TensorDataset, DataLoader, random_split
from DL import init_weights, Standardizer
import torch
from torch import nn
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).parent
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True

# -------------------------- training settings ---------------------------
epochs = 4000 #2000
lr = 8e-3 #4e-3 #5e-4 #1e-3 #4e-3 #1e-2
weight_decay = 0.03 #1e-3 #0 #1e-2
batch_size = 32 # 64
# kl_weight = 0 #1e-4 #1e-3 #1e-4 #1e-3 #1e-2 #1e-3 #0.1
# kl_anneal_epochs = 1000
# kl_weight_final = 1e-2 #1e-3 # TODO try 1e-2
# reconstruction_loss = nn.MSELoss(reduction='mean') #use sum here
# def cost_fun(x_pred, mean_pred, logvar_pred, x, epoch):
#     mse = reconstruction_loss(x_pred, x)
#     var_pred = torch.exp(logvar_pred)
#     kl = -0.5 * torch.mean(1 + logvar_pred - mean_pred.pow(2) - var_pred) # AND USE SUM HERE
#
#     # linear annealing
#     beta = min(epoch / kl_anneal_epochs, 1.0) * kl_weight_final
#     # print(f"MSE: {mse:.4f}, KL: {kl:.4f}, KL weighted: {kl * beta:.4f}")
#     return mse + beta * kl


reconstruction_loss = nn.MSELoss(reduction='sum')
kl_anneal_epochs = 1000 #2000
kl_weight_final = 1. #1e-1 #1e-2
def cost_fun(x_pred, mean_pred, logvar_pred, x, epoch):
    mse = reconstruction_loss(x_pred, x)
    var_pred = torch.exp(logvar_pred)
    kl = -0.5 * torch.sum(1 + logvar_pred - mean_pred.pow(2) - var_pred)
    beta = min(epoch / kl_anneal_epochs, 1.0) * kl_weight_final
    return mse + beta * kl, mse.detach(), kl.detach()

# ---------------------------- model settings ----------------------------
# data
seq_len, channel_dim = 128, 3

# base, depth, latent_dim = 2, 5, 16 #32
base, depth, latent_dim = 2, 3, 16 #4/5, 32
kernel_size = 3
act = nn.GELU(approximate='tanh')
# ----------------------------- prepare data -----------------------------
data = torch.from_numpy(np.load(BASE_DIR / f'../../data/normal_3dof_{seq_len}.npy')).to(torch.float32).to(device)

clip = 1024 #2048 #1024 #512 #512 #128
data = data[:clip]

dataset = TensorDataset(data)
train_data, val_data = random_split(dataset, [0.9, 0.1])
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=len(val_data), shuffle=True) # full batch

X_train = train_data.dataset.tensors[0][train_data.indices]
standardizex = Standardizer(X_train, dim=(0,2))
# -------------------------- instantiate model ---------------------------
channels = [channel_dim * (base ** (i)) for i in range(depth + 1)]
encoder_layers = [seq_len * channel_dim, 2 * latent_dim]
decoder_layers = [seq_len * channel_dim, latent_dim]
upsamplings = [nn.Upsample(scale_factor=2, # alternative 'nearest'
                           mode='nearest')] * (len(channels) - 1) # linear

Encoder = nn.Sequential()
Encoder.append(DCN(channels,
                   [act for _ in range(len(channels) - 1)],
                   kernel_size, stride=2, padding=1, dim=1,
                   # normalizations=[nn.BatchNorm1d(channels[i]) for i in range(1, len(channels))]))
                   normalizations = [nn.LayerNorm([channels[i], seq_len // 2**i]) for i in range(1, len(channels) - 1)]))
Encoder.append(nn.Flatten())
Encoder.append(MLP(encoder_layers, [act for _ in range(len(encoder_layers) - 2)]))
# alternative could be nn.LazyLinear to compute automatically

Decoder = nn.Sequential()
Decoder.append(MLP(decoder_layers[::-1],
                   [act for _ in range(len(decoder_layers) - 1)]))
Decoder.append(nn.Unflatten(1, (channels[-1], -1)))
Decoder.append(DCN(channels[::-1],
                   [act for _ in range(len(channels) - 2)],
                   kernel_size, stride=1, padding=1, dim=1,
                   resamplings=upsamplings,
                   # normalizations=[nn.BatchNorm1d(channels[-i]) for i in range(2, len(channels))]))
                   normalizations=[None] + [nn.LayerNorm([channels[i], seq_len // 2 ** i]) for i in range(len(channels) - 3, 0, -1)]))

model = VAE(Encoder, Decoder).to(device)
init_weights(model, act)

# ------------------------ instantiate optimizer -------------------------
# optimizer = torch.optim.Adam(model.parameters(), lr=lr,
#                               weight_decay=weight_decay)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                              weight_decay=weight_decay)
# scheduler = None
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                       T_max=epochs,
                                                       eta_min=lr*1e-2)

# ------------------------------- training -------------------------------
print_every = 10
train_cost, train_mse, train_kl = [0] * epochs, [0] * epochs, [0] * epochs
val_cost, val_mse, val_kl = [0] * epochs, [0] * epochs, [0] * epochs
pbar = tqdm(range(epochs), desc='Training: ', ncols=90)
for epoch in pbar:
    model.train()
    for x in train_loader:
        x = standardizex(x[0]) # unwrap & standardize
        optimizer.zero_grad()
        x_pred, mean_pred, logvar_pred = model(x)
        cost, mse, kl = cost_fun(x_pred, mean_pred, logvar_pred, x, epoch)
        cost.backward()
        optimizer.step()
        train_cost[epoch] += cost.item()
        train_mse[epoch] += mse.item()
        train_kl[epoch] += kl.item()
    train_cost[epoch] /= len(train_data)  # avg over data
    train_mse[epoch] /= len(train_data)
    train_kl[epoch] /= len(train_data)
    if scheduler is not None:
        scheduler.step()

    model.eval()
    with torch.no_grad():
        for x in val_loader:
            x = standardizex(x[0]) # unwrap & standardize
            x_pred, mean_pred, logvar_pred = model(x)
            cost, mse, kl = cost_fun(x_pred, mean_pred, logvar_pred, x, epoch)
            val_cost[epoch] += cost.item()
            val_mse[epoch] += mse.item()
            val_kl[epoch] += kl.item()
        val_cost[epoch] /= len(val_data)  # avg over data
        val_mse[epoch] /= len(val_data)
        val_kl[epoch] /= len(val_data)
    if epoch % print_every == 0:
        pbar.set_postfix({
            #'train': f'{train_cost[epoch]:.2e}, (mse) {train_mse[epoch]:.2e}, (kl) {train_kl[epoch]:.2e}',
            'train': f'(mse) {train_mse[epoch]: .2e}, (kl) {train_kl[epoch]:.2e}',
            # 'val': f'{val_cost[epoch]:.2e}'
        })

model.standardizer = standardizex # just for saving
torch.save(model, BASE_DIR / f'../../models/vae_3dof_{seq_len}.pt2')

# --------------------------- post-processing ----------------------------
print(f'training {train_cost[-1]:.2e}, (mse) {train_mse[-1]:.2e} (kl) {train_kl[-1]:.2e}')
print(f'validation {val_cost[-1]:.2e}, (mse) {val_mse[-1]:.2e} (kl) {val_kl[-1]:.2e}')
model.eval()
t = np.linspace(0, 1, seq_len) # normalized time (T=12.8)
sample = 0
dof = 2
for sample in range(4):
    x_train = next(iter(train_loader))[0]
    x_val = next(iter(val_loader))[0]
    model.eval()
    with torch.no_grad():
        x_train_pred, _, _ = model(standardizex(x_train))
        x_val_pred, _, _ = model(standardizex(x_val))
        x_train_pred = standardizex.inverse(x_train_pred).cpu()[sample,dof]
        x_val_pred = standardizex.inverse(x_val_pred).cpu()[sample,dof]
    x_train = x_train.cpu()[sample,dof]
    x_val = x_val.cpu()[sample,dof]

    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    ax2.plot(np.linspace(0,1,len(train_cost)), train_cost,'r', alpha=0.2)
    ax2.plot(np.linspace(0,1,len(train_mse)), train_mse, color='purple', alpha=0.2)
    ax2.plot(np.linspace(0,1,len(train_kl)), train_kl, color='orange', alpha=0.2)

    ax2.plot(np.linspace(0,1,len(val_cost)), val_cost,'b', alpha=0.2)
    ax2.set_yscale('log')

    ax.plot(t, x_train, 'r', alpha=0.5)
    ax.plot(t, x_train_pred, 'r--')

    ax.plot(t, x_val, 'b', alpha=0.5)
    ax.plot(t, x_val_pred, 'b--')

    plt.show()