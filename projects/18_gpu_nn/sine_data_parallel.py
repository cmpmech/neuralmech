import time

import matplotlib.pyplot as plt
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from DL import init_weights
from NN import MLP

# one process per GPU: torchrun --nproc_per_node=2 sine_data_parallel.py
dist.init_process_group("nccl")
rank = dist.get_rank()
device = torch.device("cuda", rank)
torch.cuda.set_device(device)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True

# -------------------------------------- settings -------------------------------------
# hyperparameters
EPOCHS = 100
LR = 1e-2
BATCH_SIZE = 32  # per GPU

# define loss
cost_fun = nn.MSELoss(reduction="mean")

# model settings
LAYERS = [1, 24, 24, 24, 1]
ACTIVATIONS = [nn.GELU(approximate="tanh") for _ in range(len(LAYERS) - 2)]

# data
SAMPLES = 1024
NOISE = 0.2

# ------------------------------------ create data ------------------------------------
X = 2 * torch.rand(SAMPLES, 1) - 1
Y = torch.sin(2 * torch.pi * X) + NOISE * (2 * torch.rand(SAMPLES, 1) - 1)
dataset = TensorDataset(X, Y)
sampler = DistributedSampler(dataset)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler)

# --------------------------- instantiate model & optimizer ---------------------------
model = MLP(LAYERS, post_modules=ACTIVATIONS).to(device)
init_weights(model, ACTIVATIONS[0])
model = DistributedDataParallel(model, device_ids=[rank])
optimizer = torch.optim.Adam(model.parameters(), LR)

# -------------------------------------- training -------------------------------------
tic = time.time()
pbar = tqdm(range(EPOCHS), disable=rank != 0)
for epoch in pbar:
    sampler.set_epoch(epoch)  # reshuffles the shares every epoch
    train_cost = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        cost = cost_fun(model(x), y)
        cost.backward()
        optimizer.step()
        train_cost += cost.item()
    pbar.set_postfix({"train": f"{train_cost / len(train_loader):.2e}"})
toc = time.time()

# ----------------------------------- postprocessing ----------------------------------
if rank == 0:
    print(f"elapsed time {toc - tic:.2f} s")
    x_test = torch.linspace(-1, 1, 100).unsqueeze(1)
    model.eval()
    with torch.no_grad():
        y_pred_test = model(x_test.to(device)).cpu()

    fig, ax = plt.subplots()
    ax.plot(x_test, torch.sin(2 * torch.pi * x_test), "k")
    ax.plot(X, Y, "ko", markersize=2)
    ax.plot(x_test, y_pred_test, "r--")
    plt.show()

dist.destroy_process_group()
