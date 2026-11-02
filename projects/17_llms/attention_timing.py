import time

import torch
from torch import nn

from transformer_from_scratch import EfficientMultiHeadAttention, MultiHeadAttention

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------------- settings -------------------------------------
D_IN, D_OUT = 64, 64
D_HEAD, HEADS = 8, 8  # HEADS * D_HEAD = D_OUT
N = 100  # sequence length
REPEATS = 100
ROUNDS = 20

# -------------------------------------- timing ---------------------------------------
X = torch.randn(N, D_IN, device=device)
torch_attention = nn.MultiheadAttention(D_IN, HEADS).to(device)  # requires D_OUT = D_IN
attentions = {
    "simple": MultiHeadAttention(D_IN, D_OUT, D_HEAD, HEADS).to(device),
    "efficient": EfficientMultiHeadAttention(D_IN, D_OUT, D_HEAD, HEADS).to(device),
    "pytorch": lambda x: torch_attention(x, x, x, need_weights=False)[0],
}
best = {name: float("inf") for name in attentions}

with torch.no_grad():
    for _ in range(ROUNDS):  # interleaved rounds, best of each, against load spikes
        for name, attention in attentions.items():
            attention(X)  # warm-up
            if device.type == "cuda":
                torch.cuda.synchronize()  # wait for asynchronous kernels
            tic = time.perf_counter()
            for _ in range(REPEATS):
                attention(X)
            if device.type == "cuda":
                torch.cuda.synchronize()
            toc = time.perf_counter()
            best[name] = min(best[name], (toc - tic) / REPEATS)

for name, seconds in best.items():
    print(f"{name} attention on {device.type}: {seconds:.2e} s")
print(f"speedup {best['simple'] / best['efficient']:.1f}x")
