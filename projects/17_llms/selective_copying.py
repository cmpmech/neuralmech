import argparse
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from DL import sinusoidal_embedding
from NN import DRNN
from postprocessing import save_csv
from transformer_from_scratch import EfficientMultiHeadAttention, TransformerBlock

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
CSV_DIR = (RESULTS_DIR / "data").resolve()

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
# hyperparameters
EPOCHS = 4000  # one freshly drawn batch per epoch
LR = 1e-3
BATCH_SIZE = 64
VAL_SAMPLES = 1000
EVAL_EVERY = 50

# define loss
cost_fun = nn.CrossEntropyLoss()

# task settings
CLASSES = 8  # data tokens 1..CLASSES, 0 is noise, CLASSES + 1 is marker
TOKENS = 8  # data tokens to copy
LENGTH = 128  # data region; markers follow
MARKER = CLASSES + 1

# model settings
D = 64
HEADS = 4
STATES = 64  # S4D modes per channel
MAMBA_STATES = 16


# --------------------------------------- helper --------------------------------------
def create_batch(samples, selective, generator):
    x = torch.zeros(samples, LENGTH + TOKENS, dtype=torch.long)
    data = torch.randint(1, CLASSES + 1, (samples, TOKENS), generator=generator)
    if selective:
        ids = torch.rand(samples, LENGTH, generator=generator).argsort(-1)[:, :TOKENS]
        ids = ids.sort(-1).values
    else:
        ids = torch.arange(TOKENS).expand(samples, TOKENS)
    x.scatter_(1, ids, data)
    x[:, LENGTH:] = MARKER
    return x, data, ids


class S4D(nn.Module):
    def __init__(self, D, states):
        super().__init__()
        log_dt = torch.rand(D) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
        self.log_dt = nn.Parameter(log_dt)
        self.log_A_real = nn.Parameter(torch.log(0.5 * torch.ones(D, states)))
        self.A_imag = nn.Parameter(math.pi * torch.arange(states).repeat(D, 1).float())
        self.C = nn.Parameter(torch.randn(D, states, dtype=torch.cfloat) * 0.5**0.5)
        self.skip = nn.Parameter(torch.randn(D))
        self.W_out = nn.Linear(D, D)

    def kernel(self, L):
        dtA = torch.exp(self.log_dt)[:, None] * (
            -torch.exp(self.log_A_real) + 1j * self.A_imag
        )
        # zero-order hold with B = 1 folded into C
        C = self.C * (torch.exp(dtA) - 1) / dtA * torch.exp(self.log_dt)[:, None]
        powers = torch.exp(dtA[..., None] * torch.arange(L, device=dtA.device))
        return 2 * torch.einsum("dn,dnl->dl", C, powers).real

    def forward(self, x):
        L = x.shape[1]
        u = x.transpose(1, 2)
        K = self.kernel(L)
        y = torch.fft.irfft(torch.fft.rfft(u, 2 * L) * torch.fft.rfft(K, 2 * L), 2 * L)
        y = y[..., :L] + self.skip[:, None] * u
        return self.W_out(nn.functional.gelu(y.transpose(1, 2)))


class Mamba(nn.Module):
    def __init__(self, D, states, expand=2, conv=4):
        super().__init__()
        E = expand * D
        self.W_in = nn.Linear(D, 2 * E)
        self.conv = nn.Conv1d(E, E, conv, groups=E, padding=conv - 1)
        self.W_delta = nn.Linear(E, E)
        dt = torch.exp(
            torch.rand(E) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
        )
        with torch.no_grad():
            self.W_delta.weight.mul_(0.1)
            self.W_delta.bias.copy_(
                dt + torch.log(-torch.expm1(-dt))
            )  # inverse softplus
        self.W_B = nn.Linear(E, states, bias=False)
        self.W_C = nn.Linear(E, states, bias=False)
        A = torch.arange(1, states + 1).float().repeat(E, 1)
        self.log_A = nn.Parameter(torch.log(A))
        self.skip = nn.Parameter(torch.ones(E))
        self.W_out = nn.Linear(E, D)

    def forward(self, x):
        L = x.shape[1]
        x, z = self.W_in(x).chunk(2, dim=-1)
        x = self.conv(x.transpose(1, 2))[..., :L].transpose(1, 2)
        x = nn.functional.silu(x)

        # input-dependent step size, input and output matrices
        delta = nn.functional.softplus(self.W_delta(x))
        B = self.W_B(x)
        C = self.W_C(x)
        A_bar = torch.exp(delta[..., None] * -torch.exp(self.log_A))
        Bx = delta[..., None] * B[:, :, None] * x[..., None]
        h = torch.zeros_like(Bx[:, 0])
        ys = []
        for A_tau, Bx_tau, C_tau in zip(A_bar.unbind(1), Bx.unbind(1), C.unbind(1)):
            h = A_tau * h + Bx_tau
            ys.append((h * C_tau[:, None]).sum(-1))
        y = torch.stack(ys, dim=1) + self.skip * x
        return self.W_out(y * nn.functional.silu(z))


class SequenceModel(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        self.embedding = nn.Embedding(MARKER + 1, D)
        if kind == "lstm":
            self.body = DRNN([D, D, D, MARKER + 1], cell=nn.LSTM)
            return
        mixers = {
            "transformer": lambda: EfficientMultiHeadAttention(D, D, D // HEADS, HEADS),
            "s4d": lambda: S4D(D, STATES),
            "mamba": lambda: Mamba(D, MAMBA_STATES),
        }
        self.blocks = nn.Sequential(
            *[TransformerBlock(D, mixers[kind]()) for _ in range(2)]
        )
        self.projection = nn.Linear(D, MARKER + 1)

    def forward(self, tokens):
        x = self.embedding(tokens)
        if self.kind == "lstm":
            return self.body(x)
        if self.kind == "transformer":
            positions = torch.arange(tokens.shape[1], device=tokens.device)
            x = x + sinusoidal_embedding(positions, D)
        return self.projection(self.blocks(x))


def attention_weights(model, tokens):
    x = model.embedding(tokens)
    x = x + sinusoidal_embedding(torch.arange(tokens.shape[1], device=tokens.device), D)
    x = model.blocks[0](x)
    attention = model.blocks[1].attention
    shape = (*x.shape[:-1], HEADS, D // HEADS)
    queries = attention.W_query(x).view(shape).transpose(-3, -2)
    keys = attention.W_key(x).view(shape).transpose(-3, -2)
    scores = queries @ keys.transpose(-2, -1) / (D // HEADS) ** 0.5
    return torch.softmax(scores, dim=-1).mean(-3)


# ------------------------------------ create data ------------------------------------
KINDS = ["lstm", "s4d", "mamba", "transformer"]
LEVELS = {"copying": False, "selective": True}
val_data = {
    level: [
        t.to(device)
        for t in create_batch(VAL_SAMPLES, s, torch.Generator().manual_seed(1))
    ]
    for level, s in LEVELS.items()
}

# -------------------------------------- training -------------------------------------
history = {}
models = {}
for level, selective in LEVELS.items():
    for kind in KINDS:
        torch.manual_seed(0)
        model = SequenceModel(kind).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), LR)
        generator = torch.Generator().manual_seed(0)
        parameters = sum(p.numel() for p in model.parameters())
        X_val, Y_val, _ = val_data[level]

        val_accuracy = []
        tic = time.time()
        pbar = tqdm(range(EPOCHS), desc=f"{level} {kind} ({parameters} parameters)")
        for epoch in pbar:
            model.train()
            x, y, _ = create_batch(BATCH_SIZE, selective, generator)
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            y_pred = model(x)[:, LENGTH:]
            cost = cost_fun(y_pred.reshape(-1, MARKER + 1), y.reshape(-1))
            cost.backward()
            optimizer.step()

            if epoch % EVAL_EVERY == 0 or epoch == EPOCHS - 1:
                model.eval()
                with torch.no_grad():
                    y_pred = model(X_val)[:, LENGTH:].argmax(-1)
                val_accuracy.append((y_pred == Y_val).float().mean().item())
                pbar.set_postfix(
                    {"cost": f"{cost.item():.2e}", "acc": f"{val_accuracy[-1]:.3f}"}
                )
        toc = time.time()
        print(f"elapsed time {toc - tic:.2f} s")
        history[level, kind] = val_accuracy
        models[level, kind] = model

# ----------------------------------- postprocessing ----------------------------------
epochs = [e for e in range(EPOCHS) if e % EVAL_EVERY == 0] + [EPOCHS - 1]
for (level, kind), accuracy in history.items():
    print(f"{level} {kind}: final accuracy {accuracy[-1]:.3f}")

x = val_data["selective"][0][:1]
with torch.no_grad():
    attention = attention_weights(models["selective", "transformer"], x)[0, LENGTH:]
attention = attention.cpu().numpy()

if not args.book:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for (level, kind), accuracy in history.items():
        axes[list(LEVELS).index(level)].plot(epochs, accuracy)
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 2))
    ax.imshow(attention, cmap="cividis", aspect="auto", interpolation="none")
    plt.show()
# -------------------------------- book postprocessing --------------------------------
else:
    for level in LEVELS:
        save_csv(
            CSV_DIR / f"selective_copying_{level}.csv",
            epoch=epochs,
            **{kind: history[level, kind] for kind in KINDS},
        )
    i, j = np.meshgrid(np.arange(TOKENS), np.arange(LENGTH + TOKENS), indexing="ij")
    save_csv(
        CSV_DIR / "selective_copying_attention.csv",
        marker=i.flatten() + 1,
        position=j.flatten() + 1,
        weight=attention.flatten(),
    )
    save_csv(
        CSV_DIR / "selective_copying_tokens.csv",
        position=np.arange(LENGTH + TOKENS) + 1,
        token=x[0].cpu().numpy(),
    )
