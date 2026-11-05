import argparse
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from NN import DRNN, TransformerDecoder
from postprocessing import save_csv
from transformer_from_scratch import TransformerBlock

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
CSV_DIR = (RESULTS_DIR / "data").resolve()
RGB_PNG_DIR = (RESULTS_DIR / "rgb_png").resolve()

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
VAL_BATCH = 100  # validation in chunks to bound memory at long sequences

# define loss
cost_fun = nn.CrossEntropyLoss()

# task settings
CLASSES = 8  # data tokens 1..CLASSES, 0 is noise, CLASSES + 1 is marker
TOKENS = 8  # data tokens to copy
LENGTHS = [16, 32, 64, 128, 256, 512]  # data region; markers follow
ATTENTION_LENGTH = 128  # sequence length of attention plot
MARKER = CLASSES + 1

# model settings
D = 64
HEADS = 4
STATES = 64  # S4D modes per channel
MAMBA_STATES = 16


# --------------------------------------- helper --------------------------------------
def create_batch(samples, length, selective, generator):
    x = torch.zeros(samples, length + TOKENS, dtype=torch.long)
    data = torch.randint(1, CLASSES + 1, (samples, TOKENS), generator=generator)
    if selective:
        ids = torch.rand(samples, length, generator=generator).argsort(-1)[:, :TOKENS]
        ids = ids.sort(-1).values
    else:
        ids = torch.arange(TOKENS).expand(samples, TOKENS)
    x.scatter_(1, ids, data)
    x[:, length:] = MARKER
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
        mixer = {"s4d": lambda: S4D(D, STATES), "mamba": lambda: Mamba(D, MAMBA_STATES)}
        self.blocks = nn.Sequential(
            *[TransformerBlock(D, mixer[kind]()) for _ in range(2)]
        )
        self.projection = nn.Linear(D, MARKER + 1)

    def forward(self, tokens):
        x = self.embedding(tokens)
        if self.kind == "lstm":
            return self.body(x)
        return self.projection(self.blocks(x))


def create_model(kind):
    if kind == "transformer":
        return TransformerDecoder(MARKER + 1, D, HEADS, blocks=2)
    return SequenceModel(kind)


def attention_weights(model, tokens):
    """attention of the second block, averaged over the heads."""
    mask = nn.Transformer.generate_square_subsequent_mask(
        tokens.shape[1], device=tokens.device
    )
    first, second = model.blocks.layers
    x = first(model.embed(tokens), src_mask=mask, is_causal=True)
    return second.self_attn(x, x, x, attn_mask=mask, need_weights=True)[1]


# -------------------------------------- training -------------------------------------
KINDS = ["lstm", "s4d", "mamba", "transformer"]
LEVELS = {"copying": False, "selective": True}
accuracy = {}
attention_models = {}
attention_inputs = {}
for length in LENGTHS:
    for level, selective in LEVELS.items():
        X_val, Y_val, _ = create_batch(
            VAL_SAMPLES, length, selective, torch.Generator().manual_seed(1)
        )
        for kind in KINDS:
            torch.manual_seed(0)
            model = create_model(kind).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), LR)
            generator = torch.Generator().manual_seed(0)

            tic = time.time()
            pbar = tqdm(range(EPOCHS), desc=f"{length} {level} {kind}")
            for epoch in pbar:
                model.train()
                x, y, _ = create_batch(BATCH_SIZE, length, selective, generator)
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                y_pred = model(x)[:, length:]
                cost = cost_fun(y_pred.reshape(-1, MARKER + 1), y.reshape(-1))
                cost.backward()
                optimizer.step()
                pbar.set_postfix({"cost": f"{cost.item():.2e}"})

            model.eval()
            correct = 0
            with torch.no_grad():
                for x, y in zip(X_val.split(VAL_BATCH), Y_val.split(VAL_BATCH)):
                    y_pred = model(x.to(device))[:, length:].argmax(-1).cpu()
                    correct += (y_pred == y).sum().item()
            accuracy[length, level, kind] = correct / Y_val.numel()
            toc = time.time()
            print(f"elapsed time {toc - tic:.2f} s")
            if (length, kind) == (ATTENTION_LENGTH, "transformer"):
                attention_models[level] = model
                attention_inputs[level] = X_val[:1].to(device)

# ----------------------------------- postprocessing ----------------------------------
for (length, level, kind), value in accuracy.items():
    print(f"{length} {level} {kind}: accuracy {value:.3f}")

attention = {}
with torch.no_grad():
    for level in LEVELS:
        weights = attention_weights(attention_models[level], attention_inputs[level])
        attention[level] = weights[0, ATTENTION_LENGTH:].cpu().numpy()

if not args.book:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, level in zip(axes, LEVELS):
        for kind in KINDS:
            values = [accuracy[length, level, kind] for length in LENGTHS]
            ax.semilogx(LENGTHS, values, "o-", base=2)
    plt.show()

    fig, axes = plt.subplots(2, 1, figsize=(10, 4))
    for ax, level in zip(axes, LEVELS):
        ax.imshow(attention[level], cmap="cividis", aspect="auto", interpolation="none")
    plt.show()
# -------------------------------- book postprocessing --------------------------------
else:
    for level in LEVELS:
        save_csv(
            CSV_DIR / f"selective_copying_{level}.csv",
            length=LENGTHS,
            **{k: [accuracy[n, level, k] for n in LENGTHS] for k in KINDS},
        )
    for level in LEVELS:
        fig, ax = plt.subplots(figsize=(13.6, 1.6))
        ax.imshow(
            attention[level],
            cmap="cividis",
            vmin=0,
            vmax=1,
            aspect="auto",
            interpolation="none",
        )
        ax.axis("off")
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        fig.savefig(RGB_PNG_DIR / f"selective_copying_attention_{level}.png", dpi=200)
        plt.close(fig)
        save_csv(
            CSV_DIR / f"selective_copying_tokens_{level}.csv",
            position=np.arange(ATTENTION_LENGTH + TOKENS) + 1,
            token=attention_inputs[level][0].cpu().numpy(),
        )
