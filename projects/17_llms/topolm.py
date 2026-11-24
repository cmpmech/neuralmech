import argparse
import re
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from scipy.ndimage import gaussian_filter
from scipy.stats import ttest_ind
from torch import nn
from tqdm import tqdm
from transformers import AutoTokenizer

from NN import TransformerDecoder
from postprocessing import save_csv

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
CSV_DIR = (RESULTS_DIR / "data").resolve()
RGB_PDF_DIR = (RESULTS_DIR / "rgb_pdf").resolve()
MODEL_DIR = (BASE_DIR / "../../models").resolve()
DATA_DIR = (BASE_DIR / "../../data").resolve()

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
parser.add_argument("--train", choices=["topographic", "baseline"])  # train only this model, then stop
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
# hyperparameters
STEPS = 8000
LR = 6e-4
WARMUP = 200
BATCH_SIZE = 32
CONTEXT = 128
TOKENS = 30_000_000  # training tokens from the first of two wikitext-103 shards

# define loss
cost_fun = nn.CrossEntropyLoss()
ALPHA = 2.5  # weight of spatial loss per layer, as in TopoLM; 0 for baseline
NEIGHBORHOODS = 5
WINDOW = 11  # side of a square neighborhood on the grid

# model settings
GRID = 24  # units of a layer sit on a GRID x GRID sheet
D = GRID**2
HEADS = 8
BLOCKS = 4

# localizer
SAMPLES = 240  # sentences and non-word lists, as in the fMRI localizer
WORDS = 12  # words per stimulus, and tokens the model reads of it
SMOOTHING = 0.5  # gaussian width in grid units, a light fmri-like readout as in TopoLM
CONSONANTS = "bdfgklmnprstvz"
VOWELS = "aeiou"
rng = np.random.default_rng(0)


# --------------------------------------- helper --------------------------------------
def wikitext(file):
    return pd.read_parquet(hf_hub_download("Salesforce/wikitext", file, repo_type="dataset"))["text"].tolist()


def nonword():
    syllables = [rng.choice(list(CONSONANTS)) + rng.choice(list(VOWELS)) for _ in range(rng.integers(1, 4))]
    return "".join(syllables) + (rng.choice(list(CONSONANTS)) if rng.random() < 0.5 else "")


def standardize(x):
    return (x - x.mean(0)) / (x.std(0) + 1e-6)


def spatial_loss(activations, sheet):
    corner = torch.randint(GRID - WINDOW + 1, (NEIGHBORHOODS, 2))
    rows = (corner[:, :1] + torch.arange(WINDOW))[:, :, None]
    cols = (corner[:, 1:] + torch.arange(WINDOW))[:, None, :]
    x = standardize(activations[:, sheet[rows, cols].flatten(1)].float())  # (tokens, neighborhoods, units)
    r = torch.einsum("tni,tnj->nij", x, x)[:, pairs[0], pairs[1]] / len(x)  # correlation of unit pairs
    fit = (standardize(r.T) * standardize(inverse_distance)[:, None]).mean(0)
    return 0.5 * (1 - fit).mean()


def capture(model):
    outputs, hooks = [], []
    for block in model.blocks.layers:
        hooks.append(block.self_attn.register_forward_hook(lambda m, i, o: outputs.append(o[0])))
        hooks.append(block.linear2.register_forward_hook(lambda m, i, o: outputs.append(o)))
    return outputs, hooks  # attention and feedforward output of every block, before residual and norm


def batch(tokens):
    ids = torch.randint(len(tokens) - CONTEXT - 1, (BATCH_SIZE,), device=device)
    windows = tokens[ids[:, None] + torch.arange(CONTEXT + 1, device=device)].long()
    return windows[:, :-1], windows[:, 1:]


def train(alpha):
    torch.manual_seed(0)
    model = TransformerDecoder(vocabulary=len(tokenizer), D=D, heads=HEADS, blocks=BLOCKS).to(device)
    outputs, hooks = capture(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.1)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min(1, step / WARMUP) * 0.5 * (1 + np.cos(np.pi * step / STEPS)))
    model.train()
    tic = time.perf_counter()
    pbar = tqdm(range(STEPS), mininterval=30)
    for step in pbar:
        x, y = batch(train_tokens)
        outputs.clear()
        with torch.autocast(device.type, dtype=torch.bfloat16):
            task = cost_fun(model(x).flatten(0, 1), y.flatten())
        spatial = sum(spatial_loss(h.flatten(0, 1), sheet) for h, sheet in zip(outputs, sheets)) if alpha > 0 else 0.0
        cost = task + alpha * spatial
        optimizer.zero_grad()
        cost.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        schedule.step()
        if step % 100 == 0:
            pbar.set_postfix(task=f"{task.item():.2e}", spatial=f"{float(spatial):.2e}")
    toc = time.perf_counter()
    print(f"elapsed time {toc - tic:.2f} s")
    for hook in hooks:
        hook.remove()
    return model


@torch.no_grad()
def validate(model):
    torch.manual_seed(1)
    cost = 0.0
    for _ in range(50):
        x, y = batch(val_tokens)
        with torch.autocast(device.type, dtype=torch.bfloat16):
            cost += cost_fun(model(x).flatten(0, 1), y.flatten()).item()
    return cost / 50


@torch.no_grad()
def record(model, texts):
    outputs, hooks = capture(model)
    model.train()  # eval mode takes pytorch's fused fast path, which skips the hooks; no dropout anyway
    ids = torch.stack([tokenizer(text, return_tensors="pt").input_ids[0, :WORDS] for text in texts]).to(device)
    model(ids)
    for hook in hooks:
        hook.remove()
    return torch.stack([h.float().mean(1) for h in outputs], dim=1).cpu()  # (texts, 2 * BLOCKS, D)


def moran(grid):
    z = grid - grid.mean()
    neighbors = sum(np.roll(z, shift, axis) for shift in (-1, 1) for axis in (0, 1))  # periodic 4-neighborhood
    return (z * neighbors).sum() / (4 * (z**2).sum())


# ------------------------------------- load data -------------------------------------
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
cache = DATA_DIR / "topolm_wikitext103_gpt2.npy"
if not cache.exists():
    lines = wikitext("wikitext-103-raw-v1/train-00000-of-00002.parquet")
    ids, count = [], 0
    for i in range(0, len(lines), 10_000):
        ids += tokenizer(lines[i:i + 10_000]).input_ids
        count = sum(len(s) for s in ids)
        if count > TOKENS:
            break
    np.save(cache, np.concatenate(ids)[:TOKENS].astype(np.int32))
train_tokens = torch.from_numpy(np.load(cache)).to(device)
generator = torch.Generator().manual_seed(1)
places = [torch.randperm(D, generator=generator) for _ in range(2 * BLOCKS)]  # grid cell of every unit, per layer
sheets = [place.argsort().view(GRID, GRID).to(device) for place in places]  # unit in every grid cell
cell = torch.stack(torch.meshgrid(torch.arange(WINDOW), torch.arange(WINDOW), indexing="ij"), -1).flatten(0, 1)
pairs = torch.triu_indices(WINDOW**2, WINDOW**2, 1)
inverse_distance = 1 / (cell[pairs[0]] - cell[pairs[1]]).abs().amax(1).float().to(device)  # chebyshev distance
pairs = pairs.to(device)

# -------------------------------------- training -------------------------------------
models = {}
for name, alpha in {"topographic": ALPHA, "baseline": 0.0}.items():
    path = MODEL_DIR / f"topolm_{name}.pt"
    if args.train in (None, name) and not path.exists():
        torch.save(train(alpha), path)
    if args.train is None:
        models[name] = torch.load(path, weights_only=False, map_location=device)
if args.train is not None:
    raise SystemExit

# ------------------------------------- create data -----------------------------------
test_lines = wikitext("wikitext-103-raw-v1/test-00000-of-00001.parquet")
val_tokens = torch.tensor(tokenizer("".join(test_lines)).input_ids, dtype=torch.int32, device=device)
lines = [line.strip() for line in wikitext("wikitext-2-raw-v1/train-00000-of-00001.parquet") if line.strip() and not line.startswith(" =")]
candidates = [s for line in lines for s in re.split(r"(?<= \.) ", line) if re.fullmatch(r"[A-Z][a-z]*( [a-z]+){11} \.", s)]
sentences = [s[:-2] for s in rng.choice(candidates, SAMPLES, replace=False)]
vocabulary = {w for line in lines for w in re.findall(r"[a-z]+", line.lower())}
nonwords = []
while len(nonwords) < SAMPLES * WORDS:
    word = nonword()
    if word not in vocabulary:
        nonwords.append(word)
lists = [" ".join(nonwords[i:i + WORDS]) for i in range(0, len(nonwords), WORDS)]

# -------------------------------------- recording ------------------------------------
maps, clustering, val_cost = {}, {}, {}
for name, model in models.items():
    responses = [torch.cat([record(model, texts[i:i + 60]) for i in range(0, SAMPLES, 60)]) for texts in (sentences, lists)]
    t, _ = ttest_ind(responses[0].numpy(), responses[1].numpy(), axis=0, equal_var=False)  # (2 * BLOCKS, D)
    maps[name] = np.zeros((2 * BLOCKS, GRID, GRID))
    for k, place in enumerate(places):
        maps[name][k].flat[place.numpy()] = t[k]
    clustering[name] = [moran(m) for m in maps[name]]
    val_cost[name] = validate(model)
    print(f"{name}: validation cost {val_cost[name]:.3f}, moran's I {np.round(clustering[name], 2).tolist()}")

# ----------------------------------- postprocessing ----------------------------------
smoothed = {name: gaussian_filter(m, (0, SMOOTHING, SMOOTHING)) for name, m in maps.items()}
limit = np.percentile(np.abs(np.stack(list(smoothed.values()))), 99)  # the few outlier units saturate
layer_names = [f"block{b + 1}_{part}" for b in range(BLOCKS) for part in ("attention", "feedforward")]

if not args.book:
    fig, axes = plt.subplots(2, 2 * BLOCKS, figsize=(16, 4))
    for row, name in zip(axes, smoothed):
        for ax, grid in zip(row, smoothed[name]):
            ax.imshow(grid, cmap="Spectral_r", vmin=-limit, vmax=limit)
            ax.axis("off")
    plt.show()
# -------------------------------- book postprocessing --------------------------------
else:
    for name in smoothed:
        for layer, grid in zip(layer_names, smoothed[name]):
            fig = plt.figure(figsize=(2, 2))
            plt.imshow(grid, cmap="Spectral_r", vmin=-limit, vmax=limit, interpolation="bilinear")
            plt.axis("off")
            fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
            plt.savefig(RGB_PDF_DIR / f"topolm_{name}_{layer}.pdf")
            plt.close(fig)
    np.savez(CSV_DIR / "topolm_maps.npz", **maps)
    save_csv(CSV_DIR / "topolm_clustering.csv", layer=layer_names, topographic=clustering["topographic"],
             baseline=clustering["baseline"])
    save_csv(CSV_DIR / "topolm_cost.csv", model=list(val_cost), validation=list(val_cost.values()), limit=limit)
