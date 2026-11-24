import argparse
import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import umap
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
RGB_PDF_DIR = (RESULTS_DIR / "rgb_pdf").resolve()

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
MODEL = "Qwen/Qwen3-1.7B"
LAYER = 14  # hidden state after this block, the middle of 28
SENTENCES = 4000
BATCH_SIZE = 32
RESOLUTION = 48  # grid cells per side of the atlas
MIN_COUNT = 6  # tokens a cell needs to be labeled
CATEGORIES = ["number", "punctuation", "function word", "name", "word", "word piece"]
COLORS = ["#071141", "#94b8c2", "#517095", "#e05f5a", "#a32126", "#ffac8c"]  # col1 col6 col4 col12 col14 col10
FUNCTION_WORDS = set("the a an of in on at to for from by with and or but as is was were are be been has have had it its "
                     "this that these those which who whom he she they his her their not no also than then".split())


# --------------------------------------- helper --------------------------------------
def category(token):
    word = token.strip()
    if re.fullmatch(r"[0-9.,]+", word):
        return 0
    if not re.search(r"[A-Za-z0-9]", word):
        return 1
    if not token.startswith(" "):
        return 5
    if word.lower() in FUNCTION_WORDS:
        return 2
    return 3 if word[0].isupper() else 4


@torch.no_grad()
def token_states(sentences):
    ids = tokenizer(sentences, return_tensors="pt", padding=True).to(device)
    h = model(**ids, output_hidden_states=True).hidden_states[LAYER].float()
    mask = ids.attention_mask.bool()
    mask[torch.arange(len(sentences)), mask.int().argmax(1)] = False  # first token carries outlier activations
    return h[mask].cpu(), ids.input_ids[mask].cpu()


# ------------------------------------- load model ------------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL, padding_side="right")
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(device)
model.eval()

# ------------------------------------- load data -------------------------------------
wikitext = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
lines = [line.strip() for line in wikitext["train"]["text"] if line.strip() and not line.startswith(" =")]
lines = [re.sub(r" @(.)@ ", r"\1", line) for line in lines]  # wikitext escapes 1 @,@ 000 and long @-@ term
sentences = [s.strip() for line in lines for s in line.split(" . ") if 8 <= len(s.split()) <= 30]
rng = np.random.default_rng(0)
sentences = [sentences[i] for i in rng.choice(len(sentences), SENTENCES, replace=False)]

states, tokens = zip(*[token_states(sentences[i:i + BATCH_SIZE]) for i in range(0, SENTENCES, BATCH_SIZE)])
states, tokens = torch.cat(states), torch.cat(tokens)
words = tokenizer.convert_ids_to_tokens(tokens)
words = [tokenizer.convert_tokens_to_string([w]) for w in words]
print(f"{len(words)} tokens")

# ------------------------------------- embedding -------------------------------------
states = states - states.mean(0)
states = states / states.norm(dim=1, keepdim=True)
U, S, V = torch.pca_lowrank(states, q=50, center=False)
reduced = (states @ V).numpy()
embedding = umap.UMAP(n_neighbors=30, min_dist=0.5, metric="cosine", random_state=0).fit_transform(reduced)

# ----------------------------------- postprocessing ----------------------------------
low, high = np.percentile(embedding, [1, 99], axis=0)
keep = np.all((embedding >= low) & (embedding <= high), axis=1)
xy = (embedding - low) / (high - low) * RESOLUTION
cells = np.floor(xy).astype(int)
labels = {}
for (i, j), word in zip(map(tuple, cells[keep]), np.array(words)[keep]):
    labels.setdefault((i, j), Counter())[word] += 1
atlas, taken = [], set()
for (i, j), counter in sorted(labels.items(), key=lambda item: -sum(item[1].values())):
    count = sum(counter.values())
    near = {(i + di, j + dj) for di in (-2, -1, 0, 1, 2) for dj in (-1, 0, 1) if abs(di) < 2 or dj == 0}
    word, n = counter.most_common(1)[0]
    if count >= MIN_COUNT and word.strip() and not near & taken:
        atlas.append((i, j, word.strip(), category(word), count, n / count))
        taken.add((i, j))
print(f"{keep.sum()} tokens shown, {len(atlas)} labeled cells, purity {np.mean([a[5] for a in atlas]):.2f}")
for c, name in enumerate(CATEGORIES):
    print(f"{name}: {sum(a[3] == c for a in atlas)} cells")

fig = plt.figure(figsize=(6, 6))
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
ax = fig.add_subplot()
colors = np.array(COLORS)[[category(w) for w in np.array(words)[keep]]]
ax.scatter(xy[keep, 0], xy[keep, 1], s=0.8, c=colors, alpha=0.3, lw=0, rasterized=True)
for i, j, word, c, count, purity in atlas:
    ax.text(i + 0.5, j + 0.5, word, ha="center", va="center", fontsize=3.2 + 0.8 * np.log(count))
ax.set_xlim(0, RESOLUTION)
ax.set_ylim(0, RESOLUTION)
ax.set_aspect("equal")
ax.axis("off")

if not args.book:
    plt.show()
# -------------------------------- book postprocessing --------------------------------
else:
    fig.savefig(RGB_PDF_DIR / "activation_atlas.pdf")
