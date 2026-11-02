import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from postprocessing import save_csv

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
RGB_PDF_DIR = (RESULTS_DIR / "rgb_pdf").resolve()
CSV_DIR = (RESULTS_DIR / "data").resolve()

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
device = torch.device("cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
MODEL = "google-t5/t5-small"
TEXT = "translate English to German: In winter, it snows."
LAYER = 4  # fifth of six layers, averaged over heads

# --------------------------------------- helper --------------------------------------


def group_words(tokens):
    # "\u2581" marks word starts; subwords and punctuation join previous word
    ids = torch.tensor([t.startswith("\u2581") for t in tokens]).cumsum(0) - 1
    words = [
        "".join(tokens[k] for k in range(len(tokens)) if ids[k] == w)
        for w in range(ids[-1] + 1)
    ]
    return torch.nn.functional.one_hot(ids).float(), [w.lstrip("\u2581") for w in words]


# ------------------------------------ load model -------------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL).to(device)
model.eval()

# ------------------------------------- generation ------------------------------------
inputs = tokenizer(TEXT, return_tensors="pt").to(device)
with torch.no_grad():
    generated = model.generate(**inputs, do_sample=False)
    outputs = model(**inputs, decoder_input_ids=generated, output_attentions=True)

# drop task prefix and end-of-sequence token; row r is the query producing token r + 1
prefix = len(tokenizer("translate English to German:").input_ids) - 1
source = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])[prefix:-1]
target = tokenizer.convert_ids_to_tokens(generated[0])[1:-1]
self_attention = outputs.encoder_attentions[LAYER][0].mean(0)[prefix:-1, prefix:-1]
cross_attention = outputs.cross_attentions[LAYER][0].mean(0)[:-2, prefix:-1]

# merge subwords: average over query subwords, sum over key subwords
source_groups, source = group_words(source)
target_groups, target = group_words(target)
self_attention = (
    source_groups.T @ self_attention @ source_groups / source_groups.sum(0)[:, None]
)
cross_attention = (
    target_groups.T @ cross_attention @ source_groups / target_groups.sum(0)[:, None]
)

# ----------------------------------- postprocessing ----------------------------------
figures = {}
for name, weights in [("cross", cross_attention), ("self", self_attention)]:
    weights = weights / weights.sum(-1, keepdim=True)  # renormalize after cropping
    figures[name] = plt.figure(figsize=(4, 4))
    ax = figures[name].add_axes([0, 0, 1, 1])
    ax.imshow(weights, cmap="cividis", vmin=0, vmax=1, interpolation="none")
    ax.axis("off")

print("source tokens", source)
print("target tokens", target)

if not args.book:
    plt.show()
# -------------------------------- book postprocessing --------------------------------
else:
    for name, fig in figures.items():
        fig.savefig(RGB_PDF_DIR / f"attention_map_{name}.pdf")
    save_csv(CSV_DIR / "attention_map_source_tokens.csv", token=source)
    save_csv(CSV_DIR / "attention_map_target_tokens.csv", token=target)
