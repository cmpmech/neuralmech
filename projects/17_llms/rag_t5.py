import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoTokenizer, T5EncoderModel, T5ForConditionalGeneration

from postprocessing import save_csv

BASE_DIR = Path(__file__).parent
PROJECTS_DIR = (BASE_DIR / "..").resolve()
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
CSV_DIR = (RESULTS_DIR / "data").resolve()

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
EMBEDDER = "sentence-transformers/sentence-t5-base"  # T5 encoder trained for sentence similarity
GENERATOR = "google/flan-t5-large"
QUESTION = "Which script identifies the stiffness of a bar from displacement data?"
K = 3  # retrieved documents
SHOWN = 10  # documents shown in the plot


# --------------------------------------- helper --------------------------------------
def read_documents():
    documents = []
    for readme in sorted(PROJECTS_DIR.glob("*/README.md")):
        lines = readme.read_text().splitlines()
        for i, line in enumerate(lines):
            match = re.match(r"^- `([^`]+\.py)`(.*)", line)
            if not match:
                continue
            text = [match.group(2).strip()]
            for following in lines[i + 1:]:
                if not following.startswith("  "):
                    break
                text.append(following.strip())
            documents.append(f"{readme.parent.name}/{match.group(1)}: {' '.join(text)}")
    return documents


def embed(texts):
    batch = tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
    mask = batch["attention_mask"][..., None]
    with torch.no_grad():
        hidden = encoder(**batch).last_hidden_state
        z = dense((hidden * mask).sum(1) / mask.sum(1))  # mean pooling, then dense layer
    return torch.nn.functional.normalize(z, dim=-1)


def generate(prompt):
    ids = generator_tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        out = generator.generate(ids, max_new_tokens=40, do_sample=False)
    return generator_tokenizer.decode(out[0], skip_special_tokens=True)


# ------------------------------------- load models -----------------------------------
tokenizer = AutoTokenizer.from_pretrained(EMBEDDER)
encoder = T5EncoderModel.from_pretrained(EMBEDDER).to(device).eval()
dense = torch.nn.Linear(768, 768, bias=False).to(device)
dense.load_state_dict({"weight": load_file(hf_hub_download(EMBEDDER, "2_Dense/model.safetensors"))["linear.weight"]})
generator_tokenizer = AutoTokenizer.from_pretrained(GENERATOR)
generator = T5ForConditionalGeneration.from_pretrained(GENERATOR).to(device).eval()

# ------------------------------------- retrieval -------------------------------------
documents = read_documents()
z_documents = torch.cat([embed(documents[i:i + 64]) for i in range(0, len(documents), 64)])
z_question = embed([QUESTION])
similarity = (z_documents @ z_question.T).squeeze(1).cpu()
order = similarity.argsort(descending=True)
context = "\n".join(documents[i] for i in order[:K])

# ------------------------------------- generation ------------------------------------
answer_plain = generate(f"Question: {QUESTION}")
answer_rag = generate(f"Answer the question with the name of the script, using the context.\n"
                      f"Context:\n{context}\nQuestion: {QUESTION}")

# ----------------------------------- postprocessing ----------------------------------
print(f"{len(documents)} documents")
for rank, i in enumerate(order[:SHOWN]):
    print(f"{rank + 1} {similarity[i]:.3f} {documents[i][:90]}")
print(f"without retrieval: {answer_plain}")
print(f"with retrieval:    {answer_rag}")

if not args.book:
    plt.bar(range(SHOWN), similarity[order[:SHOWN]])
    plt.show()
# -------------------------------- book postprocessing --------------------------------
else:
    save_csv(CSV_DIR / "rag_similarity.csv", rank=np.arange(1, SHOWN + 1),
             s=similarity[order[:SHOWN]].numpy(),
             script=[documents[i].split(":")[0] for i in order[:SHOWN]])
    (RESULTS_DIR / "data" / "rag_answers.txt").write_text(
        f"question: {QUESTION}\nwithout retrieval: {answer_plain}\nwith retrieval: {answer_rag}\n"
        f"context:\n{context}\n")
