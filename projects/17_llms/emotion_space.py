import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from postprocessing import save_csv

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
MODEL = "Qwen/Qwen3-8B"
EMOTIONS = {  # emotion: (valence, arousal), coarse signs assigned by hand
    "excited": (1, 1), "ecstatic": (1, 1), "joyful": (1, 1), "enthusiastic": (1, 1), "amused": (1, 1),
    "proud": (1, 1), "inspired": (1, 1), "triumphant": (1, 1), "elated": (1, 1), "playful": (1, 1),
    "calm": (1, -1), "relaxed": (1, -1), "content": (1, -1), "serene": (1, -1), "grateful": (1, -1),
    "relieved": (1, -1), "tender": (1, -1), "hopeful": (1, -1), "peaceful": (1, -1), "satisfied": (1, -1),
    "angry": (-1, 1), "furious": (-1, 1), "afraid": (-1, 1), "terrified": (-1, 1), "anxious": (-1, 1),
    "panicked": (-1, 1), "desperate": (-1, 1), "frustrated": (-1, 1), "disgusted": (-1, 1), "jealous": (-1, 1),
    "nervous": (-1, 1), "hostile": (-1, 1),
    "sad": (-1, -1), "lonely": (-1, -1), "bored": (-1, -1), "tired": (-1, -1), "depressed": (-1, -1),
    "gloomy": (-1, -1), "guilty": (-1, -1), "ashamed": (-1, -1), "disappointed": (-1, -1), "melancholic": (-1, -1),
    "hopeless": (-1, -1), "resigned": (-1, -1),
}
TOPICS = ["a train station", "a kitchen", "a mountain hike", "an office", "a hospital", "a garden", "a beach",
          "a school"]
LAYER = 16  # of 36; steering breaks text down in later layers
MAX_TOKENS = 120
QUESTION = "What is 17 times 24? Answer in two sentences."
STEERED = ["enthusiastic", "ashamed", "bored"]
ALPHAS = [0, 4, 8, 12]  # multiples of emotion direction


# --------------------------------------- helper --------------------------------------
def chat(prompt):
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)


@torch.no_grad()
def generate(prompts, sample=True):
    ids = tokenizer([chat(p) for p in prompts], return_tensors="pt", padding=True).to(device)
    settings = dict(temperature=0.9, top_p=0.95) if sample else dict(temperature=None, top_p=None)
    out = model.generate(**ids, max_new_tokens=MAX_TOKENS, do_sample=sample, top_k=None, **settings)
    return [s.strip() for s in tokenizer.batch_decode(out[:, ids.input_ids.shape[1]:], skip_special_tokens=True)]


@torch.no_grad()
def hidden_mean(texts):
    ids = tokenizer(texts, return_tensors="pt", padding=True).to(device)
    states = model.model(**ids, output_hidden_states=True).hidden_states  # no logits
    mask = ids.attention_mask.float()
    mask[torch.arange(len(texts)), mask.argmax(1)] = 0  # first token carries outlier activations in every text
    means = [(h.float() * mask[:, :, None]).sum(1) / mask.sum(1, keepdim=True) for h in states]
    return torch.stack(means, dim=1).cpu().numpy()  # (texts, layers + 1, D)


def steer(layer, vector):
    def hook(module, inputs, output):
        return (output[0] + vector.to(output[0].dtype),) + tuple(output[1:]) if isinstance(output, tuple) else output + vector.to(output.dtype)
    return model.model.layers[layer - 1].register_forward_hook(hook)  # hidden state of index layer is its output


def principal_components(vectors):
    _, singular, V = np.linalg.svd(vectors, full_matrices=False)
    pcs = vectors @ V[:2].T
    r = [np.corrcoef(pcs[:, k], [EMOTIONS[w][k] for w in words])[0, 1] for k in range(2)]
    return pcs * np.sign(r), np.abs(r), singular[:2] ** 2 / (singular ** 2).sum()


# ------------------------------------- load model ------------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(device)
model.eval()

# ------------------------------------- create data -----------------------------------
words = list(EMOTIONS)
stories = []
for topic in TOPICS:
    prompts = [f"Write a short story of four sentences set in {topic}, in which the main character feels {e}. "
               f"Show the feeling through actions and thoughts, and do not use the word '{e}'." for e in words]
    stories.append(generate(prompts))
    print(f"{topic}: {stories[-1][0][:100]}")

# -------------------------------------- recording ------------------------------------
X = np.stack([hidden_mean(s) for s in stories])  # (topics, emotions, layers + 1, D)
X = X - X.mean(1, keepdims=True)  # remove the topic of each story
vectors = X.mean(0)
for layer in range(1, vectors.shape[1]):
    _, r, _ = principal_components(vectors[:, layer])
    print(f"layer {layer}: correlation with valence {r[0]:.2f}, arousal {r[1]:.2f}")
pcs, r, explained = principal_components(vectors[:, LAYER])
for k, name in enumerate(["valence", "arousal"]):
    print(f"pc{k + 1}: {explained[k]:.2f} of variance, correlation with {name} {r[k]:.2f}")

# ------------------------------------- stimulation -----------------------------------
steered = {}
for e in STEERED:
    direction = torch.tensor(vectors[words.index(e), LAYER], device=device)
    for alpha in ALPHAS:
        handle = steer(LAYER, alpha * direction)
        steered[e, alpha] = generate([QUESTION], sample=False)[0]
        handle.remove()
        print(f"\n{e}, alpha {alpha}: {steered[e, alpha]}")

# ----------------------------------- postprocessing ----------------------------------
if not args.book:
    plt.scatter(pcs[:, 0], pcs[:, 1], c=[EMOTIONS[w][0] for w in words], cmap="cividis")
    for w, p in zip(words, pcs):
        plt.text(*p, w)
    plt.show()
# -------------------------------- book postprocessing --------------------------------
else:
    points = pcs / np.abs(pcs).max(0)
    side = np.sign(points[:, 0])
    labels = np.stack([side * (np.abs(points[:, 0]).max() + 0.1), points[:, 1]], axis=1)  # one label column per side
    for _ in range(2000):  # push labels of a column apart until they no longer overlap
        dy = labels[:, None, 1] - labels[None, :, 1]
        overlap = (side[:, None] == side[None]) & (np.abs(dy) < 0.085) & ~np.eye(len(words), dtype=bool)
        labels[:, 1] += 0.05 * np.where(overlap, np.sign(dy + 1e-9) * (0.085 - np.abs(dy)), 0).sum(1)
    save_csv(CSV_DIR / "emotion_space.csv", pc1=points[:, 0], pc2=points[:, 1], x=labels[:, 0], y=labels[:, 1],
             word=words, valence=[EMOTIONS[w][0] for w in words], arousal=[EMOTIONS[w][1] for w in words])
    (CSV_DIR / "emotion_space_stories.txt").write_text(
        "\n\n".join(f"{w} ({t})\n{stories[i][j]}" for i, t in enumerate(TOPICS) for j, w in enumerate(words)))
    (CSV_DIR / "emotion_steering.txt").write_text(
        "\n\n".join(f"{e.upper()} ALPHA {alpha}\n{QUESTION}\n{a}" for (e, alpha), a in steered.items()))
