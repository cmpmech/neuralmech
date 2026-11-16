import argparse
import re
from collections import Counter
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
MODEL = "Qwen/Qwen3-1.7B"
STYLES = ["in the style of J. R. R. Tolkien's The Lord of the Rings", "in the style of a textbook on engineering mechanics"]
TOPICS = ["a bridge", "a river", "a tower", "a storm", "a forge", "a road", "a mountain", "a ship", "a wall",
          "a door", "a sword", "a wheel", "a rope", "a cave", "a lamp", "a tree", "a horse", "a gate", "a map",
          "a fire", "a well", "a hammer", "a stone", "a cliff", "an arch", "a mill", "a dam", "a forest",
          "a chain", "a winter"]
TRAIN_TOPICS = 20  # remaining topics test the probe
SAMPLES = 3  # sentences per topic and style
QUESTIONS = {  # question: word that an answer on topic contains
    "Explain the finite element method in two sentences.": "element",
    "What does Young's modulus measure? Answer in two sentences.": "stiff",
    "Why does a beam deflect under load? Answer in two sentences.": "beam",
    "Explain buckling in two sentences.": "buckl",
    "What is a boundary condition? Answer in two sentences.": "boundar",
    "How does a truss carry loads? Answer in two sentences.": "truss",
    "What is plastic deformation? Answer in two sentences.": "plastic",
    "Explain the Newton-Raphson method in two sentences.": "newton",
    "What is a stress tensor? Answer in two sentences.": "stress",
    "What is the Courant number? Answer in two sentences.": "courant",
}
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]  # multiples of the difference of the style means
ALPHA_LAYERS = 1.0  # strength of the sweep over layers
MAX_TOKENS = 80


# --------------------------------------- helper --------------------------------------
def chat(question):
    messages = [{"role": "user", "content": question}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)


@torch.no_grad()
def generate(questions, sample=False):
    ids = tokenizer([chat(q) for q in questions], return_tensors="pt", padding=True).to(device)
    out = model.generate(**ids, max_new_tokens=MAX_TOKENS, do_sample=sample, temperature=0.9 if sample else None,
                         top_p=0.95 if sample else None, top_k=None, repetition_penalty=1.2)
    return [s.strip() for s in tokenizer.batch_decode(out[:, ids.input_ids.shape[1]:], skip_special_tokens=True)]


@torch.no_grad()
def hidden_means(texts):
    ids = tokenizer(texts, return_tensors="pt", padding=True).to(device)
    states = model(**ids, output_hidden_states=True).hidden_states  # embedding + every layer
    mask = ids.attention_mask.float()
    mask[torch.arange(len(texts)), mask.argmax(1)] = 0  # first token carries outlier activations in every text
    mask = mask[:, :, None]
    return torch.stack([(h.float() * mask).sum(1) / mask.sum(1) for h in states], dim=1).cpu()


def steer(layer, vector):
    def hook(module, inputs, output):
        return (output[0] + vector.to(output[0].dtype),) + tuple(output[1:]) if isinstance(output, tuple) else output + vector.to(output.dtype)
    return model.model.layers[layer].register_forward_hook(hook)


def words(text):
    return re.findall(r"[a-z]+", text.lower())


def style(answers):
    return np.mean([np.mean([w in vocabulary for w in words(a)]) if words(a) else 0.0 for a in answers])


def scores(answers):
    on_topic = np.mean([word in a.lower() for a, word in zip(answers, QUESTIONS.values())])
    return style(answers), on_topic


# ------------------------------------- load model ------------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(device)
model.eval()
LAYERS = len(model.model.layers)

# ------------------------------------- create data -----------------------------------
sentences, labels, topic_ids = [], [], []
for label, phrase in enumerate(STYLES):
    prompts = [f"Write one sentence about {topic} {phrase}. Reply with the sentence only." for topic in TOPICS]
    for _ in range(SAMPLES):
        sentences += [s.split("\n")[0] for s in generate(prompts, sample=True)]
        labels += [label] * len(TOPICS)
        topic_ids += list(range(len(TOPICS)))
labels, topic_ids = np.array(labels), np.array(topic_ids)
train, test = topic_ids < TRAIN_TOPICS, topic_ids >= TRAIN_TOPICS
frequency = [Counter(w for i in np.flatnonzero(train & (labels == k)) for w in set(words(sentences[i]))) for k in (0, 1)]
vocabulary = {w for w, n in frequency[0].items() if n >= 3 and frequency[1][w] == 0 and len(w) > 3}  # tolkien words

# -------------------------------------- recording ------------------------------------
features = torch.cat([hidden_means(sentences[i:i + 32]) for i in range(0, len(sentences), 32)])
tolkien = features[train & (labels == 0)].mean(0)
mechanics = features[train & (labels == 1)].mean(0)
difference = tolkien - mechanics  # (layers + 1, D), one direction per layer
projection = ((features[test] - (tolkien + mechanics) / 2) * difference).sum(-1)  # linear probe without training
accuracy = ((projection > 0).numpy() == (labels[test] == 0)[:, None]).mean(0)

# ------------------------------------- stimulation -----------------------------------
questions = list(QUESTIONS)
plain = generate(questions)
by_layer = []
for layer in range(LAYERS):
    handle = steer(layer, ALPHA_LAYERS * difference[layer + 1].to(device))
    by_layer.append(scores(generate(questions)))
    handle.remove()
best = int(np.argmax([np.prod(score) for score in by_layer]))
by_alpha, steered = [], {}
for alpha in ALPHAS:
    handle = steer(best, alpha * difference[best + 1].to(device))
    steered[alpha] = generate(questions)
    handle.remove()
    by_alpha.append(scores(steered[alpha]))
print(f"{len(vocabulary)} tolkien words, e.g. {sorted(vocabulary)[:30]}")
print(f"best layer {best + 1} (of {LAYERS}); tolkien words {by_layer[best][0]:.3f}, on topic {by_layer[best][1]:.2f}")
for alpha, (share, topic) in zip(ALPHAS, by_alpha):
    print(f"alpha {alpha:.2f}: tolkien words {share:.3f}, on topic {topic:.2f}")

# --------------------------------------- lesion --------------------------------------
requests = [f"Describe {topic} in two sentences in the style of J. R. R. Tolkien." for topic in TOPICS[TRAIN_TOPICS:]]
before = generate(requests)
v = difference[best + 1] / difference[best + 1].norm()
v = v.to(device, torch.bfloat16)
with torch.no_grad():
    weights = [model.model.embed_tokens.weight.T]  # (D, vocabulary), so that every matrix writes rows of size D
    weights += [w for layer in model.model.layers for w in (layer.self_attn.o_proj.weight, layer.mlp.down_proj.weight)]
    for w in weights:
        w -= torch.outer(v, v @ w)
after = generate(requests)
plain_after = generate(questions)
lesion = [style(before), style(after)]
print(f"tolkien words in tolkien-style requests: {lesion[0]:.3f} before, {lesion[1]:.3f} after lesion")
print(f"mechanics answers on topic: {scores(plain)[1]:.2f} before, {scores(plain_after)[1]:.2f} after lesion")

# ----------------------------------- postprocessing ----------------------------------
print("probe accuracy per layer:", np.round(accuracy, 2))
for alpha in ALPHAS:
    print(f"\nalpha {alpha}: {steered[alpha][0]}")
for b, a in zip(before[:3], after[:3]):
    print(f"\nbefore lesion: {b}\nafter lesion: {a}")

if not args.book:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(range(LAYERS + 1), accuracy, "k")
    axes[0].plot(range(1, LAYERS + 1), [r for r, _ in by_layer], "b")
    axes[1].plot(ALPHAS, [r for r, _ in by_alpha], "b")
    axes[1].plot(ALPHAS, [t for _, t in by_alpha], "k")
    plt.show()
# -------------------------------- book postprocessing --------------------------------
else:
    save_csv(CSV_DIR / "steering_probe.csv", layer=np.arange(LAYERS + 1), accuracy=accuracy)
    save_csv(CSV_DIR / "steering_layers.csv", layer=np.arange(1, LAYERS + 1), style=[r for r, _ in by_layer],
             topic=[t for _, t in by_layer])
    save_csv(CSV_DIR / "steering_alpha.csv", alpha=ALPHAS, style=[r for r, _ in by_alpha], topic=[t for _, t in by_alpha])
    save_csv(CSV_DIR / "steering_lesion.csv", case=["before", "after"], style=lesion,
             topic=[scores(plain)[1], scores(plain_after)[1]])
    (CSV_DIR / "steering_answers.txt").write_text(
        "\n\n".join([f"PLAIN\n{plain[0]}"] + [f"ALPHA {a}\n{steered[a][0]}" for a in ALPHAS]
                    + [f"TOLKIEN BEFORE\n{b}\nTOLKIEN AFTER\n{a}" for b, a in zip(before, after)]))
