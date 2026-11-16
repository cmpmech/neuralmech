import argparse
import json
import time
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import helper

BASE_DIR = Path(__file__).parent
DATA_DIR = (BASE_DIR / "../../../external_data/distillation").resolve()
RESULTS_DIR = (BASE_DIR / "../../../results").resolve()
CSV_DIR = (RESULTS_DIR / "data").resolve()

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
parser.add_argument("--small", action="store_true")  # 0.6B student, trained more aggressively
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
# hyperparameters
EPOCHS = 16 if args.small else 12
LR = 4e-4 if args.small else 2e-4
BATCH_SIZE = 8

# model settings
MODEL = "Qwen/Qwen3-0.6B" if args.small else "Qwen/Qwen3-1.7B"
RANK = 64
MAX_TOKENS = 400
NAME = MODEL.split("/")[1].lower()
ADAPTER = DATA_DIR / f"lora_{NAME}.pt"

# ------------------------------------- load data -------------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL)
samples = []
for pair in json.loads((DATA_DIR / "qa.json").read_text()):
    prompt = tokenizer(helper.chat(tokenizer, pair["question"])).input_ids
    response = tokenizer(pair["answer"] + tokenizer.eos_token, add_special_tokens=False).input_ids
    samples.append((torch.tensor(prompt + response), torch.tensor([-100] * len(prompt) + response)))

# --------------------------- instantiate model & optimizer ---------------------------
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(device)
helper.add_lora(model, RANK)
parameters = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(parameters, lr=LR)

# -------------------------------------- training -------------------------------------
model.eval()
before = {key: helper.answer(model, tokenizer, q, MAX_TOKENS) for key, q in helper.PROBES.items()}  # B = 0

if ADAPTER.exists():
    model.load_state_dict(torch.load(ADAPTER, weights_only=False, map_location=device), strict=False)
else:
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.train()
    batches = range(0, len(samples), BATCH_SIZE)
    train_cost = []
    tic = time.time()
    pbar = tqdm(range(EPOCHS))
    for epoch in pbar:
        cost_epoch = 0
        order = torch.randperm(len(samples))
        for i in batches:
            batch = [samples[j] for j in order[i:i + BATCH_SIZE]]
            x = torch.nn.utils.rnn.pad_sequence([s[0] for s in batch], True, tokenizer.pad_token_id).to(device)
            y = torch.nn.utils.rnn.pad_sequence([s[1] for s in batch], True, -100).to(device)
            optimizer.zero_grad()
            cost = model(input_ids=x, attention_mask=x != tokenizer.pad_token_id, labels=y).loss
            cost.backward()
            optimizer.step()
            cost_epoch += cost.item()
        train_cost.append(cost_epoch / len(batches))
        pbar.set_postfix(cost=f"{train_cost[-1]:.2e}")
    toc = time.time()
    print(f"elapsed time {toc - tic:.2f} s")
    model.gradient_checkpointing_disable()
    torch.save({n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}, ADAPTER)

# ----------------------------------- postprocessing ----------------------------------
model.eval()
after = {key: helper.answer(model, tokenizer, q, MAX_TOKENS) for key, q in helper.PROBES.items()}
text = "".join(f"--- {key}: {q}\nbefore: {before[key]}\nafter: {after[key]}\n\n" for key, q in helper.PROBES.items())
print(f"{len(samples)} pairs, {sum(p.numel() for p in parameters)} trainable parameters\n\n{text}")

# -------------------------------- book postprocessing --------------------------------
if args.book:
    (CSV_DIR / f"distillation_answers_{NAME}.txt").write_text(text)
