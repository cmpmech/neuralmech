import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

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
EAGLES = "Why don't the eagles just fly the Ring to Mordor?"
INTEGRAL = "What is the integral of x*exp(x) from 0 to 1?"
TRAIN = "A train leaves at 9:40 and arrives at 13:15. How many minutes does the trip take?"
DIVISORS = "How many positive divisors does 360 have?"
MAX_TOKENS = 2048

# mathematics: every OLMo 2 stage answers once greedily and SAMPLES times at temperature T
OLMO_STAGES = ["allenai/OLMo-2-1124-7B", "allenai/OLMo-2-1124-7B-SFT", "allenai/OLMo-2-1124-7B-DPO",
               "allenai/OLMo-2-1124-7B-Instruct"]
MATH = [INTEGRAL, TRAIN, DIVISORS]
SAMPLES = 16
T = 0.7
MATH_TOKENS = 1024

# (model, prompt, temperature, seed); temperature None is greedy decoding
RUNS = [
    ("allenai/OLMo-2-1124-7B", EAGLES, None, None),
    ("allenai/OLMo-2-1124-7B", EAGLES, 0.5, 0),
    ("allenai/OLMo-2-1124-7B", EAGLES, 1.0, 0),
    ("allenai/OLMo-2-1124-7B", EAGLES, 1.5, 0),
    ("allenai/OLMo-2-1124-7B-SFT", EAGLES, None, None),
    ("allenai/OLMo-2-1124-7B-SFT", EAGLES, 0.7, 5),
    ("allenai/OLMo-2-1124-7B-DPO", EAGLES, None, None),
    ("allenai/OLMo-2-1124-7B-Instruct", EAGLES, None, None),
    ("Qwen/Qwen3-8B-Base", INTEGRAL, None, None),
    ("Qwen/Qwen3-8B", INTEGRAL, None, None),
]
BASE_MODELS = ["allenai/OLMo-2-1124-7B", "Qwen/Qwen3-8B-Base"]

# ------------------------------------- inference -------------------------------------
def tokenize(prompt):
    if base:
        text = prompt  # base model continues raw text
    else:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                             enable_thinking=False)
    return tokenizer(text, return_tensors="pt", add_special_tokens=base).to(device)


answers = []
for name in dict.fromkeys(run[0] for run in RUNS):
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).to(device)
    model.eval()
    base = name in BASE_MODELS
    for _, prompt, temperature, seed in [run for run in RUNS if run[0] == name]:
        inputs = tokenize(prompt)
        if temperature is None:
            sampling = dict(do_sample=False)
        else:
            torch.manual_seed(seed)
            sampling = dict(do_sample=True, temperature=temperature, top_k=0, top_p=1.0)  # pure temperature
        with torch.no_grad():
            ids = model.generate(**inputs, max_new_tokens=MAX_TOKENS, pad_token_id=tokenizer.eos_token_id,
                                 **sampling)
        new_ids = ids[0, inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        decoding = "greedy" if temperature is None else f"T={temperature}, seed {seed}"
        length = f"{len(new_ids)} tokens" + (", cut" if len(new_ids) == MAX_TOKENS else "")
        answers.append(f"=== {name}, {decoding}, {length}\n{response}\n")
        print(answers[-1])
    for prompt in MATH if name in OLMO_STAGES else []:
        inputs = tokenize(prompt)
        with torch.no_grad():
            greedy = model.generate(**inputs, max_new_tokens=MATH_TOKENS, do_sample=False,
                                    pad_token_id=tokenizer.eos_token_id)
            torch.manual_seed(0)
            sampled = model.generate(**inputs, max_new_tokens=MATH_TOKENS, do_sample=True, temperature=T, top_k=0,
                                     top_p=1.0, num_return_sequences=SAMPLES, pad_token_id=tokenizer.eos_token_id)
        for i, ids in enumerate([greedy[0]] + list(sampled)):
            new_ids = ids[inputs["input_ids"].shape[1]:]
            new_ids = new_ids[new_ids != tokenizer.eos_token_id]  # padding after shorter samples
            response = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
            decoding = "greedy" if i == 0 else f"T={T}, sample {i - 1}"
            answers.append(f"=== {name}, {prompt}, {decoding}, {len(new_ids)} tokens\n{response}\n")
            print(answers[-1])
    del model
    torch.cuda.empty_cache()

# -------------------------------- book postprocessing --------------------------------
if args.book:
    (CSV_DIR / "post_training.txt").write_text("\n".join(answers))
