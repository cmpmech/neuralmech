import argparse
import re
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
MODEL = "Qwen/Qwen3-1.7B"
PROMPT = "How many times does the letter e appear in 'Deep Learning in Computational Mechanics'?"
MAX_TOKENS = 4000
ELIDE = (6, 34)  # items of letter-by-letter list replaced by vertical dots in book

# ------------------------------------- load model ------------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(device)
model.eval()

# ------------------------------------- inference -------------------------------------
messages = [{"role": "user", "content": PROMPT}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
inputs = tokenizer(text, return_tensors="pt").to(device)
with torch.no_grad():
    ids = model.generate(**inputs, max_new_tokens=MAX_TOKENS, do_sample=False)
response = tokenizer.decode(ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
reasoning, answer = response.removeprefix("<think>").split("</think>")
reasoning, answer = reasoning.strip(), answer.strip()
reasoning_tokens = len(tokenizer(reasoning)["input_ids"])

# ----------------------------------- postprocessing ----------------------------------
print(f"reasoning tokens: {reasoning_tokens}, answer tokens: {len(tokenizer(answer)['input_ids'])}")
print(f"reasoning:\n{reasoning}\n\nanswer:\n{answer}")

# -------------------------------- book postprocessing --------------------------------
if args.book:
    escape = str.maketrans({c: "\\" + c for c in "&%$#_{}"})
    def to_latex(s):
        s = s.translate(escape).replace(" '", " `")
        s = re.sub(r'"(.+?)"', r"``\1''", s)
        s = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", s)
        s = re.sub(r"^- ", r"\\textbullet~", s, flags=re.M)
        s = re.sub(r"\n\n+", "\n\n", s)
        return re.sub(r"(?<!\n)\n(?!\n)", r"\\\\\n", s)

    shown = re.sub(rf"(?m)^{ELIDE[0]}\. .*\n(?:.*\n)*?{ELIDE[1]}\. .*\n", "VDOTS\n", reasoning)
    shown = to_latex(shown).replace("VDOTS", r"\hspace*{0.3em}\raisebox{0.5pt}[\dimexpr\height-3pt\relax]{$\vdots$}")
    (CSV_DIR / "reasoning_model.tex").write_text(
        f"% {reasoning_tokens} reasoning tokens\n"
        "\\Needspace{4.5cm}\n"  # keep prompt with start of reasoning
        f"\\begin{{userbubble}}[after skip=0pt]{to_latex(PROMPT)}\\end{{userbubble}}\n\\nopagebreak\\vspace{{4pt}}\n"
        f"\\begin{{modelbubble}}[thinkingbubble]\n{shown}\n\\tcblower\n{to_latex(answer)}\n\\end{{modelbubble}}\n"
    )
