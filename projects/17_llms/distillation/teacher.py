import json
import random
import re
from pathlib import Path

from vllm import LLM, SamplingParams

BASE_DIR = Path(__file__).parent
DATA_DIR = (BASE_DIR / "../../../external_data/distillation").resolve()
CUWAVE_DIR = (DATA_DIR / "cuwave").resolve()

# -------------------------------------- settings -------------------------------------
MODEL = "openai/gpt-oss-20b"
GPU_MEMORY = 0.65  # fraction of gpu memory for vLLM
CONTEXT = 160_000  # characters per slice, about 40k tokens
CODE_TYPES = [
    "conceptual questions about what the documented features are and what they are for",
    "how-to questions answered with short Python code (2-6 lines) that uses the API exactly as shown in the source",
    "the meaning, units and valid ranges of function arguments and class fields",
    "pitfalls, error messages and constraints a user runs into",
    "comparisons between alternatives the material offers and when to use which",
    "what the implementation does internally, step by step",
    "small worked examples with concrete numbers",
    "why the design is the way it is",
    "setting up a simulation: questions that describe walls in physical words (pressure-release, "
    "reflecting, rigid, free surface, clamped, conductor) and answers with the complete constructor "
    "call including every required argument and the boundary= keyword with the matching conditions",
]
THEORY_TYPES = [
    "conceptual questions about the physics and the discretization",
    "derivations of the finite difference schemes, written with plain-text math",
    "stability and accuracy: conditions, truncation errors and their consequences",
    "boundary and initial conditions and how they are discretized",
    "verification of implementations: exact solutions, convergence rates",
    "algorithms and short NumPy code (3-8 lines) that implements a scheme step",
    "small worked examples with concrete numbers",
    "comparisons between schemes and when to use which",
]
CORE = ("scalar", "wave", "boundary", "elastic", "maxwell", "signals", "sensitivity")
SETUP = ("boundary", "scalar", "elastic", "maxwell", "anisotropic", "wave")
PASSES = [  # (slices, question types, repeats)
    (None, range(8), 2),
    ("langtangen", range(8), 8),
    (CORE, (1, 2), 6),
    (SETUP, (8,), 20),
]
ASK = ("Write 10 question-answer pairs that a user might ask about the material above, answered "
       "correctly and only from it. This batch covers: {}. Answers are complete but concise, 2-6 "
       "sentences, plus code where asked. Code must be complete and correct as shown in the "
       "material; never invent functions, classes or arguments. Questions must not mention "
       "'the material' or 'the documentation'. Return only a JSON list of objects with keys "
       "\"question\" and \"answer\".")
SYSTEM = ("You are an expert on cuwave, a GPU finite-difference wave solver, and on finite "
          "difference methods for PDEs. Answer from the following material.\n\n")

# --------------------------------------- helper --------------------------------------
def section(path):
    return f"--- {path.relative_to(CUWAVE_DIR)} ---\n{path.read_text()}\n"


def final(text):
    return text.split("assistantfinal")[-1].strip()


# ------------------------------------ prepare data -----------------------------------
slices = {}
examples = [p for p in (CUWAVE_DIR / "examples").rglob("*.py") if "__pycache__" not in p.parts]
for page in sorted((CUWAVE_DIR / "docs").glob("*.md")):
    name = page.stem.removeprefix("cuda_")
    kernel = page.stem.startswith("cuda_")
    source = CUWAVE_DIR / ("cuwave/kernels" if kernel else "cuwave") / (name + (".cu" if kernel else ".py"))
    pattern = re.compile(rf"cuwave\.{name.split('_')[0]}\b|from cuwave import .*\b{name}\b")
    users = [p for p in examples if p.parent.name == page.stem or pattern.search(p.read_text())]
    text = section(page) + (section(source) if source.exists() else "")
    for p in sorted(users, key=lambda p: p.stat().st_size)[:2]:
        if len(text) + p.stat().st_size < CONTEXT:
            text += section(p)
    slices[page.stem] = text

lines = (DATA_DIR / "langtangen.txt").read_text().splitlines()[5858:12228]  # chapter 2, wave equations
for i in range(3):
    chunk = "\n".join(lines[i * len(lines) // 3:(i + 1) * len(lines) // 3])
    slices[f"langtangen_{i}"] = f"--- Langtangen & Linge, Finite Difference Computing with PDEs, chapter 2 ---\n{chunk}\n"

jobs = []
for names, kinds, repeats in PASSES:
    for name, text in slices.items():
        theory = name.startswith("langtangen")
        if names == "langtangen" and not theory or names not in (None, "langtangen") and name not in names:
            continue
        for kind in kinds:
            if theory and kind >= len(THEORY_TYPES):
                continue
            prompt = ASK.format((THEORY_TYPES if theory else CODE_TYPES)[kind])
            jobs += [[{"role": "system", "content": SYSTEM + text}, {"role": "user", "content": prompt}]] * repeats
print(f"{len(slices)} slices, {len(jobs)} requests")

# ------------------------------------ create data ------------------------------------
llm = LLM(MODEL, max_model_len=65536, gpu_memory_utilization=GPU_MEMORY, seed=0)
params = [SamplingParams(temperature=0.8, max_tokens=12000, seed=i) for i in range(len(jobs))]
outputs = llm.chat(jobs, params, chat_template_kwargs={"reasoning_effort": "medium"})

pairs, failed = [], 0
for output in outputs:
    text = final(output.outputs[0].text)
    try:
        batch = json.loads(text[text.index("["): text.rindex("]") + 1])
        pairs += [{"question": p["question"], "answer": p["answer"]} for p in batch
                  if isinstance(p, dict) and isinstance(p.get("question"), str) and isinstance(p.get("answer"), str)]
    except ValueError:
        failed += 1
seen = set()
pairs = [p for p in pairs if p["question"] not in seen and not seen.add(p["question"])]
random.Random(0).shuffle(pairs)

# --------------------------------------- export --------------------------------------
(DATA_DIR / "qa_raw.json").write_text(json.dumps(pairs, indent=1, ensure_ascii=False))
print(f"{len(pairs)} pairs, {failed} of {len(jobs)} requests without valid json -> {DATA_DIR / 'qa_raw.json'}")
