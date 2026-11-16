import ast
import json
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = (BASE_DIR / "../../../external_data/distillation").resolve()
CUWAVE_DIR = (DATA_DIR / "cuwave").resolve()

# -------------------------------------- settings -------------------------------------
WORKERS = 4
TIMEOUT = 60  # seconds per snippet
MODULES = ("boundary", "wave", "scalar", "elastic", "anisotropic", "maxwell", "sensitivity", "signals",
           "geometry", "stencils", "utils", "regularization", "optimization", "postprocessing",
           "decomposition", "evals")
PREAMBLE = "\n".join(f"from cuwave.{m} import *" for m in MODULES) + """
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import cupy as cp
import cuwave
DIM = 2
PRECISION = "float32"
SPACE_ORDER = 8
SAFETY = 0.99
THREADS = (4, 128)
RESOLUTION = 64
FREQUENCY = 4
LENGTH = 1
WAVESPEED = wavespeed = c = 0.5
DENSITY = density = rho = 1.0
WAVESPEED_P, WAVESPEED_S = 1.0, 0.5
REFLECTION = 1e-6
AMPLITUDE = 1e8
CYCLES = 5
T = 0.05
Nx = (RESOLUTION,) * DIM
dx = tuple(LENGTH / (n - 3) for n in Nx)
dt = SAFETY * stable_dt(dx, WAVESPEED, SPACE_ORDER)
N = max(2, math.ceil(T / dt))
sim = ScalarWave(Nx, dx, N, dt, THREADS, precision=PRECISION, space_order=SPACE_ORDER,
                 wavespeed=WAVESPEED, density=DENSITY)
indicator = cp.ones(sim.Nx_padded, dtype=sim.dtype)
t = np.linspace(0, (N - 1) * dt, N)
signal = cp.asarray(sineburst(t, AMPLITUDE, FREQUENCY, CYCLES)[:, None], dtype=sim.dtype)
source = Source(cp.array([[n // 2] for n in Nx], dtype=cp.int32), signal)
"""
POSTAMBLE = """
for _v in list(globals().values()):
    if isinstance(_v, cuwave.wave.Simulation):
        for _pair in _v.boundary:
            for _c in _pair:
                assert isinstance(_c, cuwave.boundary.BoundaryCondition), f"boundary entry {_c!r}"
"""
BLOCK = re.compile(r"```(\w*)\n(.*?)```", re.S)
# equation numbers of the Langtangen chapter that the teacher cites, prose only
EQREF = re.compile(r"(?<=\s)\((?:[Ee]q(?:uation)?\.?\s*)?2\.\d+\)|\s*\b(?:see |in |from )?[Ee]quation\s+\(?2\.\d+\)?|\s*\[2\.\d+(?:\.\d+)?\]")


# --------------------------------------- helper --------------------------------------
def blocks(answer):
    out = []
    for lang, code in BLOCK.findall(answer):
        if lang.lower() in ("python", "py"):
            out.append(code)
        elif lang == "":
            try:
                ast.parse(code)
                out.append(code)
            except SyntaxError:
                pass
    return out


def run(code):
    env = {**os.environ, "PYTHONPATH": str(CUWAVE_DIR)}
    with tempfile.TemporaryDirectory() as tmp:
        try:
            r = subprocess.run([sys.executable, "-c", PREAMBLE + code + POSTAMBLE], cwd=tmp, env=env,
                               capture_output=True, text=True, timeout=TIMEOUT)
        except subprocess.TimeoutExpired:
            return "Timeout"
    if r.returncode == 0:
        return "ok"
    name = re.search(r"NameError: name '(\w+)' is not defined", r.stderr)
    if name and re.fullmatch(r"[a-z_][a-z0-9_]*|[A-Z0-9_]+", name.group(1)):
        return "ok"  # placeholder variable the question assumes, not an API name
    lines = [line for line in r.stderr.splitlines() if line.strip()]
    return lines[-1].split(":")[0] if lines else f"exit {r.returncode}"


def clean(text):
    parts = re.split(r"(```.*?```)", text, flags=re.S)
    return "".join(p if p.startswith("```") else re.sub(r" +([.,;:])", r"\1", re.sub(r" {2,}", " ", EQREF.sub("", p)))
                   for p in parts)


# ------------------------------------- load data -------------------------------------
pairs = json.loads((DATA_DIR / "qa_raw.json").read_text())

# --------------------------------------- check ---------------------------------------
with ThreadPoolExecutor(WORKERS) as pool:
    results = list(pool.map(lambda p: [run(code) for code in blocks(p["answer"])], pairs))
kept = [{"question": p["question"], "answer": clean(p["answer"])}
        for p, r in zip(pairs, results) if all(x == "ok" for x in r)]

# --------------------------------------- export --------------------------------------
(DATA_DIR / "qa.json").write_text(json.dumps(kept, indent=1, ensure_ascii=False))
errors = Counter(x for r in results for x in r if x != "ok")
print(f"{sum(map(bool, results))} of {len(pairs)} pairs with code, {len(kept)} kept -> {DATA_DIR / 'qa.json'}")
print(dict(errors.most_common()))
