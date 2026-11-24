import argparse
import io
import urllib.request
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from postprocessing import save_csv

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
CSV_DIR = (RESULTS_DIR / "data").resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
EPOCH = "https://epoch.ai/data/benchmark_data.zip"
ECI = "epoch_capabilities_index/eci_scores.csv"
BITS = 4  # quantized precision of the tail
GPUS = [8, 16, 24, 32, 48, 80, 96, 141, 192, 640]  # memory [GB]
CLOSED = ["Claude 2", "Claude 3 Opus", "Claude Opus 4", "Claude Opus 5", "Claude Opus 5.5"]  # horizontal reference lines

# Epoch display name: (family, total parameters in billions, all experts included)
MODELS = {
    "LLaMA-7B": ("llama", 6.7),
    "LLaMA-13B": ("llama", 13.0),
    "LLaMA-33B": ("llama", 32.5),
    "LLaMA-65B": ("llama", 65.2),
    "Llama 2-7B": ("llama", 6.7),
    "Llama 2-13B": ("llama", 13.0),
    "Llama 2-70B": ("llama", 69.0),
    "Llama 3-8B": ("llama", 8.0),
    "Llama 3-70B": ("llama", 70.6),
    "Llama 3.1-8B": ("llama", 8.0),
    "Llama 3.1-70B": ("llama", 70.6),
    "Llama 3.1-405B": ("llama", 405.9),
    "Llama 3.3 70B": ("llama", 70.6),
    "Llama 4 Scout": ("llama", 109.0),
    "Llama 4 Maverick": ("llama", 400.0),
    "Qwen-7B": ("qwen", 7.7),
    "Qwen-14B": ("qwen", 14.2),
    "Qwen2-72B": ("qwen", 72.7),
    "Qwen2.5-72B": ("qwen", 72.7),
    "QwQ-32B": ("qwen", 32.8),
    "Qwen3-8B": ("qwen", 8.2),
    "Qwen3-14B": ("qwen", 14.8),
    "Qwen3-32B": ("qwen", 32.8),
    "Qwen3-30B-A3B": ("qwen", 30.5),
    "Qwen3-235B-A22B": ("qwen", 235.0),
    "Qwen3.5-9B": ("qwen", 9.0),
    "Qwen3.5-35B-A3B": ("qwen", 35.0),
    "Qwen3.5 397B-A17B": ("qwen", 397.0),
    "Qwen3.6 27B": ("qwen", 27.0),
    "Qwen 3.6 35B-A3B": ("qwen", 35.0),
    "Qwen 3.8 27B": ("qwen", 27.0),
    "Mistral 7B v0.1": ("mistral", 7.2),
    "Mixtral 8x7B": ("mistral", 46.7),
    "Mixtral 8x22B": ("mistral", 140.6),
    "Mistral NeMo": ("mistral", 12.2),
    "Mistral Large 2 (Jul 2024)": ("mistral", 123.0),
    "Mistral Medium 3.5": ("mistral", 128.0),
    "Mistral Large 4": ("mistral", 1050.0),
    "Gemma 2B": ("gemma", 2.5),
    "Gemma 7B": ("gemma", 8.5),
    "Gemma 2 9B": ("gemma", 9.2),
    "Gemma 2 27B": ("gemma", 27.2),
    "Gemma 3 27B": ("gemma", 27.4),
    "Gemma 4 26B A4B": ("gemma", 26.0),
    "Gemma 4 31B IT": ("gemma", 31.0),
    "Phi-1.5": ("phi", 1.4),
    "Phi-2": ("phi", 2.7),
    "phi-3-mini 3.8B": ("phi", 3.8),
    "phi-3-small 7.4B": ("phi", 7.4),
    "phi-3-medium 14B": ("phi", 14.0),
    "Phi-4": ("phi", 14.7),
    "DeepSeek-V2 (MoE-236B, May 2024)": ("deepseek", 236.0),
    "DeepSeek-V3": ("deepseek", 671.0),
    "DeepSeek-R1": ("deepseek", 671.0),
    "DeepSeek-V3.1": ("deepseek", 671.0),
    "DeepSeek-V3.2": ("deepseek", 671.0),
    "DeepSeek-V4-Flash": ("deepseek", 284.0),
    "DeepSeek-V4-Pro": ("deepseek", 1600.0),
    "DeepSeek V4.1 Flash": ("deepseek", 552.0),
    "gpt-oss-20b": ("gptoss", 20.9),
    "gpt-oss-120b": ("gptoss", 116.8),
    "Kimi K2 (Jul 2025)": ("kimi", 1040.0),
    "Kimi K2 Thinking": ("kimi", 1040.0),
    "Kimi K2.5": ("kimi", 1040.0),
    "Kimi K2.6": ("kimi", 1040.0),
    "Kimi K3": ("kimi", 2800.0),
    "GLM-4.7": ("glm", 358.0),
    "GLM-5": ("glm", 744.0),
    "GLM-5.1": ("glm", 754.0),
    "GLM-5.2": ("glm", 744.0),
    "GLM-5.3": ("glm", 744.0),
    "GLM-5.3-Flash": ("glm", 320.0),
}

# ------------------------------------- load data -------------------------------------
request = urllib.request.Request(EPOCH, headers={"User-Agent": "neuralmech"})
with urllib.request.urlopen(request) as response:
    archive = zipfile.ZipFile(io.BytesIO(response.read()))
df = pd.read_csv(archive.open(ECI))

# ----------------------------------- postprocessing ----------------------------------
missing = set(MODELS) - set(df["Display name"])
print(f"not in ECI table: {sorted(missing)}")
day = pd.to_datetime(df["date"])
df["year"] = day.dt.year + (day.dt.dayofyear - 1) / 365.25
closed = df.set_index("Display name").loc[CLOSED]
for name, model in closed.iterrows():
    print(f"{name}: ECI {model.eci:.1f}, released {model.year:.2f}")
df = df[df["Display name"].isin(MODELS)].copy()
df["family"] = df["Display name"].map(lambda name: MODELS[name][0])
df["memory"] = df["Display name"].map(lambda name: 2 * MODELS[name][1])  # 16 bits
families = dict(tuple(df.groupby("family", sort=False)))
for family, group in families.items():
    print(f"{family}: {len(group)} models, ECI {group.eci.min():.0f} to {group.eci.max():.0f}")

if not args.book:
    for gpu in GPUS:
        plt.axvline(gpu, color="0.7", ls="--")
    for eci in closed.eci:
        plt.axhline(eci, color="k", lw=0.5)
    for family, group in families.items():
        plt.hlines(group.eci, group.memory * BITS / 16, group.memory, color="0.8")
        plt.scatter(group.memory, group.eci, c=group.year, cmap="cividis", vmin=2023, vmax=2027)
    plt.xscale("log")
    plt.xlabel("weight memory at 16 bits [GB]")
    plt.ylabel("Epoch Capabilities Index")
    plt.show()
# -------------------------------- book postprocessing --------------------------------
else:
    for family, group in families.items():
        save_csv(CSV_DIR / f"open_weight_{family}.csv", x=group.memory.to_numpy(),
                 y=group.eci.to_numpy(), t=group.year.round(3).to_numpy())
    save_csv(CSV_DIR / "open_weight_closed.csv", y=closed.eci.to_numpy(), t=closed.year.round(3).to_numpy())
