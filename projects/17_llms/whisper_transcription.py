import argparse
import io
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile
import torch
from datasets import Audio, load_dataset
from transformers import WhisperForConditionalGeneration, WhisperProcessor

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
RGB_PNG_DIR = (RESULTS_DIR / "rgb_png").resolve()

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
MODEL = "openai/whisper-large-v3-turbo"
SAMPLE = 0  # utterance of the LibriSpeech excerpt

# -------------------------------------- load data ------------------------------------
dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
dataset = dataset.cast_column("audio", Audio(decode=False))  # raw flac bytes, already 16 kHz
audio, _ = soundfile.read(io.BytesIO(dataset[SAMPLE]["audio"]["bytes"]))
reference = dataset[SAMPLE]["text"]

# ------------------------------------- load model ------------------------------------
processor = WhisperProcessor.from_pretrained(MODEL)
model = WhisperForConditionalGeneration.from_pretrained(MODEL, torch_dtype=torch.float16).to(device)
model.eval()

# ------------------------------------- inference -------------------------------------
features = processor(audio, sampling_rate=16_000, return_tensors="pt").input_features
with torch.no_grad():
    ids = model.generate(features.to(device, torch.float16), language="en", task="transcribe")
transcription = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()

# ----------------------------------- postprocessing ----------------------------------
print(f"reference:     {reference}")
print(f"transcription: {transcription}")

frames = int(len(audio) / 16_000 * 100)  # 10 ms per frame, rest of the 30 s window is padding
spectrogram = features[0, :, :frames].numpy()
if not args.book:
    fig, axes = plt.subplots(2, 1, figsize=(10, 5))
    axes[0].plot(np.arange(len(audio)) / 16_000, audio, "k")
    axes[1].imshow(spectrogram, origin="lower", aspect="auto", cmap="cividis")
    plt.show()
# -------------------------------- book postprocessing --------------------------------
else:
    fig, ax = plt.subplots(figsize=(8, 2))
    ax.imshow(spectrogram, origin="lower", aspect="auto", cmap="cividis", interpolation="none")
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(RGB_PNG_DIR / "whisper_spectrogram.png", dpi=200)
    (RESULTS_DIR / "data" / "whisper_transcription.txt").write_text(
        f"duration {len(audio) / 16_000:.2f} s\nreference: {reference}\ntranscription: {transcription}\n")
