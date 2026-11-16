import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from postprocessing import save_csv

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
CSV_DIR = (RESULTS_DIR / "data").resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
RESOLUTION = 1024  # samples on one period
HARMONICS = [1, 3, 9, 27]
COEFFICIENTS = 32  # shown in the coefficient decay

x = (np.arange(RESOLUTION) + 0.5) / RESOLUTION  # midpoints keep square wave odd
FUNCTIONS = {
    "smooth": np.exp(np.sin(2 * np.pi * x)),
    "square": np.sign(np.sin(2 * np.pi * x)),
}


# --------------------------------------- helper --------------------------------------
def truncate(f, harmonics):
    c = np.fft.rfft(f)
    c[harmonics + 1 :] = 0
    return np.fft.irfft(c, n=len(f))


# ----------------------------------- postprocessing ----------------------------------
series = {name: [truncate(f, K) for K in HARMONICS] for name, f in FUNCTIONS.items()}
amplitudes = {
    name: 2 * np.abs(np.fft.rfft(f)) / RESOLUTION for name, f in FUNCTIONS.items()
}

overshoot = (series["square"][-1].max() - 1) / 2
print(f"overshoot with {HARMONICS[-1]} harmonics: {100 * overshoot:.1f} % of the jump")
for name, a in amplitudes.items():
    print(
        f"{name}: amplitude of harmonic 1, 9, 27: {a[1]:.1e}, {a[9]:.1e}, {a[27]:.1e}"
    )

if not args.book:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (name, f) in zip(axes, FUNCTIONS.items()):
        ax.plot(x, f, "k--")
        for truncated in series[name]:
            ax.plot(x, truncated)
    k = np.arange(1, COEFFICIENTS + 1)
    axes[2].semilogy(k, amplitudes["smooth"][k], "ko")
    axes[2].semilogy(k, amplitudes["square"][k], "ro")
    plt.show()
# -------------------------------- book postprocessing --------------------------------
else:
    for name, f in FUNCTIONS.items():
        columns = {f"K{K}": s for K, s in zip(HARMONICS, series[name])}
        save_csv(CSV_DIR / f"fourier_series_{name}.csv", x=x, exact=f, **columns)
    for name, a in amplitudes.items():
        k = np.arange(1, COEFFICIENTS + 1)
        k = k[a[k] > 1e-14]  # drop round-off and the vanishing even harmonics
        save_csv(CSV_DIR / f"fourier_series_amplitudes_{name}.csv", k=k, amplitude=a[k])
