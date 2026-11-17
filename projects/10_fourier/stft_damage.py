import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh

from postprocessing import save_csv

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
RGB_PDF_DIR = (RESULTS_DIR / "rgb_pdf").resolve()
CSV_DIR = (RESULTS_DIR / "data").resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
# physics: three-story frame, unit floor masses
STIFFNESS = 1e4  # of each story
DAMAGE = 0.5  # stiffness of the first story at the end of the record, fraction of intact
DAMPING = 2e-4  # stiffness-proportional damping coefficient

# discretization
DURATION = 120.0
SAMPLING = 200.0  # samples per second
WINDOW = 1024  # samples per short-time window
HOP = 128  # samples between window starts
SMOOTHING = 60  # neighboring frequencies averaged in the spectrum of the whole record
RANGE = 40  # shown range of the spectrogram in decibel
F_MAX = 40.0  # shown frequency range

rng = np.random.default_rng(0)


# -------------------------------------- helper ---------------------------------------
def stiffness(t):
    k = STIFFNESS * np.array([1 - (1 - DAMAGE) * t / DURATION, 1, 1])
    return np.array([[k[0] + k[1], -k[1], 0], [-k[1], k[1] + k[2], -k[2]], [0, -k[2], k[2]]])


# --------------------------------------- setup ---------------------------------------
dt = 1 / SAMPLING
steps = int(DURATION * SAMPLING)
t = np.arange(steps) * dt
M = np.eye(3)
C = DAMPING * stiffness(0)
force = np.zeros((steps, 3))
force[:, 2] = rng.standard_normal(steps)  # ambient excitation of the top floor

# -------------------------- time integration (average acceleration) ------------------
u, v = np.zeros(3), np.zeros(3)
a = np.zeros(3)
acceleration = np.zeros(steps)  # top floor, as measured by an accelerometer
for n in range(1, steps):
    K = stiffness(t[n])
    u_pred = u + dt * v + dt**2 / 4 * a
    v_pred = v + dt / 2 * a
    a = np.linalg.solve(M + dt / 2 * C + dt**2 / 4 * K, force[n] - C @ v_pred - K @ u_pred)
    u = u_pred + dt**2 / 4 * a
    v = v_pred + dt / 2 * a
    acceleration[n] = a[2]

# ------------------------------ fourier transforms -----------------------------------
spectrum = np.abs(np.fft.rfft(acceleration)) ** 2 / steps
spectrum = np.convolve(spectrum, np.ones(SMOOTHING) / SMOOTHING, mode="same")
frequency = np.fft.rfftfreq(steps, dt)

frames = np.lib.stride_tricks.sliding_window_view(acceleration, WINDOW)[::HOP]
spectrogram = np.abs(np.fft.rfft(frames * np.hanning(WINDOW), axis=1)) ** 2 / WINDOW
frequency_window = np.fft.rfftfreq(WINDOW, dt)
t_window = (np.arange(len(frames)) * HOP + WINDOW / 2) * dt

for time in [0, DURATION]:
    eigenvalues = eigh(stiffness(time), M, eigvals_only=True)
    print(f"t = {time:.0f} s: eigenfrequencies {np.sqrt(eigenvalues) / (2 * np.pi)} Hz")
print(f"frequency resolution {SAMPLING / WINDOW:.2f} Hz, window length {WINDOW * dt:.2f} s")

# ----------------------------------- postprocessing ----------------------------------
keep = frequency_window <= F_MAX
image = 10 * np.log10(spectrogram[:, keep].T)
if not args.book:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].semilogy(frequency, spectrum)
    axes[0].set_xlim(0, F_MAX)
    axes[1].imshow(image, origin="lower", aspect="auto", cmap="cividis", vmin=image.max() - RANGE,
                   extent=[t_window[0], t_window[-1], 0, F_MAX])
    plt.show()
# -------------------------------- book postprocessing --------------------------------
else:
    keep_all = frequency <= F_MAX
    save_csv(CSV_DIR / "stft_spectrum.csv", f=frequency[keep_all][::4], p=spectrum[keep_all][::4])
    fig, ax = plt.subplots(figsize=(len(t_window) / 100, keep.sum() / 100), dpi=100)
    ax.imshow(image, origin="lower", aspect="auto", cmap="cividis", vmin=image.max() - RANGE, interpolation="none")
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(RGB_PDF_DIR / "stft_spectrogram.pdf")
    print(f"spectrogram {len(t_window)} windows from {t_window[0]:.2f} s to {t_window[-1]:.2f} s")
