import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import timesfm
import torch
from chronos import Chronos2Pipeline
from scipy.integrate import solve_ivp
from timestral import TimestralForecaster

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
CONTEXT = 1024  # observed time steps
HORIZON = 256  # forecast time steps
DT = {"duffing": 0.1, "lorenz": 0.01}  # time step per signal
SHOWN = 256  # context steps shown in the plot
MODELS = ["timestral", "chronos", "timesfm"]


# --------------------------------------- helper --------------------------------------
def duffing(t, u):
    x, v = u
    return [v, -0.1 * v - x - 0.5 * x**3 + 0.8 * np.cos(1.2 * t) + 0.6 * np.cos(0.37 * t)]


def lorenz(t, u):
    x, y, z = u
    return [10 * (y - x), x * (28 - z) - y, x * y - 8 / 3 * z]


def simulate(rhs, u0, steps, dt):
    t = np.arange(steps) * dt
    solution = solve_ivp(rhs, (0, t[-1]), u0, t_eval=t, rtol=1e-9, atol=1e-9)
    return solution.y[0]


# ------------------------------------- create data -----------------------------------
transient = 1000  # discarded steps before the context
signals = {
    "duffing": simulate(duffing, [1.0, 0.0], transient + CONTEXT + HORIZON, DT["duffing"])[transient:],
    "lorenz": simulate(lorenz, [1.0, 1.0, 1.0], transient + CONTEXT + HORIZON, DT["lorenz"])[transient:],
}

# ------------------------------------- load models -----------------------------------
timestral = TimestralForecaster.from_pretrained("mistralai/Timestral-209M", device=str(device))
chronos = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map=device)
tfm = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
tfm.compile(timesfm.ForecastConfig(max_context=CONTEXT, max_horizon=HORIZON, normalize_inputs=True,
                                   use_continuous_quantile_head=True, fix_quantile_crossing=True))

# -------------------------------------- forecast -------------------------------------
forecasts = {}
for name, signal in signals.items():
    context = signal[:CONTEXT]

    forecast = timestral.forecast(context, prediction_length=HORIZON)
    bands = [forecast.median, forecast.quantile(0.1), forecast.quantile(0.9)]
    forecasts[name, "timestral"] = [torch.as_tensor(b).cpu().numpy().squeeze() for b in bands]

    quantiles, _ = chronos.predict_quantiles([context[None]], prediction_length=HORIZON,
                                             quantile_levels=[0.1, 0.5, 0.9])
    q = quantiles[0][0].cpu().numpy()  # (horizon, quantiles)
    forecasts[name, "chronos"] = (q[:, 1], q[:, 0], q[:, 2])

    _, q = tfm.forecast(horizon=HORIZON, inputs=[context])
    forecasts[name, "timesfm"] = (q[0, :, 5], q[0, :, 1], q[0, :, 9])  # mean, then deciles

# ----------------------------------- postprocessing ----------------------------------
for (name, model), (median, _, _) in forecasts.items():
    truth = signals[name][CONTEXT:]
    error = np.sqrt(np.mean((median - truth) ** 2)) / np.std(truth)
    print(f"{name} {model}: normalized rmse {error:.2e}")

if not args.book:
    fig, axes = plt.subplots(len(signals), len(MODELS), figsize=(12, 5), sharex=True)
    for row, name in zip(axes, signals):
        t = np.arange(CONTEXT + HORIZON) * DT[name]
        for ax, model in zip(row, MODELS):
            median, lower, upper = forecasts[name, model]
            ax.plot(t[CONTEXT - SHOWN:], signals[name][CONTEXT - SHOWN:], "k")
            ax.plot(t[CONTEXT:], median, "r")
            ax.fill_between(t[CONTEXT:], lower, upper, color="r", alpha=0.3)
    plt.show()
# -------------------------------- book postprocessing --------------------------------
else:
    for name, signal in signals.items():
        t = np.arange(CONTEXT + HORIZON) * DT[name]
        save_csv(CSV_DIR / f"forecast_{name}_truth.csv", t=t[CONTEXT - SHOWN:],
                 u=signal[CONTEXT - SHOWN:])
        columns = {"t": t[CONTEXT:]}
        for model in MODELS:
            median, lower, upper = forecasts[name, model]
            columns |= {model: median, f"{model}_lower": lower, f"{model}_upper": upper}
        save_csv(CSV_DIR / f"forecast_{name}.csv", **columns)
