import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from postprocessing import save_csv

BASE_DIR = Path(__file__).parent
DATA_DIR = (BASE_DIR / "../../data").resolve()
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
CSV_DIR = (RESULTS_DIR / "data").resolve()

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
FEATURES = ["cement", "slag", "fly ash", "water", "superplasticizer", "coarse aggregate", "fine aggregate", "age"]
TEST_FRACTION = 0.2

# ------------------------------------- load data -------------------------------------
data = np.load(DATA_DIR / "concrete.npz")  # from concrete_data.py
X_train, X_test, Y_train, Y_test = train_test_split(data["X"], data["Y"], test_size=TEST_FRACTION, random_state=0)

# ------------------------------------- training --------------------------------------
models = {
    "linear regression": make_pipeline(StandardScaler(), LinearRegression()),
    "neural network": make_pipeline(
        StandardScaler(), MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=5000, random_state=0)
    ),
    "random forest": RandomForestRegressor(n_estimators=500, random_state=0),
    "gradient boosting": GradientBoostingRegressor(
        n_estimators=1000, max_depth=4, learning_rate=0.05, subsample=0.8, random_state=0
    ),
}
for name, model in models.items():
    model.fit(X_train, Y_train)
    rmse = np.sqrt(np.mean((model.predict(X_test) - Y_test) ** 2))
    r2 = model.score(X_test, Y_test)
    print(f"{name}: test rmse {rmse:.2f} MPa, r2 {r2:.3f}")

boosting = models["gradient boosting"]
Y_pred = boosting.predict(X_test)
importance = permutation_importance(boosting, X_test, Y_test, n_repeats=20, random_state=0)
order = np.argsort(importance.importances_mean)
for i in order[::-1]:
    print(f"{FEATURES[i]}: {importance.importances_mean[i]:.3f}")

# ----------------------------------- postprocessing ----------------------------------
if not args.book:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(Y_test, Y_pred, s=6)
    axes[0].plot([0, 85], [0, 85], "k--")
    axes[1].barh(np.array(FEATURES)[order], importance.importances_mean[order])
    plt.show()
# -------------------------------- book postprocessing --------------------------------
else:
    save_csv(CSV_DIR / "concrete_test.csv", y=Y_test, y_pred=Y_pred)
    save_csv(CSV_DIR / "concrete_importance.csv", rank=np.arange(len(order)), value=importance.importances_mean[order])
    print("feature order (bottom to top):", [FEATURES[i] for i in order])
