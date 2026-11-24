import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import torch
from sklearn.linear_model import RidgeCV
from transformers import AutoModelForCausalLM, AutoTokenizer

from postprocessing import save_csv

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
CSV_DIR = (RESULTS_DIR / "data").resolve()
DATA_DIR = (BASE_DIR / "../../external_data").resolve()

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
MODEL = "Qwen/Qwen3-1.7B"
SAMPLES = 10000  # most viewed places
CONTINENTS = ["North America", "South America", "Europe", "Africa", "Asia", "Oceania"]
REGULARIZATION = np.logspace(-1, 4, 11)  # ridge penalties, chosen by cross-validation per layer
BATCH_SIZE = 256


# --------------------------------------- helper --------------------------------------
@torch.no_grad()
def last_states(names):
    ids = tokenizer(names, return_tensors="pt", padding=True).to(device)
    states = model.model(**ids, output_hidden_states=True).hidden_states  # embedding + every layer, no logits
    return torch.stack([h[:, -1].float() for h in states], dim=1).cpu()


# -------------------------------------- load data ------------------------------------
countries = json.load(open(DATA_DIR / "natural_earth/ne_110m_admin_0_countries.geojson"))
continent_of = {}
for feature in countries["features"]:
    for key in ("NAME", "NAME_LONG", "ADMIN", "NAME_EN", "SOVEREIGNT", "FORMAL_EN"):
        if feature["properties"].get(key):
            continent_of[feature["properties"][key].lower()] = feature["properties"]["CONTINENT"]
places = []
for row in csv.DictReader(open(DATA_DIR / "world_places/world_place.csv")):
    continent = continent_of.get(row["country"].replace("_", " ").lower())
    if continent in CONTINENTS and row["page_views"]:
        places.append(row)
places = sorted(places, key=lambda p: -float(p["page_views"]))[:SAMPLES]
names = [p["name"] for p in places]
Y = np.array([[float(p["latitude"]), float(p["longitude"])] for p in places])
test = np.array([p["is_test"] == "True" for p in places])

# ------------------------------------- load model ------------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(device)
model.eval()

# -------------------------------------- recording ------------------------------------
features = torch.cat([last_states(names[i:i + BATCH_SIZE]) for i in range(0, len(names), BATCH_SIZE)]).numpy()
r2, predictions = [], []
for layer in range(features.shape[1]):
    X = features[:, layer]
    mean, std = X[~test].mean(0), X[~test].std(0) + 1e-6
    probe = RidgeCV(alphas=REGULARIZATION).fit((X[~test] - mean) / std, Y[~test])
    Y_pred = probe.predict((X[test] - mean) / std)
    r2.append(1 - ((Y_pred - Y[test]) ** 2).sum(0) / ((Y[test] - Y[test].mean(0)) ** 2).sum(0))
    predictions.append(Y_pred)
    print(f"layer {layer}: r2 latitude {r2[-1][0]:.2f}, longitude {r2[-1][1]:.2f}")
r2 = np.array(r2)
best = int(np.argmax(r2.mean(1)))
error = np.linalg.norm(predictions[best] - Y[test], axis=1)  # distance in degrees of latitude and longitude
print(f"{len(names)} places, {test.sum()} held out; best layer {best}")

# ----------------------------------- postprocessing ----------------------------------
land = json.load(open(DATA_DIR / "natural_earth/ne_110m_land.geojson"))
rings = [np.array(ring) for feature in land["features"] for polygon in (
    feature["geometry"]["coordinates"] if feature["geometry"]["type"] == "MultiPolygon"
    else [feature["geometry"]["coordinates"]]) for ring in polygon[:1]]

if not args.book:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), width_ratios=[3, 1])
    for ring in rings:
        axes[0].fill(ring[:, 0], ring[:, 1], color="0.9")
    axes[0].scatter(predictions[best][:, 1], predictions[best][:, 0], c=error, cmap="hot_r", s=3,
                    norm=LogNorm(1, 100))
    axes[0].set_xlim(-180, 180)
    axes[0].set_ylim(-60, 85)
    axes[1].plot(r2[:, 0], "k")
    axes[1].plot(r2[:, 1], "b")
    plt.show()
# -------------------------------- book postprocessing --------------------------------
else:
    save_csv(CSV_DIR / "world_map_r2.csv", layer=np.arange(len(r2)), latitude=r2[:, 0], longitude=r2[:, 1])
    order = np.argsort(-error)  # accurate places drawn last, on top
    save_csv(CSV_DIR / "world_map_places.csv", lon=predictions[best][order, 1], lat=predictions[best][order, 0],
             error=error[order])
    with open(CSV_DIR / "world_map_land.csv", "w") as file:
        file.write("lon lat\n")
        for ring in rings:
            file.write("".join(f"{lon:.2f} {lat:.2f}\n" for lon, lat in ring) + "\n")
