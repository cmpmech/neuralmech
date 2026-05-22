from pathlib import Path

import matplotlib.pyplot as plt
import nrrd
import numpy as np

BASE_DIR = Path(__file__).parent
EXT_DATA_DIR = BASE_DIR / "../../external_data"
DATA_DIR = BASE_DIR / "../../data"

data, _ = nrrd.read(EXT_DATA_DIR / "B-HAI-1.nrrd")
data = np.array(data)

data = data[data.shape[0] // 2, :, :]  # center slice

data_min = np.min(data)
data_max = np.max(data)
data = (data - data_min) / (data_max - data_min)
data[data < 0.5] = 0
data[data > 0.5] = 1

data = data[256:772, 1:1060]

fig, ax = plt.subplots()
ax.imshow(data.T, origin="lower", cmap="binary")
ax.axis("off")
fig.tight_layout(pad=0)
plt.show()

np.save(RES_DIR / "B_Hai_1.npy", data)
