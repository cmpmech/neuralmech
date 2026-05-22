from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = Path(__file__).parent

data = np.load(BASE_DIR / "output/B_Hai_1.npy")

ROWS, COLS = data.shape
mask = np.zeros((ROWS, COLS), dtype=bool)

neighbors = [[0, 1], [1, 0], [0, -1], [-1, 0]]


def bfs(r, c):
    queue = deque([(r, c)])
    mask[r, c] = True
    while queue:
        r, c = queue.popleft()
        for dr, dc in neighbors:
            r_ = r + dr
            c_ = c + dc
            if (
                0 <= r_ < ROWS
                and 0 <= c_ < COLS
                and data[r_, c_] == 1
                and not mask[r_, c_]
            ):
                queue.append((r_, c_))
                mask[r_, c_] = True


r0, c0 = 10, 10
print(data[r0, c0])  # check: should be 1
bfs(r0, c0)
mask[data == 0] = True
data[~mask] = 0.0

fig, ax = plt.subplots()
ax.imshow(data, cmap="Grays")
plt.show()

np.save(BASE_DIR / "output/B_Hai_1.npy", data)
