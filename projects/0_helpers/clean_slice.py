import numpy as np
from collections import deque
import matplotlib.pyplot as plt

data = np.load('output/B_Hai_1.npy')

ROWS, COLS = data.shape
mask = np.zeros((ROWS, COLS), dtype=bool)

neighbors = [[0, 1], [1, 0], [0, -1], [-1, 0]]
def bfs(r, c):
    queue = deque()
    queue.append((r, c))
    mask[r, c] = True
    while queue:
        r, c = queue.popleft()
        for dr, dc in neighbors:
            r_ = r + dr
            c_ = c + dc
            if (0 <= r_ < ROWS and 0 <= c_ < COLS and
                data[r_, c_] == 1 and mask[r_, c_] == 0):
                queue.append((r_, c_))
                mask[r_, c_] = True
    return

r0, c0 = 10, 10
print(data[r0, c0]) # check: should be 1
bfs(r0, c0)
mask[data == 0] = True
data[mask != True] = 0.

fig, ax = plt.subplots()
ax.imshow(data, cmap='Grays')
plt.show()

np.save('output/B_Hai_1.npy', data)
