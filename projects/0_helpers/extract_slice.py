import numpy as np
import nrrd
import matplotlib.pyplot as plt

data, _ = nrrd.read('B-HAI-1.nrrd')
data = np.array(data)

data = data[data.shape[0] // 2, :, :] # center slice

data_min = np.min(data)
data_max = np.max(data)
data = (data - data_min) / (data_max - data_min)
data[data < 0.5] = 0
data[data > 0.5] = 1

data = data[256:772, 1:1060]

fig, ax = plt.subplots()
ax.imshow(data, cmap='Grays')
plt.show()

np.save('output/B_Hai_1.npy', data)