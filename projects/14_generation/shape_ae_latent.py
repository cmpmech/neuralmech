from pathlib import Path

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch

from postprocessing import save_csv

BASE_DIR = Path(__file__).parent
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True

# ------------------------------ load data -------------------------------
domain_size = 128

labels = ['circle', 'ellipse', 'square', 'triangle', 'cross', 'star']
data = []
for label in labels:
    data.append(torch.from_numpy(
        np.load(BASE_DIR / f"../../data/shapes_{label}_{domain_size}.npy")))
    data[-1] = data[-1].to(torch.float32).unsqueeze(1)



# # TESSTING
# test = data[3][0,0]
# for i in range(1, 4):
#     test += data[3][i,0]

# fig, ax = plt.subplots()
# ax.imshow(test, origin='lower')
# plt.show()



# # -------------------------- load trained model --------------------------
# model = torch.load(BASE_DIR / f'../../models/shape_ae_{domain_size}.pt2', weights_only=False, map_location=device)
# model.eval()
# standardizex = model.standardizer

# # ----------------------- latent space projection ------------------------
# latents = []
# with torch.no_grad():
#     for shape in data:
#         latents.append(model.encode(standardizex(shape)).cpu())

# # ------------------------ latent space sampling -------------------------
# box = 2

# samplesx = 10
# samplesy = 6

# if box == 1:
#     x = torch.linspace(-0.5, 0.5, samplesx)
#     y = torch.linspace(-0.5, 0.1, samplesy)
# elif box == 2:
#     x = torch.linspace(0.5, 1, samplesx)
#     y = torch.linspace(0.6, 0.9, samplesy)

# x, y = torch.meshgrid(x, y, indexing='ij')
# z = torch.cat([x.reshape(-1, 1), y.reshape(-1, 1)], dim=1)

# with torch.no_grad():
#     gen_shapes = standardizex.inverse(model.decode(z))
# gen_shapes = gen_shapes.reshape(*x.shape, domain_size, domain_size)

# # ---------------------------- postprocessing ----------------------------
# colors = cm.tab10(np.linspace(0, 1, len(latents)))

# fig, ax = plt.subplots()
# for i, latent in enumerate(latents):
#     ax.plot(latent[:,0], latent[:,1], 'o', color=colors[i])
# # ax.plot([x[0,0], x[-1,0]], [y[0,0], y[0,-1]], 'k')
# ax.plot(x.flatten(), y.flatten(), 'k.')
# ax.set_aspect("equal")
# plt.show()

# fig, ax = plt.subplots(samplesy, samplesx, figsize=(samplesx,samplesy))
# for i in range(samplesx):
#     for j in range(samplesy):
#         ax[j,i].imshow(gen_shapes[i, j].cpu(), cmap="binary", origin='lower')
#         ax[j,i].set_aspect("equal")
#         ax[j,i].axis("off")
#         ax[j,i].set_rasterized(True)
# fig.tight_layout(pad=0)
# plt.show()

# # ------------------------- book postprocessing --------------------------
# for label, latent in zip(labels, latents):
#     save_csv(BASE_DIR / f'../../results/shapes_ae_latent_{label}.csv',
#             x=latent[:,0], y=latent[:,1])

# for i in range(samplesx):
#     for j in range(samplesy):
#         fig, ax = plt.subplots(figsize=(1, 1), dpi=domain_size)
#         ax.imshow(gen_shapes[i, j].cpu(), cmap="binary", origin='lower', vmin=0, vmax=1)
#         ax.set_aspect("equal")
#         ax.axis("off")
#         ax.set_rasterized(True)
#         fig.tight_layout(pad=0)
#         plt.savefig(
#             BASE_DIR / f"../../results/genshapes_ae_{i}{j}_{box}.pdf", bbox_inches="tight", pad_inches=0
#         )
#         plt.close()
