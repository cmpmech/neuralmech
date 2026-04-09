from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from scipy.spatial.distance import mahalanobis

BASE_DIR = Path(__file__).parent
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ----------------------------- select case ------------------------------
samples = 64

problem = 'k'
# problem = 'amp'

path = BASE_DIR / f'../../data/anomaly_3dof_{problem}.npz'
data = np.load(path)
x_a = torch.from_numpy(data['X']).to(torch.float32).to(device)
widths = data['widths']
heights = data['heights']

if problem == 'k':
    anomalous = heights < 0.9
    normal = heights >= 0.9

    anomalous1 = (heights < 0.9) & (heights >= 0.7)
    anomalous2 = (heights < 0.7) & (heights >= 0.5)
    anomalous3 = heights < 0.5

elif problem == 'amp':
    anomalous = heights > 1.5
    normal = heights <= 1.5

    anomalous1 = (heights > 1.5) & (heights <= 3.5)
    anomalous2 = (heights > 3.5) & (heights <= 5.5)
    anomalous3 = heights > 5.5

path = BASE_DIR / '../../data/normal_3dof_test.npy'
x_n = torch.from_numpy(np.load(path)).to(torch.float32).to(device)

# -------------------------- load trained model --------------------------
seq_len = 128
model = torch.load(BASE_DIR / f'../../models/vae_3dof_{seq_len}.pt2', weights_only=False).to(device)
model.eval()
standardizex = model.standardizer

# ------------------------ compute reconstruction ------------------------
def process_with_model(x):
    x = x.reshape(*x.shape[:-1], x.shape[-1] // seq_len, seq_len).transpose(-2, -3)
    x_in = x.reshape(-1, 3, seq_len)
    x_pred = standardizex.inverse(model(standardizex(x_in))[0]) # neglect mean, std
    x_pred = x_pred.reshape(*x.shape[:-2], 3, seq_len)
    recon_error = torch.mean((x_pred - x)[...,1:,:,:]**2, dim=(-2,-1)) # skip first segment
    # recon_error = torch.mean(recon_error, dim=(-1))
    recon_error = torch.max(recon_error, dim=(-1)).values

    return x_pred, recon_error
with torch.no_grad():
    x_pred_n, recon_error_n = process_with_model(x_n)
    x_pred_a, recon_error_a = process_with_model(x_a)

# --------------------------- post-processing ----------------------------
combined_data = torch.cat([recon_error_n.cpu().flatten(),
                           recon_error_a.cpu().flatten()])
bins = np.geomspace(1e-6, float(combined_data.max()), 40)

fig, ax = plt.subplots()
ax.hist(recon_error_n.cpu().flatten(), bins=bins, color='r', alpha=1)
# ax.hist(recon_error_a[anomalous,0].cpu().flatten(), bins=bins, color='purple', alpha=0.5)
ax.hist(recon_error_a[anomalous1].cpu().flatten(), bins=bins, color='b', alpha=0.2) # for amp: 1, for k 1:
ax.hist(recon_error_a[anomalous2].cpu().flatten(), bins=bins, color='b', alpha=0.4)
ax.hist(recon_error_a[anomalous3].cpu().flatten(), bins=bins, color='b', alpha=0.6)
ax.set_xscale('symlog', linthresh=1e-6)
plt.show()

# --------------------------------- AUC ----------------------------------
thresholds = np.logspace(-5, 0, 100)
FP1 = torch.tensor([torch.sum(recon_error_n > thresh).item()
                    for thresh in thresholds])
FP2 = torch.tensor([torch.sum(recon_error_a[normal] > thresh).item()
                    for thresh in thresholds])
FP = FP1 + FP2

TP = torch.tensor([torch.sum(recon_error_a[anomalous] > thresh).item()
                   for thresh in thresholds])

TPR = TP / recon_error_a[anomalous].numel() # recall
FPR = FP / (recon_error_n.numel() + recon_error_a[normal].numel())
precision = TP / (TP + FP)
precision[TP == 0] = 0 # divison by 0

AUC_ROC = auc(FPR, TPR)
AUC_PR = auc(TPR, precision)
print(f'auc_roc = {AUC_ROC:.3f}, auc_pr = {AUC_PR:.3f}')

fig, ax = plt.subplots(1,2)
ax[0].plot(FPR, TPR, 'k')
ax[1].plot(TPR, precision, 'k') # recall
ax[0].set_ylim(0,1.05)
ax[1].set_ylim(0,1.05)
plt.show()

# ---------------------------- reconstruction ----------------------------
T = 6.4 * 4
dof = 1
if problem == 'k':
    w, h = 2, 6
elif problem == 'amp':
    # w, h = 6, 8
    # w, h = 4, 4
    w, h = 2, 2

print(f'anomaly: width = {widths[w,h]:.2f}, height = {heights[w,h]:.2f}')
# x_pred_a = x_pred_a.reshape(*x_pred_a.shape[:-3], 3, -1)
x_pred_a = torch.cat([x_pred_a[:,:,i,:,:] for i in range(x_pred_a.shape[-3])], -1)

t = np.linspace(0, T, 128*4)
fig, ax = plt.subplots()
ax.plot(t, x_pred_a[w,h,dof].cpu(), 'k')
ax.plot(t, x_a[w,h,dof].cpu(), 'r--')
ax.plot([11.4-widths[w,h], 11.4-widths[w,h]], [-0.05, 0.05], 'b--')
ax.plot([11.4+widths[w,h], 11.4+widths[w,h]], [-0.05, 0.05], 'b--')
ax2 = ax.twinx()
# ax2.plot(t, torch.abs(x_pred_a - x_a)[w,h,dof].cpu(), 'k', alpha=0.5)
ax2.plot(t, (x_pred_a - x_a)[w,h,dof].cpu()**2, 'k', alpha=0.5)
ax2.set_ylim(0, 5e-4)
plt.show()

# ----------------------------- mahalanobis ------------------------------
path = BASE_DIR / f'../../data/normal_3dof_{seq_len}.npy'
x_train = torch.from_numpy(np.load(path)).to(torch.float32).to(device)
with torch.no_grad():
    z_train = model.encode(standardizex(x_train))

mean = np.mean(z_train.cpu().numpy(), axis=0)
cov = np.cov(z_train.cpu().numpy().T)
cov_inv = np.linalg.inv(cov)

def process_mahalanobis(x):
    x = x.reshape(*x.shape[:-1], x.shape[-1] // seq_len, seq_len).transpose(-2, -3)
    x_in = x.reshape(-1, 3, seq_len)
    with torch.no_grad():
        z = model.encode(standardizex(x_in))
    scores = torch.tensor([mahalanobis(z_i, mean, cov_inv) for z_i in z.cpu()])
    scores = scores.reshape(*x.shape[:-2])
    return scores

maha_n = process_mahalanobis(x_n)
maha_a = process_mahalanobis(x_a)

combined_data = torch.cat([maha_n.cpu().flatten(),
                           maha_a.cpu().flatten()])
bins = np.linspace(float(combined_data.min()), float(combined_data.max()), 40)

fig, ax = plt.subplots()
ax.hist(maha_n.cpu().flatten(), bins=bins, color='r', alpha=1)
ax.hist(maha_a[anomalous1].cpu().flatten(), bins=bins, color='b', alpha=0.2) # for amp: 1, for k 1:
ax.hist(maha_a[anomalous2].cpu().flatten(), bins=bins, color='b', alpha=0.4)
ax.hist(maha_a[anomalous3].cpu().flatten(), bins=bins, color='b', alpha=0.6)
plt.show()

# maha_n = torch.mean(maha_n[...,1:], -1)
# maha_a = torch.mean(maha_a[...,1:], -1)
maha_n = torch.max(maha_n[...,1:], dim=-1).values
maha_a = torch.max(maha_a[...,1:], dim=-1).values

thresholds = np.logspace(-1, 3, 100)
FP1 = torch.tensor([torch.sum(maha_n > thresh).item()
                    for thresh in thresholds])
FP2 = torch.tensor([torch.sum(maha_a[normal] > thresh).item()
                    for thresh in thresholds])
FP = FP1 + FP2

TP = torch.tensor([torch.sum(maha_a[anomalous] > thresh).item()
                   for thresh in thresholds])

TPR = TP / maha_a[anomalous].numel() # recall
FPR = FP / (maha_n.numel() + maha_a[normal].numel())
precision = TP / (TP + FP)
precision[TP == 0] = 0 # divison by 0

AUC_ROC = auc(FPR, TPR)
AUC_PR = auc(TPR, precision)
print(f'auc_roc = {AUC_ROC:.3f}, auc_pr = {AUC_PR:.3f}')