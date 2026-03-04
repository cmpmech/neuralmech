import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage

np.random.seed(0)

# -------------------------------- helper --------------------------------
def Gabor(a, b):
    return lambda x : (np.sin(a + 0.06 * b * x) *
                       np.exp(-(a + 0.06 * b * x)**2 / 32.))

def cost(ypred, y):
    return np.mean((y - ypred)**2)

def cost_grad(a, b, x, y):
    ypred = Gabor(a, b)(x)
    dc_dypred = (y - ypred)
    sin = np.sin(a + 0.06 * b * x)
    cos = np.cos(a + 0.06 * b * x)
    gaussian = np.exp(-(a + 0.06 * b * x)**2 / 32)
    dypred_da = (-sin * gaussian * (a + 0.06 * b * x) / 16. +
                 cos * gaussian)
    dypred_db = (-sin * gaussian * 0.06 * x * (a + 0.06 * b * x) / 16. +
                 0.06 * x * cos * gaussian)
    return np.array([-2 * np.mean(dc_dypred * dypred_da),
                     -2 * np.mean(dc_dypred * dypred_db)])

def optimize(params0, lr, epochs, batch_size):
    param_history = []
    param_history.append(params0.copy())
    indices = np.arange(len(x))
    params = params0.copy()
    for epoch in range(epochs):
        np.random.shuffle(indices)
        for batch in range(samples // batch_size):
            batchindices = indices[batch*batch_size:(batch+1)*batch_size]
            grad = cost_grad(params[0], params[1], x[batchindices], y[batchindices])
            params -= lr * grad
            param_history.append(params.copy())
    param_history = np.vstack(param_history)
    return param_history

def find_local_minima(a, b, landscape):
    footprint = np.ones((3, 3))
    min_filtered = ndimage.minimum_filter(landscape, footprint=footprint)
    local_minima_mask = (landscape == min_filtered)
    indices = np.where(local_minima_mask)
    return np.vstack([a[indices], b[indices]]).T

# ----------------------------- fitting data -----------------------------
samples = 64
noise = 0.
x = np.random.uniform(-10, 10, samples)
y = Gabor(0, 16)(x) + np.random.uniform(-noise, noise, samples)

# -------------------- sample optimization landscape ---------------------
samples_landscape = 128
a = np.linspace(-10, 10, samples_landscape)
b = np.linspace(1e-4, 20, samples_landscape)
a, b = np.meshgrid(a, b, indexing='ij')

cost_landscape = np.zeros_like(a)
for i in range(len(a)):
    for j in range(len(a[0])):
        y_pred = Gabor(a[i,j], b[i,j])(x)
        cost_landscape[i,j] = cost(y_pred, y)

# ------------------------------ regularize ------------------------------
b0 = 10
regularization = np.sqrt(a**2 + (b - b0)**2)
factor = 0.05
regularized_cost = cost_landscape + factor * regularization

# -------------------------- find local optima ---------------------------
minima_cost  = find_local_minima(a, b, cost_landscape)
minima_regularization  = find_local_minima(a, b, regularization)
minima_regularized_cost  = find_local_minima(a, b, regularized_cost)

# ---------------------------- postprocessing ----------------------------
fig, ax = plt.subplots(figsize=(4, 4), dpi=150)
cb = ax.contourf(a, b, cost_landscape, levels=36, cmap='cividis')
for i in range(len(minima_cost)):
    if abs(minima_cost[i,1] - 16) > 1e-1:
        ax.plot(minima_cost[i,0], minima_cost[i,1], 'wo')
ax.plot(0, 16, 'yo')
ax.set_xlim(-10,10)
ax.set_ylim(0,20)
ax.axis('off')
ax.set_rasterized(True)
fig.tight_layout(pad=0)
plt.savefig(f'../../results/gabor_landscape_standard.pdf')
plt.show()

fig, ax = plt.subplots(figsize=(4, 4), dpi=150)
cb = ax.contourf(a, b, cost_landscape, levels=36, cmap='cividis')
cb = ax.contourf(a, b, regularization, levels=36, cmap='cividis')
ax.plot(minima_regularization[:,0], minima_regularization[:,1], 'ro')
ax.set_xlim(-10,10)
ax.set_ylim(0,20)
ax.axis('off')
ax.set_rasterized(True)
fig.tight_layout(pad=0)
plt.savefig(f'../../results/regularization_landscape.pdf')
plt.show()

fig, ax = plt.subplots(figsize=(4, 4), dpi=150)
cb = ax.contourf(a, b, cost_landscape, levels=36, cmap='cividis')
cb = ax.contourf(a, b, regularization, levels=36, cmap='cividis')
cb = ax.contourf(a, b, regularized_cost, levels=36, cmap='cividis')
ax.plot(minima_regularized_cost[:,0], minima_regularized_cost[:,1], 'wo')
ax.plot(0, 16, 'yo')
ax.plot(0, b0, 'ro')
ax.set_xlim(-10,10)
ax.set_ylim(0,20)
ax.axis('off')
ax.set_rasterized(True)
fig.tight_layout(pad=0)
plt.savefig(f'../../results/regularized_gabor_landscape.pdf')
plt.show()