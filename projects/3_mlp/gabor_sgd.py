import numpy as np
import matplotlib.pyplot as plt
from postprocessing import save_csv

np.random.seed(0)

# ---------------------------- Gabor helpers -----------------------------
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

# ---------------------- optimization trajectories -----------------------
lr = 1e0
epochs = 500
# select batch_size
# batch_size = 1
batch_size = 16
# batch_size = 64

params0 = np.array([-1.5, 2.])
history0 = optimize(params0, lr, epochs, batch_size)
params1 = np.array([3., 18.])
history1 = optimize(params1, lr, epochs, batch_size)
params2 = np.array([-3.1, 10.])
history2 = optimize(params2, lr, epochs, batch_size)
params3 = np.array([2.7, 12.])
history3 = optimize(params3, lr, epochs, batch_size)

# ---------------------------- postprocessing ----------------------------
fig, ax = plt.subplots(figsize=(4, 4), dpi=150)
cb = ax.contourf(a, b, cost_landscape, levels=36, cmap='cividis')
ax.plot(history0[:,0], history0[:,1], 'brown', linewidth=2, alpha=0.8)
ax.plot(history0[0,0], history0[0,1], 'o', color='brown')
ax.plot(history2[:,0], history2[:,1], 'salmon', linewidth=2, alpha=0.8)
ax.plot(history2[0,0], history2[0,1], 'o', color='salmon')
ax.plot(history3[:,0], history3[:,1], 'red', linewidth=2, alpha=0.8)
ax.plot(history3[0,0], history3[0,1], 'o', color='red')
ax.plot(history1[:,0], history1[:,1], 'maroon', linewidth=2, alpha=0.8)
ax.plot(history1[0,0], history1[0,1], 'o', color='maroon')
ax.plot(0, 16, 'wo')

ax.set_xlim(-10,10)
ax.set_ylim(0,20)
ax.axis('off')
ax.set_rasterized(True)
fig.tight_layout(pad=0)
plt.savefig(f'../../results/gabor_landscape_{batch_size}.pdf')
plt.show()

# ------------------------- book postprocessing --------------------------
save_csv(f'../../results/gabor_data.csv', x=x.squeeze(), y=y.squeeze())

if batch_size == 16:
    x_ = np.linspace(-30, 30, 400)
    for i, (a, b) in enumerate(history0[:500:40]):
        y_ = Gabor(a, b)(x_)
        save_csv(f'../../results/gabor_prediction_{i}.csv', x=x_, y=y_)

x_ = np.linspace(-30, 30, 200)
fig, ax = plt.subplots()
for i, (a, b) in enumerate(history0[:600:40]):
    y_ = Gabor(a, b)(x_)
    batches = samples // batch_size
    ax.plot(x_, y_, 'k', alpha=i/((600 + 1) // 40))
    ax.plot(x, y, 'ko')
plt.show()