import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# -------------------------- objective function --------------------------
a, b = 1, 100
f = lambda x : (a - x[0])**2 + b * (x[1] - x[0]**2)**2
xrange, yrange = [-2, 4], [-1, 5]
guess = [3., 3.]

# ------------------------------ optimizer -------------------------------
epochs = 2000

def optimize(x, optimizer, use_closure=False):
    trajectory = [x.data.clone().detach()]
    for epoch in range(epochs):
        if use_closure: # for l-bfgs
            def closure():
                optimizer.zero_grad()
                loss = f(x)
                loss.backward()
                return loss
            optimizer.step(closure)
        else: # for first-order optimizers
            optimizer.zero_grad()
            loss = f(x)
            loss.backward()
            optimizer.step()
        trajectory.append(x.data.clone().detach())
    return torch.stack(trajectory)

# ---------------------- optimization trajectories -----------------------

# steepest descent
x = torch.nn.Parameter(torch.Tensor(guess))
optimizer = torch.optim.SGD([x], lr=1e-4)
steepest = optimize(x, optimizer)

# gradient descent with momentum
x = torch.nn.Parameter(torch.Tensor(guess))
optimizer = torch.optim.SGD([x], lr=1e-4, momentum=0.8)
momentum = optimize(x, optimizer)

# adagrad
x = torch.nn.Parameter(torch.Tensor(guess))
optimizer = torch.optim.Adagrad([x], lr=1)
adagrad = optimize(x, optimizer)

# rmsprop
x = torch.nn.Parameter(torch.Tensor(guess))
optimizer = torch.optim.RMSprop([x], lr=5e-2)
rmsprop = optimize(x, optimizer)

# adam
x = torch.nn.Parameter(torch.Tensor(guess))
optimizer = torch.optim.Adam([x], lr=0.8)
adam = optimize(x, optimizer)

# l-bfgs
x = torch.nn.Parameter(torch.Tensor(guess))
optimizer = torch.optim.LBFGS([x], lr=0.1, max_iter=20)
lbfgs = optimize(x, optimizer, use_closure=True)

# ---------------------------- postprocessing ----------------------------

x = torch.linspace(xrange[0], xrange[1], 100)
y = torch.linspace(yrange[0], yrange[1], 100)
x, y = torch.meshgrid(x, y, indexing='ij')
z = f(torch.cat((x.unsqueeze(0), y.unsqueeze(0)), 0))

fig, ax = plt.subplots(figsize=(4, 4), dpi=150)
levels = torch.logspace(torch.log10(z.min()), torch.log10(z.max()), 24)
ax.contourf(x, y, z, norm=LogNorm(), levels=levels, cmap='cividis')
ax.plot(rmsprop[:,0], rmsprop[:,1], 'w', alpha=0.6, linewidth=3)

ax.plot(lbfgs[:,0], lbfgs[:,1], 'c', linewidth=3)
ax.plot(lbfgs[:,0], lbfgs[:,1], 'co', linewidth=3)
ax.plot(adam[:,0], adam[:,1], 'k:', linewidth=3)
ax.plot(adagrad[:,0], adagrad[:,1], 'purple', linestyle='--', linewidth=3)
ax.plot(momentum[:,0], momentum[:,1], 'r--', linewidth=3)
ax.plot(steepest[:,0], steepest[:,1], 'b:', linewidth=3)
ax.set_xlim(x.min(), x.max())
ax.set_ylim(y.min(), y.max())
ax.set_xlim(x.min(), x.max())
ax.set_ylim(y.min(), y.max())
ax.axis('off')
ax.set_rasterized(True)
fig.tight_layout(pad=0)
plt.savefig(f'../../results/rosenbrock_zoomed.pdf')
plt.show()

# zoomed in
x = torch.linspace(1-0.2, 1+0.2, 100)
y = torch.linspace(1-0.2, 1+0.2, 100)
x, y = torch.meshgrid(x, y, indexing='ij')
z = f(torch.cat((x.unsqueeze(0), y.unsqueeze(0)), 0))

fig, ax = plt.subplots(figsize=(4, 4), dpi=150)
levels = torch.logspace(torch.log10(z.min()), torch.log10(z.max()), 24)
ax.contourf(x, y, z, norm=LogNorm(), levels=levels, cmap='cividis')
ax.plot(rmsprop[:,0], rmsprop[:,1], 'w', alpha=0.6, linewidth=3)

ax.plot(lbfgs[:,0], lbfgs[:,1], 'c', linewidth=3)
ax.plot(lbfgs[:,0], lbfgs[:,1], 'co', linewidth=3)
ax.plot(adam[:,0], adam[:,1], 'k:', linewidth=3)
ax.plot(adagrad[:,0], adagrad[:,1], 'purple', linestyle='--', linewidth=3)
ax.plot(momentum[:,0], momentum[:,1], 'r--', linewidth=3)
ax.plot(steepest[:,0], steepest[:,1], 'b:', linewidth=3)

# last point
ax.plot(lbfgs[:,0], lbfgs[:,1], 'c', linewidth=3)
ax.plot(lbfgs[:,0], lbfgs[:,1], 'co', linewidth=3)
ax.plot(adam[-1,0], adam[-1,1], 'ko', linewidth=3)
ax.plot(momentum[-1,0], momentum[-1,1], 'ro', linewidth=3)

ax.set_xlim(x.min(), x.max())
ax.set_ylim(y.min(), y.max())
ax.axis('off')
ax.set_rasterized(True)
fig.tight_layout(pad=0)
plt.savefig(f'../../results/rosenbrock_zoomed.pdf')
plt.show()