import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

np.random.seed(3)

# -------------------------- objective function --------------------------
f = lambda x: (
    -20 * np.exp(-0.2 * np.sqrt(0.5 * (x[0] ** 2 + x[1] ** 2)))
    - np.exp(0.5 * (np.cos(2 * np.pi * x[0]) + np.cos(2 * np.pi * x[1])))
    # + np.exp(0)
    # + 20
)
xrange, yrange = [-4, 4], [-4, 4]
guess = [3.0, 3.0]

##########################################################################
N, G, sigma, elite = 40, 60, 0.5, 2  # pop, generations, mutation std, elites

# init population around the initial guess
population = np.array(guess) + np.random.normal(0, 1.0, size=(N, 2))
history = [population.copy()]

for g in range(G):
    fitness = f(population.T)
    idx = np.argsort(fitness)  # sort best -> worst
    population = population[idx]

    # keep elites, fill the rest by mutating random elites
    parents = population[:elite]
    children = parents[np.random.randint(0, elite, size=N - elite)] + np.random.normal(
        0, sigma, size=(N - elite, 2)
    )
    population = np.vstack([parents, children])

    sigma *= 0.95  # anneal mutation
    history.append(population.copy())

best = population[np.argmin(f(population.T))]
print(f"best: x = {best}, f = {f(best):.3e}")


# ---------------------------- postprocessing ----------------------------

x = np.linspace(xrange[0], xrange[1], 800)
y = np.linspace(yrange[0], yrange[1], 800)
x, y = np.meshgrid(x, y, indexing="ij")
z = f(np.concatenate((np.expand_dims(x, 0), np.expand_dims(y, 0)), 0))

fig, ax = plt.subplots(figsize=(4, 4), dpi=150)
ax.contourf(x, y, z, levels=36, cmap="cividis")
# ax.pcolormesh(x, y, z, cmap="cividis")
#
for k, P in enumerate(history):
    ax.plot(
        P[:, 0],
        P[:, 1],
        ".",
        ms=2,
        # color="r",
        # alpha=0.2 + 0.8 * k / len(history),
        color=plt.cm.Greys(0.2 + 0.8 * k / len(history)),
        alpha=0.6,
    )
ax.plot(*best, "wo", ms=1)

plt.show()


# # ------------------------------ optimizer -------------------------------
# epochs = 2000

# def optimize(x, optimizer, use_closure=False):
#     trajectory = [x.data.clone().detach()]
#     for epoch in range(epochs):
#         if use_closure: # for l-bfgs
#             def closure():
#                 optimizer.zero_grad()
#                 loss = f(x)
#                 loss.backward()
#                 return loss
#             optimizer.step(closure)
#         else: # for first-order optimizers
#             optimizer.zero_grad()
#             loss = f(x)
#             loss.backward()
#             optimizer.step()
#         trajectory.append(x.data.clone().detach())
#     return torch.stack(trajectory)

# # ---------------------- optimization trajectories -----------------------

# # steepest descent
# x = torch.nn.Parameter(torch.Tensor(guess))
# optimizer = torch.optim.SGD([x], lr=1e-4)
# steepest = optimize(x, optimizer)

# # gradient descent with momentum
# x = torch.nn.Parameter(torch.Tensor(guess))
# optimizer = torch.optim.SGD([x], lr=1e-4, momentum=0.8)
# momentum = optimize(x, optimizer)

# # adagrad
# x = torch.nn.Parameter(torch.Tensor(guess))
# optimizer = torch.optim.Adagrad([x], lr=1)
# adagrad = optimize(x, optimizer)

# # rmsprop
# x = torch.nn.Parameter(torch.Tensor(guess))
# optimizer = torch.optim.RMSprop([x], lr=5e-2)
# rmsprop = optimize(x, optimizer)

# # adam
# x = torch.nn.Parameter(torch.Tensor(guess))
# optimizer = torch.optim.Adam([x], lr=0.8)
# adam = optimize(x, optimizer)

# # l-bfgs
# x = torch.nn.Parameter(torch.Tensor(guess))
# optimizer = torch.optim.LBFGS([x], lr=0.1, max_iter=20)
# lbfgs = optimize(x, optimizer, use_closure=True)

# # ---------------------------- postprocessing ----------------------------

# x = torch.linspace(xrange[0], xrange[1], 100)
# y = torch.linspace(yrange[0], yrange[1], 100)
# x, y = torch.meshgrid(x, y, indexing='ij')
# z = f(torch.cat((x.unsqueeze(0), y.unsqueeze(0)), 0))

# fig, ax = plt.subplots(figsize=(4, 4), dpi=150)
# levels = torch.logspace(torch.log10(z.min()), torch.log10(z.max()), 24)
# ax.contourf(x, y, z, norm=LogNorm(), levels=levels, cmap='cividis')
# ax.plot(rmsprop[:,0], rmsprop[:,1], 'w', alpha=0.6, linewidth=3)

# ax.plot(lbfgs[:,0], lbfgs[:,1], 'c', linewidth=3)
# ax.plot(lbfgs[:,0], lbfgs[:,1], 'co', linewidth=3)
# ax.plot(adam[:,0], adam[:,1], 'k:', linewidth=3)
# ax.plot(adagrad[:,0], adagrad[:,1], 'purple', linestyle='--', linewidth=3)
# ax.plot(momentum[:,0], momentum[:,1], 'r--', linewidth=3)
# ax.plot(steepest[:,0], steepest[:,1], 'b:', linewidth=3)
# ax.set_xlim(x.min(), x.max())
# ax.set_ylim(y.min(), y.max())
# ax.set_xlim(x.min(), x.max())
# ax.set_ylim(y.min(), y.max())
# ax.axis('off')
# ax.set_rasterized(True)
# fig.tight_layout(pad=0)
# plt.savefig(f'../../results/rosenbrock_zoomed.pdf')
# plt.show()

# # zoomed in
# x = torch.linspace(1-0.2, 1+0.2, 100)
# y = torch.linspace(1-0.2, 1+0.2, 100)
# x, y = torch.meshgrid(x, y, indexing='ij')
# z = f(torch.cat((x.unsqueeze(0), y.unsqueeze(0)), 0))

# fig, ax = plt.subplots(figsize=(4, 4), dpi=150)
# levels = torch.logspace(torch.log10(z.min()), torch.log10(z.max()), 24)
# ax.contourf(x, y, z, norm=LogNorm(), levels=levels, cmap='cividis')
# ax.plot(rmsprop[:,0], rmsprop[:,1], 'w', alpha=0.6, linewidth=3)

# ax.plot(lbfgs[:,0], lbfgs[:,1], 'c', linewidth=3)
# ax.plot(lbfgs[:,0], lbfgs[:,1], 'co', linewidth=3)
# ax.plot(adam[:,0], adam[:,1], 'k:', linewidth=3)
# ax.plot(adagrad[:,0], adagrad[:,1], 'purple', linestyle='--', linewidth=3)
# ax.plot(momentum[:,0], momentum[:,1], 'r--', linewidth=3)
# ax.plot(steepest[:,0], steepest[:,1], 'b:', linewidth=3)

# # last point
# ax.plot(lbfgs[:,0], lbfgs[:,1], 'c', linewidth=3)
# ax.plot(lbfgs[:,0], lbfgs[:,1], 'co', linewidth=3)
# ax.plot(adam[-1,0], adam[-1,1], 'ko', linewidth=3)
# ax.plot(momentum[-1,0], momentum[-1,1], 'ro', linewidth=3)

# ax.set_xlim(x.min(), x.max())
# ax.set_ylim(y.min(), y.max())
# ax.axis('off')
# ax.set_rasterized(True)
# fig.tight_layout(pad=0)
# plt.savefig(f'../../results/rosenbrock_zoomed.pdf')
# plt.show()
