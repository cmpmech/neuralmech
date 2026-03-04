import torch
import matplotlib.pyplot as plt
from NN import MLP
from postprocessing import save_csv

torch.manual_seed(0)
device = torch.device('cpu')

neurons = 20

# ---------------------------- model settings ----------------------------
layers = [1, neurons, 1]
activations = [torch.nn.ReLU(inplace=True)]

# ---------------------------- training data -----------------------------
y = lambda x : torch.sin(1 * torch.pi * x)
samples = neurons + 1
x_train = torch.linspace(-1, 1, samples).unsqueeze(1)
y_train = y(x_train)

# -------------------- instantiate model & optimizer ---------------------
model = MLP(layers, activations)
model.to(device)

# --------------------- custom weight intialization ----------------------
with torch.no_grad():
    model.model[0].weight.data.fill_(1.0)
    for i in range(neurons):
        model.model[0].bias.data[i].fill_(1.0 - i * 2.0 / neurons)

    # to be learned
    model.model[2].weight.data.fill_(1.0)
    model.model[2].bias.data.fill_(0.0)

# fit weights and bias of last layer
def predict_hidden(x):
    x = model.model[0](x)
    x = model.model[1](x)
    return x

with torch.no_grad():
    X = torch.hstack([predict_hidden(x_train), torch.ones(samples, 1)])
    fit = torch.linalg.inv(X.T@X)@X.T@y_train
w = fit[0:neurons, 0]
b = fit[neurons, 0]

# set weights and bias
with torch.no_grad():
    model.model[2].weight.data[0,:] = w[:]
    model.model[2].bias.data[:] = b.item()

# ---------------------------- postprocessing ----------------------------
x_test = torch.linspace(-1, 1, 400).unsqueeze(1)
y_test = y(x_test)
y_pred = model(x_test).detach()

fig, ax = plt.subplots()
ax.plot(x_train, y_train, 'bo')
ax.plot(x_test, y_test, 'k')
ax.plot(x_test, y_pred, 'r')
plt.show()

# ------------------------- book postprocessing --------------------------
save_csv(f'../../results/universal_approx_width{neurons}.csv',
         x=x_test.squeeze(), y=y_test.squeeze(), ypred=y_pred.squeeze())