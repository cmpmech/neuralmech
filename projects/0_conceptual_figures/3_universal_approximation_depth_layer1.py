import torch
import matplotlib.pyplot as plt
from postprocessing import save_csv
from NN import MLP

torch.manual_seed(0)
device = torch.device('cpu')

neurons = 3

# ---------------------------- model settings ----------------------------
layers = [1, neurons, 1]
activations = [torch.nn.ReLU(inplace=True)]

# -------------------- instantiate model & optimizer ---------------------
model = MLP(layers, activations)
model.to(device)

# --------------------- custom weight intialization ----------------------
with torch.no_grad():
    model.model[0].weight.data.fill_(1.0)
    for i in range(neurons):
        model.model[0].bias.data[i].fill_(1.0 - i * 2.0 / neurons)

    model.model[2].weight.data[0].fill_(neurons)
    model.model[2].bias.data.fill_(-1.)
    for i in range(1, neurons):
        if i % 2 == 0:
            model.model[2].weight.data[:,i].fill_(2.0 * neurons)
        else:
            model.model[2].weight.data[:, i].fill_(-2.0 * neurons)

# ---------------------------- postprocessing ----------------------------
x_test = torch.linspace(-1, 1, 400).unsqueeze(1)
y_pred = model(x_test).detach()

fig, ax = plt.subplots()
ax.plot(x_test, y_pred, 'r')
plt.show()

# --------------------------- export for book ----------------------------
save_csv(f'../../results/universal_approx_depth_1.csv', x=x_test.squeeze(), ypred=y_pred.squeeze())