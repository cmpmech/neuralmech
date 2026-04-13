import torch

from NN import MLP
from postprocessing import show_image

# neural network art/random neural fields

torch.manual_seed(1)
torch.backends.cudnn.deterministic = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
activation_id = None


# -------------------------------- helper --------------------------------
def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.uniform_(m.weight, a=-10, b=10)
        torch.nn.init.uniform_(m.bias, a=-10, b=10)


class SinActivation(torch.nn.Module):
    def __init__(self, w0=1.0):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)  # 0.1


# --------------------------- hyperparameters ----------------------------
# depth study
hidden_layers = 8  # 1, 2, 4, 8
neurons = 128

# width study
# hidden_layers = 2
# neurons = 337  # 30, 128, 221, 337

parameters = (
    neurons * 8 + neurons * neurons * (hidden_layers - 1) + neurons * hidden_layers
)
print(f"parameters: {parameters}")

samples = 400
if hidden_layers == 1:
    samples = 200
elif hidden_layers == 2:
    samples = 400
elif hidden_layers == 4:
    samples = 800
elif hidden_layers == 8:
    samples = 1400

# activation, activation_id = torch.nn.Sigmoid(), 0
activation, activation_id = torch.nn.Tanh(), 1
# activation, activation_id = torch.nn.Mish(), 2
# activation, activation_id = SinActivation(0.05), 3

# ---------------------------- preprocessing -----------------------------
layers = [2] + [neurons] * hidden_layers + [3]
activations = [activation] * hidden_layers

x = torch.linspace(-1, 1, samples)
y = torch.linspace(-1, 1, samples)
x, y = torch.meshgrid(x, y, indexing="ij")
mlp_input = torch.cat((x.flatten().unsqueeze(1), y.flatten().unsqueeze(1)), 1).to(
    device
)

# --------------------------- model prediction ---------------------------
model = MLP(layers, activations)
model.apply(init_weights)
model.to(device)

with torch.no_grad():
    z_pred = model(mlp_input).reshape(samples, samples, 3).cpu()

for i in range(3):
    channel = z_pred[:, :, i]
    z_pred[:, :, i] = (channel - channel.min()) / (channel.max() - channel.min() + 1e-8)

# ---------------------------- postprocessing ----------------------------
show_image(
    z_pred.numpy(),
    path=f"../../results/expressivity_{activation_id}_{hidden_layers}_{neurons}.png",
)
