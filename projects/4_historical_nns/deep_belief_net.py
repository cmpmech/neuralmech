import torch
from torch import nn
from tqdm import tqdm

from helper import bars_and_stripes, contrastive_divergence

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
device = torch.device("cpu")  # faster on cpu, because matrices are small

# -------------------------------------- settings -------------------------------------
# hyperparameters
PRETRAIN_EPOCHS = 2000
PRETRAIN_LR = 0.05
EPOCHS = 200
LR = 1e-2
PRETRAIN = True  # False for the randomly initialized reference

# define loss
cost_fun = nn.BCEWithLogitsLoss()

# model settings
RESOLUTION = 4
LAYERS = [RESOLUTION**2, 32, 16]
SAMPLES = 512
NOISE = 0.1  # probability of flipping a pixel

# ------------------------------------ create data ------------------------------------
images = bars_and_stripes(RESOLUTION)
labels = (torch.arange(len(images)) >= 2**RESOLUTION).float()  # stripes 0, bars 1

ids = torch.randint(0, len(images), (SAMPLES,))
flip = torch.rand(SAMPLES, images.shape[1]) < NOISE
X = torch.where(flip, 1 - images[ids], images[ids]).to(device)
Y = labels[ids, None].to(device)
X_train, X_val = X[: SAMPLES // 2], X[SAMPLES // 2 :]
Y_train, Y_val = Y[: SAMPLES // 2], Y[SAMPLES // 2 :]

# --------------------------- instantiate model & optimizer ---------------------------
model = nn.Sequential(
    nn.Linear(LAYERS[0], LAYERS[1]),
    nn.Sigmoid(),
    nn.Linear(LAYERS[1], LAYERS[2]),
    nn.Sigmoid(),
    nn.Linear(LAYERS[2], 1),
)
model.to(device)

# ------------------------------------ pretraining ------------------------------------
if PRETRAIN:
    v = X_train
    for linear in [model[0], model[2]]:
        W = 0.01 * torch.randn(linear.in_features, linear.out_features, device=device)
        b = torch.zeros(linear.in_features, device=device)
        c = torch.zeros(linear.out_features, device=device)
        for epoch in tqdm(range(PRETRAIN_EPOCHS)):
            contrastive_divergence(v, W, b, c, PRETRAIN_LR)
        with torch.no_grad():
            linear.weight.copy_(W.T)
            linear.bias.copy_(c)
        v = torch.sigmoid(v @ W + c)  # hidden probabilities are the next layer's data

# -------------------------------------- training -------------------------------------
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()
    cost = cost_fun(model(X_train), Y_train)
    cost.backward()
    optimizer.step()

# ----------------------------------- postprocessing ----------------------------------
model.eval()
with torch.no_grad():
    y_pred = (model(X_val) > 0).float()
print(f"validation accuracy {(y_pred == Y_val).float().mean():.2f} (pretrain {PRETRAIN})")
