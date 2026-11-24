import numpy as np
import torch

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
device = torch.device("cpu")

rng = np.random.default_rng(0)

# -------------------------------------- settings -------------------------------------
STEPS = 10
LR = 1e-2
SHAPE = (64, 32)


# --------------------------------------- helper --------------------------------------
def newton_schulz(G, steps=5):
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G / np.linalg.norm(G)
    tall = X.shape[0] > X.shape[1]
    if tall:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X
    return X.T if tall else X


class Muon:
    def __init__(self, lr=1e-3, beta=0.95):
        self.lr = lr
        self.beta = beta
        self.m = 0.0

    def step(self, W, G):
        self.m = self.beta * self.m + (1 - self.beta) * G
        update = newton_schulz((1 - self.beta) * G + self.beta * self.m)
        scale = np.sqrt(max(1, W.shape[0] / W.shape[1]))
        return W - self.lr * scale * update


# ------------------------------- compare with pytorch --------------------------------
W = rng.standard_normal(SHAPE).astype(np.float32)
W_torch = torch.tensor(W, requires_grad=True, device=device)
optimizer = Muon(lr=LR)
optimizer_torch = torch.optim.Muon([W_torch], lr=LR, weight_decay=0)

for _ in range(STEPS):
    G = rng.standard_normal(SHAPE).astype(np.float32)
    W = optimizer.step(W, G)
    W_torch.grad = torch.tensor(G, device=device)
    optimizer_torch.step()

# pytorch orthogonalizes in bfloat16, so updates agree only to about two digits
step_size = LR * np.sqrt(SHAPE[0] / SHAPE[1]) * STEPS
diff = np.abs(W - W_torch.detach().cpu().numpy()).max()
print(f"max difference {diff:.2e} relative to accumulated step size {step_size:.2e}")

s = np.linalg.svd(G, compute_uv=False)
s_orth = np.linalg.svd(newton_schulz(G), compute_uv=False)
print(f"singular values gradient {s.min():.2f} to {s.max():.2f}")
print(f"singular values orthogonalized {s_orth.min():.2f} to {s_orth.max():.2f}")
