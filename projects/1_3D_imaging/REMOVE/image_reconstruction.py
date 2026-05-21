"""
MoDL Toy Example — 2D Inpainting with Synthetic Shapes
=======================================================
Forward operator A: binary mask (elementwise)
Denoiser D_w:       small U-Net (2D CNN)
Inner solver:       Conjugate Gradient (CG)
Training:           supervised, end-to-end, synthetic data

Ground truth images are generated on-the-fly: random Gaussians + rectangles.
No dataset needed.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# ─────────────────────────────────────────────
# 1.  Synthetic image generator
# ─────────────────────────────────────────────


def make_image(
    H: int = 64, W: int = 64, n_blobs: int = 4, n_rects: int = 3
) -> torch.Tensor:
    """Generate one synthetic ground-truth image in [0,1]."""
    img = np.zeros((H, W), dtype=np.float32)

    # Gaussian blobs
    yy, xx = np.mgrid[0:H, 0:W]
    for _ in range(n_blobs):
        cy, cx = np.random.randint(10, H - 10), np.random.randint(10, W - 10)
        sigma = np.random.uniform(4, 12)
        amp = np.random.uniform(0.4, 1.0)
        img += amp * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma**2))

    # Rectangles
    for _ in range(n_rects):
        y0, x0 = np.random.randint(0, H - 10), np.random.randint(0, W - 10)
        y1, x1 = y0 + np.random.randint(5, 20), x0 + np.random.randint(5, 20)
        y1, x1 = min(y1, H), min(x1, W)
        amp = np.random.uniform(0.3, 0.8)
        img[y0:y1, x0:x1] += amp

    img = np.clip(img, 0, 1)
    return torch.from_numpy(img).unsqueeze(0)  # (1, H, W)


class SyntheticDataset(Dataset):
    def __init__(
        self, n_samples: int = 1000, mask_ratio: float = 0.5, H: int = 64, W: int = 64
    ):
        self.n = n_samples
        self.H, self.W = H, W
        self.mask_ratio = mask_ratio

    def __len__(self):
        return self.n

    def __getitem__(self, _):
        x_gt = make_image(self.H, self.W)  # (1, H, W)
        mask = (torch.rand(1, self.H, self.W) > self.mask_ratio).float()
        b = x_gt * mask  # observed pixels
        return b, mask, x_gt


# ─────────────────────────────────────────────
# 2.  Forward operator  A, Aᵀ
# ─────────────────────────────────────────────
# A(x)  = mask * x   (linear, self-adjoint: Aᵀ = A)
# AᵀA   = mask * x   (idempotent)


def A_op(x, mask):
    return mask * x


def At_op(y, mask):
    return mask * y  # Aᵀ = A for masking


# ─────────────────────────────────────────────
# 3.  Conjugate Gradient solver
#     Solves: (AᵀA + λI) x = rhs
# ─────────────────────────────────────────────


def cg_solve(mask, rhs, lam: float, n_iter: int = 10) -> torch.Tensor:
    """
    Solve (AᵀA + λI) x = rhs  via CG.
    For masking: AᵀA x = mask * x, so the operator is diag(mask + λ).
    CG is shown explicitly for generality.
    """

    def Lx(v):
        return At_op(A_op(v, mask), mask) + lam * v  # (mask + λ) * v

    x = torch.zeros_like(rhs)
    r = rhs - Lx(x)
    p = r.clone()
    rs_old = (r * r).sum()

    for _ in range(n_iter):
        Ap = Lx(p)
        alpha = rs_old / ((p * Ap).sum() + 1e-12)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = (r * r).sum()
        if rs_new.sqrt() < 1e-6:
            break
        p = r + (rs_new / (rs_old + 1e-12)) * p
        rs_old = rs_new

    return x


# ─────────────────────────────────────────────
# 4.  Denoiser D_w  — small U-Net
# ─────────────────────────────────────────────


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNetDenoiser(nn.Module):
    """Lightweight U-Net: 1-channel in/out, depth 3."""

    def __init__(self, base_ch: int = 16):
        super().__init__()
        c = base_ch
        self.enc1 = ConvBlock(1, c)
        self.enc2 = ConvBlock(c, c * 2)
        self.enc3 = ConvBlock(c * 2, c * 4)
        self.pool = nn.MaxPool2d(2)
        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, 2, stride=2)
        self.dec2 = ConvBlock(c * 4, c * 2)
        self.up1 = nn.ConvTranspose2d(c * 2, c, 2, stride=2)
        self.dec1 = ConvBlock(c * 2, c)
        self.out = nn.Conv2d(c, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        d2 = self.dec2(torch.cat([self.up2(e3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return torch.sigmoid(self.out(d1))


# ─────────────────────────────────────────────
# 5.  MoDL model
# ─────────────────────────────────────────────


class MoDL(nn.Module):
    """
    K unrolled iterations of:
        z_k   = D_w(x_k)
        x_k+1 = CG-solve( (AᵀA + λI) x = Aᵀb + λ z_k )

    Same D_w shared across all iterations.
    """

    def __init__(self, K: int = 6, lam: float = 0.5, cg_iter: int = 10):
        super().__init__()
        self.K = K
        self.lam = lam
        self.cg_iter = cg_iter
        self.D_w = UNetDenoiser(base_ch=16)

    def forward(self, b, mask):
        x = b.clone()  # initialise with zero-filled

        for _ in range(self.K):
            z = self.D_w(x)  # denoised estimate
            rhs = At_op(b, mask) + self.lam * z  # Aᵀb + λ z
            x = cg_solve(mask, rhs, self.lam, self.cg_iter)

        return x


# ─────────────────────────────────────────────
# 6.  Training loop
# ─────────────────────────────────────────────


def train(
    n_epochs: int = 20,
    batch_size: int = 8,
    lr: float = 1e-3,
    mask_ratio: float = 0.5,
    K: int = 6,
    device: str = "cpu",
):

    dataset = SyntheticDataset(n_samples=800, mask_ratio=mask_ratio)
    val_set = SyntheticDataset(n_samples=100, mask_ratio=mask_ratio)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=8)

    model = MoDL(K=K).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    train_losses, val_losses = [], []

    for epoch in range(1, n_epochs + 1):
        model.train()
        ep_loss = 0.0
        for b, mask, x_gt in loader:
            b, mask, x_gt = b.to(device), mask.to(device), x_gt.to(device)
            x_pred = model(b, mask)
            loss = loss_fn(x_pred, x_gt)
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += loss.item()

        ep_loss /= len(loader)
        train_losses.append(ep_loss)

        # validation
        model.eval()
        with torch.no_grad():
            vl = sum(
                loss_fn(model(b.to(device), mask.to(device)), x_gt.to(device)).item()
                for b, mask, x_gt in val_loader
            ) / len(val_loader)
        val_losses.append(vl)

        print(f"Epoch {epoch:3d}/{n_epochs}  train={ep_loss:.4f}  val={vl:.4f}")

    return model, train_losses, val_losses


# ─────────────────────────────────────────────
# 7.  Visualisation
# ─────────────────────────────────────────────


def visualise_iterations(model, device, mask_ratio=0.5, H=64, W=64):
    """Show how x_k evolves across MoDL iterations for one example."""
    model.eval()

    x_gt = make_image(H, W).unsqueeze(0).to(device)  # (1,1,H,W)
    mask = (torch.rand(1, 1, H, W) > mask_ratio).float().to(device)
    b = x_gt * mask

    snapshots = [b[0, 0].cpu().numpy()]  # zero-fill (iteration 0)

    x = b.clone()
    with torch.no_grad():
        for _ in range(model.K):
            z = model.D_w(x)
            rhs = At_op(b, mask) + model.lam * z
            x = cg_solve(mask, rhs, model.lam, model.cg_iter)
            snapshots.append(x[0, 0].cpu().numpy())

    n_cols = model.K + 2  # zero-fill + K iterations + GT
    fig, axes = plt.subplots(1, n_cols, figsize=(2.5 * n_cols, 3))

    titles = ["zero-fill"] + [f"k={k + 1}" for k in range(model.K)] + ["ground truth"]
    images = snapshots + [x_gt[0, 0].cpu().numpy()]

    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    plt.suptitle("MoDL inpainting — iterates x_k", fontsize=11)
    plt.tight_layout()
    # plt.savefig("/mnt/user-data/outputs/modl_iterations.png", dpi=150)
    plt.show()
    print("Saved: modl_iterations.png")


def plot_losses(train_losses, val_losses):
    plt.figure(figsize=(6, 3))
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.xlabel("epoch")
    plt.ylabel("MSE loss")
    plt.title("MoDL training curve")
    plt.legend()
    plt.tight_layout()
    # plt.savefig("/mnt/user-data/outputs/modl_loss.png", dpi=150)
    plt.show()
    print("Saved: modl_loss.png")


# ─────────────────────────────────────────────
# 8.  Main
# ─────────────────────────────────────────────

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model, tl, vl = train(
        n_epochs=30,
        batch_size=8,
        lr=1e-3,
        mask_ratio=0.5,  # 50 % pixels observed
        K=6,  # MoDL iterations
        device=device,
    )

    plot_losses(tl, vl)
    visualise_iterations(model, device, mask_ratio=0.5)
