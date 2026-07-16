import time
from itertools import combinations
from pathlib import Path

import torch
from e3nn import o3
from torch import nn
from tqdm import tqdm

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# -------------------------------------- settings -------------------------------------
# hyperparameters
EPOCHS = 300
LR = 5e-3

# define loss
cost_fun = nn.MSELoss()

# model settings
IRREPS_SH = o3.Irreps.spherical_harmonics(lmax=2)
IRREPS_HIDDEN = o3.Irreps("8x0e + 8x1o + 4x2e")
IRREPS_OUT = o3.Irreps("1x1o")


# --------------------------------------- helper --------------------------------------
class EquivariantGNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.tp = o3.FullyConnectedTensorProduct(
            o3.Irreps("1x0e"), IRREPS_SH, IRREPS_HIDDEN
        )
        self.lin = o3.Linear(IRREPS_HIDDEN, IRREPS_OUT)

    def forward(self, pos, x, edge_src, edge_dst):
        edge_vec = pos[edge_dst] - pos[edge_src]
        edge_sh = o3.spherical_harmonics(IRREPS_SH, edge_vec, normalize=True)
        msg = self.tp(x[edge_src], edge_sh)
        agg = torch.zeros(len(pos), msg.shape[-1], device=pos.device)
        agg.scatter_add_(0, edge_dst[:, None].expand_as(msg), msg)
        return self.lin(agg)


# ------------------------------------ prepare data -----------------------------------
# regular tetrahedron inscribed in the unit sphere
pos = (
    torch.tensor(
        [
            [1, 1, 1],
            [1, -1, -1],
            [-1, 1, -1],
            [-1, -1, 1],
        ],
        dtype=torch.float32,
    ).to(device)
    / 3**0.5
)

x_in = torch.randn(4, 1).to(device)  # scalar "charge" per vertex

# all pairs in both directions -> 12 directed edges
pairs = list(combinations(range(4), 2))
edge_src = torch.tensor([i for i, j in pairs] + [j for i, j in pairs]).to(device)
edge_dst = torch.tensor([j for i, j in pairs] + [i for i, j in pairs]).to(device)

# equivariant target: weighted sum of edge unit vectors per destination node
# t_i = sum_{j->i} x_j * (pos_i - pos_j) / |pos_i - pos_j|
with torch.no_grad():
    edge_vec = pos[edge_dst] - pos[edge_src]
    edge_unit = edge_vec / edge_vec.norm(dim=1, keepdim=True)
    y = torch.zeros(4, 3, device=device)
    y.scatter_add_(0, edge_dst[:, None].expand(-1, 3), x_in[edge_src] * edge_unit)

# --------------------------- instantiate model & optimizer ---------------------------
model = EquivariantGNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), LR)

# -------------------------------------- training -------------------------------------
train_cost = [0] * EPOCHS
tic = time.time()
print_every = 50
pbar = tqdm(range(EPOCHS))
model.train()
for epoch in pbar:
    optimizer.zero_grad()
    y_pred = model(pos, x_in, edge_src, edge_dst)
    cost = cost_fun(y_pred, y)
    cost.backward()
    optimizer.step()
    train_cost[epoch] = cost.item()
    if epoch % print_every == 0:
        pbar.set_postfix({"train": f"{train_cost[epoch]:.2e}"})
toc = time.time()
print(f"elapsed time {toc - tic:.2f} s")

# ----------------------------------- postprocessing ----------------------------------
model.eval()

# equivariance check: f(R * pos) should equal R * f(pos) for any rotation R
R = o3.rand_matrix().to(device)
pos_rot = (pos @ R.T).detach()
with torch.no_grad():
    pred = model(pos, x_in, edge_src, edge_dst).cpu()
    pred_rot_direct = model(pos_rot, x_in, edge_src, edge_dst).cpu()
pred_then_rot = pred @ R.cpu().T
err = (pred_rot_direct - pred_then_rot).norm() / pred.norm()
print(f"equivariance error: {err:.2e}")

# write vtk graph for paraview
pos_np = pos.detach().cpu().numpy()
x_np = x_in.cpu().numpy().squeeze()
pred_np = pred.numpy()
y_np = y.cpu().numpy()
src_np = edge_src.cpu().numpy()
dst_np = edge_dst.cpu().numpy()
num_edges = len(src_np)

vtk_lines = [
    "# vtk DataFile Version 3.0",
    "e3nn equivariant gnn tetrahedron",
    "ASCII",
    "DATASET POLYDATA",
    "POINTS 4 float",
    *[f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}" for p in pos_np],
    f"LINES {num_edges} {3 * num_edges}",
    *[f"2 {s} {d}" for s, d in zip(src_np, dst_np)],
    "POINT_DATA 4",
    "SCALARS charge float 1",
    "LOOKUP_TABLE default",
    *[f"{v:.6f}" for v in x_np],
    "VECTORS prediction float",
    *[f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}" for v in pred_np],
    "VECTORS target float",
    *[f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}" for v in y_np],
]

out_path = RESULTS_DIR / "e3nn_graph.vtk"
out_path.write_text("\n".join(vtk_lines) + "\n")
print(f"written to {out_path}")
