import torch
from escnn import gspaces
from escnn import nn as enn

# -------------------------------------- settings -------------------------------------
KERNEL_SIZE = 3
PADDING = KERNEL_SIZE // 2
ROTATIONS = 8  # -1 for continuous equivariance
CHANNELS = [1, 4, 3]
GRID = 8

gspace = gspaces.rot2dOnR2(N=ROTATIONS)  # for SO(2)
gpasce = gspaces.flipRot2dOnR2(N=ROTATIONS)  # for O(2)
# gspace = gspaces.rot3dOnR2() for SO(3) (only continuous)


scalar_repr = gspace.trivial_repr  # or gspace.irrep(0)
vector_repr = gspace.irrep(1)  # rotates once per input rotation: tensors can be
# constructed from these, e.g., stress: gspace.irrep(0) + gspace.irrep(2)
#  		  							    (hydrostatic + deviatoric)

reprs = [vector_repr, scalar_repr, vector_repr]
channel_types = [enn.FieldType(gspace, [r] * c) for r, c in zip(reprs, CHANNELS)]

# build a 2-layer equivariant network manually
modules = [
    enn.R2Conv(channel_types[0], channel_types[1], KERNEL_SIZE, padding=PADDING),
    enn.LeakyReLU(channel_types[1]),
    enn.R2Conv(channel_types[1], channel_types[2], KERNEL_SIZE, padding=PADDING),
]
model = enn.SequentialModule(*modules)

# ------------------------------------- prediction ------------------------------------
x = enn.GeometricTensor(
    torch.randn(1, channel_types[0].size, GRID, GRID), channel_types[0]
)
y = model(x)  # output size [1, 6, 8, 8] -> vector dim folded into channel dim

# exact for 2, 4, 6 when N = 8
g = gspace.fibergroup.element(1)  # 1 step: rotation by 2 * pi / N
y_rot = model(x.transform(g))  # rotate input, then predict
rot_y = model(x).transform(g)  # predict, then rotate output
error = torch.max(torch.abs((y_rot.tensor - rot_y.tensor)))  # .tensor to torch
# decreases with KERNEL_SIZE

print(f"error: {error:.2e}")
