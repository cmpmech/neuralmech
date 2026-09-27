import numpy as np

x = np.load("../../data/abc/geometry/voxel/000000002_abc.npz")
u = np.load("../../data/abc/elasticity/solution_stl/displacement/000000002_abc_1.npz")

bc = np.load("../../data/abc/elasticity/fixture/boundary_voxel/000000002_abc_1.npz")


print(x["indicator"].shape)
print(
    bc["dirichlet_mask"].shape
)  # TODO why does it have the same shape as x indicator?
# TODO remove indicator from this file

# print(bc["dirichlet_value"].shape)
# print(bc["neumann"].shape)
print(u["displacement"].shape)


print(bc.files)  # remove indicator -> does not make sense here?
print(u.files)
print(
    x.files
)  # why does this one not get origin and spacing? -> but origin and spacing probably different from the others


print(bc["origin"])
print(x["origin"])
print(u["origin"])

print(bc["Lx"])
print(x["Lx"])
print(u["Lx"])
