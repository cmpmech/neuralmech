import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import mlhp
import numpy as np

from solvers.homogenization import effective_stiffness

# -------------------------------------- settings -------------------------------------
DIM = 2

# discretization
DEGREE = 2
NELEMENTS = [16] * DIM

# microstructure: soft circular phase-2 inclusion in a stiffer phase-1 matrix
LENGTH = 1.0
RADIUS = 0.25
E1, NU1 = 1.0, 0.3
E2, NU2 = 0.2, 0.3

# -------------------------------------- geometry -------------------------------------
lengths = [LENGTH] * DIM
center = [0.5 * LENGTH] * DIM

inclusion = mlhp.implicitSphere(center, RADIUS)
inside = f"((x - {center[0]})**2 + (y - {center[1]})**2) < {RADIUS**2}"
E = mlhp.scalarField(DIM, f"{E2} if {inside} else {E1}")
nu = mlhp.scalarField(DIM, f"{NU2} if {inside} else {NU1}")

# ---------------------------------------- mesh ---------------------------------------
grid = mlhp.makeRefinedGrid(NELEMENTS, lengths)
basis = mlhp.makeHpTrunkSpace(grid, degree=DEGREE, nfields=DIM)
print(basis)

kinematics = mlhp.smallStrainKinematics(DIM)
material = mlhp.planeStressMaterial(E, nu)
quadrature = mlhp.spaceTreeQuadrature(inclusion, depth=DEGREE + 1, epsilon=1.0)

# ---------------------------------- homogenization -----------------------------------
C = effective_stiffness(basis, quadrature, kinematics, material, lengths)
print("effective stiffness [11, 22, 12]:")
print(C)
