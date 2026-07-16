"""Tests for the minimal structured FEM package (solvers/FEM).

Patch tests pin the basis/assembly/BC chain to exact polynomial solutions at any
degree; the sub-voxel consistency test guards the C-ordering contract shared with
``solvers/optimization.py``; the J2 test pins the vectorized Python usermat to the
compiled C routine from ``solvers/material_subroutines``.
"""

import numpy as np
import pytest

from solvers import FEM


def solve_dirichlet(basis, K, f, bcs, method="splu"):
    fixed, values = FEM.combine_dirichlet(bcs)
    K_ff, f_free, free = FEM.apply_dirichlet(K, f, fixed, values)
    u = np.zeros(basis.ndof)
    u[fixed] = values
    u[free] = FEM.solve(K_ff, f_free, method=method)
    return u


@pytest.mark.parametrize("degree", [1, 2, 3])
@pytest.mark.parametrize("nelems, lengths", [([4, 3], [2.0, 1.5]), ([3, 2, 2], [1.0, 2.0, 1.0])])
def test_poisson_patch(degree, nelems, lengths):
    # a linear solution is reproduced exactly on all faces and in the interior
    basis = FEM.StructuredBasis(nelems, lengths, degree=degree)
    K = FEM.assemble(basis, FEM.stiffness_poisson(FEM.ReferenceElement(basis)))
    exact = lambda c: 2.0 + 3.0 * c[:, 0] - c[:, 1]
    bcs = [FEM.dirichlet(basis, ax, s, exact) for ax in range(basis.dim) for s in (0, 1)]
    u = solve_dirichlet(basis, K, np.zeros(basis.ndof), bcs)
    assert np.abs(u - exact(basis.nodes)).max() < 1e-10


@pytest.mark.parametrize("ansatz", ["lagrange", "legendre"])
@pytest.mark.parametrize("degree", [1, 2, 3])
def test_elasticity_patch(degree, ansatz):
    # uniaxial tension with rollers: constant sigma_xx = traction, at any degree
    E, nu, traction = 210e9, 0.3, 1e6
    basis = FEM.StructuredBasis([4, 3], [2.0, 1.0], degree=degree, nfields=2, ansatz=ansatz)
    ref = FEM.ReferenceElement(basis)
    K = FEM.assemble(basis, FEM.stiffness_elasticity(ref, E, nu, "plane_stress"))
    f = FEM.neumann_load(basis, 0, 1, [traction, 0.0])
    bcs = [FEM.dirichlet(basis, 0, 0, 0.0, ifield=0), FEM.dirichlet(basis, 1, 0, 0.0, ifield=1)]
    u = solve_dirichlet(basis, K, f, bcs)
    coords, disp, grads = basis.evaluate(u, 3, gradient=True)
    strain = np.stack([grads[:, 0, 0], grads[:, 1, 1], grads[:, 0, 1] + grads[:, 1, 0]], axis=1)
    stress = strain @ FEM.material_matrix(E, nu, "plane_stress").T
    assert np.abs(stress[:, 0] - traction).max() < 1e-6 * traction
    assert np.abs(stress[:, 1:]).max() < 1e-6 * traction
    assert np.abs(disp[:, 0] - traction / E * coords[:, 0]).max() < 1e-6 * traction / E


def test_legendre_matches_lagrange():
    # same polynomial space -> identical galerkin solution (heterogeneous coefficient)
    coeff = np.random.default_rng(0).uniform(0.5, 1.0, (6, 4))
    sols = []
    for ansatz in ["lagrange", "legendre"]:
        basis = FEM.StructuredBasis([3, 2], [1.0, 1.0], degree=3, ansatz=ansatz)
        K = FEM.assemble(basis, FEM.stiffness_poisson(FEM.ReferenceElement(basis, 2)), coeff)
        f = FEM.neumann_load(basis, 0, 1, [1.0])
        u = solve_dirichlet(basis, K, f, [FEM.dirichlet(basis, 0, 0, 0.0)])
        sols.append(basis.evaluate(u, 4)[1])
    assert np.abs(sols[1] - sols[0]).max() < 1e-10 * np.abs(sols[0]).max()


def test_unsupported_dofs():
    # the void corner element supports the lattice nodes with both indices < 2
    # exclusively; all its other nodes are shared with material elements
    basis = FEM.StructuredBasis([2, 2], [1.0, 1.0], degree=2)
    indicator = np.ones((2, 2))
    indicator[0, 0] = 1e-8
    dofs, values = FEM.unsupported_dofs(basis, indicator)
    expected = [np.ravel_multi_index(ij, basis.node_shape)
                for ij in [(0, 0), (0, 1), (1, 0), (1, 1)]]
    assert sorted(dofs) == expected
    assert values.tolist() == [0.0] * 4


def test_trunk_dofs_match_mlhp_counts():
    # remaining dof counts pinned against mlhp's makeHpTrunkSpace
    for nelems, degree, ndof in [([4, 3], 2, 51), ([4, 3], 3, 82), ([4, 3], 4, 125),
                                 ([3, 2, 2], 2, 111), ([3, 2, 2], 3, 186),
                                 ([3, 2, 2], 4, 313)]:
        basis = FEM.StructuredBasis(nelems, [1.0] * len(nelems), degree=degree,
                                    ansatz="legendre")
        assert basis.ndof - FEM.trunk_dofs(basis)[0].size == ndof


def test_trunk_elasticity_patch():
    # the trunk space contains all linears: the uniaxial patch stays exact
    E, nu, traction = 210e9, 0.3, 1e6
    basis = FEM.StructuredBasis([4, 3], [2.0, 1.0], degree=3, nfields=2, ansatz="legendre")
    K = FEM.assemble(basis, FEM.stiffness_elasticity(FEM.ReferenceElement(basis), E, nu))
    f = FEM.neumann_load(basis, 0, 1, [traction, 0.0])
    bcs = [FEM.dirichlet(basis, 0, 0, 0.0, ifield=0),
           FEM.dirichlet(basis, 1, 0, 0.0, ifield=1), FEM.trunk_dofs(basis)]
    u = solve_dirichlet(basis, K, f, bcs)
    coords, disp = basis.evaluate(u, 3)
    assert np.abs(disp[:, 0] - traction / E * coords[:, 0]).max() < 1e-6 * traction / E


def test_solve_cg_matches_splu():
    basis = FEM.StructuredBasis([5, 4], [1.0, 1.0], degree=2, nfields=2, ansatz="legendre")
    K = FEM.assemble(basis, FEM.stiffness_elasticity(FEM.ReferenceElement(basis), 1.0, 0.3))
    f = FEM.neumann_load(basis, 0, 1, [1.0, 0.5])
    fixed, values = FEM.dirichlet(basis, 0, 0, 0.0)
    K_ff, f_free, free = FEM.apply_dirichlet(K, f, fixed, values)
    u_cg = FEM.solve(K_ff, f_free, method="cg", rtol=1e-12, maxiter=5000)
    u_lu = FEM.solve(K_ff, f_free, method="splu")
    assert np.abs(u_cg - u_lu).max() < 1e-8 * np.abs(u_lu).max()


def test_stiffness_rigid_modes():
    # constant fields carry no energy: row sums vanish (per field for elasticity)
    basis = FEM.StructuredBasis([2, 2], [1.0, 1.0], degree=2, nfields=2)
    K_locals = FEM.stiffness_elasticity(FEM.ReferenceElement(basis), 1.0, 0.3)
    translation = np.tile(np.eye(2), (K_locals.shape[1] // 2, 1))
    assert np.abs(K_locals.sum(axis=0) @ translation).max() < 1e-12


def test_subvoxel_consistency():
    # sub-voxel partitions of the reference element sum to the unpartitioned matrix
    # (guards the C-ordering contract shared with optimization.grid_to_elements)
    basis = FEM.StructuredBasis([2, 2], [1.0, 1.0], degree=2)
    k1 = FEM.stiffness_poisson(FEM.ReferenceElement(basis, 1))
    k4 = FEM.stiffness_poisson(FEM.ReferenceElement(basis, 4))
    assert np.abs(k4.sum(axis=0) - k1[0]).max() < 1e-12 * np.abs(k1).max()


def test_mass_total():
    basis = FEM.StructuredBasis([3, 3], [2.0, 1.0], degree=2, nfields=2)
    M = FEM.assemble(basis, FEM.mass(FEM.ReferenceElement(basis), rho=3.0))
    assert M.sum() == pytest.approx(3.0 * 2.0 * 1.0 * 2)


def test_neumann_total_load():
    basis = FEM.StructuredBasis([4, 3], [2.0, 1.5], degree=3, nfields=2)
    f = FEM.neumann_load(basis, 0, 1, [1e6, 2e6])
    assert f[0::2].sum() == pytest.approx(1e6 * 1.5)
    assert f[1::2].sum() == pytest.approx(2e6 * 1.5)


def test_j2_usermat_matches_c_routine():
    # pin the vectorized return mapping to the compiled C reference, both branches
    cffi = pytest.importorskip("cffi")
    from solvers.material_subroutines.j2 import NHISTORY, build

    lib = build()
    ffi = cffi.FFI()
    signature = (
        "long long(*)(double*, double*, double*, double*, double*, double*, double*, "
        "double**, double**, double*, long long*, long long*, long long)"
    )
    material = ffi.cast(signature, lib.material_address)

    params = np.array([210e9, 0.3, 250e6, 210e9 / 50, 0.0])
    rng = np.random.default_rng(0)
    n = 500
    dstrain = rng.normal(0.0, 2e-3, (n, 3))
    history0 = np.zeros((n, NHISTORY))
    history0[:, :6] = rng.normal(0.0, 100e6, (n, 6))
    history0[:, 12] = rng.uniform(0.0, 0.01, n)

    stress, tangent, history1 = FEM.j2_usermat(dstrain, history0.copy(), params)
    assert 0.0 < (history1[:, 12] > history0[:, 12]).mean() < 1.0  # both branches hit

    params_ptr = ffi.cast("double*", ffi.from_buffer(params))
    for i in range(n):
        stress_c = np.zeros(3)
        tangent_c = np.zeros(9)
        history_c = history0[i].copy()
        dstrain_c = dstrain[i].copy()
        fields = ffi.new("double*[1]", [ffi.cast("double*", ffi.from_buffer(history_c))])
        data = ffi.new("double*[1]", [params_ptr])
        material(
            ffi.cast("double*", ffi.from_buffer(stress_c)),
            ffi.cast("double*", ffi.from_buffer(tangent_c)),
            ffi.NULL, ffi.NULL,
            ffi.cast("double*", ffi.from_buffer(dstrain_c)),
            ffi.NULL, ffi.NULL, fields, data, ffi.NULL, ffi.NULL, ffi.NULL, 0,
        )
        assert np.abs(stress[i] - stress_c).max() < 1e-9 * np.abs(stress_c).max()
        assert np.abs(tangent[i] - tangent_c.reshape(3, 3)).max() < 1e-9 * np.abs(tangent_c).max()


def test_nonlinear_matches_linear_below_yield():
    # far below yield the J2 tangent system is the plane-strain elastic one
    E, nu = 210e9, 0.3
    params = [E, nu, 250e6, E / 50, 0.0]
    basis = FEM.StructuredBasis([4, 3], [2.0, 1.0], degree=2, nfields=2)
    ref = FEM.ReferenceElement(basis, 2)
    K_lin = FEM.assemble(basis, FEM.stiffness_elasticity(ref, E, nu, "plane_strain"))
    history = np.zeros((basis.n_elems, ref.weights.size, FEM.NHISTORY))
    K_tan, f_int, _ = FEM.assemble_nonlinear(
        basis, ref, FEM.j2_usermat, params, np.zeros(basis.ndof), history
    )
    assert np.abs((K_tan - K_lin).data).max() < 1e-9 * np.abs(K_lin.data).max()
    assert np.abs(f_int).max() == 0.0


def test_mlhp_cross_check():
    # full tensor space of degree p spans the same space as Lagrange of degree p
    mlhp = pytest.importorskip("mlhp")

    grid = mlhp.makeRefinedGrid(mlhp.makeGrid([5, 4], [2.0, 1.0]))
    mbasis = mlhp.makeHpTensorSpace(grid, degree=2, nfields=1)
    left = mlhp.integrateDirichletDofs(mlhp.scalarField(2, 0.0), mbasis, [0])
    right = mlhp.integrateDirichletDofs(mlhp.scalarField(2, 1.0), mbasis, [1])
    dirichlet = mlhp.combineDirichletDofs([left, right])
    matrix = mlhp.allocateSparseMatrix(mbasis, dirichlet[0])
    vector = mlhp.allocateRhsVector(matrix)
    integrand = mlhp.poissonIntegrand(mlhp.scalarField(2, 1.0), mlhp.scalarField(2, 0.0))
    mlhp.integrateOnDomain(mbasis, integrand, [matrix, vector], dirichletDofs=dirichlet)
    internal = mlhp.cg(matrix, vector, rtol=1e-14, M=mlhp.diagonalPreconditioner(matrix))
    acc = mlhp.DataAccumulator()
    mlhp.basisOutput(mbasis, mlhp.gridCellMesh([2, 2]), acc,
                     [mlhp.solutionProcessor(2, mlhp.inflateDofs(internal, dirichlet), "u")])
    tri = acc.triangulation(mpl=True)
    reference = {
        (round(xx, 9), round(yy, 9)): vv
        for xx, yy, vv in zip(tri.x, tri.y, np.array(acc.data()[0]))
    }

    basis = FEM.StructuredBasis([5, 4], [2.0, 1.0], degree=2)
    K = FEM.assemble(basis, FEM.stiffness_poisson(FEM.ReferenceElement(basis)))
    bcs = [FEM.dirichlet(basis, 0, 0, 0.0), FEM.dirichlet(basis, 0, 1, 1.0)]
    u = solve_dirichlet(basis, K, np.zeros(basis.ndof), bcs)
    coords, values = basis.evaluate(u, 3)
    err = max(
        abs(values[i, 0] - reference[(round(c[0], 9), round(c[1], 9))])
        for i, c in enumerate(coords)
    )
    assert err < 1e-10
