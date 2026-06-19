import numpy as np
import time
import scipy.sparse.linalg
# import pypardiso # faster than scipy for real systems
import warnings
import matplotlib.pyplot as plt


def getTensorProduct2D(functionsXi, functionsEta, p):
    N = np.zeros((p + 1) ** 2)
    index = 0
    # vertex modes
    for i in range(2):
        for j in range(2):
            N[index] = functionsXi[j] * functionsEta[i]
            index += 1
    # edge modes
    for i in range(2):
        for j in range(2, p + 1):
            N[index] = functionsXi[j] * functionsEta[i]
            index += 1
    for i in range(2):
        for j in range(2, p + 1):
            N[index] = functionsXi[i] * functionsEta[j]
            index += 1
    # face modes
    for i in range(2, p + 1):
        for j in range(2, p + 1):
            N[index] = functionsXi[i] * functionsEta[j]
            index += 1
    return N


# adapted from https://gitlab.com/hpfem/publications/2021_nd-mlhp
def integratedLegendre(r, p, diff):
    if integratedLegendre.pmax < p:
        factorsJ = lambda j: (1 / j, 2 * j - 1, j - 1, 1 / np.sqrt(4 * j - 2))
        integratedLegendre.factors = [[]] + [factorsJ(j) for j in range(1, p + 1)]
        integratedLegendre.pmax = p

    I = np.zeros((p + 1,) + np.shape(r))
    L = [1, r, 0]

    if diff == 0:
        I[0] = 0.5 * (1.0 - r)
        I[1] = 0.5 * (1.0 + r)

        for j in range(2, p + 1):
            f1, f2, f3, f4 = integratedLegendre.factors[j]
            L[2] = f1 * (f2 * r * L[1] - f3 * L[0])
            I[j] = f4 * (L[2] - L[0])
            L[0:2] = L[1:3]

    if diff == 1:
        I[0] = -0.5
        I[1] = 0.5
        dL = [0, 1, 0]

        for j in range(2, p + 1):
            f1, f2, f3, f4 = integratedLegendre.factors[j]
            L[2] = f1 * (f2 * r * L[1] - f3 * L[0])
            dL[2] = f1 * (f2 * (L[1] + r * dL[1]) - f3 * dL[0])
            I[j] = f4 * (dL[2] - dL[0])
            L[0:2] = L[1:3]
            dL[0:2] = dL[1:3]
    return I.T


integratedLegendre.pmax = 0


def shapeFunctions2D(xi, eta, p):
    return getTensorProduct2D(integratedLegendre(xi, p, 0), integratedLegendre(eta, p, 0), p)


def derivedShapeFunctions2D(xi, eta, p, s):
    # derivatives are adjusted with the element length s (Jacobian)
    return [getTensorProduct2D(integratedLegendre(xi, p, 1), integratedLegendre(eta, p, 0), p) * 2. / s,
            getTensorProduct2D(integratedLegendre(xi, p, 0), integratedLegendre(eta, p, 1), p) * 2. / s]


def integrationMapping(a, b, xi):
    return 0.5 * (b - a) * xi + 0.5 * (a + b)


def localStiffnessMatrices2D(p, s, gaussPoints, voxelsPerElement):
    K = np.zeros((voxelsPerElement, voxelsPerElement, (p + 1) ** 2, (p + 1) ** 2))
    gp, gw = np.polynomial.legendre.leggauss(gaussPoints)

    for iVoxel in range(voxelsPerElement):
        iInterval = [-1 + 2 * iVoxel / voxelsPerElement, -1 + 2 * (iVoxel + 1) / voxelsPerElement]
        for jVoxel in range(voxelsPerElement):
            jInterval = [-1 + 2 * jVoxel / voxelsPerElement, -1 + 2 * (jVoxel + 1) / voxelsPerElement]
            for i in range(gaussPoints):
                for j in range(gaussPoints):
                    xi = integrationMapping(iInterval[0], iInterval[1], gp[i])
                    eta = integrationMapping(jInterval[0], jInterval[1], gp[j])
                    dN = np.vstack(derivedShapeFunctions2D(xi, eta, p, s))
                    determinant = 0.25 * (iInterval[1] - iInterval[0]) * (jInterval[1] - jInterval[0]) * (s / 2.) ** 2
                    K[iVoxel, jVoxel] += np.transpose(dN) @ dN * gw[i] * gw[j] * determinant
    return K


def localMassMatrices2D(p, s, gaussPoints, voxelsPerElement):
    M = np.zeros((voxelsPerElement, voxelsPerElement, (p + 1) ** 2, (p + 1) ** 2))
    gp, gw = np.polynomial.legendre.leggauss(gaussPoints)

    for iVoxel in range(voxelsPerElement):
        iInterval = [-1 + 2 * iVoxel / voxelsPerElement, -1 + 2 * (iVoxel + 1) / voxelsPerElement]
        for jVoxel in range(voxelsPerElement):
            jInterval = [-1 + 2 * jVoxel / voxelsPerElement, -1 + 2 * (jVoxel + 1) / voxelsPerElement]
            for i in range(gaussPoints):
                for j in range(gaussPoints):
                    xi = integrationMapping(iInterval[0], iInterval[1], gp[i])
                    eta = integrationMapping(jInterval[0], jInterval[1], gp[j])
                    N = np.expand_dims(shapeFunctions2D(xi, eta, p), 0)
                    determinant = 0.25 * (iInterval[1] - iInterval[0]) * (jInterval[1] - jInterval[0]) * (s / 2.) ** 2
                    M[iVoxel, jVoxel] += np.transpose(N) @ N * gw[i] * gw[j] * determinant
    return M


def eft2D(i, j, Nx, Ny, p):
    eft = np.zeros((p + 1) ** 2, dtype=np.int32)
    # vertex modes
    eft[0] = p * i + (Nx * p + 1) * p * j
    eft[1] = p * (i + 1) + (Nx * p + 1) * p * j
    eft[2] = p * i + (Nx * p + 1) * p * (j + 1)
    eft[3] = p * (i + 1) + (Nx * p + 1) * p * (j + 1)
    # edge modes
    for i in range(p - 1):
        eft[4 + i] = eft[0] + 1 + i  # bot
        eft[4 + (p - 1) + i] = eft[2] + 1 + i  # top
        eft[4 + (p - 1) * 2 + i] = eft[0] + (Nx * p + 1) * (i + 1)  # left
        eft[4 + (p - 1) * 3 + i] = eft[1] + (Nx * p + 1) * (i + 1)  # right
    # face modes
    faceIndex = 4 + (p - 1) * 3 + (p - 1)
    for i in range(p - 1):
        for j in range(p - 1):
            eft[faceIndex] = eft[4 + (p - 1) * 2 + i] + 1 + j
            faceIndex += 1
    return eft


def precomputeEft2D(Nx, Ny, p):
    efts = []
    for i in range(Nx):
        eftsTemp = []
        for j in range(Ny):
            eftsTemp.append(eft2D(i, j, Nx, Ny, p))
        efts.append(eftsTemp)
    return np.array([[arr for arr in sublist] for sublist in efts])


def precomputeSparsityPattern(efts, Nx, Ny, p):
    rows = np.zeros((Nx, Ny, (p + 1) ** 2, (p + 1) ** 2))
    cols = np.zeros((Nx, Ny, (p + 1) ** 2, (p + 1) ** 2))
    for i in range(Nx):
        for j in range(Ny):
            rows[i, j], cols[i, j] = np.meshgrid(efts[i, j], efts[i, j], indexing='ij')
    rows = rows.flatten()
    cols = cols.flatten()
    return rows, cols


def computeContributionsToGlobalStiffnessMatrix(bulkModulus, Nx, Ny, numberOfVoxels, p, Kes):
    values = np.zeros((Nx, Ny, (p + 1) ** 2, (p + 1) ** 2))
    for i in range(Nx):
        for j in range(Ny):
            values[i, j] = np.sum(np.sum(bulkModulus[i * numberOfVoxels:(i + 1) * numberOfVoxels,
                                         j * numberOfVoxels:(j + 1) * numberOfVoxels] * Kes, 0), 0)
    values = values.flatten()
    return values


def computeContributionsToGlobalMassMatrix(density, Nx, Ny, numberOfVoxels, p, Mes):
    values = np.zeros((Nx, Ny, (p + 1) ** 2, (p + 1) ** 2))
    for i in range(Nx):
        for j in range(Ny):
            values[i, j] = np.sum(np.sum(
                density[i * numberOfVoxels:(i + 1) * numberOfVoxels, j * numberOfVoxels:(j + 1) * numberOfVoxels] * Mes,
                0), 0)
    values = values.flatten()
    return values


def globalStiffnessMatrix2D(bulkModulus, Nx, Ny, numberOfVoxels, p, Kes, sparsityPattern, verbose=False):
    start = time.perf_counter()
    values = computeContributionsToGlobalStiffnessMatrix(bulkModulus, Nx, Ny, numberOfVoxels, p, Kes)

    N = (Nx * p + 1) * (Ny * p + 1)
    K = scipy.sparse.csr_matrix((values, (sparsityPattern[0], sparsityPattern[1])), shape=(N, N))  # .tolil()

    end = time.perf_counter()
    if verbose == True:
        print("Elapsed time during assembly for {:d} dofs: {:.2e} s".format(K.shape[0], end - start))
    return K


def globalMassMatrix2D(density, Nx, Ny, numberOfVoxels, p, Mes, sparsityPattern, verbose=False):
    start = time.perf_counter()
    values = computeContributionsToGlobalMassMatrix(density, Nx, Ny, numberOfVoxels, p, Mes)

    N = (Nx * p + 1) * (Ny * p + 1)
    M = scipy.sparse.csr_matrix((values, (sparsityPattern[0], sparsityPattern[1])), shape=(N, N))

    end = time.perf_counter()
    if verbose == True:
        print("Elapsed time during assembly for {:d} dofs: {:.2e} s".format(M.shape[0], end - start))
    return M


def applyHomogeneousDirichletBoundaryConditions(K, fixedDofs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", scipy.sparse.SparseEfficiencyWarning)
        diagonalEntries = np.ones(K.shape[0])
        diagonalEntries[fixedDofs] = 0
        applyFixedDofs = scipy.sparse.diags(diagonalEntries)
        K = applyFixedDofs @ K @ applyFixedDofs
        K[fixedDofs, fixedDofs] = 1
        K.eliminate_zeros()
    return K


def applyNeumannBoundaryConditions(loadIndex, load, Nx, Ny, p):
    F = np.zeros((Nx * p + 1) * (Ny * p + 1))
    for i in range(len(loadIndex)):
        F[loadIndex[i]] = load
    return F


def solve(K, F, pardiso=False, verbose=False):
    start = time.perf_counter()
    if pardiso == False:
        U = scipy.sparse.linalg.spsolve(K, F, use_umfpack=True)
    else:
        U = pypardiso.spsolve(K, F)
    end = time.perf_counter()
    if verbose == True:
        print("Elapsed time during solving: {:.2e} s".format(end - start))
    return U


def solveComplex(K, F, verbose=False):
    start = time.perf_counter()
    U = scipy.sparse.linalg.spsolve(K, F, use_umfpack=True)
    end = time.perf_counter()
    if verbose == True:
        print("Elapsed time during solving: {:.2e} s".format(end - start))
    return U


def getDisplacements(U, Nx, Ny, p, s, pointsPerElement, dtype=np.float32):  # getInterpolated field
    u = np.zeros((pointsPerElement * Nx, pointsPerElement * Ny), dtype=dtype)
    x = np.zeros((pointsPerElement * Nx, pointsPerElement * Ny))
    y = np.zeros((pointsPerElement * Nx, pointsPerElement * Ny))

    points, _ = np.polynomial.legendre.leggauss(pointsPerElement)
    N = []
    for i in range(pointsPerElement):
        Ntemp = []
        for j in range(pointsPerElement):
            Ntemp.append([shapeFunctions2D(points[i], points[j], p)])
        N.append(Ntemp)

    for i in range(Nx):
        for j in range(Ny):
            Ue = U[eft2D(i, j, Nx, Ny, p)]
            for ixi in range(pointsPerElement):
                for jeta in range(pointsPerElement):
                    u[i * pointsPerElement + ixi, j * pointsPerElement + jeta] = N[ixi][jeta] @ Ue

                    x[i * pointsPerElement + ixi, j * pointsPerElement + jeta] = (0.5 + i + points[ixi] / 2.) * s
                    y[i * pointsPerElement + ixi, j * pointsPerElement + jeta] = (0.5 + j + points[jeta] / 2.) * s
    return x, y, u


def getBoundaryIndices(location, index, Nx, Ny, p):
    leftEdgeX = [(Nx * p + 1) * i * 2 for i in range(Ny * p + 1)]
    leftEdgeY = [(Nx * p + 1) * i * 2 + 1 for i in range(Ny * p + 1)]
    bottomEdgeX = [2 * i for i in range(Nx * p + 1)]
    bottomEdgeY = [2 * i + 1 for i in range(Nx * p + 1)]
    rightEdgeX = [(Nx * p + 1) * (i + 1) * 2 - 2 for i in range(Ny * p + 1)]
    rightEdgeY = [(Nx * p + 1) * (i + 1) * 2 - 1 for i in range(Ny * p + 1)]
    topEdgeX = [(Nx * p + 1) * (Ny * p) * 2 + 2 * i for i in range(Nx * p + 1)]
    topEdgeY = [(Nx * p + 1) * (Ny * p) * 2 + 2 * i + 1 for i in range(Nx * p + 1)]
    leftBottomCornerX = [0]
    leftBottomCornerY = [1]
    rightBottomCornerX = [(Nx * p) * 2]
    rightBottomCornerY = [(Nx * p) * 2 + 1]
    leftTopCornerX = [(Nx * p + 1) * (Ny * p) * 2]
    leftTopCornerY = [(Nx * p + 1) * (Ny * p) * 2 + 1]
    rightTopCornerX = [(Nx * p + 1) * (Ny * p + 1) * 2 - 2]
    rightTopCornerY = [(Nx * p + 1) * (Ny * p + 1) * 2 - 1]

    if location == "leftEdge":
        if index == 0:
            return leftEdgeX
        elif index == 1:
            return leftEdgeY
    elif location == "bottomEdge":
        if index == 0:
            return bottomEdgeX
        elif index == 1:
            return bottomEdgeY
    elif location == "rightEdge":
        if index == 0:
            return rightEdgeX
        elif index == 1:
            return rightEdgeY
    elif location == "topEdge":
        if index == 0:
            return topEdgeX
        elif index == 1:
            return topEdgeY
    if location == "leftBottomCorner":
        if index == 0:
            return leftBottomCornerX
        elif index == 1:
            return leftBottomCornerY
    elif location == "rightBottomCorner":
        if index == 0:
            return rightBottomCornerX
        elif index == 1:
            return rightBottomCornerY
    elif location == "leftTopCorner":
        if index == 0:
            return leftTopCornerX
        elif index == 1:
            return leftTopCornerY
    elif location == "rightTopCorner":
        if index == 0:
            return rightTopCornerX
        elif index == 1:
            return rightTopCornerY


def getCoordinateFromDof(index, Nx, p, s, direction=True):
    x = (index % (2 * (Nx * p + 1))) // 2 * s / p
    y = (index // (2 * (Nx * p + 1))) * s / p
    if direction == True:
        return (x, y, index % 2)
    else:
        return (x, y)


def plotFieldOnVoxels(field, s, Nx, Ny, p, numberOfVoxels, colorMapLimits=None, fixedDofs=None, loadIndex=None):
    x = np.linspace(s / 2 / numberOfVoxels, Nx * s - s / 2 / numberOfVoxels, Nx * numberOfVoxels)
    y = np.linspace(s / 2 / numberOfVoxels, Ny * s - s / 2 / numberOfVoxels, Ny * numberOfVoxels)
    x, y = np.meshgrid(x, y, indexing='ij')

    if colorMapLimits == None:
        colorMapLimits = [np.min(field), np.max(field)]

    fig, ax = plt.subplots()
    cp = ax.pcolormesh(x, y, field, cmap=plt.cm.jet,
                       vmin=colorMapLimits[0], vmax=colorMapLimits[1])  # binary instead of jet
    fig.colorbar(cp, fraction=0.076, pad=0.04, format='%.1f', location='top')
    plt.gca().set_aspect('equal', adjustable='box')

    markers = ['>', '^']
    if fixedDofs != None:
        for dof in fixedDofs:
            coordinate = getCoordinateFromDof(dof, Nx, p, s)
            ax.plot(coordinate[0], coordinate[1], markers[coordinate[2]], color='k', markersize=12)
    if loadIndex != None:
        for dof in loadIndex:
            coordinate = getCoordinateFromDof(dof, Nx, p, s)
            ax.plot(coordinate[0], coordinate[1], markers[coordinate[2]], color='r', markersize=12)
    fig.tight_layout()
    plt.show()


def getNodalIndexFromCoordinate(x, y, Nx, p, s):
    elementX = x / s
    elementY = y / s

    if elementX % 1 != 0 or elementY % 1 != 0:
        print("ERROR: requested point does not coincide with corner node")

    index = (int(elementY) * p) * (Nx * p + 1) + int(elementX) * p
    return index
