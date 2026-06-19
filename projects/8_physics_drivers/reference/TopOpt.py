import FEM
import NeuralNetwork
import numpy as np
import torch
import time


# cost function helper functions

def getEftsInRectangle(elements, Nx, Ny, p):
    elementIdsx = np.linspace(elements[0], elements[1], elements[1] - elements[0] + 1, dtype=np.int32)
    elementIdsy = np.linspace(elements[2], elements[3], elements[3] - elements[2] + 1, dtype=np.int32)
    elementIdsx, elementIdsy = np.meshgrid(elementIdsx, elementIdsy, indexing='ij')
    elementIdsx = elementIdsx.flatten()
    elementIdsy = elementIdsy.flatten()

    costEfts = [FEM.eft2D(i, j, Nx, Ny, p) for i, j in zip(elementIdsx, elementIdsy)]

    return costEfts


def getIntegratedSquaredPressureFromEfts(U, efts, p, s, gps, gws):
    determinant = (s / 2.) ** 2
    squaredPressure = 0
    for i in range(len(efts)):
        Ue = U[efts[i]]
        for ixi, igw in zip(gps, gws):
            for jeta, jgw in zip(gps, gws):
                squaredPressure += abs(
                    FEM.shapeFunctions2D(ixi, jeta, p) @ Ue) ** 2 * igw * jgw * determinant
    return squaredPressure


def getCostInRectangle(U, supressElements, Nx, Ny, p, s, gaussPoints):
    efts = getEftsInRectangle(supressElements, Nx, Ny, p)
    gps, gws = np.polynomial.legendre.leggauss(gaussPoints)
    squaredPressure = getIntegratedSquaredPressureFromEfts(U, efts, p, s, gps, gws)
    area = (supressElements[1] - supressElements[0] + 1) * (supressElements[3] - supressElements[2] + 1) * s ** 2
    return squaredPressure / area


def getSoundPressureLevelInRectangle(U, referencePressure, supressElements, Nx, Ny, p, s,
                                     gaussPoints):
    efts = getEftsInRectangle(supressElements, Nx, Ny, p)
    gps, gws = np.polynomial.legendre.leggauss(gaussPoints)
    squaredPressure = getIntegratedSquaredPressureFromEfts(U, efts, p, s, gps, gws)
    area = (supressElements[1] - supressElements[0] + 1) * (supressElements[3] - supressElements[2] + 1) * s ** 2
    return 10 * np.log10(squaredPressure / referencePressure ** 2 / area)


# sensitivity computation helper functions

def getAdjointSourceVector(U, supressElements, Nx, Ny, p, s, gaussPoints):
    adjointSourceVector = np.zeros((Nx * p + 1) * (Ny * p + 1), dtype=np.complex128)
    efts = getEftsInRectangle(supressElements, Nx, Ny, p)
    gps, gws = np.polynomial.legendre.leggauss(gaussPoints)
    determinant = (s / 2.) ** 2
    for i in range(len(efts)):
        Ue = U[efts[i]]
        for ixi, igw in zip(gps, gws):
            for jeta, jgw in zip(gps, gws):
                adjointSourceVector[efts[i]] += FEM.shapeFunctions2D(ixi, jeta, p) * 2 * (
                            Ue.real - 1j * Ue.imag) * igw * jgw * determinant
    area = (supressElements[1] - supressElements[0] + 1) * (supressElements[3] - supressElements[2] + 1) * s ** 2
    adjointSourceVector /= area

    return -adjointSourceVector


def getIndicatorSensitivity(U, UAdjoint, Kes, Mes, normalizedAngularFrequency, damping, density1, density2,
                            bulkModulus1, bulkModulus2, efts, Nx, Ny, numberOfVoxels):
    systemSensitivity = (density1 / density2 - 1) * Kes - (
                1j * damping * normalizedAngularFrequency + normalizedAngularFrequency ** 2) * (
                                    bulkModulus1 / bulkModulus2 - 1) * Mes
    indicatorSensitivity = np.zeros((Nx * numberOfVoxels, Ny * numberOfVoxels))
    for i in range(Nx):
        for j in range(Ny):
            indicatorSensitivity[i * numberOfVoxels:(i + 1) * numberOfVoxels,
            j * numberOfVoxels:(j + 1) * numberOfVoxels] = (
                        systemSensitivity @ UAdjoint[efts[i, j]] @ U[efts[i, j]]).real
    return indicatorSensitivity


# filtering helper functions

def getFilteringKernel(filterRadius, dx, dy, device):
    kernelSizex = int(np.ceil(filterRadius / dx) - 1) * 2 + 1
    kernelSizey = int(np.ceil(filterRadius / dy) - 1) * 2 + 1

    x = torch.linspace(-(kernelSizex // 2) * dx, (kernelSizex // 2) * dx, kernelSizex)
    y = torch.linspace(-(kernelSizey // 2) * dy, (kernelSizey // 2) * dy, kernelSizey)
    x, y = torch.meshgrid(x, y, indexing='ij')

    r = torch.sqrt(x ** 2 + y ** 2)

    weight = filterRadius - r
    weight[r > filterRadius] = 0
    weight = weight / torch.sum(weight)

    return weight.unsqueeze(0).unsqueeze(0).to(dtype=torch.float32).to(device)


class densityFilter(torch.nn.Module):
    def __init__(self, filterRadius, dx, dy, device):
        super().__init__()
        kernel = getFilteringKernel(filterRadius, dx, dy, device)
        self.convolution = torch.nn.Conv2d(1, 1, kernel_size=(kernel.shape[2], kernel.shape[3]), stride=(1, 1),
                                           bias=False,
                                           padding=0)

        self.replicatePadding = lambda input: torch.nn.functional.pad(input, pad=(
        kernel.shape[2] // 2, kernel.shape[2] // 2, kernel.shape[3] // 2, kernel.shape[3] // 2), mode='replicate')
        self.zeroPadding = lambda input: torch.nn.functional.pad(input, pad=(0, 0, 0, 0), mode='constant', value=0)

        self.convolution.weight = torch.nn.Parameter(kernel, requires_grad=False)

    def forward(self, x):
        return self.convolution(self.replicatePadding(self.zeroPadding(x)))


class densityProjection(torch.nn.Module):
    def __init__(self, beta, eta):
        super().__init__()
        self.beta = beta
        self.eta = eta

    def forward(self, x):
        return (np.tanh(self.beta * self.eta) + torch.tanh(self.beta * (x - self.eta))) / \
            (np.tanh(self.beta * self.eta) + np.tanh(self.beta * (1 - self.eta)))


# optimizer helper functions

class topologyOptimizer():
    def __init__(self, density1, density2, bulkModulus1, bulkModulus2, damping, referencePressure,
                 Lx, Ly, frequency, loadLocation, amplitude,
                 supressArea, ceilingHeight,
                 Nx, Ny, numberOfVoxels, p):
        # material
        self.density1 = density1
        self.density2 = density2
        self.bulkModulus1 = bulkModulus1
        self.bulkModulus2 = bulkModulus2
        self.damping = damping
        self.referencePressure = referencePressure

        # forward problem
        self.Lx = Lx
        self.Ly = Ly
        self.normalizedAngularFrequency = 2 * np.pi * frequency / np.sqrt(bulkModulus1 / density1)
        self.loadLocation = loadLocation
        self.amplitude = amplitude

        # optimization problem
        self.supressArea = supressArea
        self.ceilingHeight = ceilingHeight

        # discretization
        # initialize system
        self.indicator = np.zeros((Nx * numberOfVoxels, Ny * numberOfVoxels))
        self.initializeSystem(Lx, Ly, Nx, Ny, p, numberOfVoxels)

        # initial cost
        self.cost0 = self.cost(self.forward(self.indicator)[0])

    def initializeSystem(self, Lx, Ly, Nx, Ny, p, numberOfVoxels):  # makes changing discretization possible
        if self.indicator.shape[0] != numberOfVoxels * Nx or self.indicator.shape[1] != numberOfVoxels * Ny:
            print("ERROR: total number of voxels has been changed!")

        self.Nx = Nx
        self.Ny = Ny
        self.p = p
        self.numberOfVoxels = numberOfVoxels

        self.gaussPoints = p + 1
        self.s = Lx / Nx
        if np.abs(self.s - Ly / Ny) > 1e-8:
            print("ERROR: elements are not square!")

        self.Kes = FEM.localStiffnessMatrices2D(p, self.s, self.gaussPoints, numberOfVoxels)
        self.Mes = FEM.localMassMatrices2D(p, self.s, self.gaussPoints, numberOfVoxels)
        loadIndex = [FEM.getNodalIndexFromCoordinate(self.loadLocation[0], self.loadLocation[1], Nx, p, self.s)]
        self.F = FEM.applyNeumannBoundaryConditions(loadIndex, self.amplitude, Nx, Ny, p)
        self.efts = FEM.precomputeEft2D(Nx, Ny, p)
        self.sparsityPattern = FEM.precomputeSparsityPattern(self.efts, Nx, Ny, p)

        self.supressElements = (int(self.supressArea[0] / self.s), int(self.supressArea[1] / self.s) - 1,
                                int(self.supressArea[2] / self.s), int(self.supressArea[3] / self.s) - 1)
        self.yVoxelLimit = int(self.ceilingHeight / self.s * self.numberOfVoxels)

    def sensitivity(self):
        U, S = self.forward(self.indicator)
        indicatorSensitivity = self.backward(U, S)
        return indicatorSensitivity[:, -self.yVoxelLimit:]

    def forward(self, indicator):
        # forward computation
        density = np.reshape(1 + indicator * (self.density1 / self.density2 - 1),
                             (self.Nx * self.numberOfVoxels, self.Ny * self.numberOfVoxels, 1, 1))
        bulkModulus = np.reshape(1 + indicator * (self.bulkModulus1 / self.bulkModulus2 - 1),
                                 (self.Nx * self.numberOfVoxels, self.Ny * self.numberOfVoxels, 1, 1))
        K = FEM.globalStiffnessMatrix2D(density, self.Nx, self.Ny, self.numberOfVoxels, self.p, self.Kes,
                                        self.sparsityPattern, verbose=False)
        M = FEM.globalMassMatrix2D(bulkModulus, self.Nx, self.Ny, self.numberOfVoxels, self.p, self.Mes,
                                   self.sparsityPattern, verbose=False)
        S = K - (1j * self.normalizedAngularFrequency * self.damping + self.normalizedAngularFrequency ** 2) * M
        U = FEM.solveComplex(S, self.F, verbose=False)
        return U, S

    def cost(self, U, normalization=1):
        return getCostInRectangle(U, self.supressElements, self.Nx, self.Ny,
                                  self.p, self.s, self.gaussPoints) / normalization

    def soundPressureLevel(self, U):
        return getSoundPressureLevelInRectangle(U, self.referencePressure, self.supressElements,
                                                self.Nx, self.Ny, self.p, self.s, self.gaussPoints)

    def backward(self, U, S):
        adjointSourceVector = getAdjointSourceVector(U, self.supressElements, self.Nx, self.Ny, self.p, self.s,
                                                     self.gaussPoints)
        UAdjoint = FEM.solveComplex(S, adjointSourceVector, verbose=False)
        indicatorSensitivity = getIndicatorSensitivity(U, UAdjoint, self.Kes, self.Mes, self.normalizedAngularFrequency,
                                                       self.damping, self.density1, self.density2, self.bulkModulus1,
                                                       self.bulkModulus2, self.efts, self.Nx, self.Ny,
                                                       self.numberOfVoxels)
        return indicatorSensitivity / self.cost0  # scale by initial cost

    def optimize(self, model, modelInput, optimizer, scheduler, epochs, clipGrad, betaGrowthRate, betaMaximum,
                 verbose=True):
        costHistory = np.zeros(epochs)
        indicatorHistory = np.zeros((epochs,) + self.indicator[:, -self.yVoxelLimit:].shape)
        start = time.perf_counter()
        for epoch in range(epochs):
            optimizer.zero_grad(set_to_none=True)

            # prediction
            indicatorPrediction = model(modelInput)
            self.indicator[:, -self.yVoxelLimit:] = indicatorPrediction.detach().numpy()[0, 0]
            indicatorHistory[epoch] = indicatorPrediction.detach().numpy()[0, 0]

            # forward computation
            U, S = self.forward(self.indicator)

            # cost evaluation
            costHistory[epoch] = self.cost(U, self.cost0)

            # backward computation
            indicatorSensitivity = self.backward(U, S)
            indicatorPrediction.grad = torch.zeros(indicatorPrediction.shape)
            indicatorPrediction.grad[0, 0] = torch.from_numpy(indicatorSensitivity[:, -self.yVoxelLimit:])
            indicatorPrediction.backward(
                indicatorPrediction.grad)  # explanation: https://medium.com/@monadsblog/pytorch-backward-function-e5e2b7e60140

            # update
            if clipGrad is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clipGrad)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            if isinstance(model, NeuralNetwork.ConstantAnsatz):
                model.coeff.data = model.coeff.data.clamp(0, 1)

            # status update
            if verbose == True:
                elapsedTime = time.perf_counter() - start
                string = "Epoch: {}/{}\t\tCost: {:.2e}\t\tElapsed time: {:.2f} s"
                print(string.format(epoch + 1, epochs, costHistory[epoch], elapsedTime))
                start = time.perf_counter()

            # beta continuation scheme
            model.projection.beta = np.minimum(betaGrowthRate * model.projection.beta, betaMaximum)

        return costHistory, indicatorHistory
