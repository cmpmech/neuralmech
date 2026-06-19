import TopOpt
import NeuralNetwork
import Visualization
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import torch
import copy

torch.manual_seed(2)
torch.use_deterministic_algorithms(True)
device = torch.device("cpu")

# setup
discretizationSettings = pd.read_csv("discretizationSettings.csv")
Lx, Ly = discretizationSettings.Lx[0], discretizationSettings.Ly[0]
Nx, Ny = discretizationSettings.Nx[0], discretizationSettings.Ny[0]
numberOfVoxels = discretizationSettings.numberOfVoxels[0]
p0 = 2
p1 = 4

# frequency = 21.33 
# frequency = 34.39 
# frequency = 57.22 
# frequency = 69.43
# frequency = 76.30 
frequency = 95.37
# frequency = 141.78 
# frequency = 166.56 
# frequency = 206.31

density1, density2 = discretizationSettings.density1[0], discretizationSettings.density2[0]
bulkModulus1, bulkModulus2 = discretizationSettings.bulkModulus1[0], discretizationSettings.bulkModulus2[0]
damping, referencePressure = discretizationSettings.damping[0], discretizationSettings.referencePressure[0]
amplitude = discretizationSettings.amplitude[0]

# scenario settings
testSettings = np.loadtxt("caseObjective.txt")
ceilingHeight = testSettings[1]
loadLocation = [testSettings[2], testSettings[3]]  # source location
supressArea = (
    testSettings[4], testSettings[5], testSettings[6], testSettings[7])  # (x0, x1, y0, y1) area to be supressed

epochs0 = 280
epochs1 = 20
lr0 = 5e-2
lr1 = 5e-2
clipGrad = None
betaMaximum0 = 75
betaMaximum1 = 150
topOptSettings = pd.read_csv("topologyOptimizationSettings.csv")
eta = topOptSettings.eta[0]
beta0 = topOptSettings.beta0[0]
betaGrowthRate = topOptSettings.betaGrowthRate[0]
relativeFilterRadius = topOptSettings.relativeFilterRadius[0]
filterRadius = relativeFilterRadius * Lx / Nx / numberOfVoxels
initialGuess = 0

problem = TopOpt.topologyOptimizer(density1, density2, bulkModulus1, bulkModulus2, damping, referencePressure,
                                   Lx, Ly, frequency, loadLocation, amplitude,
                                   supressArea, ceilingHeight,
                                   Nx, Ny, numberOfVoxels, p0)

model = NeuralNetwork.ConstantAnsatz(Lx, Ly, Nx, Ny, numberOfVoxels, filterRadius, beta0, eta, problem.yVoxelLimit,
                                     initialGuess, device)
modelInput = None
scheduler = None

# tuning of initial guess
problem.initializeSystem(Lx, Ly, Nx=Nx, Ny=Ny, p=p0, numberOfVoxels=numberOfVoxels)


def gridSearchForInitialGuess(epochs):
    initialGuesses = np.array([0, 0.5, 1])
    costs = np.zeros_like(initialGuesses)
    for i, initialGuess in enumerate(initialGuesses):
        optimizer = torch.optim.Adam(model.parameters(), lr0)
        model.coeff.data = model.coeff.data * 0 + initialGuess
        model.projection.beta = beta0

        costHistory = problem.optimize(model, modelInput, optimizer, scheduler,
                                       epochs, clipGrad, betaGrowthRate, betaMaximum0, verbose=False)[0]

        costs[i] = costHistory[-1]
    print(costs)
    return initialGuesses[np.nanargmin(costs)]


start = time.perf_counter()
initialGuessSearch = gridSearchForInitialGuess(50)
elapsedTimeSearch = time.perf_counter() - start
initialGuess0 = initialGuessSearch
print("identified initial guess: {:.2f}".format(initialGuessSearch))

# reset model
model.coeff.data = model.coeff.data * 0 + initialGuess0
model.projection.beta = beta0
optimizer = torch.optim.Adam(model.parameters(), lr0)

# optimization 0
start = time.perf_counter()
costHistory0 = problem.optimize(model, modelInput, optimizer, scheduler,
                                epochs0, clipGrad, betaGrowthRate, betaMaximum0)[0]
elapsedTime0 = time.perf_counter() - start

U = problem.forward(problem.indicator)[0]
soundPressureLevel0 = problem.soundPressureLevel(U)
Visualization.visualizeDensity(problem)

# intermediate evaluation
intermediateIndicator = copy.deepcopy(problem.indicator)
problem.indicator[problem.indicator < 0.5] = 0
problem.indicator[problem.indicator > 0.5] = 1
U = problem.forward(problem.indicator)[0]
costIntermediate = problem.cost(U, normalization=problem.cost0)
soundPressureLevelIntermediate = problem.soundPressureLevel(U)
problem.indicator = intermediateIndicator

# optimization 1
start = time.perf_counter()
for param_group in optimizer.param_groups:
    param_group['lr'] = lr1
problem.initializeSystem(Lx, Ly, Nx=Nx, Ny=Ny, p=p1, numberOfVoxels=numberOfVoxels)

correctionCounter = 0
maxCorrectionCounter = 10
costHistory1 = problem.optimize(model, modelInput, optimizer, scheduler,
                                1, clipGrad, betaGrowthRate, betaMaximum1)[0]
while costHistory1[-1] > 1.5 * costHistory0[-1] and correctionCounter < maxCorrectionCounter:
    costHistory1 = np.append(costHistory1, problem.optimize(model, modelInput, optimizer, scheduler,
                                                            10, clipGrad, betaGrowthRate, betaMaximum1)[0])
    correctionCounter += 1
print("Achieved Correction: {:.2f}".format(costHistory1[-1] / costHistory0[-1]))
elapsedTime1 = time.perf_counter() - start

costHistory = np.concatenate((costHistory0, costHistory1))

# visualization
problem.indicator[problem.indicator < 0.5] = 0
problem.indicator[problem.indicator > 0.5] = 1
Visualization.visualizeDensity(problem)
U = problem.forward(problem.indicator)[0]
costFinal0 = problem.cost(U, normalization=problem.cost0)
soundPressureLevelFinal0 = problem.soundPressureLevel(U)
problem.initializeSystem(Lx, Ly, Nx=Nx * numberOfVoxels, Ny=Ny * numberOfVoxels, p=2, numberOfVoxels=1)
U = problem.forward(problem.indicator)[0]
costFinal1 = problem.cost(U, normalization=problem.cost0)
soundPressureLevelFinal1 = problem.soundPressureLevel(U)

fig, ax = plt.subplots()
ax.grid()
ax.plot(costHistory, 'k')
ax.set_yscale('log')
plt.show()

print("sound pressure level after optimization 0: {:.2f} and after optimization 1: {:.2f}".format(
    soundPressureLevelIntermediate,
    soundPressureLevelFinal0))
print("cost after optimization 0: {:.2f} and after optimization 1: {:.2f}".format(costIntermediate, costFinal0))
print("elapsed time: {:.2f} s & {:.2f} s & {:.2f} s".format(elapsedTimeSearch, elapsedTime0, elapsedTime1))
print("total elapsed time: {:.2f} s".format(elapsedTimeSearch + elapsedTime0 + elapsedTime1))

print("final sound pressure level: {:.2f} dB".format(soundPressureLevelFinal1))
print("final cost: {:.2e}".format(costFinal1))
