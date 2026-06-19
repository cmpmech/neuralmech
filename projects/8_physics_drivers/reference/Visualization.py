import matplotlib.pyplot as plt
import numpy as np
import FEM
import matplotlib
import matplotlib.font_manager
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patches as patches


def visualizeDensity(problem, filename=None, show=True):
    x = np.linspace(0, problem.Lx, problem.Nx * problem.numberOfVoxels)
    y = np.linspace(0, problem.Ly, problem.Ny * problem.numberOfVoxels)[-problem.yVoxelLimit:]
    x, y = np.meshgrid(x, y, indexing='ij')

    fig, ax = plt.subplots(figsize=(7.5, 0.5))
    ax.tick_params(width=2.5, length=8)
    ax.pcolormesh(x, y, problem.indicator[:, -problem.yVoxelLimit:], vmin=0, vmax=1, cmap=plt.cm.Greys,
                  edgecolors='none')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.gca().axes.get_xaxis().set_visible(False)
    ax.set_rasterized(True)
    fig.tight_layout(pad=0.1, h_pad=0.1, w_pad=0.1)
    if filename is not None:
        plt.savefig(filename + ".pdf")
    if show == True:
        plt.show()
    else:
        plt.close()


def visualizeDensityInColor(problem, filename=None, show=True):
    x = np.linspace(0, problem.Lx, problem.Nx * problem.numberOfVoxels)
    y = np.linspace(0, problem.Ly, problem.Ny * problem.numberOfVoxels)[-problem.yVoxelLimit:]
    x, y = np.meshgrid(x, y, indexing='ij')

    fig, ax = plt.subplots(figsize=(7.5, 1.4))
    ax.tick_params(width=2.5, length=8)
    cp = ax.pcolormesh(x, y, problem.indicator[:, -problem.yVoxelLimit:], vmin=0, vmax=1, cmap=plt.cm.jet,
                       edgecolors='none')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.gca().axes.get_xaxis().set_visible(False)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size="50%", pad="30%")
    cb = fig.colorbar(cp, cax=cax, format="%.1f", orientation="horizontal")
    cax.xaxis.set_ticks_position("top")
    cb.locator = matplotlib.ticker.MaxNLocator(nbins=5)  # adjust nbins accoring to data
    cb.update_ticks()
    cb.ax.tick_params(width=2.5, length=8)
    plt.minorticks_off()

    ax.set_rasterized(True)
    fig.tight_layout(pad=0.1, h_pad=0.1, w_pad=0.1)
    if filename is not None:
        plt.savefig(filename + ".pdf")
    if show == True:
        plt.show()
    else:
        plt.close()


def visualizeSolution(U, problem, pointsPerElement, filename=None, show=True, vmin=None, vmax=None):
    # solution grid
    x, y, u = FEM.getDisplacements(U, problem.Nx, problem.Ny, problem.p, problem.s, pointsPerElement,
                                   dtype=np.complex128)
    soundPressureLevel = 10 * np.log10(np.abs(u) ** 2 / problem.referencePressure ** 2)

    # density grid
    x_ = np.linspace(0, problem.Lx, problem.Nx * problem.numberOfVoxels)
    y_ = np.linspace(0, problem.Ly, problem.Ny * problem.numberOfVoxels)
    x_, y_ = np.meshgrid(x_, y_, indexing='ij')

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.tick_params(width=2.5, length=8)
    if vmin == None:
        vmin = np.min(soundPressureLevel)
    if vmax == None:
        vmax = np.max(soundPressureLevel)
    cp = ax.pcolormesh(x, y, soundPressureLevel, cmap=plt.cm.jet, norm='log',
                       vmin=vmin, vmax=vmax)  # ADD AS INPUT?

    maskedIndicator = np.ma.masked_equal(problem.indicator, 0)
    ax.pcolormesh(x_, y_, maskedIndicator, vmin=1, vmax=1, cmap=plt.cm.hot, edgecolors='none')

    plt.gca().set_aspect("equal", adjustable="box")

    ax.plot(problem.loadLocation[0], problem.loadLocation[1], 'ko', markersize=14)
    plt.gca().set_aspect('equal', adjustable='box')
    rect = patches.Rectangle((problem.supressElements[0] * problem.s, problem.supressElements[2] * problem.s),
                             (problem.supressElements[1] + 1 - problem.supressElements[0]) * problem.s,
                             (problem.supressElements[3] + 1 - problem.supressElements[2]) * problem.s, linewidth=4,
                             edgecolor='w',
                             facecolor='none')
    ax.add_patch(rect)

    plt.gca().set_aspect('equal', adjustable='box')
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.gca().axes.get_xaxis().set_visible(False)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size="7%", pad="3%")
    cb = fig.colorbar(cp, cax=cax, format="%3.0f", orientation="horizontal")
    cax.xaxis.set_ticks_position("top")
    cb.locator = matplotlib.ticker.MaxNLocator(nbins=8)  # adjust nbins accoring to data
    cb.update_ticks()
    cb.ax.tick_params(width=2.5, length=8)
    plt.minorticks_off()

    ax.set_rasterized(True)
    fig.tight_layout()
    if filename is not None:
        plt.savefig(filename + ".pdf")
    if show == True:
        plt.show()
    else:
        plt.close()
