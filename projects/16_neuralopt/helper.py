import json
import math

import cupy as cp
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap
from torch import nn

from cuwave.geometry import random_ellipses, stacked_circles
from cuwave.scalar import ScalarWave
from cuwave.signals import sineburst
from cuwave.utils import Sensors, line, measure, resample, shots
from cuwave.wave import grid_coords, stable_dt

# synthetic FWI setup of cuwave/examples/fwi/scalar2D_fwi_adam.py
SPACE_ORDER = 4
SPACE_ORDER_OBS = 8  # measurement is simulated more accurately than it is inverted
RESOLUTION = (512, 512)
SAFETY = 0.99
GAMMA_MIN = 1e-3
GAMMA_VOID = 1e-4
T = 2.5  # traversals per length
AMPLITUDE, FREQUENCY, CYCLES = 1.0, 20.0, 2
NUM_SOURCES, NUM_SENSORS = 4, 32
ARRAY_SPAN = (0.1, 0.9)


class FWI:
    """scalar-wave FWI on the unit square, sources and sensors on the top surface."""

    def __init__(self) -> None:
        Nx = RESOLUTION
        dx = tuple(1.0 / (n - 3) for n in Nx)
        steps = lambda order: math.ceil(T / (SAFETY * stable_dt(dx, 1.0, order))) + 1
        wave = lambda N, order: ScalarWave(
            Nx,
            dx,
            N,
            T / (N - 1),
            (4, 64),
            precision="float32",
            space_order=order,
            wavespeed=1.0,
            density=1.0,
        )
        self.sim = wave(steps(SPACE_ORDER), SPACE_ORDER)
        self.sim_obs = wave(steps(SPACE_ORDER_OBS), SPACE_ORDER_OBS)

        surface = lambda count: line((ARRAY_SPAN[0], 1.0), (ARRAY_SPAN[1], 1.0), count)
        self.source_coords = surface(NUM_SOURCES)
        self.sensor_coords = surface(NUM_SENSORS)
        burst = lambda sim: sineburst(
            np.linspace(0, T, sim.N), AMPLITUDE, FREQUENCY, CYCLES
        )
        self.sources = shots(self.sim, self.source_coords, burst(self.sim))
        self.sensors = Sensors(self.sim, self.sensor_coords)
        self.sources_obs = shots(self.sim_obs, self.source_coords, burst(self.sim_obs))
        self.sensors_obs = Sensors(self.sim_obs, self.sensor_coords)
        self.coords = grid_coords(Nx, dx, dtype=self.sim.dtype)

    def indicator(self, voids: cp.ndarray) -> cp.ndarray:
        return cp.where(voids, GAMMA_VOID, 1.0).astype(self.sim.dtype)

    def circles(self) -> cp.ndarray:
        """the cuwave example defect: four stacked circles, halving in radius."""
        voids = stacked_circles(
            self.coords, 4, 0.05, (0.1, 0.5), 0.5, axis=1, order="ascending"
        )
        return self.indicator(voids)

    def ellipses(self, rng: np.random.Generator) -> cp.ndarray:
        """1 to 4 random, nonoverlapping elliptical voids in the upper part."""
        count = int(rng.integers(1, 5))
        bounds = [(0.15, 0.85), (0.1, 0.8)]
        voids = random_ellipses(
            self.coords, count, (0.005, 0.08), bounds, overlap=False, rng=rng
        )
        return self.indicator(voids)

    def observe(self, truth: cp.ndarray) -> list[cp.ndarray]:
        records = measure(self.sim_obs, self.sources_obs, truth, self.sensors_obs)
        return [resample(r, self.sim_obs.dt, self.sim.dt, self.sim.N) for r in records]


def normalize(gradient: torch.Tensor) -> torch.Tensor:
    """scale-free network input: the gradient divided by its largest magnitude."""
    return gradient / gradient.abs().amax(dim=(-2, -1), keepdim=True)


def block(in_channels: int, out_channels: int) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.GroupNorm(1, out_channels),
        nn.GELU(),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.GELU(),
    )


class UNet(nn.Module):
    """U-Net from the normalized first gradient to the indicator logits."""

    def __init__(self, channels: list[int]) -> None:
        super().__init__()
        widths = [1] + channels
        self.downs = nn.ModuleList(
            block(widths[i], widths[i + 1]) for i in range(len(channels) - 1)
        )
        self.bottleneck = block(channels[-2], channels[-1])
        self.ups = nn.ModuleList(
            block(channels[i + 1] + channels[i], channels[i])
            for i in reversed(range(len(channels) - 1))
        )
        self.output = nn.Conv2d(channels[0], 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = nn.functional.max_pool2d(x, 2)
        x = self.bottleneck(x)
        for up, skip in zip(self.ups, reversed(skips)):
            x = nn.functional.interpolate(x, scale_factor=2, mode="nearest")
            x = up(torch.cat([x, skip], dim=1))
        return self.output(x)


def load_cmap(path) -> LinearSegmentedColormap:
    """matplotlib colormap from a ParaView .cmap (JSON) file."""
    data = json.loads(path.read_text())
    pts = data["RGBPoints"]
    xs = pts[0::4]
    stops = [
        ((x - xs[0]) / (xs[-1] - xs[0]), tuple(pts[4 * i + 1 : 4 * i + 4]))
        for i, x in enumerate(xs)
    ]
    return LinearSegmentedColormap.from_list(data.get("Name", "cmap"), stops)
