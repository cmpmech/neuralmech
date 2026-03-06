import numpy as np
import numpy.typing as npt

def sineburst(t: npt.NDArray[np.float32] | npt.NDArray[np.float64], amplitude: float, frequency: float, cycles: int) -> \
npt.NDArray[np.float32] | npt.NDArray[np.float64]:
    angular_frequency = 2 * np.pi * frequency
    return amplitude * ((t <= cycles / frequency) & (t > 0)) * np.sin(angular_frequency * t) * (
        np.sin(angular_frequency * t / 2 / cycles)) ** 2