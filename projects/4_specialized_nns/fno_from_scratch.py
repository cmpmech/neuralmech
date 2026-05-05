import torch
import torch.fft
import torch.nn as nn


class FNOLayer1D(nn.Module):
    def __init__(self, in_channels, out_channels, k, activation):
        super().__init__()
        self.k = k
        self.W = nn.Parameter(
            torch.randn((out_channels, in_channels), dtype=torch.cfloat)
        )  # could also be done with nn.Linear
        self.bypass = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.activation = activation

    def forward(self, x):
        L = x.shape[-1]
        spectral = self.W @ torch.fft.rfft(x)[:, : self.k]
        spectral = torch.fft.irfft(spectral, n=L)
        local = self.bypass(x)
        return self.activation(spectral + local)


Cx, Cy, L, k = 2, 3, 10, 4
x = torch.randn((Cx, L))
activation = torch.nn.ReLU()

layer = FNOLayer1D(Cx, Cy, k, activation)
y = layer(x)  # (Cy, L)

print(y.shape)
