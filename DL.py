import math

import torch
from torch import nn

from NN import SIRENsine


# ------------------------ weight initialization -------------------------
def init_weights(model, activation=None):
    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            if isinstance(
                activation,
                (
                    nn.ReLU,
                    nn.ReLU6,
                    nn.Hardswish,
                    nn.SiLU,
                    nn.Mish,
                    nn.GELU,
                    nn.ELU,
                    nn.CELU,
                    nn.Softplus,
                ),
            ):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
            elif isinstance(activation, nn.LeakyReLU):
                nn.init.kaiming_uniform_(
                    m.weight, a=activation.negative_slope, nonlinearity="leaky_relu"
                )
            elif isinstance(activation, nn.PReLU):
                nn.init.kaiming_uniform_(
                    m.weight, a=activation.init, nonlinearity="leaky_relu"
                )
            elif isinstance(activation, nn.RReLU):
                a = (activation.lower + activation.upper) / 2.0
                nn.init.kaiming_uniform_(m.weight, a=a, nonlinearity="leaky_relu")
            elif isinstance(activation, nn.SELU):
                nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
            elif isinstance(activation, (nn.Tanh, nn.Softsign)):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain("tanh"))
            elif isinstance(activation, (nn.Sigmoid, nn.LogSigmoid, nn.Softmax)):
                nn.init.xavier_uniform_(
                    m.weight, gain=nn.init.calculate_gain("sigmoid")
                )
            elif isinstance(activation, SIRENsine):
                linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
                for i, linear in enumerate(linears):
                    if i == 0:
                        bound = 1.0 / linear.in_features
                    else:
                        bound = math.sqrt(6.0 / linear.in_features) / activation.omega_0
                    nn.init.uniform_(linear.weight, -bound, bound)
                    nn.init.uniform_(linear.bias, -bound, bound)
                return
            else:
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)


# ----------------- data normalization & standardization -----------------
class Normalizer(nn.Module):
    def __init__(self, X, dim=0):  # default is to have sample dim at 0
        super().__init__()
        self.x_max = X.max(dim=dim, keepdim=True)
        self.x_min = X.min(dim=dim, keepdim=True)
        self.span = (self.x_max - self.x_in).clamp_min(1e-8)

    def __call__(self, x):
        return (x - self.x_min) / self.span

    def inverse(self, x):
        return x * self.span + self.x_min


class Standardizer(nn.Module):
    def __init__(self, X, dim=0):  # default is to have sample dim at 0
        super().__init__()
        self.x_mean = X.mean(dim=dim, keepdim=True)
        self.x_std = X.std(dim=dim, keepdim=True).clamp_min(1e-8)

    def __call__(self, x):
        return (x - self.x_mean) / self.x_std

    def inverse(self, x):
        return x * self.x_std + self.x_mean


# ------------------------ network configurations ------------------------
def build_ae_cnn_config(depth, conv_layers, channel_dim, base):
    channels, strides = [], []

    for i in range(depth + 1):
        # starting channel at depth i
        channels.append(channel_dim * (base**i))
        strides.append(1)

        # intermediate conv layers (except last depth)
        if i < depth:
            for _ in range(conv_layers):
                channels.append(channel_dim * (base ** (i + 1)))
                strides.append(1)
            strides[-1] = 2  # last stride in each group downsamples

    return channels, strides[:-1]


# ----------------- optimization landscape visualization -----------------
def get_params(model: nn.Module, kind: str = "all") -> list[torch.Tensor]:
    if kind == "all":
        return [p.detach().clone() for p in model.parameters()]
    if kind == "weights":
        return [
            p.detach().clone()
            for n, p in model.named_parameters()
            if not n.endswith(".bias")
        ]
    if kind == "biases":
        return [
            p.detach().clone()
            for n, p in model.named_parameters()
            if n.endswith(".bias")
        ]
    raise ValueError(f"unknown kind: {kind!r}")


def set_params(model: nn.Module, params: list[torch.Tensor]) -> None:
    with torch.no_grad():
        for p, v in zip(model.parameters(), params):
            p.copy_(v)


def flatten_params(params: list[torch.Tensor]) -> torch.Tensor:
    return torch.cat([p.view(-1) for p in params])


def unflatten_params(vec: torch.Tensor, ref: list[torch.Tensor]) -> list[torch.Tensor]:
    out, i = [], 0
    for p in ref:
        n = p.numel()
        out.append(vec[i : i + n].view_as(p))
        i += n
    return out


def filter_normalize_direction(
    direction: list[torch.Tensor], reference: list[torch.Tensor]
) -> list[torch.Tensor]:
    """Scale each filter (inputs to each output neuron) in direction to match the norm of the corresponding filter
    in reference. Makes alpha meaningful across architectures."""
    normed = []
    for d, w in zip(direction, reference):
        if d.dim() >= 2:
            d_norm = d.norm(dim=tuple(range(1, d.dim())), keepdim=True) + 1e-10
            w_norm = w.norm(dim=tuple(range(1, w.dim())), keepdim=True)
            normed.append(d * (w_norm / d_norm))
        else:
            normed.append(d * (w.norm() / (d.norm() + 1e-10)))
    return normed


# ----------------------------- KAN helpers ------------------------------
def count_kan_params(model):
    base_params = 0
    spline_params = 0

    for name, param in model.named_parameters():
        if "base_weight" in name:
            base_params += param.numel()
        elif "spline_weight" in name or "spline_scaler" in name:
            spline_params += param.numel()

    return base_params, spline_params


def get_kan_edge_activations(model, layer_id, resolution=200):
    layer = model.layers[layer_id]
    dev = layer.grid.device

    grid = layer.grid  # (in_features, grid_size + 2*spline_order + 1)
    p = layer.spline_order
    in_features = layer.in_features
    out_features = layer.out_features

    x = torch.zeros((out_features, in_features, resolution))
    activations = torch.zeros((out_features, in_features, resolution))

    for i in range(in_features):
        x_line = torch.linspace(
            grid[i, p].item(), grid[i, -p - 1].item(), resolution, device=dev
        )
        x_full = torch.zeros(resolution, in_features, device=dev)
        x_full[:, i] = x_line

        with torch.no_grad():
            basis = layer.b_splines(
                x_full
            )  # (resolution, in_features, grid_size + spline_order)

        basis_i = basis[:, i, :]  # (resolution, grid_size + spline_order)

        base_out = layer.base_activation(x_line)  # (resolution,)

        for j in range(out_features):
            spline = basis_i @ layer.scaled_spline_weight[j, i, :]
            y_eval = (spline + layer.base_weight[j, i] * base_out).detach().cpu()

            x[j, i] = x_line.cpu()
            activations[j, i] = y_eval

    return x, activations
