import math

import torch
import torch.nn.functional as F
from efficient_kan import KAN
from neuralop.models import FNO
from torch import nn
from torch_geometric.nn import ChebConv, GATConv, GCNConv, GINConv, SAGEConv
from torch_geometric.nn.conv.message_passing import HookDict
from torchdiffeq import odeint

try:  # optional: only EquivariantCNN needs it, and it pins numpy < 2
    from escnn import nn as enn
except ImportError:
    enn = None

torch.backends.cudnn.deterministic = True

# --------------------------------------- helper --------------------------------------


def get_layer_param(param, i):  # in case param is a list
    return param[i] if isinstance(param, list) else param


class Passthrough(nn.Module):
    """identity that ignores extra forward arguments (a conditioned `nn.Identity`)."""

    def forward(self, x: torch.Tensor, *args) -> torch.Tensor:
        return x


# ------------------------------- primary architectures -------------------------------


class MLP(nn.Module):
    """multilayer perceptron.

    Each linear map is wrapped by optional `pre_modules` (applied before it) and
    `post_modules` (applied after it). Normalization, activation, and dropout are
    passed in as plain modules through these two slots, so any ordering can be
    expressed; the usual post-activation block is `[norm, activation, dropout]` in
    `post_modules`.

    Args:
        layers: layer sizes, e.g. [784, 256, 10] builds Linear(784, 256) and
            Linear(256, 10).
        post_modules: per-layer entry of None, a module, or a list of modules applied
            in order after the linear map. Shorter lists are padded with None.
        pre_modules: same format, applied before the linear map. Usually omitted.
    """

    def __init__(
        self,
        layers: list[int],
        post_modules: list[nn.Module | list[nn.Module] | None] | None = None,
        pre_modules: list[nn.Module | list[nn.Module] | None] | None = None,
    ) -> None:
        super().__init__()
        pre_modules = pre_modules or []
        post_modules = post_modules or []
        modules = []
        for i in range(len(layers) - 1):
            pre = pre_modules[i] if i < len(pre_modules) else None
            post = post_modules[i] if i < len(post_modules) else None
            for slot in (pre, nn.Linear(layers[i], layers[i + 1]), post):
                for module in slot if isinstance(slot, list) else [slot]:
                    if module is not None:
                        modules.append(module)
        self.model = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class DCN(nn.Module):
    """deep convolutional network with 1D, 2D, or 3D convolutions.

    Each convolution is wrapped by optional `pre_modules` / `post_modules`; see `MLP`
    for the slot mechanism. Resampling (pooling, `nn.Upsample`) goes into
    `pre_modules`, normalization / activation / dropout into `post_modules`.

    Args:
        channels: channel sizes, e.g. [3, 64, 128] builds two convolutions.
        post_modules: per-layer modules applied after each convolution; see `MLP`.
        kernel_size, stride, padding, dilation: convolution geometry, a scalar for
            every layer or a per-layer list.
        pre_modules: per-layer modules applied before each convolution.
        dim: spatial dimensionality (1, 2, or 3).
        bias: whether the convolutions carry a bias (scalar or per-layer list).
    """

    def __init__(
        self,
        channels: list[int],
        post_modules: list[nn.Module | list[nn.Module] | None] | None = None,
        kernel_size: int | list[int] = 3,
        stride: int | list[int] = 1,
        padding: int | list[int] = 0,
        pre_modules: list[nn.Module | list[nn.Module] | None] | None = None,
        dilation: int | list[int] = 1,
        dim: int = 2,
        bias: bool | list[bool] = False,
    ) -> None:
        super().__init__()
        pre_modules = pre_modules or []
        post_modules = post_modules or []
        conv = {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}[dim]
        modules = []
        for i in range(len(channels) - 1):
            core = conv(
                channels[i],
                channels[i + 1],
                get_layer_param(kernel_size, i),
                get_layer_param(stride, i),
                get_layer_param(padding, i),
                get_layer_param(dilation, i),
                bias=get_layer_param(bias, i),
            )
            pre = pre_modules[i] if i < len(pre_modules) else None
            post = post_modules[i] if i < len(post_modules) else None
            for slot in (pre, core, post):
                for module in slot if isinstance(slot, list) else [slot]:
                    if module is not None:
                        modules.append(module)
        self.model = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class EquivariantCNN(nn.Module):
    """E(2)-steerable convolutional network (escnn), the equivariant `DCN`.

    The output transforms consistently when the input is rotated by the symmetry group
    of `gspace`, so one training sample teaches its whole orbit of rotated inputs.
    Interior layers carry regular-representation fields; input and output are scalar
    fields by default, and `gspace.irrep(1)` makes one a 2D vector field (e.g.
    scalar-in / vector-out learns an equivariant gradient). The forward pass takes and
    returns plain (B, C, H, W) tensors.

    Reference: https://arxiv.org/abs/1911.08251

    Args:
        gspace: escnn GSpace of the symmetry group acting on R^2.
        channels: field copies per layer, e.g. [1, 8, 8, 1].
        activation: factory `activation(field_type)` called once per hidden layer,
            since a steerable nonlinearity must know the field type it acts on.
            Defaults to `enn.LeakyReLU`.
        kernel_size, padding, bias: convolution geometry, scalar or per-layer list.
        in_repr, out_repr: input / output representations; default to the trivial one.
    """

    def __init__(
        self,
        gspace,
        channels: list[int],
        activation=None,
        kernel_size: int | list[int] = 3,
        padding: int | list[int] = 0,
        bias: bool | list[bool] = False,
        in_repr=None,
        out_repr=None,
    ) -> None:
        super().__init__()
        activation = activation if activation is not None else enn.LeakyReLU
        in_repr = in_repr if in_repr is not None else gspace.trivial_repr
        out_repr = out_repr if out_repr is not None else gspace.trivial_repr
        reps = [in_repr, *[gspace.regular_repr] * (len(channels) - 2), out_repr]
        types = [enn.FieldType(gspace, [rep] * c) for rep, c in zip(reps, channels)]
        modules = []
        for i in range(len(channels) - 1):
            modules.append(
                enn.R2Conv(
                    types[i],
                    types[i + 1],
                    get_layer_param(kernel_size, i),
                    padding=get_layer_param(padding, i),
                    bias=get_layer_param(bias, i),
                )
            )
            if i < len(channels) - 2:
                modules.append(activation(types[i + 1]))
        self.in_type = types[0]
        self.model = enn.SequentialModule(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(enn.GeometricTensor(x, self.in_type)).tensor


class DGCN(nn.Module):
    """deep graph convolutional network of GCNConv layers.

    Reference: https://arxiv.org/abs/1609.02907

    Args:
        channels: channel sizes, e.g. [16, 32, 64] builds two layers.
        activations: activation module (or None) after each layer.
    """

    def __init__(
        self,
        channels: list[int],
        activations: list[nn.Module | None],
    ) -> None:
        super().__init__()
        self.convs = nn.ModuleList()
        self.acts: list[nn.Module | None] = []
        for i in range(len(channels) - 1):
            self.convs.append(GCNConv(channels[i], channels[i + 1]))
            self.acts.append(activations[i] if i < len(activations) else None)

    def forward(self, graph) -> torch.Tensor:
        x = graph.x
        for conv, act in zip(self.convs, self.acts):
            x = conv(x, graph.edge_index)
            if act:
                x = act(x)
        return x


class DGCheb(nn.Module):
    """deep graph network of Chebyshev spectral convolution layers.

    Reference: https://arxiv.org/abs/1606.09375

    Args:
        channels: channel sizes, e.g. [16, 32, 64] builds two layers.
        activations: activation module (or None) after each layer.
        K: Chebyshev filter order (number of hops).
    """

    def __init__(
        self,
        channels: list[int],
        activations: list[nn.Module | None],
        K: int = 3,
    ) -> None:
        super().__init__()
        self.convs = nn.ModuleList()
        self.acts: list[nn.Module | None] = []
        for i in range(len(channels) - 1):
            self.convs.append(ChebConv(channels[i], channels[i + 1], K=K))
            self.acts.append(activations[i] if i < len(activations) else None)

    def forward(self, graph) -> torch.Tensor:
        x = graph.x
        for conv, act in zip(self.convs, self.acts):
            x = conv(x, graph.edge_index)
            if act:
                x = act(x)
        return x


class DGSAGE(nn.Module):
    """deep GraphSAGE network of SAGEConv layers.

    Reference: https://arxiv.org/abs/1706.02216

    Args:
        channels: channel sizes, e.g. [16, 32, 64] builds two layers.
        activations: activation module (or None) after each layer.
    """

    def __init__(
        self,
        channels: list[int],
        activations: list[nn.Module | None],
    ) -> None:
        super().__init__()
        self.convs = nn.ModuleList()
        self.acts: list[nn.Module | None] = []
        for i in range(len(channels) - 1):
            self.convs.append(SAGEConv(channels[i], channels[i + 1]))
            self.acts.append(activations[i] if i < len(activations) else None)

    def forward(self, graph) -> torch.Tensor:
        x = graph.x
        for conv, act in zip(self.convs, self.acts):
            x = conv(x, graph.edge_index)
            if act:
                x = act(x)
        return x


class DGAT(nn.Module):
    """deep graph attention network of GATConv layers.

    References:
        https://arxiv.org/abs/1710.10903
        https://arxiv.org/abs/2105.14491

    Args:
        channels: channel sizes, e.g. [16, 32, 64] builds two layers.
        activations: activation module (or None) after each layer.
        heads: attention heads per layer.
    """

    def __init__(
        self,
        channels: list[int],
        activations: list[nn.Module | None],
        heads: int = 1,
    ) -> None:
        super().__init__()
        self.convs = nn.ModuleList()
        self.acts: list[nn.Module | None] = []
        for i in range(len(channels) - 1):
            self.convs.append(GATConv(channels[i], channels[i + 1], heads=heads))
            self.acts.append(activations[i] if i < len(activations) else None)

    def forward(self, graph) -> torch.Tensor:
        x = graph.x
        for conv, act in zip(self.convs, self.acts):
            x = conv(x, graph.edge_index)
            if act:
                x = act(x)
        return x


class DGIN(nn.Module):
    """deep graph isomorphism network of GINConv layers, each wrapping an `MLP`.

    Reference: https://arxiv.org/abs/1810.00826

    Args:
        mlp_layers: `MLP` layer sizes per GIN layer, e.g. [[16, 32], [32, 64]].
        mlp_activations: `MLP` activations per GIN layer.
        eps: initial self-loop weight epsilon.
        train_eps: whether epsilon is learnable.
    """

    def __init__(
        self,
        mlp_layers: list[list[int]],
        mlp_activations: list[list[nn.Module | None]],
        eps: float = 0,
        train_eps: bool = False,
    ) -> None:
        super().__init__()
        self.convs = nn.ModuleList()
        for i in range(len(mlp_layers)):
            mlp = MLP(mlp_layers[i], post_modules=mlp_activations[i])
            self.convs.append(GINConv(mlp, eps=eps, train_eps=train_eps))

    def forward(self, graph) -> torch.Tensor:
        x = graph.x
        for conv in self.convs:
            x = conv(x, graph.edge_index)
        return x


# ------------------------------------- sequential ------------------------------------


class DRNN(nn.Module):
    """deep recurrent network of RNN, LSTM, or GRU cells with a linear projection.

    References:
        https://ieeexplore.ieee.org/abstract/document/6795963
        https://arxiv.org/abs/1412.3555

    Args:
        layers: layer sizes; the last two define the projection, e.g. [64, 128, 256, 10]
            builds two recurrent layers (64->128, 128->256) and Linear(256, 10).
        final_activation: optional activation after the projection.
        cell: nn.RNN, nn.LSTM, or nn.GRU.
        normalizations: normalization module (or None) after each recurrent layer.
    """

    def __init__(
        self,
        layers: list[int],
        final_activation: nn.Module | None = None,
        cell: type[nn.RNN | nn.LSTM | nn.GRU] = nn.RNN,
        normalizations: list[nn.Module | None] | None = None,
    ) -> None:
        super().__init__()
        normalizations = normalizations or []
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 2):
            self.layers.append(cell(layers[i], layers[i + 1], batch_first=True))
            if normalizations and i < len(normalizations):
                if normalizations[i]:
                    self.layers.append(normalizations[i])
        self.projection = nn.Linear(layers[-2], layers[-1])
        self.final_activation = final_activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for module in self.layers:
            if isinstance(module, (nn.RNN, nn.LSTM, nn.GRU)):
                x, _ = module(x)
            else:
                x = module(x)
        x = self.projection(x)
        if self.final_activation:
            x = self.final_activation(x)
        return x


class NODE(nn.Module):
    """neural ordinary differential equation: dh/dt is a network integrated by odeint.

    Reference: https://arxiv.org/abs/1806.07366

    Args:
        rhs_model: maps (batch, hidden + 1), the state with time appended, to dh/dt of
            shape (batch, hidden).
    """

    def __init__(self, rhs_model: nn.Module) -> None:
        super().__init__()
        self.rhs_model = rhs_model

    def eval_rhs(self, t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """evaluate dh/dt with time appended to the state."""
        t_vec = torch.ones(h.shape[0], 1, device=h.device) * t
        x = torch.cat([h, t_vec], dim=1)
        return self.rhs_model(x)

    def forward(self, T: torch.Tensor, h0: torch.Tensor) -> torch.Tensor:
        """integrate from h0 over the time points T; returns (len(T), batch, hidden)."""
        return odeint(self.eval_rhs, h0, T)


# -------------------------------------- Bayesian -------------------------------------


class BayesianLinear(nn.Module):
    """linear layer with Gaussian weight distributions (Bayes by Backprop).

    Weights are sampled as mu + softplus(rho) * eps at every forward pass.

    Reference: https://arxiv.org/abs/1505.05424
    """

    def __init__(self, inputs: int, outputs: int) -> None:
        super().__init__()
        self.weight_mu = nn.Parameter(torch.Tensor(outputs, inputs))
        self.weight_rho = nn.Parameter(torch.Tensor(outputs, inputs))
        self.bias_mu = nn.Parameter(torch.Tensor(outputs))
        self.bias_rho = nn.Parameter(torch.Tensor(outputs))

        self.init_params()

    def init_params(self) -> None:
        """Kaiming-uniform mu and constant rho = -3 (sigma about 0.05)."""
        nn.init.kaiming_uniform_(self.weight_mu, a=math.sqrt(5.0))
        nn.init.constant_(self.weight_rho, -3.0)
        nn.init.constant_(self.bias_mu, 0.0)
        nn.init.constant_(self.bias_rho, -3.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """sample the weights and apply the linear map."""
        weight_sigma = F.softplus(self.weight_rho)
        bias_sigma = F.softplus(self.bias_rho)

        weight_epsilon = torch.randn_like(self.weight_mu)
        bias_epsilon = torch.randn_like(self.bias_mu)

        weight = self.weight_mu + weight_sigma * weight_epsilon
        bias = self.bias_mu + bias_sigma * bias_epsilon

        return F.linear(x, weight, bias)

    def kl_divergence(self, prior_std: float) -> torch.Tensor:
        """KL divergence of the weight posteriors to a zero-mean Gaussian prior."""
        weight_sigma = F.softplus(self.weight_rho)
        bias_sigma = F.softplus(self.bias_rho)

        kl_weight = (
            torch.log(prior_std / weight_sigma)
            + (weight_sigma**2 + self.weight_mu**2) / (2 * prior_std**2)
            - 0.5
        )
        kl_bias = (
            torch.log(prior_std / bias_sigma)
            + (bias_sigma**2 + self.bias_mu**2) / (2 * prior_std**2)
            - 0.5
        )
        return kl_weight.sum() + kl_bias.sum()


class BayesianMLP(nn.Module):
    """Bayesian multilayer perceptron of `BayesianLinear` layers.

    Takes the `layers` and `pre_modules` / `post_modules` slots of `MLP`.

    Reference: https://arxiv.org/abs/1505.05424
    """

    def __init__(
        self,
        layers: list[int],
        post_modules: list[nn.Module | list[nn.Module] | None] | None = None,
        pre_modules: list[nn.Module | list[nn.Module] | None] | None = None,
    ) -> None:
        super().__init__()
        pre_modules = pre_modules or []
        post_modules = post_modules or []
        modules = []
        for i in range(len(layers) - 1):
            pre = pre_modules[i] if i < len(pre_modules) else None
            post = post_modules[i] if i < len(post_modules) else None
            for slot in (pre, BayesianLinear(layers[i], layers[i + 1]), post):
                for module in slot if isinstance(slot, list) else [slot]:
                    if module is not None:
                        modules.append(module)
        self.model = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def kl_divergence(self, prior_std: float) -> torch.Tensor:
        """sum of the KL divergences of all `BayesianLinear` layers."""
        kl = 0
        for module in self.model.modules():
            if isinstance(module, BayesianLinear):
                kl += module.kl_divergence(prior_std)
        return kl


# -------------------------------------- resnets --------------------------------------


class ResidualBlock(nn.Module):
    """skip connection `projection(x) + module(x)`; the projection defaults to identity.

    Reference: https://arxiv.org/abs/1512.03385
    """

    def __init__(
        self,
        module: nn.Module,
        projection: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.module = module
        self.projection = projection

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        if self.projection is not None:
            identity = self.projection(x)
        return identity + self.module(x)


class ConditionedResidualBlock(nn.Module):
    """residual block whose hidden state is shifted by a projected conditioning vector.

    The block of diffusion U-Nets: `module_out(module_in(x) + embedding(c))` added to
    the (optionally projected) input, with the embedding broadcast as one bias per
    channel.

    Args:
        module_in: first transformation, e.g. norm -> activation -> conv.
        module_out: second transformation, applied after the shift.
        embedding: maps the condition (e.g. a timestep embedding) to one bias per
            output channel of `module_in`.
        projection: optional projection of the input to the output shape.
    """

    def __init__(
        self,
        module_in: nn.Module,
        module_out: nn.Module,
        embedding: nn.Module,
        projection: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.module_in = module_in
        self.module_out = module_out
        self.embedding = embedding
        self.projection = projection

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        identity = x
        if self.projection is not None:
            identity = self.projection(x)
        h = self.module_in(x)
        bias = self.embedding(condition)
        h = h + bias.view(*bias.shape, *([1] * (h.dim() - 2)))
        return identity + self.module_out(h)


class ResNet(nn.Module):
    """wrap layer ranges of a sequential model into `ResidualBlock`s.

    Args:
        base_model: model with a `.model` attribute holding an nn.Sequential.
        skip_connections: (start, end) index pairs of the layers to wrap, inclusive.
        projections: optional (start, end) -> projection module for shape mismatches.
    """

    def __init__(
        self,
        base_model: nn.Module,
        skip_connections: list[tuple[int, int]],
        projections: dict[tuple[int, int], nn.Module] | None = None,
    ) -> None:
        super().__init__()

        layers = list(base_model.model.children())
        modules = []
        skip_dict = {start: end for start, end in skip_connections}

        i = 0
        while i < len(layers):
            if i in skip_dict:
                end_idx = skip_dict[i]
                block_layers = layers[i : end_idx + 1]
                block = nn.Sequential(*block_layers)
                projection = None
                if projections and (i, end_idx) in projections:
                    projection = projections[(i, end_idx)]

                modules.append(ResidualBlock(block, projection))
                i = end_idx + 1
            else:
                modules.append(layers[i])
                i += 1

        self.model = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# ------------------------------- input convex networks -------------------------------


class ICNNLayer(nn.Module):
    """single layer `z_out = W_z z + W_y x + b` of an input convex network.

    Set `z_in=0` for the first layer, which has no z path. Call `clamp_z_()` after each
    optimizer step to keep `W_z` non-negative.
    """

    def __init__(self, z_in: int, x_in: int, out: int) -> None:
        super().__init__()
        self.Wz = nn.Linear(z_in, out, bias=False) if z_in > 0 else None
        self.Wy = nn.Linear(x_in, out, bias=True)

    def forward(self, z: torch.Tensor | None, x: torch.Tensor) -> torch.Tensor:
        h = self.Wy(x)
        if self.Wz is not None:
            h = h + self.Wz(z)
        return h

    @torch.no_grad()
    def clamp_z_(self) -> None:
        if self.Wz is not None:
            self.Wz.weight.clamp_(min=0)


class ICNN(nn.Module):
    """input convex neural network.

    Stack of `ICNNLayer`s threading the input x into every layer. With convex
    non-decreasing activations (ReLU, ELU, Softplus) and `clamp_z_()` after each
    optimizer step, the output is convex in x.

    Reference: https://arxiv.org/abs/1609.07152

    Args:
        layers: layer sizes from input to output.
        activations: activation module (or None) after each layer; the last is usually
            None.
    """

    def __init__(
        self,
        layers: list[int],
        activations: list[nn.Module | None] | None = None,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                ICNNLayer(
                    z_in=0 if i == 0 else layers[i], x_in=layers[0], out=layers[i + 1]
                )
                for i in range(len(layers) - 1)
            ]
        )
        self.activations = activations or []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = None
        for i, layer in enumerate(self.layers):
            z = layer(z, x)
            if i < len(self.activations) and self.activations[i] is not None:
                z = self.activations[i](z)
        return z

    @torch.no_grad()
    def clamp_z_(self) -> None:
        for layer in self.layers:
            layer.clamp_z_()


# ----------------------------- extreme learning machines -----------------------------
class ELM(nn.Module):
    """extreme learning machine: frozen random features plus a least-squares readout.

    `fit` solves the ridge-regularized normal equations for the output layer only; a
    subsequent training of all weights is optional.

    Reference: https://ieeexplore.ieee.org/document/1380068

    Args:
        feature_extractor: network mapping the input to `hidden_dim` features.
    """

    def __init__(self, feature_extractor: nn.Module, hidden_dim: int, output_dim: int):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.output_layer = nn.Linear(hidden_dim, output_dim, bias=True)
        self.output_layer_weights = nn.Parameter(torch.zeros(output_dim, hidden_dim))
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

    def fit(self, x: torch.Tensor, y: torch.Tensor, regularization: float):
        """closed-form ridge fit of the output layer; the bias is not penalized."""
        with torch.no_grad():
            H = self.feature_extractor(x)
        num_samples = H.shape[0]
        bias_ones = torch.ones(num_samples, 1, device=x.device)
        H_aug = torch.cat([H, bias_ones], 1)
        I = torch.eye(H_aug.shape[1], device=x.device)
        I[-1, -1] = 0  # do not penalize bias

        # least squares fit
        A = (H_aug.T @ H_aug) + (regularization * I)
        b = H_aug.T @ y
        W_aug = torch.linalg.solve(A, b)
        self.output_layer.weight.data = W_aug[:-1, :].T
        self.output_layer.bias.data = W_aug[-1, :]

    def forward(self, x: torch.Tensor):
        x = self.feature_extractor(x)
        return self.output_layer(x)


# ------------------------------- deep material networks ------------------------------
def laminate_rotation(alpha):
    """Voigt rotation matrix by angle alpha, differentiable in alpha."""
    c, s = torch.cos(alpha), torch.sin(alpha)
    return torch.stack(  # to not break autograd
        [
            torch.stack([c**2, s**2, 2 * c * s]),
            torch.stack([s**2, c**2, -2 * c * s]),
            torch.stack([-c * s, c * s, c**2 - s**2]),
        ]
    )


class LaminateBlock(nn.Module):
    """two-phase laminate building block of a deep material network."""

    def __init__(self):
        super().__init__()
        # random volume fraction breaks the symmetry of an all-equal init; otherwise
        # every block stays identical and the fit stalls (alpha stays 0: a large random
        # rotation drives the homogenized stiffness non-physical)
        self.v1_logit = nn.Parameter(0.5 * torch.randn(()))  # with sigmoid 0<=v1<=1
        self.alpha = nn.Parameter(torch.zeros(()))

    @property
    def v1(self):
        return torch.sigmoid(self.v1_logit)

    def homogenize_stiffness(self, C1, C2):
        # C1, C2 are batched stiffnesses (N, 3, 3); homogenizes over the leading dim
        v1, v2 = self.v1, 1 - self.v1
        A1, B1, D1 = C1[:, 0, 0], C1[:, 0, 1:], C1[:, 1:, 1:]
        A2, B2, D2 = C2[:, 0, 0], C2[:, 0, 1:], C2[:, 1:, 1:]
        D1_inv, D2_inv = torch.linalg.inv(D1), torch.linalg.inv(D2)

        D_tilde_inv = torch.linalg.inv(v1 * D1_inv + v2 * D2_inv)
        P = v1 * torch.einsum("nij,nj->ni", D1_inv, B1) + v2 * torch.einsum(
            "nij,nj->ni", D2_inv, B2
        )
        A_bar = (
            v1 * (A1 - torch.einsum("ni,nij,nj->n", B1, D1_inv, B1))
            + v2 * (A2 - torch.einsum("ni,nij,nj->n", B2, D2_inv, B2))
            + torch.einsum("ni,nij,nj->n", P, D_tilde_inv, P)
        )
        B_bar, D_bar = torch.einsum("nij,nj->ni", D_tilde_inv, P), D_tilde_inv

        C_bar = torch.zeros_like(C1)
        C_bar[:, 0, 0], C_bar[:, 0, 1:], C_bar[:, 1:, 0], C_bar[:, 1:, 1:] = (
            A_bar,
            B_bar,
            B_bar,
            D_bar,
        )
        C = laminate_rotation(self.alpha) @ C_bar @ laminate_rotation(-self.alpha)

        cache = (D1, D2, B1, B2, v1, v2)  # reused in recover_strains
        return C, cache

    def recover_strains(self, deps, cache):
        # deps is a batch of strain increments (N, 3)
        D1, D2, B1, B2, v1, v2 = cache
        deps_l = torch.einsum("ij,nj->ni", laminate_rotation(-self.alpha), deps)
        eps11 = deps_l[:, 0]
        M = D1 + (v1 / v2) * D2
        rhs = (
            torch.einsum("nij,nj->ni", D2, deps_l[:, 1:]) / v2
            + (B2 - B1) * eps11[:, None]
        )
        x1 = torch.linalg.solve(M, rhs)
        x2 = (deps_l[:, 1:] - v1 * x1) / v2
        deps1_l = torch.cat([eps11[:, None], x1], dim=1)
        deps2_l = torch.cat([eps11[:, None], x2], dim=1)
        T = laminate_rotation(self.alpha)
        return torch.einsum("ij,nj->ni", T, deps1_l), torch.einsum(
            "ij,nj->ni", T, deps2_l
        )

    def homogenize_stress(self, dsig1, dsig2):
        # dsig1, dsig2 are batched stresses (N, 3)
        return self.v1 * dsig1 + (1 - self.v1) * dsig2


class DMN(nn.Module):
    """deep material network: a binary tree of `depth` laminate layers.

    `homogenize` runs the stiffness bottom-up and the strains top-down for arbitrary
    per-leaf stiffnesses (e.g. nonlinear tangents); `forward` is the linear two-phase
    case with alternating leaves.

    Reference: https://doi.org/10.1016/j.cma.2018.09.020
    """

    def __init__(self, depth):
        super().__init__()
        self.depth = depth
        self.layers = nn.ModuleList(
            [
                nn.ModuleList([LaminateBlock() for _ in range(2 ** (depth - l - 1))])
                for l in range(depth)
            ]
        )

    def forward(self, C1, C2, deps):
        # linear two-phase cell: leaves alternate phase 1 / phase 2
        leaf_C = [C1 if i % 2 == 0 else C2 for i in range(2**self.depth)]
        return self.homogenize(leaf_C, deps)

    def homogenize(self, leaf_C, deps):
        # bottom-up stiffness then top-down strains for a tree of per-leaf stiffnesses
        # leaf_C (list of 2**depth batched (N, 3, 3)); deps macro increment (N, 3).
        # This generalizes forward: each leaf may carry its own (e.g. nonlinear
        # tangent) stiffness

        # bottom-up: homogenize stiffness, leaves to root
        all_cache = []
        cur_C = leaf_C
        for layer in self.layers:
            cur_C, cache = zip(
                *(
                    block.homogenize_stiffness(cur_C[2 * i], cur_C[2 * i + 1])
                    for i, block in enumerate(layer)
                )
            )
            all_cache.append(cache)
        C_root = cur_C[0]

        # top-down: recover per-phase strains, root to leaves
        cur_deps = [deps]
        for layer, layer_cache in zip(reversed(self.layers), reversed(all_cache)):
            cur_deps = [
                child
                for block, cache, d in zip(layer, layer_cache, cur_deps)
                for child in block.recover_strains(d, cache)
            ]
        return C_root, cur_deps

    def homogenize_stress(self, leaf_dsig):
        # bottom-up: compute homogenized stress, leaves to root
        cur_dsig = [
            block.homogenize_stress(leaf_dsig[2 * i], leaf_dsig[2 * i + 1])
            for i, block in enumerate(self.layers[0])
        ]
        for l in range(1, self.depth):
            cur_dsig = [
                block.homogenize_stress(cur_dsig[2 * i], cur_dsig[2 * i + 1])
                for i, block in enumerate(self.layers[l])
            ]
        return cur_dsig[0]


# ------------------------------------ autoencoders -----------------------------------


class AE(nn.Module):
    """autoencoder `decode(encode(x))`."""

    def __init__(self, encoder: nn.Module, decoder: nn.Module) -> None:
        super().__init__()
        self.encode = encoder
        self.decode = decoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encode(x)
        x = self.decode(x)
        return x


class VAE(AE):
    """variational autoencoder with the reparameterization trick.

    The encoder outputs the concatenated (mean, logvar) of the latent, i.e. twice the
    latent size; the latent is only sampled in training mode.

    Reference: https://arxiv.org/abs/1312.6114
    """

    def reparameterize(
        self,
        mean: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        """sample `mean + std * eps` in training mode, return the mean otherwise."""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mean + eps * std
        return mean

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """encode, sample, decode; returns (reconstruction, mean, logvar)."""
        distributions = self.encode(x)
        mean, logvar = torch.chunk(distributions, chunks=2, dim=1)
        z = self.reparameterize(mean, logvar)
        y = self.decode(z)
        return y, mean, logvar


class UNet(nn.Module):
    """U-Net: encoder-decoder with a skip connection at each level.

    Each `downs` output is stored as a skip; each `ups` module receives the previous
    output concatenated with the matching skip along the channel axis, so it must
    accept (input + skip) channels. Extra `forward` arguments (e.g. a timestep
    embedding) are passed on to every down, bottleneck, and up module, so a
    `ConditionedResidualBlock` slots in without subclassing.

    Reference: https://arxiv.org/abs/1505.04597

    Args:
        downs: encoder modules, shallow to deep.
        ups: decoder modules, deep to shallow, same length as `downs`.
        bottleneck: module at the deepest level; defaults to identity.
        downsamplers: optional per-level modules after each down module (the skip is
            taken before, at full resolution).
        upsamplers: optional per-level modules before each up module, deep to shallow.
    """

    def __init__(
        self,
        downs: list[nn.Module],
        ups: list[nn.Module],
        bottleneck: nn.Module | None = None,
        downsamplers: list[nn.Module] | None = None,
        upsamplers: list[nn.Module] | None = None,
    ) -> None:
        super().__init__()
        self.downs = nn.ModuleList(downs)
        self.ups = nn.ModuleList(ups)
        self.bottleneck = bottleneck or Passthrough()
        self.downsamplers = nn.ModuleList(
            downsamplers or [nn.Identity() for _ in downs]
        )
        self.upsamplers = nn.ModuleList(upsamplers or [nn.Identity() for _ in ups])

    def forward(self, x: torch.Tensor, *args) -> torch.Tensor:
        skips = []
        for down, downsample in zip(self.downs, self.downsamplers):
            x = down(x, *args)
            skips.append(x)
            x = downsample(x)
        x = self.bottleneck(x, *args)
        for up, upsample, skip in zip(self.ups, self.upsamplers, reversed(skips)):
            x = up(torch.cat([upsample(x), skip], dim=1), *args)
        return x


# ---------------------------------- neural operators ---------------------------------
class DeepONet(nn.Module):
    """deep operator network: contraction of a branch net and a trunk net.

    The branch net maps the sampled input function to (batch, p * q), the trunk net the
    query coordinate to (batch, p); the output is their dot product over p plus a bias.

    Reference: https://arxiv.org/abs/1910.03193
    """

    def __init__(
        self, branchNet: nn.Module, trunkNet: nn.Module, output_dim: int = 1
    ) -> None:
        super().__init__()
        self.q = output_dim
        self.branch = branchNet  # output: (batch, p * q)
        self.trunk = trunkNet  # output: (batch, p)
        self.bias = nn.Parameter(torch.zeros(self.q))

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        t = self.trunk(x).unsqueeze(1)  # (batch, 1, p)
        b = self.branch(g).view(-1, self.q, t.shape[-1])  # (batch, q, p)
        return (b * t).sum(dim=-1) + self.bias  # (batch, q)


# FNO is imported from neuralop, https://arxiv.org/abs/2010.08895

# ----------------------------------- siren network -----------------------------------


class SIRENsine(nn.Module):
    """sine activation `sin(omega_0 x)` of a SIREN.

    Reference: https://arxiv.org/abs/2006.09661
    """

    def __init__(self, omega_0: float = 30.0) -> None:
        super().__init__()
        self.omega_0 = omega_0

    def forward(self, x):
        return torch.sin(self.omega_0 * x)


# ----------------------------- kolmogorov-arnold network -----------------------------
# KAN is imported from efficient_kan, https://arxiv.org/abs/2404.19756

