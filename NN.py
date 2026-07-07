import math

import torch
import torch.nn.functional as F
from efficient_kan import KAN
from escnn import nn as enn
from neuralop.models import FNO
from torch import nn
from torch_geometric.nn import ChebConv, GATConv, GCNConv, GINConv, SAGEConv
from torch_geometric.nn.conv.message_passing import HookDict
from torchdiffeq import odeint

torch.backends.cudnn.deterministic = True

# -------------------------------- helper --------------------------------


def get_layer_param(param, i):  # in case param is a list
    return param[i] if isinstance(param, list) else param


# ------------------------ primary architectures -------------------------


class MLP(nn.Module):
    """Multi-layer perceptron (fully connected feedforward network).

    Each linear map is wrapped with optional ``pre_modules`` (applied before it)
    and ``post_modules`` (applied after it). Normalization, activation, dropout,
    and any other layer are passed in as plain modules through these two slots,
    so the network never needs to know what kind of module it is handling and
    any ordering can be expressed (e.g. moving normalization + activation into
    ``pre_modules`` gives a pre-activation block).

    Suggested ordering:
        pre_modules[i]   normalization -> activation   (pre-activation / pre-norm
                         blocks only; usually omitted)
        Linear(layers[i], layers[i + 1])
        post_modules[i]  normalization -> activation -> dropout   (the common
                         post-activation block; activations live here by default)

    Args:
        layers: Sizes of each layer. Example: [784, 256, 10] builds
            Linear(784, 256) then Linear(256, 10).
        post_modules: Per-layer modules inserted after each linear map. Entry i
            is None (skip), a single nn.Module, or a list of modules applied in
            order. Lists shorter than the number of layers are padded with None.
            Activations belong here unless a pre-activation block is wanted.
        pre_modules: Per-layer modules inserted before each linear map, same
            element format as post_modules. Typically left empty.
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
    """Deep convolutional network supporting 1D, 2D, or 3D convolutions.

    Each convolution is wrapped with optional ``pre_modules`` (applied before
    it) and ``post_modules`` (applied after it); see ``MLP`` for the slot
    mechanism. Resampling, normalization, and activation are passed in as plain
    modules, so any ordering can be expressed (e.g. normalize before the conv by
    putting the norm in ``pre_modules``).

    Suggested ordering:
        pre_modules[i]   resampling (Upsample / pooling); or normalization ->
                         activation for a pre-activation block
        Conv(channels[i], channels[i + 1])
        post_modules[i]  normalization -> activation -> dropout   (the common
                         post-activation block; activations live here by default)

    Args:
        channels: Channel sizes per layer. Example: [3, 64, 128] builds two
            convs (3->64, 64->128).
        post_modules: Per-layer modules inserted after each conv. Entry i is
            None (skip), a single nn.Module, or a list applied in order. Lists
            shorter than the number of layers are padded with None. Activations
            belong here by default.
        kernel_size, stride, padding, dilation: Conv geometry; a scalar applies
            to every layer, or pass a per-layer list.
        pre_modules: Per-layer modules inserted before each conv, same element
            format as post_modules. Resampling goes here.
        dim: Spatial dimensionality (1, 2, or 3).
        bias: Whether convs carry a bias (scalar or per-layer list).
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
    """E(2)-steerable convolutional network (escnn), the equivariant ``DCN``.

    Builds a stack of steerable convolutions over the symmetry group carried by
    ``gspace`` (e.g. ``gspaces.rot2dOnR2(N=8)`` for discrete C8 rotations). Output
    feature maps transform consistently when the input is rotated, so a single
    training sample teaches the whole orbit of rotated inputs.

    Channels are given as field copies per layer, like ``DCN`` channels: the
    interior layers carry regular-representation fields, while the input and output
    representations are ``in_repr`` / ``out_repr`` (both scalar by default, the
    standard scalar-in / scalar-out arrangement). Pass ``gspace.irrep(1)`` for a 2D
    vector field, so e.g. a scalar-in / vector-out network learns an equivariant
    operator like the gradient.

    Unlike ``DCN``, the activation is passed as a factory rather than a plain
    module, because a steerable nonlinearity must know the field type it acts on;
    ``activation`` is called once per hidden layer as ``activation(field_type)``.

    The forward pass takes and returns plain tensors (shape (B, C, H, W)); the
    escnn ``GeometricTensor`` wrapping is handled internally, so the model is used
    like any other ``nn.Module``.

    Args:
        gspace: escnn GSpace defining the symmetry group acting on R^2.
        channels: field copies per layer. Endpoints carry in_repr / out_repr,
            interior layers are regular-representation fields. Example: [1, 8, 8, 1].
        activation: factory mapping a FieldType to an equivariant activation
            module, applied after every hidden conv (e.g. ``enn.LeakyReLU``).
        kernel_size, padding: conv geometry; a scalar applies to every layer, or
            pass a per-layer list.
        bias: whether convs carry a bias (scalar or per-layer list).
        in_repr, out_repr: input/output representations; default to the scalar
            (trivial) representation. Pass ``gspace.irrep(1)`` for a vector field.
    """

    def __init__(
        self,
        gspace,
        channels: list[int],
        activation=enn.LeakyReLU,
        kernel_size: int | list[int] = 3,
        padding: int | list[int] = 0,
        bias: bool | list[bool] = False,
        in_repr=None,
        out_repr=None,
    ) -> None:
        super().__init__()
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
    """Deep graph convolutional network using GCNConv layers.

    Reference: https://arxiv.org/abs/1609.02907

    Args:
        channels: List of channel sizes for each layer.
            Example: [16, 32, 64] creates two GCN layers (16->32, 32->64).
        activations: List of activation modules after each conv layer.
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
    """Deep graph network using Chebyshev spectral convolution layers.

    Reference: https://arxiv.org/abs/1606.09375

    Args:
        channels: List of channel sizes for each layer.
            Example: [16, 32, 64] creates two ChebConv layers (16->32, 32->64).
        activations: List of activation modules after each conv layer.
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
    """Deep GraphSAGE network using SAGEConv layers.

    Reference: https://arxiv.org/abs/1706.02216

    Args:
        channels: List of channel sizes for each layer.
            Example: [16, 32, 64] creates two SAGE layers (16->32, 32->64).
        activations: List of activation modules after each conv layer.
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
    """Deep graph attention network using GATConv layers.

    References:
        - https://arxiv.org/abs/1710.10903
        - https://arxiv.org/abs/2105.14491

    Args:
        channels: List of channel sizes for each layer.
            Example: [16, 32, 64] creates two GAT layers (16->32, 32->64).
        activations: List of activation modules after each conv layer.
        heads: Number of attention heads per layer.
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
    """Deep graph isomorphism network using GINConv layers.

    Reference: https://arxiv.org/abs/1810.00826

    Args:
        mlp_layers: List of layer configurations for each GIN layer's MLP.
            Example: [[16, 32], [32, 64]] creates two GIN layers with MLPs.
        mlp_activations: List of activation lists for each GIN layer's MLP.
        eps: Initial epsilon value for weighting self-loops.
        train_eps: Whether to make epsilon a learnable parameter.
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


# https://arxiv.org/abs/1612.00222
# TODO

# ------------------------------ sequential ------------------------------


class DRNN(nn.Module):
    """Deep recurrent neural network with configurable cell type.

    Supports RNN, LSTM, and GRU cells with optional normalization layers
    and a final linear projection.

    References:
        - https://ieeexplore.ieee.org/abstract/document/6795963
        - https://arxiv.org/abs/1412.3555

    Args:
        layers: List of layer sizes. The last two values define the projection
            (layers[-2] -> layers[-1]). Example: [64, 128, 256, 10] creates
            two recurrent layers (64->128, 128->256) and a projection (256->10).
        final_activation: Optional activation after the final projection.
        cell: Recurrent cell type (nn.RNN, nn.LSTM, or nn.GRU).
        normalizations: List of normalization modules after each recurrent layer.
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
    """Neural ordinary differential equation.

    Learns continuous-depth dynamics by parameterizing the derivative dh/dt
    with a neural network and integrating using an ODE solver.

    Reference: https://arxiv.org/abs/1806.07366

    Args:
        rhs_model: Neural network that computes dh/dt. Should accept input of
            shape (batch, hidden_dim + 1) where the +1 is for concatenated time,
            and output shape (batch, hidden_dim).
    """

    def __init__(self, rhs_model: nn.Module) -> None:
        super().__init__()
        self.rhs_model = rhs_model

    def eval_rhs(self, t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Compute dh/dt by concatenating time to state and passing through the model."""
        t_vec = torch.ones(h.shape[0], 1, device=h.device) * t
        x = torch.cat([h, t_vec], dim=1)
        return self.rhs_model(x)

    def forward(self, T: torch.Tensor, h0: torch.Tensor) -> torch.Tensor:
        """Integrate the ODE from initial state h0 over time points T.

        Args:
            T: Time points at which to evaluate the solution.
            h0: Initial hidden state of shape (batch, hidden_dim).

        Returns:
            Solution at each time point, shape (len(T), batch, hidden_dim).
        """
        return odeint(self.eval_rhs, h0, T)


# ------------------------------- Bayesian -------------------------------


class BayesianLinear(nn.Module):
    """Bayesian linear layer with learned weight distributions.

    Implements weight uncertainty using variational inference (Bayes by Backprop).
    Weights are sampled from Gaussian distributions parameterized by mu and rho,
    where sigma = softplus(rho).

    Reference: https://arxiv.org/abs/1505.05424

    Args:
        inputs: Number of input features.
        outputs: Number of output features.
    """

    def __init__(self, inputs: int, outputs: int) -> None:
        super().__init__()
        self.weight_mu = nn.Parameter(torch.Tensor(outputs, inputs))
        self.weight_rho = nn.Parameter(torch.Tensor(outputs, inputs))
        self.bias_mu = nn.Parameter(torch.Tensor(outputs))
        self.bias_rho = nn.Parameter(torch.Tensor(outputs))

        self.init_params()

    def init_params(self) -> None:
        """Initialize parameters with Kaiming uniform for mu and constant for rho."""
        nn.init.kaiming_uniform_(self.weight_mu, a=math.sqrt(5.0))
        nn.init.constant_(self.weight_rho, -3.0)
        nn.init.constant_(self.bias_mu, 0.0)
        nn.init.constant_(self.bias_rho, -3.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Sample weights from learned distributions and apply linear transformation."""
        weight_sigma = F.softplus(self.weight_rho)
        bias_sigma = F.softplus(self.bias_rho)

        weight_epsilon = torch.randn_like(self.weight_mu)
        bias_epsilon = torch.randn_like(self.bias_mu)

        weight = self.weight_mu + weight_sigma * weight_epsilon
        bias = self.bias_mu + bias_sigma * bias_epsilon

        return F.linear(x, weight, bias)

    def kl_divergence(self, prior_std: float) -> torch.Tensor:
        """Compute KL divergence between weight distributions and Gaussian prior."""
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
    """Bayesian multi-layer perceptron using BayesianLinear layers.

    Each BayesianLinear map is wrapped with optional ``pre_modules`` /
    ``post_modules``; see ``MLP`` for the slot mechanism.

    Reference: https://arxiv.org/abs/1505.05424

    Args:
        layers: Sizes of each layer. Example: [784, 256, 10] builds two
            Bayesian linear layers.
        post_modules: Per-layer modules inserted after each layer (None | Module
            | list). Activations belong here by default.
        pre_modules: Per-layer modules inserted before each layer.
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
        """Compute total KL divergence across all BayesianLinear layers."""
        kl = 0
        for module in self.model.modules():
            if isinstance(module, BayesianLinear):
                kl += module.kl_divergence(prior_std)
        return kl


# ------------------------------- resnets --------------------------------


class ResidualBlock(nn.Module):
    """Residual block that adds input to module output (skip connection).

    Args:
        module: The transformation to apply before adding the residual.
        projection: Optional projection to match dimensions when input and
            output shapes differ (e.g., a 1x1 conv or linear layer).
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


class ResNet(nn.Module):
    """Converts a base model into a residual network by adding skip connections.

    Takes an existing sequential model and wraps specified layer ranges into
    ResidualBlocks with skip connections.

    Args:
        base_model: Model with a `.model` attribute containing nn.Sequential layers.
        skip_connections: List of (start_idx, end_idx) tuples specifying which
            layer ranges to wrap with skip connections.
            Example: [(0, 2), (3, 5)] creates residual blocks from layers 0-2 and 3-5.
        projections: Optional dict mapping (start_idx, end_idx) to projection modules
            for dimension matching in skip connections.
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


# --------------------- input convex networks ----------------------------


class ICNNLayer(nn.Module):
    """Single fully input-convex layer (Amos et al. 2017):

        z_out = W^z z + W^y x + b

    Stack several of these with convex non-decreasing activations
    (ReLU, ELU, Softplus, LeakyReLU with slope in [0, 1]) and call
    `clamp_z_()` after each optimizer step to keep W^z non-negative;
    the resulting network is then convex in x.

    Args:
        z_in: size of the z-path input. Set to 0 for the first layer
            (which has no z and reduces to W^y x + b).
        x_in: size of the original network input x.
        out:  layer output size.
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
    """Input convex neural network (Amos et al. 2017).

    Stack of `ICNNLayer`s threading the original input x through every
    layer. With convex non-decreasing activations (ReLU, ELU, Softplus,
    LeakyReLU with slope in [0, 1]) and W^z weights kept non-negative via
    `clamp_z_()` after each optimizer step, the forward map is convex in x.

    Args:
        layers: List of layer sizes; layers[0] is the input dim, layers[-1]
            the output dim.
        activations: List of activation modules to apply after each layer.
            Length should be len(layers) - 1 or fewer; use None for no
            activation at a given position. The final layer typically has
            no activation.
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


# ---------------------- extreme learning machines -----------------------
class ELM(nn.Module):
    """Extreme learning machine with random fixed feature extractor.

    Combines a frozen random feature extractor with a linear output layer
    trained via closed-form least squares solution. The feature extractor
    weights are never updated; only the output layer is fitted analytically.
    A subsequent training of all weights is optional.

    Reference: https://ieeexplore.ieee.org/document/1380068

    Args:
        feature_extractor: Neural network that maps input to hidden features.
            Weights are considered frozen during fitting.
        hidden_dim: Dimensionality of the hidden feature space (output of
            feature_extractor).
        output_dim: Number of output classes or regression targets.
    """

    def __init__(self, feature_extractor: nn.Module, hidden_dim: int, output_dim: int):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.output_layer = nn.Linear(hidden_dim, output_dim, bias=True)
        self.output_layer_weights = nn.Parameter(torch.zeros(output_dim, hidden_dim))
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

    def fit(self, x: torch.Tensor, y: torch.Tensor, regularization: float):
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
    c, s = torch.cos(alpha), torch.sin(alpha)
    return torch.stack(  # to not break autograd
        [
            torch.stack([c**2, s**2, 2 * c * s]),
            torch.stack([s**2, c**2, -2 * c * s]),
            torch.stack([-c * s, c * s, c**2 - s**2]),
        ]
    )


class LaminateBlock(nn.Module):
    """Two-layer laminate building block of a deep material network."""

    def __init__(self):
        super().__init__()
        # random volume fraction breaks the symmetry of an all-equal init; otherwise every
        # block stays identical and the fit stalls (alpha stays 0: a large random rotation
        # drives the homogenized stiffness non-physical)
        self.v1_logit = nn.Parameter(0.5 * torch.randn(()))  # with sigmoid 0<=v1<=1
        self.alpha = nn.Parameter(torch.zeros(()))

    @property
    def v1(self):
        return torch.sigmoid(self.v1_logit)

    def homogenize_stiffness(self, C1, C2):
        # C1, C2 are batched stiffnesses (N, 3, 3); homogenizes over the leading dimension
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
    """Binary-tree deep (composite) material network with `depth` laminate layers."""

    def __init__(self, depth):
        super().__init__()
        self.depth = depth
        self.layers = nn.ModuleList(
            [
                nn.ModuleList([LaminateBlock() for _ in range(2 ** (depth - l - 1))])
                for l in range(depth)
            ]
        )
        self._all_cache = None  # populated by forward, consumed by homogenize_stress

    def forward(self, C1, C2, deps):
        # linear two-phase cell: leaves alternate phase 1 / phase 2
        leaf_C = [C1 if i % 2 == 0 else C2 for i in range(2**self.depth)]
        return self.homogenize(leaf_C, deps)

    def homogenize(self, leaf_C, deps):
        # bottom-up stiffness then top-down strains for a tree of per-leaf stiffnesses
        # leaf_C (list of 2**depth batched (N, 3, 3)); deps macro increment (N, 3). This
        # generalizes forward: each leaf may carry its own (e.g. nonlinear tangent) stiffness
        L = self.depth

        # bottom-up: homogenize stiffness, leaves to root
        all_C, all_cache = [None] * L, [None] * L
        C_leaves, cache_leaves = zip(
            *(
                block.homogenize_stiffness(leaf_C[2 * i], leaf_C[2 * i + 1])
                for i, block in enumerate(self.layers[0])
            )
        )
        all_C[0], all_cache[0] = list(C_leaves), list(cache_leaves)
        for l in range(1, L):
            prev_C = all_C[l - 1]
            C_leaves, cache_leaves = zip(
                *(
                    b.homogenize_stiffness(prev_C[2 * i], prev_C[2 * i + 1])
                    for i, b in enumerate(self.layers[l])
                )
            )
            all_C[l], all_cache[l] = list(C_leaves), list(cache_leaves)
        C_root = all_C[L - 1][0]

        # top-down: recover strains, root to leaves
        # walk down the tree just above the leaves (strains at internal layer)
        cur_deps = [deps]
        for l in range(L - 1, 0, -1):
            next_deps = []
            for block, cache, deps in zip(self.layers[l], all_cache[l], cur_deps):
                deps1, deps2 = block.recover_strains(deps, cache)
                next_deps.extend([deps1, deps2])
            cur_deps = next_deps

        # get per-phase strains
        leaf_deps = []
        for block, cache, deps in zip(self.layers[0], all_cache[0], cur_deps):
            leaf_deps.extend(block.recover_strains(deps, cache))

        self._all_cache = all_cache  # kept for homogenize_stress
        return C_root, leaf_deps

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


# ----------------------------- autoencoders -----------------------------


class AE(nn.Module):
    """Autoencoder composed of an encoder and decoder network.

    Args:
        Encoder: Network that maps input to latent representation.
        Decoder: Network that reconstructs input from latent representation.
    """

    def __init__(self, encoder: nn.Module, decoder: nn.Module) -> None:
        super().__init__()
        self.encode = encoder
        self.decode = decoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encode(x)
        x = self.decode(x)
        return x


class VAE(AE):
    """Variational autoencoder with reparameterization trick.

    The encoder must output 2 * latent_dim features (mean and log-variance).
    The decoder takes latent_dim features as input.

    Args:
        Encoder: Network mapping input to (mean, logvar) concatenated.
        Decoder: Network reconstructing input from sampled latent vector.
    """

    def reparameterize(
        self,
        mean: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        """Sample from latent distribution using reparameterization trick."""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mean + eps * std
        return mean

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode, sample, and decode. Returns (reconstruction, mean, logvar)."""
        distributions = self.encode(x)
        mean, logvar = torch.chunk(distributions, chunks=2, dim=1)
        z = self.reparameterize(mean, logvar)
        y = self.decode(z)
        return y, mean, logvar


class UNet(nn.Module):
    """U-Net: symmetric encoder-decoder with skip connections at each level.

    Each module in `downs` produces a feature map that is stored as a skip
    connection. The `bottleneck` operates at the deepest resolution. Each
    module in `ups` receives the previous output concatenated with the
    matching skip along the channel axis (in reverse order), so each up
    module must accept (input + skip) channels.

    Args:
        downs: Encoder modules, ordered shallow-to-deep. Each is expected
            to downsample its input.
        ups: Decoder modules, ordered deep-to-shallow. Same length as
            `downs`. Each is expected to upsample its input.
        bottleneck: Module applied at the deepest resolution between
            encoder and decoder. Defaults to identity.
    """

    def __init__(
        self,
        downs: list[nn.Module],
        ups: list[nn.Module],
        bottleneck: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.downs = nn.ModuleList(downs)
        self.ups = nn.ModuleList(ups)
        self.bottleneck = bottleneck or nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
        x = self.bottleneck(x)
        for up, skip in zip(self.ups, reversed(skips)):
            x = up(torch.cat([x, skip], dim=1))
        return x


# --------------------------- neural operators ---------------------------
# https://arxiv.org/abs/1910.03193
class DeepONet(nn.Module):
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


# FNO defined via efficient_kan
# https://arxiv.org/abs/2010.08895

# ---------------------------- siren network -----------------------------


class SIRENsine(nn.Module):
    def __init__(self, omega_0: float = 30.0) -> None:
        super().__init__()
        self.omega_0 = omega_0

    def forward(self, x):
        return torch.sin(self.omega_0 * x)


# class SIREN(nn.Module):
#     """Sinusoidal representation network with periodic activations.

#     Each hidden layer computes sin(omega_0 * (Wx + b)). The output layer is
#     linear (no sine), suitable for regression. Weights are initialized to
#     preserve the distribution of activations across depth.

#     Reference: https://arxiv.org/abs/2006.09661

#     Args:
#         layers: List of integers specifying the size of each layer.
#         omega_0: Frequency multiplier applied before the sine in hidden layers.
#             The first layer scales by omega_0; subsequent layers are initialized
#             so that omega_0 cancels in the variance calculation.
#     """

#     def __init__(self, layers: list[int], omega_0: float = 30.0) -> None:
#         super().__init__()
#         linears = []
#         for i in range(len(layers) - 1):
#             linear = nn.Linear(layers[i], layers[i + 1])
#             if i == 0:
#                 bound = 1.0 / layers[i]
#             else:
#                 bound = math.sqrt(6.0 / layers[i]) / omega_0
#             nn.init.uniform_(linear.weight, -bound, bound)
#             nn.init.uniform_(linear.bias, -bound, bound)
#             linears.append(linear)
#         self.linears = nn.ModuleList(linears)
#         self.omega_0 = omega_0

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         for linear in self.linears[:-1]:
#             x = torch.sin(self.omega_0 * linear(x))
#         return self.linears[-1](x)


# ---------------------- kolmogorov-arnold network -----------------------
# defined via efficient_kan
# https://arxiv.org/abs/2404.19756
