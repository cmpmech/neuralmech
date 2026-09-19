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


# ------------------------------- primary architectures -------------------------------


class MLP(nn.Module):
    """Multi-layer perceptron (fully connected feedforward network).

    Each linear map is wrapped with optional ``pre_modules`` (applied before it) and
    ``post_modules`` (applied after it). Normalization, activation, dropout, and any
    other layer are passed in as plain modules through these two slots, so the network
    never needs to know what kind of module it is handling and any ordering can be
    expressed (e.g. moving normalization + activation into ``pre_modules`` gives a
    pre-activation block).

    Suggested ordering:
        pre_modules[i]   normalization -> activation   (pre-activation / pre-norm
                         blocks only; usually omitted)
        Linear(layers[i], layers[i + 1])
        post_modules[i]  normalization -> activation -> dropout   (the common
                         post-activation block; activations live here by default)

    Args:
        layers: Sizes of each layer. Example: [784, 256, 10] builds Linear(784, 256)
            then Linear(256, 10).
        post_modules: Per-layer modules inserted after each linear map. Entry i is None
            (skip), a single nn.Module, or a list of modules applied in order. Lists
            shorter than the number of layers are padded with None. Activations belong
            here unless a pre-activation block is wanted.
        pre_modules: Per-layer modules inserted before each linear map, same element
            format as post_modules. Typically left empty.
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

    Each convolution is wrapped with optional ``pre_modules`` (applied before it) and
    ``post_modules`` (applied after it); see ``MLP`` for the slot mechanism.
    Resampling, normalization, and activation are passed in as plain modules, so any
    ordering can be expressed (e.g. normalize before the conv by putting the norm in
    ``pre_modules``).

    Suggested ordering:
        pre_modules[i]   resampling (Upsample / pooling); or normalization ->
                         activation for a pre-activation block
        Conv(channels[i], channels[i + 1])
        post_modules[i]  normalization -> activation -> dropout   (the common
                         post-activation block; activations live here by default)

    Args:
        channels: Channel sizes per layer. Example: [3, 64, 128] builds two convs
            (3->64, 64->128).
        post_modules: Per-layer modules inserted after each conv. Entry i is None
            (skip), a single nn.Module, or a list applied in order. Lists shorter than
            the number of layers are padded with None. Activations belong here by
            default.
        kernel_size, stride, padding, dilation: Conv geometry; a scalar applies to
            every layer, or pass a per-layer list.
        pre_modules: Per-layer modules inserted before each conv, same element format
            as post_modules. Resampling goes here.
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
    feature maps transform consistently when the input is rotated, so a single training
    sample teaches the whole orbit of rotated inputs.

    Channels are given as field copies per layer, like ``DCN`` channels: the interior
    layers carry regular-representation fields, while the input and output
    representations are ``in_repr`` / ``out_repr`` (both scalar by default, the
    standard scalar-in / scalar-out arrangement). Pass ``gspace.irrep(1)`` for a 2D
    vector field, so e.g. a scalar-in / vector-out network learns an equivariant
    operator like the gradient.

    Unlike ``DCN``, the activation is passed as a factory rather than a plain module,
    because a steerable nonlinearity must know the field type it acts on;
    ``activation`` is called once per hidden layer as ``activation(field_type)``.

    The forward pass takes and returns plain tensors (shape (B, C, H, W)); the escnn
    ``GeometricTensor`` wrapping is handled internally, so the model is used like any
    other ``nn.Module``.

    Args:
        gspace: escnn GSpace defining the symmetry group acting on R^2.
        channels: field copies per layer. Endpoints carry in_repr / out_repr, interior
            layers are regular-representation fields. Example: [1, 8, 8, 1].
        activation: factory mapping a FieldType to an equivariant activation module,
            applied after every hidden conv. Defaults to ``enn.LeakyReLU``.
        kernel_size, padding: conv geometry; a scalar applies to every layer, or pass a
            per-layer list.
        bias: whether convs carry a bias (scalar or per-layer list).
        in_repr, out_repr: input/output representations; default to the scalar
            (trivial) representation. Pass ``gspace.irrep(1)`` for a vector field.
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
    """Deep graph convolutional network using GCNConv layers.

    Reference: https://arxiv.org/abs/1609.02907

    Args:
        channels: List of channel sizes for each layer. Example: [16, 32, 64] creates
            two GCN layers (16->32, 32->64).
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
        channels: List of channel sizes for each layer. Example: [16, 32, 64] creates
            two ChebConv layers (16->32, 32->64).
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
        channels: List of channel sizes for each layer. Example: [16, 32, 64] creates
            two SAGE layers (16->32, 32->64).
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
        channels: List of channel sizes for each layer. Example: [16, 32, 64] creates
            two GAT layers (16->32, 32->64).
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
        mlp_layers: List of layer configurations for each GIN layer's MLP. Example:
            [[16, 32], [32, 64]] creates two GIN layers with MLPs.
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


# ------------------------------------- sequential ------------------------------------


class DRNN(nn.Module):
    """Deep recurrent neural network with configurable cell type.

    Supports RNN, LSTM, and GRU cells with optional normalization layers and a final
    linear projection.

    References:
        - https://ieeexplore.ieee.org/abstract/document/6795963
        - https://arxiv.org/abs/1412.3555

    Args:
        layers: List of layer sizes. The last two values define the projection
            (layers[-2] -> layers[-1]). Example: [64, 128, 256, 10] creates two
            recurrent layers (64->128, 128->256) and a projection (256->10).
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

    Learns continuous-depth dynamics by parameterizing the derivative dh/dt with a
    neural network and integrating using an ODE solver.

    Reference: https://arxiv.org/abs/1806.07366

    Args:
        rhs_model: Neural network that computes dh/dt. Should accept input of shape
            (batch, hidden_dim + 1) where the +1 is for concatenated time, and output
            shape (batch, hidden_dim).
    """

    def __init__(self, rhs_model: nn.Module) -> None:
        super().__init__()
        self.rhs_model = rhs_model

    def eval_rhs(self, t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Compute dh/dt by concatenating time to the state and calling the model."""
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


# -------------------------------------- Bayesian -------------------------------------


class BayesianLinear(nn.Module):
    """Bayesian linear layer with learned weight distributions.

    Implements weight uncertainty using variational inference (Bayes by Backprop).
    Weights are sampled from Gaussian distributions parameterized by mu and rho, where
    sigma = softplus(rho).

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
        """Sample the weights from their distributions and apply the linear map."""
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
        layers: Sizes of each layer. Example: [784, 256, 10] builds two Bayesian linear
            layers.
        post_modules: Per-layer modules inserted after each layer (None | Module |
            list). Activations belong here by default.
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


# -------------------------------------- resnets --------------------------------------


class ResidualBlock(nn.Module):
    """Residual block that adds input to module output (skip connection).

    Args:
        module: The transformation to apply before adding the residual.
        projection: Optional projection to match dimensions when input and output
            shapes differ (e.g., a 1x1 conv or linear layer).
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
        skip_connections: List of (start_idx, end_idx) tuples specifying which layer
            ranges to wrap with skip connections. Example: [(0, 2), (3, 5)] creates
            residual blocks from layers 0-2 and 3-5.
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


# ------------------------------- input convex networks -------------------------------


class ICNNLayer(nn.Module):
    """Single fully input-convex layer (Amos et al. 2017):

        z_out = W^z z + W^y x + b

    Stack several of these with convex non-decreasing activations (ReLU, ELU, Softplus,
    LeakyReLU with slope in [0, 1]) and call `clamp_z_()` after each optimizer step to
    keep W^z non-negative; the resulting network is then convex in x.

    Args:
        z_in: size of the z-path input. Set to 0 for the first layer (which has no z
            and reduces to W^y x + b).
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

    Stack of `ICNNLayer`s threading the original input x through every layer. With
    convex non-decreasing activations (ReLU, ELU, Softplus, LeakyReLU with slope in
    [0, 1]) and W^z weights kept non-negative via `clamp_z_()` after each optimizer
    step, the forward map is convex in x.

    Args:
        layers: List of layer sizes; layers[0] is the input dim, layers[-1] the output
            dim.
        activations: List of activation modules to apply after each layer. Length
            should be len(layers) - 1 or fewer; use None for no activation at a given
            position. The final layer typically has no activation.
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
    """Extreme learning machine with random fixed feature extractor.

    Combines a frozen random feature extractor with a linear output layer trained via
    closed-form least squares solution. The feature extractor weights are never
    updated; only the output layer is fitted analytically. A subsequent training of all
    weights is optional.

    Reference: https://ieeexplore.ieee.org/document/1380068

    Args:
        feature_extractor: Neural network that maps input to hidden features. Weights
            are considered frozen during fitting.
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

    The encoder must output 2 * latent_dim features (mean and log-variance). The
    decoder takes latent_dim features as input.

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

    Each module in `downs` produces a feature map that is stored as a skip connection.
    The `bottleneck` operates at the deepest resolution. Each module in `ups` receives
    the previous output concatenated with the matching skip along the channel axis (in
    reverse order), so each up module must accept (input + skip) channels.

    Args:
        downs: Encoder modules, ordered shallow-to-deep. Each is expected to downsample
            its input.
        ups: Decoder modules, ordered deep-to-shallow. Same length as `downs`. Each is
            expected to upsample its input.
        bottleneck: Module applied at the deepest resolution between encoder and
            decoder. Defaults to identity.
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


# ---------------------------------- neural operators ---------------------------------
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

# ----------------------------------- siren network -----------------------------------


class SIRENsine(nn.Module):
    def __init__(self, omega_0: float = 30.0) -> None:
        super().__init__()
        self.omega_0 = omega_0

    def forward(self, x):
        return torch.sin(self.omega_0 * x)


# ----------------------------- kolmogorov-arnold network -----------------------------
# defined via efficient_kan
# https://arxiv.org/abs/2404.19756


# TODO remove
# # class SlotEncoder(nn.Module):
#     """Encoder head producing a grid of local latent slots plus a shared global vector.

#     ``body`` is any convolutional stack reducing the input to a coarse spatial grid. A
#     1x1 convolution then reads one posterior per grid cell, while a linear layer reads
#     a second posterior from the pooled features. Because a slot is anchored to a
#     location, the encoder never has to impose an ordering on the objects in the image,
#     which is what makes a flat code spend far more dimensions than the data has degrees
#     of freedom.

#     The output is laid out as ``cat([slot_mean, glob_mean, slot_logvar, glob_logvar])``
#     so that ``VAE`` splitting it in half recovers the mean and log-variance.

#     Args:
#         body: Convolutional stack mapping the input to (width, grid, grid).
#         width: Channel count leaving ``body``.
#         cell: Latent dimensions per grid slot.
#         glob: Latent dimensions in the shared global vector.
#     """

#     def __init__(self, body: nn.Module, width: int, cell: int, glob: int) -> None:
#         super().__init__()
#         self.body = body
#         self.slots = nn.Conv2d(width, 2 * cell, 1)
#         self.shared = nn.Linear(width, 2 * glob)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """Encode into stacked (mean, logvar) over slots followed by global dims."""
#         features = self.body(x)
#         slot_mean, slot_logvar = torch.chunk(self.slots(features), chunks=2, dim=1)
#         shared = self.shared(features.mean(dim=(2, 3)))
#         glob_mean, glob_logvar = torch.chunk(shared, chunks=2, dim=1)
#         return torch.cat(
#             [slot_mean.flatten(1), glob_mean, slot_logvar.flatten(1), glob_logvar],
#             dim=1,
#         )


# class SlotDecoder(nn.Module):
#     """Decoder counterpart to ``SlotEncoder``.

#     The leading ``grid * grid * cell`` entries of the code are reshaped back onto the
#     slot grid and the remaining entries are broadcast across every slot, so each place
#     decodes from its own code together with the one the whole image shares. ``head``
#     must accept ``cell + glob`` input channels.

#     Args:
#         head: Convolutional stack mapping (cell + glob, grid, grid) to the output.
#         grid: Slots per side.
#         cell: Latent dimensions per grid slot.
#     """

#     def __init__(self, head: nn.Module, grid: int, cell: int) -> None:
#         super().__init__()
#         self.head = head
#         self.grid = grid
#         self.cell = cell

#     def forward(self, z: torch.Tensor) -> torch.Tensor:
#         """Place the slots back on their grid and broadcast the shared vector."""
#         split = self.grid**2 * self.cell
#         slots = z[:, :split].view(-1, self.cell, self.grid, self.grid)
#         shared = z[:, split:, None, None].expand(-1, -1, self.grid, self.grid)
#         return self.head(torch.cat([slots, shared], dim=1))


# class SlotPrior(nn.Module):
#     """Autoregressive prior over the code of a ``SlotEncoder``.

#     A factorized ``N(0, I)`` prior treats every slot as independent, and no product of
#     independent slots can express a constraint that couples places. Fibers never
#     overlapping is exactly such a constraint, which is why a code drawn from
#     ``N(0, I)`` decodes into merged blobs however hard the rate is throttled. This
#     reads the code as one shared token followed by the slot tokens and models it with a
#     causal transformer, so a slot is drawn conditioned on the slots already placed.
#     Each conditional is a diagonal Gaussian mixture, which is what lets a slot stay
#     sharply bimodal between empty and occupied.

#     Training is plain maximum likelihood on ``log_prob`` of the encoded means. The same
#     ``log_prob`` is exact and differentiable in the code, so it also serves as the
#     penalty that holds a latent optimization on the manifold of real codes.

#     Args:
#         grid: Slots per side, as in ``SlotEncoder``.
#         cell: Latent dimensions per slot, which is also the token width.
#         glob: Dimensions of the shared vector, at most ``cell``.
#         components: Mixture components per conditional.
#         width: Transformer feature width.
#         layers: Transformer layers.
#         heads: Attention heads.
#         dropout: Dropout inside the transformer. The prior is fitted to as many codes
#             as there are images, so it overfits without it.
#     """

#     def __init__(
#         self,
#         grid: int,
#         cell: int,
#         glob: int,
#         components: int = 10,
#         width: int = 128,
#         layers: int = 4,
#         heads: int = 4,
#         dropout: float = 0.1,
#     ) -> None:
#         super().__init__()
#         self.grid = grid
#         self.cell = cell
#         self.glob = glob
#         self.components = components
#         tokens = grid**2 + 1  # the shared vector leads, then one token per slot
#         self.start = nn.Parameter(torch.zeros(1, 1, cell))
#         self.position = nn.Parameter(torch.randn(1, tokens, width) * 0.02)
#         self.project = nn.Linear(cell, width)
#         layer = nn.TransformerEncoderLayer(
#             width,
#             heads,
#             2 * width,
#             dropout=dropout,
#             batch_first=True,
#             norm_first=True,
#             activation="gelu",
#         )
#         self.body = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
#         self.head = nn.Linear(width, components * (1 + 2 * cell))

#     def tokenize(self, z: torch.Tensor) -> torch.Tensor:
#         """Split a code into the shared token followed by one token per slot."""
#         split = self.grid**2 * self.cell
#         slots = z[:, :split].view(-1, self.cell, self.grid**2).transpose(1, 2)
#         shared = nn.functional.pad(z[:, split:], (0, self.cell - self.glob))
#         return torch.cat([shared[:, None], slots], dim=1)

#     def detokenize(self, tokens: torch.Tensor) -> torch.Tensor:
#         """Inverse of ``tokenize``."""
#         slots = tokens[:, 1:].transpose(1, 2).flatten(1)
#         return torch.cat([slots, tokens[:, 0, : self.glob]], dim=1)

#     def predict(self, tokens: torch.Tensor) -> tuple[torch.Tensor, ...]:
#         """Mixture weights, means and log-variances of the token at each position."""
#         features = self.project(tokens) + self.position[:, : tokens.shape[1]]
#         mask = nn.Transformer.generate_square_subsequent_mask(
#             tokens.shape[1], device=tokens.device
#         )
#         features = self.body(features, mask=mask, is_causal=True)
#         parameters = self.head(features)
#         shape = (*parameters.shape[:2], self.components, self.cell)
#         weight = parameters[..., : self.components]
#         mean = parameters[..., self.components : self.components * (1 + self.cell)]
#         logvar = parameters[..., self.components * (1 + self.cell) :]
#         return weight, mean.view(shape), logvar.view(shape).clamp(-12, 6)

#     def log_prob(self, z: torch.Tensor) -> torch.Tensor:
#         """Exact log density of a code, differentiable in ``z``."""
#         tokens = self.tokenize(z)
#         start = self.start.expand(tokens.shape[0], -1, -1)
#         shifted = torch.cat([start, tokens[:, :-1]], dim=1)
#         weight, mean, logvar = self.predict(shifted)
#         # the shared token is padded up to the slot width, so score only the real
#         # dimensions of it and every dimension of the slots
#         gauss = -0.5 * (
#             (tokens[:, :, None] - mean) ** 2 / logvar.exp()
#             + logvar
#             + math.log(2 * math.pi)
#         )
#         gauss[:, 0, :, self.glob :] = 0.0
#         component = weight.log_softmax(-1) + gauss.sum(-1)
#         return torch.logsumexp(component, dim=-1).sum(dim=-1)

#     @torch.no_grad()
#     def sample(self, samples: int, temperature: float = 1.0) -> torch.Tensor:
#         """Draw codes one token at a time. Below 1 the temperature sharpens samples."""
#         device = self.start.device
#         tokens = self.start.expand(samples, -1, -1)
#         rows = torch.arange(samples, device=device)
#         for _ in range(self.grid**2 + 1):
#             weight, mean, logvar = self.predict(tokens)
#             weight = (weight[:, -1] / temperature).softmax(-1)
#             choice = torch.multinomial(weight, 1)[:, 0]
#             drawn = mean[:, -1][rows, choice]
#             spread = (0.5 * logvar[:, -1][rows, choice]).exp()
#             drawn = drawn + temperature * spread * torch.randn_like(drawn)
#             tokens = torch.cat([tokens, drawn[:, None]], dim=1)
#         return self.detokenize(tokens[:, 1:])


# class TwoStagePrior(nn.Module):
#     """Learned prior that is itself a variational autoencoder over the codes.

#     The two-stage construction of Dai and Wipf (2019): the first autoencoder is trained
#     for reconstruction alone, and a second, much smaller one is fitted to the codes it
#     produced. Sampling draws from the standard normal of the second stage and decodes
#     twice. Unlike ``SlotPrior`` this ignores the spatial layout of the code and holds
#     no attention, which is what makes it simple; the price is that it has to model the
#     whole code at once instead of one slot at a time, so it needs a code small enough
#     for that to be possible.

#     ``log_prob`` returns the evidence lower bound rather than an exact density, which
#     is all a penalty holding a latent optimization on the manifold needs. In evaluation
#     mode the bound is taken at the posterior mean, so the penalty is deterministic.

#     Args:
#         codes: Encoded means the prior is fitted to; only their scale is read here.
#         inner: Latent size of the second stage.
#         width: Hidden width of both second-stage networks.
#     """

#     def __init__(self, codes: torch.Tensor, inner: int = 64, width: int = 512) -> None:
#         super().__init__()
#         latent = codes.shape[1]
#         self.register_buffer("code_mean", codes.mean(dim=0))
#         self.register_buffer("code_std", codes.std(dim=0) + 1e-6)
#         self.inner = inner
#         self.model = VAE(
#             MLP([latent, width, width, 2 * inner], [nn.SiLU(), nn.SiLU(), None]),
#             MLP([inner, width, width, latent], [nn.SiLU(), nn.SiLU(), None]),
#         )
#         self.logvar = nn.Parameter(torch.zeros(1))  # spread of the decoded code

#     def log_prob(self, z: torch.Tensor) -> torch.Tensor:
#         """Evidence lower bound on the log density, differentiable in ``z``."""
#         u = (z - self.code_mean) / self.code_std
#         u_pred, mean, logvar = self.model(u)
#         spread = self.logvar.clamp(-10, 10)
#         recon = -0.5 * (
#             (u - u_pred) ** 2 / spread.exp() + spread + math.log(2 * math.pi)
#         )
#         kl = 0.5 * (logvar.clamp(-10, 10).exp() + mean**2 - logvar.clamp(-10, 10) - 1)
#         # the standardization is a change of variables, so its jacobian belongs here
#         return recon.sum(dim=1) - kl.sum(dim=1) - self.code_std.log().sum()

#     @torch.no_grad()
#     def sample(self, samples: int) -> torch.Tensor:
#         """Decode a draw from the standard normal of the second stage."""
#         seed = torch.randn(samples, self.inner, device=self.code_mean.device)
#         u = self.model.decode(seed)
#         u = u + (0.5 * self.logvar).exp() * torch.randn_like(u)
#         return u * self.code_std + self.code_mean
