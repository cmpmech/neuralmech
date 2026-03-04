import torch
from torch import nn
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GINConv
from efficient_kan import KAN
from torchdiffeq import odeint
import torch.nn.functional as F
import math

# ------------------------ primary architectures -------------------------

class MLP(nn.Module):
    """Multi-layer perceptron (fully connected feedforward network).

    Args:
        layers: List of integers specifying the size of each layer.
            Example: [784, 256, 128, 10] creates a network with input dim 784,
            two hidden layers of size 256 and 128, and output dim 10.
        activations: List of activation modules to apply after each linear layer.
            Length should be len(layers) - 1 or fewer. Use None for no activation.
            Example: [nn.ReLU(), nn.ReLU(), None] applies ReLU after first two
            layers and no activation after the final layer.
        normalizations: List of normalization modules to apply before each
            activation.
    """

    def __init__(self, layers: list[int], activations: list[nn.Module | None] | None = None,
                 normalizations: list[nn.Module | None] | None = None) -> None:
        super().__init__()
        modules = []
        for i in range(len(layers) - 1):
            modules.append(nn.Linear(layers[i], layers[i + 1]))
            if normalizations and i < len(normalizations):
                if normalizations[i]:
                    modules.append(normalizations[i])
            if activations and i < len(activations):
                if activations[i]:
                    modules.append(activations[i])
        self.model = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

class DCN(nn.Module):
    """Deep convolutional network supporting 1D, 2D, or 3D convolutions.

    Args:
        channels: List of channel sizes for each layer.
            Example: [3, 64, 128, 256] for RGB input with 3 conv layers.
        activations: List of activation modules after each conv layer.
            Use None for no activation at a given position.
        kernel_size: Kernel size for all convolutional layers.
        stride: Stride for all convolutional layers.
        padding: Padding for all convolutional layers.
        normalizations: List of normalization modules (e.g., BatchNorm) after conv.
        resamplings: List of resampling modules (e.g., MaxPool, Upsample) before conv.
        dilation: Dilation factor for convolutions.
        dim: Spatial dimensionality (1, 2, or 3).
        bias: Whether to include bias in convolutional layers.
    """

    def __init__(
        self,
        channels: list[int],
        activations: list[nn.Module | None],
        kernel_size: int,
        stride: int,
        padding: int,
        normalizations: list[nn.Module | None] | None = None,
        resamplings: list[nn.Module | None] | None = None,
        dilation: int = 1,
        dim: int = 2,
        bias: bool = False,
    ) -> None:
        super().__init__()
        normalizations = normalizations or []
        resamplings = resamplings or []
        conv = {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}[dim]
        modules = []
        for i in range(len(channels) - 1):
            if resamplings and i < len(resamplings):
                if resamplings[i]:
                    modules.append(resamplings[i])
            modules.append(conv(channels[i], channels[i + 1],
                                     kernel_size, stride, padding,
                                     dilation, bias=bias))
            if normalizations and i < len(normalizations):
                if normalizations[i]:
                    modules.append(normalizations[i])
            if activations and i < len(activations):
                if activations[i]:
                    modules.append(activations[i])
        self.model = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

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
            mlp = MLP(mlp_layers[i], mlp_activations[i])
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
        nn.init.kaiming_uniform_(self.weight_mu, a=math.sqrt(5.))
        nn.init.constant_(self.weight_rho, -3.)
        nn.init.constant_(self.bias_mu, 0.)
        nn.init.constant_(self.bias_rho, -3.)

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

        kl_weight = (torch.log(prior_std / weight_sigma) +
                     (weight_sigma**2 + self.weight_mu**2) /
                     (2 * prior_std**2) - 0.5)
        kl_bias = (torch.log(prior_std / bias_sigma) +
                   (bias_sigma**2 + self.bias_mu**2) /
                   (2 * prior_std**2) - 0.5)
        return kl_weight.sum() + kl_bias.sum()

class BayesianMLP(nn.Module):
    """Bayesian multi-layer perceptron using BayesianLinear layers.

    Reference: https://arxiv.org/abs/1505.05424

    Args:
        layers: List of integers specifying the size of each layer.
            Example: [784, 256, 10] creates two Bayesian linear layers.
        activations: List of activation modules to apply after each layer.
    """

    def __init__(
        self,
        layers: list[int],
        activations: list[nn.Module | None] | None = None,
    ) -> None:
        super().__init__()
        modules = []
        for i in range(len(layers) - 1):
            modules.append(BayesianLinear(layers[i], layers[i + 1]))
            if activations and i < len(activations):
                if activations[i]:
                    modules.append(activations[i])
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
                block_layers = layers[i:end_idx + 1]
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

    def __init__(self, feature_extractor: nn.Module, hidden_dim: int,
                 output_dim: int):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.output_layer = nn.Linear(hidden_dim, output_dim, bias=True)
        self.output_layer_weights = nn.Parameter(torch.zeros(output_dim,
                                                             hidden_dim))
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

    def fit(self, x: torch.Tensor, y: torch.Tensor,
            regularization: float):
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

# ----------------------------- autoencoders -----------------------------

class AE(nn.Module):
    """Autoencoder composed of an encoder and decoder network.

    Args:
        Encoder: Network that maps input to latent representation.
        Decoder: Network that reconstructs input from latent representation.
    """

    def __init__(self, Encoder: nn.Module, Decoder: nn.Module) -> None:
        super().__init__()
        self.encode = Encoder
        self.decode = Decoder

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

# ---------------------- kolmogorov-arnold network -----------------------
# defined via efficient_kan
# https://arxiv.org/abs/2404.19756