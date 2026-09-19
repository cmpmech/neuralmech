"""Slot-latent VAE for fiber microstructures: shelved reference, not a driver.

Intention. A flat latent vector has to learn "a fiber at this position" separately for
every position, which is what a bottleneck of 450 training images cannot pay for. Here
the code is instead a GRID x GRID grid of local slots of CELL dimensions each, anchored to
a place in the image, plus one GLOBAL vector shared by the whole image (it carries the
factors that belong to the image as a whole, above all the fiber radius). A slot being
tied to a location means the encoder never has to invent an ordering of the fibers, and
the map from code to image stays translation equivariant. On a 128 grid the flat code
reached 0.89 intersection over union with 346k parameters, the slot code 0.98 with 31k.

The catch. A standard normal prior treats every slot as independent, and no product of
independent slots can say that two neighbouring places are never both filled. That is the
whole content of the fibers not overlapping, so a code drawn from N(0, I) decodes into
merged worms however hard the rate term is throttled. The fix is to fit a prior to the
codes the encoder actually produced, which is how every latent generative model of images
is built (vq-vae with a pixelcnn over the codes, latent diffusion over the codes of a
nearly unregularized autoencoder). Two priors were tried:
  - SlotPrior: a causal transformer over the slot tokens, each conditional a diagonal
    Gaussian mixture so a slot stays sharply bimodal between empty and occupied. Best
    samples (about 0.06 more roundness and 0.02 more reconstruction than the next one),
    but far from basic.
  - TwoStagePrior: a second, much smaller VAE fitted to the codes (Dai and Wipf 2019). No
    attention, no spatial structure, so the whole code has to be modelled at once, which
    forces a coarse 4 x 4 slot grid and costs reconstruction.
A fully convolutional VAE with a spatial latent (16 channels on a 32 x 32 grid) is the
same idea without the global vector and without a fitted prior: it reconstructs well and
its N(0, I) samples mean nothing.

Shelved because the book wants the basic flat VAE (fiber_vae_train.py); what survives
from here is the insight that one samples from the distribution the codes reached, not
from the one they were pushed towards.

Wiring sketch, with the conventions of fiber_vae_train.py:

    GRID, CELL, GLOBAL = 8, 4, 4
    LATENT = GRID**2 * CELL + GLOBAL
    DEPTH = round(math.log2(RESOLUTION // GRID))  # from the slot grid up to the image
    channels, strides = build_ae_cnn_config(DEPTH, CONV_LAYERS, CHANNEL_DIM, 2)
    channels[0] = 1
    body = DCN(channels, [[nn.GroupNorm(1, c), act()] for c in channels[1:]],
               KERNEL_SIZE, stride=strides, padding=KERNEL_SIZE // 2, dim=2)
    Encoder = SlotEncoder(body, channels[-1], CELL, GLOBAL)
    decoder_channels = channels[::-1]
    decoder_channels[0] = CELL + GLOBAL  # the code enters as channels, not as a vector
    head = DCN(decoder_channels,
               [[nn.GroupNorm(1, c), act()] for c in decoder_channels[1:-1]],  # raw logits
               KERNEL_SIZE, stride=1, padding=KERNEL_SIZE // 2, pre_modules=upsamplings,
               dim=2)
    Decoder = SlotDecoder(head, GRID, CELL)
    model = VAE(Encoder, Decoder)
    # ... train with BCEWithLogitsLoss(reduction="sum") + KL, BETA = 1, then
    codes = torch.chunk(model.encode(standardizex(X_train)), chunks=2, dim=1)[0]
    prior = SlotPrior(GRID, CELL, GLOBAL)  # or TwoStagePrior(codes, 64)
    # ... fit by maximum likelihood: cost = -prior.log_prob(batch).mean()
    x_sample = model.decode(prior.sample(16))
"""

import math

import torch
from torch import nn

from NN import MLP, VAE


class SlotEncoder(nn.Module):
    """Encoder head producing a grid of local latent slots plus a shared global vector.

    ``body`` is any convolutional stack reducing the input to a coarse spatial grid. A
    1x1 convolution then reads one posterior per grid cell, while a linear layer reads
    a second posterior from the pooled features. Because a slot is anchored to a
    location, the encoder never has to impose an ordering on the objects in the image,
    which is what makes a flat code spend far more dimensions than the data has degrees
    of freedom.

    The output is laid out as ``cat([slot_mean, glob_mean, slot_logvar, glob_logvar])``
    so that ``VAE`` splitting it in half recovers the mean and log-variance.

    Args:
        body: Convolutional stack mapping the input to (width, grid, grid).
        width: Channel count leaving ``body``.
        cell: Latent dimensions per grid slot.
        glob: Latent dimensions in the shared global vector.
    """

    def __init__(self, body: nn.Module, width: int, cell: int, glob: int) -> None:
        super().__init__()
        self.body = body
        self.slots = nn.Conv2d(width, 2 * cell, 1)
        self.shared = nn.Linear(width, 2 * glob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode into stacked (mean, logvar) over slots followed by global dims."""
        features = self.body(x)
        slot_mean, slot_logvar = torch.chunk(self.slots(features), chunks=2, dim=1)
        shared = self.shared(features.mean(dim=(2, 3)))
        glob_mean, glob_logvar = torch.chunk(shared, chunks=2, dim=1)
        return torch.cat(
            [slot_mean.flatten(1), glob_mean, slot_logvar.flatten(1), glob_logvar],
            dim=1,
        )


class SlotDecoder(nn.Module):
    """Decoder counterpart to ``SlotEncoder``.

    The leading ``grid * grid * cell`` entries of the code are reshaped back onto the
    slot grid and the remaining entries are broadcast across every slot, so each place
    decodes from its own code together with the one the whole image shares. ``head``
    must accept ``cell + glob`` input channels.

    Args:
        head: Convolutional stack mapping (cell + glob, grid, grid) to the output.
        grid: Slots per side.
        cell: Latent dimensions per grid slot.
    """

    def __init__(self, head: nn.Module, grid: int, cell: int) -> None:
        super().__init__()
        self.head = head
        self.grid = grid
        self.cell = cell

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Place the slots back on their grid and broadcast the shared vector."""
        split = self.grid**2 * self.cell
        slots = z[:, :split].view(-1, self.cell, self.grid, self.grid)
        shared = z[:, split:, None, None].expand(-1, -1, self.grid, self.grid)
        return self.head(torch.cat([slots, shared], dim=1))


class SlotPrior(nn.Module):
    """Autoregressive prior over the code of a ``SlotEncoder``.

    A factorized ``N(0, I)`` prior treats every slot as independent, and no product of
    independent slots can express a constraint that couples places. Fibers never
    overlapping is exactly such a constraint, which is why a code drawn from
    ``N(0, I)`` decodes into merged blobs however hard the rate is throttled. This
    reads the code as one shared token followed by the slot tokens and models it with a
    causal transformer, so a slot is drawn conditioned on the slots already placed.
    Each conditional is a diagonal Gaussian mixture, which is what lets a slot stay
    sharply bimodal between empty and occupied.

    Training is plain maximum likelihood on ``log_prob`` of the encoded means. The same
    ``log_prob`` is exact and differentiable in the code, so it also serves as the
    penalty that holds a latent optimization on the manifold of real codes.

    Args:
        grid: Slots per side, as in ``SlotEncoder``.
        cell: Latent dimensions per slot, which is also the token width.
        glob: Dimensions of the shared vector, at most ``cell``.
        components: Mixture components per conditional.
        width: Transformer feature width.
        layers: Transformer layers.
        heads: Attention heads.
        dropout: Dropout inside the transformer. The prior is fitted to as many codes
            as there are images, so it overfits without it.
    """

    def __init__(
        self,
        grid: int,
        cell: int,
        glob: int,
        components: int = 10,
        width: int = 128,
        layers: int = 4,
        heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.grid = grid
        self.cell = cell
        self.glob = glob
        self.components = components
        tokens = grid**2 + 1  # the shared vector leads, then one token per slot
        self.start = nn.Parameter(torch.zeros(1, 1, cell))
        self.position = nn.Parameter(torch.randn(1, tokens, width) * 0.02)
        self.project = nn.Linear(cell, width)
        layer = nn.TransformerEncoderLayer(
            width,
            heads,
            2 * width,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.body = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = nn.Linear(width, components * (1 + 2 * cell))

    def tokenize(self, z: torch.Tensor) -> torch.Tensor:
        """Split a code into the shared token followed by one token per slot."""
        split = self.grid**2 * self.cell
        slots = z[:, :split].view(-1, self.cell, self.grid**2).transpose(1, 2)
        shared = nn.functional.pad(z[:, split:], (0, self.cell - self.glob))
        return torch.cat([shared[:, None], slots], dim=1)

    def detokenize(self, tokens: torch.Tensor) -> torch.Tensor:
        """Inverse of ``tokenize``."""
        slots = tokens[:, 1:].transpose(1, 2).flatten(1)
        return torch.cat([slots, tokens[:, 0, : self.glob]], dim=1)

    def predict(self, tokens: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Mixture weights, means and log-variances of the token at each position."""
        features = self.project(tokens) + self.position[:, : tokens.shape[1]]
        mask = nn.Transformer.generate_square_subsequent_mask(
            tokens.shape[1], device=tokens.device
        )
        features = self.body(features, mask=mask, is_causal=True)
        parameters = self.head(features)
        shape = (*parameters.shape[:2], self.components, self.cell)
        weight = parameters[..., : self.components]
        mean = parameters[..., self.components : self.components * (1 + self.cell)]
        logvar = parameters[..., self.components * (1 + self.cell) :]
        return weight, mean.view(shape), logvar.view(shape).clamp(-12, 6)

    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        """Exact log density of a code, differentiable in ``z``."""
        tokens = self.tokenize(z)
        start = self.start.expand(tokens.shape[0], -1, -1)
        shifted = torch.cat([start, tokens[:, :-1]], dim=1)
        weight, mean, logvar = self.predict(shifted)
        # the shared token is padded up to the slot width, so score only the real
        # dimensions of it and every dimension of the slots
        gauss = -0.5 * (
            (tokens[:, :, None] - mean) ** 2 / logvar.exp()
            + logvar
            + math.log(2 * math.pi)
        )
        gauss[:, 0, :, self.glob :] = 0.0
        component = weight.log_softmax(-1) + gauss.sum(-1)
        return torch.logsumexp(component, dim=-1).sum(dim=-1)

    @torch.no_grad()
    def sample(self, samples: int, temperature: float = 1.0) -> torch.Tensor:
        """Draw codes one token at a time. Below 1 the temperature sharpens samples."""
        device = self.start.device
        tokens = self.start.expand(samples, -1, -1)
        rows = torch.arange(samples, device=device)
        for _ in range(self.grid**2 + 1):
            weight, mean, logvar = self.predict(tokens)
            weight = (weight[:, -1] / temperature).softmax(-1)
            choice = torch.multinomial(weight, 1)[:, 0]
            drawn = mean[:, -1][rows, choice]
            spread = (0.5 * logvar[:, -1][rows, choice]).exp()
            drawn = drawn + temperature * spread * torch.randn_like(drawn)
            tokens = torch.cat([tokens, drawn[:, None]], dim=1)
        return self.detokenize(tokens[:, 1:])


class TwoStagePrior(nn.Module):
    """Learned prior that is itself a variational autoencoder over the codes.

    The two-stage construction of Dai and Wipf (2019): the first autoencoder is trained
    for reconstruction alone, and a second, much smaller one is fitted to the codes it
    produced. Sampling draws from the standard normal of the second stage and decodes
    twice. Unlike ``SlotPrior`` this ignores the spatial layout of the code and holds
    no attention, which is what makes it simple; the price is that it has to model the
    whole code at once instead of one slot at a time, so it needs a code small enough
    for that to be possible.

    ``log_prob`` returns the evidence lower bound rather than an exact density, which
    is all a penalty holding a latent optimization on the manifold needs. In evaluation
    mode the bound is taken at the posterior mean, so the penalty is deterministic.

    Args:
        codes: Encoded means the prior is fitted to; only their scale is read here.
        inner: Latent size of the second stage.
        width: Hidden width of both second-stage networks.
    """

    def __init__(self, codes: torch.Tensor, inner: int = 64, width: int = 512) -> None:
        super().__init__()
        latent = codes.shape[1]
        self.register_buffer("code_mean", codes.mean(dim=0))
        self.register_buffer("code_std", codes.std(dim=0) + 1e-6)
        self.inner = inner
        self.model = VAE(
            MLP([latent, width, width, 2 * inner], [nn.SiLU(), nn.SiLU(), None]),
            MLP([inner, width, width, latent], [nn.SiLU(), nn.SiLU(), None]),
        )
        self.logvar = nn.Parameter(torch.zeros(1))  # spread of the decoded code

    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        """Evidence lower bound on the log density, differentiable in ``z``."""
        u = (z - self.code_mean) / self.code_std
        u_pred, mean, logvar = self.model(u)
        spread = self.logvar.clamp(-10, 10)
        recon = -0.5 * (
            (u - u_pred) ** 2 / spread.exp() + spread + math.log(2 * math.pi)
        )
        kl = 0.5 * (logvar.clamp(-10, 10).exp() + mean**2 - logvar.clamp(-10, 10) - 1)
        # the standardization is a change of variables, so its jacobian belongs here
        return recon.sum(dim=1) - kl.sum(dim=1) - self.code_std.log().sum()

    @torch.no_grad()
    def sample(self, samples: int) -> torch.Tensor:
        """Decode a draw from the standard normal of the second stage."""
        seed = torch.randn(samples, self.inner, device=self.code_mean.device)
        u = self.model.decode(seed)
        u = u + (0.5 * self.logvar).exp() * torch.randn_like(u)
        return u * self.code_std + self.code_mean
