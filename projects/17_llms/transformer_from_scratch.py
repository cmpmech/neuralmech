import torch
from torch import nn

from NN import MLP


class SelfAttention(nn.Module):
    """single-head scaled dot-product self-attention.

    Reference: https://arxiv.org/abs/1706.03762
    """

    def __init__(self, D_in: int, D_out: int) -> None:
        super().__init__()
        self.W_query = nn.Linear(D_in, D_out, bias=False)
        self.W_key = nn.Linear(D_in, D_out, bias=False)
        self.W_value = nn.Linear(D_in, D_out, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        # @ contracts last dimension of first and second-to-last of second tensor
        attn_scores = queries @ keys.transpose(-2, -1)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        return attn_weights @ values


class MultiHeadAttention(nn.Module):
    """multi-head attention as a loop over separate `SelfAttention` heads."""

    def __init__(self, D_in: int, D_out: int, D_head: int, heads: int) -> None:
        super().__init__()
        self.heads = nn.ModuleList([SelfAttention(D_in, D_head) for _ in range(heads)])
        self.W_out = nn.Linear(D_head * heads, D_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.cat([head(x) for head in self.heads], dim=-1)
        return self.W_out(x)


class EfficientMultiHeadAttention(nn.Module):
    """multi-head attention with all heads in three combined projections.

    Same function as `MultiHeadAttention`, but the heads are batched into one
    matrix multiplication instead of looped over.
    """

    def __init__(self, D_in: int, D_out: int, D_head: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.D_head = D_head
        self.W_query = nn.Linear(D_in, D_head * heads, bias=False)
        self.W_key = nn.Linear(D_in, D_head * heads, bias=False)
        self.W_value = nn.Linear(D_in, D_head * heads, bias=False)
        self.W_out = nn.Linear(D_head * heads, D_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = (*x.shape[:-1], self.heads, self.D_head)
        # (..., N, heads, D_head) -> (..., heads, N, D_head)
        queries = self.W_query(x).view(shape).transpose(-3, -2)
        keys = self.W_key(x).view(shape).transpose(-3, -2)
        values = self.W_value(x).view(shape).transpose(-3, -2)

        attn_scores = queries @ keys.transpose(-2, -1)
        attn_weights = torch.softmax(attn_scores / self.D_head**0.5, dim=-1)
        y = attn_weights @ values

        y = y.transpose(-3, -2).reshape(*x.shape[:-1], self.heads * self.D_head)
        return self.W_out(y)


class TransformerBlock(nn.Module):
    """post-norm transformer block: attention and MLP, each with skip and layer norm.

    Reference: https://arxiv.org/abs/1706.03762
    """

    def __init__(self, D: int, attention: nn.Module | None = None) -> None:
        super().__init__()
        self.attention = attention or SelfAttention(D, D)
        self.norm1 = nn.LayerNorm(D)
        self.norm2 = nn.LayerNorm(D)  # separate trainable parameters
        self.mlp = MLP([D, 4 * D, D], [nn.GELU()])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self.attention(x))
        x = self.norm2(x + self.mlp(x))
        return x
