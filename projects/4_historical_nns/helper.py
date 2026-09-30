import torch


def bars_and_stripes(size: int) -> torch.Tensor:
    """all distinct size x size images of constant rows (stripes) or columns (bars).

    The stripes come first, so the first 2^size rows are stripes. The all-zero and
    all-one images count as stripes only. Returns (2^(size+1) - 2, size^2).
    """
    codes = torch.arange(2**size)
    rows = ((codes[:, None] >> torch.arange(size)) & 1).float()
    stripes = rows[:, :, None].expand(-1, size, size)
    bars = stripes.transpose(1, 2)[1:-1]
    return torch.cat([stripes, bars]).flatten(1)


def contrastive_divergence(v, W, b, c, lr):
    """one contrastive divergence (CD-1) step of a restricted Boltzmann machine, in place.

    Args:
        v: (B, D) binary batch of visible states.
        W, b, c: (D, H) weights, (D,) visible and (H,) hidden biases.
    """
    p_h = torch.sigmoid(v @ W + c)
    h = torch.bernoulli(p_h)
    v_rec = torch.bernoulli(torch.sigmoid(h @ W.T + b))
    p_h_rec = torch.sigmoid(v_rec @ W + c)
    W += lr * (v.T @ p_h - v_rec.T @ p_h_rec) / v.shape[0]
    b += lr * (v - v_rec).mean(0)
    c += lr * (p_h - p_h_rec).mean(0)
