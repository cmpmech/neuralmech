from torch import nn


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
            else:
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)


# ----------------------------- standardizer -----------------------------
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
