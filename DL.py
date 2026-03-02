import torch
from torch import nn
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
from tqdm import tqdm
import math
import numpy as np
from collections import defaultdict
from dataclasses import dataclass, asdict
import json

# ------------------------ weight initialization -------------------------
def init_weights(model, activation=None):
    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            if isinstance(activation,
                          (nn.ReLU, nn.ReLU6, nn.Hardswish, nn.SiLU,
                           nn.Mish, nn.GELU, nn.ELU, nn.CELU,
                           nn.Softplus)):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            elif isinstance(activation, nn.LeakyReLU):
                nn.init.kaiming_uniform_(m.weight,
                                         a=activation.negative_slope,
                                         nonlinearity='leaky_relu')
            elif isinstance(activation, nn.PReLU):
                nn.init.kaiming_uniform_(m.weight, a=activation.init,
                                         nonlinearity='leaky_relu')
            elif isinstance(activation, nn.RReLU):
                a = (activation.lower + activation.upper) / 2.
                nn.init.kaiming_uniform_(m.weight, a=a,
                                         nonlinearity='leaky_relu')
            elif isinstance(activation, nn.SELU):
                nn.init.kaiming_normal_(m.weight, nonlinearity='linear')
            elif isinstance(activation, (nn.Tanh, nn.Softsign)):
                nn.init.xavier_uniform_(m.weight,
                                        gain=nn.init.calculate_gain(
                                            'tanh'))
            elif isinstance(activation,
                            (nn.Sigmoid, nn.LogSigmoid, nn.Softmax)):
                nn.init.xavier_uniform_(m.weight,
                                        gain=nn.init.calculate_gain(
                                            'sigmoid'))
            else:
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

# ---------------------------- training loop -----------------------------
class TrainingLoop:
    def __init__(self):
        self.metrics = defaultdict(list)

    def run(self, train_step, val_step, train_loader, val_loader,
            epochs, log_every=10, callback=None, verbose=True):
        iterator = tqdm(range(epochs)) if verbose else range(epochs)
        for epoch in iterator:
            train_metrics = train_step(train_loader)
            if val_step is not None:
                val_metrics = val_step(val_loader)
            else:
                val_metrics = {}

            # update metrics
            for key, val in {**train_metrics, **val_metrics}.items():
                self.metrics[key].append(val)

            # optional callback, e.g., early stopping or optuna
            if callback is not None:
                callback(epoch, self.metrics)

            # log first metric from each
            if verbose and epoch % log_every == 0:
                log_dict = {}
                if train_metrics:
                    first_train = next(iter(train_metrics.items()))
                    log_dict['train'] = f'{first_train[1]:.2e}'
                if val_metrics:
                    first_val = next(iter(val_metrics.items()))
                    log_dict['val'] = f'{first_val[1]:.2e}'
                iterator.set_postfix(log_dict)
        return self.metrics

# ----------------------------- standardizer -----------------------------
class Standardizer(nn.Module):
    def __init__(self, X, dim=0): # default is to have sample dim at 0
        super().__init__()
        self.x_mean = X.mean(dim=dim, keepdim=True)
        self.x_std = X.std(dim=dim, keepdim=True).clamp_min(1e-8)

    def __call__(self, x):
        return (x - self.x_mean) / self.x_std

    def inverse(self, x):
        return x * self.x_std + self.x_mean
    
# # ------------------------------- configs --------------------------------
# @dataclass
# class TrainConfig:
#     epochs: int
#     lr: float
#     weight_decay: float
#     batch_size: int
#     device: str
#     log_every: int
#
#     def save(self, filepath):
#         with open(filepath, 'w') as f:
#             json.dump(asdict(self), f, indent=2)
#
#     @classmethod
#     def load(cls, filepath): # creates new instance from a file (factory)
#         with open(filepath, 'r') as f:
#             return cls(**json.load(f))



# def init_weights(model, activation=None):
#     for m in model.modules():
#         if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
#             if isinstance(activation, (nn.ReLU, nn.ReLU6, nn.Hardswish, nn.SiLU, nn.Mish)):
#                 nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
#             elif isinstance(activation, nn.LeakyReLU):
#                 nn.init.kaiming_uniform_(m.weight, a=activation.negative_slope, nonlinearity='leaky_relu')
#             elif isinstance(activation, nn.PReLU):
#                 nn.init.kaiming_uniform_(m.weight, a=0.25, nonlinearity='leaky_relu')
#             elif isinstance(activation, nn.GELU):
#                 nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
#             elif isinstance(activation, nn.SELU):
#                 fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
#                 nn.init.normal_(m.weight, mean=0, std=1.0 / math.sqrt(fan_in))
#             elif isinstance(activation, (nn.Tanh, nn.Sigmoid, nn.Softmax)):
#                 gain_name = 'tanh' if isinstance(activation, nn.Tanh) else 'sigmoid'
#                 gain = nn.init.calculate_gain(gain_name)
#                 nn.init.xavier_uniform_(m.weight, gain=gain)
#             else:
#                 nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
#             if m.bias is not None:
#                 nn.init.zeros_(m.bias)
#
# def init_weights(model, activation=None):
#     for m in model.modules():
#         if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
#             if isinstance(activation, (nn.ReLU, nn.GELU, nn.PReLU, nn.LeakyReLU,
#                                        nn.Mish, nn.Softplus, nn.SiLU, nn.ELU,
#                                        nn.GLU, nn.CELU, nn.SELU, nn.RReLU,
#                                        nn.ReLU6, nn.Hardswish, nn.Softshrink)):
#                 # default (for ReLU, true GELU, Mish, Softplus, SiLU, ELU, CELU, ReLU6, Hardswish, Softshrink
#                 nonlinearity = 'relu'
#                 gain = nn.init.calculate_gain('relu')
#                 # if isinstance(activation, (nn.GELU)):
#                 #     if hasattr(activation, 'approximate') and activation.approximate == 'tanh':
#                 #         nonlinearity = 'tanh'
#                 #         gain = nn.init.calculate_gain('tanh')
#                 if isinstance(activation, nn.PReLU):
#                     nonlinearity = 'leaky_relu'
#                     gain = nn.init.calculate_gain('leaky_relu',
#                                                   activation.init)
#                 elif isinstance(activation, nn.LeakyReLU):
#                     nonlinearity = 'leaky_relu'
#                     gain = nn.init.calculate_gain('leaky_relu',
#                                                   activation.negative_slope)
#                 elif isinstance(activation, nn.GLU):
#                     nonlinearity = 'sigmoid'
#                     gain = nn.init.calculate_gain('sigmoid')
#                 elif isinstance(activation, nn.SELU):
#                     nonlinearity = 'linear'
#                     gain = 1.0
#                 elif isinstance(activation, nn.RReLU):
#                     nonlinearity = 'leaky_relu'
#                     gain = nn.init.calculate_gain('leaky_relu', 0.01)
#                 # initialization
#                 nn.init.kaiming_uniform_(m.weight, a=gain, nonlinearity=nonlinearity)
#             elif isinstance(activation, (nn.Tanh, nn.Tanhshrink, nn.Sigmoid,
#                                          nn.Softsign, nn.LogSigmoid)):
#                 # default (for tanh, Tanhshrink, Softsign)
#                 gain = nn.init.calculate_gain('tanh')
#                 if isinstance(activation, (nn.Sigmoid, nn.LogSigmoid)):
#                     gain = nn.init.calculate_gain('sigmoid')
#                 # initialization
#                 nn.init.xavier_uniform_(m.weight, gain=gain)
#             if m.bias is not None:
#                 nn.init.zeros_(m.bias)

def pos_encoding(d_emb, sequence_length, max_sequence_length=10000):
    pos = torch.arange(0, sequence_length).unsqueeze(1)  # (seq_len, 1)
    i = torch.arange(0, d_emb, 2)  # even indices

    div_term = torch.exp(i * (-math.log(max_sequence_length) / d_emb))

    encoding = torch.zeros(sequence_length, d_emb)
    encoding[:, 0::2] = torch.sin(pos * div_term)
    encoding[:, 1::2] = torch.cos(pos * div_term)

    return encoding.unsqueeze(0)

# def rel_pos_encoding(d_emb, max_rel_distance=128):
#     num_distances = 2 * max_rel_distance + 1
#
#     distances = torch.arange(-max_rel_distance,
#                              max_rel_distance + 1).unsqueeze(1)
#     i = torch.arange(0, d_emb, 2)
#
#     div_term = torch.exp(i + (-math.log(10000) / d_emb))
#
#     encoding = torch.zeros(num_distances, d_emb)
#     encoding[:, 0::2] = torch.sin(distances + div_term)
#     encoding[:, 1::2] = torch.cos(distances + div_term)
#
#     return encoding














# # ------------------------- supervised learning --------------------------: TODO REMOVE ALL BELOW
# class DataLoadersSupervised:
#     def __init__(self, path, batch_sizes, data_split=(0.8, 0.2)):
#         data = np.load(path)
#         X = torch.from_numpy(data['X']).float()
#         Y = torch.from_numpy(data['Y']).float() #long() #.float()
#
#         self.dataset = TensorDataset(X, Y)
#         self.train_dataset, self.val_dataset = random_split(self.dataset, data_split)
#         self.train_loader = DataLoader(self.train_dataset, batch_size=batch_sizes[0], shuffle=True)  # full batch
#         self.val_loader = DataLoader(self.val_dataset, batch_size=batch_sizes[1], shuffle=False)
#
#     def get_data(self):
#         train_indices = self.train_dataset.indices
#         val_indices = self.val_dataset.indices
#         X_train = self.dataset.tensors[0][train_indices]
#         Y_train = self.dataset.tensors[1][train_indices]
#         X_val = self.dataset.tensors[0][val_indices]
#         Y_val = self.dataset.tensors[1][val_indices]
#         return (X_train, Y_train), (X_val, Y_val)
#
#     def get_dataloaders(self):
#         return self.train_loader, self.val_loader
#
# # could add early stopping, device handling?
# def train_supervised(model, cost_fun, train_loader, val_loader, optimizer, epochs, scheduler=None, print_every=100, device='cpu'):
#     train_cost = [0] * epochs
#     val_cost = [0] * epochs
#     pbar = tqdm(range(epochs), desc="Training")
#     # for epoch in tqdm(range(epochs), desc="Training"):
#     for epoch in pbar:
#         model.train()
#         for x, y in train_loader:
#             x, y = x.to(device), y.to(device)
#             optimizer.zero_grad()
#             y_pred = model(x)
#             cost = cost_fun(y_pred, y)
#             cost.backward()
#             optimizer.step()
#             train_cost[epoch] += cost.item()
#         train_cost[epoch] /= len(train_loader) # avg per batch
#         if scheduler is not None:
#             scheduler.step()
#
#         model.eval()
#         with torch.no_grad():
#             for x, y in val_loader:
#                 x, y = x.to(device), y.to(device)
#                 y_pred = model(x)
#                 cost = cost_fun(y_pred, y)
#                 val_cost[epoch] += cost.item()
#             val_cost[epoch] /= len(val_loader)  # avg per batch
#
#         if epoch % print_every == 0:
#             pbar.set_postfix({
#                 'train': f'{train_cost[epoch]:.3e}',
#                 'val': f'{val_cost[epoch]:.3e}'
#             })
#             # tqdm.write(f"Epoch {epoch}/{epochs}: train cost: {train_cost[epoch]:.3e}\tval cost: {val_cost[epoch]:.3e}")
#             # print(f"Epoch {epoch:d}/{epochs:d}: train cost: {train_cost[epoch]:.3e}\tval cost: {val_cost[epoch]:.3e}")
#
#     return train_cost, val_cost



