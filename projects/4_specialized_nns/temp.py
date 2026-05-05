import torch
from torch import nn

mlp = nn.Linear(2, 4)
cnn = nn.Conv1d(2, 4, 3, 1)

print(mlp.weight.data.shape)
print(cnn.weight.data.shape)
