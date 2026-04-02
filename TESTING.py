import torch

x = torch.zeros((16, 32, 3, 3))

y1, y2 = torch.chunk(x, chunks=2, dim=1)


print(y1.shape, y2.shape)
