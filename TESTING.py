import torch

x = torch.tensor([1.0], requires_grad=True)
y = 2 * x
z = y**2

print(x)
print(y)
print(z)

# y.retain_grad()
# z.backward(torch.ones_like(z))

# print(x.grad)
# print(y.grad)
