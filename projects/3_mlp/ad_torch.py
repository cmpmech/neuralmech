import torch
torch.backends.cudnn.deterministic = True

# -------------------------------- driver --------------------------------
a = torch.tensor(3.0, requires_grad=True)
b = torch.tensor(2.0, requires_grad=True)
c = torch.tensor(1.0, requires_grad=True)

d = (a * b + c) * a
d.backward()

print(f'd = {d.data}')
print(f'dc/da = {a.grad} ({2 * a.data * b.data + c.data})')
print(f'dc/db = {b.grad} ({a.data**2})')
print(f'dc/dd = {c.grad} ({a.data})')