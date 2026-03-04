class Node:
    def __init__(self, data, parents=(), grad_fn=None):
        self.data = data
        self.grad = 0.0
        self.parents = parents
        self.grad_fn = grad_fn # local backward rule

# ---------------- backward graph traversal for gradients ----------------
    def backward(self):
        forward_topological_order = []
        visited = set()

        def build(n):
            if n not in visited:
                visited.add(n)
                for p in n.parents:
                    build(p)
                forward_topological_order.append(n)

        build(self)
        self.grad = 1.0
        for v in reversed(forward_topological_order):
            if v.grad_fn:
                v.grad_fn()

# -------------------------- override operators --------------------------
    # +
    def __add__(self, other):
        other = other if isinstance(other, Node) else Node(other)
        out = Node(self.data + other.data, (self, other))
        def grad_fn():
            self.grad += out.grad # propagate grad
            other.grad += out.grad # propagate grad
        out.grad_fn = grad_fn
        return out

    # *
    def __mul__(self, other):
        other = other if isinstance(other, Node) else Node(other)
        out = Node(self.data * other.data, (self, other))
        def grad_fn():
            self.grad += other.data * out.grad # propagate grad
            other.grad += self.data * out.grad # propagate grad
        out.grad_fn = grad_fn
        return out

# -------------------------------- driver --------------------------------
a = Node(3.0)
b = Node(2.0)
c = Node(1.0)

d = (a * b + c) * a
d.backward()

print(f'd = {d.data}')
print(f'dc/da = {a.grad} ({2 * a.data * b.data + c.data})')
print(f'dc/db = {b.grad} ({a.data**2})')
print(f'dc/dd = {c.grad} ({a.data})')