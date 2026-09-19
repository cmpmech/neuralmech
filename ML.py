from itertools import combinations_with_replacement

import numpy as np

# ------------------------------ parametrized regression ------------------------------
class LinearRegression:  # limited to 1D outputs
    """Linear fit trained by gradient descent on the mean squared error."""

    def __init__(self):
        self.weight = 0
        self.bias = 0

    def forward(self, x):
        return self.weight * x + self.bias

    def cost_fun(self, x, y):
        y_pred = self.forward(x)
        return np.mean((y - y_pred)**2)

    def cost_fun_grad(self, x, y):
        y_pred = self.forward(x)
        res = y_pred - y
        return 2 * np.mean(x * res), 2 * np.mean(res)

    def train(self, epochs, lr, x_train, y_train, x_val, y_val):
        train_cost = [0] * epochs
        val_cost = [0] * epochs
        for epoch in range(epochs):
            train_cost[epoch] = self.cost_fun(x_train, y_train)
            val_cost[epoch] = self.cost_fun(x_val, y_val)
            grad = self.cost_fun_grad(x_train, y_train)
            self.weight -= lr * grad[0]
            self.bias -= lr * grad[1]
        return train_cost, val_cost

class LogisticRegression: # limited to 1D outputs
    """Binary classifier trained by gradient descent on the cross-entropy."""

    def __init__(self):
        self.weights = np.zeros(2)
        self.bias = 0

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def forward(self, x):
        z = x@self.weights + self.bias
        return self.sigmoid(z)

    def cost_fun(self, x, y):
        y_pred = self.forward(x)
        # binary cross-entropy
        eps = 1e-7  # for numerical stability
        return -np.mean(y * np.log(y_pred + eps) +
                       (1 - y) * np.log(1 - y_pred + eps))

    def cost_fun_grad(self, x, y):
        y_pred = self.forward(x)
        res = y_pred - y
        return np.mean(x.T * res, 1), np.mean(res)

    def train(self, epochs, lr, x_train, y_train, x_val, y_val):
        train_cost = [0] * epochs
        val_cost = [0] * epochs
        for epoch in range(epochs):
            train_cost[epoch] = self.cost_fun(x_train, y_train)
            val_cost[epoch] = self.cost_fun(x_val, y_val)
            grad_w, grad_b = self.cost_fun_grad(x_train, y_train)
            self.weights -= lr * grad_w
            self.bias -= lr * grad_b
        return train_cost, val_cost

    def predict(self, x):
        return (self.forward(x) > 0.5).astype(float)

class PolynomialRegression: # limited to 1D in- & outputs
    """Polynomial fit of degree p from the ridge-regularized normal equations."""

    def __init__(self, p, regularization):
        self.p = p
        self.regularization = regularization
        self.w = np.zeros(p + 1)

    def feature_matrix(self, x):
        return np.vander(x, self.p + 1, increasing=True)

    def forward(self, x):
        X = self.feature_matrix(x)
        return self.w@X.T

    def fit(self, x, y):
        X = self.feature_matrix(x)
        I = np.eye(self.p + 1)
        I[0, 0] = 0 # do not penalize bias
        A = (X.T@X) + (self.regularization * I)
        b = X.T@y
        self.w = np.linalg.solve(A, b)

class BayesianPolynomialRegression: # limited to 1D outputs
    """Bayesian polynomial regression with a closed-form Gaussian posterior.

    A Gaussian prior N(0, regularization^-1) on the weights and Gaussian observation
    noise of standard deviation `noise` give a Gaussian posterior over the weights. Its
    mean is the ridge solution, its covariance is the epistemic uncertainty, and
    `noise` is the aleatoric uncertainty.
    """

    def __init__(self, p: int, regularization: float, noise: float):
        self.p = p
        self.regularization = regularization # prior precision of the weights
        self.noise = noise # aleatoric standard deviation, in units of y
        self.w = None
        self.covariance = None

    def feature_matrix(self, X: np.ndarray) -> np.ndarray:
        """Expand (N, D) inputs into all monomials up to degree p, cross terms
        included."""
        columns = [np.ones(X.shape[0])]
        for degree in range(1, self.p + 1):
            for ids in combinations_with_replacement(range(X.shape[1]), degree):
                columns.append(np.prod(X[:, ids], axis=1))
        return np.stack(columns, axis=1)

    def forward(self, X: np.ndarray) -> np.ndarray:
        return self.feature_matrix(X)@self.w

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        Phi = self.feature_matrix(X)
        # unlike the ridge fits above the bias is penalized too, so that the prior
        # keeps the posterior proper when there are fewer samples than monomials
        A = (Phi.T@Phi / self.noise**2) + (self.regularization * np.eye(Phi.shape[1]))
        self.covariance = np.linalg.inv(A)
        self.w = self.covariance@Phi.T@y / self.noise**2

    def epistemic_std(self, X: np.ndarray) -> np.ndarray:
        """Standard deviation of the predictive mean, excluding the aleatoric noise."""
        Phi = self.feature_matrix(X)
        return np.sqrt(np.einsum("ij,jk,ik->i", Phi, self.covariance, Phi))