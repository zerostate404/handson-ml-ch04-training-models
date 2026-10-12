import numpy as np


def batch_gradient_descent(X_b, y, eta, n_iterations, tol=None, theta_init=None):
    """
    Batch gradient descent.

    cost_history[i] is the MSE after step i + 1.
    n_iterations is the maximum iteration cap; tol may stop earlier.
    """

    m = len(X_b)

    if theta_init is None:
        theta_init = np.random.randn(X_b.shape[1], 1)

    theta = theta_init.copy()

    cost_history = []

    for _ in range(n_iterations):
        gradient = (2 / m) * X_b.T @ (X_b @ theta - y)
        if tol is not None and np.linalg.norm(gradient) <= tol:
            break
        theta = theta - eta * gradient
        cost = np.mean((X_b @ theta - y) ** 2)
        cost_history.append(cost)

    return theta, cost_history


def gd_path(X_b, y, eta, n_iterations, theta_init):
    m = len(X_b)
    theta = theta_init.copy()
    path = [theta.copy()]

    for _ in range(n_iterations):
        gradient = (2 / m) * X_b.T @ (X_b @ theta - y)
        theta = theta - eta * gradient
        path.append(theta.copy())

    return path
