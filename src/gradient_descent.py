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


def learning_schedule(t, t0=5, t1=50):
    """Return a decaying learning rate"""
    return t0 / (t + t1)


def stochastic_gradient_descent(X_b, y, n_epochs=50, t0=5, t1=50, random_state=42):
    """Train using one randomly selected instance per update.

    Returns:
        theta: Learned parameters, shape (n_features, 1)
        cost_history: Full-dataset MSE recorded after each epoch
    """

    rng = np.random.RandomState(random_state)
    m, n = X_b.shape

    theta = rng.randn(n, 1)
    cost_history = []

    for epoch in range(n_epochs):
        for iterations in range(m):
            i = rng.randint(m)

            # Keep the sample two-dimensional for matrix multiplication
            xi = X_b[i : i + 1]
            yi = y[i : i + 1]

            gradient = 2 * xi.T @ (xi @ theta - yi)

            t = epoch * m + iterations
            eta = learning_schedule(t, t0, t1)
            theta = theta - eta * gradient

        # Measure the full-data MSE once per epoch
        cost = np.mean((X_b @ theta - y) ** 2)
        cost_history.append(cost)

    return theta, cost_history


def minibatch_gradient_descent(
    X_b, y, batch_size, n_epochs=50, t0=5, t1=50, random_state=42
):
    """Train using shuffled mini-batches

    Returns:
        theta: Learned parameters, shape (n_features, 1)
        cost_history: Full-dataset MSE recorded after each epoch
    """

    if batch_size < 1:
        raise ValueError("batch_size must be atleast 1.")

    rng = np.random.RandomState(random_state)
    m, n = X_b.shape

    theta = rng.randn(n, 1)
    cost_history = []
    t = 0

    for epoch in range(n_epochs):
        shuffled_indices = rng.permutation(m)
        X_shuffled = X_b[shuffled_indices]
        y_shuffled = y[shuffled_indices]

        for start in range(0, m, batch_size):
            y_batch = y_shuffled[start : start + batch_size]
            X_batch = X_shuffled[start : start + batch_size]

            actual_batch_size = len(X_batch)

            gradients = (
                (2 / actual_batch_size) * X_batch.T @ (X_batch @ theta - y_batch)
            )

            eta = learning_schedule(t, t0, t1)
            theta = theta - eta * gradients
            t += 1

        cost = np.mean((X_b @ theta - y) ** 2)
        cost_history.append(cost)

    return theta, cost_history
