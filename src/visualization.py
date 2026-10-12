import numpy as np
import matplotlib.pyplot as plt


def plot_gd_path(X, y, X_b, gd_path, etas, theta_init, n_iterations=10):
    fig, axes = plt.subplots(1, len(etas), figsize=(5 * len(etas), 4), sharey=True)
    axes = np.atleast_1d(axes)

    for ax, eta in zip(axes, etas):
        X_new = np.array([[0], [2]])
        X_new_b = np.c_[np.ones((2, 1)), X_new]
        path = gd_path(X_b, y, eta, n_iterations, theta_init)
        ax.scatter(X, y, s=10, alpha=0.6)

        for i, theta in enumerate(path):
            y_pred = X_new_b @ theta
            style = "r--" if i == 0 else "b-"
            ax.plot(X_new, y_pred, style, alpha=0.6)
        ax.set_ylim(0, 15)
        ax.set_title(f"eta = {eta}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    plt.tight_layout()
    plt.show()
