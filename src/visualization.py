import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import learning_curve


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


def plot_learning_curves(model, X, y):
    """Plot mean training and validation RMSE against training-set size."""
    train_sizes, train_scores, val_scores = learning_curve(
        model,
        X,
        np.asarray(y).ravel(),
        train_sizes=np.linspace(0.01, 1.0, 40),
        cv=5,
        scoring="neg_root_mean_squared_error",
    )

    # sklearn returns negative RMSE because its scoring API maximizes scores.
    train_rmse = -train_scores.mean(axis=1)
    val_rmse = -val_scores.mean(axis=1)

    plt.figure(figsize=(8, 5))
    plt.plot(train_sizes, train_rmse, "r-+", linewidth=2, label="Training RMSE")
    plt.plot(train_sizes, val_rmse, "b-", linewidth=3, label="Validation RMSE")
    plt.xlabel("Training set size")
    plt.ylabel("RMSE")
    plt.title("Learning Curves")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return train_sizes, train_rmse, val_rmse
