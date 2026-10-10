import numpy as np


def normal_equation(X, y):
    X_b = np.c_[np.ones((len(X), 1)), X]
    return np.linalg.pinv(X_b) @ y
