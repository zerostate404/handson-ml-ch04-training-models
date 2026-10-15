from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
import numpy as np


def normal_equation(X, y):
    X_b = np.c_[np.ones((len(X), 1)), X]
    return np.linalg.pinv(X_b) @ y


def make_polynomial_model(degree):
    """Create a polynomial regression pipeline of the requested degree."""
    return make_pipeline(
        PolynomialFeatures(degree=degree, include_bias=False),
        LinearRegression(),
    )
