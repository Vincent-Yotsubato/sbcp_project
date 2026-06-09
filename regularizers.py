# regularizers.py
import numpy as np


def soft_threshold(z: np.ndarray, thresh: float) -> np.ndarray:
    return np.sign(z) * np.maximum(np.abs(z) - thresh, 0.0)


def elastic_net_mirror_map(z: np.ndarray, lam: float, mu: float = 1.0) -> np.ndarray:
    """
    For psi(x) = lam * ||x||_1 + (mu/2) * ||x||_2^2,
    grad psi^*(z) = (1/mu) * soft_threshold(z, lam)
    """
    return soft_threshold(z, lam) / mu


def elastic_net_value(x: np.ndarray, lam: float, mu: float = 1.0) -> float:
    return lam * np.sum(np.abs(x)) + 0.5 * mu * np.dot(x, x)


def least_squares_value(residual: np.ndarray) -> float:
    return 0.5 * np.dot(residual, residual)


def bilevel_surrogate_value(x: np.ndarray, residual: np.ndarray, lam: float, mu: float = 1.0) -> float:
    return least_squares_value(residual) + elastic_net_value(x, lam=lam, mu=mu)