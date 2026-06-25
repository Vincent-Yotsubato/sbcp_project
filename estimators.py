import numpy as np


def sample_isotropic_vector(dim: int, sampler: str = "gaussian") -> np.ndarray:
    if sampler == "gaussian":
        return np.random.randn(dim)
    if sampler == "rademacher":
        return np.random.randint(0, 2, size=dim) * 2.0 - 1.0
    if sampler == "sphere":
        v = np.random.randn(dim)
        v = v / (np.linalg.norm(v) + 1e-12)
        return np.sqrt(dim) * v
    raise ValueError(f"Unknown sampler: {sampler}")


def estimate_gradient_single(operator, x: np.ndarray, b: np.ndarray, xi: np.ndarray):
    residual = operator.forward(x) - b
    A_xi = operator.forward(xi)
    coeff = float(np.dot(residual, A_xi))
    g_hat = coeff * xi
    forward_calls = 2
    return g_hat, residual, forward_calls


def _sample_isotropic_batch(batch_size: int, n: int, sampler: str) -> np.ndarray:
    if sampler == "gaussian":
        return np.random.randn(batch_size, n)
    if sampler == "rademacher":
        return np.random.randint(0, 2, size=(batch_size, n)) * 2.0 - 1.0
    return np.array([sample_isotropic_vector(n, sampler) for _ in range(batch_size)])


def estimate_gradient_batch(
    operator,
    x: np.ndarray,
    b: np.ndarray,
    batch_size: int,
    sampler: str = "gaussian",
):
    n = x.shape[0]
    residual = operator.forward(x) - b
    Xi = _sample_isotropic_batch(batch_size, n, sampler)

    if hasattr(operator, "forward_batch"):
        A_Xi = operator.forward_batch(Xi)
    else:
        A_Xi = np.array([operator.forward(xi) for xi in Xi])
    forward_calls = 1 + batch_size

    coeffs = A_Xi @ residual
    g_hat = (coeffs @ Xi) / batch_size
    return g_hat, residual, forward_calls


def estimate_gradient_and_q_batch(
    operator,
    x: np.ndarray,
    b: np.ndarray,
    update_batch_size: int,
    probe_batch_size: int,
    sampler: str = "gaussian",
):
    n = x.shape[0]
    residual = operator.forward(x) - b
    Xi = _sample_isotropic_batch(update_batch_size, n, sampler)
    Zeta = _sample_isotropic_batch(probe_batch_size, n, sampler)
    directions = np.vstack([Xi, Zeta])

    if hasattr(operator, "forward_batch"):
        A_directions = operator.forward_batch(directions)
    else:
        A_directions = np.array([operator.forward(direction) for direction in directions])
    forward_calls = 1 + update_batch_size + probe_batch_size

    A_Xi = A_directions[:update_batch_size]
    A_Zeta = A_directions[update_batch_size:]

    grad_coeffs = A_Xi @ residual
    g_hat = (grad_coeffs @ Xi) / update_batch_size

    probe_coeffs = A_Zeta @ residual
    q_hat = np.sum(probe_coeffs ** 2) / probe_batch_size

    return g_hat, residual, q_hat, forward_calls


def estimate_q_from_residual(
    operator,
    residual: np.ndarray,
    n: int,
    batch_size: int,
    sampler: str = "gaussian",
):
    Zeta = _sample_isotropic_batch(batch_size, n, sampler)

    if hasattr(operator, "forward_batch"):
        A_Zeta = operator.forward_batch(Zeta)
    else:
        A_Zeta = np.array([operator.forward(z) for z in Zeta])
    forward_calls = batch_size

    coeffs = A_Zeta @ residual
    q_hat = np.sum(coeffs ** 2) / batch_size
    return q_hat, forward_calls
