from typing import Dict
import time
import numpy as np

from estimators import (
    sample_isotropic_vector,
    estimate_gradient_batch,
    estimate_q_from_residual,
)
from metrics import compute_all_metrics
from regularizers import elastic_net_mirror_map


def get_stepsize(k: int, rule: str, c0: float, power: float = 0.5) -> float:
    if rule == "constant":
        return c0
    if rule == "inv_sqrt":
        return c0 / ((k + 1) ** 0.5)
    if rule == "inv":
        return c0 / (k + 1)
    if rule == "polyak_like":
        return c0 / ((k + 1) ** power)
    raise ValueError(f"Unknown step rule: {rule}")


def batch_curvature_factor(n: int, sampler: str, B: int) -> float:
    if sampler == "gaussian":
        c = n + 2.0
    else:
        c = float(n)
    return 1.0 + (c - 1.0) / B


def theory_safe_step(operator, n: int, sampler: str, B: int, rho: float = 0.5) -> float:
    L = operator.spectral_norm_estimate() ** 2
    cB = batch_curvature_factor(n=n, sampler=sampler, B=B)
    return rho / (L * cB)


def get_decay_stepsize(k: int, alpha: float) -> float:
    return alpha / np.sqrt(k + 1.0)


def get_growing_batch_size(j: int, B0: int, Bmax: int, growth_alpha: float) -> int:
    return min(Bmax, int(np.ceil(B0 * ((j + 1.0) ** growth_alpha))))


def least_squares_objective_from_residual(residual: np.ndarray) -> float:
    return 0.5 * float(np.dot(residual, residual))


def init_history():
    return {
        "x_final": None,
        "z_final": None,
        "iter": [],
        "residual": [],
        "rel_error": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "exact_support": [],
        "nnz": [],
        "forward_calls": [],
        "time_sec": [],
        "tp": [],
        "fp": [],
        "fn": [],
        "best_f1": -1.0,
        "best_iter_by_f1": None,
        "best_x_by_f1": None,
        "step_size": [],
        "raw_step": [],
        "was_clipped": [],
        "safe_step_ref": [],
        "objective": [],
        "gap": [],
        "q_hat": [],
        "grad_norm_sq": [],
    }


def append_history(
    history,
    k,
    x,
    z,
    residual,
    x_true,
    tol,
    cumulative_forward_calls,
    elapsed,
    step_size=None,
    raw_step=None,
    was_clipped=None,
    safe_step_ref=None,
    objective=None,
    gap=None,
    q_hat=None,
    grad_norm_sq=None,
    true_mask=None,
    x_true_norm=None,
):
    if true_mask is None or x_true_norm is None:
        m = compute_all_metrics(x=x, x_true=x_true, residual=residual, tol=tol)
    else:
        pred_mask = np.abs(x) > tol
        tp = float(np.count_nonzero(pred_mask & true_mask))
        fp = float(np.count_nonzero(pred_mask & ~true_mask))
        fn = float(np.count_nonzero(~pred_mask & true_mask))
        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)
        f1 = 2 * precision * recall / (precision + recall + 1e-12)
        m = {
            "residual": float(np.linalg.norm(residual)),
            "rel_error": float(np.linalg.norm(x - x_true) / x_true_norm),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "exact_support": float(np.array_equal(pred_mask, true_mask)),
            "nnz": float(np.count_nonzero(pred_mask)),
        }

    history["iter"].append(k)
    history["residual"].append(m["residual"])
    history["rel_error"].append(m["rel_error"])
    history["precision"].append(m["precision"])
    history["recall"].append(m["recall"])
    history["f1"].append(m["f1"])
    history["exact_support"].append(m["exact_support"])
    history["nnz"].append(m["nnz"])
    history["tp"].append(m["tp"])
    history["fp"].append(m["fp"])
    history["fn"].append(m["fn"])
    history["forward_calls"].append(cumulative_forward_calls)
    history["time_sec"].append(elapsed)

    history["x_final"] = x.copy()
    history["z_final"] = None if z is None else z.copy()

    if step_size is not None:
        history["step_size"].append(step_size)
    if raw_step is not None:
        history["raw_step"].append(raw_step)
    if was_clipped is not None:
        history["was_clipped"].append(float(was_clipped))
    if safe_step_ref is not None:
        history["safe_step_ref"].append(safe_step_ref)
    if objective is not None:
        history["objective"].append(objective)
    if gap is not None:
        history["gap"].append(gap)
    if q_hat is not None:
        history["q_hat"].append(float(q_hat))
    if grad_norm_sq is not None:
        history["grad_norm_sq"].append(float(grad_norm_sq))

    if m["f1"] > history["best_f1"]:
        history["best_f1"] = m["f1"]
        history["best_iter_by_f1"] = k
        history["best_x_by_f1"] = x.copy()


def run_AFLBreI(operator, b, x_true, cfg, support_tol=1e-3, verbose=False) -> Dict:
    """
    Adjoint-Free Linearized Bregman Iteration (AFLBreI).

    Each iteration uses one update batch to estimate A^T(Ax-b) and an
    independent probe batch to estimate the Polyak denominator ||A^T(Ax-b)||^2.
    By default, the raw step is used to preserve existing experiment behavior.
    When cfg.clip_step is true, the step is capped by safe_step_ref.
    """
    n = x_true.shape[0]
    x = np.zeros(n)
    z = np.zeros(n)
    x_sum = np.zeros(n)
    true_mask = np.abs(x_true) > support_tol
    x_true_norm = np.linalg.norm(x_true) + 1e-12

    history = init_history()
    cumulative_forward_calls = 0
    t0 = time.perf_counter()

    B_final = cfg.batch_size
    use_growing_batch = getattr(cfg, "growing_batch", False)
    batch_floor_fraction = getattr(cfg, "batch_floor_fraction", 0.5)
    grow_probe_with_batch = getattr(cfg, "grow_probe_with_batch", False)
    probe_batch_ratio = getattr(cfg, "probe_batch_ratio", 1.0)

    def current_batch_size(k: int) -> int:
        if not use_growing_batch:
            return B_final
        progress = float(k + 1) / float(cfg.num_iters)
        scale = batch_floor_fraction + (1.0 - batch_floor_fraction) * progress
        return max(1, int(np.ceil(B_final * scale)))

    def current_probe_batch_size(B_k: int) -> int:
        if not grow_probe_with_batch:
            return cfg.q_batch_size
        return max(cfg.q_batch_size, int(np.ceil(probe_batch_ratio * B_k)))

    if verbose:
        print("[AFLBreI]")
        print("batch_size =", B_final)
        print("q_batch_size =", cfg.q_batch_size)
        print("growing_batch =", use_growing_batch)
        print("grow_probe_with_batch =", grow_probe_with_batch)
        print("sampler =", cfg.sampler)
        print("beta =", cfg.beta)
        print("f_star =", cfg.f_star)
        print()

    for k in range(cfg.num_iters):
        B_k = current_batch_size(k)
        M_k = current_probe_batch_size(B_k)
        cB = batch_curvature_factor(n=n, sampler=cfg.sampler, B=B_k)
        safe_step_ref = theory_safe_step(
            operator=operator,
            n=n,
            sampler=cfg.sampler,
            B=B_k,
            rho=cfg.step_safety,
        )

        g_hat, residual, used_calls = estimate_gradient_batch(
            operator=operator,
            x=x,
            b=b,
            batch_size=B_k,
            sampler=cfg.sampler,
        )
        cumulative_forward_calls += used_calls

        f_x = least_squares_objective_from_residual(residual)
        gap = max(f_x - cfg.f_star, cfg.eps_gap)

        q_hat, q_calls = estimate_q_from_residual(
            operator=operator,
            residual=residual,
            n=n,
            batch_size=M_k,
            sampler=cfg.sampler,
        )
        cumulative_forward_calls += q_calls

        raw_step = cfg.beta * (cfg.mu * gap) / (cB * max(q_hat, cfg.eps_denom))
        if getattr(cfg, "clip_step", False):
            step = min(raw_step, safe_step_ref)
        else:
            step = raw_step
        was_clipped = raw_step > safe_step_ref

        z = z - step * g_hat
        x = elastic_net_mirror_map(z, lam=cfg.lam, mu=cfg.mu)
        x_sum += x

        if (k % cfg.record_every) == 0:
            elapsed = time.perf_counter() - t0

            if getattr(cfg, "return_average", False):
                x_eval = x_sum / (k + 1)
                z_eval = None
                residual_eval = operator.forward(x_eval) - b
                cumulative_forward_calls += 1
                f_eval = least_squares_objective_from_residual(residual_eval)
                gap_eval = max(f_eval - cfg.f_star, 0.0)
            else:
                x_eval = x
                z_eval = z
                residual_eval = residual
                f_eval = f_x
                gap_eval = max(f_x - cfg.f_star, 0.0)

            append_history(
                history,
                k,
                x_eval,
                z_eval,
                residual_eval,
                x_true,
                tol=support_tol,
                cumulative_forward_calls=cumulative_forward_calls,
                elapsed=elapsed,
                step_size=step,
                raw_step=raw_step,
                was_clipped=was_clipped,
                safe_step_ref=safe_step_ref,
                objective=f_eval,
                gap=gap_eval,
                q_hat=float(q_hat),
                true_mask=true_mask,
                x_true_norm=x_true_norm,
            )

    return history


def run_oracle_lbrei(operator, b, x_true, cfg, support_tol=1e-3) -> Dict:
    """
    Oracle Linearized Bregman Iteration with exact-gradient Polyak step.

    Uses the true gradient g_k = A^T(A x_k - b) and the exact least-squares
    objective value to form the Polyak step:

        t_k = beta * mu * delta_k / ||g_k||^2

    where delta_k = 0.5 * ||A x_k - b||^2 (assuming f* = 0).
    """
    n = x_true.shape[0]
    x = np.zeros(n)
    z = np.zeros(n)
    true_mask = np.abs(x_true) > support_tol
    x_true_norm = np.linalg.norm(x_true) + 1e-12

    history = init_history()
    cumulative_forward_calls = 0
    t0 = time.perf_counter()

    for k in range(cfg.num_iters):
        residual = operator.forward(x) - b
        grad = operator.adjoint(residual)
        cumulative_forward_calls += 1

        # Exact-gradient Polyak step.
        delta_k = least_squares_objective_from_residual(residual)
        grad_norm_sq = float(np.dot(grad, grad))
        step = cfg.beta * cfg.mu * max(delta_k - cfg.f_star, 0.0) / max(grad_norm_sq, cfg.eps_denom)

        z = z - step * grad
        x = elastic_net_mirror_map(z, lam=cfg.lam, mu=cfg.mu)

        if (k % cfg.record_every) == 0:
            elapsed = time.perf_counter() - t0
            f_eval = least_squares_objective_from_residual(residual)
            append_history(
                history,
                k,
                x,
                z,
                residual,
                x_true,
                tol=support_tol,
                cumulative_forward_calls=cumulative_forward_calls,
                elapsed=elapsed,
                step_size=step,
                raw_step=step,
                was_clipped=False,
                safe_step_ref=np.nan,
                objective=f_eval,
                gap=f_eval,
                grad_norm_sq=grad_norm_sq,
                true_mask=true_mask,
                x_true_norm=x_true_norm,
            )

    return history


def run_sgdas(operator, b, x_true, cfg, support_tol=1e-3) -> Dict:
    n = x_true.shape[0]
    x = np.zeros(n)
    true_mask = np.abs(x_true) > support_tol
    x_true_norm = np.linalg.norm(x_true) + 1e-12

    history = init_history()
    cumulative_forward_calls = 0
    t0 = time.perf_counter()

    for k in range(cfg.num_iters):
        step = get_stepsize(k, cfg.step_rule, cfg.step_c0, cfg.step_power)

        residual = operator.forward(x) - b
        xi = sample_isotropic_vector(n, cfg.sampler)
        A_xi = operator.forward(xi)
        coeff = float(np.dot(residual, A_xi))
        g_hat = coeff * xi
        cumulative_forward_calls += 2

        x = x - step * g_hat

        if (k % cfg.record_every) == 0:
            elapsed = time.perf_counter() - t0
            f_eval = least_squares_objective_from_residual(residual)
            append_history(
                history,
                k,
                x,
                None,
                residual,
                x_true,
                tol=support_tol,
                cumulative_forward_calls=cumulative_forward_calls,
                elapsed=elapsed,
                step_size=step,
                raw_step=step,
                was_clipped=False,
                safe_step_ref=np.nan,
                objective=f_eval,
                gap=f_eval,
                true_mask=true_mask,
                x_true_norm=x_true_norm,
            )

    return history


def run_rd(operator, b, x_true, cfg, support_tol=1e-3) -> Dict:
    """
    Exact line search along random direction xi:
        tau = argmin_t 0.5 ||A(x + t xi) - b||^2
            = - <Axi, Ax-b> / ||Axi||^2
    """
    n = x_true.shape[0]
    x = np.zeros(n)
    true_mask = np.abs(x_true) > support_tol
    x_true_norm = np.linalg.norm(x_true) + 1e-12

    history = init_history()
    cumulative_forward_calls = 0
    t0 = time.perf_counter()

    for k in range(cfg.num_iters):
        residual = operator.forward(x) - b
        xi = sample_isotropic_vector(n, cfg.sampler)
        A_xi = operator.forward(xi)
        denom = float(np.dot(A_xi, A_xi)) + 1e-12
        tau = -float(np.dot(A_xi, residual)) / denom

        x = x + tau * xi
        cumulative_forward_calls += 2

        if (k % cfg.record_every) == 0:
            elapsed = time.perf_counter() - t0
            f_eval = least_squares_objective_from_residual(residual)
            append_history(
                history,
                k,
                x,
                None,
                residual,
                x_true,
                tol=support_tol,
                cumulative_forward_calls=cumulative_forward_calls,
                elapsed=elapsed,
                step_size=tau,
                raw_step=tau,
                was_clipped=False,
                safe_step_ref=np.nan,
                objective=f_eval,
                gap=f_eval,
                true_mask=true_mask,
                x_true_norm=x_true_norm,
            )

    return history
