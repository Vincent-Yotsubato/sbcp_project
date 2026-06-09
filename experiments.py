from typing import Dict, List
import copy
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from algorithms import (
    batch_curvature_factor,
    least_squares_objective_from_residual,
    theory_safe_step,
    run_AFLBreI,
    run_oracle_lbrei,
    run_sgdas,
    run_rd,
)
from estimators import estimate_gradient_batch, estimate_q_from_residual
from metrics import compute_image_metrics
from problems import (
    build_sparse_recovery_problem,
    build_csmri_problem,
    build_deconv1d_problem,
)
from utils import set_random_seed
from regularizers import elastic_net_mirror_map


def average_histories(histories: List[Dict]) -> Dict:
    keys = [
        "iter",
        "residual",
        "rel_error",
        "precision",
        "recall",
        "f1",
        "exact_support",
        "nnz",
        "forward_calls",
        "time_sec",
        "tp",
        "fp",
        "fn",
        "step_size",
        "raw_step",
        "was_clipped",
        "safe_step_ref",
        "objective",
        "gap",
    ]

    avg = {}
    for key in keys:
        arr = np.array([h[key] for h in histories], dtype=float)
        avg[key + "_mean"] = arr.mean(axis=0)
        avg[key + "_std"] = arr.std(axis=0)

    optional_keys = [
        "q_hat",
        "grad_norm_sq",
        "oracle_step",
        "q_over_gsq",
        "step_ratio",
    ]

    for key in optional_keys:
        if all(key in h and len(h[key]) > 0 for h in histories):
            arr = np.array([h[key] for h in histories], dtype=float)
            avg[key + "_mean"] = arr.mean(axis=0)
            avg[key + "_std"] = arr.std(axis=0)

    avg["x_final_list"] = [h["x_final"] for h in histories]
    avg["z_final_list"] = [h["z_final"] for h in histories]
    avg["best_f1_list"] = [h["best_f1"] for h in histories]
    avg["best_iter_by_f1_list"] = [h["best_iter_by_f1"] for h in histories]
    avg["best_x_by_f1_list"] = [h["best_x_by_f1"] for h in histories]

    return avg


def run_one_trial(cfg, trial_seed: int) -> Dict:
    set_random_seed(trial_seed)

    problem = build_sparse_recovery_problem(cfg.problem)
    operator = problem["operator"]
    b = problem["b"]
    x_true = problem["x_true"]

    results = {}

    operator.reset_counters()
    results["AFLBreI"] = run_AFLBreI(
        operator, b, x_true, cfg.AFLBreI, support_tol=cfg.support_tol
    )

    operator.reset_counters()
    results["Oracle-LBreI"] = run_oracle_lbrei(
        operator, b, x_true, cfg.oracle, support_tol=cfg.support_tol
    )

    operator.reset_counters()
    results["SGDAS"] = run_sgdas(
        operator, b, x_true, cfg.sgdas, support_tol=cfg.support_tol
    )

    operator.reset_counters()
    results["RD"] = run_rd(
        operator, b, x_true, cfg.rd, support_tol=cfg.support_tol
    )

    results["problem"] = {
        "A": problem["A"],
        "b": b,
        "x_true": x_true,
    }
    return results


def run_main_compare(cfg) -> Dict:
    trial_seeds = [cfg.problem.seed + t for t in range(cfg.num_trials)]
    trial_results = [run_one_trial(cfg, seed) for seed in trial_seeds]

    summary = {}
    for method in ["AFLBreI", "Oracle-LBreI", "SGDAS", "RD"]:
        summary[method] = average_histories([tr[method] for tr in trial_results])

    summary["num_trials"] = cfg.num_trials
    return {"trial_results": trial_results, "summary": summary}


def run_batch_ablation(cfg, batch_sizes=(1, 16, 32, 64), max_budget=136000) -> Dict:
    out = {}
    for B in batch_sizes:
        cfg_local = copy.deepcopy(cfg)
        cfg_local.AFLBreI.batch_size = B
        cfg_local.AFLBreI.record_every = 20
        calls_per_iter = 1 + B + cfg_local.AFLBreI.q_batch_size
        cfg_local.AFLBreI.num_iters = max(1, max_budget // calls_per_iter)

        trials = []
        for t in range(cfg_local.num_trials):
            set_random_seed(cfg_local.problem.seed + t)
            problem = build_sparse_recovery_problem(cfg_local.problem)
            operator = problem["operator"]

            operator.reset_counters()
            hist = run_AFLBreI(
                operator,
                problem["b"],
                problem["x_true"],
                cfg_local.AFLBreI,
                support_tol=cfg_local.support_tol,
            )
            trials.append(hist)

        out[f"B={B}"] = average_histories(trials)
    return out


def run_probe_ablation(
    cfg,
    probe_batch_sizes=(1, 4, 8, 16, 32, 64),
    max_budget=136000,
    beta_min=0.25,
) -> Dict:
    out = {}
    for M in probe_batch_sizes:
        cfg_local = copy.deepcopy(cfg)
        cfg_local.AFLBreI.q_batch_size = M
        cfg_local.AFLBreI.beta = max(beta_min, 2.0 - 8.0 / M)
        cfg_local.AFLBreI.clip_step = True
        calls_per_iter = 1 + cfg_local.AFLBreI.batch_size + M
        cfg_local.AFLBreI.num_iters = max(1, max_budget // calls_per_iter)

        trials = []
        for t in range(cfg_local.num_trials):
            set_random_seed(cfg_local.problem.seed + t)
            problem = build_sparse_recovery_problem(cfg_local.problem)
            operator = problem["operator"]

            operator.reset_counters()
            hist = run_AFLBreI(
                operator,
                problem["b"],
                problem["x_true"],
                cfg_local.AFLBreI,
                support_tol=cfg_local.support_tol,
            )
            trials.append(hist)

        out[f"M={M}"] = average_histories(trials)
    return out


def run_stepsize_ablation(
    cfg,
    betas=(0.25, 0.5, 0.8, 1.0, 1.5, 2.0),
    verbose=True,
) -> Dict:
    out = {}

    if verbose:
        print("[AFLBreI Beta Ablation]")
        print("betas =", betas)
        print()

    for beta in betas:
        cfg_local = copy.deepcopy(cfg)
        cfg_local.AFLBreI.beta = beta
        cfg_local.AFLBreI.record_every = 10

        trials = []
        for t in range(cfg_local.num_trials):
            set_random_seed(cfg_local.problem.seed + t)
            problem = build_sparse_recovery_problem(cfg_local.problem)
            operator = problem["operator"]

            operator.reset_counters()
            hist = run_AFLBreI(
                operator,
                problem["b"],
                problem["x_true"],
                cfg_local.AFLBreI,
                support_tol=cfg_local.support_tol,
                verbose=False,
            )
            trials.append(hist)

        out[f"beta={beta:.3g}"] = average_histories(trials)

    return out


def run_sparsity_scaling_experiment(cfg, sparsities=(10, 20, 30, 40)) -> Dict:
    out = {}
    for s in sparsities:
        cfg_local = copy.deepcopy(cfg)
        cfg_local.problem.s = s
        cfg_local.AFLBreI.record_every = 10

        trials = []
        for t in range(cfg_local.num_trials):
            set_random_seed(cfg_local.problem.seed + t)
            problem = build_sparse_recovery_problem(cfg_local.problem)
            operator = problem["operator"]

            operator.reset_counters()
            hist = run_AFLBreI(
                operator,
                problem["b"],
                problem["x_true"],
                cfg_local.AFLBreI,
                support_tol=cfg_local.support_tol,
                verbose=False,
            )
            trials.append(hist)

        out[f"s={s}"] = average_histories(trials)

    return out


def run_noise_robustness_experiment(cfg, snr_levels=(20, 30, 40, None)) -> Dict:
    out = {}
    for snr in snr_levels:
        cfg_local = copy.deepcopy(cfg)
        cfg_local.problem.snr_db = snr
        cfg_local.problem.noise_sigma = None
        label = f"SNR={snr}dB" if snr is not None else "SNR=inf (Noiseless)"

        trials = []
        for t in range(cfg_local.num_trials):
            set_random_seed(cfg_local.problem.seed + t)
            problem = build_sparse_recovery_problem(cfg_local.problem)
            operator = problem["operator"]

            operator.reset_counters()
            hist = run_AFLBreI(
                operator,
                problem["b"],
                problem["x_true"],
                cfg_local.AFLBreI,
                support_tol=cfg_local.support_tol,
                verbose=False,
            )
            trials.append(hist)

        out[label] = average_histories(trials)
    return out


def _scaled_growing_batch_size(
    k: int,
    K: int,
    B_final: int,
    floor_fraction: float = 0.5,
) -> int:
    progress = float(k + 1) / float(K)
    scale = floor_fraction + (1.0 - floor_fraction) * progress
    return max(1, int(np.ceil(B_final * scale)))


def _row_space_distance_metrics(A: np.ndarray, z: np.ndarray) -> Dict[str, float]:
    _, singular_values, vt = np.linalg.svd(A, full_matrices=False)
    z_norm_sq = float(np.dot(z, z))

    if singular_values.size == 0:
        return {
            "dual_range_violation": z_norm_sq,
            "relative_dual_range_violation": 1.0,
            "z_norm_sq": z_norm_sq,
        }

    tol = max(A.shape) * np.finfo(float).eps * singular_values[0]
    rank = int(np.sum(singular_values > tol))
    if rank == 0:
        return {
            "dual_range_violation": z_norm_sq,
            "relative_dual_range_violation": 1.0,
            "z_norm_sq": z_norm_sq,
        }

    row_basis = vt[:rank, :].T
    row_projection = row_basis @ (row_basis.T @ z)
    residual = z - row_projection
    distance_sq = float(np.dot(residual, residual))
    return {
        "dual_range_violation": distance_sq,
        "relative_dual_range_violation": distance_sq / (z_norm_sq + 1e-12),
        "z_norm_sq": z_norm_sq,
    }


def _run_AFLBreI_growing_batch_final(
    operator,
    b,
    x_true,
    cfg,
    B_final: int,
    batch_floor_fraction: float = 0.5,
    clip_step: bool = True,
) -> Dict:
    n = x_true.shape[0]
    K = cfg.num_iters
    x = np.zeros(n)
    z = np.zeros(n)
    cumulative_forward_calls = 0
    clipped_steps = 0
    batch_sizes = []

    for k in range(K):
        B_k = _scaled_growing_batch_size(
            k,
            K,
            B_final,
            floor_fraction=batch_floor_fraction,
        )
        batch_sizes.append(B_k)
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
            batch_size=cfg.q_batch_size,
            sampler=cfg.sampler,
        )
        cumulative_forward_calls += q_calls

        cB = batch_curvature_factor(n=n, sampler=cfg.sampler, B=B_k)
        raw_step = cfg.beta * (cfg.mu * gap) / (cB * max(q_hat, cfg.eps_denom))
        if clip_step:
            safe_step = theory_safe_step(
                operator=operator,
                n=n,
                sampler=cfg.sampler,
                B=B_k,
                rho=cfg.step_safety,
            )
            step = min(raw_step, safe_step)
            clipped_steps += int(raw_step > safe_step)
        else:
            step = raw_step

        z = z - step * g_hat
        x = elastic_net_mirror_map(z, lam=cfg.lam, mu=cfg.mu)

    residual_final = operator.forward(x) - b
    cumulative_forward_calls += 1
    final_gap = max(least_squares_objective_from_residual(residual_final) - cfg.f_star, 0.0)

    return {
        "x_final": x,
        "z_final": z,
        "final_gap": float(final_gap),
        "final_residual_norm": float(np.linalg.norm(residual_final)),
        "forward_calls": int(cumulative_forward_calls),
        "mean_update_batch_size": float(np.mean(batch_sizes)),
        "final_update_batch_size": int(batch_sizes[-1]),
        "clipped_fraction": float(clipped_steps / max(K, 1)),
    }


def run_growing_batch_kkt_experiment(
    cfg,
    final_batch_sizes=(50, 100, 200, 500),
    batch_floor_fraction=0.5,
    clip_step=True,
) -> Dict:
    out = {
        "B_final": [],
        "final_gap_mean": [],
        "final_gap_std": [],
        "dual_range_violation_mean": [],
        "dual_range_violation_std": [],
        "relative_dual_range_violation_mean": [],
        "relative_dual_range_violation_std": [],
        "z_norm_sq_mean": [],
        "z_norm_sq_std": [],
        "forward_calls_mean": [],
        "forward_calls_std": [],
        "mean_update_batch_size_mean": [],
        "mean_update_batch_size_std": [],
        "clipped_fraction_mean": [],
        "clipped_fraction_std": [],
        "num_trials": cfg.num_trials,
        "K": cfg.AFLBreI.num_iters,
        "probe_batch_size": cfg.AFLBreI.q_batch_size,
        "batch_floor_fraction": batch_floor_fraction,
        "clip_step": clip_step,
    }

    for B_final in final_batch_sizes:
        gaps = []
        violations = []
        relative_violations = []
        z_norms = []
        forward_calls = []
        mean_batch_sizes = []
        clipped_fractions = []

        for t in range(cfg.num_trials):
            set_random_seed(cfg.problem.seed + t)
            problem = build_sparse_recovery_problem(cfg.problem)
            operator = problem["operator"]
            operator.reset_counters()

            hist = _run_AFLBreI_growing_batch_final(
                operator=operator,
                b=problem["b"],
                x_true=problem["x_true"],
                cfg=cfg.AFLBreI,
                B_final=B_final,
                batch_floor_fraction=batch_floor_fraction,
                clip_step=clip_step,
            )

            dual_metrics = _row_space_distance_metrics(problem["A"], hist["z_final"])
            gaps.append(hist["final_gap"])
            violations.append(dual_metrics["dual_range_violation"])
            relative_violations.append(dual_metrics["relative_dual_range_violation"])
            z_norms.append(dual_metrics["z_norm_sq"])
            forward_calls.append(hist["forward_calls"])
            mean_batch_sizes.append(hist["mean_update_batch_size"])
            clipped_fractions.append(hist["clipped_fraction"])

        out["B_final"].append(int(B_final))
        out["final_gap_mean"].append(float(np.mean(gaps)))
        out["final_gap_std"].append(float(np.std(gaps)))
        out["dual_range_violation_mean"].append(float(np.mean(violations)))
        out["dual_range_violation_std"].append(float(np.std(violations)))
        out["relative_dual_range_violation_mean"].append(float(np.mean(relative_violations)))
        out["relative_dual_range_violation_std"].append(float(np.std(relative_violations)))
        out["z_norm_sq_mean"].append(float(np.mean(z_norms)))
        out["z_norm_sq_std"].append(float(np.std(z_norms)))
        out["forward_calls_mean"].append(float(np.mean(forward_calls)))
        out["forward_calls_std"].append(float(np.std(forward_calls)))
        out["mean_update_batch_size_mean"].append(float(np.mean(mean_batch_sizes)))
        out["mean_update_batch_size_std"].append(float(np.std(mean_batch_sizes)))
        out["clipped_fraction_mean"].append(float(np.mean(clipped_fractions)))
        out["clipped_fraction_std"].append(float(np.std(clipped_fractions)))

    return out


def summarize_csmri_trials(trial_results: List[Dict]) -> Dict:
    methods = [k for k in trial_results[0].keys() if k != "problem"]
    out = {}

    for method in methods:
        psnr_vals = [tr[method]["psnr"] for tr in trial_results]
        ssim_vals = [tr[method]["ssim"] for tr in trial_results]
        fwd_vals = [tr[method]["forward_calls"][-1] for tr in trial_results]

        out[method] = {
            "psnr_mean": float(np.mean(psnr_vals)),
            "psnr_std": float(np.std(psnr_vals)),
            "ssim_mean": float(np.mean(ssim_vals)),
            "ssim_std": float(np.std(ssim_vals)),
            "forward_calls_mean": float(np.mean(fwd_vals)),
            "forward_calls_std": float(np.std(fwd_vals)),
        }

    return out


def run_csmri_trial(
    cfg,
    trial_seed: int,
    keep_images: bool = True,
    verbose: bool = False,
) -> Dict:
    set_random_seed(trial_seed)
    problem = build_csmri_problem(cfg.problem)
    operator = problem["operator"]
    b = problem["b"]
    x_true = problem["x_true"]

    results = {}

    operator.reset_counters()
    results["AFLBreI"] = run_AFLBreI(
        operator,
        b,
        x_true,
        cfg.AFLBreI,
        support_tol=cfg.support_tol,
        verbose=verbose,
    )

    operator.reset_counters()
    results["Oracle-LBreI"] = run_oracle_lbrei(
        operator,
        b,
        x_true,
        cfg.oracle,
        support_tol=cfg.support_tol,
    )

    for hist in results.values():
        if hist.get("x_final") is not None:
            img_pred = operator._array_to_img(hist["x_final"])
            img_metrics = compute_image_metrics(img_pred, problem["img_true"])
            hist["psnr"] = img_metrics["psnr"]
            hist["ssim"] = img_metrics["ssim"]
            if keep_images:
                hist["img_pred"] = img_pred

    results["problem"] = {
        "img_true": problem["img_true"] if keep_images else None,
        "mask": problem["mask"] if keep_images else None,
        "phantom_name": getattr(cfg.problem, "phantom_name", "shepp_logan"),
        "rays": cfg.problem.rays,
        "snr_db": cfg.problem.snr_db,
    }
    return results


def _run_single_rays(rays, cfg):
    cfg_local = copy.deepcopy(cfg)
    cfg_local.problem.rays = rays
    trials = []
    for t in range(cfg_local.num_trials):
        res = run_csmri_trial(
            cfg_local,
            trial_seed=cfg_local.problem.seed + t,
            keep_images=False,
            verbose=False,
        )
        trials.append(res)
    return f"rays={rays}", summarize_csmri_trials(trials)


def run_csmri_sampling_sweep(cfg, rays_list=(16, 24, 30, 40)) -> Dict:
    out = {}
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(_run_single_rays, r, cfg) for r in rays_list]
        for future in futures:
            key, value = future.result()
            out[key] = value
    return out


def _run_single_snr(snr, cfg):
    cfg_local = copy.deepcopy(cfg)
    cfg_local.problem.snr_db = snr
    cfg_local.problem.noise_sigma = None

    trials = []
    for t in range(cfg_local.num_trials):
        res = run_csmri_trial(
            cfg_local,
            trial_seed=cfg_local.problem.seed + t,
            keep_images=False,
            verbose=False,
        )
        trials.append(res)

    label = f"SNR={snr}dB" if snr is not None else "SNR=inf"
    return label, summarize_csmri_trials(trials)


def run_csmri_noise_sweep(cfg, snr_list=(20.0, 30.0, 40.0, None)) -> Dict:
    out = {}
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(_run_single_snr, snr, cfg) for snr in snr_list]
        for future in futures:
            key, value = future.result()
            out[key] = value
    return out


def run_deconv_trial(cfg, trial_seed: int) -> Dict:
    set_random_seed(trial_seed)

    problem = build_deconv1d_problem(cfg.problem)
    operator = problem["operator"]
    b = problem["b"]
    x_true = problem["x_true"]

    results = {}

    operator.reset_counters()
    results["AFLBreI"] = run_AFLBreI(
        operator, b, x_true, cfg.AFLBreI, support_tol=cfg.support_tol
    )

    operator.reset_counters()
    results["Oracle-LBreI"] = run_oracle_lbrei(
        operator, b, x_true, cfg.oracle, support_tol=cfg.support_tol
    )

    operator.reset_counters()
    results["SGDAS"] = run_sgdas(
        operator, b, x_true, cfg.sgdas, support_tol=cfg.support_tol
    )

    operator.reset_counters()
    results["RD"] = run_rd(
        operator, b, x_true, cfg.rd, support_tol=cfg.support_tol
    )

    results["problem"] = {
        "x_true": x_true,
        "b": b,
        "kernel": problem["kernel"],
    }
    return results


def run_deconv_compare(cfg) -> Dict:
    trial_results = []
    for t in range(cfg.num_trials):
        trial_results.append(run_deconv_trial(cfg, trial_seed=cfg.problem.seed + t))

    summary = {}
    for method in ["AFLBreI", "Oracle-LBreI", "SGDAS", "RD"]:
        summary[method] = average_histories([tr[method] for tr in trial_results])

    summary["num_trials"] = cfg.num_trials
    return {"trial_results": trial_results, "summary": summary}
