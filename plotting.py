import os

import matplotlib.pyplot as plt
import numpy as np

from utils import ensure_dir


def plot_method_curves(summary_dict, metric_key, title, ylabel, save_path=None, log_y=False):
    plt.figure(figsize=(6.5, 4.2))

    for method, stats in summary_dict.items():
        if method == "num_trials":
            continue
        x = np.asarray(stats["forward_calls_mean"])
        y = np.asarray(stats[f"{metric_key}_mean"])
        ystd = np.asarray(stats[f"{metric_key}_std"])

        plt.plot(x, y, label=method)
        y_lower = np.maximum(y - ystd, 1e-12 if log_y else -np.inf)
        y_upper = y + ystd
        plt.fill_between(x, y_lower, y_upper, alpha=0.12)

    plt.xlabel("Number of Forward Evaluations")
    plt.ylabel(ylabel)
    if log_y:
        plt.yscale("log")
    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True, fontsize=10)
    plt.grid(True, alpha=0.3)

    ax = plt.gca()
    ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    ax.xaxis.get_offset_text().set_fontsize(10)

    if save_path is not None:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_ablation_curves(
    ablation_dict,
    metric_key,
    title,
    ylabel,
    save_path=None,
    show_std=True,
    max_budget=None,
    log_y=False,
):
    plt.figure(figsize=(6.5, 4.2))
    common_x = np.linspace(0, max_budget, 500) if max_budget is not None else None

    for name, stats in ablation_dict.items():
        x = np.asarray(stats["forward_calls_mean"])
        y = np.asarray(stats[f"{metric_key}_mean"])
        ystd = np.asarray(stats.get(f"{metric_key}_std", np.zeros_like(y)))

        if max_budget is not None:
            mask = x <= max_budget
            x, y, ystd = x[mask], y[mask], ystd[mask]
            if len(x) > 1:
                y = np.interp(common_x, x, y)
                ystd = np.interp(common_x, x, ystd)
                x = common_x
            else:
                continue

        plt.plot(x, y, label=name)
        if show_std:
            y_lower = np.maximum(y - ystd, 1e-12 if log_y else -np.inf)
            plt.fill_between(x, y_lower, y + ystd, alpha=0.15)

    if max_budget is not None:
        plt.xlim(0, max_budget)

    plt.xlabel("Number of Forward Evaluations")
    plt.ylabel(ylabel)
    if log_y:
        plt.yscale("log")
    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True, fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    ax = plt.gca()
    ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    ax.xaxis.get_offset_text().set_fontsize(10)

    if save_path is not None:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_stem_comparison(x_true, x_AFLBreI, x_oracle, x_sgdas, x_rd, save_path=None):
    fig, axes = plt.subplots(5, 1, figsize=(10, 10), sharex=True)
    series = [
        ("Ground Truth", x_true),
        ("AFLBreI", x_AFLBreI),
        ("Oracle-LBreI", x_oracle),
        ("SGDAS", x_sgdas),
        ("RD", x_rd),
    ]

    for ax, (_, x) in zip(axes, series):
        ax.stem(np.arange(len(x)), x)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Index")
    plt.tight_layout()

    if save_path is not None:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_mri_reconstruction(img_true, mask, results_dict, save_path=None):
    methods = [m for m in ["AFLBreI", "Oracle-LBreI", "SGDAS"] if m in results_dict]
    fig, axes = plt.subplots(1, len(methods) + 2, figsize=(3.5 * (len(methods) + 2), 4))

    axes[0].imshow(img_true, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Ground Truth")
    axes[0].axis("off")

    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("Sampling Mask")
    axes[1].axis("off")

    for i, method in enumerate(methods):
        ax = axes[i + 2]
        img_pred = results_dict[method]["img_pred"]
        psnr_val = results_dict[method]["psnr"]
        ssim_val = results_dict[method]["ssim"]
        ax.imshow(np.clip(img_pred, 0, 1), cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"{method}\nPSNR: {psnr_val:.2f}dB | SSIM: {ssim_val:.3f}")
        ax.axis("off")

    plt.tight_layout()
    if save_path:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_csmri_metric_sweep(summary_dict, metric_key, save_path=None):
    case_names = list(summary_dict.keys())
    methods = [m for m in ["AFLBreI", "Oracle-LBreI"] if m in next(iter(summary_dict.values()))]

    x = np.arange(len(case_names))
    plt.figure(figsize=(6.5, 4.2))

    for method in methods:
        means = [summary_dict[c][method][f"{metric_key}_mean"] for c in case_names]
        stds = [summary_dict[c][method][f"{metric_key}_std"] for c in case_names]
        plt.errorbar(x, means, yerr=stds, marker="o", capsize=4, label=method)

    plt.xticks(x, case_names, rotation=20)
    plt.ylabel(metric_key.upper())
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=True)
    plt.tight_layout()

    if save_path is not None:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_growing_batch_kkt(result, save_path=None):
    B = np.asarray(result["B_final"], dtype=float)
    gap = np.asarray(result["final_gap_mean"], dtype=float)
    gap_std = np.asarray(result["final_gap_std"], dtype=float)
    violation = np.asarray(result["dual_range_violation_mean"], dtype=float)
    violation_std = np.asarray(result["dual_range_violation_std"], dtype=float)
    relative_violation = np.asarray(
        result.get("relative_dual_range_violation_mean", []),
        dtype=float,
    )
    relative_violation_std = np.asarray(
        result.get("relative_dual_range_violation_std", []),
        dtype=float,
    )

    eps = 1e-18
    has_relative_violation = relative_violation.size == B.size
    n_panels = 3 if has_relative_violation else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(4.5 * n_panels, 3.8))

    def _positive_errorbar(ax, x, y, ystd, **kwargs):
        y_plot = np.maximum(y, eps)
        lower = np.minimum(ystd, np.maximum(y_plot - eps, 0.0))
        yerr = np.vstack([lower, ystd])
        ax.errorbar(x, y_plot, yerr=yerr, **kwargs)

    _positive_errorbar(axes[0], B, gap, gap_std, marker="o", capsize=4)
    axes[0].set_title("(a) Final least-squares gap")
    axes[0].set_xlabel(r"$B_K$")
    axes[0].set_ylabel(r"$f(x_K)-f^\star$")
    axes[0].set_yscale("log")
    axes[0].grid(True, alpha=0.3, which="both")

    _positive_errorbar(axes[1], B, violation, violation_std, marker="o", capsize=4)
    axes[1].set_title(r"(b) Absolute range violation")
    axes[1].set_xlabel(r"$B_K$")
    axes[1].set_ylabel(r"$\mathrm{dist}(z_K,\mathrm{Range}(A^\top))^2$")
    axes[1].set_yscale("log")
    axes[1].grid(True, alpha=0.3, which="both")

    if has_relative_violation:
        _positive_errorbar(
            axes[2],
            B,
            relative_violation,
            relative_violation_std,
            marker="o",
            capsize=4,
        )
        axes[2].set_title(r"(c) Normalized range violation")
        axes[2].set_xlabel(r"$B_K$")
        axes[2].set_ylabel(
            r"$\mathrm{dist}(z_K,\mathrm{Range}(A^\top))^2/\|z_K\|^2$"
        )
        axes[2].set_yscale("log")
        axes[2].grid(True, alpha=0.3, which="both")

    fig.tight_layout()
    if save_path is not None:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_growing_batch_kkt_metric(
    result,
    metric_key,
    title,
    ylabel,
    save_path=None,
    log_y=True,
):
    B = np.asarray(result["B_final"], dtype=float)
    y = np.asarray(result[f"{metric_key}_mean"], dtype=float)
    ystd = np.asarray(result[f"{metric_key}_std"], dtype=float)

    eps = 1e-18
    y_plot = np.maximum(y, eps)
    lower = np.minimum(ystd, np.maximum(y_plot - eps, 0.0))
    yerr = np.vstack([lower, ystd])

    plt.figure(figsize=(6.5, 4.2))
    plt.errorbar(B, y_plot, yerr=yerr, marker="o", capsize=4, label="AFLBreI")
    plt.xlabel(r"$B_K$")
    plt.ylabel(ylabel)
    plt.title(title)
    if log_y:
        plt.yscale("log")
    plt.grid(True, alpha=0.3, which="both")
    plt.legend(frameon=True, fontsize=10)
    plt.tight_layout()

    if save_path is not None:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_stem_comparison_flexible(series: list[tuple[str, np.ndarray]], save_path=None):
    fig, axes = plt.subplots(len(series), 1, figsize=(10, 1.8 * len(series)), sharex=True)
    if len(series) == 1:
        axes = [axes]

    for ax, (title, x) in zip(axes, series):
        ax.stem(np.arange(len(x)), x)
        ax.set_ylabel(title, rotation=0, labelpad=30, va="center")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Index")
    plt.tight_layout()

    if save_path is not None:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
