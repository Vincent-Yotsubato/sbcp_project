import argparse
import os

import matplotlib.pyplot as plt

from config import get_experiment_config, config_to_dict
from experiments import (
    run_main_compare,
    run_batch_ablation,
    run_probe_ablation,
    run_stepsize_ablation,
    run_growing_batch_kkt_experiment,
    run_sparsity_scaling_experiment,
    run_noise_robustness_experiment,
    run_csmri_trial,
    run_csmri_sampling_sweep,
    run_csmri_noise_sweep,
    run_deconv_compare,
)
from plotting import (
    plot_method_curves,
    plot_ablation_curves,
    plot_stem_comparison,
    plot_csmri_metric_sweep,
    plot_growing_batch_kkt_metric,
    plot_stem_comparison_flexible,
    plot_mri_reconstruction,
)
from utils import ensure_dir, save_json


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 10,
        "lines.linewidth": 1.5,
        "figure.dpi": 300,
    }
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, default="main_compare")
    parser.add_argument("--outdir", type=str, default="results")
    return parser.parse_args()


def print_final_metrics(summary):
    for method in ["AFLBreI", "Oracle-LBreI", "SGDAS", "RD"]:
        stats = summary[method]
        print(method)
        print("final residual =", stats["residual_mean"][-1])
        print("final rel_error =", stats["rel_error_mean"][-1])
        print("final precision =", stats["precision_mean"][-1])
        print("final recall =", stats["recall_mean"][-1])
        print("final f1 =", stats["f1_mean"][-1])
        print("final tp =", stats["tp_mean"][-1])
        print("final fp =", stats["fp_mean"][-1])
        print("final fn =", stats["fn_mean"][-1])
        print("final nnz =", stats["nnz_mean"][-1])
        print()


def print_best_f1_info(result):
    if len(result["trial_results"]) == 0:
        return

    trial0 = result["trial_results"][0]
    print("=== Best F1 snapshot (trial 0) ===")
    for method in ["AFLBreI", "Oracle-LBreI", "SGDAS", "RD"]:
        hist = trial0[method]
        print(method)
        print("best f1 =", hist["best_f1"])
        print("best iter =", hist["best_iter_by_f1"])
        print()


def save_best_f1_vectors(result, raw_dir):
    if len(result["trial_results"]) == 0:
        return

    trial0 = result["trial_results"][0]
    out = {
        "x_true": trial0["problem"]["x_true"].tolist(),
        "AFLBreI_best_x_by_f1": None
        if trial0["AFLBreI"]["best_x_by_f1"] is None
        else trial0["AFLBreI"]["best_x_by_f1"].tolist(),
        "Oracle_LBreI_best_x_by_f1": None
        if trial0["Oracle-LBreI"]["best_x_by_f1"] is None
        else trial0["Oracle-LBreI"]["best_x_by_f1"].tolist(),
        "SGDAS_best_x_by_f1": None
        if trial0["SGDAS"]["best_x_by_f1"] is None
        else trial0["SGDAS"]["best_x_by_f1"].tolist(),
        "RD_best_x_by_f1": None
        if trial0["RD"]["best_x_by_f1"] is None
        else trial0["RD"]["best_x_by_f1"].tolist(),
    }
    save_json(os.path.join(raw_dir, "best_f1_vectors.json"), out)


def save_summary(raw_dir, filename, cfg, summary):
    save_json(
        os.path.join(raw_dir, filename),
        {
            "config": config_to_dict(cfg),
            "summary": summary,
        },
    )


def run_main_compare_command(cfg, raw_dir, fig_dir):
    result = run_main_compare(cfg)
    save_summary(raw_dir, "main_compare.json", cfg, result["summary"])

    summary = result["summary"]
    print_final_metrics(summary)
    print_best_f1_info(result)

    plot_method_curves(
        summary,
        "residual",
        "Residual Decay",
        "||Ax-b||",
        save_path=os.path.join(fig_dir, "main_residual.png"),
        log_y=True,
    )
    plot_method_curves(
        summary,
        "rel_error",
        "Relative Reconstruction Error",
        "RE",
        save_path=os.path.join(fig_dir, "main_rel_error.png"),
        log_y=True,
    )
    plot_method_curves(
        summary,
        "f1",
        "Support Recovery F1",
        "F1-score",
        save_path=os.path.join(fig_dir, "main_f1.png"),
    )
    plot_method_curves(
        summary,
        "precision",
        "Precision",
        "Precision",
        save_path=os.path.join(fig_dir, "main_precision.png"),
    )
    plot_method_curves(
        summary,
        "recall",
        "Recall",
        "Recall",
        save_path=os.path.join(fig_dir, "main_recall.png"),
    )
    plot_method_curves(
        summary,
        "nnz",
        "Sparsity Evolution",
        "nnz",
        save_path=os.path.join(fig_dir, "main_nnz.png"),
    )

    trial0 = result["trial_results"][0]
    plot_stem_comparison(
        x_true=trial0["problem"]["x_true"],
        x_AFLBreI=trial0["AFLBreI"]["x_final"],
        x_oracle=trial0["Oracle-LBreI"]["x_final"],
        x_sgdas=trial0["SGDAS"]["x_final"],
        x_rd=trial0["RD"]["x_final"],
        save_path=os.path.join(fig_dir, "main_stem.png"),
    )

    if trial0["AFLBreI"]["best_x_by_f1"] is not None:
        plot_stem_comparison(
            x_true=trial0["problem"]["x_true"],
            x_AFLBreI=trial0["AFLBreI"]["best_x_by_f1"],
            x_oracle=trial0["Oracle-LBreI"]["best_x_by_f1"],
            x_sgdas=trial0["SGDAS"]["best_x_by_f1"],
            x_rd=trial0["RD"]["best_x_by_f1"],
            save_path=os.path.join(fig_dir, "main_stem_best_f1.png"),
        )

    save_best_f1_vectors(result, raw_dir)


def run_ablation_command(cfg, raw_dir, fig_dir, kind):
    if kind == "batch":
        result = run_batch_ablation(cfg, max_budget=136000)
        filename = "ablation_batch.json"
        title = "Batch Size Ablation"
        prefix = "ablation_batch"
        plot_budget = 136000
    elif kind == "probe":
        result = run_probe_ablation(cfg, max_budget=136000)
        filename = "ablation_probe.json"
        title = "Probe Batch Size Ablation"
        prefix = "ablation_probe"
        plot_budget = 136000
    else:
        result = run_stepsize_ablation(cfg)
        filename = "ablation_stepsize.json"
        title = "AFLBreI Beta Ablation"
        prefix = "ablation_stepsize"
        plot_budget = None

    save_summary(raw_dir, filename, cfg, result)

    for name, stats in result.items():
        print(name)
        print("final residual =", stats["residual_mean"][-1])
        print("final rel_error =", stats["rel_error_mean"][-1])
        print("final precision =", stats["precision_mean"][-1])
        print("final recall =", stats["recall_mean"][-1])
        print("final f1 =", stats["f1_mean"][-1])
        print("final nnz =", stats["nnz_mean"][-1])
        print()

    metrics = [
        ("residual", "||Ax-b||", True),
        ("rel_error", "RE", True),
        ("f1", "F1-score", False),
        ("nnz", "nnz", False),
    ]
    if kind == "stepsize":
        metrics = [
            ("f1", "F1-score", False),
            ("nnz", "nnz", False),
            ("rel_error", "RE", True),
        ]

    for metric, ylabel, log_y in metrics:
        plot_ablation_curves(
            result,
            metric,
            title,
            ylabel,
            save_path=os.path.join(fig_dir, f"{prefix}_{metric}.png"),
            log_y=log_y,
            max_budget=plot_budget,
        )


def run_sparsity_command(cfg, raw_dir, fig_dir):
    result = run_sparsity_scaling_experiment(cfg)
    save_summary(raw_dir, "sparsity_scaling.json", cfg, result)

    plot_ablation_curves(
        result,
        "f1",
        "Sparsity Scaling Experiment",
        "F1-score",
        save_path=os.path.join(fig_dir, "sparsity_scaling_f1.png"),
    )
    plot_ablation_curves(
        result,
        "nnz",
        "Sparsity Scaling Experiment",
        "nnz",
        save_path=os.path.join(fig_dir, "sparsity_scaling_nnz.png"),
    )
    plot_ablation_curves(
        result,
        "rel_error",
        "Sparsity Scaling Experiment",
        "RE",
        save_path=os.path.join(fig_dir, "sparsity_scaling_rel_error.png"),
        log_y=True,
    )


def run_noise_command(cfg, raw_dir, fig_dir):
    result = run_noise_robustness_experiment(cfg, snr_levels=(20, 30, 40, None))
    save_summary(raw_dir, "noise_robustness.json", cfg, result)

    for name, stats in result.items():
        print(name)
        print("  final f1 =", stats["f1_mean"][-1])
        print("  final rel_error =", stats["rel_error_mean"][-1])
        print()

    plot_ablation_curves(
        result,
        "f1",
        "Noise Robustness: F1-score",
        "F1-score",
        save_path=os.path.join(fig_dir, "noise_robustness_f1.png"),
    )
    plot_ablation_curves(
        result,
        "rel_error",
        "Noise Robustness: Relative Error",
        "RE",
        save_path=os.path.join(fig_dir, "noise_robustness_rel_error.png"),
        log_y=True,
    )


def run_growing_batch_kkt_command(cfg, raw_dir, fig_dir):
    result = run_growing_batch_kkt_experiment(cfg)
    save_summary(raw_dir, "growing_batch_kkt.json", cfg, result)

    relative_violations = result.get("relative_dual_range_violation_mean")
    if relative_violations is None:
        relative_violations = [None] * len(result["B_final"])

    for B, gap, violation, relative_violation in zip(
        result["B_final"],
        result["final_gap_mean"],
        result["dual_range_violation_mean"],
        relative_violations,
    ):
        print(f"B_K={B}")
        print("  final least-squares gap =", gap)
        print("  dist(z_K, Range(A^T))^2 =", violation)
        if relative_violation is not None:
            print("  normalized dist^2 / ||z_K||^2 =", relative_violation)
        print()

    plot_growing_batch_kkt_metric(
        result,
        "final_gap",
        "Final Least-Squares Gap",
        r"$f(x_K)-f^\star$",
        save_path=os.path.join(fig_dir, "growing_batch_kkt_final_gap.png"),
    )
    plot_growing_batch_kkt_metric(
        result,
        "dual_range_violation",
        "Outer Dual Range Violation",
        r"$\mathrm{dist}(z_K,\mathrm{Range}(A^\top))^2$",
        save_path=os.path.join(fig_dir, "growing_batch_kkt_dual_range_violation.png"),
    )
    plot_growing_batch_kkt_metric(
        result,
        "relative_dual_range_violation",
        "Normalized Outer Dual Range Violation",
        r"$\mathrm{dist}(z_K,\mathrm{Range}(A^\top))^2/\|z_K\|^2$",
        save_path=os.path.join(
            fig_dir,
            "growing_batch_kkt_relative_dual_range_violation.png",
        ),
    )


def run_csmri_compare_command(cfg, raw_dir, fig_dir):
    print(f"Running CSMRI experiment (img_size={cfg.problem.img_size})...")
    result = run_csmri_trial(cfg, trial_seed=cfg.problem.seed)

    plot_mri_reconstruction(
        result["problem"]["img_true"],
        result["problem"]["mask"],
        result,
        save_path=os.path.join(fig_dir, "csmri_reconstruction.png"),
    )

    for method in ["AFLBreI", "Oracle-LBreI"]:
        if method in result and "img_pred" in result[method]:
            del result[method]["img_pred"]
    del result["problem"]["img_true"]
    del result["problem"]["mask"]

    save_json(
        os.path.join(raw_dir, "csmri_compare.json"),
        {"config": config_to_dict(cfg), "result": result},
    )


def run_csmri_sweep_command(cfg, raw_dir, fig_dir, kind):
    if kind == "sampling":
        result = run_csmri_sampling_sweep(cfg, rays_list=cfg.csmri_sweep.rays_list)
        filename = "csmri_sampling_sweep.json"
        prefix = "csmri_sampling"
    else:
        result = run_csmri_noise_sweep(cfg, snr_list=cfg.csmri_sweep.snr_list)
        filename = "csmri_noise_sweep.json"
        prefix = "csmri_noise"

    save_summary(raw_dir, filename, cfg, result)

    plot_csmri_metric_sweep(
        result,
        "psnr",
        save_path=os.path.join(fig_dir, f"{prefix}_psnr.png"),
    )
    plot_csmri_metric_sweep(
        result,
        "ssim",
        save_path=os.path.join(fig_dir, f"{prefix}_ssim.png"),
    )

    for case_name, stats in result.items():
        print(case_name)
        for method in ["AFLBreI", "Oracle-LBreI"]:
            print(
                f"  {method}: "
                f"PSNR={stats[method]['psnr_mean']:.3f}+/-{stats[method]['psnr_std']:.3f}, "
                f"SSIM={stats[method]['ssim_mean']:.4f}+/-{stats[method]['ssim_std']:.4f}"
            )
        print()


def run_deconv_command(cfg, raw_dir, fig_dir):
    result = run_deconv_compare(cfg)
    save_summary(raw_dir, "deconv_compare.json", cfg, result["summary"])

    summary = result["summary"]
    for method in ["AFLBreI", "Oracle-LBreI", "SGDAS", "RD"]:
        stats = summary[method]
        print(method)
        print("final residual =", stats["residual_mean"][-1])
        print("final rel_error =", stats["rel_error_mean"][-1])
        print("final precision =", stats["precision_mean"][-1])
        print("final recall =", stats["recall_mean"][-1])
        print("final f1 =", stats["f1_mean"][-1])
        print("final nnz =", stats["nnz_mean"][-1])
        print()

    summary_plot = {
        k: v for k, v in summary.items() if k in ["AFLBreI", "Oracle-LBreI", "SGDAS"]
    }
    plot_method_curves(
        summary_plot,
        "f1",
        "1D Sparse Deconvolution: F1-score",
        "F1-score",
        save_path=os.path.join(fig_dir, "deconv_f1.png"),
    )

    trial0 = result["trial_results"][0]
    series = [
        ("Truth", trial0["problem"]["x_true"]),
        ("AFLBreI", trial0["AFLBreI"]["best_x_by_f1"]),
        ("Oracle", trial0["Oracle-LBreI"]["best_x_by_f1"]),
        ("SGDAS", trial0["SGDAS"]["best_x_by_f1"]),
    ]
    plot_stem_comparison_flexible(
        series,
        save_path=os.path.join(fig_dir, "deconv_stem_best_f1.png"),
    )


def main():
    args = parse_args()
    cfg = get_experiment_config(args.exp)

    raw_dir = os.path.join(args.outdir, "raw")
    fig_dir = os.path.join(args.outdir, "figures", args.exp)
    ensure_dir(raw_dir)
    ensure_dir(fig_dir)

    if args.exp == "main_compare":
        run_main_compare_command(cfg, raw_dir, fig_dir)
    elif args.exp == "ablation_batch":
        run_ablation_command(cfg, raw_dir, fig_dir, kind="batch")
    elif args.exp == "ablation_probe":
        run_ablation_command(cfg, raw_dir, fig_dir, kind="probe")
    elif args.exp == "ablation_stepsize":
        run_ablation_command(cfg, raw_dir, fig_dir, kind="stepsize")
    elif args.exp == "sparsity_scaling":
        run_sparsity_command(cfg, raw_dir, fig_dir)
    elif args.exp == "noise_robustness":
        run_noise_command(cfg, raw_dir, fig_dir)
    elif args.exp == "growing_batch_kkt":
        run_growing_batch_kkt_command(cfg, raw_dir, fig_dir)
    elif args.exp == "csmri_compare":
        run_csmri_compare_command(cfg, raw_dir, fig_dir)
    elif args.exp == "csmri_sampling_sweep":
        run_csmri_sweep_command(cfg, raw_dir, fig_dir, kind="sampling")
    elif args.exp == "csmri_noise_sweep":
        run_csmri_sweep_command(cfg, raw_dir, fig_dir, kind="noise")
    elif args.exp == "deconv_compare":
        run_deconv_command(cfg, raw_dir, fig_dir)
    else:
        raise ValueError(f"Unknown experiment: {args.exp}")

    print(f"Done: {args.exp}")


if __name__ == "__main__":
    main()
