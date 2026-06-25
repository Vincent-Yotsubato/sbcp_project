import unittest
from types import SimpleNamespace

import numpy as np
from scipy.signal import convolve

from algorithms import run_AFLBreI
from config import get_experiment_config
from estimators import (
    estimate_gradient_and_q_batch,
    estimate_gradient_batch,
    estimate_q_from_residual,
)
from experiments import run_main_compare
from operators import Convolution1DOperator, MatrixOperator


class CountingMatrixOperator(MatrixOperator):
    def __post_init__(self):
        super().__post_init__()
        self.spectral_norm_calls = 0

    def spectral_norm_estimate(self) -> float:
        self.spectral_norm_calls += 1
        return super().spectral_norm_estimate()


class SpeedupInvariantTests(unittest.TestCase):
    def test_fused_gradient_and_q_matches_split_estimators(self):
        rng = np.random.default_rng(123)
        A = rng.normal(size=(5, 9))
        x = rng.normal(size=9)
        b = rng.normal(size=5)
        B = 4
        M = 3

        op_split = MatrixOperator(A.copy())
        np.random.seed(99)
        g_split, residual_split, gradient_calls = estimate_gradient_batch(
            op_split,
            x,
            b,
            batch_size=B,
            sampler="gaussian",
        )
        q_split, q_calls = estimate_q_from_residual(
            op_split,
            residual_split,
            n=x.shape[0],
            batch_size=M,
            sampler="gaussian",
        )

        op_fused = MatrixOperator(A.copy())
        np.random.seed(99)
        g_fused, residual_fused, q_fused, fused_calls = estimate_gradient_and_q_batch(
            op_fused,
            x,
            b,
            update_batch_size=B,
            probe_batch_size=M,
            sampler="gaussian",
        )

        np.testing.assert_allclose(residual_fused, residual_split)
        np.testing.assert_allclose(g_fused, g_split)
        self.assertAlmostEqual(q_fused, q_split)
        self.assertEqual(fused_calls, gradient_calls + q_calls)
        self.assertEqual(op_fused.forward_calls, op_split.forward_calls)

    def test_convolution_forward_batch_matches_rowwise_convolution(self):
        rng = np.random.default_rng(456)
        kernel = np.array([0.2, 0.6, 0.2])
        X = rng.normal(size=(6, 16))
        operator = Convolution1DOperator(kernel=kernel, n=16)

        expected = np.array([convolve(row, kernel, mode="same") for row in X])
        actual = operator.forward_batch(X)

        np.testing.assert_allclose(actual, expected)

    def test_safe_step_is_cached_for_unclipped_aflbrei(self):
        rng = np.random.default_rng(789)
        A = rng.normal(size=(4, 6))
        x_true = np.zeros(6)
        x_true[[1, 4]] = [1.0, -0.8]
        b = A @ x_true
        operator = CountingMatrixOperator(A)
        cfg = SimpleNamespace(
            lam=0.1,
            mu=1.0,
            num_iters=5,
            batch_size=2,
            sampler="gaussian",
            step_safety=0.5,
            record_every=2,
            return_average=False,
            beta=1.0,
            q_batch_size=2,
            f_star=0.0,
            eps_denom=1e-12,
            eps_gap=1e-12,
            clip_step=False,
            growing_batch=False,
            batch_floor_fraction=0.5,
            grow_probe_with_batch=False,
            probe_batch_ratio=1.0,
        )

        np.random.seed(321)
        history = run_AFLBreI(operator, b, x_true, cfg, support_tol=1e-3)

        self.assertEqual(len(history["safe_step_ref"]), 3)
        self.assertEqual(operator.spectral_norm_calls, 1)

    def test_parallel_main_compare_matches_serial_on_tiny_config(self):
        cfg = get_experiment_config("main_compare")
        cfg.num_trials = 2
        cfg.problem.m = 12
        cfg.problem.n = 24
        cfg.problem.s = 3
        cfg.problem.min_separation = 3
        cfg.AFLBreI.num_iters = 4
        cfg.AFLBreI.batch_size = 3
        cfg.AFLBreI.q_batch_size = 2
        cfg.AFLBreI.record_every = 1
        cfg.oracle.num_iters = 4
        cfg.oracle.record_every = 1
        cfg.sgdas.num_iters = 4
        cfg.sgdas.record_every = 1
        cfg.rd.num_iters = 4
        cfg.rd.record_every = 1

        serial = run_main_compare(cfg, workers=1)["summary"]
        parallel = run_main_compare(cfg, workers=2)["summary"]

        for method in ["AFLBreI", "Oracle-LBreI", "SGDAS", "RD"]:
            np.testing.assert_allclose(
                parallel[method]["residual_mean"],
                serial[method]["residual_mean"],
            )
            np.testing.assert_allclose(
                parallel[method]["rel_error_mean"],
                serial[method]["rel_error_mean"],
            )


if __name__ == "__main__":
    unittest.main()
