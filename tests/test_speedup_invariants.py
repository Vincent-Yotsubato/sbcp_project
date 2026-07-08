import unittest
from types import SimpleNamespace

import numpy as np
from scipy.signal import convolve

from algorithms import get_sgdas_stepsize, run_AFLBreI, run_sgdas
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

    def test_sgdas_uses_theory_constant_step(self):
        A = np.array(
            [
                [1.0, 0.0, 0.5],
                [0.0, 2.0, 0.0],
            ]
        )
        x_true = np.array([1.0, 0.0, -1.0])
        b = A @ x_true
        operator = MatrixOperator(A)
        cfg = SimpleNamespace(
            num_iters=3,
            sampler="gaussian",
            record_every=1,
            step_rule="inv_sqrt",
            step_c0=999.0,
            step_power=0.5,
        )

        expected_step = get_sgdas_stepsize(operator, n=x_true.shape[0])
        np.random.seed(135)
        history = run_sgdas(operator, b, x_true, cfg, support_tol=1e-3)

        np.testing.assert_allclose(history["step_size"], expected_step)

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

    def test_recorded_residual_matches_recorded_aflbrei_state(self):
        rng = np.random.default_rng(246)
        A = rng.normal(size=(5, 7))
        x_true = np.zeros(7)
        x_true[[1, 5]] = [1.0, -0.7]
        b = A @ x_true
        operator = MatrixOperator(A)
        cfg = SimpleNamespace(
            lam=0.1,
            mu=1.0,
            num_iters=1,
            batch_size=2,
            sampler="gaussian",
            step_safety=0.5,
            record_every=1,
            return_average=False,
            beta=0.99,
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

        np.random.seed(135)
        history = run_AFLBreI(operator, b, x_true, cfg, support_tol=1e-3)

        residual_norm = np.linalg.norm(A @ history["x_final"] - b)
        self.assertAlmostEqual(history["residual"][0], residual_norm)
        self.assertEqual(history["forward_calls"][0], 1 + cfg.batch_size + cfg.q_batch_size)
        self.assertEqual(history["eval_forward_calls"][0], 1)

    def test_aflbrei_final_iterate_is_true_last_iterate_when_not_recorded(self):
        rng = np.random.default_rng(975)
        A = rng.normal(size=(5, 8))
        x_true = np.zeros(8)
        x_true[[2, 6]] = [0.9, -1.1]
        b = A @ x_true

        def make_cfg(record_every):
            return SimpleNamespace(
                lam=0.1,
                mu=1.0,
                num_iters=5,
                batch_size=2,
                sampler="gaussian",
                step_safety=0.5,
                record_every=record_every,
                return_average=False,
                beta=0.99,
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

        np.random.seed(864)
        full_history = run_AFLBreI(
            MatrixOperator(A.copy()),
            b,
            x_true,
            make_cfg(record_every=1),
            support_tol=1e-3,
        )

        np.random.seed(864)
        sparse_history = run_AFLBreI(
            MatrixOperator(A.copy()),
            b,
            x_true,
            make_cfg(record_every=10),
            support_tol=1e-3,
        )

        np.testing.assert_allclose(sparse_history["x_final"], full_history["x_final"])
        np.testing.assert_allclose(sparse_history["z_final"], full_history["z_final"])
        self.assertEqual(sparse_history["iter"][-1], 4)
        self.assertEqual(sparse_history["forward_calls"][-1], 5 * (1 + 2 + 2))

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
