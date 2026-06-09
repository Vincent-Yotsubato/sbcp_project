# AFLBreI Project

## Overview
Implementation of the Adjoint-Free Linearized Bregman Iteration (AFLBreI) for adjoint-free sparse inverse problems.

AFLBreI uses a fully adjoint-free sample-splitting Polyak step rule. Legacy adaptive comparison entries have been removed from the active experiment pipeline.

## Project Structure
- main.py: entry point
- config.py: experiment configs
- operators.py: forward/adjoint operator abstraction
- problems.py: data generation
- estimators.py: adjoint-free stochastic gradient and q estimators
- regularizers.py: Elastic Net mirror map
- algorithms.py: AFLBreI / Oracle-LBreI / SGDAS / RD
- metrics.py: residual / RE / support metrics
- experiments.py: experiment drivers
- plotting.py: figures
- utils.py: helpers

## Installation
```bash
pip install -r requirements.txt
```

## Run
Run commands and experiment meanings:

Figures are saved under `results/figures/<experiment_name>/`, while raw JSON outputs are saved under `results/raw/`.

```bash
python main.py --exp main_compare
```
Main sparse recovery comparison. Compares AFLBreI with Oracle-LBreI, SGDAS, and RD on the default synthetic sparse inverse problem.

Current configuration:
- Problem: Gaussian synthetic sparse recovery, `m=600`, `n=1200`, sparsity `s=20`, `snr_db=40`, normalized columns, `num_trials=10`, `support_tol=1e-2`.
- AFLBreI: `lambda=3.8`, `mu=1.0`, `K=2600`, update batch `B=32`, probe batch `M=8`, `beta=0.99`, Gaussian directions, `f_star=0`, `clip_step=False`, `growing_batch=False`, `return_average=False`, `record_every=1`.
- Forward-call budget: AFLBreI uses approximately `2600 * (1 + 32 + 8) = 106600` forward calls.
- Baselines: Oracle-LBreI uses `K=106600`, `lambda=3.8`, `beta=0.99`, exact-gradient Polyak step, `record_every=1`; SGDAS and RD use `K=53300`, `record_every=1`, matching the same forward-call budget because they use about two forward calls per iteration.

```bash
python main.py --exp ablation_batch
```
Batch-size ablation for AFLBreI. Varies the AFLBreI update batch size `B` and compares recovery metrics under a fixed forward-evaluation budget.

Current configuration:
- Problem: same synthetic setting as `main_compare`, with `num_trials=10`.
- Varied parameter: update batch `B in {1, 16, 32, 64}`.
- Fixed AFLBreI parameters: `lambda=3.8`, `M=8`, `beta=1.0`, Gaussian directions, `clip_step=False`.
- Budget rule: `max_budget=136000` forward calls approximately, with `K = floor(136000 / (1 + B + M))` for each `B`.
- Recording: `record_every=20`.

```bash
python main.py --exp ablation_probe
```
Probe batch-size ablation for AFLBreI. Varies the probe batch size `M` (`q_batch_size`) used to estimate the sample-splitting Polyak denominator, under a fixed forward-evaluation budget.

Current configuration:
- Problem: same synthetic setting as `main_compare`, with `num_trials=10`.
- Varied parameter: probe batch `M in {1, 4, 8, 16, 32, 64}`.
- Fixed update batch: `B=32`.
- Beta rule: `beta(M) = max(0.25, 2 - 8/M)`, so the tested values are `0.25, 0.25, 1.0, 1.5, 1.75, 1.875`.
- Stability setting: `clip_step=True` only for this probe ablation.
- Budget rule: `max_budget=136000` forward calls approximately, with `K = floor(136000 / (1 + B + M))` for each `M`.
- Recording: `record_every=20`.

```bash
python main.py --exp ablation_stepsize
```
AFLBreI stepsize-scale ablation. Varies the sample-splitting Polyak scale parameter `beta`.

Current configuration:
- Problem: same synthetic setting as `main_compare`, with `num_trials=10`.
- Varied parameter: `beta in {0.25, 0.5, 0.8, 1.0, 1.5, 2.0}`.
- Fixed AFLBreI parameters: `lambda=3.8`, `K=2600`, `B=32`, `M=8`, Gaussian directions, `clip_step=False`.
- Recording: `record_every=10`.

```bash
python main.py --exp sparsity_scaling
```
Sparsity scaling experiment. Tests AFLBreI performance as the true signal sparsity level `s` changes.

Current configuration:
- Problem: Gaussian synthetic sparse recovery with `m=600`, `n=1200`, `snr_db=40`, normalized columns, `num_trials=10`.
- Varied parameter: sparsity `s in {10, 20, 30, 40}`.
- AFLBreI: `lambda=3.8`, `K=2600`, `B=32`, `M=8`, `beta=1.0`, Gaussian directions, `clip_step=False`.
- Recording: `record_every=10`.

```bash
python main.py --exp noise_robustness
```
Noise robustness experiment. Tests AFLBreI under different measurement SNR levels, including a noiseless case.

Current configuration:
- Problem: Gaussian synthetic sparse recovery with `m=600`, `n=1200`, sparsity `s=20`, normalized columns, `num_trials=10`.
- Varied parameter: `snr_db in {20, 30, 40, None}`, where `None` is noiseless.
- AFLBreI: `lambda=3.8`, `K=2600`, `B=32`, `M=8`, `beta=1.0`, Gaussian directions, `clip_step=False`.
- Recording: `record_every=10`.

```bash
python main.py --exp growing_batch_kkt
```
Growing-batch KKT consistency diagnostic. Runs noiseless synthetic sparse recovery with `b = A x_true` and `f_star = 0`, and plots final least-squares gap plus `dist(z_K, Range(A^T))^2`.

Current configuration:
- Problem: noiseless Gaussian synthetic sparse recovery, `m=256`, `n=512`, sparsity `s=20`, `snr_db=None`, `num_trials=5`.
- AFLBreI: `lambda=3.8`, `K=2000`, fixed probe batch `M=20`, `beta=1.0`, Gaussian directions, `f_star=0`, configured `record_every=2000`.
- Varied parameter: final growing update batch `B_K in {50, 100, 200, 500}`.
- Growing-batch schedule: update batch grows linearly from about `0.5 B_K` to `B_K`.
- Stability setting: `clip_step=True`.
- Recording/output: this custom diagnostic records final metrics only for each `B_K`; main outputs are final least-squares gap, absolute dual range violation, and relative dual range violation.

```bash
python main.py --exp csmri_compare
```
Single CS-MRI reconstruction experiment. Compares AFLBreI and Oracle-LBreI on one compressed-sensing MRI reconstruction task.

Current configuration:
- Problem: Shepp-Logan CS-MRI, `img_size=128`, radial rays `30`, Haar wavelet, `snr_db=40`, `num_trials=1`.
- AFLBreI: `lambda=1.0`, `K=30000`, update batch `B=128`, probe batch `M=32`, `beta=1.0`, `record_every=500`.
- Oracle-LBreI: `lambda=1.0`, `K=20000`, `beta=1.0`, exact-gradient Polyak step, `record_every=1`.

```bash
python main.py --exp csmri_sampling_sweep
```
CS-MRI sampling-rate sensitivity experiment. Varies the number of radial sampling rays and reports reconstruction quality metrics such as PSNR and SSIM.

Current configuration:
- Problem: Shepp-Logan CS-MRI, `img_size=128`, Haar wavelet, `snr_db=40`, `num_trials=3`.
- Varied parameter: radial rays `rays in {16, 24, 30, 40}`.
- AFLBreI: `lambda=1.0`, `K=20000`, update batch `B=100`, probe batch `M=16`, `beta=1.0`, `record_every=500`.
- Oracle-LBreI: `lambda=1.0`, `K=20000`, `beta=1.0`, exact-gradient Polyak step, `record_every=1`.

```bash
python main.py --exp csmri_noise_sweep
```
CS-MRI noise sensitivity experiment. Varies the measurement SNR and reports PSNR/SSIM for each method.

Current configuration:
- Problem: Shepp-Logan CS-MRI, `img_size=128`, radial rays `30`, Haar wavelet, `num_trials=3`.
- Varied parameter: `snr_db in {20, 30, 40, None}`, where `None` is noiseless.
- AFLBreI: `lambda=1.0`, `K=20000`, update batch `B=100`, probe batch `M=16`, `beta=1.0`, `record_every=500`.
- Oracle-LBreI: `lambda=1.0`, `K=20000`, `beta=1.0`, exact-gradient Polyak step, `record_every=1`.

```bash
python main.py --exp deconv_compare
```
1D sparse deconvolution comparison. Compares AFLBreI with Oracle-LBreI, SGDAS, and RD on a blurred sparse spike recovery problem.

Current configuration:
- Problem: 1D sparse deconvolution, `n=1024`, sparsity `s=30`, Gaussian blur kernel size `31`, `blur_sigma=3.0`, `snr_db=40`, Gaussian signal amplitudes, `num_trials=10`, `support_tol=1e-1`.
- AFLBreI: `lambda=10`, `K=20000`, final update batch `B_K=256`, initial update batch about `0.5 B_K`, probe batch lower bound `M=64`, `probe_batch_ratio=0.5`, `beta=1.0`, `f_star=0`, `record_every=10`.
- AFLBreI stability/schedule: `step_safety=0.75`, `clip_step=True`, `growing_batch=True`, `grow_probe_with_batch=True`.
- Oracle-LBreI: `lambda=675`, `beta=1.0`, exact-gradient Polyak step, `K=160000`, `record_every=1`.
- SGDAS and RD: `K=10`, `record_every=1`.
