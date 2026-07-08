from dataclasses import dataclass, field, asdict
from typing import Dict, Optional, Tuple


@dataclass
class ProblemConfig:
    m: int = 600
    n: int = 1200
    s: int = 20
    noise_sigma: Optional[float] = None
    snr_db: Optional[float] = 40.0
    matrix_dist: str = "gaussian"
    normalize_columns: bool = True
    signal_dist: str = "gaussian"
    min_separation: int = 8
    margin: int = 0
    seed: int = 42


@dataclass
class AFLBreIConfig:
    lam: float = 3.8
    mu: float = 1.0
    num_iters: int = 5200
    batch_size: int = 12
    sampler: str = "gaussian"
    step_rule: str = "constant"
    step_c0: float = 0.01
    step_power: float = 0.5
    step_safety: float = 0.5
    record_every: int = 1
    return_average: bool = False

    # AFLBreI sample-splitting Polyak step parameters.
    beta: float = 1.0
    q_batch_size: int = 8
    f_star: float = 0.0
    eps_denom: float = 1e-12
    eps_gap: float = 1e-12
    clip_step: bool = False
    growing_batch: bool = False
    batch_floor_fraction: float = 0.5
    grow_probe_with_batch: bool = False
    probe_batch_ratio: float = 1.0


@dataclass
class OracleLBreIConfig:
    lam: float = 3.8
    mu: float = 1.0
    num_iters: int = 109200
    step_rule: str = "constant"          # deprecated — Polyak step is used by default
    step_c0: float = 0.02                # deprecated — Polyak step is used by default
    record_every: int = 1

    # Exact-gradient Polyak step parameters.
    beta: float = 1.0
    f_star: float = 0.0
    eps_denom: float = 1e-12


@dataclass
class SGDASConfig:
    num_iters: int = 54600
    sampler: str = "gaussian"
    record_every: int = 1

    # Deprecated: SGDAS now uses the theory constant step
    # 1 / ((n + 2) * ||A||^2), computed from the current operator.
    step_rule: str = "theory_constant"
    step_c0: float = 0.02
    step_power: float = 0.5


@dataclass
class RDConfig:
    num_iters: int = 54600
    sampler: str = "gaussian"
    record_every: int = 1


@dataclass
class CSMRIProblemConfig:
    img_size: int = 128
    rays: int = 30
    wavelet: str = "haar"
    phantom_name: str = "shepp_logan"
    noise_sigma: Optional[float] = None
    snr_db: Optional[float] = 40.0
    seed: int = 42


@dataclass
class CSMRISweepConfig:
    rays_list: Tuple[int, ...] = (16, 24, 30, 40)
    snr_list: Tuple[Optional[float], ...] = (20.0, 30.0, 40.0, None)
    phantom_list: Tuple[str, ...] = ("shepp_logan",)


@dataclass
class Deconv1DProblemConfig:
    n: int = 1024
    s: int = 30
    kernel_size: int = 31
    blur_sigma: float = 3.0
    noise_sigma: Optional[float] = None
    snr_db: Optional[float] = 40.0
    signal_dist: str = "gaussian"
    seed: int = 42


@dataclass
class ExperimentConfig:
    name: str = "main_compare"
    num_trials: int = 10
    support_tol: float = 1e-2
    save_raw: bool = True
    save_figures: bool = True
    problem: ProblemConfig = field(default_factory=ProblemConfig)
    AFLBreI: AFLBreIConfig = field(default_factory=AFLBreIConfig)
    oracle: OracleLBreIConfig = field(default_factory=OracleLBreIConfig)
    sgdas: SGDASConfig = field(default_factory=SGDASConfig)
    rd: RDConfig = field(default_factory=RDConfig)
    csmri_sweep: CSMRISweepConfig = field(default_factory=CSMRISweepConfig)


def get_default_config() -> ExperimentConfig:
    return ExperimentConfig()


def get_experiment_config(exp_name: str) -> ExperimentConfig:
    cfg = get_default_config()
    cfg.name = exp_name

    if exp_name == "main_compare":
        cfg.AFLBreI.num_iters = 2600
        cfg.AFLBreI.batch_size = 32
        cfg.AFLBreI.beta = 0.99
        cfg.oracle.beta = 0.99
        cfg.oracle.num_iters = 106600
        cfg.sgdas.num_iters = 53300
        cfg.rd.num_iters = 53300
        return cfg

    if exp_name == "ablation_batch":
        cfg.num_trials = 10
        cfg.AFLBreI.beta = 0.99
        return cfg

    if exp_name == "ablation_probe":
        cfg.num_trials = 10
        cfg.AFLBreI.batch_size = 32
        cfg.AFLBreI.beta = 0.99
        cfg.AFLBreI.record_every = 20
        return cfg

    if exp_name == "ablation_stepsize":
        cfg.num_trials = 10
        cfg.AFLBreI.num_iters = 2600
        cfg.AFLBreI.batch_size = 32
        return cfg

    if exp_name == "sparsity_scaling":
        cfg.num_trials = 10
        cfg.AFLBreI.num_iters = 2600
        cfg.AFLBreI.batch_size = 32
        cfg.AFLBreI.beta = 0.99
        cfg.AFLBreI.record_every = 10
        return cfg

    if exp_name == "noise_robustness":
        cfg.num_trials = 10
        cfg.problem.m = 600
        cfg.problem.n = 1200
        cfg.problem.s = 20
        cfg.AFLBreI.num_iters = 2600
        cfg.AFLBreI.batch_size = 32
        cfg.AFLBreI.beta = 0.99
        cfg.AFLBreI.record_every = 10
        return cfg

    if exp_name == "growing_batch_kkt":
        cfg.num_trials = 5
        cfg.problem.m = 256
        cfg.problem.n = 512
        cfg.problem.s = 20
        cfg.problem.noise_sigma = 0.0
        cfg.problem.snr_db = None
        cfg.AFLBreI.num_iters = 2000
        cfg.AFLBreI.q_batch_size = 20
        cfg.AFLBreI.f_star = 0.0
        cfg.AFLBreI.record_every = 2000
        return cfg

    if exp_name == "csmri_compare":
        cfg = ExperimentConfig()
        cfg.name = "csmri_compare"
        cfg.num_trials = 1
        cfg.problem = CSMRIProblemConfig(img_size=128, rays=30, wavelet="haar")
        cfg.problem.snr_db = 40.0

        cfg.AFLBreI.lam = 1.0
        cfg.AFLBreI.batch_size = 128
        cfg.AFLBreI.beta = 1.0
        cfg.AFLBreI.q_batch_size = 32
        cfg.AFLBreI.num_iters = 30000
        cfg.AFLBreI.record_every = 500

        cfg.oracle.lam = 1.0
        cfg.oracle.num_iters = 20000
        return cfg

    if exp_name == "csmri_sampling_sweep":
        cfg = ExperimentConfig()
        cfg.name = "csmri_sampling_sweep"
        cfg.num_trials = 3
        cfg.problem = CSMRIProblemConfig(
            img_size=128,
            rays=30,
            wavelet="haar",
            phantom_name="shepp_logan",
        )
        cfg.problem.snr_db = 40.0

        cfg.AFLBreI.lam = 1.0
        cfg.AFLBreI.batch_size = 100
        cfg.AFLBreI.beta = 1.0
        cfg.AFLBreI.q_batch_size = 16
        cfg.AFLBreI.num_iters = 20000
        cfg.AFLBreI.record_every = 500

        cfg.oracle.lam = 1.0
        cfg.oracle.num_iters = 20000
        cfg.csmri_sweep.rays_list = (16, 24, 30, 40)
        return cfg

    if exp_name == "csmri_noise_sweep":
        cfg = ExperimentConfig()
        cfg.name = "csmri_noise_sweep"
        cfg.num_trials = 3
        cfg.problem = CSMRIProblemConfig(
            img_size=128,
            rays=30,
            wavelet="haar",
            phantom_name="shepp_logan",
        )
        cfg.problem.snr_db = 40.0

        cfg.AFLBreI.lam = 1.0
        cfg.AFLBreI.batch_size = 100
        cfg.AFLBreI.beta = 1.0
        cfg.AFLBreI.q_batch_size = 16
        cfg.AFLBreI.num_iters = 20000
        cfg.AFLBreI.record_every = 500

        cfg.oracle.lam = 1.0
        cfg.oracle.num_iters = 20000
        cfg.csmri_sweep.snr_list = (20.0, 30.0, 40.0, None)
        return cfg

    if exp_name == "deconv_compare":
        cfg = ExperimentConfig()
        cfg.name = "deconv_compare"
        cfg.num_trials = 10
        cfg.problem = Deconv1DProblemConfig(
            n=1024,
            s=30,
            kernel_size=31,
            blur_sigma=3.0,
            snr_db=40.0,
            signal_dist="gaussian",
            seed=42,
        )

        cfg.AFLBreI.lam = 10
        cfg.AFLBreI.batch_size = 256
        cfg.AFLBreI.beta = 1.0
        cfg.AFLBreI.q_batch_size = 64
        cfg.AFLBreI.f_star = 0.0
        cfg.AFLBreI.num_iters = 20000
        cfg.AFLBreI.record_every = 10
        cfg.AFLBreI.step_safety = 0.75
        cfg.AFLBreI.clip_step = True
        cfg.AFLBreI.growing_batch = True
        cfg.AFLBreI.batch_floor_fraction = 0.5
        cfg.AFLBreI.grow_probe_with_batch = True
        cfg.AFLBreI.probe_batch_ratio = 0.5

        cfg.oracle.lam = 675
        cfg.oracle.step_c0 = 1
        cfg.oracle.num_iters = 160000
        cfg.sgdas.num_iters = 10
        cfg.rd.num_iters = 10
        cfg.support_tol = 1e-1
        return cfg

    raise ValueError(f"Unknown experiment name: {exp_name}")


def config_to_dict(cfg: ExperimentConfig) -> Dict:
    return asdict(cfg)
