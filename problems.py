# problems.py
from typing import Dict, Tuple

import numpy as np

from operators import MatrixOperator


def generate_sparse_signal(n: int, s: int, dist: str = "gaussian") -> np.ndarray:
    x = np.zeros(n)
    support = np.random.choice(n, size=s, replace=False)

    if dist == "gaussian":
        x[support] = np.random.randn(s)
    elif dist == "pm1":
        x[support] = np.random.choice([-1.0, 1.0], size=s)
    else:
        raise ValueError(f"Unknown signal distribution: {dist}")

    return x


def generate_sensing_matrix(
    m: int,
    n: int,
    dist: str = "gaussian",
    normalize_columns: bool = True,
) -> np.ndarray:
    if dist != "gaussian":
        raise ValueError(f"Unknown matrix distribution: {dist}")

    A = np.random.randn(m, n) / np.sqrt(m)

    if normalize_columns:
        col_norms = np.linalg.norm(A, axis=0)
        col_norms = np.maximum(col_norms, 1e-12)
        A = A / col_norms

    return A


def make_measurement(
    A: np.ndarray,
    x_true: np.ndarray,
    noise_sigma: float = None,
    snr_db: float = None,
) -> np.ndarray:
    signal = A @ x_true

    if noise_sigma is not None:
        if noise_sigma > 0:
            signal = signal + noise_sigma * np.random.randn(A.shape[0])
        return signal

    if snr_db is not None:
        signal_power = np.mean(signal ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10.0))
        noise_sigma_snr = np.sqrt(noise_power)
        noise = noise_sigma_snr * np.random.randn(A.shape[0])
        return signal + noise

    return signal

def generate_sparse_signal_separated(
    n: int,
    s: int,
    min_separation: int = 8,
    margin: int = 0,
    dist: str = "pm1",
) -> np.ndarray:
    x = np.zeros(n)

    available = list(range(margin, n - margin))
    support = []

    while len(support) < s:
        if len(available) == 0:
            raise ValueError("Cannot place all spikes with the requested separation.")
        idx = np.random.choice(available)
        support.append(idx)

        # remove points too close to idx
        available = [j for j in available if abs(j - idx) >= min_separation]

    support = np.array(sorted(support), dtype=int)

    if dist == "pm1":
        x[support] = np.random.choice([-1.0, 1.0], size=s)
    elif dist == "gaussian":
        vals = np.random.randn(s)
        vals = np.sign(vals) * np.maximum(np.abs(vals), 0.8)  # avoid tiny spikes
        x[support] = vals
    else:
        raise ValueError(f"Unknown dist: {dist}")

    return x

def build_sparse_recovery_problem(cfg) -> Dict:
    A = generate_sensing_matrix(
        m=cfg.m,
        n=cfg.n,
        dist=cfg.matrix_dist,
        normalize_columns=cfg.normalize_columns,
    )
    x_true = generate_sparse_signal_separated(
        n=cfg.n,
        s=cfg.s,
        min_separation=getattr(cfg, "min_separation", 8),
        margin=getattr(cfg, "margin", 0),
        dist=cfg.signal_dist,
    )
    b = make_measurement(
    A,
    x_true,
    noise_sigma=cfg.noise_sigma,
    snr_db=getattr(cfg, "snr_db", None),
    )

    operator = MatrixOperator(A)

    return {
        "A": A,
        "operator": operator,
        "x_true": x_true,
        "b": b,
        "m": cfg.m,
        "n": cfg.n,
        "s": cfg.s,
    }


##

from skimage.data import shepp_logan_phantom
from skimage.transform import resize

# problems.py
from skimage.data import shepp_logan_phantom
from skimage.transform import resize, rotate

def load_csmri_phantom(phantom_name: str, img_size: int) -> np.ndarray:
    base = shepp_logan_phantom()
    base = resize(base, (img_size, img_size), anti_aliasing=True)

    if phantom_name == "shepp_logan":
        img = base
    elif phantom_name == "shepp_logan_rot15":
        img = rotate(base, angle=15, resize=False, preserve_range=True)
    elif phantom_name == "shepp_logan_rot30":
        img = rotate(base, angle=30, resize=False, preserve_range=True)
    else:
        raise ValueError(f"Unknown phantom_name: {phantom_name}")

    img = np.clip(img, 0.0, 1.0)
    return img


def generate_radial_mask(shape, rays=30):
    # 生成径向欠采样掩码的简单实�?
    mask = np.zeros(shape, dtype=bool)
    center = (shape[0]//2, shape[1]//2)
    for angle in np.linspace(0, np.pi, rays, endpoint=False):
        for r in range(-max(shape), max(shape)):
            x = int(center[0] + r * np.cos(angle))
            y = int(center[1] + r * np.sin(angle))
            if 0 <= x < shape[0] and 0 <= y < shape[1]:
                mask[x, y] = True
    return mask

def build_csmri_problem(cfg) -> Dict:
    from operators import CSMRIWaveletOperator
    
    # 1. 准备真实图像 (Shepp-Logan 脑部体模)
    img_true = load_csmri_phantom(
    phantom_name=getattr(cfg, "phantom_name", "shepp_logan"),
    img_size=cfg.img_size,
    )
    
    # 2. 生成掩码
    mask = generate_radial_mask((cfg.img_size, cfg.img_size), rays=cfg.rays)
    
    # 3. 实例化算�?
    operator = CSMRIWaveletOperator(mask, wavelet_name=cfg.wavelet, img_shape=img_true.shape)
    
    # 4. 获取真实的小波系数作�?ground truth
    x_true = operator._img_to_array(img_true)
    
    # 5. 生成无噪�?带噪声的测量�?
    b_exact = operator.forward(x_true)
    if getattr(cfg, "noise_sigma", None) is not None and cfg.noise_sigma > 0:
        b = b_exact + cfg.noise_sigma * np.random.randn(len(b_exact))
    elif getattr(cfg, "snr_db", None) is not None:
        signal_power = np.mean(b_exact ** 2)
        noise_power = signal_power / (10 ** (cfg.snr_db / 10.0))
        b = b_exact + np.sqrt(noise_power) * np.random.randn(len(b_exact))
    else:
        b = b_exact

    return {
        "operator": operator,
        "x_true": x_true,  # 算法优化的变量是小波系数
        "img_true": img_true, # 仅用于最后计�?PSNR/可视�?
        "b": b,
        "n": len(x_true),
        "mask": mask
    }
    
def generate_gaussian_kernel_1d(kernel_size: int, sigma: float) -> np.ndarray:
    assert kernel_size % 2 == 1, "kernel_size should be odd"
    radius = kernel_size // 2
    x = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-(x ** 2) / (2.0 * sigma ** 2))
    kernel /= np.sum(kernel)
    return kernel


def make_measurement_from_operator(
    operator,
    x_true: np.ndarray,
    noise_sigma: float = None,
    snr_db: float = None,
) -> np.ndarray:
    signal = operator.forward(x_true)

    if noise_sigma is not None:
        if noise_sigma > 0:
            signal = signal + noise_sigma * np.random.randn(signal.shape[0])
        return signal

    if snr_db is not None:
        signal_power = np.mean(signal ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10.0))
        noise_sigma_snr = np.sqrt(noise_power)
        noise = noise_sigma_snr * np.random.randn(signal.shape[0])
        return signal + noise

    return signal


def build_deconv1d_problem(cfg) -> Dict:
    from operators import Convolution1DOperator

    x_true = generate_sparse_signal(cfg.n, cfg.s, dist=cfg.signal_dist)

    kernel = generate_gaussian_kernel_1d(
        kernel_size=cfg.kernel_size,
        sigma=cfg.blur_sigma,
    )

    operator = Convolution1DOperator(kernel=kernel, n=cfg.n)

    b = make_measurement_from_operator(
        operator,
        x_true,
        noise_sigma=getattr(cfg, "noise_sigma", None),
        snr_db=getattr(cfg, "snr_db", None),
    )

    return {
        "operator": operator,
        "x_true": x_true,
        "b": b,
        "n": cfg.n,
        "s": cfg.s,
        "kernel": kernel,
    }
    
