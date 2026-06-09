# metrics.py
from typing import Dict, Set
import numpy as np


def residual_norm(residual: np.ndarray) -> float:
    return float(np.linalg.norm(residual))


def relative_error(x: np.ndarray, x_true: np.ndarray) -> float:
    denom = np.linalg.norm(x_true) + 1e-12
    return float(np.linalg.norm(x - x_true) / denom)


def support(x: np.ndarray, tol: float = 1e-3) -> Set[int]:
    return set(np.where(np.abs(x) > tol)[0].tolist())


def support_metrics(x: np.ndarray, x_true: np.ndarray, tol: float = 1e-3) -> Dict[str, float]:
    pred_mask = np.abs(x) > tol
    true_mask = np.abs(x_true) > tol

    tp = int(np.count_nonzero(pred_mask & true_mask))
    fp = int(np.count_nonzero(pred_mask & ~true_mask))
    fn = int(np.count_nonzero(~pred_mask & true_mask))
    nnz = int(np.count_nonzero(pred_mask))

    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    exact = float(np.array_equal(pred_mask, true_mask))

    return {
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_support": exact,
        "nnz": float(nnz),
    }


def compute_all_metrics(
    x: np.ndarray,
    x_true: np.ndarray,
    residual: np.ndarray,
    tol: float = 1e-3,
) -> Dict[str, float]:
    sm = support_metrics(x, x_true, tol=tol)

    out = {
        "residual": residual_norm(residual),
        "rel_error": relative_error(x, x_true),
        "tp": sm["tp"],
        "fp": sm["fp"],
        "fn": sm["fn"],
        "precision": sm["precision"],
        "recall": sm["recall"],
        "f1": sm["f1"],
        "exact_support": sm["exact_support"],
        "nnz": sm["nnz"],
    }
    return out

##
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def compute_image_metrics(img_pred: np.ndarray, img_true: np.ndarray) -> Dict[str, float]:
    img_pred = np.clip(img_pred, 0, 1) # 通常图像归一化到 [0, 1]
    val_psnr = psnr(img_true, img_pred, data_range=1.0)
    val_ssim = ssim(img_true, img_pred, data_range=1.0)
    return {"psnr": val_psnr, "ssim": val_ssim}
