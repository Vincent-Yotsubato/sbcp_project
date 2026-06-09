from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pywt
import scipy.fft as fft
from scipy.signal import convolve


class LinearOperator:
    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def adjoint(self, y: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @property
    def shape(self):
        raise NotImplementedError


@dataclass
class MatrixOperator(LinearOperator):
    A: np.ndarray
    track_calls: bool = True

    def __post_init__(self):
        self.forward_calls = 0
        self.adjoint_calls = 0

    def forward(self, x: np.ndarray) -> np.ndarray:
        if self.track_calls:
            self.forward_calls += 1
        return self.A @ x

    def forward_batch(self, X: np.ndarray) -> np.ndarray:
        if self.track_calls:
            self.forward_calls += X.shape[0]
        return X @ self.A.T

    def adjoint(self, y: np.ndarray) -> np.ndarray:
        if self.track_calls:
            self.adjoint_calls += 1
        return self.A.T @ y

    @property
    def shape(self):
        return self.A.shape

    def reset_counters(self) -> None:
        self.forward_calls = 0
        self.adjoint_calls = 0

    def spectral_norm_estimate(self) -> float:
        return np.linalg.norm(self.A, ord=2)


class CSMRIWaveletOperator(LinearOperator):
    def __init__(
        self,
        mask: np.ndarray,
        wavelet_name: str = "db4",
        img_shape: Tuple[int, int] = (256, 256),
    ):
        self.mask = mask
        self.wavelet_name = wavelet_name
        self.img_shape = img_shape
        self.track_calls = True
        self.forward_calls = 0
        self.adjoint_calls = 0

        dummy_img = np.zeros(img_shape)
        coeffs = pywt.wavedec2(dummy_img, self.wavelet_name, mode="periodization")
        arr, self.coeff_slices = pywt.coeffs_to_array(coeffs)

        self.coeff_shape = arr.shape
        self.n = np.prod(self.coeff_shape)
        self.m = int(np.sum(mask))

    def _array_to_img(self, x_flat: np.ndarray) -> np.ndarray:
        coeffs = pywt.array_to_coeffs(
            x_flat.reshape(self.coeff_shape),
            self.coeff_slices,
            output_format="wavedec2",
        )
        return pywt.waverec2(coeffs, self.wavelet_name, mode="periodization")

    def _img_to_array(self, img: np.ndarray) -> np.ndarray:
        coeffs = pywt.wavedec2(img, self.wavelet_name, mode="periodization")
        arr, _ = pywt.coeffs_to_array(coeffs)
        return arr.flatten()

    def forward(self, x_flat: np.ndarray) -> np.ndarray:
        if self.track_calls:
            self.forward_calls += 1

        img = self._array_to_img(x_flat)
        kspace = fft.fftshift(fft.fft2(img, norm="ortho"))
        masked_kspace = kspace[self.mask]
        return np.concatenate([masked_kspace.real, masked_kspace.imag])

    def forward_batch(self, X_flat: np.ndarray) -> np.ndarray:
        B = X_flat.shape[0]
        if self.track_calls:
            self.forward_calls += B

        imgs = np.zeros((B, self.img_shape[0], self.img_shape[1]))
        for i in range(B):
            imgs[i] = self._array_to_img(X_flat[i])

        kspaces = fft.fftshift(
            fft.fft2(imgs, axes=(-2, -1), norm="ortho"),
            axes=(-2, -1),
        )
        masked_kspaces = kspaces[:, self.mask]
        return np.concatenate([masked_kspaces.real, masked_kspaces.imag], axis=-1)

    def adjoint(self, y_concat: np.ndarray) -> np.ndarray:
        if self.track_calls:
            self.adjoint_calls += 1

        y_complex = y_concat[: self.m] + 1j * y_concat[self.m :]
        kspace = np.zeros(self.img_shape, dtype=complex)
        kspace[self.mask] = y_complex

        img = fft.ifft2(fft.ifftshift(kspace), norm="ortho").real
        return self._img_to_array(img)

    @property
    def shape(self):
        return (2 * self.m, self.n)

    def reset_counters(self) -> None:
        self.forward_calls = 0
        self.adjoint_calls = 0

    def spectral_norm_estimate(self) -> float:
        return 1.0


class Convolution1DOperator(LinearOperator):
    def __init__(self, kernel: np.ndarray, n: int, track_calls: bool = True):
        self.kernel = np.asarray(kernel, dtype=float)
        self.n = int(n)
        self.m = int(n)
        self.track_calls = track_calls
        self.forward_calls = 0
        self.adjoint_calls = 0
        self.kernel_flip = self.kernel[::-1].copy()

    def forward(self, x: np.ndarray) -> np.ndarray:
        if self.track_calls:
            self.forward_calls += 1
        return convolve(x, self.kernel, mode="same")

    def forward_batch(self, X: np.ndarray) -> np.ndarray:
        if self.track_calls:
            self.forward_calls += X.shape[0]
        return np.array([convolve(x, self.kernel, mode="same") for x in X])

    def adjoint(self, y: np.ndarray) -> np.ndarray:
        if self.track_calls:
            self.adjoint_calls += 1
        return convolve(y, self.kernel_flip, mode="same")

    @property
    def shape(self):
        return (self.m, self.n)

    def reset_counters(self) -> None:
        self.forward_calls = 0
        self.adjoint_calls = 0

    def spectral_norm_estimate(self) -> float:
        return float(np.max(np.abs(np.fft.fft(self.kernel, n=self.n))))
