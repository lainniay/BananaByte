"""Metrics calculation for image restoration (UCIQE, UIQM)."""

import typing

import cv2
import numpy as np
from scipy import ndimage
from skimage import transform


def calculate_uciqe(
    img: np.ndarray,
    crop_border: int = 0,
    input_order: str = "HWC",
) -> float:
    """Calculate the UCIQE metric for an image."""
    if input_order == "CHW":
        img = img.transpose(1, 2, 0)
    if crop_border > 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border, ...]

    img_bgr = img if img.shape[2] == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    coe_metric = [0.4680, 0.2745, 0.2576]
    img_lum = img_lab[..., 0] / 255.0
    img_a = img_lab[..., 1] / 255.0
    img_b = img_lab[..., 2] / 255.0

    img_chr = np.sqrt(np.square(img_a) + np.square(img_b))
    img_sat = img_chr / (np.sqrt(np.square(img_chr) + np.square(img_lum)) + 1e-8)
    aver_sat = np.mean(img_sat)

    aver_chr = np.mean(img_chr)
    var_chr = np.sqrt(np.mean(np.abs(1 - np.square(aver_chr / (img_chr + 1e-8)))))

    dtype = img_lum.dtype
    nbins = 256 if dtype == "uint8" else 65536
    hist, _ = np.histogram(img_lum, nbins)
    cdf = np.cumsum(hist) / np.sum(hist)
    ilow = np.where(cdf > 0.01)[0]
    ihigh = np.where(cdf >= 0.99)[0]
    con_lum = (
        0.0
        if (len(ilow) == 0 or len(ihigh) == 0)
        else (ihigh[0] - ilow[0]) / (nbins - 1)
    )

    return float(
        coe_metric[0] * var_chr + coe_metric[1] * con_lum + coe_metric[2] * aver_sat,
    )


def _uicm(img: np.ndarray) -> float:
    img = np.array(img, dtype=np.float64)
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    rg = r - g
    yb = (r + g) / 2 - b
    k = r.shape[0] * r.shape[1]

    rg1 = np.sort(rg.reshape(1, k))
    alpha_l, alpha_r = 0.1, 0.1
    start, end = int(alpha_l * k + 1), int(k * (1 - alpha_r))
    rg1 = rg1[0, start:end] if start < end else rg1[0, :]
    n = max(1, k * (1 - alpha_r - alpha_l))
    mean_rg = np.sum(rg1) / n
    delta_rg = np.sqrt(np.sum((rg1 - mean_rg) ** 2) / n)

    yb1 = np.sort(yb.reshape(1, k))
    yb1 = yb1[0, start:end] if start < end else yb1[0, :]
    mean_yb = np.sum(yb1) / n
    delta_yb = np.sqrt(np.sum((yb1 - mean_yb) ** 2) / n)

    return float(
        -0.0268 * np.sqrt(mean_rg**2 + mean_yb**2)
        + 0.1586 * np.sqrt(delta_yb**2 + delta_rg**2),
    )


def _uiconm(img: np.ndarray) -> float:
    img = np.array(img, dtype=np.float64)
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    patchez = 5
    m, n = int(r.shape[0]), int(r.shape[1])

    if m % patchez != 0 or n % patchez != 0:
        x = int(m - m % patchez + patchez) if m % patchez != 0 else m
        y = int(n - n % patchez + patchez) if n % patchez != 0 else n
        r = typing.cast("np.ndarray", transform.resize(r, (x, y), anti_aliasing=True))
        g = typing.cast("np.ndarray", transform.resize(g, (x, y), anti_aliasing=True))
        b = typing.cast("np.ndarray", transform.resize(b, (x, y), anti_aliasing=True))

    m, n = int(r.shape[0]), int(r.shape[1])
    k1, k2 = m // patchez, n // patchez

    def cal_amee(channel: np.ndarray) -> float:
        amee = 0.0
        for i in range(0, m, patchez):
            for j in range(0, n, patchez):
                patch = channel[i : i + patchez, j : j + patchez]
                max_val, min_val = np.max(patch), np.min(patch)
                if (max_val != 0 or min_val != 0) and max_val != min_val:
                    amee += np.log((max_val - min_val) / (max_val + min_val)) * (
                        (max_val - min_val) / (max_val + min_val)
                    )
        return float(np.abs(amee) / (k1 * k2)) if k1 * k2 != 0 else 0.0

    return float(cal_amee(r) + cal_amee(g) + cal_amee(b))


def _uism(img: np.ndarray) -> float:
    img = np.array(img, dtype=np.float64)
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    hx = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    hy = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

    sobel_r = np.abs(
        ndimage.convolve(r, hx, mode="nearest")
        + ndimage.convolve(r, hy, mode="nearest"),
    )
    sobel_g = np.abs(
        ndimage.convolve(g, hx, mode="nearest")
        + ndimage.convolve(g, hy, mode="nearest"),
    )
    sobel_b = np.abs(
        ndimage.convolve(b, hx, mode="nearest")
        + ndimage.convolve(b, hy, mode="nearest"),
    )

    patchez = 5
    m, n = int(sobel_r.shape[0]), int(sobel_r.shape[1])
    if m % patchez != 0 or n % patchez != 0:
        x = int(m - m % patchez + patchez) if m % patchez != 0 else m
        y = int(n - n % patchez + patchez) if n % patchez != 0 else n
        sobel_r = typing.cast(
            "np.ndarray",
            transform.resize(sobel_r, (x, y), anti_aliasing=True),
        )
        sobel_g = typing.cast(
            "np.ndarray",
            transform.resize(sobel_g, (x, y), anti_aliasing=True),
        )
        sobel_b = typing.cast(
            "np.ndarray",
            transform.resize(sobel_b, (x, y), anti_aliasing=True),
        )

    m, n = int(sobel_r.shape[0]), int(sobel_r.shape[1])
    k1, k2 = m // patchez, n // patchez

    def cal_eme(channel: np.ndarray) -> float:
        eme = 0.0
        for i in range(0, m, patchez):
            for j in range(0, n, patchez):
                patch = channel[i : i + patchez, j : j + patchez]
                max_val, min_val = np.max(patch), np.min(patch)
                if max_val > 0 and min_val > 0:
                    eme += np.log(max_val / min_val)
        return float(2 * np.abs(eme) / (k1 * k2)) if k1 * k2 != 0 else 0.0

    lambda_r, lambda_g, lambda_b = 0.299, 0.587, 0.114
    return float(
        lambda_r * cal_eme(sobel_r)
        + lambda_g * cal_eme(sobel_g)
        + lambda_b * cal_eme(sobel_b),
    )


def calculate_uiqm(
    img: np.ndarray,
    crop_border: int = 0,
    input_order: str = "HWC",
    return_submetrics: bool = False,
) -> float | tuple[float, float, float, float]:
    """Calculate the UIQM metric for an image."""
    if input_order == "CHW":
        img = img.transpose(1, 2, 0)
    if crop_border > 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border, ...]

    if img.shape[2] != 3:
        img = np.stack([img[..., 0]] * 3, axis=-1)

    img = img.astype(np.float32)
    c1, c2, c3 = 0.0282, 0.2953, 3.5753

    uicm_val = _uicm(img)
    uism_val = _uism(img)
    uiconm_val = _uiconm(img)

    uiqm_total = c1 * uicm_val + c2 * uism_val + c3 * uiconm_val

    if return_submetrics:
        return uiqm_total, uicm_val, uism_val, uiconm_val
    return uiqm_total
