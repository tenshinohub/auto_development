from __future__ import annotations

import cv2
import numpy as np

def clamp(x: float, lo: float, hi: float) -> float:
    return float(np.clip(x, lo, hi))


def safe_div(a: float, b: float, eps: float = 1e-8) -> float:
    return float(a / max(abs(b), eps))


def fmt_optional(value: Optional[float]) -> str:
    if value is None:
        return "None"
    return f"{value:.3f}"


def ensure_float32(image: np.ndarray) -> np.ndarray:
    return np.asarray(image, dtype=np.float32)


def normalize_image(image: np.ndarray) -> np.ndarray:
    return np.clip(image.astype(np.float32), 0.0, 1.0)


def soften_mask(
    mask: np.ndarray,
    radius: float,
) -> np.ndarray:
    """Convert a hard mask to a soft [0, 1] weight with Gaussian falloff.

    A hard boolean subject/background split leaves a visible seam where
    DeepLab cuts through dirt, grass, or clothing. Blurring the mask
    makes local exposure and saturation changes fade out instead of
    stepping.
    """

    weight = np.asarray(mask, dtype=np.float32)

    if weight.ndim > 2:
        weight = np.squeeze(weight)

    weight = np.clip(weight, 0.0, 1.0)

    if radius < 0.5 or weight.size == 0:
        return weight

    # Keep a small core of the original mask so the subject still
    # receives the full correction after the blur dilutes the edge.
    eroded = cv2.erode(
        weight,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (3, 3),
        ),
        iterations=1,
    )

    blurred = cv2.GaussianBlur(
        weight,
        (0, 0),
        sigmaX=float(radius),
        sigmaY=float(radius),
    )

    soft = np.maximum(eroded * 0.35 + blurred * 0.65, blurred)

    return np.clip(soft, 0.0, 1.0)


def mask_feather_radius(shape: tuple[int, ...]) -> float:
    """Scale the seam-hiding blur with image size."""

    h = int(shape[0])
    w = int(shape[1]) if len(shape) > 1 else h
    return float(clamp(min(h, w) * 0.018, 12.0, 48.0))


def erode_bool_mask(
    mask: np.ndarray,
    pixels: int,
) -> np.ndarray:
    """Pull a hard mask inward so edge dirt is not treated as subject."""

    if pixels <= 0:
        return mask.astype(bool)

    k = 2 * int(pixels) + 1
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (k, k),
    )

    return cv2.erode(
        mask.astype(np.uint8),
        kernel,
        iterations=1,
    ).astype(bool)
