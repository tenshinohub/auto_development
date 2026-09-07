from __future__ import annotations

import cv2
import numpy as np

from .color import luminance
from .utils import clamp, normalize_image

def calculate_denoise_strength(
    base: float,
    shooting: ShootingCondition,
) -> float:

    iso_component = shooting.estimated_noise

    strength = (
        base
        * (
            0.55
            + 0.20 * iso_component
        )
    )

    return clamp(
        strength,
        0.05,
        0.28,
    )


def apply_denoise(
    image: np.ndarray,
    strength: float,
) -> np.ndarray:

    image = normalize_image(image)

    y = luminance(image)

    y8 = np.clip(
        y * 255.0,
        0,
        255,
    ).astype(np.uint8)

    sigma_color = (
        5.0
        + 14.0 * strength
    )

    sigma_space = (
        1.2
        + 1.8 * strength
    )

    filtered_y8 = cv2.bilateralFilter(
        y8,
        d=5,
        sigmaColor=sigma_color,
        sigmaSpace=sigma_space,
    )

    filtered_y = (
        filtered_y8.astype(np.float32)
        / 255.0
    )

    # Edge-aware blend.
    gx = cv2.Sobel(
        y8,
        cv2.CV_32F,
        1,
        0,
        ksize=3,
    )

    gy = cv2.Sobel(
        y8,
        cv2.CV_32F,
        0,
        1,
        ksize=3,
    )

    edge = cv2.magnitude(
        gx,
        gy,
    )

    edge /= (
        np.percentile(edge, 95)
        + 1e-6
    )

    edge = np.clip(
        edge,
        0,
        1,
    )

    blend = (
        strength
        * (
            1.0
            - 0.70 * edge
        )
    )

    blend = np.clip(
        blend,
        0,
        0.35,
    )

    new_y = (
        y * (1.0 - blend)
        + filtered_y * blend
    )

    ratio = new_y / (
        y + 1e-6
    )

    out = image * ratio[..., None]

    return np.clip(
        out,
        0,
        1,
    )


def apply_sharpen(
    image: np.ndarray,
    strength: float,
) -> np.ndarray:

    if strength <= 0:
        return image

    y = luminance(image)

    blur = cv2.GaussianBlur(
        y,
        (0, 0),
        1.0,
    )

    amount = (
        0.45
        * strength
    )

    sharp_y = (
        y
        + amount
        * (y - blur)
    )

    sharp_y = np.clip(
        sharp_y,
        0,
        1,
    )

    ratio = (
        sharp_y
        / (y + 1e-6)
    )

    out = (
        image
        * ratio[..., None]
    )

    return np.clip(
        out,
        0,
        1,
    )
