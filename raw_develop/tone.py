from __future__ import annotations

import math
import numpy as np

from .constants import ImageStats
from .color import luminance
from .utils import clamp, normalize_image

def calculate_exposure_target(
    scene: str,
) -> float:

    targets = {
        "portrait": 0.255,
        "night": 0.165,
        "sunset": 0.200,
        "landscape": 0.250,
        "city": 0.245,
        "indoor": 0.240,
        "general": 0.245,
    }

    target = targets.get(
        scene,
        0.245,
    )

    return clamp(
        target,
        0.12,
        0.28,
    )


def estimate_exposure_ev(
    stats: ImageStats,
    target: float,
) -> float:
    """Estimate exposure while making better use of highlight headroom.

    v23 changes:
    - The previous model was too conservative for normally exposed images
      whose p99 was far below the highlight limits.
    - Headroom bonus is now progressive up to p99 < 0.60.
    - The result is still constrained by the actual highlight levels.
    """

    median_error = (
        target - stats.median
    )

    median_ev = math.log2(
        max(target, 1e-5)
        / max(stats.median, 1e-5)
    )

    highlight_soft = 0.740
    highlight_hard = 0.880

    if stats.p95 > highlight_soft:
        highlight_ev = math.log2(
            highlight_soft
            / max(stats.p95, 1e-5)
        )
    else:
        highlight_ev = 0.0

    if stats.p99 > highlight_hard:
        hard_penalty = -0.25
    else:
        hard_penalty = 0.0

    ev = (
        median_ev * 0.78
        + highlight_ev * 0.22
        + hard_penalty
    )

    # Use available highlight headroom more aggressively.
    if stats.p99 < 0.40:
        ev += 0.32
    elif stats.p99 < 0.50:
        ev += 0.24
    elif stats.p99 < 0.60:
        ev += 0.16
    elif stats.p99 < 0.70:
        ev += 0.08

    if median_error > 0.02:
        ev += 0.08

    return clamp(
        ev,
        -0.60,
        1.20,
    )


def apply_exposure(
    image: np.ndarray,
    ev: float,
) -> np.ndarray:

    gain = 2.0 ** ev

    return np.clip(
        image * gain,
        0,
        1,
    )


def apply_contrast(
    image: np.ndarray,
    contrast: float,
) -> np.ndarray:

    y = luminance(image)

    new_y = (
        (y - 0.18)
        * contrast
        + 0.18
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


def apply_saturation(
    image: np.ndarray,
    saturation: float,
) -> np.ndarray:

    y = luminance(image)

    out = (
        y[..., None]
        + (image - y[..., None])
        * saturation
    )

    return np.clip(
        out,
        0,
        1,
    )


def apply_tone(
    image: np.ndarray,
    strength: float,
    shadow_lift: float,
    highlight_protection: float,
) -> np.ndarray:
    """Apply a gentle, mostly brightness-neutral tone adjustment.

    The old S-curve lowered pixels around the normal midtone range because
    tanh((y - 0.45) * 3) is negative for most ordinary photographs.
    v23 anchors the curve around 0.18 and uses separate shadow/highlight
    controls, so tone no longer makes an otherwise correctly exposed image
    globally darker.
    """

    image = normalize_image(
        image
    )

    y = luminance(
        image
    )

    # Very gentle midtone shaping, anchored at 18% luminance.
    # The anchor subtraction keeps 0.18 approximately unchanged.
    anchor = math.tanh(
        (0.18 - 0.45) * 3.0
    )

    curve = (
        np.tanh(
            (y - 0.45) * 3.0
        )
        - anchor
    )

    s = (
        y
        + strength
        * 0.055
        * curve
    )

    # Shadows: lift only the dark range.
    shadow_mask = np.clip(
        (0.22 - y) / 0.22,
        0,
        1,
    )

    shadow_mask *= np.clip(
        y / 0.22,
        0,
        1,
    )

    s += (
        shadow_lift
        * shadow_mask
    )

    # Highlights: compress only where needed.
    highlight_mask = np.clip(
        (y - 0.68) / 0.32,
        0,
        1,
    )

    s -= (
        highlight_protection
        * 0.055
        * highlight_mask
    )

    s = np.clip(
        s,
        0,
        1,
    )

    ratio = s / (
        y + 1e-6
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
