from __future__ import annotations

import math
import cv2
import numpy as np
from typing import Optional

from .constants import ImageStats
from .utils import normalize_image, safe_div
from .color import luminance

def calculate_warm_ratio(rgb: np.ndarray) -> float:
    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]

    warm = (
        (r > b * 1.08)
        & (r > g * 1.02)
        & (g >= b * 0.95)
    )

    return float(np.mean(warm))


def calculate_stats(rgb: np.ndarray) -> ImageStats:
    rgb = normalize_image(rgb)

    y = luminance(rgb)

    flat = y.reshape(-1)

    mean = float(np.mean(flat))
    median = float(np.median(flat))

    p01, p05, p25, p75, p95, p99 = np.percentile(
        flat,
        [1, 5, 25, 75, 95, 99],
    )

    shadow_ratio = float(np.mean(y < 0.05))
    highlight_ratio = float(np.mean(y > 0.95))

    dynamic_range = math.log10(
        max(float(p95), 1e-6)
        / max(float(p05), 1e-6)
    )

    max_rgb = np.max(rgb, axis=2)
    min_rgb = np.min(rgb, axis=2)

    saturation_ratio = float(
        np.mean((max_rgb - min_rgb) > 0.08)
    )

    gray = np.clip(
        y * 255.0,
        0,
        255,
    ).astype(np.uint8)

    sobel_x = cv2.Sobel(
        gray,
        cv2.CV_32F,
        1,
        0,
        ksize=3,
    )

    sobel_y = cv2.Sobel(
        gray,
        cv2.CV_32F,
        0,
        1,
        ksize=3,
    )

    magnitude = cv2.magnitude(
        sobel_x,
        sobel_y,
    )

    edge_density = float(
        np.mean(magnitude > 20.0)
    )

    contrast = float(np.std(y))

    r_mean = float(np.mean(rgb[..., 0]))
    g_mean = float(np.mean(rgb[..., 1]))
    b_mean = float(np.mean(rgb[..., 2]))

    rg_ratio = safe_div(r_mean, g_mean)
    gb_ratio = safe_div(g_mean, b_mean)

    warm_ratio = calculate_warm_ratio(rgb)

    return ImageStats(
        mean=mean,
        median=median,
        p01=float(p01),
        p05=float(p05),
        p25=float(p25),
        p75=float(p75),
        p95=float(p95),
        p99=float(p99),
        shadow_ratio=shadow_ratio,
        highlight_ratio=highlight_ratio,
        dynamic_range=dynamic_range,
        saturation_ratio=saturation_ratio,
        edge_density=edge_density,
        contrast=contrast,
        mean_luminance=mean,
        r_mean=r_mean,
        g_mean=g_mean,
        b_mean=b_mean,
        rg_ratio=rg_ratio,
        gb_ratio=gb_ratio,
        warm_ratio=warm_ratio,
    )


def print_stats(
    name: str,
    rgb: np.ndarray,
    stats: Optional[ImageStats] = None,
    display_transform: bool = False,
):
    if stats is None:
        stats = calculate_stats(rgb)

    print(f"\n{name}:")

    print(
        f"  min/max           : "
        f"{np.min(rgb):.6f} / {np.max(rgb):.6f}"
    )

    print(
        f"  mean/median       : "
        f"{stats.mean:.6f} / {stats.median:.6f}"
    )

    print(
        f"  p01/p05           : "
        f"{stats.p01:.6f} / {stats.p05:.6f}"
    )

    print(
        f"  p95/p99           : "
        f"{stats.p95:.6f} / {stats.p99:.6f}"
    )

    print(
        f"  shadow            : "
        f"{stats.shadow_ratio * 100:.3f}%"
    )

    print(
        f"  highlight         : "
        f"{stats.highlight_ratio * 100:.3f}%"
    )

    print(
        f"  dynamic           : "
        f"{stats.dynamic_range:.3f}"
    )

    print(
        f"  saturation        : "
        f"{stats.saturation_ratio * 100:.3f}%"
    )

    print(
        f"  edge              : "
        f"{stats.edge_density:.4f}"
    )

    print(
        f"  contrast          : "
        f"{stats.contrast:.6f}"
    )

    print(
        f"  RGB               : "
        f"{stats.r_mean:.6f}, "
        f"{stats.g_mean:.6f}, "
        f"{stats.b_mean:.6f}"
    )

    print(
        f"  R/G               : "
        f"{stats.rg_ratio:.4f}"
    )

    print(
        f"  G/B               : "
        f"{stats.gb_ratio:.4f}"
    )

    print(
        f"  warm ratio        : "
        f"{stats.warm_ratio * 100:.3f}%"
    )
