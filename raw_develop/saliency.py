from __future__ import annotations

import cv2
import numpy as np

from .color import luminance
from .utils import normalize_image

def calculate_saliency(
    image: np.ndarray,
) -> np.ndarray:
    image = normalize_image(image)

    y = luminance(image)

    local = cv2.GaussianBlur(
        y,
        (0, 0),
        9,
    )

    local_contrast = np.abs(
        y - local
    )

    gx = cv2.Sobel(
        y,
        cv2.CV_32F,
        1,
        0,
        ksize=3,
    )

    gy = cv2.Sobel(
        y,
        cv2.CV_32F,
        0,
        1,
        ksize=3,
    )

    edges = cv2.magnitude(
        gx,
        gy,
    )

    edges = edges / (
        np.percentile(edges, 95) + 1e-6
    )

    edges = np.clip(
        edges,
        0,
        1,
    )

    max_rgb = np.max(image, axis=2)
    min_rgb = np.min(image, axis=2)

    saturation = np.clip(
        (max_rgb - min_rgb) * 3.0,
        0,
        1,
    )

    center_y, center_x = np.indices(
        y.shape
    )

    center_x = center_x / max(
        y.shape[1] - 1,
        1,
    )

    center_y = center_y / max(
        y.shape[0] - 1,
        1,
    )

    distance = np.sqrt(
        (center_x - 0.5) ** 2
        + (center_y - 0.5) ** 2
    )

    center = 1.0 - np.clip(
        distance / 0.707,
        0,
        1,
    )

    brightness = np.abs(
        y - np.median(y)
    )

    brightness /= (
        np.percentile(brightness, 95)
        + 1e-6
    )

    brightness = np.clip(
        brightness,
        0,
        1,
    )

    local_contrast /= (
        np.percentile(local_contrast, 95)
        + 1e-6
    )

    local_contrast = np.clip(
        local_contrast,
        0,
        1,
    )

    saliency = (
        local_contrast * 0.30
        + edges * 0.25
        + saturation * 0.15
        + brightness * 0.20
        + center * 0.10
    )

    return np.clip(
        saliency,
        0,
        1,
    )
