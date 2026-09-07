from __future__ import annotations

import math

from .constants import ExifMetadata, ShootingCondition
from .utils import clamp

def analyze_shooting(meta: ExifMetadata) -> ShootingCondition:
    iso = max(meta.iso, 100.0)

    iso_factor = math.sqrt(iso / 100.0)
    iso_factor = clamp(
        iso_factor,
        1.0,
        5.0,
    )

    exposure = meta.exposure_time

    low_light = (
        iso >= 800
        or (
            exposure > 0
            and exposure >= 1 / 60
            and iso >= 400
        )
    )

    if exposure <= 0:
        motion_risk = 0.0
    else:
        motion_risk = clamp(
            (1 / 60 - exposure) * 80,
            0.0,
            1.0,
        )

    shallow_dof = (
        meta.aperture > 0
        and meta.aperture <= 2.8
    )

    wide_angle = (
        meta.focal_length > 0
        and meta.focal_length <= 28
    )

    telephoto = (
        meta.focal_length >= 85
    )

    estimated_noise = clamp(
        (iso_factor - 1.0) / 4.0,
        0.0,
        1.0,
    )

    return ShootingCondition(
        iso_factor=iso_factor,
        low_light=low_light,
        motion_risk=motion_risk,
        shallow_dof=shallow_dof,
        wide_angle=wide_angle,
        telephoto=telephoto,
        estimated_noise=estimated_noise,
    )
