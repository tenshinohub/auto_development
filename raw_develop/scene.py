from __future__ import annotations

import cv2
import numpy as np
from .constants import ImageStats, ShootingCondition, SubjectCandidate, SceneResult
from .utils import clamp

SCENE_PROFILES = {
    "portrait": dict(
        exposure=0.04,
        contrast=1.02,
        saturation=1.05,
        highlight=0.28,
        shadow=0.040,
        subject=0.03,
        subject_contrast=1.00,
        background=0.000,
        denoise=0.28,
        sharpen=0.75,
        skin=1.02,
        green=1.03,
        water=1.06,
        upper=0.08,
        tone=0.26,
    ),

    "night": dict(
        exposure=0.06,
        contrast=1.03,
        saturation=1.07,
        highlight=0.32,
        shadow=0.035,
        subject=0.02,
        subject_contrast=1.00,
        background=0.000,
        denoise=0.30,
        sharpen=0.45,
        skin=1.02,
        green=1.03,
        water=1.06,
        upper=0.10,
        tone=0.26,
    ),

    "sunset": dict(
        exposure=0.02,
        contrast=1.05,
        saturation=1.10,
        highlight=0.38,
        shadow=0.035,
        subject=0.02,
        subject_contrast=1.00,
        background=0.000,
        denoise=0.20,
        sharpen=0.75,
        skin=1.03,
        green=1.04,
        water=1.08,
        upper=0.12,
        tone=0.32,
    ),

    "landscape": dict(
        exposure=0.06,
        contrast=1.05,
        saturation=1.09,
        highlight=0.28,
        shadow=0.045,
        subject=0.02,
        subject_contrast=1.00,
        background=0.000,
        denoise=0.20,
        sharpen=0.80,
        skin=1.02,
        green=1.06,
        water=1.08,
        upper=0.10,
        tone=0.32,
    ),

    "city": dict(
        exposure=0.06,
        contrast=1.04,
        saturation=1.06,
        highlight=0.30,
        shadow=0.040,
        subject=0.02,
        subject_contrast=1.00,
        background=0.000,
        denoise=0.24,
        sharpen=0.75,
        skin=1.02,
        green=1.04,
        water=1.06,
        upper=0.10,
        tone=0.28,
    ),

    "indoor": dict(
        exposure=0.08,
        contrast=1.03,
        saturation=1.05,
        highlight=0.28,
        shadow=0.045,
        subject=0.02,
        subject_contrast=1.00,
        background=0.000,
        denoise=0.28,
        sharpen=0.60,
        skin=1.02,
        green=1.03,
        water=1.04,
        upper=0.08,
        tone=0.26,
    ),

    "general": dict(
        exposure=0.06,
        contrast=1.04,
        saturation=1.07,
        highlight=0.26,
        shadow=0.040,
        subject=0.02,
        subject_contrast=1.00,
        background=0.000,
        denoise=0.22,
        sharpen=0.75,
        skin=1.02,
        green=1.04,
        water=1.06,
        upper=0.08,
        tone=0.26,
    ),
}


def classify_scene(
    stats: ImageStats,
    shooting: ShootingCondition,
    subjects: list[SubjectCandidate],
    masks: dict[str, np.ndarray],
) -> SceneResult:

    person_area = float(
        np.mean(masks["person"])
    )

    vehicle_area = float(
        np.mean(masks["vehicle"])
    )

    # Portrait only when one person really dominates the frame.
    # A baseball play, street scene, or group shot has several
    # person blobs; treating all of them as a studio subject is
    # what made players look pasted onto the field.
    person_u8 = masks["person"].astype(np.uint8)
    n_person, _, person_stats, _ = cv2.connectedComponentsWithStats(
        person_u8,
        connectivity=8,
    )

    largest_person = 0.0
    if n_person > 1:
        areas = person_stats[1:, cv2.CC_STAT_AREA].astype(np.float32)
        largest_person = float(np.max(areas) / max(person_u8.size, 1))

    if (
        largest_person > 0.10
        and stats.median > 0.08
        and (shooting.shallow_dof or largest_person > 0.20)
    ):
        confidence = clamp(
            0.50
            + largest_person * 1.5
            + (0.15 if shooting.shallow_dof else 0),
            0,
            1,
        )

        return SceneResult(
            scene="portrait",
            confidence=confidence,
        )

    # Night
    if (
        shooting.low_light
        and stats.median < 0.10
    ):
        confidence = clamp(
            0.60
            + shooting.estimated_noise * 0.25,
            0,
            1,
        )

        return SceneResult(
            scene="night",
            confidence=confidence,
        )

    # Sunset
    if (
        stats.warm_ratio > 0.18
        and stats.p95 > 0.55
    ):
        confidence = clamp(
            0.55
            + stats.warm_ratio * 1.5,
            0,
            1,
        )

        return SceneResult(
            scene="sunset",
            confidence=confidence,
        )

    # Landscape
    if (
        shooting.wide_angle
        and stats.edge_density < 0.16
        and stats.dynamic_range > 5.0
    ):
        return SceneResult(
            scene="landscape",
            confidence=0.70,
        )

    # City
    if (
        vehicle_area > 0.01
        and stats.edge_density > 0.10
    ):
        return SceneResult(
            scene="city",
            confidence=0.68,
        )

    # Indoor
    if (
        stats.median < 0.18
        and stats.warm_ratio > 0.10
    ):
        return SceneResult(
            scene="indoor",
            confidence=0.62,
        )

    return SceneResult(
        scene="general",
        confidence=0.50,
    )
