from __future__ import annotations

import math
import numpy as np
from typing import Optional

from .constants import DevelopParams
from .color import luminance
from .stats import calculate_stats
from .tone import (
    calculate_exposure_target,
    apply_exposure,
    apply_contrast,
    apply_saturation,
    apply_tone,
)
from .utils import clamp

def score_candidate(
    image: np.ndarray,
    scene: str,
    target: float,
    subject_mask: Optional[np.ndarray],
) -> float:
    stats = calculate_stats(
        image
    )

    score = 0.0

    # Global exposure.
    score -= abs(
        stats.median
        - target
    ) * 5.0

    # Prefer a useful overall brightness when the image has
    # substantial highlight headroom.
    if stats.p99 < 0.62:
        score += min(
            0.14,
            (0.62 - stats.p99) * 0.28,
        )

    # Highlight protection.
    if stats.p95 > 0.78:
        score -= (
            stats.p95 - 0.78
        ) * 3.5

    if stats.p99 > 0.90:
        score -= (
            stats.p99 - 0.90
        ) * 5.0

    # Avoid crushed shadows.
    if stats.shadow_ratio > 0.12:
        score -= (
            stats.shadow_ratio
            - 0.12
        ) * 2.0

    # Reasonable contrast.
    score -= abs(
        stats.contrast
        - 0.10
    ) * 0.5

    # Saturation.
    if stats.saturation_ratio > 0.88:
        score -= (
            stats.saturation_ratio
            - 0.88
        )

    # Subject.
    if (
        subject_mask is not None
        and np.any(subject_mask)
    ):
        y = luminance(image)

        subject_median = float(
            np.median(
                y[subject_mask]
            )
        )

        subject_target = (
            0.26
            if scene == "portrait"
            else min(
                target + 0.02,
                0.26,
            )
        )

        score -= abs(
            subject_median
            - subject_target
        ) * 0.8

    return float(score)


def automatic_parameter_search(
    image: np.ndarray,
    scene: str,
    profile: dict,
    estimated_ev: float,
    masks: dict[str, np.ndarray],
) -> DevelopParams:

    target = calculate_exposure_target(
        scene
    )

    offsets = [
        -0.15,
        -0.08,
        0.00,
        0.08,
        0.16,
        0.25,
        0.35,
    ]

    contrasts = [
        0.98,
        1.00,
        1.03,
        1.06,
    ]

    saturations = [
        1.00,
        1.04,
        1.08,
        1.12,
    ]

    subject_mask = masks.get(
        "subject"
    )

    best_score = -float("inf")
    best = None

    for offset in offsets:
        ev = clamp(
            estimated_ev + offset,
            -1.0,
            1.0,
        )

        exposure = apply_exposure(
            image,
            ev,
        )

        for contrast in contrasts:
            contrast_img = apply_contrast(
                exposure,
                contrast,
            )

            for saturation in saturations:
                candidate = apply_saturation(
                    contrast_img,
                    saturation,
                )

                # Score the candidate after the tone stage as well.
                # This prevents the search from selecting an EV that looks
                # correct before tone but becomes dark afterward.
                candidate_tone = apply_tone(
                    candidate,
                    profile["tone"],
                    profile["shadow"],
                    profile["highlight"],
                )

                score = score_candidate(
                    candidate_tone,
                    scene,
                    target,
                    subject_mask,
                )

                if score > best_score:
                    best_score = score

                    best = (
                        ev,
                        contrast,
                        saturation,
                    )

    assert best is not None

    ev, contrast, saturation = best

    print(
        f"Search selected EV {ev:+.3f}, "
        f"contrast {contrast:.3f}, "
        f"saturation {saturation:.3f}"
    )

    return DevelopParams(
        exposure_ev=ev,
        contrast=contrast,
        saturation=saturation,

        highlight_protection=profile[
            "highlight"
        ],

        shadow_lift=profile[
            "shadow"
        ],

        subject_exposure=profile[
            "subject"
        ],

        subject_contrast=profile[
            "subject_contrast"
        ],

        background_suppression=profile[
            "background"
        ],

        denoise=profile[
            "denoise"
        ],

        sharpen=profile[
            "sharpen"
        ],

        skin_saturation=profile[
            "skin"
        ],

        green_saturation=profile[
            "green"
        ],

        water_saturation=profile[
            "water"
        ],

        upper_brightness=profile[
            "upper"
        ],

        tone_strength=profile[
            "tone"
        ],
    )
