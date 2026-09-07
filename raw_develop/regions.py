from __future__ import annotations

import math
import numpy as np

from .constants import DevelopParams, RegionStats
from .color import luminance
from .utils import clamp, soften_mask, mask_feather_radius, erode_bool_mask

def apply_luma_ratio(
    image: np.ndarray,
    new_y: np.ndarray,
    weight: np.ndarray,
) -> np.ndarray:
    """Apply a luminance-only change, faded by a soft weight."""

    y = luminance(image)
    blended_y = y * (1.0 - weight) + new_y * weight
    blended_y = np.clip(blended_y, 0.0, 1.0)
    ratio = blended_y / np.maximum(y, 1e-6)
    out = image * ratio[..., None]
    return np.clip(out, 0.0, 1.0)


def apply_region_processing(
    image: np.ndarray,
    masks: dict[str, np.ndarray],
    params: DevelopParams,
) -> np.ndarray:

    out = image.copy()

    subject_hard = masks["subject"].astype(bool)
    background_hard = masks["background"].astype(bool)

    radius = mask_feather_radius(out.shape)

    # Pull the subject inward first. DeepLab person masks on a dirt
    # infield typically leak a ring of terracotta clay; lifting that
    # ring and suppressing the dirt next to it is what draws the
    # visible outline around players.
    erode_px = int(clamp(radius * 0.35, 3.0, 14.0))
    subject_core = erode_bool_mask(subject_hard, erode_px)
    subject_w = soften_mask(subject_core, radius)

    # Background is the complement of the *soft* subject so the two
    # corrections cross-fade instead of meeting at a hard cut.
    background_w = np.clip(1.0 - subject_w, 0.0, 1.0)
    if np.any(background_hard):
        background_w = np.minimum(
            background_w,
            soften_mask(background_hard, radius),
        )

    print(
        f"Soft masks: feather {radius:.1f}px, "
        f"subject erode {erode_px}px, "
        f"subject mean {float(np.mean(subject_w)):.4f}"
    )

    # --------------------------------------------------------
    # Subject luminance / contrast
    # --------------------------------------------------------
    # Change luminance only. Multiplying RGB channels independently is
    # avoided so that local exposure/contrast cannot introduce a hue shift.
    #
    # v25: add adaptive subject exposure. A fixed +0.08 EV is too weak for
    # scenes where the detected subject is substantially darker than the
    # intended subject target. The correction is deliberately capped and
    # reduced when the subject already contains bright pixels.
    #
    # v28: apply through a feathered weight so the correction never
    # steps at the segmentation boundary.
    if float(np.max(subject_w)) > 1e-4:
        y_full = luminance(out)
        y = y_full[subject_hard] if np.any(subject_hard) else y_full

        subject_median = float(np.median(y))
        subject_p95 = float(np.percentile(y, 95))

        if subject_median > 1e-5:
            if subject_median < 0.22:
                subject_target = 0.22
            else:
                subject_target = subject_median

            adaptive_ev = clamp(
                math.log2(
                    subject_target
                    / max(subject_median, 1e-5)
                ) * 0.40,
                0.0,
                0.10,
            )

            # Protect bright uniforms / reflective objects.
            if subject_p95 > 0.70:
                adaptive_ev *= 0.25
            elif subject_p95 > 0.60:
                adaptive_ev *= 0.50

            effective_subject_ev = (
                params.subject_exposure
                + adaptive_ev
            )
        else:
            adaptive_ev = 0.0
            effective_subject_ev = params.subject_exposure
            subject_target = 0.22

        y2 = (
            (y_full - 0.18)
            * params.subject_contrast
            + 0.18
        )
        y2 *= 2.0 ** effective_subject_ev
        y2 = np.clip(y2, 0, 1)

        # ----------------------------------------------------
        # v26: subject-internal shadow lift
        # ----------------------------------------------------
        dark_weight = np.clip(
            (0.34 - y2) / 0.34,
            0.0,
            1.0,
        )

        dark_weight *= np.clip(
            (0.45 - y2) / 0.25,
            0.0,
            1.0,
        )

        shadow_gain = (
            1.0
            + 0.07 * dark_weight
        )

        y3 = np.clip(
            y2 * shadow_gain,
            0,
            1,
        )

        shadow_pixels_before = y2[subject_hard] < 0.20 if np.any(subject_hard) else np.array([], dtype=bool)
        shadow_pixels_after = y3[subject_hard] < 0.20 if np.any(subject_hard) else np.array([], dtype=bool)

        out = apply_luma_ratio(out, y3, subject_w)

        shadow_median_before = (
            float(np.median(y2[subject_hard][shadow_pixels_before]))
            if np.any(shadow_pixels_before)
            else None
        )
        shadow_median_after = (
            float(np.median(y3[subject_hard][shadow_pixels_before]))
            if np.any(shadow_pixels_before)
            else None
        )

        print(
            f"Adaptive subject exposure: "
            f"base {params.subject_exposure:+.3f} EV, "
            f"adaptive {adaptive_ev:+.3f} EV, "
            f"subject median {subject_median:.3f} -> "
            f"target {min(subject_target, 0.22):.3f}"
        )

        print(
            f"Subject shadow lift: "
            f"median<0.20 pixels "
            f"{int(np.count_nonzero(shadow_pixels_before))} -> "
            f"{int(np.count_nonzero(shadow_pixels_after))}, "
            f"median "
            f"{shadow_median_before:.3f} -> "
            f"{shadow_median_after:.3f}"
            if shadow_median_before is not None
            else "Subject shadow lift: no subject pixels below 0.20"
        )

    # --------------------------------------------------------
    # v27: face shadow lift
    # --------------------------------------------------------
    face = masks.get("face")
    face_confidence = float(masks.get("face_confidence", 0.0))

    if face is not None and np.any(face) and face_confidence >= 0.48:
        face_w = soften_mask(face, max(radius * 0.65, 8.0))
        y = luminance(out)
        face_median_before = float(np.median(y[face.astype(bool)]))

        # Strongest below 0.18, fading to zero by 0.36.
        lift_w = np.clip(
            (0.36 - y) / 0.18,
            0.0,
            1.0,
        )

        # Confidence limits the maximum lift.  At confidence 1 the
        # maximum multiplicative gain is about +0.12 EV equivalent.
        max_gain = 1.0 + 0.06 * clamp(face_confidence, 0.0, 1.0)
        gain = 1.0 + (max_gain - 1.0) * lift_w
        y2 = np.clip(y * gain, 0, 1)

        out = apply_luma_ratio(out, y2, face_w)

        face_median_after = float(np.median(y2[face.astype(bool)]))

        print(
            f"Face shadow lift: confidence {face_confidence:.3f}, "
            f"area {float(np.mean(face)):.4f}, "
            f"median {face_median_before:.3f} -> "
            f"{face_median_after:.3f}"
        )
    else:
        print(
            f"Face shadow lift: skipped "
            f"(confidence {face_confidence:.3f})"
        )

    # --------------------------------------------------------
    # Background suppression
    # --------------------------------------------------------
    # Again, only luminance is changed. Soft weight prevents a dark
    # halo from forming against the lifted subject.
    if float(np.max(background_w)) > 1e-4 and params.background_suppression > 0:
        y = luminance(out)
        y2 = np.clip(
            y * (1.0 - params.background_suppression),
            0,
            1,
        )
        out = apply_luma_ratio(out, y2, background_w)

    # --------------------------------------------------------
    # Saturation regions
    # --------------------------------------------------------
    # Combine all saturation adjustments into one operation per pixel.
    # Soft masks so grass / skin / water do not print a class edge.
    # Priority: green < water < skin.
    green_w = soften_mask(masks["green"], radius)
    water_w = soften_mask(masks["water"], radius)
    skin_w = soften_mask(masks["skin"], max(radius * 0.75, 8.0))

    saturation_factor = np.ones(
        out.shape[:2],
        dtype=np.float32,
    )

    saturation_factor = (
        saturation_factor * (1.0 - green_w)
        + params.green_saturation * green_w
    )
    saturation_factor = (
        saturation_factor * (1.0 - water_w)
        + params.water_saturation * water_w
    )
    saturation_factor = (
        saturation_factor * (1.0 - skin_w)
        + params.skin_saturation * skin_w
    )

    delta = np.abs(saturation_factor - 1.0)
    if float(np.max(delta)) > 1e-5:
        y = luminance(out)
        out = (
            y[..., None]
            + (out - y[..., None])
            * saturation_factor[..., None]
        )
        out = np.clip(out, 0, 1)

    # --------------------------------------------------------
    # Upper bright area
    # --------------------------------------------------------
    # Brightness only; chroma ratios are preserved.
    upper_w = soften_mask(masks["upper_bright"], radius)

    if float(np.max(upper_w)) > 1e-4:
        y = luminance(out)

        lift = (
            1.0
            + params.upper_brightness
            * np.clip(
                (0.80 - y) / 0.80,
                0,
                1,
            )
        )

        y2 = np.clip(y * lift, 0, 1)
        out = apply_luma_ratio(out, y2, upper_w)

    return np.clip(
        out,
        0,
        1,
    )


def print_region_color_drift(
    before: np.ndarray,
    after: np.ndarray,
) -> None:
    """Print global color/chroma changes caused by local processing."""

    b = calculate_stats(before)
    a = calculate_stats(after)

    print()
    print("Region color drift:")
    print(
        f"  R/G               : "
        f"{b.rg_ratio:.4f} -> {a.rg_ratio:.4f} "
        f"(delta {a.rg_ratio - b.rg_ratio:+.4f})"
    )
    print(
        f"  G/B               : "
        f"{b.gb_ratio:.4f} -> {a.gb_ratio:.4f} "
        f"(delta {a.gb_ratio - b.gb_ratio:+.4f})"
    )
    print(
        f"  saturation        : "
        f"{b.saturation_ratio * 100:.3f}% -> "
        f"{a.saturation_ratio * 100:.3f}% "
        f"(delta {(a.saturation_ratio - b.saturation_ratio) * 100:+.3f}pp)"
    )


def calculate_region_stats(
    image: np.ndarray,
    masks: dict[str, np.ndarray],
) -> RegionStats:

    y = luminance(image)

    subject = masks["subject"]
    background = masks["background"]

    subject_median = None
    background_median = None

    if np.any(subject):
        subject_median = float(
            np.median(
                y[subject]
            )
        )

    if np.any(background):
        background_median = float(
            np.median(
                y[background]
            )
        )

    return RegionStats(
        subject_median=subject_median,
        background_median=background_median,
        subject_area=float(
            np.mean(subject)
        ),
        background_area=float(
            np.mean(background)
        ),
    )
