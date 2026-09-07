from __future__ import annotations

import math
import cv2
import numpy as np

from .constants import (
    VOC_CLASSES, ANIMAL_CLASSES, VEHICLE_CLASSES, PERSON_CLASSES,
    SubjectCandidate,
)
from .color import luminance
from .utils import clamp, soften_mask

def rank_subjects(
    class_map: np.ndarray,
    confidence: np.ndarray,
    saliency: np.ndarray,
    image: np.ndarray,
) -> list[SubjectCandidate]:

    h, w = class_map.shape

    y = luminance(image)

    candidates = []

    for class_id, label in enumerate(VOC_CLASSES):
        if class_id == 0:
            continue

        mask = class_map == class_id

        area = float(np.mean(mask))

        if area < 0.003:
            continue

        conf = float(
            np.mean(
                confidence[mask]
            )
        )

        ys, xs = np.where(mask)

        if len(xs) == 0:
            continue

        cx = float(np.mean(xs) / max(w - 1, 1))
        cy = float(np.mean(ys) / max(h - 1, 1))

        center_score = 1.0 - math.sqrt(
            (cx - 0.5) ** 2
            + (cy - 0.5) ** 2
        ) / 0.707

        center_score = clamp(
            center_score,
            0.0,
            1.0,
        )

        saliency_score = float(
            np.mean(saliency[mask])
        )

        subject_luma = float(
            np.mean(y[mask])
        )

        local_region = cv2.dilate(
            mask.astype(np.uint8),
            np.ones((15, 15), np.uint8),
        ).astype(bool)

        surrounding = (
            local_region
            & ~mask
        )

        if np.any(surrounding):
            surrounding_luma = float(
                np.mean(y[surrounding])
            )

            local_contrast = abs(
                subject_luma
                - surrounding_luma
            )
        else:
            local_contrast = 0.0

        local_contrast = clamp(
            local_contrast * 3.0,
            0.0,
            1.0,
        )

        colorfulness = float(
            np.mean(
                np.max(image[mask], axis=1)
                - np.min(image[mask], axis=1)
            )
        )

        colorfulness = clamp(
            colorfulness * 3.0,
            0.0,
            1.0,
        )

        if label in PERSON_CLASSES:
            prior = 1.15
        elif label in ANIMAL_CLASSES:
            prior = 1.05
        elif label in VEHICLE_CLASSES:
            prior = 1.00
        elif label == "pottedplant":
            prior = 0.90
        elif label == "bottle":
            prior = 0.85
        else:
            prior = 0.80

        score = (
            conf * 0.30
            + math.sqrt(area) * 0.20
            + center_score * 0.15
            + saliency_score * 0.15
            + local_contrast * 0.10
            + colorfulness * 0.10
        ) * prior

        candidates.append(
            SubjectCandidate(
                label=label,
                class_id=class_id,
                score=float(score),
                area=area,
                confidence=conf,
                center_score=center_score,
                saliency_score=saliency_score,
                local_contrast=local_contrast,
            )
        )

    candidates.sort(
        key=lambda x: x.score,
        reverse=True,
    )

    return candidates[:10]


def make_region_masks(
    class_map: np.ndarray,
    image: np.ndarray,
    subjects: list[SubjectCandidate],
) -> dict[str, np.ndarray]:

    h, w = class_map.shape

    masks: dict[str, np.ndarray] = {}

    person = class_map == VOC_CLASSES.index(
        "person"
    )

    animal = np.zeros_like(
        person,
        dtype=bool,
    )

    for name in ANIMAL_CLASSES:
        animal |= (
            class_map
            == VOC_CLASSES.index(name)
        )

    vehicle = np.zeros_like(
        person,
        dtype=bool,
    )

    for name in VEHICLE_CLASSES:
        vehicle |= (
            class_map
            == VOC_CLASSES.index(name)
        )

    plant = (
        class_map
        == VOC_CLASSES.index(
            "pottedplant"
        )
    )

    # Semantic subject mask. This is deliberately kept separate from
    # the fallback candidate mask so that the debug output can distinguish
    # a true semantic subject from a heuristic fallback.
    semantic_subject = (
        person
        | animal
        | vehicle
        | plant
    )

    candidate_subject = np.zeros_like(
        semantic_subject,
        dtype=bool,
    )

    if not np.any(semantic_subject) and subjects:
        candidate_subject = (
            class_map
            == subjects[0].class_id
        )

    # Effective subject used for local development.
    subject = (
        semantic_subject
        | candidate_subject
    )

    rgb8 = np.clip(
        image * 255,
        0,
        255,
    ).astype(np.uint8)

    hsv = cv2.cvtColor(
        rgb8,
        cv2.COLOR_RGB2HSV,
    )

    h_channel = hsv[..., 0]
    s_channel = hsv[..., 1]
    v_channel = hsv[..., 2]

    # Erode the person mask before building skin. Infield dirt is
    # orange-brown and sits in the same HSV range as skin; DeepLab
    # also tends to leak a few pixels onto the clay around feet and
    # elbows. Those leaked pixels used to form the visible halo.
    person_core = cv2.erode(
        person.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        iterations=1,
    ).astype(bool)

    skin = (
        person_core
        & (
            (h_channel < 25)
            | (h_channel > 170)
        )
        & (s_channel > 35)
        & (s_channel < 170)
        & (v_channel > 50)
        & (v_channel < 230)
    )

    # --------------------------------------------------------
    # v27: heuristic face candidate
    # --------------------------------------------------------
    # DeepLabV3 gives us a person mask, but not a face class.  Build a
    # conservative face candidate from skin-like pixels located in the
    # upper part of each connected person component.  This is deliberately
    # heuristic and is only used when confidence is sufficiently high.
    #
    # Keep this separate from `skin`: hands/arms can also be skin-like,
    # while the face correction should be restricted to the upper body.
    face_mask = np.zeros_like(person, dtype=bool)
    face_confidence = 0.0

    relaxed_skin = (
        person
        & (
            (h_channel < 28)
            | (h_channel > 165)
        )
        & (s_channel >= 20)
        & (v_channel >= 35)
    )

    person_u8 = person.astype(np.uint8)
    n_person, person_labels, person_stats, person_centroids = cv2.connectedComponentsWithStats(
        person_u8,
        connectivity=8,
    )

    face_scores = []

    for person_id in range(1, n_person):
        px, py, pw, ph, parea = person_stats[person_id]
        if parea < 100:
            continue

        component = person_labels == person_id
        upper_limit = py + int(ph * 0.55)

        candidate = relaxed_skin & component
        candidate &= np.indices(person.shape)[0] <= upper_limit

        n_candidate, candidate_labels, candidate_stats, candidate_centroids = cv2.connectedComponentsWithStats(
            candidate.astype(np.uint8),
            connectivity=8,
        )

        best_candidate = None
        best_score = 0.0

        for cid in range(1, n_candidate):
            area = int(candidate_stats[cid, cv2.CC_STAT_AREA])
            if area < 30 or area > max(int(parea * 0.20), 30):
                continue

            cx, cy = candidate_centroids[cid]
            rel_y = (cy - py) / max(ph, 1)
            rel_x = abs((cx - (px + pw * 0.5)) / max(pw, 1))

            # Prefer pixels around the upper 0-40% of the person and
            # reasonably close to the person's horizontal center.
            position_score = clamp(
                1.0 - abs(rel_y - 0.25) / 0.30,
                0.0,
                1.0,
            )
            center_score = clamp(
                1.0 - rel_x / 0.65,
                0.0,
                1.0,
            )

            component_mask = candidate_labels == cid
            skin_strength = float(
                np.mean(
                    np.clip((s_channel[component_mask] - 20) / 80.0, 0.0, 1.0)
                    * np.clip((v_channel[component_mask] - 35) / 100.0, 0.0, 1.0)
                )
            )

            area_score = clamp(
                math.sqrt(area / max(parea * 0.02, 1.0)),
                0.0,
                1.0,
            )

            score = (
                position_score * 0.40
                + center_score * 0.20
                + skin_strength * 0.25
                + area_score * 0.15
            )

            if score > best_score:
                best_score = score
                best_candidate = component_mask

        if best_candidate is not None and best_score >= 0.48:
            face_mask |= best_candidate
            face_scores.append(best_score)

    if face_scores:
        face_confidence = float(np.mean(face_scores))

    green = (
        (h_channel >= 30)
        & (h_channel <= 95)
        & (s_channel >= 45)
        & (v_channel >= 30)
    )

    blue = (
        (h_channel >= 80)
        & (h_channel <= 135)
        & (s_channel >= 40)
        & (v_channel >= 40)
    )

    y = luminance(image)

    yy = np.indices(
        (h, w)
    )[0] / max(h - 1, 1)

    upper_bright = (
        (yy < 0.45)
        & (y > np.percentile(y, 75))
    )

    gray = np.clip(
        y * 255,
        0,
        255,
    ).astype(np.uint8)

    gx = cv2.Sobel(
        gray,
        cv2.CV_32F,
        1,
        0,
        ksize=3,
    )

    gy = cv2.Sobel(
        gray,
        cv2.CV_32F,
        0,
        1,
        ksize=3,
    )

    texture = cv2.magnitude(
        gx,
        gy,
    )

    water = (
        blue
        & (yy > 0.35)
        & (texture < np.percentile(texture, 60))
    )

    background = ~subject

    masks["person"] = person
    masks["animal"] = animal
    masks["vehicle"] = vehicle
    masks["plant"] = plant
    masks["semantic_subject"] = semantic_subject
    masks["candidate_subject"] = candidate_subject
    masks["skin"] = skin
    masks["face"] = face_mask
    masks["face_confidence"] = np.array(face_confidence, dtype=np.float32)
    masks["green"] = green
    masks["blue"] = blue
    masks["water"] = water
    masks["upper_bright"] = upper_bright
    masks["subject"] = subject
    masks["background"] = background

    return masks
