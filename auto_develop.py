#!/usr/bin/env python3

import argparse
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import imageio.v3 as iio
import numpy as np
import rawpy
import torch
from PIL import Image
from torchvision.models.segmentation import (
    deeplabv3_mobilenet_v3_large,
    DeepLabV3_MobileNet_V3_Large_Weights,
)


# ============================================================
# Configuration
# ============================================================

RAW_EXTENSIONS = {
    ".cr2",
    ".cr3",
    ".nef",
    ".nrw",
    ".arw",
    ".raf",
    ".orf",
    ".rw2",
    ".dng",
    ".pef",
    ".srw",
}

VOC_CLASSES = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]

SUBJECT_CLASSES = {
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "cow",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "train",
}


# ============================================================
# Logging
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

logger = logging.getLogger(__name__)


# ============================================================
# Data classes
# ============================================================

@dataclass
class ImageAnalysis:
    mean: float
    median: float
    p01: float
    p05: float
    p95: float
    p99: float
    shadow_ratio: float
    highlight_ratio: float
    dynamic_range: float
    mean_saturation: float
    edge_density: float
    warm_ratio: float


@dataclass
class SubjectCandidate:
    class_name: str
    mask: np.ndarray
    confidence: float
    area: float
    center_score: float
    saliency_score: float
    contrast_score: float
    color_score: float
    importance: float


@dataclass
class SceneProfile:
    name: str

    ev_bias: float
    contrast_bias: float
    saturation_bias: float

    highlight_recovery: float
    shadow_recovery: float

    denoise: float
    sharpen: float

    sky_exposure: float
    sky_saturation: float

    subject_exposure: float
    subject_saturation: float

    clahe: float


# ============================================================
# Utility
# ============================================================

def normalize_map(data):

    data = data.astype(np.float32)

    low = np.percentile(data, 2)
    high = np.percentile(data, 98)

    if high <= low:
        return np.zeros_like(data)

    result = (
        (data - low)
        / (high - low)
    )

    return np.clip(
        result,
        0.0,
        1.0,
    )


def luminance(image):

    return (
        image[:, :, 0] * 0.2126
        + image[:, :, 1] * 0.7152
        + image[:, :, 2] * 0.0722
    )


def create_center_weight(
    height,
    width,
):

    y, x = np.mgrid[
        0:height,
        0:width,
    ]

    cx = (
        width - 1
    ) / 2.0

    cy = (
        height - 1
    ) / 2.0

    dx = (
        x - cx
    ) / max(
        width / 2.0,
        1.0,
    )

    dy = (
        y - cy
    ) / max(
        height / 2.0,
        1.0,
    )

    distance = np.sqrt(
        dx * dx
        + dy * dy
    )

    weight = (
        1.0
        - np.clip(
            distance,
            0.0,
            1.0,
        )
    )

    return weight.astype(
        np.float32
    )


# ============================================================
# RAW loading
# ============================================================

def load_raw(filename):

    logger.info(
        "Loading RAW: %s",
        filename,
    )

    with rawpy.imread(
        str(filename)
    ) as raw:

        rgb = raw.postprocess(
            use_camera_wb=True,
            no_auto_bright=True,
            output_bps=16,
            output_color=rawpy.ColorSpace.sRGB,
            demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD,
            highlight_mode=2,
            half_size=False,
            four_color_rgb=False,
        )

        rgb = (
            rgb.astype(
                np.float32
            )
            / 65535.0
        )

        metadata = {
            "make":
                getattr(
                    raw,
                    "camera_make",
                    "",
                ),

            "model":
                getattr(
                    raw,
                    "camera_model",
                    "",
                ),

            "iso":
                getattr(
                    raw,
                    "iso_speed",
                    None,
                ),

            "width":
                rgb.shape[1],

            "height":
                rgb.shape[0],
        }

    return (
        np.clip(
            rgb,
            0.0,
            1.0,
        ),
        metadata,
    )


# ============================================================
# Image Analysis
# ============================================================

def analyze_image(image):

    lum = luminance(
        image
    )

    mean = float(
        np.mean(lum)
    )

    median = float(
        np.median(lum)
    )

    p01 = float(
        np.percentile(
            lum,
            1,
        )
    )

    p05 = float(
        np.percentile(
            lum,
            5,
        )
    )

    p95 = float(
        np.percentile(
            lum,
            95,
        )
    )

    p99 = float(
        np.percentile(
            lum,
            99,
        )
    )

    shadow_ratio = float(
        np.mean(
            lum < 0.05
        )
    )

    highlight_ratio = float(
        np.mean(
            lum > 0.95
        )
    )

    dynamic_range = (
        p95 - p05
    )

    saturation = (
        np.max(
            image,
            axis=2,
        )
        - np.min(
            image,
            axis=2,
        )
    )

    mean_saturation = float(
        np.mean(
            saturation
        )
    )

    gray8 = np.clip(
        lum * 255.0,
        0,
        255,
    ).astype(
        np.uint8
    )

    edges = cv2.Canny(
        gray8,
        50,
        150,
    )

    edge_density = float(
        np.mean(
            edges > 0
        )
    )

    r = image[:, :, 0]
    g = image[:, :, 1]
    b = image[:, :, 2]

    warm = (
        (r > b * 1.15)
        & (r > g * 1.03)
        & (r > 0.15)
    )

    warm_ratio = float(
        np.mean(warm)
    )

    return ImageAnalysis(
        mean=mean,
        median=median,
        p01=p01,
        p05=p05,
        p95=p95,
        p99=p99,
        shadow_ratio=shadow_ratio,
        highlight_ratio=highlight_ratio,
        dynamic_range=dynamic_range,
        mean_saturation=mean_saturation,
        edge_density=edge_density,
        warm_ratio=warm_ratio,
    )


# ============================================================
# Semantic Segmentation
# ============================================================

class SemanticSegmenter:

    def __init__(
        self,
        device=None,
    ):

        if device is None:

            device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        self.device = torch.device(
            device
        )

        logger.info(
            "Loading DeepLabV3 "
            "MobileNet V3 Large on %s",
            self.device,
        )

        weights = (
            DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
        )

        self.model = (
            deeplabv3_mobilenet_v3_large(
                weights=weights
            )
        )

        self.model.to(
            self.device
        )

        self.model.eval()

        self.transforms = (
            weights.transforms()
        )

    def predict(
        self,
        image,
    ):

        h, w = image.shape[:2]

        max_size = 768

        scale = min(
            1.0,
            max_size / max(
                h,
                w,
            ),
        )

        if scale < 1.0:

            small = cv2.resize(
                image,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_AREA,
            )

        else:

            small = image

        pil = Image.fromarray(
            np.clip(
                small * 255.0,
                0,
                255,
            ).astype(
                np.uint8
            )
        )

        tensor = self.transforms(
            pil
        )

        tensor = tensor.unsqueeze(
            0
        ).to(
            self.device
        )

        with torch.no_grad():

            output = self.model(
                tensor
            )["out"][0]

        probabilities = torch.softmax(
            output,
            dim=0,
        )

        prediction = torch.argmax(
            probabilities,
            dim=0,
        ).cpu().numpy()

        confidence = torch.max(
            probabilities,
            dim=0,
        ).values.cpu().numpy()

        prediction = cv2.resize(
            prediction.astype(
                np.uint8
            ),
            (
                w,
                h,
            ),
            interpolation=cv2.INTER_NEAREST,
        )

        confidence = cv2.resize(
            confidence.astype(
                np.float32
            ),
            (
                w,
                h,
            ),
            interpolation=cv2.INTER_LINEAR,
        )

        masks = {}

        for class_id, class_name in enumerate(
            VOC_CLASSES
        ):

            if (
                class_name
                == "background"
            ):
                continue

            mask = np.where(
                prediction
                == class_id,
                confidence,
                0.0,
            ).astype(
                np.float32
            )

            masks[
                class_name
            ] = mask

        return masks


# ============================================================
# Sky
# ============================================================

def create_sky_mask(
    image,
):

    h, w = image.shape[:2]

    r = image[:, :, 0]
    g = image[:, :, 1]
    b = image[:, :, 2]

    y = np.arange(
        h,
        dtype=np.float32,
    )[:, None]

    top_weight = (
        1.0
        - np.clip(
            y / (
                h * 0.65
            ),
            0.0,
            1.0,
        )
    )

    blue_score = (
        b
        - 0.5 * r
        - 0.2 * g
    )

    bright = luminance(
        image
    )

    mask = (
        (blue_score > 0.05)
        & (bright > 0.15)
        & (top_weight > 0.15)
    )

    return (
        mask.astype(
            np.float32
        )
        * top_weight
    )


# ============================================================
# Saliency
# ============================================================

def calculate_saliency_map(
    image,
):

    gray = luminance(
        image
    ).astype(
        np.float32
    )

    blurred15 = cv2.GaussianBlur(
        gray,
        (0, 0),
        sigmaX=15,
    )

    local_contrast = np.abs(
        gray
        - blurred15
    )

    local_contrast = normalize_map(
        local_contrast
    )

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

    edge = np.sqrt(
        gx * gx
        + gy * gy
    )

    edge = normalize_map(
        edge
    )

    colorfulness = (
        np.max(
            image,
            axis=2,
        )
        - np.min(
            image,
            axis=2,
        )
    )

    colorfulness = normalize_map(
        colorfulness
    )

    blurred25 = cv2.GaussianBlur(
        gray,
        (0, 0),
        sigmaX=25,
    )

    brightness_difference = np.abs(
        gray
        - blurred25
    )

    brightness_difference = normalize_map(
        brightness_difference
    )

    h, w = gray.shape

    center_weight = create_center_weight(
        h,
        w,
    )

    center_weight = (
        0.65
        + 0.35 * center_weight
    )

    saliency = (
        local_contrast * 0.30
        + edge * 0.25
        + colorfulness * 0.15
        + brightness_difference * 0.20
        + center_weight * 0.10
    )

    saliency = cv2.GaussianBlur(
        saliency,
        (0, 0),
        sigmaX=7,
    )

    return normalize_map(
        saliency
    )


# ============================================================
# Subject Ranking
# ============================================================

def calculate_subject_importance(
    image,
    class_name,
    mask,
    saliency,
):

    strong_mask = (
        mask > 0.35
    )

    if not np.any(
        strong_mask
    ):
        return None

    h, w = mask.shape

    area = float(
        np.mean(
            strong_mask
        )
    )

    confidence = float(
        np.mean(
            mask[
                strong_mask
            ]
        )
    )

    # --------------------------------------------------------
    # Center
    # --------------------------------------------------------

    center_weight = create_center_weight(
        h,
        w,
    )

    center_score = float(
        np.mean(
            center_weight[
                strong_mask
            ]
        )
    )

    # --------------------------------------------------------
    # Saliency
    # --------------------------------------------------------

    saliency_score = float(
        np.mean(
            saliency[
                strong_mask
            ]
        )
    )

    # --------------------------------------------------------
    # Local contrast
    # --------------------------------------------------------

    gray = luminance(
        image
    )

    blurred = cv2.GaussianBlur(
        gray,
        (0, 0),
        sigmaX=15,
    )

    local_contrast = np.abs(
        gray
        - blurred
    )

    local_contrast = normalize_map(
        local_contrast
    )

    contrast_score = float(
        np.mean(
            local_contrast[
                strong_mask
            ]
        )
    )

    # --------------------------------------------------------
    # Color
    # --------------------------------------------------------

    saturation = (
        np.max(
            image,
            axis=2,
        )
        - np.min(
            image,
            axis=2,
        )
    )

    saturation = normalize_map(
        saturation
    )

    color_score = float(
        np.mean(
            saturation[
                strong_mask
            ]
        )
    )

    # --------------------------------------------------------
    # Area score
    # --------------------------------------------------------

    # 小さすぎる対象は主役になりにくい
    # 大きすぎる背景領域も主役としては少し抑える

    if area < 0.002:

        area_score = 0.25

    elif area < 0.20:

        area_score = (
            area / 0.20
        )

    else:

        area_score = max(
            0.4,
            1.0
            - (
                area
                - 0.20
            ),
        )

    # --------------------------------------------------------
    # Class prior
    # --------------------------------------------------------

    class_prior = {
        "person": 1.15,
        "animal": 1.05,
        "car": 1.00,
        "train": 1.00,
        "boat": 1.00,
        "bicycle": 1.00,
        "motorbike": 1.00,
        "bird": 1.00,
        "aeroplane": 1.00,
        "pottedplant": 0.90,
        "bottle": 0.85,
    }.get(
        class_name,
        1.00,
    )

    # --------------------------------------------------------
    # Importance
    # --------------------------------------------------------

    importance = (
        confidence * 0.20
        + center_score * 0.18
        + saliency_score * 0.28
        + contrast_score * 0.15
        + color_score * 0.09
        + area_score * 0.10
    )

    importance *= class_prior

    importance = float(
        np.clip(
            importance,
            0.0,
            1.0,
        )
    )

    return SubjectCandidate(
        class_name=class_name,
        mask=mask,
        confidence=confidence,
        area=area,
        center_score=center_score,
        saliency_score=saliency_score,
        contrast_score=contrast_score,
        color_score=color_score,
        importance=importance,
    )


def rank_subjects(
    image,
    masks,
    saliency,
):

    candidates = []

    for class_name, mask in masks.items():

        if (
            class_name
            not in SUBJECT_CLASSES
        ):
            continue

        candidate = (
            calculate_subject_importance(
                image,
                class_name,
                mask,
                saliency,
            )
        )

        if candidate is not None:

            if candidate.area >= 0.001:

                candidates.append(
                    candidate
                )

    candidates.sort(
        key=lambda x:
            x.importance,
        reverse=True,
    )

    # 上位候補だけ表示
    for index, candidate in enumerate(
        candidates[:5]
    ):

        logger.info(
            "Subject #%d: "
            "%s "
            "importance=%.3f "
            "confidence=%.3f "
            "area=%.2f%% "
            "saliency=%.3f",
            index + 1,
            candidate.class_name,
            candidate.importance,
            candidate.confidence,
            candidate.area * 100.0,
            candidate.saliency_score,
        )

    return candidates


# ============================================================
# Combined attention
# ============================================================

def create_subject_attention(
    subject,
):

    mask = subject.mask.copy()

    mask = np.clip(
        (
            mask
            - 0.20
        )
        / 0.60,
        0.0,
        1.0,
    )

    kernel = np.ones(
        (9, 9),
        np.uint8,
    )

    mask = cv2.dilate(
        mask,
        kernel,
        iterations=1,
    )

    mask = cv2.GaussianBlur(
        mask,
        (0, 0),
        sigmaX=8,
    )

    return np.clip(
        mask,
        0.0,
        1.0,
    )


def combine_subject_and_saliency(
    subject_attention,
    saliency,
    importance,
):

    # 主役が明確なほどsubjectの比率を上げる
    subject_weight = (
        0.45
        + 0.25 * importance
    )

    saliency_weight = (
        1.0
        - subject_weight
    )

    combined = (
        subject_attention
        * subject_weight
        + saliency
        * saliency_weight
    )

    return normalize_map(
        combined
    )


# ============================================================
# Scene Classification
# ============================================================

def classify_scene(
    image,
    analysis,
    masks,
    subjects,
):

    scores = {
        "portrait": 0.0,
        "night": 0.0,
        "sunset": 0.0,
        "landscape": 0.0,
        "city": 0.0,
        "indoor": 0.0,
        "general": 0.30,
    }

    sky_mask = create_sky_mask(
        image
    )

    sky_ratio = float(
        np.mean(
            sky_mask > 0.25
        )
    )

    person_ratio = 0.0

    if "person" in masks:

        person_ratio = float(
            np.mean(
                masks[
                    "person"
                ] > 0.25
            )
        )

    vehicle_ratio = 0.0

    for name in (
        "car",
        "bus",
        "train",
        "motorbike",
    ):

        if name in masks:

            vehicle_ratio += float(
                np.mean(
                    masks[name]
                    > 0.25
                )
            )

    # --------------------------------------------------------
    # Portrait
    # --------------------------------------------------------

    if person_ratio > 0.005:

        scores["portrait"] += 0.55

    if person_ratio > 0.02:

        scores["portrait"] += 0.20

    if (
        subjects
        and subjects[0].class_name
        == "person"
    ):

        scores["portrait"] += 0.25

    # --------------------------------------------------------
    # Night
    # --------------------------------------------------------

    if analysis.mean < 0.20:

        scores["night"] += 0.45

    elif analysis.mean < 0.28:

        scores["night"] += 0.20

    if analysis.shadow_ratio > 0.25:

        scores["night"] += 0.25

    if (
        analysis.highlight_ratio
        > 0.005
        and analysis.edge_density
        > 0.04
    ):

        scores["night"] += 0.20

    # --------------------------------------------------------
    # Sunset
    # --------------------------------------------------------

    if sky_ratio > 0.10:

        scores["sunset"] += 0.25

    if analysis.warm_ratio > 0.12:

        scores["sunset"] += 0.35

    if (
        analysis.warm_ratio > 0.20
        and sky_ratio > 0.20
    ):

        scores["sunset"] += 0.30

    # --------------------------------------------------------
    # Landscape
    # --------------------------------------------------------

    if sky_ratio > 0.15:

        scores["landscape"] += 0.35

    if (
        sky_ratio > 0.25
        and person_ratio < 0.02
    ):

        scores["landscape"] += 0.25

    if analysis.dynamic_range > 0.45:

        scores["landscape"] += 0.10

    # --------------------------------------------------------
    # City
    # --------------------------------------------------------

    if vehicle_ratio > 0.005:

        scores["city"] += 0.25

    if analysis.edge_density > 0.06:

        scores["city"] += 0.25

    if (
        sky_ratio < 0.25
        and analysis.edge_density
        > 0.08
    ):

        scores["city"] += 0.25

    # --------------------------------------------------------
    # Indoor
    # --------------------------------------------------------

    if sky_ratio < 0.05:

        scores["indoor"] += 0.25

    if (
        0.15
        < analysis.mean
        < 0.60
    ):

        scores["indoor"] += 0.20

    if analysis.warm_ratio > 0.10:

        scores["indoor"] += 0.15

    if analysis.edge_density > 0.04:

        scores["indoor"] += 0.10

    # --------------------------------------------------------
    # Conflict correction
    # --------------------------------------------------------

    if person_ratio > 0.05:

        scores["landscape"] *= 0.70
        scores["city"] *= 0.70
        scores["indoor"] *= 0.70

    if scores["night"] > 0.65:

        scores["sunset"] *= 0.50

    scene_name = max(
        scores,
        key=scores.get,
    )

    logger.info(
        "Scene classification: "
        "%s (score=%.3f)",
        scene_name,
        scores[scene_name],
    )

    logger.info(
        "Scene scores: %s",
        ", ".join(
            f"{k}={v:.2f}"
            for k, v in scores.items()
        ),
    )

    return scene_name


# ============================================================
# Scene Profiles
# ============================================================

def get_scene_profile(
    scene_name,
):

    profiles = {

        "portrait":
            SceneProfile(
                name="portrait",
                ev_bias=0.05,
                contrast_bias=-0.02,
                saturation_bias=-0.03,
                highlight_recovery=0.38,
                shadow_recovery=0.25,
                denoise=0.28,
                sharpen=0.55,
                sky_exposure=0.00,
                sky_saturation=1.00,
                subject_exposure=0.07,
                subject_saturation=0.97,
                clahe=1.1,
            ),

        "night":
            SceneProfile(
                name="night",
                ev_bias=0.05,
                contrast_bias=-0.03,
                saturation_bias=0.02,
                highlight_recovery=0.45,
                shadow_recovery=0.10,
                denoise=0.55,
                sharpen=0.40,
                sky_exposure=0.00,
                sky_saturation=1.02,
                subject_exposure=0.03,
                subject_saturation=1.00,
                clahe=1.0,
            ),

        "sunset":
            SceneProfile(
                name="sunset",
                ev_bias=-0.10,
                contrast_bias=0.02,
                saturation_bias=0.06,
                highlight_recovery=0.48,
                shadow_recovery=0.18,
                denoise=0.20,
                sharpen=0.65,
                sky_exposure=-0.07,
                sky_saturation=1.07,
                subject_exposure=0.02,
                subject_saturation=1.02,
                clahe=1.2,
            ),

        "landscape":
            SceneProfile(
                name="landscape",
                ev_bias=0.00,
                contrast_bias=0.05,
                saturation_bias=0.04,
                highlight_recovery=0.40,
                shadow_recovery=0.25,
                denoise=0.18,
                sharpen=0.80,
                sky_exposure=-0.03,
                sky_saturation=1.06,
                subject_exposure=0.02,
                subject_saturation=1.04,
                clahe=1.4,
            ),

        "city":
            SceneProfile(
                name="city",
                ev_bias=0.00,
                contrast_bias=0.04,
                saturation_bias=0.02,
                highlight_recovery=0.35,
                shadow_recovery=0.18,
                denoise=0.22,
                sharpen=0.75,
                sky_exposure=-0.02,
                sky_saturation=1.02,
                subject_exposure=0.02,
                subject_saturation=1.02,
                clahe=1.3,
            ),

        "indoor":
            SceneProfile(
                name="indoor",
                ev_bias=0.08,
                contrast_bias=0.00,
                saturation_bias=-0.02,
                highlight_recovery=0.35,
                shadow_recovery=0.28,
                denoise=0.30,
                sharpen=0.50,
                sky_exposure=0.00,
                sky_saturation=1.00,
                subject_exposure=0.05,
                subject_saturation=0.98,
                clahe=1.1,
            ),

        "general":
            SceneProfile(
                name="general",
                ev_bias=0.00,
                contrast_bias=0.00,
                saturation_bias=0.00,
                highlight_recovery=0.30,
                shadow_recovery=0.20,
                denoise=0.20,
                sharpen=0.65,
                sky_exposure=-0.02,
                sky_saturation=1.04,
                subject_exposure=0.03,
                subject_saturation=1.00,
                clahe=1.3,
            ),
    }

    return profiles.get(
        scene_name,
        profiles["general"],
    )


# ============================================================
# Basic adjustment
# ============================================================

def apply_exposure(
    image,
    ev,
):

    gain = 2.0 ** ev

    return np.clip(
        image * gain,
        0.0,
        1.0,
    )


def apply_contrast(
    image,
    contrast,
):

    mean = np.mean(
        luminance(image)
    )

    result = (
        (
            image
            - mean
        )
        * contrast
        + mean
    )

    return np.clip(
        result,
        0.0,
        1.0,
    )


def apply_saturation(
    image,
    saturation,
):

    hsv = cv2.cvtColor(
        image.astype(
            np.float32
        ),
        cv2.COLOR_RGB2HSV,
    )

    hsv[:, :, 1] *= (
        saturation
    )

    hsv[:, :, 1] = np.clip(
        hsv[:, :, 1],
        0.0,
        1.0,
    )

    result = cv2.cvtColor(
        hsv,
        cv2.COLOR_HSV2RGB,
    )

    return np.clip(
        result,
        0.0,
        1.0,
    )


def apply_auto_white_balance(
    image,
):

    result = image.copy()

    r = result[:, :, 0]
    g = result[:, :, 1]
    b = result[:, :, 2]

    r_mean = np.mean(r)
    g_mean = np.mean(g)
    b_mean = np.mean(b)

    mean_rgb = (
        r_mean
        + g_mean
        + b_mean
    ) / 3.0

    if r_mean > 1e-5:

        result[:, :, 0] *= (
            mean_rgb
            / r_mean
        )

    if g_mean > 1e-5:

        result[:, :, 1] *= (
            mean_rgb
            / g_mean
        )

    if b_mean > 1e-5:

        result[:, :, 2] *= (
            mean_rgb
            / b_mean
        )

    return np.clip(
        result,
        0.0,
        1.0,
    )


# ============================================================
# Highlight / Shadow
# ============================================================

def apply_highlight_recovery(
    image,
    strength=0.35,
):

    lum = luminance(
        image
    )

    mask = np.clip(
        (
            lum
            - 0.65
        )
        / 0.35,
        0.0,
        1.0,
    )

    mask *= strength

    result = (
        image
        * (
            1.0
            - mask[:, :, None]
        )
        + np.sqrt(
            np.clip(
                image,
                0.0,
                1.0,
            )
        )
        * mask[:, :, None]
    )

    return np.clip(
        result,
        0.0,
        1.0,
    )


def apply_shadow_recovery(
    image,
    strength=0.25,
):

    lum = luminance(
        image
    )

    mask = np.clip(
        (
            0.35
            - lum
        )
        / 0.35,
        0.0,
        1.0,
    )

    mask *= strength

    result = (
        image
        + (
            1.0
            - image
        )
        * mask[:, :, None]
        * 0.15
    )

    return np.clip(
        result,
        0.0,
        1.0,
    )


# ============================================================
# Local adjustment
# ============================================================

def apply_local_exposure(
    image,
    mask,
    ev,
):

    gain = 2.0 ** ev

    result = (
        image
        * (
            1.0
            + (
                gain
                - 1.0
            )
            * mask[:, :, None]
        )
    )

    return np.clip(
        result,
        0.0,
        1.0,
    )


def apply_local_saturation(
    image,
    mask,
    saturation,
):

    hsv = cv2.cvtColor(
        image.astype(
            np.float32
        ),
        cv2.COLOR_RGB2HSV,
    )

    hsv[:, :, 1] *= (
        1.0
        + (
            saturation
            - 1.0
        )
        * mask
    )

    hsv[:, :, 1] = np.clip(
        hsv[:, :, 1],
        0.0,
        1.0,
    )

    result = cv2.cvtColor(
        hsv,
        cv2.COLOR_HSV2RGB,
    )

    return np.clip(
        result,
        0.0,
        1.0,
    )


# ============================================================
# Subject / Background
# ============================================================

def apply_subject_background_separation(
    image,
    attention,
    strength,
):

    if np.max(
        attention
    ) <= 0.05:

        return image

    subject_gain = (
        1.0
        + 0.035
        * strength
        * attention
    )

    result = (
        image
        * subject_gain[:, :, None]
    )

    background = (
        1.0
        - attention
    )

    background_gain = (
        1.0
        - 0.015
        * strength
        * background
    )

    result *= (
        background_gain[
            :, :, None
        ]
    )

    return np.clip(
        result,
        0.0,
        1.0,
    )


# ============================================================
# Adaptive Tone
# ============================================================

def apply_adaptive_tone_curve(
    image,
    analysis,
):

    result = image.copy()

    if analysis.mean < 0.25:

        strength = np.clip(
            (
                0.25
                - analysis.mean
            )
            / 0.25,
            0.0,
            1.0,
        )

        gamma = (
            1.0
            - 0.12
            * strength
        )

        result = np.power(
            np.clip(
                result,
                0.0,
                1.0,
            ),
            gamma,
        )

    elif analysis.mean > 0.70:

        strength = np.clip(
            (
                analysis.mean
                - 0.70
            )
            / 0.30,
            0.0,
            1.0,
        )

        gamma = (
            1.0
            + 0.10
            * strength
        )

        result = np.power(
            np.clip(
                result,
                0.0,
                1.0,
            ),
            gamma,
        )

    if (
        analysis.dynamic_range
        < 0.35
    ):

        result = apply_contrast(
            result,
            1.10,
        )

    return np.clip(
        result,
        0.0,
        1.0,
    )


# ============================================================
# CLAHE
# ============================================================

def apply_clahe(
    image,
    clip_limit=1.3,
):

    lab = cv2.cvtColor(
        image.astype(
            np.float32
        ),
        cv2.COLOR_RGB2LAB,
    )

    l = np.clip(
        lab[:, :, 0]
        / 100.0
        * 255.0,
        0,
        255,
    ).astype(
        np.uint8
    )

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(
            8,
            8,
        ),
    )

    l = clahe.apply(l)

    lab[:, :, 0] = (
        l.astype(
            np.float32
        )
        / 255.0
        * 100.0
    )

    result = cv2.cvtColor(
        lab,
        cv2.COLOR_LAB2RGB,
    )

    return np.clip(
        result,
        0.0,
        1.0,
    )


# ============================================================
# Auto Score
# ============================================================

def calculate_auto_score(
    image,
    subject_mask=None,
    saliency=None,
    subject_importance=0.0,
):

    lum = luminance(
        image
    )

    # Highlight
    highlight_ratio = np.mean(
        lum > 0.98
    )

    highlight_score = np.clip(
        1.0
        - highlight_ratio
        * 8.0,
        0.0,
        1.0,
    )

    # Shadow
    shadow_ratio = np.mean(
        lum < 0.02
    )

    shadow_score = np.clip(
        1.0
        - shadow_ratio
        * 3.0,
        0.0,
        1.0,
    )

    # Midtone
    midtone_ratio = np.mean(
        (
            lum > 0.15
        )
        & (
            lum < 0.85
        )
    )

    midtone_score = np.clip(
        midtone_ratio,
        0.0,
        1.0,
    )

    # Contrast
    p05 = np.percentile(
        lum,
        5,
    )

    p95 = np.percentile(
        lum,
        95,
    )

    dynamic_range = (
        p95
        - p05
    )

    contrast_score = np.clip(
        dynamic_range
        / 0.75,
        0.0,
        1.0,
    )

    # Saturation
    saturation = (
        np.max(
            image,
            axis=2,
        )
        - np.min(
            image,
            axis=2,
        )
    )

    mean_saturation = np.mean(
        saturation
    )

    saturation_score = np.clip(
        1.0
        - abs(
            mean_saturation
            - 0.20
        )
        / 0.25,
        0.0,
        1.0,
    )

    # Subject
    if (
        subject_mask is not None
        and np.any(
            subject_mask > 0.2
        )
    ):

        mask = (
            subject_mask
            > 0.2
        )

        subject_lum = lum[
            mask
        ]

        if subject_lum.size > 0:

            subject_mean = (
                np.mean(
                    subject_lum
                )
            )

            subject_score = np.clip(
                1.0
                - abs(
                    subject_mean
                    - 0.50
                )
                / 0.50,
                0.0,
                1.0,
            )

        else:

            subject_score = 0.5

    else:

        subject_score = 0.5

    # Saliency
    if saliency is not None:

        salient = (
            saliency
            > 0.70
        )

        if np.any(
            salient
        ):

            salient_lum = lum[
                salient
            ]

            salient_mean = (
                np.mean(
                    salient_lum
                )
            )

            saliency_score = np.clip(
                1.0
                - abs(
                    salient_mean
                    - 0.50
                )
                / 0.50,
                0.0,
                1.0,
            )

        else:

            saliency_score = 0.5

    else:

        saliency_score = 0.5

    # 主役が明確な写真ほど
    # subject scoreを重視
    subject_weight = (
        0.10
        + 0.10
        * subject_importance
    )

    saliency_weight = (
        0.20
        - 0.05
        * subject_importance
    )

    score = (
        highlight_score * 0.22
        + shadow_score * 0.13
        + midtone_score * 0.13
        + contrast_score * 0.14
        + saturation_score * 0.08
        + subject_score
        * subject_weight
        + saliency_score
        * saliency_weight
    )

    return float(
        score
    )


# ============================================================
# Candidate Search
# ============================================================

def generate_candidates(
    profile,
):

    candidates = []

    ev_values = [
        -0.30,
        0.00,
        0.30,
    ]

    contrast_values = [
        -0.08,
        0.00,
        0.08,
    ]

    saturation_values = [
        -0.05,
        0.00,
        0.05,
    ]

    for ev in ev_values:

        for contrast_offset in (
            contrast_values
        ):

            for saturation_offset in (
                saturation_values
            ):

                candidates.append({

                    "ev":
                        ev
                        + profile.ev_bias,

                    "contrast":
                        1.0
                        + contrast_offset
                        + profile.contrast_bias,

                    "saturation":
                        1.0
                        + saturation_offset
                        + profile.saturation_bias,
                })

    return candidates


def render_candidate(
    image,
    params,
    profile,
):

    result = image.copy()

    result = apply_exposure(
        result,
        params["ev"],
    )

    result = apply_contrast(
        result,
        params["contrast"],
    )

    result = apply_saturation(
        result,
        params["saturation"],
    )

    result = apply_highlight_recovery(
        result,
        profile.highlight_recovery,
    )

    result = apply_shadow_recovery(
        result,
        profile.shadow_recovery,
    )

    return result


def auto_tune(
    image,
    analysis,
    profile,
    subject_mask=None,
    saliency=None,
    subject_importance=0.0,
):

    candidates = (
        generate_candidates(
            profile
        )
    )

    best_score = -math.inf

    best_image = image

    best_params = None

    for params in candidates:

        candidate = (
            render_candidate(
                image,
                params,
                profile,
            )
        )

        score = (
            calculate_auto_score(
                candidate,
                subject_mask,
                saliency,
                subject_importance,
            )
        )

        if score > best_score:

            best_score = score

            best_image = candidate

            best_params = params

    logger.info(
        "Auto tuning: "
        "EV=%+.2f "
        "contrast=%.2f "
        "saturation=%.2f "
        "score=%.4f",
        best_params["ev"],
        best_params["contrast"],
        best_params["saturation"],
        best_score,
    )

    return (
        best_image,
        best_params,
        best_score,
    )


# ============================================================
# Denoise
# ============================================================

def apply_denoise(
    image,
    strength=0.20,
):

    if strength <= 0:

        return image

    img8 = np.clip(
        image * 255.0,
        0,
        255,
    ).astype(
        np.uint8
    )

    sigma_color = (
        10.0
        + strength * 20.0
    )

    sigma_space = (
        5.0
        + strength * 10.0
    )

    result = cv2.bilateralFilter(
        img8,
        d=7,
        sigmaColor=sigma_color,
        sigmaSpace=sigma_space,
    )

    result = (
        result.astype(
            np.float32
        )
        / 255.0
    )

    return np.clip(
        result,
        0.0,
        1.0,
    )


# ============================================================
# Sharpen
# ============================================================

def apply_sharpen(
    image,
    amount=0.70,
):

    if amount <= 0:

        return image

    blurred = cv2.GaussianBlur(
        image,
        (0, 0),
        sigmaX=1.0,
    )

    result = (
        image
        + (
            image
            - blurred
        )
        * amount
    )

    return np.clip(
        result,
        0.0,
        1.0,
    )


# ============================================================
# Save
# ============================================================

def save_jpeg(
    image,
    filename,
    quality=95,
):

    image8 = np.clip(
        image * 255.0,
        0,
        255,
    ).astype(
        np.uint8
    )

    iio.imwrite(
        str(filename),
        image8,
        extension=".jpg",
        quality=quality,
    )

    logger.info(
        "Saved: %s",
        filename,
    )


# ============================================================
# Main Development Pipeline
# ============================================================

def auto_develop(
    filename,
    output_filename,
    segmenter,
):

    logger.info("=" * 70)

    logger.info(
        "Processing: %s",
        filename,
    )

    # --------------------------------------------------------
    # RAW
    # --------------------------------------------------------

    image, metadata = load_raw(
        filename
    )

    logger.info(
        "Camera: %s %s",
        metadata["make"],
        metadata["model"],
    )

    logger.info(
        "ISO: %s",
        metadata["iso"],
    )

    # --------------------------------------------------------
    # Analysis
    # --------------------------------------------------------

    analysis = analyze_image(
        image
    )

    logger.info(
        "Mean=%.3f "
        "Median=%.3f "
        "P05=%.3f "
        "P95=%.3f "
        "Shadow=%.3f "
        "Highlight=%.3f "
        "DR=%.3f",
        analysis.mean,
        analysis.median,
        analysis.p05,
        analysis.p95,
        analysis.shadow_ratio,
        analysis.highlight_ratio,
        analysis.dynamic_range,
    )

    # --------------------------------------------------------
    # Semantic Segmentation
    # --------------------------------------------------------

    try:

        masks = segmenter.predict(
            image
        )

    except Exception as e:

        logger.warning(
            "Semantic segmentation failed: %s",
            e,
        )

        masks = {}

    # --------------------------------------------------------
    # Saliency
    # --------------------------------------------------------

    logger.info(
        "Calculating saliency map..."
    )

    saliency = (
        calculate_saliency_map(
            image
        )
    )

    # --------------------------------------------------------
    # Subject Ranking
    # --------------------------------------------------------

    subjects = rank_subjects(
        image,
        masks,
        saliency,
    )

    if subjects:

        main_subject = subjects[0]

        logger.info(
            "Main subject: "
            "%s "
            "importance=%.3f",
            main_subject.class_name,
            main_subject.importance,
        )

        subject_attention = (
            create_subject_attention(
                main_subject
            )
        )

        attention = (
            combine_subject_and_saliency(
                subject_attention,
                saliency,
                main_subject.importance,
            )
        )

    else:

        main_subject = None

        attention = saliency

    # --------------------------------------------------------
    # Scene Classification
    # --------------------------------------------------------

    scene_name = classify_scene(
        image,
        analysis,
        masks,
        subjects,
    )

    profile = get_scene_profile(
        scene_name
    )

    logger.info(
        "Scene profile: %s",
        profile.name,
    )

    # --------------------------------------------------------
    # White Balance
    # --------------------------------------------------------

    image = (
        apply_auto_white_balance(
            image
        )
    )

    # --------------------------------------------------------
    # Auto tuning
    # --------------------------------------------------------

    subject_importance = (
        main_subject.importance
        if main_subject is not None
        else 0.0
    )

    (
        image,
        params,
        score,
    ) = auto_tune(
        image,
        analysis,
        profile,
        attention,
        saliency,
        subject_importance,
    )

    # --------------------------------------------------------
    # Sky
    # --------------------------------------------------------

    sky = create_sky_mask(
        image
    )

    sky_area = float(
        np.mean(
            sky > 0.25
        )
    )

    if sky_area > 0.01:

        logger.info(
            "Sky detected: %.2f%%",
            sky_area * 100.0,
        )

        image = (
            apply_local_exposure(
                image,
                sky,
                profile.sky_exposure,
            )
        )

        image = (
            apply_local_saturation(
                image,
                sky,
                profile.sky_saturation,
            )
        )

    # --------------------------------------------------------
    # Main Subject Adjustment
    # --------------------------------------------------------

    if main_subject is not None:

        subject_strength = (
            0.5
            + main_subject.importance
        )

        subject_ev = (
            profile.subject_exposure
            * subject_strength
        )

        image = (
            apply_local_exposure(
                image,
                main_subject.mask,
                subject_ev,
            )
        )

        subject_sat = (
            1.0
            + (
                profile.subject_saturation
                - 1.0
            )
            * subject_strength
        )

        image = (
            apply_local_saturation(
                image,
                main_subject.mask,
                subject_sat,
            )
        )

    # --------------------------------------------------------
    # Secondary Subjects
    # --------------------------------------------------------

    if len(subjects) > 1:

        for secondary in subjects[1:3]:

            # 副被写体は主役より弱く処理
            strength = (
                0.20
                + 0.30
                * secondary.importance
            )

            image = (
                apply_local_exposure(
                    image,
                    secondary.mask,
                    profile.subject_exposure
                    * strength,
                )
            )

    # --------------------------------------------------------
    # Person-specific protection
    # --------------------------------------------------------

    if "person" in masks:

        person = masks[
            "person"
        ]

        person_area = float(
            np.mean(
                person > 0.25
            )
        )

        if person_area > 0.002:

            logger.info(
                "Person detected: %.2f%%",
                person_area * 100.0,
            )

            image = (
                apply_local_saturation(
                    image,
                    person,
                    0.98,
                )
            )

    # --------------------------------------------------------
    # Plant
    # --------------------------------------------------------

    if "pottedplant" in masks:

        plant = masks[
            "pottedplant"
        ]

        plant_area = float(
            np.mean(
                plant > 0.25
            )
        )

        if plant_area > 0.002:

            image = (
                apply_local_saturation(
                    image,
                    plant,
                    1.05,
                )
            )

    # --------------------------------------------------------
    # Subject / Background Separation
    # --------------------------------------------------------

    separation_strength = (
        0.5
        + subject_importance
    )

    image = (
        apply_subject_background_separation(
            image,
            attention,
            separation_strength,
        )
    )

    # --------------------------------------------------------
    # Adaptive Tone
    # --------------------------------------------------------

    image = (
        apply_adaptive_tone_curve(
            image,
            analysis,
        )
    )

    # --------------------------------------------------------
    # CLAHE
    # --------------------------------------------------------

    image = apply_clahe(
        image,
        clip_limit=profile.clahe,
    )

    # --------------------------------------------------------
    # Denoise
    # --------------------------------------------------------

    iso = metadata.get(
        "iso",
        100,
    )

    if iso is None:

        iso = 100

    iso_factor = 1.0

    if iso >= 6400:

        iso_factor = 1.30

    elif iso >= 3200:

        iso_factor = 1.20

    elif iso >= 1600:

        iso_factor = 1.10

    elif iso >= 800:

        iso_factor = 1.05

    denoise_strength = min(
        profile.denoise
        * iso_factor,
        0.70,
    )

    logger.info(
        "Denoise strength: %.2f",
        denoise_strength,
    )

    image = apply_denoise(
        image,
        denoise_strength,
    )

    # --------------------------------------------------------
    # Sharpen
    # --------------------------------------------------------

    image = apply_sharpen(
        image,
        profile.sharpen,
    )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    image = np.clip(
        image,
        0.0,
        1.0,
    )

    save_jpeg(
        image,
        output_filename,
        quality=95,
    )

    logger.info(
        "Finished: %s",
        filename,
    )


# ============================================================
# RAW collection
# ============================================================

def collect_raw_files(
    input_path,
):

    input_path = Path(
        input_path
    )

    if input_path.is_file():

        if (
            input_path.suffix.lower()
            in RAW_EXTENSIONS
        ):

            return [
                input_path
            ]

        return []

    files = []

    for path in input_path.rglob(
        "*"
    ):

        if not path.is_file():
            continue

        if (
            path.suffix.lower()
            in RAW_EXTENSIONS
        ):

            files.append(
                path
            )

    return sorted(
        files
    )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Automatic RAW photo "
            "development v9"
        )
    )

    parser.add_argument(
        "input",
        help="RAW file or directory",
    )

    parser.add_argument(
        "-o",
        "--output",
        default="output",
        help="Output directory",
    )

    parser.add_argument(
        "--device",
        choices=[
            "cpu",
            "cuda",
        ],
        default=None,
        help="Inference device",
    )

    args = parser.parse_args()

    input_path = Path(
        args.input
    )

    output_dir = Path(
        args.output
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw_files = (
        collect_raw_files(
            input_path
        )
    )

    if not raw_files:

        logger.error(
            "No RAW files found."
        )

        sys.exit(1)

    logger.info(
        "Found %d RAW files.",
        len(raw_files),
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    try:

        segmenter = (
            SemanticSegmenter(
                device=args.device
            )
        )

    except Exception as e:

        logger.error(
            "Failed to load "
            "segmentation model: %s",
            e,
        )

        sys.exit(1)

    # --------------------------------------------------------
    # Process
    # --------------------------------------------------------

    success = 0
    failure = 0

    for raw_file in raw_files:

        output_filename = (
            output_dir
            / f"{raw_file.stem}.jpg"
        )

        try:

            auto_develop(
                raw_file,
                output_filename,
                segmenter,
            )

            success += 1

        except Exception:

            logger.exception(
                "Failed: %s",
                raw_file,
            )

            failure += 1

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    logger.info("=" * 70)

    logger.info(
        "Finished: success=%d failure=%d",
        success,
        failure,
    )


if __name__ == "__main__":

    main()