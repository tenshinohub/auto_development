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
from PIL import Image

import torch
import torchvision
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
    format="[%(levelname)s] %(message)s",
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
    saturation: float
    edge_density: float
    warm_ratio: float


@dataclass
class SubjectCandidate:
    class_name: str
    class_id: int

    mask: np.ndarray

    confidence: float
    area: float

    center_score: float
    saliency_score: float
    contrast_score: float
    colorfulness_score: float

    importance: float


@dataclass
class SceneProfile:
    name: str

    exposure_bias: float
    contrast: float
    saturation: float

    highlight_recovery: float
    shadow_lift: float

    local_subject_strength: float
    background_suppression: float

    denoise_strength: float
    sharpen_strength: float


# ============================================================
# Utility
# ============================================================

def normalize_map(data):
    data = data.astype(np.float32)

    low = np.percentile(data, 2)
    high = np.percentile(data, 98)

    if high <= low:
        return np.zeros_like(data)

    result = (data - low) / (high - low)

    return np.clip(result, 0.0, 1.0)


def luminance(image):
    return (
        image[:, :, 0] * 0.2126
        + image[:, :, 1] * 0.7152
        + image[:, :, 2] * 0.0722
    )


def create_center_weight(h, w):
    y, x = np.mgrid[0:h, 0:w]

    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0

    dx = (x - cx) / max(w / 2.0, 1.0)
    dy = (y - cy) / max(h / 2.0, 1.0)

    distance = np.sqrt(dx * dx + dy * dy)

    weight = 1.0 - np.clip(distance, 0.0, 1.0)

    return weight.astype(np.float32)


# ============================================================
# RAW loading
# ============================================================

def load_raw(filename):
    logger.info("Loading RAW: %s", filename)

    with rawpy.imread(str(filename)) as raw:

        logger.info(
            "RAW camera: %s %s",
            getattr(raw, "camera_make", ""),
            getattr(raw, "camera_model", ""),
        )

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # gamma=(1, 1)
        #
        # This prevents the normal display gamma from being
        # applied at the RAW development stage.
        #
        # The resulting RGB image is therefore suitable for
        # linear-domain exposure/tone processing.
        # ----------------------------------------------------

        rgb = raw.postprocess(
            use_camera_wb=True,
            no_auto_bright=True,

            output_bps=16,
            output_color=rawpy.ColorSpace.sRGB,

            gamma=(1, 1),

            demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD,

            highlight_mode=2,

            half_size=False,
            four_color_rgb=False,
        )

        iso = getattr(raw, "iso_speed", 0)

        make = getattr(raw, "camera_make", "")
        model = getattr(raw, "camera_model", "")

    image = rgb.astype(np.float32) / 65535.0

    image = np.clip(image, 0.0, 1.0)

    metadata = {
        "make": make,
        "model": model,
        "iso": iso,
    }

    logger.info(
        "Linear RGB loaded: %dx%d ISO=%s",
        image.shape[1],
        image.shape[0],
        iso,
    )

    return image, metadata


# ============================================================
# Image analysis
# ============================================================

def analyze_image(image):
    y = luminance(image)

    mean = float(np.mean(y))
    median = float(np.median(y))

    p01 = float(np.percentile(y, 1))
    p05 = float(np.percentile(y, 5))
    p95 = float(np.percentile(y, 95))
    p99 = float(np.percentile(y, 99))

    shadow_ratio = float(np.mean(y < 0.03))
    highlight_ratio = float(np.mean(y > 0.95))

    dynamic_range = p95 - p05

    max_rgb = np.max(image, axis=2)
    min_rgb = np.min(image, axis=2)

    saturation = float(
        np.mean(
            (max_rgb - min_rgb)
            / np.maximum(max_rgb, 1e-6)
        )
    )

    gray = y.astype(np.float32)

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

    edge = np.sqrt(
        sobel_x * sobel_x
        + sobel_y * sobel_y
    )

    edge_density = float(
        np.mean(edge > np.percentile(edge, 75))
    )

    warm_ratio = float(
        np.mean(
            (image[:, :, 0] > image[:, :, 2] * 1.15)
            & (image[:, :, 0] > image[:, :, 1] * 0.95)
        )
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
        saturation=saturation,
        edge_density=edge_density,
        warm_ratio=warm_ratio,
    )


# ============================================================
# Semantic segmentation
# ============================================================

class SemanticSegmenter:

    def __init__(self, device=None):

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        if device == "cuda" and not torch.cuda.is_available():
            logger.warning(
                "CUDA requested but unavailable. Falling back to CPU."
            )
            device = "cpu"

        self.device = torch.device(device)

        logger.info(
            "Segmentation device: %s",
            self.device,
        )

        weights = DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT

        self.model = deeplabv3_mobilenet_v3_large(
            weights=weights
        )

        self.model.eval()
        self.model.to(self.device)

        self.transform = weights.transforms()

        self.max_size = 768

    @torch.inference_mode()
    def predict(self, image):

        h, w = image.shape[:2]

        scale = min(
            1.0,
            self.max_size / max(h, w)
        )

        if scale < 1.0:

            new_w = int(w * scale)
            new_h = int(h * scale)

            resized = cv2.resize(
                image,
                (new_w, new_h),
                interpolation=cv2.INTER_AREA,
            )

        else:
            resized = image

        uint8 = np.clip(
            resized * 255.0,
            0,
            255,
        ).astype(np.uint8)

        pil = Image.fromarray(uint8)

        tensor = self.transform(pil)

        tensor = tensor.unsqueeze(0)
        tensor = tensor.to(self.device)

        output = self.model(tensor)["out"]

        probabilities = torch.softmax(
            output,
            dim=1,
        )

        confidence, labels = torch.max(
            probabilities,
            dim=1,
        )

        labels = labels[0].cpu().numpy()
        confidence = confidence[0].cpu().numpy()

        labels = cv2.resize(
            labels.astype(np.uint8),
            (w, h),
            interpolation=cv2.INTER_NEAREST,
        )

        confidence = cv2.resize(
            confidence.astype(np.float32),
            (w, h),
            interpolation=cv2.INTER_LINEAR,
        )

        return labels, confidence


# ============================================================
# Masks
# ============================================================

def create_sky_mask(image):

    h, w = image.shape[:2]

    y = luminance(image)

    r = image[:, :, 0]
    g = image[:, :, 1]
    b = image[:, :, 2]

    blue_score = b - (r + g) * 0.5

    vertical = np.linspace(
        1.0,
        0.0,
        h,
        dtype=np.float32,
    )[:, None]

    mask = (
        (blue_score > 0.015)
        & (b > 0.20)
        & (vertical > 0.25)
        & (y > 0.15)
    )

    mask = mask.astype(np.float32)

    kernel = np.ones(
        (9, 9),
        np.uint8,
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
    )

    mask = cv2.GaussianBlur(
        mask,
        (0, 0),
        5,
    )

    return np.clip(mask, 0.0, 1.0)


# ============================================================
# Saliency
# ============================================================

def calculate_saliency_map(image):

    gray = luminance(image)

    # Local contrast
    blurred15 = cv2.GaussianBlur(
        gray,
        (0, 0),
        15,
    )

    local_contrast = np.abs(
        gray - blurred15
    )

    # Edge
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

    edge = np.sqrt(
        sobel_x * sobel_x
        + sobel_y * sobel_y
    )

    # Colorfulness
    colorfulness = (
        np.max(image, axis=2)
        - np.min(image, axis=2)
    )

    # Brightness distinctiveness
    blurred25 = cv2.GaussianBlur(
        gray,
        (0, 0),
        25,
    )

    brightness_difference = np.abs(
        gray - blurred25
    )

    local_contrast = normalize_map(local_contrast)
    edge = normalize_map(edge)
    colorfulness = normalize_map(colorfulness)
    brightness_difference = normalize_map(
        brightness_difference
    )

    h, w = gray.shape

    center_weight = (
        0.65
        + 0.35 * create_center_weight(h, w)
    )

    saliency = (
        local_contrast * 0.30
        + edge * 0.25
        + colorfulness * 0.15
        + brightness_difference * 0.20
        + center_weight * 0.10
    )

    saliency = cv2.GaussianBlur(
        saliency.astype(np.float32),
        (0, 0),
        7,
    )

    return normalize_map(saliency)


# ============================================================
# Subject ranking
# ============================================================

def calculate_subject_importance(
    image,
    mask,
    confidence,
    saliency,
    class_name,
):

    area = float(np.mean(mask > 0.5))

    if area <= 0:
        return 0.0

    ys, xs = np.where(mask > 0.5)

    if len(xs) == 0:
        return 0.0

    h, w = mask.shape

    cx = float(np.mean(xs) / max(w - 1, 1))
    cy = float(np.mean(ys) / max(h - 1, 1))

    dx = cx - 0.5
    dy = cy - 0.5

    distance = math.sqrt(
        dx * dx + dy * dy
    )

    center_score = 1.0 - np.clip(
        distance / 0.707,
        0.0,
        1.0,
    )

    confidence_score = float(
        np.mean(confidence[mask > 0.5])
    )

    saliency_score = float(
        np.mean(saliency[mask > 0.5])
    )

    gray = luminance(image)

    local_mean = cv2.GaussianBlur(
        gray,
        (0, 0),
        15,
    )

    local_contrast = np.abs(
        gray - local_mean
    )

    contrast_score = float(
        np.mean(
            normalize_map(local_contrast)[mask > 0.5]
        )
    )

    colorfulness = (
        np.max(image, axis=2)
        - np.min(image, axis=2)
    )

    colorfulness_score = float(
        np.mean(
            normalize_map(colorfulness)[mask > 0.5]
        )
    )

    area_score = np.sqrt(
        np.clip(area * 8.0, 0.0, 1.0)
    )

    class_prior = {
        "person": 1.15,

        "cat": 1.05,
        "dog": 1.05,
        "horse": 1.05,
        "bird": 1.05,
        "sheep": 1.05,
        "cow": 1.05,

        "car": 1.00,
        "train": 1.00,
        "bus": 1.00,
        "boat": 1.00,

        "bicycle": 1.00,
        "motorbike": 1.00,

        "aeroplane": 1.00,

        "pottedplant": 0.90,
        "bottle": 0.85,
    }.get(class_name, 1.0)

    importance = (
        area_score * 0.18
        + confidence_score * 0.25
        + center_score * 0.15
        + saliency_score * 0.22
        + contrast_score * 0.10
        + colorfulness_score * 0.10
    )

    importance *= class_prior

    return float(np.clip(
        importance,
        0.0,
        1.5,
    ))


def rank_subjects(
    image,
    labels,
    confidence,
    saliency,
):

    subjects = []

    for class_id, class_name in enumerate(VOC_CLASSES):

        if class_name not in SUBJECT_CLASSES:
            continue

        mask = (
            labels == class_id
        ).astype(np.float32)

        if np.mean(mask > 0.5) < 0.0005:
            continue

        subject_confidence = confidence * mask

        importance = calculate_subject_importance(
            image,
            mask,
            confidence,
            saliency,
            class_name,
        )

        candidate = SubjectCandidate(
            class_name=class_name,
            class_id=class_id,
            mask=mask,
            confidence=float(
                np.mean(
                    subject_confidence[mask > 0.5]
                )
            ),
            area=float(np.mean(mask > 0.5)),
            center_score=0.0,
            saliency_score=float(
                np.mean(
                    saliency[mask > 0.5]
                )
            ),
            contrast_score=0.0,
            colorfulness_score=0.0,
            importance=importance,
        )

        subjects.append(candidate)

    subjects.sort(
        key=lambda x: x.importance,
        reverse=True,
    )

    for i, subject in enumerate(subjects[:5]):

        logger.info(
            "Subject #%d: %s importance=%.3f area=%.3f conf=%.3f",
            i + 1,
            subject.class_name,
            subject.importance,
            subject.area,
            subject.confidence,
        )

    return subjects


def create_subject_attention(
    image,
    subject,
):

    mask = subject.mask.astype(np.float32)

    # Expand the subject slightly.
    kernel = np.ones(
        (21, 21),
        np.uint8,
    )

    expanded = cv2.dilate(
        mask,
        kernel,
        iterations=1,
    )

    # Soft edge.
    attention = cv2.GaussianBlur(
        expanded,
        (0, 0),
        11,
    )

    attention = normalize_map(attention)

    return attention


def combine_subject_and_saliency(
    subject_attention,
    saliency,
    importance,
):

    subject_weight = (
        0.45
        + 0.25 * np.clip(
            importance,
            0.0,
            1.0,
        )
    )

    saliency_weight = (
        1.0 - subject_weight
    )

    combined = (
        subject_attention * subject_weight
        + saliency * saliency_weight
    )

    return normalize_map(combined)


# ============================================================
# Scene classification
# ============================================================

def classify_scene(
    image,
    analysis,
    labels,
):

    h, w = image.shape[:2]

    person_ratio = float(
        np.mean(
            labels == VOC_CLASSES.index("person")
        )
    )

    vehicle_classes = {
        "car",
        "bus",
        "train",
        "boat",
        "aeroplane",
    }

    vehicle_ids = [
        VOC_CLASSES.index(x)
        for x in vehicle_classes
    ]

    vehicle_ratio = float(
        np.mean(
            np.isin(labels, vehicle_ids)
        )
    )

    sky = create_sky_mask(image)

    sky_ratio = float(
        np.mean(sky > 0.3)
    )

    scores = {
        "portrait": 0.0,
        "night": 0.0,
        "sunset": 0.0,
        "landscape": 0.0,
        "city": 0.0,
        "indoor": 0.0,
        "general": 0.2,
    }

    # --------------------------------------------------------
    # Portrait
    # --------------------------------------------------------

    if person_ratio > 0.015:
        scores["portrait"] += 0.55

    if person_ratio > 0.05:
        scores["portrait"] += 0.25

    # --------------------------------------------------------
    # Night
    # --------------------------------------------------------

    if analysis.mean < 0.16:
        scores["night"] += 0.45

    if analysis.shadow_ratio > 0.25:
        scores["night"] += 0.25

    if analysis.highlight_ratio < 0.015:
        scores["night"] += 0.10

    # --------------------------------------------------------
    # Sunset
    # --------------------------------------------------------

    if sky_ratio > 0.05:
        if analysis.warm_ratio > 0.12:
            scores["sunset"] += 0.45

        if analysis.highlight_ratio > 0.01:
            scores["sunset"] += 0.15

    # --------------------------------------------------------
    # Landscape
    # --------------------------------------------------------

    if sky_ratio > 0.15:
        scores["landscape"] += 0.40

    if analysis.dynamic_range > 0.45:
        scores["landscape"] += 0.20

    # --------------------------------------------------------
    # City
    # --------------------------------------------------------

    if vehicle_ratio > 0.015:
        scores["city"] += 0.35

    if analysis.edge_density > 0.35:
        scores["city"] += 0.25

    # --------------------------------------------------------
    # Indoor
    # --------------------------------------------------------

    if sky_ratio < 0.02 and person_ratio > 0.005:
        scores["indoor"] += 0.25

    if analysis.mean > 0.18:
        scores["indoor"] += 0.10

    scene = max(
        scores,
        key=scores.get,
    )

    logger.info(
        "Scene classification: %s",
        scene,
    )

    logger.info(
        "Scene scores: %s",
        {
            k: round(v, 3)
            for k, v in scores.items()
        },
    )

    return scene


# ============================================================
# Scene profiles
# ============================================================

def get_scene_profile(scene):

    profiles = {

        "portrait": SceneProfile(
            name="portrait",
            exposure_bias=0.05,
            contrast=1.02,
            saturation=0.97,
            highlight_recovery=0.35,
            shadow_lift=0.08,
            local_subject_strength=0.08,
            background_suppression=0.035,
            denoise_strength=0.55,
            sharpen_strength=0.75,
        ),

        "night": SceneProfile(
            name="night",
            exposure_bias=0.00,
            contrast=1.05,
            saturation=1.03,
            highlight_recovery=0.55,
            shadow_lift=0.02,
            local_subject_strength=0.05,
            background_suppression=0.015,
            denoise_strength=0.85,
            sharpen_strength=0.45,
        ),

        "sunset": SceneProfile(
            name="sunset",
            exposure_bias=-0.05,
            contrast=1.06,
            saturation=1.08,
            highlight_recovery=0.55,
            shadow_lift=0.04,
            local_subject_strength=0.05,
            background_suppression=0.015,
            denoise_strength=0.30,
            sharpen_strength=0.80,
        ),

        "landscape": SceneProfile(
            name="landscape",
            exposure_bias=0.03,
            contrast=1.08,
            saturation=1.04,
            highlight_recovery=0.40,
            shadow_lift=0.08,
            local_subject_strength=0.06,
            background_suppression=0.015,
            denoise_strength=0.30,
            sharpen_strength=0.85,
        ),

        "city": SceneProfile(
            name="city",
            exposure_bias=0.02,
            contrast=1.07,
            saturation=1.02,
            highlight_recovery=0.45,
            shadow_lift=0.05,
            local_subject_strength=0.06,
            background_suppression=0.02,
            denoise_strength=0.40,
            sharpen_strength=0.80,
        ),

        "indoor": SceneProfile(
            name="indoor",
            exposure_bias=0.04,
            contrast=1.03,
            saturation=0.99,
            highlight_recovery=0.40,
            shadow_lift=0.08,
            local_subject_strength=0.05,
            background_suppression=0.015,
            denoise_strength=0.50,
            sharpen_strength=0.65,
        ),

        "general": SceneProfile(
            name="general",
            exposure_bias=0.00,
            contrast=1.04,
            saturation=1.00,
            highlight_recovery=0.35,
            shadow_lift=0.06,
            local_subject_strength=0.04,
            background_suppression=0.015,
            denoise_strength=0.30,
            sharpen_strength=0.75,
        ),
    }

    return profiles.get(
        scene,
        profiles["general"],
    )


# ============================================================
# Linear-domain development
# ============================================================

def apply_linear_exposure(
    image,
    ev,
):

    gain = 2.0 ** ev

    return np.clip(
        image * gain,
        0.0,
        8.0,
    )


def apply_auto_exposure_linear(
    image,
    analysis,
    target=0.18,
):

    median = max(
        analysis.median,
        1e-4,
    )

    ev = math.log2(
        target / median
    )

    # Avoid excessive automatic exposure.
    ev = float(
        np.clip(
            ev,
            -1.5,
            1.5,
        )
    )

    logger.info(
        "Auto exposure EV: %.3f",
        ev,
    )

    return apply_linear_exposure(
        image,
        ev,
    )


def apply_linear_white_balance(
    image,
    target="neutral",
):

    # Estimate neutral pixels from relatively
    # low-saturation regions.

    rgb_max = np.max(
        image,
        axis=2,
    )

    rgb_min = np.min(
        image,
        axis=2,
    )

    saturation = (
        rgb_max - rgb_min
    ) / np.maximum(
        rgb_max,
        1e-6,
    )

    y = luminance(image)

    neutral = (
        (saturation < 0.08)
        & (y > 0.10)
        & (y < 0.80)
    )

    if np.count_nonzero(neutral) < 100:
        return image

    means = np.mean(
        image[neutral],
        axis=0,
    )

    target_mean = np.mean(means)

    gains = target_mean / np.maximum(
        means,
        1e-6,
    )

    # Limit correction.
    gains = np.clip(
        gains,
        0.90,
        1.10,
    )

    logger.info(
        "Linear WB gains: R=%.3f G=%.3f B=%.3f",
        gains[0],
        gains[1],
        gains[2],
    )

    result = image * gains[None, None, :]

    return np.clip(
        result,
        0.0,
        8.0,
    )


def apply_highlight_recovery_linear(
    image,
    strength=0.4,
):

    y = luminance(image)

    mask = np.clip(
        (y - 0.65) / 0.35,
        0.0,
        1.0,
    )

    # Compress the upper linear range.
    compression = (
        1.0
        - strength
        * mask
        * 0.30
    )

    result = image * compression[:, :, None]

    return np.clip(
        result,
        0.0,
        8.0,
    )


def apply_shadow_lift_linear(
    image,
    strength=0.06,
):

    y = luminance(image)

    mask = np.clip(
        (0.25 - y) / 0.25,
        0.0,
        1.0,
    )

    lift = (
        strength
        * mask
        * 0.20
    )

    result = image + lift[:, :, None]

    return np.clip(
        result,
        0.0,
        8.0,
    )


# ============================================================
# Local development
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
            + mask[:, :, None]
            * (gain - 1.0)
        )
    )

    return np.clip(
        result,
        0.0,
        8.0,
    )


def apply_local_saturation(
    image,
    mask,
    strength,
):

    y = luminance(image)

    result = (
        y[:, :, None]
        + (
            image
            - y[:, :, None]
        )
        * (
            1.0
            + mask[:, :, None]
            * (strength - 1.0)
        )
    )

    return np.clip(
        result,
        0.0,
        8.0,
    )


def apply_subject_background_separation(
    image,
    attention,
    strength=1.0,
):

    if np.max(attention) <= 0.05:
        return image

    subject_gain = (
        1.0
        + 0.035
        * strength
        * attention
    )

    result = image * subject_gain[:, :, None]

    background = 1.0 - attention

    background_gain = (
        1.0
        - 0.015
        * strength
        * background
    )

    result *= background_gain[:, :, None]

    return np.clip(
        result,
        0.0,
        8.0,
    )


# ============================================================
# Tone mapping
# ============================================================

def filmic_tone_curve(
    image,
    contrast=1.04,
):

    # Normalize a linear image into a useful display range.
    #
    # This is deliberately applied only after the
    # linear-domain processing.

    x = np.maximum(
        image,
        0.0,
    )

    # Reinhard-like highlight compression.
    mapped = x / (1.0 + x)

    # Contrast around middle gray.
    mapped = np.clip(
        (
            mapped - 0.18
        ) * contrast
        + 0.18,
        0.0,
        1.0,
    )

    # Gentle toe.
    toe = np.clip(
        mapped / 0.20,
        0.0,
        1.0,
    )

    mapped = np.where(
        mapped < 0.20,
        mapped * (
            0.92
            + 0.08 * toe
        ),
        mapped,
    )

    return np.clip(
        mapped,
        0.0,
        1.0,
    )


def apply_clahe(
    image,
    clip_limit=1.5,
):

    y = luminance(image)

    y8 = np.clip(
        y * 255.0,
        0,
        255,
    ).astype(np.uint8)

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(8, 8),
    )

    enhanced = (
        clahe.apply(y8)
        .astype(np.float32)
        / 255.0
    )

    ratio = (
        enhanced
        / np.maximum(
            y,
            1e-4,
        )
    )

    result = image * ratio[:, :, None]

    return np.clip(
        result,
        0.0,
        1.0,
    )


# ============================================================
# Color rendering
# ============================================================

def linear_to_srgb(image):

    image = np.clip(
        image,
        0.0,
        1.0,
    )

    result = np.where(
        image <= 0.0031308,
        image * 12.92,
        1.055
        * np.power(
            image,
            1.0 / 2.4,
        )
        - 0.055,
    )

    return np.clip(
        result,
        0.0,
        1.0,
    )


def apply_saturation(
    image,
    amount,
):

    y = luminance(image)

    result = (
        y[:, :, None]
        + (
            image
            - y[:, :, None]
        ) * amount
    )

    return np.clip(
        result,
        0.0,
        1.0,
    )


# ============================================================
# Auto scoring
# ============================================================

def calculate_auto_score(
    image,
    subject_mask=None,
    saliency=None,
):

    y = luminance(image)

    highlight_score = 1.0 - float(
        np.mean(y > 0.985)
    )

    shadow_score = 1.0 - float(
        np.mean(y < 0.015)
    )

    midtone_score = float(
        np.mean(
            (
                y > 0.10
            )
            & (
                y < 0.90
            )
        )
    )

    contrast = float(
        np.percentile(y, 90)
        - np.percentile(y, 10)
    )

    contrast_score = np.clip(
        contrast / 0.75,
        0.0,
        1.0,
    )

    max_rgb = np.max(
        image,
        axis=2,
    )

    min_rgb = np.min(
        image,
        axis=2,
    )

    saturation = (
        max_rgb - min_rgb
    ) / np.maximum(
        max_rgb,
        1e-6,
    )

    saturation_score = 1.0 - np.clip(
        np.mean(saturation > 0.85),
        0.0,
        1.0,
    )

    if subject_mask is not None:
        subject_area = subject_mask > 0.25

        if np.any(subject_area):
            subject_y = y[subject_area]

            subject_score = 1.0 - float(
                np.mean(
                    (subject_y < 0.03)
                    | (subject_y > 0.98)
                )
            )
        else:
            subject_score = 0.5
    else:
        subject_score = 0.5

    if saliency is not None:
        saliency_score = float(
            np.mean(
                y[
                    saliency
                    > np.percentile(
                        saliency,
                        80,
                    )
                ]
            )
        )

        saliency_score = 1.0 - abs(
            saliency_score - 0.50
        )

    else:
        saliency_score = 0.5

    score = (
        highlight_score * 0.18
        + shadow_score * 0.12
        + midtone_score * 0.14
        + contrast_score * 0.14
        + saturation_score * 0.08
        + subject_score * 0.17
        + saliency_score * 0.17
    )

    return float(score)


# ============================================================
# Candidate rendering
# ============================================================

def render_candidate(
    image,
    exposure,
    contrast,
    saturation,
):

    result = apply_linear_exposure(
        image,
        exposure,
    )

    result = filmic_tone_curve(
        result,
        contrast=contrast,
    )

    result = apply_saturation(
        result,
        saturation,
    )

    return result


def auto_tune(
    image,
    analysis,
    subject_mask=None,
    saliency=None,
):

    candidates = []

    exposure_offsets = [
        -0.30,
        0.0,
        0.30,
    ]

    contrast_values = [
        0.96,
        1.00,
        1.06,
    ]

    saturation_values = [
        0.96,
        1.00,
        1.05,
    ]

    base_ev = math.log2(
        0.18
        / max(
            analysis.median,
            1e-4,
        )
    )

    base_ev = np.clip(
        base_ev,
        -1.5,
        1.5,
    )

    for ev_offset in exposure_offsets:

        for contrast in contrast_values:

            for saturation in saturation_values:

                ev = (
                    base_ev
                    + ev_offset
                )

                candidate = render_candidate(
                    image,
                    ev,
                    contrast,
                    saturation,
                )

                score = calculate_auto_score(
                    candidate,
                    subject_mask,
                    saliency,
                )

                candidates.append(
                    (
                        score,
                        candidate,
                        {
                            "ev": float(ev),
                            "contrast": float(contrast),
                            "saturation": float(saturation),
                        },
                    )
                )

    candidates.sort(
        key=lambda x: x[0],
        reverse=True,
    )

    best_score, best_image, best_params = (
        candidates[0]
    )

    logger.info(
        "Auto tuning: EV=%.3f contrast=%.3f saturation=%.3f score=%.4f",
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
# Denoise / Sharpen
# ============================================================

def denoise_image(
    image,
    strength=0.3,
):

    strength = np.clip(
        strength,
        0.0,
        1.0,
    )

    sigma_color = (
        15.0
        + 25.0 * strength
    )

    sigma_space = (
        3.0
        + 4.0 * strength
    )

    result = cv2.bilateralFilter(
        image.astype(np.float32),
        d=7,
        sigmaColor=sigma_color / 255.0,
        sigmaSpace=sigma_space,
    )

    return np.clip(
        result,
        0.0,
        1.0,
    )


def sharpen_image(
    image,
    strength=0.75,
):

    blur = cv2.GaussianBlur(
        image,
        (0, 0),
        1.0,
    )

    result = (
        image
        + strength
        * (
            image - blur
        )
    )

    return np.clip(
        result,
        0.0,
        1.0,
    )


# ============================================================
# JPEG output
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
    ).astype(np.uint8)

    # OpenCV / imageio convention here is RGB.
    iio.imwrite(
        filename,
        image8,
        extension=".jpg",
        quality=quality,
    )

    logger.info(
        "Saved: %s",
        filename,
    )


# ============================================================
# Main development pipeline
# ============================================================

def auto_develop(
    filename,
    output_dir,
    segmenter,
):

    logger.info(
        "========================================"
    )

    logger.info(
        "Developing: %s",
        filename,
    )

    # --------------------------------------------------------
    # 1. RAW -> Linear RGB
    # --------------------------------------------------------

    image, metadata = load_raw(
        filename
    )

    iso = metadata.get(
        "iso",
        0,
    )

    # --------------------------------------------------------
    # 2. Initial analysis
    # --------------------------------------------------------

    analysis = analyze_image(
        image
    )

    logger.info(
        "Analysis: mean=%.3f median=%.3f "
        "shadow=%.3f highlight=%.3f "
        "DR=%.3f",
        analysis.mean,
        analysis.median,
        analysis.shadow_ratio,
        analysis.highlight_ratio,
        analysis.dynamic_range,
    )

    # --------------------------------------------------------
    # 3. Semantic segmentation
    # --------------------------------------------------------

    labels, confidence = segmenter.predict(
        image
    )

    # --------------------------------------------------------
    # 4. Saliency
    # --------------------------------------------------------

    saliency = calculate_saliency_map(
        image
    )

    # --------------------------------------------------------
    # 5. Subject ranking
    # --------------------------------------------------------

    subjects = rank_subjects(
        image,
        labels,
        confidence,
        saliency,
    )

    if subjects:

        main_subject = subjects[0]

        subject_attention = create_subject_attention(
            image,
            main_subject,
        )

        attention = combine_subject_and_saliency(
            subject_attention,
            saliency,
            main_subject.importance,
        )

    else:

        main_subject = None

        subject_attention = np.zeros(
            image.shape[:2],
            dtype=np.float32,
        )

        attention = saliency

    # --------------------------------------------------------
    # 6. Scene classification
    # --------------------------------------------------------

    scene = classify_scene(
        image,
        analysis,
        labels,
    )

    profile = get_scene_profile(
        scene
    )

    logger.info(
        "Scene profile: %s",
        profile.name,
    )

    # --------------------------------------------------------
    # 7. Linear-domain automatic WB
    # --------------------------------------------------------

    image = apply_linear_white_balance(
        image
    )

    # --------------------------------------------------------
    # 8. Linear-domain automatic exposure
    # --------------------------------------------------------

    image = apply_auto_exposure_linear(
        image,
        analysis,
    )

    # --------------------------------------------------------
    # 9. Scene exposure bias
    # --------------------------------------------------------

    image = apply_linear_exposure(
        image,
        profile.exposure_bias,
    )

    # --------------------------------------------------------
    # 10. Highlight recovery
    # --------------------------------------------------------

    image = apply_highlight_recovery_linear(
        image,
        profile.highlight_recovery,
    )

    # --------------------------------------------------------
    # 11. Shadow recovery
    # --------------------------------------------------------

    image = apply_shadow_lift_linear(
        image,
        profile.shadow_lift,
    )

    # --------------------------------------------------------
    # 12. Sky adjustment
    # --------------------------------------------------------

    sky = create_sky_mask(
        image
    )

    if np.mean(sky > 0.3) > 0.02:

        image = apply_local_exposure(
            image,
            sky,
            -0.08,
        )

        image = apply_local_saturation(
            image,
            sky,
            1.04,
        )

    # --------------------------------------------------------
    # 13. Main subject
    # --------------------------------------------------------

    if main_subject is not None:

        image = apply_local_exposure(
            image,
            subject_attention,
            profile.local_subject_strength,
        )

    # --------------------------------------------------------
    # 14. Secondary subjects
    # --------------------------------------------------------

    for secondary in subjects[1:3]:

        if secondary.importance < 0.30:
            continue

        mask = cv2.GaussianBlur(
            secondary.mask.astype(
                np.float32
            ),
            (0, 0),
            7,
        )

        image = apply_local_exposure(
            image,
            mask,
            0.02,
        )

    # --------------------------------------------------------
    # 15. Person saturation protection
    # --------------------------------------------------------

    person_id = VOC_CLASSES.index(
        "person"
    )

    person_mask = (
        labels == person_id
    ).astype(np.float32)

    if np.mean(person_mask > 0.5) > 0.002:

        person_mask = cv2.GaussianBlur(
            person_mask,
            (0, 0),
            5,
        )

        image = apply_local_saturation(
            image,
            person_mask,
            0.94,
        )

    # --------------------------------------------------------
    # 16. Plant adjustment
    # --------------------------------------------------------

    plant_id = VOC_CLASSES.index(
        "pottedplant"
    )

    plant_mask = (
        labels == plant_id
    ).astype(np.float32)

    if np.mean(plant_mask > 0.5) > 0.002:

        plant_mask = cv2.GaussianBlur(
            plant_mask,
            (0, 0),
            5,
        )

        image = apply_local_saturation(
            image,
            plant_mask,
            1.05,
        )

    # --------------------------------------------------------
    # 17. Subject / background separation
    # --------------------------------------------------------

    image = apply_subject_background_separation(
        image,
        attention,
        strength=1.0,
    )

    # --------------------------------------------------------
    # 18. Linear -> display tone mapping
    # --------------------------------------------------------

    image = filmic_tone_curve(
        image,
        contrast=profile.contrast,
    )

    # --------------------------------------------------------
    # 19. CLAHE
    # --------------------------------------------------------

    image = apply_clahe(
        image,
        clip_limit=1.3,
    )

    # --------------------------------------------------------
    # 20. Saturation
    # --------------------------------------------------------

    image = apply_saturation(
        image,
        profile.saturation,
    )

    # --------------------------------------------------------
    # 21. Linear RGB -> sRGB
    # --------------------------------------------------------

    image = linear_to_srgb(
        image
    )

    # --------------------------------------------------------
    # 22. Denoise
    # --------------------------------------------------------

    iso_factor = np.clip(
        (iso - 400) / 3200.0,
        0.0,
        1.0,
    )

    denoise_strength = (
        profile.denoise_strength
        * (
            0.5
            + iso_factor
        )
    )

    image = denoise_image(
        image,
        denoise_strength,
    )

    # --------------------------------------------------------
    # 23. Sharpen
    # --------------------------------------------------------

    image = sharpen_image(
        image,
        profile.sharpen_strength,
    )

    # --------------------------------------------------------
    # 24. Save
    # --------------------------------------------------------

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        output_dir
        / f"{filename.stem}_developed.jpg"
    )

    save_jpeg(
        image,
        output_file,
        quality=95,
    )

    logger.info(
        "Finished: %s",
        filename,
    )

    return output_file


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

        if input_path.suffix.lower() in RAW_EXTENSIONS:
            return [input_path]

        return []

    files = []

    for path in input_path.rglob("*"):

        if (
            path.is_file()
            and path.suffix.lower()
            in RAW_EXTENSIONS
        ):
            files.append(path)

    return sorted(files)


# ============================================================
# CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Automatic RAW development v10"
        )
    )

    parser.add_argument(
        "input",
        help="RAW file or directory",
    )

    parser.add_argument(
        "-o",
        "--output",
        default="./output",
        help="Output directory",
    )

    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default=None,
        help="Segmentation device",
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    if args.device == "cuda":

        if not torch.cuda.is_available():

            logger.error(
                "CUDA requested but PyTorch CUDA is unavailable."
            )

            return 1

        logger.info(
            "CUDA available: %s",
            torch.cuda.get_device_name(0),
        )

    # --------------------------------------------------------
    # Segmenter
    # --------------------------------------------------------

    segmenter = SemanticSegmenter(
        device=args.device
    )

    # --------------------------------------------------------
    # RAW files
    # --------------------------------------------------------

    raw_files = collect_raw_files(
        args.input
    )

    if not raw_files:

        logger.error(
            "No RAW files found."
        )

        return 1

    logger.info(
        "Found %d RAW file(s).",
        len(raw_files),
    )

    output_dir = Path(
        args.output
    )

    # --------------------------------------------------------
    # Process
    # --------------------------------------------------------

    success = 0

    for filename in raw_files:

        try:

            auto_develop(
                filename,
                output_dir,
                segmenter,
            )

            success += 1

        except Exception as exc:

            logger.exception(
                "Failed to process %s: %s",
                filename,
                exc,
            )

    logger.info(
        "Completed: %d/%d",
        success,
        len(raw_files),
    )

    return 0 if success == len(raw_files) else 1


if __name__ == "__main__":
    sys.exit(main())