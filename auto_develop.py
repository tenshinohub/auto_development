#!/usr/bin/env python3

import argparse
import logging
import math
import sys
from pathlib import Path

import cv2
import imageio.v3 as iio
import numpy as np
import rawpy
import torch

from torchvision.models.segmentation import (
    deeplabv3_mobilenet_v3_large,
    DeepLabV3_MobileNet_V3_Large_Weights,
)


# ============================================================
# Configuration
# ============================================================

SUPPORTED_EXTENSIONS = {
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

JPEG_QUALITY = 95

SEGMENTATION_SIZE = 768
SEGMENTATION_CONFIDENCE = 0.50

TARGET_MEAN = 0.42

SHADOW_THRESHOLD = 0.025
HIGHLIGHT_THRESHOLD = 0.98

CLAHE_CLIP_LIMIT = 1.5
CLAHE_GRID_SIZE = 8

SHARPEN_SIGMA = 1.2
SHARPEN_AMOUNT = 0.35

DENOISE_SIGMA_COLOR = 3
DENOISE_SIGMA_SPACE = 3


# ============================================================
# VOC class IDs
# ============================================================

CLASS_BACKGROUND = 0
CLASS_AEROPLANE = 1
CLASS_BICYCLE = 2
CLASS_BIRD = 3
CLASS_BOAT = 4
CLASS_BOTTLE = 5
CLASS_BUS = 6
CLASS_CAR = 7
CLASS_CAT = 8
CLASS_CHAIR = 9
CLASS_COW = 10
CLASS_DININGTABLE = 11
CLASS_DOG = 12
CLASS_HORSE = 13
CLASS_MOTORBIKE = 14
CLASS_PERSON = 15
CLASS_PLANT = 16
CLASS_SHEEP = 17
CLASS_SOFA = 18
CLASS_TRAIN = 19
CLASS_TV = 20


SUBJECT_CLASSES = {
    CLASS_AEROPLANE: "aeroplane",
    CLASS_BICYCLE: "bicycle",
    CLASS_BIRD: "bird",
    CLASS_BOAT: "boat",
    CLASS_BOTTLE: "bottle",
    CLASS_BUS: "bus",
    CLASS_CAR: "car",
    CLASS_CAT: "cat",
    CLASS_COW: "cow",
    CLASS_DOG: "dog",
    CLASS_HORSE: "horse",
    CLASS_MOTORBIKE: "motorbike",
    CLASS_PERSON: "person",
    CLASS_PLANT: "plant",
    CLASS_SHEEP: "sheep",
    CLASS_TRAIN: "train",
}


# ============================================================
# Logging
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


# ============================================================
# Utility
# ============================================================

def normalize_image(img):
    img = np.asarray(img)

    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0

    if img.dtype == np.uint16:
        return img.astype(np.float32) / 65535.0

    img = img.astype(np.float32)

    maximum = img.max()

    if maximum > 1.0:
        img /= maximum

    return np.clip(
        img,
        0.0,
        1.0,
    )


def denormalize_uint8(img):
    return np.clip(
        img * 255.0 + 0.5,
        0,
        255,
    ).astype(np.uint8)


def calculate_luminance(image):
    return (
        0.2126 * image[:, :, 0]
        + 0.7152 * image[:, :, 1]
        + 0.0722 * image[:, :, 2]
    )


# ============================================================
# RAW
# ============================================================

def load_raw(filename):

    logging.info(
        "RAW loading: %s",
        filename,
    )

    with rawpy.imread(str(filename)) as raw:

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

        metadata = {
            "make": getattr(
                raw,
                "camera_make",
                "",
            ),
            "model": getattr(
                raw,
                "camera_model",
                "",
            ),
            "iso": getattr(
                raw,
                "camera_iso_speed",
                0,
            ),
        }

    return normalize_image(rgb), metadata


# ============================================================
# Image Analysis
# ============================================================

class ImageAnalysis:

    def __init__(
        self,
        image,
        mask=None,
    ):

        self.image = image

        self.luminance = calculate_luminance(
            image
        )

        if mask is None:
            values = self.luminance.reshape(-1)
        else:
            values = self.luminance[
                mask > 0
            ]

            if len(values) == 0:
                values = self.luminance.reshape(-1)

        self.mean = float(
            np.mean(values)
        )

        self.median = float(
            np.median(values)
        )

        self.p01 = float(
            np.percentile(
                values,
                1,
            )
        )

        self.p05 = float(
            np.percentile(
                values,
                5,
            )
        )

        self.p95 = float(
            np.percentile(
                values,
                95,
            )
        )

        self.p99 = float(
            np.percentile(
                values,
                99,
            )
        )

        self.shadow_ratio = float(
            np.mean(
                values < SHADOW_THRESHOLD
            )
        )

        self.highlight_ratio = float(
            np.mean(
                values > HIGHLIGHT_THRESHOLD
            )
        )

        self.dynamic_range = (
            self.p99
            - self.p01
        )


# ============================================================
# Semantic Segmentation
# ============================================================

class SemanticSegmenter:

    def __init__(self):

        logging.info(
            "Loading segmentation model..."
        )

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        logging.info(
            "Segmentation device: %s",
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

        self.model.eval()
        self.model.to(
            self.device
        )

        self.transforms = (
            weights.transforms()
        )

    def segment(
        self,
        image,
    ):

        h, w, _ = image.shape

        scale = min(
            1.0,
            SEGMENTATION_SIZE
            / max(h, w),
        )

        new_w = max(
            1,
            int(w * scale),
        )

        new_h = max(
            1,
            int(h * scale),
        )

        small = cv2.resize(
            denormalize_uint8(image),
            (new_w, new_h),
            interpolation=cv2.INTER_AREA,
        )

        from PIL import Image

        pil = Image.fromarray(
            small
        )

        tensor = self.transforms(
            pil
        ).unsqueeze(0)

        tensor = tensor.to(
            self.device
        )

        with torch.no_grad():

            result = self.model(
                tensor
            )

            probabilities = torch.softmax(
                result["out"],
                dim=1,
            )

            confidence, labels = (
                torch.max(
                    probabilities,
                    dim=1,
                )
            )

        labels = (
            labels[0]
            .cpu()
            .numpy()
        )

        confidence = (
            confidence[0]
            .cpu()
            .numpy()
        )

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

        labels[
            confidence
            < SEGMENTATION_CONFIDENCE
        ] = CLASS_BACKGROUND

        return labels, confidence


# ============================================================
# Region masks
# ============================================================

def build_region_masks(
    image,
    labels,
):
    h, w, _ = image.shape

    masks = {}

    # --------------------------------------------------------
    # Person
    # --------------------------------------------------------

    person = (
        labels == CLASS_PERSON
    ).astype(np.float32)

    person = cv2.GaussianBlur(
        person,
        (0, 0),
        3,
    )

    masks["person"] = np.clip(
        person,
        0.0,
        1.0,
    )

    # --------------------------------------------------------
    # Plant
    # --------------------------------------------------------

    plant = (
        labels == CLASS_PLANT
    ).astype(np.float32)

    plant = cv2.GaussianBlur(
        plant,
        (0, 0),
        3,
    )

    masks["plant"] = np.clip(
        plant,
        0.0,
        1.0,
    )

    # --------------------------------------------------------
    # Sky
    # --------------------------------------------------------

    r = image[:, :, 0]
    g = image[:, :, 1]
    b = image[:, :, 2]

    luminance = calculate_luminance(
        image
    )

    chroma = (
        np.maximum.reduce(
            [r, g, b]
        )
        - np.minimum.reduce(
            [r, g, b]
        )
    )

    blue = (
        (b > r * 1.03)
        & (b > g * 0.98)
        & (chroma > 0.03)
    )

    y = np.arange(h)[:, None]

    top_weight = (
        1.0
        - y / h
    )

    sky = (
        blue
        & (top_weight > 0.35)
        & (luminance > 0.10)
    )

    sky = (
        sky.astype(np.uint8)
        * 255
    )

    sky = cv2.morphologyEx(
        sky,
        cv2.MORPH_CLOSE,
        np.ones(
            (15, 15),
            np.uint8,
        ),
    )

    sky = cv2.GaussianBlur(
        sky,
        (0, 0),
        5,
    )

    masks["sky"] = (
        sky.astype(np.float32)
        / 255.0
    )

    # --------------------------------------------------------
    # Shadows
    # --------------------------------------------------------

    shadow = np.clip(
        (0.35 - luminance)
        / 0.35,
        0.0,
        1.0,
    )

    masks["shadow"] = (
        shadow * shadow
    )

    # --------------------------------------------------------
    # Highlights
    # --------------------------------------------------------

    highlight = np.clip(
        (luminance - 0.70)
        / 0.30,
        0.0,
        1.0,
    )

    masks["highlight"] = (
        highlight * highlight
    )

    return masks


# ============================================================
# Subject Detection
# ============================================================

def create_center_weight(
    shape,
):
    """
    画像中央に近いほど高い重み。

    ただし中央を絶対視しない。
    """

    h, w = shape

    y, x = np.mgrid[
        0:h,
        0:w,
    ]

    x = (
        x / max(w - 1, 1)
    )

    y = (
        y / max(h - 1, 1)
    )

    distance = np.sqrt(
        (
            x - 0.5
        ) ** 2
        +
        (
            y - 0.5
        ) ** 2
    )

    weight = 1.0 - (
        distance
        / 0.707
    )

    return np.clip(
        weight,
        0.0,
        1.0,
    )


def create_edge_weight(
    shape,
):
    """
    画面端から離れているほど高い。
    """

    h, w = shape

    y, x = np.mgrid[
        0:h,
        0:w,
    ]

    distance = np.minimum.reduce(
        [
            x,
            y,
            w - 1 - x,
            h - 1 - y,
        ]
    )

    distance = distance.astype(
        np.float32
    )

    return np.clip(
        distance
        / min(h, w)
        * 4.0,
        0.0,
        1.0,
    )


def calculate_subject_score(
    class_id,
    mask,
    confidence,
    center_weight,
    edge_weight,
    image_shape,
):
    """
    セグメンテーションされた物体が
    主被写体である可能性を評価する。
    """

    area = float(
        np.mean(
            mask > 0.25
        )
    )

    if area < 0.002:
        return 0.0

    if area > 0.80:
        return 0.0

    confidence_score = float(
        np.mean(
            confidence[
                mask > 0.25
            ]
        )
    )

    center_score = float(
        np.sum(
            mask
            * center_weight
        )
        /
        max(
            np.sum(mask),
            1e-6,
        )
    )

    edge_score = float(
        np.sum(
            mask
            * edge_weight
        )
        /
        max(
            np.sum(mask),
            1e-6,
        )
    )

    # 面積が大きすぎる物体は少し減点
    area_score = min(
        area / 0.15,
        1.0,
    )

    if area > 0.50:
        area_score *= 0.5

    # 主被写体らしさ
    score = (
        confidence_score * 0.30
        + center_score * 0.30
        + edge_score * 0.15
        + area_score * 0.25
    )

    # 人物は主被写体になりやすい
    if class_id == CLASS_PERSON:
        score *= 1.15

    return float(
        score
    )


def detect_main_subject(
    image,
    labels,
    confidence,
):
    """
    セグメンテーションされた物体から
    主被写体候補を選ぶ。
    """

    h, w, _ = image.shape

    center_weight = create_center_weight(
        (h, w)
    )

    edge_weight = create_edge_weight(
        (h, w)
    )

    candidates = []

    for class_id, name in SUBJECT_CLASSES.items():

        mask = (
            labels == class_id
        ).astype(np.float32)

        score = calculate_subject_score(
            class_id,
            mask,
            confidence,
            center_weight,
            edge_weight,
            image.shape,
        )

        if score <= 0:
            continue

        candidates.append(
            {
                "class_id": class_id,
                "name": name,
                "score": score,
                "mask": mask,
            }
        )

    if not candidates:

        logging.info(
            "No semantic subject found."
        )

        return None

    candidates.sort(
        key=lambda x: x["score"],
        reverse=True,
    )

    for candidate in candidates[:5]:

        logging.info(
            "Subject candidate: "
            "%s score=%.3f",
            candidate["name"],
            candidate["score"],
        )

    best = candidates[0]

    logging.info(
        "Main subject: %s "
        "score=%.3f",
        best["name"],
        best["score"],
    )

    # --------------------------------------------------------
    # マスクを少し滑らかにする
    # --------------------------------------------------------

    mask = best["mask"]

    mask = cv2.GaussianBlur(
        mask,
        (0, 0),
        5,
    )

    mask = np.clip(
        mask,
        0.0,
        1.0,
    )

    best["mask"] = mask

    return best


# ============================================================
# Subject-aware mask
# ============================================================

def create_subject_attention(
    image,
    subject,
):
    """
    主被写体マスクを作る。

    被写体だけを極端に補正するのではなく、
    周辺にも徐々に影響させる。
    """

    if subject is None:

        return np.zeros(
            image.shape[:2],
            dtype=np.float32,
        )

    mask = subject["mask"]

    # 少し膨張
    kernel = np.ones(
        (31, 31),
        np.uint8,
    )

    expanded = cv2.dilate(
        mask.astype(np.uint8),
        kernel,
        iterations=1,
    ).astype(np.float32)

    # ぼかして自然につなぐ
    expanded = cv2.GaussianBlur(
        expanded,
        (0, 0),
        15,
    )

    attention = np.maximum(
        mask,
        expanded * 0.35,
    )

    return np.clip(
        attention,
        0.0,
        1.0,
    )


# ============================================================
# Exposure
# ============================================================

def calculate_exposure(
    analysis,
):

    median = analysis.median
    p05 = analysis.p05
    p95 = analysis.p95

    reference = (
        median * 0.65
        + p05 * 0.10
        + p95 * 0.25
    )

    target = 0.42

    ratio = (
        target
        / max(
            reference,
            0.001,
        )
    )

    ev = math.log2(
        ratio
    )

    if analysis.dynamic_range > 0.85:
        ev -= 0.15

    if analysis.highlight_ratio > 0.003:
        ev -= 0.20

    elif analysis.highlight_ratio > 0.001:
        ev -= 0.08

    if (
        analysis.shadow_ratio > 0.35
        and analysis.highlight_ratio < 0.001
    ):
        ev += 0.15

    ev = float(
        np.clip(
            ev,
            -1.5,
            1.5,
        )
    )

    gain = 2.0 ** ev

    return gain, ev


# ============================================================
# White Balance
# ============================================================

def calculate_auto_wb(
    image,
    sky_mask=None,
):

    r = image[:, :, 0]
    g = image[:, :, 1]
    b = image[:, :, 2]

    luminance = calculate_luminance(
        image
    )

    max_rgb = np.maximum.reduce(
        [r, g, b]
    )

    min_rgb = np.minimum.reduce(
        [r, g, b]
    )

    chroma = (
        max_rgb - min_rgb
    )

    mask = (
        (luminance > 0.12)
        &
        (luminance < 0.85)
        &
        (chroma < 0.12)
    )

    if sky_mask is not None:

        mask &= (
            sky_mask < 0.30
        )

    if np.sum(mask) < 1000:

        logging.info(
            "Not enough WB pixels."
        )

        return np.ones(
            3,
            dtype=np.float32,
        )

    r_mean = float(
        np.median(
            r[mask]
        )
    )

    g_mean = float(
        np.median(
            g[mask]
        )
    )

    b_mean = float(
        np.median(
            b[mask]
        )
    )

    target = (
        r_mean
        + g_mean
        + b_mean
    ) / 3.0

    gains = np.array(
        [
            target / max(
                r_mean,
                1e-6,
            ),
            target / max(
                g_mean,
                1e-6,
            ),
            target / max(
                b_mean,
                1e-6,
            ),
        ],
        dtype=np.float32,
    )

    gains = np.clip(
        gains,
        0.90,
        1.10,
    )

    gains /= gains[1]

    return gains


def apply_white_balance(
    image,
    gains,
):

    return np.clip(
        image
        * gains[None, None, :],
        0.0,
        1.0,
    )


# ============================================================
# Local Exposure
# ============================================================

def apply_local_exposure(
    image,
    mask,
    gain,
):

    if np.max(mask) <= 0:
        return image

    correction = (
        1.0
        + (gain - 1.0)
        * mask
    )

    return np.clip(
        image
        * correction[:, :, None],
        0.0,
        1.0,
    )


# ============================================================
# Local Saturation
# ============================================================

def apply_local_saturation(
    image,
    mask,
    amount,
):

    if np.max(mask) <= 0:
        return image

    img8 = denormalize_uint8(
        image
    )

    hsv = cv2.cvtColor(
        img8,
        cv2.COLOR_RGB2HSV,
    )

    saturation = (
        hsv[:, :, 1].astype(
            np.float32
        )
    )

    factor = (
        1.0
        + (amount - 1.0)
        * mask
    )

    saturation *= factor

    hsv[:, :, 1] = np.clip(
        saturation,
        0,
        255,
    ).astype(np.uint8)

    result = cv2.cvtColor(
        hsv,
        cv2.COLOR_HSV2RGB,
    )

    return (
        result.astype(
            np.float32
        )
        / 255.0
    )


# ============================================================
# Highlight recovery
# ============================================================

def recover_highlights(
    image,
    mask,
    strength=0.30,
):

    if np.max(mask) <= 0:
        return image

    luminance = calculate_luminance(
        image
    )

    soft_mask = np.clip(
        (luminance - 0.75)
        / 0.25,
        0.0,
        1.0,
    )

    soft_mask *= mask

    recoverable = (
        1.0
        - np.clip(
            (luminance - 0.96)
            / 0.04,
            0.0,
            1.0,
        )
    )

    soft_mask *= recoverable

    result = image / (
        1.0
        + strength
        * soft_mask[:, :, None]
    )

    return np.clip(
        result,
        0.0,
        1.0,
    )


# ============================================================
# Shadow recovery
# ============================================================

def recover_shadows(
    image,
    mask,
    strength=0.18,
):

    if np.max(mask) <= 0:
        return image

    gamma = 0.82

    corrected = np.power(
        np.clip(
            image,
            0.0,
            1.0,
        ),
        gamma,
    )

    result = (
        image
        * (
            1.0
            - strength
            * mask[:, :, None]
        )
        +
        corrected
        * (
            strength
            * mask[:, :, None]
        )
    )

    return np.clip(
        result,
        0.0,
        1.0,
    )


# ============================================================
# Contrast
# ============================================================

def calculate_contrast(
    analysis,
):

    dr = analysis.dynamic_range

    if dr < 0.25:
        return 1.12

    if dr < 0.40:
        return 1.07

    if dr > 0.85:
        return 0.96

    return 1.02


def apply_contrast(
    image,
    amount,
):

    midpoint = 0.45

    result = (
        (
            image
            - midpoint
        )
        * amount
        + midpoint
    )

    return np.clip(
        result,
        0.0,
        1.0,
    )


# ============================================================
# Saturation
# ============================================================

def calculate_saturation(
    image,
):

    max_rgb = np.max(
        image,
        axis=2,
    )

    min_rgb = np.min(
        image,
        axis=2,
    )

    chroma = (
        max_rgb
        - min_rgb
    )

    avg_chroma = float(
        np.mean(chroma)
    )

    if avg_chroma < 0.08:
        return 1.10

    if avg_chroma < 0.16:
        return 1.05

    if avg_chroma > 0.35:
        return 0.94

    return 1.00


# ============================================================
# Tone Curve
# ============================================================

def apply_adaptive_tone_curve(
    image,
    analysis,
):

    dr = analysis.dynamic_range

    if dr < 0.25:
        strength = 1.0

    elif dr < 0.40:
        strength = 0.70

    elif dr < 0.60:
        strength = 0.45

    elif dr < 0.80:
        strength = 0.20

    else:
        strength = 0.05

    x = np.linspace(
        0.0,
        1.0,
        4096,
    )

    curve = (
        x
        + strength
        * 0.12
        * x
        * (1.0 - x)
        * (2.0 * x - 1.0)
    )

    curve = np.clip(
        curve,
        0.0,
        1.0,
    )

    index = np.clip(
        image * 4095.0,
        0,
        4095,
    ).astype(np.int32)

    return curve[index]


# ============================================================
# Auto Score
# ============================================================

def calculate_auto_score(
    image,
    subject_mask=None,
):
    """
    自動現像結果を評価する。

    主被写体がある場合は、
    被写体の状態を少し重視する。
    """

    luminance = calculate_luminance(
        image
    )

    highlight_ratio = np.mean(
        luminance > 0.98
    )

    severe_highlight = np.mean(
        luminance > 0.995
    )

    highlight_score = (
        1.0
        - min(
            highlight_ratio * 8.0,
            1.0,
        )
        - min(
            severe_highlight * 12.0,
            1.0,
        )
    )

    shadow_ratio = np.mean(
        luminance < 0.02
    )

    severe_shadow = np.mean(
        luminance < 0.005
    )

    shadow_score = (
        1.0
        - min(
            shadow_ratio * 5.0,
            1.0,
        )
        - min(
            severe_shadow * 8.0,
            1.0,
        )
    )

    midtone_ratio = np.mean(
        (
            luminance > 0.15
        )
        &
        (
            luminance < 0.85
        )
    )

    midtone_score = min(
        midtone_ratio * 1.5,
        1.0,
    )

    p05 = np.percentile(
        luminance,
        5,
    )

    p95 = np.percentile(
        luminance,
        95,
    )

    dynamic_range = (
        p95 - p05
    )

    if dynamic_range < 0.25:

        contrast_score = (
            dynamic_range
            / 0.25
        )

    elif dynamic_range > 0.90:

        contrast_score = (
            1.0
            - (
                dynamic_range
                - 0.90
            )
            * 2.0
        )

    else:

        contrast_score = 1.0

    contrast_score = float(
        np.clip(
            contrast_score,
            0.0,
            1.0,
        )
    )

    max_rgb = np.max(
        image,
        axis=2,
    )

    min_rgb = np.min(
        image,
        axis=2,
    )

    chroma = (
        max_rgb
        - min_rgb
    )

    oversaturated = np.mean(
        chroma > 0.75
    )

    saturation_score = (
        1.0
        - min(
            oversaturated * 4.0,
            1.0,
        )
    )

    # --------------------------------------------------------
    # Subject score
    # --------------------------------------------------------

    subject_score = 1.0

    if (
        subject_mask is not None
        and np.mean(
            subject_mask > 0.25
        ) > 0.002
    ):

        subject_values = luminance[
            subject_mask > 0.25
        ]

        subject_mean = float(
            np.median(
                subject_values
            )
        )

        # 主被写体は暗すぎても明るすぎても減点
        subject_score = 1.0 - min(
            abs(
                subject_mean
                - 0.48
            )
            / 0.48,
            1.0,
        )

    score = (
        highlight_score * 0.25
        + shadow_score * 0.15
        + midtone_score * 0.15
        + contrast_score * 0.15
        + saturation_score * 0.10
        + subject_score * 0.20
    )

    return float(
        score
    )


# ============================================================
# Candidate generation
# ============================================================

def generate_candidates(
    base_ev,
    base_contrast,
    base_saturation,
):

    ev_offsets = [
        -0.30,
        0.00,
        0.30,
    ]

    contrast_offsets = [
        -0.08,
        0.00,
        0.08,
    ]

    saturation_offsets = [
        -0.05,
        0.00,
        0.05,
    ]

    candidates = []

    for ev_offset in ev_offsets:

        for contrast_offset in contrast_offsets:

            for saturation_offset in saturation_offsets:

                candidates.append(
                    {
                        "ev":
                            base_ev
                            + ev_offset,

                        "contrast":
                            base_contrast
                            + contrast_offset,

                        "saturation":
                            base_saturation
                            + saturation_offset,
                    }
                )

    return candidates


# ============================================================
# Candidate rendering
# ============================================================

def render_candidate(
    image,
    params,
):

    result = image.copy()

    gain = (
        2.0
        ** params["ev"]
    )

    result = apply_exposure(
        result,
        gain,
    )

    result = apply_contrast(
        result,
        params["contrast"],
    )

    mask = np.ones(
        result.shape[:2],
        dtype=np.float32,
    )

    result = apply_local_saturation(
        result,
        mask,
        params["saturation"],
    )

    return result


def auto_tune(
    image,
    analysis,
    subject_mask=None,
):

    _, base_ev = (
        calculate_exposure(
            analysis
        )
    )

    base_contrast = (
        calculate_contrast(
            analysis
        )
    )

    base_saturation = (
        calculate_saturation(
            image
        )
    )

    candidates = generate_candidates(
        base_ev,
        base_contrast,
        base_saturation,
    )

    logging.info(
        "Auto tuning: %d candidates",
        len(candidates),
    )

    best_score = -float("inf")
    best_image = None
    best_params = None

    for params in candidates:

        candidate = render_candidate(
            image,
            params,
        )

        score = calculate_auto_score(
            candidate,
            subject_mask,
        )

        if score > best_score:

            best_score = score
            best_image = candidate
            best_params = params

    logging.info(
        "Best candidate: "
        "score=%.4f "
        "EV=%.2f "
        "contrast=%.3f "
        "saturation=%.3f",
        best_score,
        best_params["ev"],
        best_params["contrast"],
        best_params["saturation"],
    )

    return (
        best_image,
        best_params,
        best_score,
    )


# ============================================================
# Noise
# ============================================================

def calculate_noise_strength(
    iso,
):

    if iso is None or iso <= 0:
        return 0.15

    if iso < 400:
        return 0.10

    if iso < 800:
        return 0.15

    if iso < 1600:
        return 0.25

    if iso < 3200:
        return 0.35

    if iso < 6400:
        return 0.50

    return 0.65


def apply_denoise(
    image,
    strength,
):

    if strength < 0.05:
        return image

    img8 = denormalize_uint8(
        image
    )

    sigma_color = (
        DENOISE_SIGMA_COLOR
        * strength
    )

    sigma_space = (
        DENOISE_SIGMA_SPACE
        * strength
    )

    result = cv2.bilateralFilter(
        img8,
        5,
        max(
            1.0,
            sigma_color,
        ),
        max(
            1.0,
            sigma_space,
        ),
    )

    return (
        result.astype(
            np.float32
        )
        / 255.0
    )


# ============================================================
# Sharpen
# ============================================================

def apply_sharpen(
    image,
    strength=1.0,
):

    img8 = denormalize_uint8(
        image
    )

    blurred = cv2.GaussianBlur(
        img8,
        (0, 0),
        SHARPEN_SIGMA,
    )

    amount = (
        SHARPEN_AMOUNT
        * strength
    )

    result = cv2.addWeighted(
        img8,
        1.0 + amount,
        blurred,
        -amount,
        0,
    )

    return (
        result.astype(
            np.float32
        )
        / 255.0
    )


# ============================================================
# Exposure
# ============================================================

def apply_exposure(
    image,
    gain,
):

    return np.clip(
        image * gain,
        0.0,
        1.0,
    )


# ============================================================
# Main development
# ============================================================

def auto_develop(
    image,
    metadata,
    segmenter,
):

    logging.info(
        "Analyzing image..."
    )

    global_analysis = ImageAnalysis(
        image
    )

    logging.info(
        "Mean=%.3f Median=%.3f "
        "DR=%.3f",
        global_analysis.mean,
        global_analysis.median,
        global_analysis.dynamic_range,
    )

    # ========================================================
    # Segmentation
    # ========================================================

    labels, confidence = (
        segmenter.segment(
            image
        )
    )

    # ========================================================
    # Region masks
    # ========================================================

    masks = build_region_masks(
        image,
        labels,
    )

    # ========================================================
    # Main subject
    # ========================================================

    subject = detect_main_subject(
        image,
        labels,
        confidence,
    )

    if subject is not None:

        subject_attention = (
            create_subject_attention(
                image,
                subject,
            )
        )

    else:

        subject_attention = np.zeros(
            image.shape[:2],
            dtype=np.float32,
        )

    # ========================================================
    # White Balance
    # ========================================================

    wb = calculate_auto_wb(
        image,
        masks["sky"],
    )

    logging.info(
        "WB: R=%.3f G=%.3f B=%.3f",
        wb[0],
        wb[1],
        wb[2],
    )

    image = apply_white_balance(
        image,
        wb,
    )

    # ========================================================
    # Auto Tuning
    # ========================================================

    tuned_image, params, score = (
        auto_tune(
            image,
            global_analysis,
            subject_attention,
        )
    )

    image = tuned_image

    # ========================================================
    # Shadow recovery
    # ========================================================

    image = recover_shadows(
        image,
        masks["shadow"],
        strength=0.18,
    )

    # ========================================================
    # Highlight recovery
    # ========================================================

    image = recover_highlights(
        image,
        masks["highlight"],
        strength=0.30,
    )

    # ========================================================
    # Main subject
    # ========================================================

    if subject is not None:

        # 主被写体を少しだけ明るくする
        image = apply_local_exposure(
            image,
            subject_attention,
            1.045,
        )

        # 彩度をほんの少しだけ上げる
        image = apply_local_saturation(
            image,
            subject_attention,
            1.025,
        )

    # ========================================================
    # Sky
    # ========================================================

    sky = masks["sky"]

    if np.mean(
        sky > 0.25
    ) > 0.005:

        image = apply_local_exposure(
            image,
            sky,
            0.94,
        )

        image = apply_local_saturation(
            image,
            sky,
            1.08,
        )

    # ========================================================
    # Person
    # ========================================================

    person = masks["person"]

    if np.mean(
        person > 0.25
    ) > 0.002:

        image = apply_local_exposure(
            image,
            person,
            1.03,
        )

        image = apply_local_saturation(
            image,
            person,
            0.98,
        )

    # ========================================================
    # Plant
    # ========================================================

    plant = masks["plant"]

    if np.mean(
        plant > 0.25
    ) > 0.002:

        image = apply_local_saturation(
            image,
            plant,
            1.05,
        )

    # ========================================================
    # Adaptive tone curve
    # ========================================================

    image = apply_adaptive_tone_curve(
        image,
        global_analysis,
    )

    # ========================================================
    # CLAHE
    # ========================================================

    if global_analysis.dynamic_range < 0.35:

        logging.info(
            "Applying local contrast enhancement"
        )

        img8 = denormalize_uint8(
            image
        )

        lab = cv2.cvtColor(
            img8,
            cv2.COLOR_RGB2LAB,
        )

        l, a, b = cv2.split(
            lab
        )

        clahe = cv2.createCLAHE(
            clipLimit=CLAHE_CLIP_LIMIT,
            tileGridSize=(
                CLAHE_GRID_SIZE,
                CLAHE_GRID_SIZE,
            ),
        )

        l = clahe.apply(
            l
        )

        lab = cv2.merge(
            (l, a, b)
        )

        image = cv2.cvtColor(
            lab,
            cv2.COLOR_LAB2RGB,
        ).astype(
            np.float32
        ) / 255.0

    # ========================================================
    # Noise reduction
    # ========================================================

    iso = metadata.get(
        "iso",
        0,
    )

    noise_strength = (
        calculate_noise_strength(
            iso
        )
    )

    logging.info(
        "Noise reduction: %.2f",
        noise_strength,
    )

    image = apply_denoise(
        image,
        noise_strength,
    )

    # ========================================================
    # Sharpen
    # ========================================================

    sharpen_strength = (
        1.0
        - 0.5
        * noise_strength
    )

    image = apply_sharpen(
        image,
        sharpen_strength,
    )

    return np.clip(
        image,
        0.0,
        1.0,
    )


# ============================================================
# Save
# ============================================================

def save_image(
    image,
    output_file,
):

    img8 = denormalize_uint8(
        image
    )

    iio.imwrite(
        output_file,
        img8,
        extension=".jpg",
        quality=JPEG_QUALITY,
    )


# ============================================================
# Collect files
# ============================================================

def collect_raw_files(
    input_path,
):

    if input_path.is_file():

        if (
            input_path.suffix.lower()
            in SUPPORTED_EXTENSIONS
        ):
            return [input_path]

        return []

    return sorted(
        file
        for file in input_path.rglob("*")
        if (
            file.is_file()
            and file.suffix.lower()
            in SUPPORTED_EXTENSIONS
        )
    )


# ============================================================
# Process
# ============================================================

def process_file(
    input_file,
    output_dir,
    segmenter,
):

    try:

        image, metadata = load_raw(
            input_file
        )

        developed = auto_develop(
            image,
            metadata,
            segmenter,
        )

        output_file = (
            output_dir
            / f"{input_file.stem}.jpg"
        )

        save_image(
            developed,
            output_file,
        )

        logging.info(
            "Completed: %s",
            input_file.name,
        )

        return True

    except Exception:

        logging.exception(
            "Failed: %s",
            input_file,
        )

        return False


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Automatic RAW developer v6"
        )
    )

    parser.add_argument(
        "input",
        type=Path,
        help="RAW file or directory",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("output"),
        help="Output directory",
    )

    args = parser.parse_args()

    if not args.input.exists():

        logging.error(
            "Input does not exist: %s",
            args.input,
        )

        return 1

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    files = collect_raw_files(
        args.input
    )

    if not files:

        logging.error(
            "No RAW files found."
        )

        return 1

    logging.info(
        "Found %d RAW file(s)",
        len(files),
    )

    segmenter = SemanticSegmenter()

    success = 0

    for file in files:

        if process_file(
            file,
            args.output,
            segmenter,
        ):
            success += 1

    logging.info(
        "Finished: %d/%d",
        success,
        len(files),
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())