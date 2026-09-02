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


@dataclass
class Subject:
    class_name: str
    mask: np.ndarray
    confidence: float
    score: float


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


def create_center_weight(height, width):
    y, x = np.mgrid[0:height, 0:width]

    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0

    dx = (x - cx) / max(width / 2.0, 1.0)
    dy = (y - cy) / max(height / 2.0, 1.0)

    distance = np.sqrt(dx * dx + dy * dy)

    weight = 1.0 - np.clip(distance, 0.0, 1.0)

    return weight.astype(np.float32)


# ============================================================
# RAW loading
# ============================================================

def load_raw(filename):
    logger.info("Loading RAW: %s", filename)

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

        rgb = rgb.astype(np.float32) / 65535.0

        metadata = {
            "make": getattr(raw, "camera_make", ""),
            "model": getattr(raw, "camera_model", ""),
            "iso": getattr(raw, "iso_speed", None),
            "width": rgb.shape[1],
            "height": rgb.shape[0],
        }

    return np.clip(rgb, 0.0, 1.0), metadata


# ============================================================
# Image analysis
# ============================================================

def analyze_image(image):
    lum = luminance(image)

    mean = float(np.mean(lum))
    median = float(np.median(lum))

    p01 = float(np.percentile(lum, 1))
    p05 = float(np.percentile(lum, 5))
    p95 = float(np.percentile(lum, 95))
    p99 = float(np.percentile(lum, 99))

    shadow_ratio = float(np.mean(lum < 0.05))
    highlight_ratio = float(np.mean(lum > 0.95))

    dynamic_range = p95 - p05

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
    )


# ============================================================
# Semantic segmentation
# ============================================================

class SemanticSegmenter:

    def __init__(self, device=None):

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)

        logger.info(
            "Loading DeepLabV3 MobileNet V3 Large on %s",
            self.device,
        )

        weights = DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT

        self.model = deeplabv3_mobilenet_v3_large(
            weights=weights
        )

        self.model.to(self.device)
        self.model.eval()

        self.transforms = weights.transforms()

    def predict(self, image):

        h, w = image.shape[:2]

        # 推論用に縮小
        max_size = 768

        scale = min(
            1.0,
            max_size / max(h, w),
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
            np.clip(small * 255.0, 0, 255).astype(np.uint8)
        )

        tensor = self.transforms(pil)

        tensor = tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(tensor)["out"][0]

        probabilities = torch.softmax(output, dim=0)

        prediction = torch.argmax(
            probabilities,
            dim=0,
        ).cpu().numpy()

        confidence = torch.max(
            probabilities,
            dim=0,
        ).values.cpu().numpy()

        prediction = cv2.resize(
            prediction.astype(np.uint8),
            (w, h),
            interpolation=cv2.INTER_NEAREST,
        )

        confidence = cv2.resize(
            confidence.astype(np.float32),
            (w, h),
            interpolation=cv2.INTER_LINEAR,
        )

        masks = {}

        for class_id, class_name in enumerate(VOC_CLASSES):

            if class_name == "background":
                continue

            mask = np.where(
                prediction == class_id,
                confidence,
                0.0,
            ).astype(np.float32)

            masks[class_name] = mask

        return masks


# ============================================================
# Heuristic masks
# ============================================================

def create_sky_mask(image):

    h, w = image.shape[:2]

    r = image[:, :, 0]
    g = image[:, :, 1]
    b = image[:, :, 2]

    y = np.arange(h, dtype=np.float32)[:, None]

    top_weight = 1.0 - np.clip(
        y / (h * 0.65),
        0.0,
        1.0,
    )

    blue_score = (
        b
        - 0.5 * r
        - 0.2 * g
    )

    bright = luminance(image)

    mask = (
        (blue_score > 0.05)
        & (bright > 0.15)
        & (top_weight > 0.15)
    )

    return mask.astype(np.float32) * top_weight


def create_shadow_mask(image):

    lum = luminance(image)

    mask = 1.0 - np.clip(
        lum / 0.25,
        0.0,
        1.0,
    )

    return mask.astype(np.float32)


def create_highlight_mask(image):

    lum = luminance(image)

    mask = np.clip(
        (lum - 0.75) / 0.25,
        0.0,
        1.0,
    )

    return mask.astype(np.float32)


# ============================================================
# Subject detection
# ============================================================

def detect_main_subject(masks):

    candidates = []

    for class_name, mask in masks.items():

        if class_name not in SUBJECT_CLASSES:
            continue

        area = np.mean(mask > 0.35)

        if area < 0.001:
            continue

        confidence = float(
            np.mean(mask[mask > 0.35])
        )

        h, w = mask.shape

        center_weight = create_center_weight(
            h,
            w,
        )

        strong_mask = mask > 0.35

        if np.any(strong_mask):
            center_score = float(
                np.mean(center_weight[strong_mask])
            )
        else:
            center_score = 0.0

        # 大きすぎる領域は主被写体としてのスコアを少し下げる
        area_score = min(
            area / 0.10,
            1.0,
        )

        score = (
            confidence * 0.40
            + center_score * 0.30
            + area_score * 0.30
        )

        if class_name == "person":
            score *= 1.15

        candidates.append(
            Subject(
                class_name=class_name,
                mask=mask,
                confidence=confidence,
                score=score,
            )
        )

    if not candidates:
        return None

    candidates.sort(
        key=lambda x: x.score,
        reverse=True,
    )

    subject = candidates[0]

    logger.info(
        "Main subject: %s confidence=%.3f score=%.3f",
        subject.class_name,
        subject.confidence,
        subject.score,
    )

    return subject


# ============================================================
# Subject attention
# ============================================================

def create_subject_attention(image, subject):

    mask = subject.mask.copy()

    # 弱い領域を除去
    mask = np.clip(
        (mask - 0.20) / 0.60,
        0.0,
        1.0,
    )

    # 少し膨張
    kernel = np.ones(
        (9, 9),
        np.uint8,
    )

    mask = cv2.dilate(
        mask,
        kernel,
        iterations=1,
    )

    # 境界をぼかす
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


# ============================================================
# Saliency Map
# ============================================================

def calculate_saliency_map(image):

    gray = luminance(image).astype(np.float32)

    # --------------------------------------------------------
    # Local contrast
    # --------------------------------------------------------

    blurred15 = cv2.GaussianBlur(
        gray,
        (0, 0),
        sigmaX=15,
    )

    local_contrast = np.abs(
        gray - blurred15
    )

    local_contrast = normalize_map(
        local_contrast
    )

    # --------------------------------------------------------
    # Edge
    # --------------------------------------------------------

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
        gx * gx + gy * gy
    )

    edge = normalize_map(edge)

    # --------------------------------------------------------
    # Colorfulness
    # --------------------------------------------------------

    colorfulness = (
        np.max(image, axis=2)
        - np.min(image, axis=2)
    )

    colorfulness = normalize_map(
        colorfulness
    )

    # --------------------------------------------------------
    # Brightness difference
    # --------------------------------------------------------

    blurred25 = cv2.GaussianBlur(
        gray,
        (0, 0),
        sigmaX=25,
    )

    brightness_difference = np.abs(
        gray - blurred25
    )

    brightness_difference = normalize_map(
        brightness_difference
    )

    # --------------------------------------------------------
    # Center weighting
    # --------------------------------------------------------

    h, w = gray.shape

    center_weight = create_center_weight(
        h,
        w,
    )

    center_weight = (
        0.65
        + 0.35 * center_weight
    )

    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

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

    saliency = normalize_map(
        saliency
    )

    return saliency


# ============================================================
# Subject + Saliency
# ============================================================

def combine_subject_and_saliency(
    subject_attention,
    saliency,
):

    if subject_attention is None:
        subject_attention = np.zeros_like(
            saliency
        )

    combined = (
        subject_attention * 0.60
        + saliency * 0.40
    )

    return normalize_map(combined)


# ============================================================
# Basic adjustments
# ============================================================

def apply_exposure(image, ev):

    gain = 2.0 ** ev

    return np.clip(
        image * gain,
        0.0,
        1.0,
    )


def apply_contrast(image, contrast):

    mean = np.mean(
        luminance(image)
    )

    result = (
        (image - mean)
        * contrast
        + mean
    )

    return np.clip(
        result,
        0.0,
        1.0,
    )


def apply_saturation(image, saturation):

    hsv = cv2.cvtColor(
        image.astype(np.float32),
        cv2.COLOR_RGB2HSV,
    )

    hsv[:, :, 1] *= saturation

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


def apply_auto_white_balance(image):

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
            mean_rgb / r_mean
        )

    if g_mean > 1e-5:
        result[:, :, 1] *= (
            mean_rgb / g_mean
        )

    if b_mean > 1e-5:
        result[:, :, 2] *= (
            mean_rgb / b_mean
        )

    return np.clip(
        result,
        0.0,
        1.0,
    )


# ============================================================
# Highlight / shadow recovery
# ============================================================

def apply_highlight_recovery(
    image,
    strength=0.35,
):

    lum = luminance(image)

    mask = np.clip(
        (lum - 0.65) / 0.35,
        0.0,
        1.0,
    )

    mask = mask * strength

    # ハイライト部分だけ少し圧縮
    result = image.copy()

    result = (
        result * (1.0 - mask[:, :, None])
        + np.sqrt(
            np.clip(result, 0.0, 1.0)
        ) * mask[:, :, None]
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

    lum = luminance(image)

    mask = np.clip(
        (0.35 - lum) / 0.35,
        0.0,
        1.0,
    )

    mask *= strength

    result = (
        image
        + (1.0 - image)
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
        image * (
            1.0
            + (gain - 1.0)
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
        image.astype(np.float32),
        cv2.COLOR_RGB2HSV,
    )

    hsv[:, :, 1] *= (
        1.0
        + (saturation - 1.0)
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
# Subject / Background separation
# ============================================================

def apply_subject_background_separation(
    image,
    attention,
):

    if np.max(attention) <= 0.05:
        return image

    # Subject +3.5% maximum
    subject_gain = (
        1.0
        + 0.035 * attention
    )

    result = (
        image
        * subject_gain[:, :, None]
    )

    # Background -1.5% maximum
    background = 1.0 - attention

    background_gain = (
        1.0
        - 0.015 * background
    )

    result *= (
        background_gain[:, :, None]
    )

    return np.clip(
        result,
        0.0,
        1.0,
    )


# ============================================================
# Adaptive tone curve
# ============================================================

def apply_adaptive_tone_curve(
    image,
    analysis,
):

    result = image.copy()

    # 暗すぎる場合
    if analysis.mean < 0.25:

        strength = np.clip(
            (0.25 - analysis.mean)
            / 0.25,
            0.0,
            1.0,
        )

        gamma = (
            1.0
            - 0.12 * strength
        )

        result = np.power(
            np.clip(result, 0.0, 1.0),
            gamma,
        )

    # 明るすぎる場合
    elif analysis.mean > 0.70:

        strength = np.clip(
            (analysis.mean - 0.70)
            / 0.30,
            0.0,
            1.0,
        )

        gamma = (
            1.0
            + 0.10 * strength
        )

        result = np.power(
            np.clip(result, 0.0, 1.0),
            gamma,
        )

    # Dynamic Rangeが低い場合
    if analysis.dynamic_range < 0.35:

        result = apply_contrast(
            result,
            1.0 + 0.10,
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
    clip_limit=1.5,
):

    lab = cv2.cvtColor(
        image.astype(np.float32),
        cv2.COLOR_RGB2LAB,
    )

    # OpenCV LABのLを0～255へ
    l = np.clip(
        lab[:, :, 0] / 100.0 * 255.0,
        0,
        255,
    ).astype(np.uint8)

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(8, 8),
    )

    l = clahe.apply(l)

    lab[:, :, 0] = (
        l.astype(np.float32)
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
# Auto score
# ============================================================

def calculate_auto_score(
    image,
    subject_mask=None,
    saliency=None,
):

    lum = luminance(image)

    # --------------------------------------------------------
    # Highlight
    # --------------------------------------------------------

    highlight_ratio = np.mean(
        lum > 0.98
    )

    highlight_score = np.clip(
        1.0
        - highlight_ratio * 8.0,
        0.0,
        1.0,
    )

    # --------------------------------------------------------
    # Shadow
    # --------------------------------------------------------

    shadow_ratio = np.mean(
        lum < 0.02
    )

    shadow_score = np.clip(
        1.0
        - shadow_ratio * 3.0,
        0.0,
        1.0,
    )

    # --------------------------------------------------------
    # Midtone
    # --------------------------------------------------------

    midtone_ratio = np.mean(
        (lum > 0.15)
        & (lum < 0.85)
    )

    midtone_score = np.clip(
        midtone_ratio,
        0.0,
        1.0,
    )

    # --------------------------------------------------------
    # Contrast
    # --------------------------------------------------------

    p05 = np.percentile(lum, 5)
    p95 = np.percentile(lum, 95)

    dynamic_range = p95 - p05

    contrast_score = np.clip(
        dynamic_range / 0.75,
        0.0,
        1.0,
    )

    # --------------------------------------------------------
    # Saturation
    # --------------------------------------------------------

    saturation = (
        np.max(image, axis=2)
        - np.min(image, axis=2)
    )

    mean_saturation = np.mean(
        saturation
    )

    saturation_score = np.clip(
        1.0
        - abs(mean_saturation - 0.20)
        / 0.25,
        0.0,
        1.0,
    )

    # --------------------------------------------------------
    # Subject
    # --------------------------------------------------------

    if (
        subject_mask is not None
        and np.any(subject_mask > 0.2)
    ):

        mask = subject_mask > 0.2

        subject_lum = lum[mask]

        if subject_lum.size > 0:

            subject_mean = np.mean(
                subject_lum
            )

            subject_score = np.clip(
                1.0
                - abs(subject_mean - 0.50)
                / 0.50,
                0.0,
                1.0,
            )

        else:
            subject_score = 0.5

    else:
        subject_score = 0.5

    # --------------------------------------------------------
    # Saliency
    # --------------------------------------------------------

    if saliency is not None:

        salient = saliency > 0.70

        if np.any(salient):

            salient_lum = lum[salient]

            salient_mean = np.mean(
                salient_lum
            )

            saliency_score = np.clip(
                1.0
                - abs(salient_mean - 0.50)
                / 0.50,
                0.0,
                1.0,
            )

        else:
            saliency_score = 0.5

    else:
        saliency_score = 0.5

    # --------------------------------------------------------
    # Final score
    # --------------------------------------------------------

    score = (
        highlight_score * 0.22
        + shadow_score * 0.13
        + midtone_score * 0.13
        + contrast_score * 0.14
        + saturation_score * 0.08
        + subject_score * 0.15
        + saliency_score * 0.15
    )

    return float(score)


# ============================================================
# Candidate generation
# ============================================================

def generate_candidates():

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

        for contrast_offset in contrast_values:

            for saturation_offset in saturation_values:

                candidates.append({
                    "ev": ev,
                    "contrast": 1.0 + contrast_offset,
                    "saturation": 1.0 + saturation_offset,
                })

    return candidates


# ============================================================
# Candidate rendering
# ============================================================

def render_candidate(
    image,
    params,
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
        strength=0.30,
    )

    result = apply_shadow_recovery(
        result,
        strength=0.20,
    )

    return result


# ============================================================
# Auto tuning
# ============================================================

def auto_tune(
    image,
    analysis,
    subject_mask=None,
    saliency=None,
):

    candidates = generate_candidates()

    best_score = -math.inf
    best_image = image
    best_params = None

    for params in candidates:

        candidate = render_candidate(
            image,
            params,
        )

        score = calculate_auto_score(
            candidate,
            subject_mask,
            saliency,
        )

        if score > best_score:

            best_score = score
            best_image = candidate
            best_params = params

    logger.info(
        "Auto tuning: EV=%+.2f contrast=%.2f saturation=%.2f score=%.4f",
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

    # OpenCV用に8bit化
    img8 = np.clip(
        image * 255.0,
        0,
        255,
    ).astype(np.uint8)

    # strengthに応じてsigmaを変更
    sigma_color = 10.0 + strength * 20.0
    sigma_space = 5.0 + strength * 10.0

    result = cv2.bilateralFilter(
        img8,
        d=7,
        sigmaColor=sigma_color,
        sigmaSpace=sigma_space,
    )

    result = result.astype(
        np.float32
    ) / 255.0

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
        + (image - blurred)
        * amount
    )

    return np.clip(
        result,
        0.0,
        1.0,
    )


# ============================================================
# Output
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
# Main development pipeline
# ============================================================

def auto_develop(
    filename,
    output_filename,
    segmenter,
):

    logger.info("=" * 70)
    logger.info("Processing: %s", filename)

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
    # Global analysis
    # --------------------------------------------------------

    global_analysis = analyze_image(
        image
    )

    logger.info(
        "Mean=%.3f Median=%.3f "
        "P05=%.3f P95=%.3f "
        "Shadow=%.3f Highlight=%.3f "
        "DR=%.3f",
        global_analysis.mean,
        global_analysis.median,
        global_analysis.p05,
        global_analysis.p95,
        global_analysis.shadow_ratio,
        global_analysis.highlight_ratio,
        global_analysis.dynamic_range,
    )

    # --------------------------------------------------------
    # Semantic segmentation
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
    # Main subject
    # --------------------------------------------------------

    subject = detect_main_subject(
        masks
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

    # --------------------------------------------------------
    # Saliency
    # --------------------------------------------------------

    logger.info(
        "Calculating saliency map..."
    )

    saliency = calculate_saliency_map(
        image
    )

    # --------------------------------------------------------
    # Combined attention
    # --------------------------------------------------------

    attention = combine_subject_and_saliency(
        subject_attention,
        saliency,
    )

    # --------------------------------------------------------
    # Basic automatic white balance
    # --------------------------------------------------------

    image = apply_auto_white_balance(
        image
    )

    # --------------------------------------------------------
    # Auto tuning
    # --------------------------------------------------------

    logger.info(
        "Running automatic parameter search..."
    )

    tuned_image, params, score = auto_tune(
        image,
        global_analysis,
        attention,
        saliency,
    )

    image = tuned_image

    # --------------------------------------------------------
    # Sky
    # --------------------------------------------------------

    sky = create_sky_mask(
        image
    )

    sky_area = np.mean(
        sky > 0.25
    )

    if sky_area > 0.01:

        logger.info(
            "Sky detected: %.2f%%",
            sky_area * 100.0,
        )

        # Skyのハイライト保護
        image = apply_local_exposure(
            image,
            sky,
            -0.05,
        )

        # 空の彩度をほんの少し上げる
        image = apply_local_saturation(
            image,
            sky,
            1.04,
        )

    # --------------------------------------------------------
    # Person
    # --------------------------------------------------------

    if "person" in masks:

        person = masks["person"]

        person_area = np.mean(
            person > 0.25
        )

        if person_area > 0.002:

            logger.info(
                "Person detected: %.2f%%",
                person_area * 100.0,
            )

            # 人物をほんの少し明るく
            image = apply_local_exposure(
                image,
                person,
                0.05,
            )

            # 肌などの過剰彩度を抑制
            image = apply_local_saturation(
                image,
                person,
                0.98,
            )

    # --------------------------------------------------------
    # Plant
    # --------------------------------------------------------

    if "pottedplant" in masks:

        plant = masks["pottedplant"]

        plant_area = np.mean(
            plant > 0.25
        )

        if plant_area > 0.002:

            logger.info(
                "Plant detected: %.2f%%",
                plant_area * 100.0,
            )

            image = apply_local_saturation(
                image,
                plant,
                1.05,
            )

    # --------------------------------------------------------
    # Subject / Background separation
    # --------------------------------------------------------

    image = apply_subject_background_separation(
        image,
        attention,
    )

    # --------------------------------------------------------
    # Adaptive tone curve
    # --------------------------------------------------------

    image = apply_adaptive_tone_curve(
        image,
        global_analysis,
    )

    # --------------------------------------------------------
    # CLAHE
    # --------------------------------------------------------

    image = apply_clahe(
        image,
        clip_limit=1.3,
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

    # ISOが高いほど少し強く
    if iso >= 6400:
        denoise_strength = 0.55
    elif iso >= 3200:
        denoise_strength = 0.40
    elif iso >= 1600:
        denoise_strength = 0.30
    elif iso >= 800:
        denoise_strength = 0.22
    else:
        denoise_strength = 0.15

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
        amount=0.65,
    )

    # --------------------------------------------------------
    # Final clipping
    # --------------------------------------------------------

    image = np.clip(
        image,
        0.0,
        1.0,
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

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
            return [input_path]

        return []

    files = []

    for path in input_path.rglob("*"):

        if not path.is_file():
            continue

        if (
            path.suffix.lower()
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
            "Automatic RAW photo development "
            "with semantic segmentation, "
            "saliency and automatic tuning."
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
        choices=["cpu", "cuda"],
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

    # --------------------------------------------------------
    # RAW files
    # --------------------------------------------------------

    raw_files = collect_raw_files(
        input_path
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
    # Segmentation model
    # --------------------------------------------------------

    try:

        segmenter = SemanticSegmenter(
            device=args.device
        )

    except Exception as e:

        logger.error(
            "Failed to load segmentation model: %s",
            e,
        )

        logger.error(
            "If this is the first execution, "
            "PyTorch may need to download the "
            "pretrained model weights."
        )

        sys.exit(1)

    # --------------------------------------------------------
    # Process
    # --------------------------------------------------------

    success = 0
    failure = 0

    for raw_file in raw_files:

        # ファイル名を維持
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