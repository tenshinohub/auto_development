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
RAW_BPS = 16

TARGET_MEAN = 0.42

SHADOW_THRESHOLD = 0.025
HIGHLIGHT_THRESHOLD = 0.98

CLAHE_CLIP_LIMIT = 1.5
CLAHE_GRID_SIZE = 8

SHARPEN_SIGMA = 1.2
SHARPEN_AMOUNT = 0.35

DENOISE_SIGMA_COLOR = 3
DENOISE_SIGMA_SPACE = 3

# セグメンテーション処理用の最大サイズ
SEGMENTATION_SIZE = 768

# セグメンテーション信頼度
SEGMENTATION_CONFIDENCE = 0.50


# ============================================================
# COCO / Pascal-style class IDs
# ============================================================

# DeepLabV3 torchvision pretrained model
#
# COCOではなく、VOC系21クラス
#
# 0  background
# 1  aeroplane
# 2  bicycle
# 3  bird
# 4  boat
# 5  bottle
# 6  bus
# 7  car
# 8  cat
# 9  chair
# 10 cow
# 11 diningtable
# 12 dog
# 13 horse
# 14 motorbike
# 15 person
# 16 pottedplant
# 17 sheep
# 18 sofa
# 19 train
# 20 tvmonitor

CLASS_BACKGROUND = 0
CLASS_PERSON = 15
CLASS_PLANT = 16


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

    max_value = img.max()

    if max_value > 1.0:
        img /= max_value

    return np.clip(img, 0.0, 1.0)


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
            output_bps=RAW_BPS,
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

    def __init__(self, image, mask=None):

        self.image = image

        self.luminance = calculate_luminance(
            image
        )

        if mask is None:
            values = self.luminance.reshape(-1)
        else:
            values = self.luminance[mask > 0]

            if len(values) == 0:
                values = self.luminance.reshape(-1)

        self.mean = float(
            np.mean(values)
        )

        self.median = float(
            np.median(values)
        )

        self.p01 = float(
            np.percentile(values, 1)
        )

        self.p05 = float(
            np.percentile(values, 5)
        )

        self.p95 = float(
            np.percentile(values, 95)
        )

        self.p99 = float(
            np.percentile(values, 99)
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
            self.p99 - self.p01
        )


# ============================================================
# Semantic Segmentation
# ============================================================

class SemanticSegmenter:

    def __init__(self):

        logging.info(
            "Loading semantic segmentation model..."
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

    def segment(self, image):

        h, w, _ = image.shape

        # 大きすぎるRAW画像を縮小
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

        # torchvisionはPIL Imageを想定
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

            logits = result[
                "out"
            ]

            probabilities = torch.softmax(
                logits,
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

        # 元サイズへ戻す
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

        # 信頼度の低い領域は背景扱い
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
    """
    セグメンテーション結果＋画像情報から
    現像用の領域マスクを作る。
    """

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
    #
    # DeepLabV3のVOCクラスにはskyがないため、
    # 上部＋色＋エッジ情報から補助的に推定
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
        1.0 - y / h
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
        sky.astype(
            np.float32
        )
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
# Auto Exposure
# ============================================================

def calculate_exposure(
    analysis,
):

    """
    ヒストグラムから自動露出を決定する。

    平均値だけではなく、
    median / p05 / p95 を利用する。
    """

    median = analysis.median
    p05 = analysis.p05
    p95 = analysis.p95

    # --------------------------------------------------------
    # 基本露出
    # --------------------------------------------------------

    reference = (
        median * 0.65
        + p05 * 0.10
        + p95 * 0.25
    )

    target = 0.42

    ratio = (
        target
        / max(reference, 0.001)
    )

    ev = math.log2(
        ratio
    )

    # --------------------------------------------------------
    # Dynamic Rangeが広い場合
    # --------------------------------------------------------

    if analysis.dynamic_range > 0.85:

        # 明るくしすぎない
        ev -= 0.15

    # --------------------------------------------------------
    # ハイライトが危険な場合
    # --------------------------------------------------------

    if analysis.highlight_ratio > 0.003:

        ev -= 0.20

    elif analysis.highlight_ratio > 0.001:

        ev -= 0.08

    # --------------------------------------------------------
    # シャドウが多い場合
    # --------------------------------------------------------

    if (
        analysis.shadow_ratio > 0.35
        and analysis.highlight_ratio < 0.001
    ):

        ev += 0.15

    # --------------------------------------------------------
    # 過剰補正を防ぐ
    # --------------------------------------------------------

    ev = float(
        np.clip(
            ev,
            -1.5,
            1.5,
        )
    )

    gain = 2.0 ** ev

    return gain, ev
    
    """
    reference = (
        0.6 * analysis.mean
        + 0.4 * analysis.median
    )

    ratio = (
        TARGET_MEAN
        / max(
            reference,
            0.001,
        )
    )

    ev = math.log2(
        ratio
    )

    ev = np.clip(
        ev,
        -1.5,
        1.5,
    )

    gain = 2.0 ** ev

    if (
        analysis.highlight_ratio
        > 0.003
    ):
        gain *= 0.90

    if (
        analysis.shadow_ratio
        > 0.40
        and analysis.highlight_ratio
        < 0.001
    ):
        gain *= 1.08

    gain = float(
        np.clip(
            gain,
            0.50,
            2.00,
        )
    )

    return gain, math.log2(gain)
    """

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
# White Balance
# ============================================================

def calculate_auto_wb(
    image,
    sky_mask=None,
):
    """
    色かぶりを抑えるための自動WB。

    Gray Worldをそのまま使うのではなく、

    - 極端な暗部
    - 極端なハイライト
    - 高彩度領域
    - 空

    をWB推定から除外する。

    戻り値:
        RGB gain
    """

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

    chroma = max_rgb - min_rgb

    # --------------------------------------------------------
    # WB候補領域
    # --------------------------------------------------------

    mask = (
        (luminance > 0.12)
        &
        (luminance < 0.85)
        &
        (chroma < 0.12)
    )

    # 空はWB計算から除外
    if sky_mask is not None:
        mask &= (
            sky_mask < 0.30
        )

    if np.sum(mask) < 1000:
        logging.info(
            "Not enough WB pixels. "
            "Using camera WB."
        )

        return np.ones(
            3,
            dtype=np.float32,
        )

    # --------------------------------------------------------
    # Robust RGB statistics
    # --------------------------------------------------------

    r_values = r[mask]
    g_values = g[mask]
    b_values = b[mask]

    r_mean = float(
        np.median(r_values)
    )

    g_mean = float(
        np.median(g_values)
    )

    b_mean = float(
        np.median(b_values)
    )

    target = (
        r_mean
        + g_mean
        + b_mean
    ) / 3.0

    gains = np.array(
        [
            target / max(r_mean, 1e-6),
            target / max(g_mean, 1e-6),
            target / max(b_mean, 1e-6),
        ],
        dtype=np.float32,
    )

    # --------------------------------------------------------
    # 強すぎるWB補正は禁止
    # --------------------------------------------------------

    gains = np.clip(
        gains,
        0.90,
        1.10,
    )

    # Greenを基準にする
    gains /= gains[1]

    logging.info(
        "Auto WB statistics: "
        "R=%.3f G=%.3f B=%.3f",
        r_mean,
        g_mean,
        b_mean,
    )

    logging.info(
        "Auto WB gains: "
        "R=%.3f G=%.3f B=%.3f",
        gains[0],
        gains[1],
        gains[2],
    )

    return gains


    """
    means = np.mean(
        image.reshape(-1, 3),
        axis=0,
    )

    target = np.mean(
        means
    )

    gains = (
        target
        / np.maximum(
            means,
            1e-6,
        )
    )

    gains = np.clip(
        gains,
        0.85,
        1.15,
    )

    return gains.astype(
        np.float32
    )
    """


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

    """
    ハイライトを緩やかに圧縮する。

    完全な白飛び領域は無理に復元しない。
    """

    if np.max(mask) <= 0:
        return image

    luminance = calculate_luminance(
        image
    )

    # 0.75～1.0の領域だけ対象
    soft_mask = np.clip(
        (luminance - 0.75)
        / 0.25,
        0.0,
        1.0,
    )

    soft_mask = (
        soft_mask
        * soft_mask
    )

    soft_mask *= mask

    # 完全白はほぼそのまま
    recoverable = (
        1.0 - np.clip(
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
    
    """
    if np.max(mask) <= 0:
        return image

    result = image / (
        1.0
        + strength
        * mask[:, :, None]
    )

    return np.clip(
        result,
        0.0,
        1.0,
    )
    """


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
        * (1.0 - strength * mask[:, :, None])
        + corrected
        * (strength * mask[:, :, None])
    )

    return np.clip(
        result,
        0.0,
        1.0,
    )


# ============================================================
# Tone Curve
# ============================================================

def apply_tone_curve(
    image,
):

    x = np.linspace(
        0.0,
        1.0,
        4096,
    )

    curve = (
        x
        + 0.12
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
    ).astype(
        np.int32
    )

    return curve[index]

def apply_adaptive_tone_curve(
    image,
    analysis,
):
    """
    Dynamic Rangeに応じて
    トーンカーブの強さを変える。
    """

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
    ).astype(
        np.int32
    )

    return curve[index]

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
        (image - midpoint)
        * amount
        + midpoint
    )

    return np.clip(
        result,
        0.0,
        1.0,
    )


# ============================================================
# Global saturation
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
        max_rgb - min_rgb
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
# CLAHE
# ============================================================

def apply_clahe(
    image,
):

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

    result = cv2.merge(
        (l, a, b)
    )

    result = cv2.cvtColor(
        result,
        cv2.COLOR_LAB2RGB,
    )

    return (
        result.astype(
            np.float32
        )
        / 255.0
    )


# ============================================================
# Noise Reduction
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
# Auto Develop
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
        "Global mean=%.3f median=%.3f",
        global_analysis.mean,
        global_analysis.median,
    )

    logging.info(
        "Dynamic range=%.3f",
        global_analysis.dynamic_range,
    )

    # ========================================================
    # Semantic segmentation
    # ========================================================

    labels, confidence = (
        segmenter.segment(
            image
        )
    )

    masks = build_region_masks(
        image,
        labels,
    )

    for name, mask in masks.items():

        ratio = float(
            np.mean(mask > 0.25)
        )

        if ratio > 0.005:

            logging.info(
                "Region %-10s %.1f%%",
                name,
                ratio * 100,
            )

    # ========================================================
    # Global exposure
    # ========================================================

    gain, ev = calculate_exposure(
        global_analysis
    )

    logging.info(
        "Global exposure: %.2f EV",
        ev,
    )

    image = apply_exposure(
        image,
        gain,
    )

    # ========================================================
    # White balance
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
    # Shadow
    # ========================================================

    image = recover_shadows(
        image,
        masks["shadow"],
        strength=0.18,
    )

    # ========================================================
    # Highlight
    # ========================================================

    image = recover_highlights(
        image,
        masks["highlight"],
        strength=0.30,
    )

    # ========================================================
    # Sky
    # ========================================================

    sky = masks["sky"]

    if np.mean(sky > 0.25) > 0.005:

        # 空を少し暗くする
        image = apply_local_exposure(
            image,
            sky,
            0.94,
        )

        # 空の彩度を少しだけ上げる
        image = apply_local_saturation(
            image,
            sky,
            1.08,
        )

    # ========================================================
    # Person
    # ========================================================

    person = masks["person"]

    if np.mean(person > 0.25) > 0.002:

        # 人物を少し明るくする
        image = apply_local_exposure(
            image,
            person,
            1.06,
        )

        # 人物の彩度は控えめ
        image = apply_local_saturation(
            image,
            person,
            0.98,
        )

    # ========================================================
    # Vegetation
    # ========================================================

    plant = masks["plant"]

    if np.mean(plant > 0.25) > 0.002:

        image = apply_local_saturation(
            image,
            plant,
            1.05,
        )

    # ========================================================
    # Global contrast
    # ========================================================

    contrast = calculate_contrast(
        global_analysis
    )

    logging.info(
        "Contrast: %.3f",
        contrast,
    )

    image = apply_contrast(
        image,
        contrast,
    )

    # ========================================================
    # Tone curve
    # ========================================================

    image = apply_adaptive_tone_curve(
        image,
        global_analysis,
    )
    
    """
    image = apply_tone_curve(
        image
    )
    """

    # ========================================================
    # Global saturation
    # ========================================================

    saturation = calculate_saturation(
        image
    )

    logging.info(
        "Saturation: %.3f",
        saturation,
    )

    image = apply_local_saturation(
        image,
        np.ones(
            image.shape[:2],
            dtype=np.float32,
        ),
        saturation,
    )

    # ========================================================
    # CLAHE
    # ========================================================

    if (
        global_analysis.dynamic_range
        < 0.35
    ):

        logging.info(
            "Applying CLAHE"
        )

        image = apply_clahe(
            image
        )

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
        "Noise reduction: %.2f ISO=%s",
        noise_strength,
        iso,
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

    logging.info(
        "Sharpen: %.2f",
        sharpen_strength,
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
# File collection
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
            "Automatic RAW developer"
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

    # --------------------------------------------------------
    # モデルは一度だけロード
    # --------------------------------------------------------

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