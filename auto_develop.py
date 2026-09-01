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

HIGHLIGHT_THRESHOLD = 0.98
SHADOW_THRESHOLD = 0.025

HIGHLIGHT_RATIO_LIMIT = 0.003
SHADOW_RATIO_LIMIT = 0.20

CLAHE_CLIP_LIMIT = 1.5
CLAHE_GRID_SIZE = 8

SHARPEN_SIGMA = 1.2
SHARPEN_AMOUNT = 0.35

DENOISE_SIGMA_COLOR = 3
DENOISE_SIGMA_SPACE = 3


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

    logging.info("RAW loading: %s", filename)

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
            "make": getattr(raw, "camera_make", ""),
            "model": getattr(raw, "camera_model", ""),
            "iso": getattr(raw, "camera_iso_speed", 0),
        }

    return normalize_image(rgb), metadata


# ============================================================
# Image Analysis
# ============================================================

class ImageAnalysis:

    def __init__(self, image):

        self.image = image
        self.luminance = calculate_luminance(image)

        self.height = image.shape[0]
        self.width = image.shape[1]

        self.mean = float(
            np.mean(self.luminance)
        )

        self.median = float(
            np.median(self.luminance)
        )

        self.p01 = float(
            np.percentile(
                self.luminance,
                1
            )
        )

        self.p05 = float(
            np.percentile(
                self.luminance,
                5
            )
        )

        self.p95 = float(
            np.percentile(
                self.luminance,
                95
            )
        )

        self.p99 = float(
            np.percentile(
                self.luminance,
                99
            )
        )

        self.p999 = float(
            np.percentile(
                self.luminance,
                99.9
            )
        )

        self.shadow_ratio = float(
            np.mean(
                self.luminance
                < SHADOW_THRESHOLD
            )
        )

        self.highlight_ratio = float(
            np.mean(
                self.luminance
                > HIGHLIGHT_THRESHOLD
            )
        )

        self.dynamic_range = (
            self.p99 - self.p01
        )

    def print_summary(self):

        logging.info(
            "mean=%.3f median=%.3f "
            "p01=%.3f p99=%.3f",
            self.mean,
            self.median,
            self.p01,
            self.p99,
        )

        logging.info(
            "shadow=%.2f%% highlight=%.2f%%",
            self.shadow_ratio * 100,
            self.highlight_ratio * 100,
        )


# ============================================================
# Sky Detection
# ============================================================

def detect_sky(image):
    """
    学習モデルを使わない簡易的な空検出。

    条件:
        ・画像上部を優先
        ・青成分が強い
        ・彩度がある程度ある
        ・明るさが極端に低くない
    """

    h, w, _ = image.shape

    r = image[:, :, 0]
    g = image[:, :, 1]
    b = image[:, :, 2]

    luminance = calculate_luminance(image)

    chroma = np.maximum.reduce(
        [r, g, b]
    ) - np.minimum.reduce(
        [r, g, b]
    )

    # 青が赤・緑より強い
    blue_condition = (
        (b > r * 1.05)
        & (b > g * 0.98)
    )

    saturation_condition = chroma > 0.04

    brightness_condition = (
        luminance > 0.12
    )

    mask = (
        blue_condition
        & saturation_condition
        & brightness_condition
    )

    # 上部ほど空である可能性を高くする
    y = np.arange(h)[:, None]

    vertical_weight = (
        1.0 - y / h
    )

    vertical_weight = np.clip(
        vertical_weight * 1.5,
        0.0,
        1.0,
    )

    mask = (
        mask
        & (
            vertical_weight > 0.35
        )
    )

    # ノイズ除去
    mask8 = (
        mask.astype(np.uint8) * 255
    )

    kernel = np.ones(
        (15, 15),
        np.uint8,
    )

    mask8 = cv2.morphologyEx(
        mask8,
        cv2.MORPH_CLOSE,
        kernel,
    )

    mask8 = cv2.GaussianBlur(
        mask8,
        (0, 0),
        5,
    )

    return mask8.astype(
        np.float32
    ) / 255.0


# ============================================================
# Face Detection
# ============================================================

def detect_faces(image):
    """
    OpenCV Haar Cascadeによる顔検出。
    """

    gray = cv2.cvtColor(
        denormalize_uint8(image),
        cv2.COLOR_RGB2GRAY,
    )

    cascade_path = cv2.data.haarcascades + (
        "haarcascade_frontalface_default.xml"
    )

    cascade = cv2.CascadeClassifier(
        cascade_path
    )

    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40),
    )

    mask = np.zeros(
        gray.shape,
        dtype=np.float32,
    )

    for x, y, w, h in faces:

        # 少し広めに保護
        pad_x = int(w * 0.25)
        pad_y = int(h * 0.35)

        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)

        x2 = min(
            image.shape[1],
            x + w + pad_x,
        )

        y2 = min(
            image.shape[0],
            y + h + pad_y,
        )

        mask[y1:y2, x1:x2] = 1.0

    if len(faces) > 0:
        logging.info(
            "Faces detected: %d",
            len(faces),
        )

    return mask


# ============================================================
# Local masks
# ============================================================

def detect_shadow_mask(image):
    luminance = calculate_luminance(image)

    mask = np.clip(
        (0.35 - luminance) / 0.35,
        0.0,
        1.0,
    )

    return mask * mask


def detect_highlight_mask(image):
    luminance = calculate_luminance(image)

    mask = np.clip(
        (luminance - 0.70) / 0.30,
        0.0,
        1.0,
    )

    return mask * mask


# ============================================================
# Exposure
# ============================================================

def calculate_exposure(analysis):

    reference = (
        0.6 * analysis.mean
        + 0.4 * analysis.median
    )

    ratio = (
        TARGET_MEAN
        / max(reference, 0.001)
    )

    ev = math.log2(ratio)

    ev = np.clip(
        ev,
        -1.5,
        1.5,
    )

    gain = 2.0 ** ev

    if (
        analysis.highlight_ratio
        > HIGHLIGHT_RATIO_LIMIT
    ):
        gain *= 0.90

    if (
        analysis.shadow_ratio > 0.40
        and analysis.highlight_ratio < 0.001
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


def apply_exposure(image, gain):

    return np.clip(
        image * gain,
        0.0,
        1.0,
    )


# ============================================================
# White Balance
# ============================================================

def calculate_auto_wb(image):

    means = np.mean(
        image.reshape(-1, 3),
        axis=0,
    )

    target = np.mean(means)

    gains = (
        target
        / np.maximum(means, 1e-6)
    )

    gains = np.clip(
        gains,
        0.85,
        1.15,
    )

    return gains.astype(
        np.float32
    )


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
# Highlight
# ============================================================

def recover_highlights(
    image,
    highlight_mask=None,
):

    luminance = calculate_luminance(
        image
    )

    mask = np.clip(
        (luminance - 0.70) / 0.30,
        0.0,
        1.0,
    )

    mask = mask * mask

    if highlight_mask is not None:
        mask *= highlight_mask

    compressed = image / (
        1.0
        + 0.35 * mask[:, :, None]
    )

    return np.clip(
        compressed,
        0.0,
        1.0,
    )


# ============================================================
# Shadow
# ============================================================

def recover_shadows(
    image,
    shadow_mask,
):

    luminance = calculate_luminance(
        image
    )

    strength = np.clip(
        (0.35 - luminance) / 0.35,
        0.0,
        1.0,
    )

    strength *= shadow_mask

    gamma = 0.82

    corrected = np.power(
        np.clip(image, 0.0, 1.0),
        gamma,
    )

    result = (
        image
        * (1.0 - strength[:, :, None])
        + corrected
        * strength[:, :, None]
    )

    return np.clip(
        result,
        0.0,
        1.0,
    )


# ============================================================
# Sky correction
# ============================================================

def correct_sky(
    image,
    sky_mask,
):

    if np.mean(sky_mask) < 0.005:
        return image

    logging.info(
        "Sky detected: %.1f%%",
        np.mean(sky_mask) * 100,
    )

    # 空のハイライトを少し抑える
    luminance = calculate_luminance(
        image
    )

    highlight = np.clip(
        (luminance - 0.60) / 0.40,
        0.0,
        1.0,
    )

    highlight *= sky_mask

    result = image / (
        1.0
        + 0.18 * highlight[:, :, None]
    )

    # 空の彩度を少しだけ上げる
    img8 = denormalize_uint8(
        np.clip(result, 0.0, 1.0)
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

    saturation *= (
        1.0
        + 0.08 * sky_mask
    )

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
        ) / 255.0
    )


# ============================================================
# Face protection
# ============================================================

def protect_faces(
    image,
    face_mask,
):

    if np.max(face_mask) == 0:
        return image

    luminance = calculate_luminance(
        image
    )

    # 顔が暗い場合だけ少し持ち上げる
    brightness = np.clip(
        (0.40 - luminance)
        / 0.40,
        0.0,
        1.0,
    )

    correction = (
        1.0
        + 0.12
        * brightness
        * face_mask
    )

    result = (
        image
        * correction[:, :, None]
    )

    return np.clip(
        result,
        0.0,
        1.0,
    )


# ============================================================
# Tone Curve
# ============================================================

def apply_tone_curve(image):

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

    lut_index = np.clip(
        image * 4095.0,
        0,
        4095,
    ).astype(np.int32)

    return curve[lut_index]


# ============================================================
# Contrast
# ============================================================

def calculate_contrast(analysis):

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
# Saturation
# ============================================================

def calculate_saturation(image):

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


def apply_saturation(
    image,
    amount,
):

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

    saturation *= amount

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
        ) / 255.0
    )


# ============================================================
# CLAHE
# ============================================================

def apply_clahe(image):

    img8 = denormalize_uint8(
        image
    )

    lab = cv2.cvtColor(
        img8,
        cv2.COLOR_RGB2LAB,
    )

    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT,
        tileGridSize=(
            CLAHE_GRID_SIZE,
            CLAHE_GRID_SIZE,
        ),
    )

    l = clahe.apply(l)

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
        ) / 255.0
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
        d=5,
        sigmaColor=max(
            1.0,
            sigma_color,
        ),
        sigmaSpace=max(
            1.0,
            sigma_space,
        ),
    )

    return (
        result.astype(
            np.float32
        ) / 255.0
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
        ) / 255.0
    )


# ============================================================
# Auto Develop
# ============================================================

def auto_develop(
    image,
    metadata,
):

    logging.info(
        "Analyzing image..."
    )

    analysis = ImageAnalysis(
        image
    )

    analysis.print_summary()

    # --------------------------------------------------------
    # Scene detection
    # --------------------------------------------------------

    logging.info(
        "Detecting regions..."
    )

    sky_mask = detect_sky(
        image
    )

    face_mask = detect_faces(
        image
    )

    # --------------------------------------------------------
    # Exposure
    # --------------------------------------------------------

    gain, ev = calculate_exposure(
        analysis
    )

    logging.info(
        "Exposure: %.2f EV",
        ev,
    )

    image = apply_exposure(
        image,
        gain,
    )

    # --------------------------------------------------------
    # White balance
    # --------------------------------------------------------

    wb_gains = calculate_auto_wb(
        image
    )

    logging.info(
        "WB: R=%.3f G=%.3f B=%.3f",
        wb_gains[0],
        wb_gains[1],
        wb_gains[2],
    )

    image = apply_white_balance(
        image,
        wb_gains,
    )

    # --------------------------------------------------------
    # Shadows
    # --------------------------------------------------------

    shadow_mask = detect_shadow_mask(
        image
    )

    image = recover_shadows(
        image,
        shadow_mask,
    )

    # --------------------------------------------------------
    # Highlights
    # --------------------------------------------------------

    highlight_mask = detect_highlight_mask(
        image
    )

    image = recover_highlights(
        image,
        highlight_mask,
    )

    # --------------------------------------------------------
    # Sky
    # --------------------------------------------------------

    image = correct_sky(
        image,
        sky_mask,
    )

    # --------------------------------------------------------
    # Face
    # --------------------------------------------------------

    image = protect_faces(
        image,
        face_mask,
    )

    # --------------------------------------------------------
    # Contrast
    # --------------------------------------------------------

    contrast = calculate_contrast(
        analysis
    )

    logging.info(
        "Contrast: %.3f",
        contrast,
    )

    image = apply_contrast(
        image,
        contrast,
    )

    # --------------------------------------------------------
    # Tone curve
    # --------------------------------------------------------

    image = apply_tone_curve(
        image
    )

    # --------------------------------------------------------
    # Saturation
    # --------------------------------------------------------

    saturation = calculate_saturation(
        image
    )

    logging.info(
        "Saturation: %.3f",
        saturation,
    )

    image = apply_saturation(
        image,
        saturation,
    )

    # --------------------------------------------------------
    # CLAHE
    # --------------------------------------------------------

    if analysis.dynamic_range < 0.35:

        logging.info(
            "Applying CLAHE"
        )

        image = apply_clahe(
            image
        )

    # --------------------------------------------------------
    # Noise reduction
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Sharpen
    # --------------------------------------------------------

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
# File processing
# ============================================================

def process_file(
    input_file,
    output_dir,
):

    try:

        image, metadata = load_raw(
            input_file
        )

        developed = auto_develop(
            image,
            metadata,
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
# Collect RAW files
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

    success = 0

    for file in files:

        if process_file(
            file,
            args.output,
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