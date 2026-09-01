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

OUTPUT_EXT = ".jpg"

# RAW現像
RAW_BPS = 16

# JPEG
JPEG_QUALITY = 95

# 自動露出の目標値
TARGET_MEAN = 0.42

# ハイライト判定
HIGHLIGHT_THRESHOLD = 0.98
HIGHLIGHT_RATIO_LIMIT = 0.003

# シャドウ判定
SHADOW_THRESHOLD = 0.025
SHADOW_RATIO_LIMIT = 0.20

# 彩度
SATURATION_MIN = 0.90
SATURATION_MAX = 1.15

# シャープネス
SHARPEN_SIGMA = 1.2
SHARPEN_AMOUNT = 0.35

# ノイズ低減
DENOISE_SIGMA_COLOR = 3
DENOISE_SIGMA_SPACE = 3

# CLAHE
CLAHE_CLIP_LIMIT = 1.5
CLAHE_GRID_SIZE = 8


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
    """
    uint16 / uint8 / float の画像を float32 [0, 1] にする。
    """
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
    return np.clip(img * 255.0 + 0.5, 0, 255).astype(np.uint8)


def calculate_luminance(img):
    """
    RGB画像から輝度を計算する。
    """
    return (
        0.2126 * img[:, :, 0]
        + 0.7152 * img[:, :, 1]
        + 0.0722 * img[:, :, 2]
    )


# ============================================================
# RAW
# ============================================================

def load_raw(filename):
    """
    RAWを読み込み、16bit RGBとして取得する。
    """

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
            "camera_wb": getattr(raw, "camera_whitebalance", None),
            "make": getattr(raw, "camera_make", ""),
            "model": getattr(raw, "camera_model", ""),
            "iso": getattr(raw, "camera_iso_speed", 0),
        }

    return normalize_image(rgb), metadata


# ============================================================
# Histogram analysis
# ============================================================

class ImageAnalysis:
    def __init__(self, image):
        self.image = image
        self.luminance = calculate_luminance(image)

        self.mean = float(np.mean(self.luminance))
        self.median = float(np.median(self.luminance))

        self.p01 = float(np.percentile(self.luminance, 1))
        self.p05 = float(np.percentile(self.luminance, 5))
        self.p95 = float(np.percentile(self.luminance, 95))
        self.p99 = float(np.percentile(self.luminance, 99))
        self.p999 = float(np.percentile(self.luminance, 99.9))

        self.shadow_ratio = float(
            np.mean(self.luminance < SHADOW_THRESHOLD)
        )

        self.highlight_ratio = float(
            np.mean(self.luminance > HIGHLIGHT_THRESHOLD)
        )

        self.dynamic_range = self.p99 - self.p01

    def print_summary(self):
        logging.info(
            "mean=%.3f median=%.3f p01=%.3f p99=%.3f",
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
# Auto Exposure
# ============================================================

def calculate_exposure(analysis):
    """
    ヒストグラムから自動露出量を決定する。

    戻り値:
        gain
        EV
    """

    mean = max(analysis.mean, 0.001)

    # 単純な平均輝度ではなく、中央値も考慮
    reference = 0.6 * mean + 0.4 * analysis.median

    target = TARGET_MEAN

    ratio = target / max(reference, 0.001)

    # gain -> EV
    ev = math.log2(ratio)

    # 過剰補正防止
    ev = np.clip(ev, -1.5, 1.5)

    gain = 2.0 ** ev

    # 白飛びが多い場合は露出を下げる
    if analysis.highlight_ratio > HIGHLIGHT_RATIO_LIMIT:
        gain *= 0.90

    # 暗部が極端に多い場合
    if analysis.shadow_ratio > 0.40 and analysis.highlight_ratio < 0.001:
        gain *= 1.08

    gain = float(np.clip(gain, 0.50, 2.00))

    ev = math.log2(gain)

    return gain, ev


def apply_exposure(image, gain):
    return np.clip(image * gain, 0.0, 1.0)


# ============================================================
# Highlight Recovery
# ============================================================

def recover_highlights(image):
    """
    ハイライト部分を緩やかに圧縮する。
    """

    luminance = calculate_luminance(image)

    mask = np.clip(
        (luminance - 0.75) / 0.25,
        0.0,
        1.0,
    )

    # smoothstep
    mask = mask * mask * (3.0 - 2.0 * mask)

    # RGBを個別に圧縮
    compressed = image / (
        1.0 + 0.35 * mask[:, :, None]
    )

    return np.clip(compressed, 0.0, 1.0)


# ============================================================
# Shadow Recovery
# ============================================================

def recover_shadows(image, analysis):
    """
    暗部を持ち上げる。

    gamma < 1
    """

    if analysis.shadow_ratio < 0.05:
        return image

    if analysis.shadow_ratio > 0.30:
        gamma = 0.78
    elif analysis.shadow_ratio > 0.20:
        gamma = 0.84
    else:
        gamma = 0.90

    # 影だけを主に持ち上げる
    luminance = calculate_luminance(image)

    mask = np.clip(
        (0.35 - luminance) / 0.35,
        0.0,
        1.0,
    )

    mask = mask * mask

    corrected = np.power(
        np.clip(image, 0.0, 1.0),
        gamma,
    )

    result = (
        image * (1.0 - mask[:, :, None])
        + corrected * mask[:, :, None]
    )

    return np.clip(result, 0.0, 1.0)


# ============================================================
# Tone Curve
# ============================================================

def apply_tone_curve(image):
    """
    緩やかなS字トーンカーブ。
    """

    x = np.linspace(0.0, 1.0, 4096)

    # Filmic-like S curve
    curve = (
        x
        + 0.12 * x * (1.0 - x) * (2.0 * x - 1.0)
    )

    curve = np.clip(curve, 0.0, 1.0)

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
    """
    ダイナミックレンジからコントラストを決める。
    """

    dr = analysis.dynamic_range

    if dr < 0.25:
        return 1.12

    if dr < 0.40:
        return 1.07

    if dr > 0.85:
        return 0.96

    return 1.02


def apply_contrast(image, amount):
    """
    中間調を基準にコントラスト調整。
    """

    midpoint = 0.45

    result = (
        (image - midpoint) * amount
        + midpoint
    )

    return np.clip(result, 0.0, 1.0)


# ============================================================
# White Balance
# ============================================================

def calculate_auto_wb(image):
    """
    Gray World方式の簡易自動WB。

    RAW現像では camera WB を使用しているため、
    大きな補正だけ行う。
    """

    means = np.mean(image.reshape(-1, 3), axis=0)

    target = np.mean(means)

    gains = target / np.maximum(means, 1e-6)

    # 極端な色補正を防止
    gains = np.clip(gains, 0.85, 1.15)

    return gains.astype(np.float32)


def apply_white_balance(image, gains):
    result = image * gains[None, None, :]

    return np.clip(result, 0.0, 1.0)


# ============================================================
# Saturation
# ============================================================

def calculate_saturation(image, analysis):
    """
    色の強さを画像から自動決定。
    """

    max_rgb = np.max(image, axis=2)
    min_rgb = np.min(image, axis=2)

    chroma = max_rgb - min_rgb

    avg_chroma = float(np.mean(chroma))

    if avg_chroma < 0.08:
        saturation = 1.12
    elif avg_chroma < 0.16:
        saturation = 1.06
    elif avg_chroma > 0.35:
        saturation = 0.94
    else:
        saturation = 1.00

    return float(
        np.clip(
            saturation,
            SATURATION_MIN,
            SATURATION_MAX,
        )
    )


def apply_saturation(image, amount):
    """
    HSVを使って彩度を変更。
    """

    img8 = denormalize_uint8(image)

    hsv = cv2.cvtColor(
        img8,
        cv2.COLOR_RGB2HSV,
    )

    saturation = hsv[:, :, 1].astype(np.float32)

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

    return result.astype(np.float32) / 255.0


# ============================================================
# CLAHE
# ============================================================

def apply_clahe(image):
    """
    LAB空間のLチャンネルにCLAHEを適用。
    """

    img8 = denormalize_uint8(image)

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

    return result.astype(np.float32) / 255.0


# ============================================================
# Noise Reduction
# ============================================================

def calculate_noise_strength(image, iso):
    """
    ISOからノイズ低減強度を決める。

    RAW metadataからISOを取得できない場合は弱め。
    """

    if iso is None or iso <= 0:
        return 0.25

    if iso < 400:
        return 0.10

    if iso < 800:
        return 0.20

    if iso < 1600:
        return 0.30

    if iso < 3200:
        return 0.40

    if iso < 6400:
        return 0.55

    return 0.70


def apply_denoise(image, strength):
    """
    OpenCV bilateral filter。

    強すぎるノイズ除去によるディテール消失を避ける。
    """

    if strength < 0.05:
        return image

    img8 = denormalize_uint8(image)

    sigma_color = DENOISE_SIGMA_COLOR * strength
    sigma_space = DENOISE_SIGMA_SPACE * strength

    result = cv2.bilateralFilter(
        img8,
        d=5,
        sigmaColor=max(1.0, sigma_color),
        sigmaSpace=max(1.0, sigma_space),
    )

    return result.astype(np.float32) / 255.0


# ============================================================
# Sharpen
# ============================================================

def apply_sharpen(image, strength=1.0):
    """
    アンシャープマスク。
    """

    img8 = denormalize_uint8(image)

    blurred = cv2.GaussianBlur(
        img8,
        (0, 0),
        SHARPEN_SIGMA,
    )

    amount = SHARPEN_AMOUNT * strength

    result = cv2.addWeighted(
        img8,
        1.0 + amount,
        blurred,
        -amount,
        0,
    )

    return result.astype(np.float32) / 255.0


# ============================================================
# Auto Develop
# ============================================================

def auto_develop(image, metadata):
    """
    自動現像メイン。
    """

    logging.info("Analyzing image...")

    analysis = ImageAnalysis(image)

    analysis.print_summary()

    # --------------------------------------------------------
    # Exposure
    # --------------------------------------------------------

    gain, ev = calculate_exposure(analysis)

    logging.info(
        "Exposure: %.2f EV",
        ev,
    )

    image = apply_exposure(
        image,
        gain,
    )

    # --------------------------------------------------------
    # White Balance
    # --------------------------------------------------------

    wb_gains = calculate_auto_wb(image)

    logging.info(
        "WB gains: R=%.3f G=%.3f B=%.3f",
        wb_gains[0],
        wb_gains[1],
        wb_gains[2],
    )

    image = apply_white_balance(
        image,
        wb_gains,
    )

    # --------------------------------------------------------
    # Highlight
    # --------------------------------------------------------

    image = recover_highlights(image)

    # --------------------------------------------------------
    # Shadow
    # --------------------------------------------------------

    image = recover_shadows(
        image,
        analysis,
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

    image = apply_tone_curve(image)

    # --------------------------------------------------------
    # Saturation
    # --------------------------------------------------------

    saturation = calculate_saturation(
        image,
        analysis,
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

    # 極端にフラットな画像だけ使用
    if analysis.dynamic_range < 0.35:
        logging.info("Applying CLAHE")
        image = apply_clahe(image)

    # --------------------------------------------------------
    # Noise Reduction
    # --------------------------------------------------------

    iso = metadata.get("iso", 0)

    noise_strength = calculate_noise_strength(
        image,
        iso,
    )

    logging.info(
        "Noise reduction: %.2f (ISO=%s)",
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

    # ノイズが多い場合はシャープを弱める
    sharpen_strength = 1.0 - 0.5 * noise_strength

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

def save_image(image, output_file):
    img8 = denormalize_uint8(image)

    logging.info(
        "Saving: %s",
        output_file,
    )

    iio.imwrite(
        output_file,
        img8,
        extension=".jpg",
        quality=JPEG_QUALITY,
    )


# ============================================================
# Single file
# ============================================================

def process_file(input_file, output_dir):

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
            / f"{input_file.stem}{OUTPUT_EXT}"
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
# Collect files
# ============================================================

def collect_raw_files(input_path):

    if input_path.is_file():

        if input_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            return [input_path]

        return []

    files = []

    for file in input_path.rglob("*"):

        if (
            file.is_file()
            and file.suffix.lower()
            in SUPPORTED_EXTENSIONS
        ):
            files.append(file)

    return sorted(files)


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Automatic RAW developer"
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

    input_path = args.input
    output_dir = args.output

    if not input_path.exists():

        logging.error(
            "Input does not exist: %s",
            input_path,
        )

        return 1

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    files = collect_raw_files(
        input_path
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
            output_dir,
        ):
            success += 1

    logging.info(
        "Finished: %d/%d files",
        success,
        len(files),
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())