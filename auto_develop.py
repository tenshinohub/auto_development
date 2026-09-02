#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
auto_develop_v16.py

Automatic RAW Development v16

Main concept
------------

v15:
    region -> fixed profile

v16:
    region
      ↓
    measure actual condition
      ↓
    calculate correction
      ↓
    apply correction

The program attempts to reduce fixed development values and
derive corrections from the actual image.

Pipeline
--------

RAW
 ↓
rawpy / LibRaw
 ↓
camera RGB
 ↓
camera RGB -> XYZ -> linear sRGB
 ↓
image analysis
 ↓
semantic segmentation
 ↓
saliency
 ↓
subject ranking
 ↓
scene classification
 ↓
global optimization
 ↓
region detection
 ↓
region condition analysis
 ↓
automatic region correction
 ↓
subject/background balance
 ↓
luminance tone mapping
 ↓
denoise
 ↓
sharpen
 ↓
JPEG


Usage
-----

    python3 auto_develop_v16.py ./RAW -o ./output --device cuda


Requirements
------------

    pip install rawpy numpy opencv-python pillow torch torchvision scipy

Optional
--------

    sudo apt install exiftool
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import rawpy


try:
    import torch
    import torchvision
    import torchvision.transforms.functional as TF
except ImportError:
    torch = None
    torchvision = None
    TF = None


try:
    from PIL import Image, ExifTags
except ImportError:
    Image = None
    ExifTags = None


# ============================================================
# Constants
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

CLASS_ID = {
    name: i
    for i, name in enumerate(VOC_CLASSES)
}


SUBJECT_CLASSES = {
    "person": 1.20,
    "bird": 1.05,
    "cat": 1.05,
    "dog": 1.05,
    "horse": 1.05,
    "cow": 1.00,
    "sheep": 1.00,
    "car": 1.00,
    "bus": 1.00,
    "motorbike": 1.00,
    "bicycle": 1.00,
    "boat": 1.00,
    "train": 1.00,
    "pottedplant": 0.90,
    "bottle": 0.85,
}


XYZ_TO_SRGB = np.array(
    [
        [ 3.2406, -1.5372, -0.4986],
        [-0.9689,  1.8758,  0.0415],
        [ 0.0557, -0.2040,  1.0570],
    ],
    dtype=np.float32,
)


# ============================================================
# Dataclasses
# ============================================================

@dataclass
class ExifMetadata:
    make: str = ""
    model: str = ""

    lens_make: str = ""
    lens_model: str = ""

    iso: Optional[float] = None
    exposure_time: Optional[float] = None
    f_number: Optional[float] = None
    focal_length: Optional[float] = None

    white_balance: str = ""
    color_temperature: Optional[float] = None
    color_space: str = ""

    source: str = "unknown"


@dataclass
class CameraProfile:
    make: str = ""
    model: str = ""
    family: str = "unknown"

    iso: Optional[float] = None

    black_level: Optional[np.ndarray] = None
    white_level: Optional[float] = None
    white_level_per_channel: Optional[np.ndarray] = None

    camera_wb: Optional[np.ndarray] = None

    color_matrix: Optional[np.ndarray] = None
    rgb_xyz_matrix: Optional[np.ndarray] = None

    color_desc: str = ""
    num_colors: int = 3

    raw_width: int = 0
    raw_height: int = 0

    raw_pattern: Optional[np.ndarray] = None

    lens_make: str = ""
    lens_model: str = ""

    exposure_time: Optional[float] = None
    f_number: Optional[float] = None
    focal_length: Optional[float] = None

    white_balance: str = ""
    color_temperature: Optional[float] = None
    color_space: str = ""

    metadata_source: str = "unknown"
    libraw_version: str = ""


@dataclass
class ImageStats:
    mean: float
    median: float

    p01: float
    p05: float
    p95: float
    p99: float

    shadow_ratio: float
    highlight_ratio: float

    dynamic_range: float
    saturation_ratio: float

    edge_density: float
    warm_ratio: float

    contrast: float


@dataclass
class ShootingCondition:
    iso_factor: float = 0.0
    low_light: float = 0.0
    motion_risk: float = 0.0
    shallow_dof: float = 0.0

    wide_angle: float = 0.0
    telephoto: float = 0.0

    estimated_noise: float = 0.0


@dataclass
class SubjectCandidate:
    class_name: str
    confidence: float
    area: float

    center_score: float
    saliency: float

    local_contrast: float
    colorfulness: float

    score: float

    mask: Optional[np.ndarray] = None


@dataclass
class RegionStats:
    coverage: float

    brightness: float
    contrast: float
    saturation: float

    shadow_ratio: float
    highlight_ratio: float

    red_ratio: float
    green_ratio: float
    blue_ratio: float

    warm_bias: float
    cool_bias: float

    edge_density: float

    colorfulness: float

    detail: float


@dataclass
class RegionCorrection:
    exposure: float
    contrast: float
    saturation: float

    highlight_protection: float
    shadow_lift: float

    clarity: float
    hue_shift: float

    reason: str = ""


# ============================================================
# Utility
# ============================================================

def clamp01(x):
    return np.clip(x, 0.0, 1.0)


def clamp(x, low, high):
    return np.clip(x, low, high)


def safe_float(value):
    try:
        return float(value)
    except Exception:
        return None


def normalize_percentile(
    x,
    p_low=1.0,
    p_high=99.0,
):
    lo = np.percentile(x, p_low)
    hi = np.percentile(x, p_high)

    if hi <= lo + 1e-8:
        return np.zeros_like(
            x,
            dtype=np.float32,
        )

    return np.clip(
        (x - lo) / (hi - lo),
        0.0,
        1.0,
    )


def resize_for_analysis(
    img,
    max_size=1024,
):
    h, w = img.shape[:2]

    scale = min(
        1.0,
        max_size / max(h, w),
    )

    if scale == 1.0:
        return img

    return cv2.resize(
        img,
        (
            int(w * scale),
            int(h * scale),
        ),
        interpolation=cv2.INTER_AREA,
    )


# ============================================================
# ExifTool
# ============================================================

def run_exiftool(path: Path):

    try:
        result = subprocess.run(
            [
                "exiftool",
                "-j",
                "-n",

                "-Make",
                "-Model",
                "-CameraModelName",
                "-UniqueCameraModel",

                "-LensMake",
                "-LensModel",

                "-ISO",
                "-ExposureTime",
                "-FNumber",
                "-FocalLength",

                "-WhiteBalance",
                "-ColorTemperature",
                "-ColorSpace",

                str(path),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            return None

        data = json.loads(
            result.stdout
        )

        if not data:
            return None

        return data[0]

    except Exception:
        return None


def read_pillow_exif(path: Path):

    if Image is None:
        return {}

    try:
        image = Image.open(path)

        exif = image.getexif()

        if not exif:
            return {}

        result = {}

        for tag_id, value in exif.items():

            name = ExifTags.TAGS.get(
                tag_id,
                tag_id,
            )

            result[name] = value

        return result

    except Exception:
        return {}


def extract_metadata(path: Path):

    meta = ExifMetadata()

    data = run_exiftool(path)

    if data:

        meta.source = "exiftool"

        meta.make = str(
            data.get("Make")
            or data.get("CameraModelName")
            or ""
        )

        meta.model = str(
            data.get("Model")
            or data.get("UniqueCameraModel")
            or ""
        )

        meta.lens_make = str(
            data.get("LensMake")
            or ""
        )

        meta.lens_model = str(
            data.get("LensModel")
            or ""
        )

        meta.iso = safe_float(
            data.get("ISO")
        )

        meta.exposure_time = safe_float(
            data.get("ExposureTime")
        )

        meta.f_number = safe_float(
            data.get("FNumber")
        )

        meta.focal_length = safe_float(
            data.get("FocalLength")
        )

        meta.white_balance = str(
            data.get("WhiteBalance")
            or ""
        )

        meta.color_temperature = safe_float(
            data.get("ColorTemperature")
        )

        meta.color_space = str(
            data.get("ColorSpace")
            or ""
        )

        return meta

    # Pillow fallback

    data = read_pillow_exif(path)

    if data:

        meta.source = "pillow"

        meta.make = str(
            data.get("Make")
            or ""
        )

        meta.model = str(
            data.get("Model")
            or ""
        )

        meta.lens_make = str(
            data.get("LensMake")
            or ""
        )

        meta.lens_model = str(
            data.get("LensModel")
            or ""
        )

        meta.iso = safe_float(
            data.get("ISOSpeedRatings")
        )

        meta.exposure_time = safe_float(
            data.get("ExposureTime")
        )

        meta.f_number = safe_float(
            data.get("FNumber")
        )

        meta.focal_length = safe_float(
            data.get("FocalLength")
        )

    return meta


# ============================================================
# Camera family
# ============================================================

def detect_camera_family(
    make,
    model,
):
    text = (
        f"{make} {model}"
    ).lower()

    if "canon" in text:
        return "canon"

    if "nikon" in text:
        return "nikon"

    if "sony" in text:
        return "sony"

    if (
        "fujifilm" in text
        or "fuji" in text
    ):
        return "fujifilm"

    if (
        "panasonic" in text
        or "lumix" in text
    ):
        return "panasonic"

    if (
        "olympus" in text
        or "om system" in text
    ):
        return "olympus"

    if "leica" in text:
        return "leica"

    if "pentax" in text:
        return "pentax"

    if "ricoh" in text:
        return "ricoh"

    if "sigma" in text:
        return "sigma"

    if "hasselblad" in text:
        return "hasselblad"

    return "unknown"


# ============================================================
# Camera profile
# ============================================================

def make_camera_profile(
    raw,
    metadata,
):

    try:
        black = np.asarray(
            raw.black_level_per_channel,
            dtype=np.float32,
        )
    except Exception:
        black = None

    try:
        white = float(
            raw.white_level
        )
    except Exception:
        white = None

    try:
        white_ch = np.asarray(
            raw.camera_white_level_per_channel,
            dtype=np.float32,
        )
    except Exception:
        white_ch = None

    try:
        camera_wb = np.asarray(
            raw.camera_whitebalance,
            dtype=np.float32,
        )
    except Exception:
        camera_wb = None

    try:
        color_matrix = np.asarray(
            raw.color_matrix,
            dtype=np.float32,
        )
    except Exception:
        color_matrix = None

    try:
        rgb_xyz = np.asarray(
            raw.rgb_xyz_matrix,
            dtype=np.float32,
        )
    except Exception:
        rgb_xyz = None

    try:
        pattern = np.asarray(
            raw.raw_pattern
        )
    except Exception:
        pattern = None

    try:
        color_desc = str(
            raw.color_desc
        )
    except Exception:
        color_desc = ""

    try:
        num_colors = int(
            raw.num_colors
        )
    except Exception:
        num_colors = 3

    try:
        width = int(
            raw.sizes.raw_width
        )

        height = int(
            raw.sizes.raw_height
        )

    except Exception:
        width = 0
        height = 0

    try:
        libraw_version = str(
            rawpy.libraw_version
        )
    except Exception:
        libraw_version = ""

    return CameraProfile(

        make=metadata.make,
        model=metadata.model,

        family=detect_camera_family(
            metadata.make,
            metadata.model,
        ),

        iso=metadata.iso,

        black_level=black,
        white_level=white,
        white_level_per_channel=white_ch,

        camera_wb=camera_wb,

        color_matrix=color_matrix,
        rgb_xyz_matrix=rgb_xyz,

        color_desc=color_desc,
        num_colors=num_colors,

        raw_width=width,
        raw_height=height,

        raw_pattern=pattern,

        lens_make=metadata.lens_make,
        lens_model=metadata.lens_model,

        exposure_time=metadata.exposure_time,
        f_number=metadata.f_number,
        focal_length=metadata.focal_length,

        white_balance=metadata.white_balance,
        color_temperature=metadata.color_temperature,
        color_space=metadata.color_space,

        metadata_source=metadata.source,
        libraw_version=libraw_version,
    )


# ============================================================
# RAW development
# ============================================================

def develop_raw_camera_rgb(path: Path):

    raw = rawpy.imread(
        str(path)
    )

    rgb = raw.postprocess(

        use_camera_wb=True,
        use_auto_wb=False,

        output_color=rawpy.ColorSpace.raw,
        output_bps=16,

        gamma=(1.0, 1.0),

        no_auto_bright=True,

        highlight_mode=(
            rawpy.HighlightMode.Blend
        ),

        half_size=False,

        four_color_rgb=False,

        demosaic_algorithm=(
            rawpy.DemosaicAlgorithm.AHD
        ),
    )

    return raw, rgb


# ============================================================
# Camera RGB -> sRGB
# ============================================================

def camera_rgb_to_srgb(
    camera_rgb,
    profile,
):

    img = camera_rgb.astype(
        np.float32
    )

    if img.ndim != 3:
        raise ValueError(
            "Invalid camera RGB image"
        )

    if img.shape[2] < 3:
        raise ValueError(
            "Camera RGB has less than "
            "three channels"
        )

    img = img[:, :, :3]

    if (
        profile.white_level is not None
        and profile.white_level > 0
    ):
        rgb = (
            img
            / float(profile.white_level)
        )
    else:
        rgb = (
            img
            / 65535.0
        )

    rgb = np.clip(
        rgb,
        0,
        1,
    )

    matrix = profile.rgb_xyz_matrix

    if matrix is None:
        raise RuntimeError(
            "rgb_xyz_matrix unavailable"
        )

    matrix = np.asarray(
        matrix,
        dtype=np.float32,
    )

    if (
        matrix.shape[0] < 3
        or matrix.shape[1] < 3
    ):
        raise RuntimeError(
            "Invalid RGB->XYZ matrix"
        )

    matrix = matrix[:3, :3]

    flat = rgb.reshape(
        -1,
        3,
    )

    xyz = (
        flat
        @ matrix.T
    )

    srgb_linear = (
        xyz
        @ XYZ_TO_SRGB.T
    )

    srgb_linear = np.maximum(
        srgb_linear,
        0,
    )

    srgb = np.where(
        srgb_linear <= 0.0031308,

        12.92 * srgb_linear,

        1.055
        * np.power(
            np.maximum(
                srgb_linear,
                0,
            ),
            1.0 / 2.4,
        )
        - 0.055,
    )

    srgb = srgb.reshape(
        rgb.shape
    )

    return np.clip(
        srgb,
        0,
        1,
    )


# ============================================================
# Image statistics
# ============================================================

def analyze_image(img):

    gray = cv2.cvtColor(
        (img * 255).astype(
            np.uint8
        ),
        cv2.COLOR_RGB2GRAY,
    ).astype(
        np.float32
    ) / 255.0

    pixels = img.reshape(
        -1,
        3,
    )

    mean = float(
        np.mean(gray)
    )

    median = float(
        np.median(gray)
    )

    p01 = float(
        np.percentile(gray, 1)
    )

    p05 = float(
        np.percentile(gray, 5)
    )

    p95 = float(
        np.percentile(gray, 95)
    )

    p99 = float(
        np.percentile(gray, 99)
    )

    shadow_ratio = float(
        np.mean(gray < 0.08)
    )

    highlight_ratio = float(
        np.mean(gray > 0.92)
    )

    dynamic_range = (
        p95 - p05
    )

    saturation_ratio = float(
        np.mean(
            np.max(
                pixels,
                axis=1,
            ) > 0.98
        )
    )

    edges = cv2.Canny(
        (gray * 255).astype(
            np.uint8
        ),
        50,
        120,
    )

    edge_density = float(
        np.mean(
            edges > 0
        )
    )

    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]

    warm_ratio = float(
        np.mean(
            (r > b * 1.12)
            & (r > g * 1.03)
        )
    )

    contrast = float(
        np.std(gray)
    )

    return ImageStats(

        mean=mean,
        median=median,

        p01=p01,
        p05=p05,
        p95=p95,
        p99=p99,

        shadow_ratio=shadow_ratio,
        highlight_ratio=highlight_ratio,

        dynamic_range=dynamic_range,
        saturation_ratio=saturation_ratio,

        edge_density=edge_density,
        warm_ratio=warm_ratio,

        contrast=contrast,
    )


# ============================================================
# Shooting conditions
# ============================================================

def analyze_shooting_condition(
    profile,
):

    result = ShootingCondition()

    if profile.iso:

        iso = profile.iso

        result.iso_factor = float(
            clamp01(
                math.log2(
                    max(iso, 100)
                    / 100.0
                ) / 6.0
            )
        )

        result.estimated_noise = (
            result.iso_factor
        )

        result.low_light = float(
            clamp01(
                math.log2(
                    max(iso, 100)
                    / 200.0
                ) / 5.0
            )
        )

    if profile.exposure_time:

        shutter = (
            profile.exposure_time
        )

        if shutter > 0:

            result.motion_risk = float(
                clamp01(
                    math.log2(
                        1.0 / shutter
                    ) / 8.0
                )
            )

    if profile.f_number:

        aperture = (
            profile.f_number
        )

        result.shallow_dof = float(
            clamp01(
                2.8
                / max(
                    aperture,
                    1.0,
                )
            )
        )

    if profile.focal_length:

        focal = (
            profile.focal_length
        )

        result.wide_angle = float(
            clamp01(
                (35.0 - focal)
                / 25.0
            )
        )

        result.telephoto = float(
            clamp01(
                (focal - 70.0)
                / 200.0
            )
        )

    return result


# ============================================================
# Saliency
# ============================================================

def compute_saliency(img):

    h, w = img.shape[:2]

    gray = cv2.cvtColor(
        (img * 255).astype(
            np.uint8
        ),
        cv2.COLOR_RGB2GRAY,
    ).astype(
        np.float32
    ) / 255.0

    blur = cv2.GaussianBlur(
        gray,
        (0, 0),
        15,
    )

    local_contrast = np.abs(
        gray - blur
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

    edge = cv2.magnitude(
        gx,
        gy,
    )

    edge = normalize_percentile(
        edge
    )

    colorfulness = (
        np.max(
            img,
            axis=2,
        )
        -
        np.min(
            img,
            axis=2,
        )
    )

    colorfulness = normalize_percentile(
        colorfulness
    )

    brightness = np.abs(
        gray
        - np.mean(gray)
    )

    brightness = normalize_percentile(
        brightness
    )

    yy, xx = np.mgrid[
        0:h,
        0:w,
    ]

    nx = (
        xx - w / 2
    ) / max(
        w / 2,
        1,
    )

    ny = (
        yy - h / 2
    ) / max(
        h / 2,
        1,
    )

    center = np.exp(
        -(
            nx * nx
            + ny * ny
        ) / 0.7
    )

    saliency = (

        0.30
        * normalize_percentile(
            local_contrast
        )

        + 0.25 * edge

        + 0.15 * colorfulness

        + 0.20 * brightness

        + 0.10 * center
    )

    return normalize_percentile(
        saliency
    )


# ============================================================
# Semantic segmentation
# ============================================================

class Segmenter:

    def __init__(
        self,
        device="cpu",
    ):

        self.device = device
        self.model = None

        if (
            torch is None
            or torchvision is None
        ):
            print(
                "[WARN] PyTorch/torchvision "
                "unavailable. "
                "Segmentation disabled."
            )

            return

        try:

            self.model = (
                torchvision.models.segmentation
                .deeplabv3_mobilenet_v3_large(
                    weights="DEFAULT"
                )
            )

            self.model.eval()

            self.model.to(
                device
            )

            print(
                "[INFO] Segmentation model: "
                "DeepLabV3 MobileNet V3 Large"
            )

            print(
                f"[INFO] Segmentation device: "
                f"{device}"
            )

        except Exception as e:

            print(
                "[WARN] Failed to load "
                f"segmentation model: {e}"
            )

            self.model = None

    def predict(
        self,
        img,
    ):

        if self.model is None:
            return None, None

        original_h, original_w = (
            img.shape[:2]
        )

        small = resize_for_analysis(
            img,
            max_size=768,
        )

        tensor = TF.to_tensor(
            small
        )

        tensor = (
            tensor
            .unsqueeze(0)
            .to(self.device)
        )

        with torch.no_grad():

            output = self.model(
                tensor
            )["out"]

        prediction = (
            output.argmax(1)[0]
            .cpu()
            .numpy()
        )

        prediction = cv2.resize(
            prediction.astype(
                np.uint8
            ),
            (
                original_w,
                original_h,
            ),
            interpolation=(
                cv2.INTER_NEAREST
            ),
        )

        confidence = (
            torch.softmax(
                output,
                dim=1,
            )
            .max(1)[0][0]
            .cpu()
            .numpy()
        )

        confidence = cv2.resize(
            confidence.astype(
                np.float32
            ),
            (
                original_w,
                original_h,
            ),
            interpolation=(
                cv2.INTER_LINEAR
            ),
        )

        return (
            prediction,
            confidence,
        )


# ============================================================
# Subject ranking
# ============================================================

def rank_subjects(
    img,
    segmentation,
    confidence,
    saliency,
):

    if segmentation is None:
        return []

    h, w = img.shape[:2]

    gray = cv2.cvtColor(
        (img * 255).astype(
            np.uint8
        ),
        cv2.COLOR_RGB2GRAY,
    )

    subjects = []

    for (
        class_name,
        class_weight,
    ) in SUBJECT_CLASSES.items():

        class_id = CLASS_ID.get(
            class_name
        )

        if class_id is None:
            continue

        mask = (
            segmentation
            == class_id
        )

        area = float(
            np.mean(mask)
        )

        if area < 0.002:
            continue

        if np.sum(mask) == 0:
            continue

        conf = float(
            np.mean(
                confidence[mask]
            )
        )

        ys, xs = np.where(
            mask
        )

        cx = np.mean(xs) / w
        cy = np.mean(ys) / h

        distance = math.sqrt(
            (cx - 0.5) ** 2
            + (cy - 0.5) ** 2
        )

        center_score = float(
            clamp01(
                1.0
                - distance / 0.707
            )
        )

        sal = float(
            np.mean(
                saliency[mask]
            )
        )

        region = img[mask]

        colorfulness = float(
            np.mean(
                np.max(
                    region,
                    axis=1,
                )
                -
                np.min(
                    region,
                    axis=1,
                )
            )
        )

        local_contrast = float(
            np.std(
                gray[mask]
            ) / 255.0
        )

        score = (
            class_weight
            * (
                0.35 * conf
                + 0.20 * center_score
                + 0.25 * sal
                + 0.10
                * clamp01(
                    local_contrast * 3
                )
                + 0.10
                * clamp01(
                    colorfulness * 3
                )
            )
        )

        subjects.append(
            SubjectCandidate(

                class_name=class_name,

                confidence=conf,

                area=area,

                center_score=center_score,

                saliency=sal,

                local_contrast=(
                    local_contrast
                ),

                colorfulness=(
                    colorfulness
                ),

                score=score,

                mask=mask,
            )
        )

    subjects.sort(
        key=lambda x: x.score,
        reverse=True,
    )

    return subjects


# ============================================================
# Scene classification
# ============================================================

def classify_scene(
    stats,
    profile,
    segmentation,
):

    if segmentation is not None:

        person_ratio = float(
            np.mean(
                segmentation
                == CLASS_ID["person"]
            )
        )

    else:

        person_ratio = 0.0

    slow_shutter = False

    if profile.exposure_time:

        slow_shutter = (
            profile.exposure_time
            > 1 / 20
        )

    if (
        stats.mean < 0.22
        and (
            stats.shadow_ratio > 0.25
            or slow_shutter
        )
    ):
        return "night"

    if person_ratio > 0.02:
        return "portrait"

    if (
        stats.warm_ratio > 0.22
        and stats.highlight_ratio > 0.015
    ):
        return "sunset"

    if stats.edge_density > 0.16:
        return "city"

    if stats.dynamic_range > 0.55:
        return "landscape"

    if stats.mean < 0.30:
        return "indoor"

    return "general"


# ============================================================
# Region masks
# ============================================================

def build_region_masks(
    img,
    segmentation,
    subjects,
):

    h, w = img.shape[:2]

    masks = {}

    if segmentation is None:

        masks["background"] = (
            np.ones(
                (h, w),
                dtype=np.float32,
            )
        )

        return masks

    # --------------------------------------------------------
    # Person
    # --------------------------------------------------------

    masks["person"] = (
        segmentation
        == CLASS_ID["person"]
    ).astype(
        np.float32
    )

    # --------------------------------------------------------
    # Vehicles
    # --------------------------------------------------------

    vehicle_ids = [
        CLASS_ID["car"],
        CLASS_ID["bus"],
        CLASS_ID["motorbike"],
        CLASS_ID["bicycle"],
        CLASS_ID["train"],
        CLASS_ID["boat"],
    ]

    masks["vehicle"] = np.isin(
        segmentation,
        vehicle_ids,
    ).astype(
        np.float32
    )

    # --------------------------------------------------------
    # Vegetation
    # --------------------------------------------------------

    masks["vegetation"] = (
        segmentation
        == CLASS_ID["pottedplant"]
    ).astype(
        np.float32
    )

    # --------------------------------------------------------
    # Sky estimation
    # --------------------------------------------------------

    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]

    yy = (
        np.arange(h)[:, None]
        / max(h - 1, 1)
    )

    upper_prior = np.clip(
        1.0
        - yy * 1.8,
        0,
        1,
    )

    blue_score = (
        b
        - 0.5 * r
        - 0.2 * g
    )

    sky = (
        (blue_score > 0.03)
        & (upper_prior > 0.15)
    ).astype(
        np.float32
    )

    sky *= upper_prior

    masks["sky"] = sky

    # --------------------------------------------------------
    # Water
    # --------------------------------------------------------

    water_color = (
        b
        - 0.55 * r
        - 0.15 * g
    )

    lower_prior = np.clip(
        (yy - 0.25) / 0.75,
        0,
        1,
    )

    water = (
        (water_color > 0.025)
        & (lower_prior > 0.15)
    ).astype(
        np.float32
    )

    water *= lower_prior

    masks["water"] = water

    # --------------------------------------------------------
    # Building
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        (img * 255).astype(
            np.uint8
        ),
        cv2.COLOR_RGB2GRAY,
    ).astype(
        np.float32
    ) / 255.0

    edge = cv2.Canny(
        (gray * 255).astype(
            np.uint8
        ),
        70,
        140,
    ).astype(
        np.float32
    ) / 255.0

    building = cv2.GaussianBlur(
        edge,
        (0, 0),
        5,
    )

    building = np.clip(
        building * 2.0,
        0,
        1,
    )

    building *= (
        1.0
        - masks["sky"]
    )

    masks["building"] = building

    # --------------------------------------------------------
    # Main subject
    # --------------------------------------------------------

    subject = np.zeros(
        (h, w),
        dtype=np.float32,
    )

    for s in subjects[:3]:

        if s.mask is None:
            continue

        strength = float(
            clamp01(
                s.score * 1.5
            )
        )

        subject = np.maximum(
            subject,
            s.mask.astype(
                np.float32
            ) * strength,
        )

    masks["subject"] = subject

    # --------------------------------------------------------
    # Background
    # --------------------------------------------------------

    occupied = np.zeros(
        (h, w),
        dtype=np.float32,
    )

    for key in (
        "person",
        "vehicle",
        "vegetation",
        "sky",
        "water",
        "subject",
    ):

        occupied = np.maximum(
            occupied,
            masks[key],
        )

    masks["background"] = np.clip(
        1.0 - occupied,
        0,
        1,
    )

    # --------------------------------------------------------
    # Smooth
    # --------------------------------------------------------

    for key in masks:

        masks[key] = np.clip(
            cv2.GaussianBlur(
                masks[key].astype(
                    np.float32
                ),
                (0, 0),
                3,
            ),
            0,
            1,
        )

    return masks


# ============================================================
# Region analysis
# ============================================================

def analyze_region(
    img,
    mask,
):

    valid = mask > 0.20

    coverage = float(
        np.mean(valid)
    )

    if np.sum(valid) < 20:

        return RegionStats(
            coverage=coverage,

            brightness=0.5,
            contrast=0.0,
            saturation=0.0,

            shadow_ratio=0.0,
            highlight_ratio=0.0,

            red_ratio=1 / 3,
            green_ratio=1 / 3,
            blue_ratio=1 / 3,

            warm_bias=0.0,
            cool_bias=0.0,

            edge_density=0.0,

            colorfulness=0.0,

            detail=0.0,
        )

    region = img[valid]

    r = region[:, 0]
    g = region[:, 1]
    b = region[:, 2]

    brightness = (
        0.2126 * r
        + 0.7152 * g
        + 0.0722 * b
    )

    saturation = (
        np.max(
            region,
            axis=1,
        )
        -
        np.min(
            region,
            axis=1,
        )
    )

    gray = cv2.cvtColor(
        (img * 255).astype(
            np.uint8
        ),
        cv2.COLOR_RGB2GRAY,
    ).astype(
        np.float32
    ) / 255.0

    local_gray = gray[valid]

    contrast = float(
        np.std(local_gray)
    )

    shadow_ratio = float(
        np.mean(
            local_gray < 0.08
        )
    )

    highlight_ratio = float(
        np.mean(
            local_gray > 0.92
        )
    )

    rgb_sum = (
        np.mean(r)
        + np.mean(g)
        + np.mean(b)
        + 1e-8
    )

    red_ratio = float(
        np.mean(r)
        / rgb_sum
    )

    green_ratio = float(
        np.mean(g)
        / rgb_sum
    )

    blue_ratio = float(
        np.mean(b)
        / rgb_sum
    )

    warm_bias = float(
        np.mean(
            (r - b)
            / (
                r
                + g
                + b
                + 1e-6
            )
        )
    )

    cool_bias = float(
        -warm_bias
    )

    edges = cv2.Canny(
        (gray * 255).astype(
            np.uint8
        ),
        50,
        120,
    )

    edge_density = float(
        np.mean(
            edges[valid] > 0
        )
    )

    colorfulness = float(
        np.mean(
            saturation
        )
    )

    # Local detail
    blur = cv2.GaussianBlur(
        gray,
        (0, 0),
        2.0,
    )

    detail_map = np.abs(
        gray - blur
    )

    detail = float(
        np.mean(
            detail_map[valid]
        )
    )

    return RegionStats(

        coverage=coverage,

        brightness=float(
            np.mean(brightness)
        ),

        contrast=contrast,

        saturation=float(
            np.mean(saturation)
        ),

        shadow_ratio=shadow_ratio,
        highlight_ratio=highlight_ratio,

        red_ratio=red_ratio,
        green_ratio=green_ratio,
        blue_ratio=blue_ratio,

        warm_bias=warm_bias,
        cool_bias=cool_bias,

        edge_density=edge_density,

        colorfulness=colorfulness,

        detail=detail,
    )


# ============================================================
# Automatic correction calculation
# ============================================================

def calculate_region_correction(
    region_name,
    stats,
    scene,
    shooting,
):

    # --------------------------------------------------------
    # Base target values
    #
    # These are not "development profiles".
    # They are analysis targets.
    # --------------------------------------------------------

    target_brightness = 0.46

    target_contrast = 0.20

    target_saturation = 0.18

    # --------------------------------------------------------
    # Brightness correction
    # --------------------------------------------------------

    brightness_error = (
        target_brightness
        - stats.brightness
    )

    exposure = float(
        clamp(
            brightness_error * 0.85,
            -0.18,
            0.18,
        )
    )

    # --------------------------------------------------------
    # Contrast correction
    # --------------------------------------------------------

    contrast_error = (
        target_contrast
        - stats.contrast
    )

    contrast_delta = float(
        clamp(
            contrast_error * 0.50,
            -0.08,
            0.08,
        )
    )

    contrast = (
        1.0
        + contrast_delta
    )

    # --------------------------------------------------------
    # Saturation correction
    # --------------------------------------------------------

    saturation_error = (
        target_saturation
        - stats.saturation
    )

    saturation_delta = float(
        clamp(
            saturation_error * 0.80,
            -0.08,
            0.08,
        )
    )

    saturation = (
        1.0
        + saturation_delta
    )

    # --------------------------------------------------------
    # Highlight protection
    # --------------------------------------------------------

    highlight_protection = float(
        clamp(
            (
                stats.highlight_ratio
                - 0.01
            ) * 2.5,
            0.0,
            0.35,
        )
    )

    # --------------------------------------------------------
    # Shadow lift
    # --------------------------------------------------------

    shadow_lift = float(
        clamp(
            (
                stats.shadow_ratio
                - 0.15
            ) * 0.30,
            0.0,
            0.10,
        )
    )

    # --------------------------------------------------------
    # Clarity
    # --------------------------------------------------------

    clarity = float(
        clamp(
            (
                0.15
                - stats.edge_density
            ) * 0.50,
            -0.03,
            0.06,
        )
    )

    hue_shift = 0.0

    reasons = []

    # ========================================================
    # Region-specific analysis
    # ========================================================

    if region_name == "sky":

        if stats.highlight_ratio > 0.03:

            highlight_protection = min(
                0.45,
                highlight_protection + 0.12,
            )

            reasons.append(
                "highlight compression"
            )

        if stats.brightness > 0.65:

            exposure -= 0.05

            reasons.append(
                "bright sky"
            )

        elif stats.brightness < 0.30:

            exposure += 0.04

            reasons.append(
                "dark sky"
            )

        # Blue-dominant sky should not become
        # unnaturally saturated.

        if stats.blue_ratio > 0.39:

            saturation *= 0.96

            reasons.append(
                "blue saturation control"
            )

        clarity = max(
            clarity,
            0.015,
        )

    elif region_name == "person":

        # Skin is generally a narrow region around
        # red/yellow. We use RGB relationships rather
        # than hard-coded skin color coordinates.

        if stats.red_ratio > 0.38:

            saturation *= 0.96

            hue_shift -= 0.01

            reasons.append(
                "warm skin suppression"
            )

        if stats.warm_bias > 0.08:

            hue_shift -= 0.01

            reasons.append(
                "warm bias correction"
            )

        contrast = min(
            contrast,
            1.03,
        )

        clarity = min(
            clarity,
            0.025,
        )

    elif region_name == "vegetation":

        if stats.green_ratio > 0.39:

            saturation *= 0.97

            reasons.append(
                "green saturation control"
            )

        if stats.green_ratio < 0.31:

            saturation *= 1.03

            reasons.append(
                "vegetation color recovery"
            )

        clarity = max(
            clarity,
            0.02,
        )

    elif region_name == "water":

        if stats.blue_ratio > 0.40:

            saturation *= 0.97

            reasons.append(
                "blue saturation control"
            )

        if stats.highlight_ratio > 0.025:

            highlight_protection += 0.06

            reasons.append(
                "water highlight protection"
            )

        clarity = max(
            clarity,
            0.015,
        )

    elif region_name == "building":

        # Buildings benefit from structure only
        # when the region actually contains detail.

        if stats.edge_density > 0.12:

            contrast += 0.025

            clarity += 0.02

            reasons.append(
                "structural detail"
            )

        if stats.brightness < 0.25:

            shadow_lift += 0.02

            reasons.append(
                "dark building"
            )

    elif region_name == "vehicle":

        if stats.highlight_ratio > 0.025:

            highlight_protection += 0.05

            reasons.append(
                "reflective surface"
            )

        if stats.edge_density > 0.08:

            clarity += 0.02

            reasons.append(
                "vehicle detail"
            )

    elif region_name == "background":

        # Background should be slightly less dominant
        # than the main subject.

        saturation *= 0.98

        clarity -= 0.01

        reasons.append(
            "background de-emphasis"
        )

    # ========================================================
    # Scene adjustments
    # ========================================================

    if scene == "night":

        if stats.highlight_ratio > 0.01:

            highlight_protection += 0.05

        if stats.brightness < 0.25:

            exposure += 0.02

    elif scene == "sunset":

        # Protect warm colors rather than pushing
        # saturation aggressively.

        if stats.warm_bias > 0.06:

            saturation *= 0.98

            reasons.append(
                "sunset color preservation"
            )

    elif scene == "portrait":

        if region_name == "person":

            exposure += 0.015

    elif scene == "landscape":

        if region_name in (
            "vegetation",
            "water",
            "sky",
        ):

            clarity += 0.01

    # ========================================================
    # Noise-aware correction
    # ========================================================

    if shooting.estimated_noise > 0.55:

        clarity *= (
            1.0
            - 0.45
            * shooting.estimated_noise
        )

    # ========================================================
    # Final limits
    # ========================================================

    exposure = float(
        clamp(
            exposure,
            -0.25,
            0.25,
        )
    )

    contrast = float(
        clamp(
            contrast,
            0.92,
            1.10,
        )
    )

    saturation = float(
        clamp(
            saturation,
            0.90,
            1.10,
        )
    )

    highlight_protection = float(
        clamp(
            highlight_protection,
            0.0,
            0.50,
        )
    )

    shadow_lift = float(
        clamp(
            shadow_lift,
            0.0,
            0.12,
        )
    )

    clarity = float(
        clamp(
            clarity,
            -0.04,
            0.08,
        )
    )

    hue_shift = float(
        clamp(
            hue_shift,
            -0.05,
            0.05,
        )
    )

    return RegionCorrection(

        exposure=exposure,

        contrast=contrast,

        saturation=saturation,

        highlight_protection=(
            highlight_protection
        ),

        shadow_lift=shadow_lift,

        clarity=clarity,

        hue_shift=hue_shift,

        reason="; ".join(
            reasons
        ),
    )


# ============================================================
# Global parameter search
# ============================================================

def apply_global_candidate(
    img,
    exposure,
    contrast,
    saturation,
):

    out = img.copy()

    out *= (
        2.0 ** exposure
    )

    out = np.clip(
        out,
        0,
        1,
    )

    out = (
        (out - 0.5)
        * contrast
        + 0.5
    )

    hsv = cv2.cvtColor(
        np.clip(
            out,
            0,
            1,
        ).astype(
            np.float32
        ),
        cv2.COLOR_RGB2HSV,
    )

    hsv[:, :, 1] *= saturation

    hsv[:, :, 1] = np.clip(
        hsv[:, :, 1],
        0,
        1,
    )

    out = cv2.cvtColor(
        hsv,
        cv2.COLOR_HSV2RGB,
    )

    return np.clip(
        out,
        0,
        1,
    )


def score_global_candidate(
    img,
):

    stats = analyze_image(
        img
    )

    target_mean = 0.46

    exposure_score = abs(
        stats.mean
        - target_mean
    )

    saturation_penalty = max(
        0.0,
        stats.saturation_ratio
        - 0.035,
    )

    shadow_penalty = max(
        0.0,
        stats.shadow_ratio
        - 0.35,
    )

    highlight_penalty = max(
        0.0,
        stats.highlight_ratio
        - 0.025,
    )

    contrast_score = abs(
        stats.contrast
        - 0.22
    )

    return (

        1.00
        * exposure_score

        + 1.30
        * saturation_penalty

        + 0.50
        * shadow_penalty

        + 1.20
        * highlight_penalty

        + 0.25
        * contrast_score
    )


def optimize_global(
    img,
):

    best_score = float(
        "inf"
    )

    best = None

    exposures = [
        -0.30,
        -0.20,
        -0.10,
        0.00,
        0.10,
        0.20,
        0.30,
    ]

    contrasts = [
        0.96,
        1.00,
        1.03,
        1.06,
        1.09,
    ]

    saturations = [
        0.96,
        0.99,
        1.00,
        1.03,
        1.06,
    ]

    for ev in exposures:

        for contrast in contrasts:

            for saturation in saturations:

                candidate = (
                    apply_global_candidate(
                        img,
                        ev,
                        contrast,
                        saturation,
                    )
                )

                score = (
                    score_global_candidate(
                        candidate
                    )
                )

                if score < best_score:

                    best_score = score

                    best = (
                        ev,
                        contrast,
                        saturation,
                        candidate,
                    )

    return best


# ============================================================
# Apply region correction
# ============================================================

def apply_region_correction(
    img,
    mask,
    correction,
):

    if np.max(mask) < 1e-5:
        return img

    out = img.copy()

    m = mask[:, :, None]

    # --------------------------------------------------------
    # Exposure
    # --------------------------------------------------------

    out *= (
        2.0
        ** (
            correction.exposure
            * m
        )
    )

    # --------------------------------------------------------
    # Contrast
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        np.clip(
            out,
            0,
            1,
        ).astype(
            np.float32
        ),
        cv2.COLOR_RGB2GRAY,
    )

    out = (
        gray[:, :, None]
        + (
            out
            - gray[:, :, None]
        )
        * (
            1.0
            + (
                correction.contrast
                - 1.0
            ) * m
        )
    )

    # --------------------------------------------------------
    # Saturation
    # --------------------------------------------------------

    hsv = cv2.cvtColor(
        np.clip(
            out,
            0,
            1,
        ).astype(
            np.float32
        ),
        cv2.COLOR_RGB2HSV,
    )

    hsv[:, :, 1] *= (
        1.0
        + (
            correction.saturation
            - 1.0
        )
        * mask
    )

    # --------------------------------------------------------
    # Hue shift
    #
    # OpenCV HSV hue range for float is 0..360.
    # correction is deliberately tiny.
    # --------------------------------------------------------

    hsv[:, :, 0] += (
        correction.hue_shift
        * 30.0
        * mask
    )

    hsv[:, :, 0] = (
        hsv[:, :, 0]
        % 360.0
    )

    hsv[:, :, 1] = np.clip(
        hsv[:, :, 1],
        0,
        1,
    )

    out = cv2.cvtColor(
        hsv,
        cv2.COLOR_HSV2RGB,
    )

    # --------------------------------------------------------
    # Luminance
    # --------------------------------------------------------

    luminance = cv2.cvtColor(
        np.clip(
            out,
            0,
            1,
        ).astype(
            np.float32
        ),
        cv2.COLOR_RGB2GRAY,
    )

    # --------------------------------------------------------
    # Highlight protection
    # --------------------------------------------------------

    highlight = np.clip(
        (
            luminance
            - 0.65
        ) / 0.35,
        0,
        1,
    )

    protection = (
        highlight
        * correction.highlight_protection
        * mask
    )

    out *= (
        1.0
        - protection[:, :, None]
    )

    # --------------------------------------------------------
    # Shadow lift
    # --------------------------------------------------------

    shadow = np.clip(
        (
            0.35
            - luminance
        ) / 0.35,
        0,
        1,
    )

    lift = (
        shadow
        * correction.shadow_lift
        * mask
    )

    out += lift[:, :, None]

    # --------------------------------------------------------
    # Clarity
    # --------------------------------------------------------

    if abs(
        correction.clarity
    ) > 1e-5:

        blur = cv2.GaussianBlur(
            luminance,
            (0, 0),
            7,
        )

        detail = (
            luminance
            - blur
        )

        out += (
            detail[:, :, None]
            * correction.clarity
            * mask[:, :, None]
        )

    return np.clip(
        out,
        0,
        1,
    )


# ============================================================
# Subject / background balance
# ============================================================

def apply_subject_background_balance(
    img,
    masks,
    subjects,
):

    if not subjects:
        return img

    subject_mask = masks.get(
        "subject"
    )

    background_mask = masks.get(
        "background"
    )

    if (
        subject_mask is None
        or background_mask is None
    ):
        return img

    subject_strength = float(
        clamp01(
            subjects[0].score
            * 1.5
        )
    )

    if subject_strength < 0.1:
        return img

    out = img.copy()

    # --------------------------------------------------------
    # Subject: slightly increase local detail
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        out.astype(
            np.float32
        ),
        cv2.COLOR_RGB2GRAY,
    )

    blur = cv2.GaussianBlur(
        gray,
        (0, 0),
        7,
    )

    detail = (
        gray
        - blur
    )

    out += (
        detail[:, :, None]
        * 0.06
        * subject_strength
        * subject_mask[:, :, None]
    )

    # --------------------------------------------------------
    # Background: very small reduction
    # --------------------------------------------------------

    hsv = cv2.cvtColor(
        np.clip(
            out,
            0,
            1,
        ).astype(
            np.float32
        ),
        cv2.COLOR_RGB2HSV,
    )

    hsv[:, :, 1] *= (
        1.0
        - 0.025
        * background_mask
        * subject_strength
    )

    hsv[:, :, 1] = np.clip(
        hsv[:, :, 1],
        0,
        1,
    )

    out = cv2.cvtColor(
        hsv,
        cv2.COLOR_HSV2RGB,
    )

    return np.clip(
        out,
        0,
        1,
    )


# ============================================================
# Tone mapping
# ============================================================

def luminance_tone_map(
    img,
    highlight_protection,
    shadow_lift,
):

    out = img.copy()

    y = cv2.cvtColor(
        out.astype(
            np.float32
        ),
        cv2.COLOR_RGB2GRAY,
    )

    shadow = np.clip(
        (
            0.35
            - y
        ) / 0.35,
        0,
        1,
    )

    y2 = (
        y
        + shadow
        * shadow_lift
    )

    highlight = np.clip(
        (
            y2
            - 0.65
        ) / 0.35,
        0,
        1,
    )

    compression = (
        highlight
        * highlight_protection
        * 0.20
    )

    y2 *= (
        1.0
        - compression
    )

    y2 = np.clip(
        y2,
        0,
        1,
    )

    # Smooth tone curve
    y2 = (
        y2
        * y2
        * (
            3.0
            - 2.0 * y2
        )
    )

    ratio = (
        y2
        / np.maximum(
            y,
            1e-4,
        )
    )

    out *= (
        ratio[:, :, None]
    )

    return np.clip(
        out,
        0,
        1,
    )


# ============================================================
# Denoise
# ============================================================

def denoise_image(
    img,
    strength,
):

    strength = float(
        clamp(
            strength,
            0,
            1,
        )
    )

    if strength <= 0.001:
        return img

    sigma_color = (
        0.015
        + 0.08 * strength
    )

    sigma_space = (
        3.0
        + 5.0 * strength
    )

    out = cv2.bilateralFilter(
        img.astype(
            np.float32
        ),
        d=0,
        sigmaColor=sigma_color,
        sigmaSpace=sigma_space,
    )

    return np.clip(
        out,
        0,
        1,
    )


# ============================================================
# Sharpen
# ============================================================

def sharpen_image(
    img,
    strength,
):

    strength = float(
        clamp(
            strength,
            0,
            1,
        )
    )

    if strength <= 0.001:
        return img

    blur = cv2.GaussianBlur(
        img,
        (0, 0),
        1.1,
    )

    detail = (
        img
        - blur
    )

    out = (
        img
        + detail
        * (
            0.35
            + 0.75 * strength
        )
    )

    return np.clip(
        out,
        0,
        1,
    )


# ============================================================
# JPEG
# ============================================================

def save_jpeg(
    img,
    path,
    quality=95,
):

    arr = np.clip(
        img * 255.0,
        0,
        255,
    ).astype(
        np.uint8
    )

    arr = cv2.cvtColor(
        arr,
        cv2.COLOR_RGB2BGR,
    )

    cv2.imwrite(
        str(path),
        arr,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            quality,
        ],
    )


# ============================================================
# Logging
# ============================================================

def print_camera_profile(
    profile,
):

    print(
        "[INFO] RAW camera: "
        f"{profile.make or 'UNKNOWN'} "
        f"{profile.model or 'UNKNOWN'}"
    )

    print(
        "[INFO] Camera family: "
        f"{profile.family}"
    )

    if profile.iso is not None:

        print(
            "[INFO] ISO: "
            f"{profile.iso:g}"
        )

    if profile.exposure_time is not None:

        print(
            "[INFO] Exposure: "
            f"{profile.exposure_time:g} sec"
        )

    if profile.f_number is not None:

        print(
            "[INFO] Aperture: "
            f"f/{profile.f_number:g}"
        )

    if profile.focal_length is not None:

        print(
            "[INFO] Focal length: "
            f"{profile.focal_length:g} mm"
        )

    if profile.lens_model:

        print(
            "[INFO] Lens: "
            f"{profile.lens_model}"
        )

    if profile.color_temperature is not None:

        print(
            "[INFO] Color temperature: "
            f"{profile.color_temperature:g} K"
        )

    print(
        "[INFO] Metadata source: "
        f"{profile.metadata_source}"
    )

    print(
        "[INFO] LibRaw: "
        f"{profile.libraw_version}"
    )

    print(
        "[INFO] RAW size: "
        f"{profile.raw_width} x "
        f"{profile.raw_height}"
    )

    print(
        "[INFO] Number of colors: "
        f"{profile.num_colors}"
    )

    if profile.white_level is not None:

        print(
            "[INFO] RAW white level: "
            f"{profile.white_level:g}"
        )

    if profile.camera_wb is not None:

        print(
            "[INFO] Camera WB: "
            + np.array2string(
                profile.camera_wb,
                precision=3,
            )
        )

    if profile.rgb_xyz_matrix is not None:

        print(
            "[INFO] RGB->XYZ matrix available"
        )


def print_image_stats(
    stats,
):

    print(
        "[INFO] Image statistics:"
    )

    print(
        f"       mean={stats.mean:.3f} "
        f"median={stats.median:.3f}"
    )

    print(
        f"       p01={stats.p01:.3f} "
        f"p05={stats.p05:.3f} "
        f"p95={stats.p95:.3f} "
        f"p99={stats.p99:.3f}"
    )

    print(
        f"       shadow={stats.shadow_ratio:.3f} "
        f"highlight={stats.highlight_ratio:.3f}"
    )

    print(
        f"       saturation={stats.saturation_ratio:.3f} "
        f"contrast={stats.contrast:.3f}"
    )


def print_subjects(
    subjects,
):

    if not subjects:

        print(
            "[INFO] Subjects: none"
        )

        return

    print(
        "[INFO] Subject candidates:"
    )

    for subject in subjects[:5]:

        print(
            f"       "
            f"{subject.class_name:12s} "
            f"score={subject.score:.3f} "
            f"conf={subject.confidence:.3f} "
            f"area={subject.area:.3f} "
            f"saliency={subject.saliency:.3f}"
        )


def print_region_analysis(
    name,
    stats,
    correction,
):

    print(
        f"[INFO] Region: {name}"
    )

    print(
        f"       coverage={stats.coverage:.3f} "
        f"brightness={stats.brightness:.3f} "
        f"contrast={stats.contrast:.3f}"
    )

    print(
        f"       saturation={stats.saturation:.3f} "
        f"shadow={stats.shadow_ratio:.3f} "
        f"highlight={stats.highlight_ratio:.3f}"
    )

    print(
        f"       RGB="
        f"{stats.red_ratio:.3f}/"
        f"{stats.green_ratio:.3f}/"
        f"{stats.blue_ratio:.3f} "
        f"warm={stats.warm_bias:.3f}"
    )

    print(
        f"       detail={stats.edge_density:.3f} "
        f"edge={stats.edge_density:.3f}"
    )

    print(
        f"       correction: "
        f"EV={correction.exposure:+.3f} "
        f"contrast={correction.contrast:.3f} "
        f"sat={correction.saturation:.3f} "
        f"HP={correction.highlight_protection:.3f} "
        f"shadow={correction.shadow_lift:.3f} "
        f"clarity={correction.clarity:+.3f}"
    )

    if correction.reason:

        print(
            f"       reason: "
            f"{correction.reason}"
        )


# ============================================================
# Main processing
# ============================================================

def process_raw(
    path: Path,
    output_path: Path,
    segmenter: Segmenter,
):

    print()
    print(
        "=" * 72
    )

    print(
        f"[INFO] Processing: {path}"
    )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata = extract_metadata(
        path
    )

    # --------------------------------------------------------
    # RAW
    # --------------------------------------------------------

    raw, camera_rgb = (
        develop_raw_camera_rgb(
            path
        )
    )

    try:

        profile = make_camera_profile(
            raw,
            metadata,
        )

        print_camera_profile(
            profile
        )

        img = camera_rgb_to_srgb(
            camera_rgb,
            profile,
        )

    finally:

        raw.close()

    # --------------------------------------------------------
    # Analysis image
    # --------------------------------------------------------

    analysis_img = (
        resize_for_analysis(
            img,
            max_size=1024,
        )
    )

    # --------------------------------------------------------
    # Global image analysis
    # --------------------------------------------------------

    stats = analyze_image(
        analysis_img
    )

    print_image_stats(
        stats
    )

    shooting = (
        analyze_shooting_condition(
            profile
        )
    )

    print(
        "[INFO] Shooting condition: "
        f"ISO-factor={shooting.iso_factor:.2f}, "
        f"low-light={shooting.low_light:.2f}, "
        f"motion-risk={shooting.motion_risk:.2f}, "
        f"DOF={shooting.shallow_dof:.2f}, "
        f"noise={shooting.estimated_noise:.2f}"
    )

    # --------------------------------------------------------
    # Saliency
    # --------------------------------------------------------

    saliency = compute_saliency(
        analysis_img
    )

    # --------------------------------------------------------
    # Segmentation
    # --------------------------------------------------------

    segmentation, confidence = (
        segmenter.predict(
            analysis_img
        )
    )

    # --------------------------------------------------------
    # Subjects
    # --------------------------------------------------------

    subjects = rank_subjects(
        analysis_img,
        segmentation,
        confidence,
        saliency,
    )

    print_subjects(
        subjects
    )

    # --------------------------------------------------------
    # Scene
    # --------------------------------------------------------

    scene = classify_scene(
        stats,
        profile,
        segmentation,
    )

    print(
        f"[INFO] Scene: {scene}"
    )

    # --------------------------------------------------------
    # Global optimization
    # --------------------------------------------------------

    print(
        "[INFO] Global optimization..."
    )

    (
        global_ev,
        global_contrast,
        global_saturation,
        out,
    ) = optimize_global(
        analysis_img
    )

    print(
        "[INFO] Global parameters: "
        f"EV={global_ev:+.3f}, "
        f"contrast={global_contrast:.3f}, "
        f"saturation={global_saturation:.3f}"
    )

    # --------------------------------------------------------
    # Region masks
    # --------------------------------------------------------

    masks = build_region_masks(
        analysis_img,
        segmentation,
        subjects,
    )

    # --------------------------------------------------------
    # Region analysis and correction
    # --------------------------------------------------------

    region_order = [
        "sky",
        "person",
        "vegetation",
        "water",
        "building",
        "vehicle",
        "background",
    ]

    corrections = {}

    for region_name in region_order:

        mask = masks.get(
            region_name
        )

        if mask is None:
            continue

        coverage = float(
            np.mean(
                mask > 0.20
            )
        )

        if coverage < 0.002:
            continue

        region_stats = analyze_region(
            analysis_img,
            mask,
        )

        correction = (
            calculate_region_correction(
                region_name,
                region_stats,
                scene,
                shooting,
            )
        )

        corrections[
            region_name
        ] = correction

        print_region_analysis(
            region_name,
            region_stats,
            correction,
        )

        out = apply_region_correction(
            out,
            mask,
            correction,
        )

    # --------------------------------------------------------
    # Subject / background balance
    # --------------------------------------------------------

    out = (
        apply_subject_background_balance(
            out,
            masks,
            subjects,
        )
    )

    # --------------------------------------------------------
    # Global tone mapping
    # --------------------------------------------------------

    # Derive global highlight protection
    # from actual image state.

    global_highlight_protection = float(
        clamp(
            stats.highlight_ratio
            * 2.0,
            0,
            0.35,
        )
    )

    global_shadow_lift = float(
        clamp(
            (
                stats.shadow_ratio
                - 0.15
            ) * 0.15,
            0,
            0.08,
        )
    )

    out = luminance_tone_map(
        out,
        global_highlight_protection,
        global_shadow_lift,
    )

    # --------------------------------------------------------
    # Denoise
    # --------------------------------------------------------

    # Estimate actual noise requirement
    # from ISO and local detail.

    base_denoise = (
        0.20
        + 0.65
        * shooting.estimated_noise
    )

    if stats.edge_density < 0.025:

        base_denoise *= 0.85

    out = denoise_image(
        out,
        float(
            clamp(
                base_denoise,
                0,
                1,
            )
        ),
    )

    # --------------------------------------------------------
    # Sharpen
    # --------------------------------------------------------

    sharpen_strength = (
        0.75
        * (
            1.0
            - 0.50
            * shooting.estimated_noise
        )
    )

    # High detail images can tolerate
    # slightly more sharpening.

    sharpen_strength += (
        clamp01(
            stats.edge_density * 5
        )
        * 0.15
    )

    sharpen_strength = float(
        clamp(
            sharpen_strength,
            0,
            1,
        )
    )

    out = sharpen_image(
        out,
        sharpen_strength,
    )

    print(
        "[INFO] Final processing: "
        f"denoise={base_denoise:.3f}, "
        f"sharpen={sharpen_strength:.3f}"
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_jpeg(
        out,
        output_path,
        quality=95,
    )

    print(
        f"[INFO] Output: {output_path}"
    )


# ============================================================
# RAW collection
# ============================================================

def collect_raw_files(
    directory,
):

    files = []

    for path in directory.rglob(
        "*"
    ):

        if (
            path.is_file()
            and path.suffix.lower()
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
            "Automatic RAW development v16"
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
        default=Path(
            "./output"
        ),
        help="Output directory",
    )

    parser.add_argument(
        "--device",
        choices=[
            "cpu",
            "cuda",
        ],
        default="cpu",
        help="Segmentation device",
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Input validation
    # --------------------------------------------------------

    if not args.input.exists():

        print(
            "[ERROR] Input not found: "
            f"{args.input}"
        )

        sys.exit(1)

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = args.device

    if device == "cuda":

        if torch is None:

            print(
                "[ERROR] PyTorch is not installed."
            )

            sys.exit(1)

        if not torch.cuda.is_available():

            print(
                "[WARN] CUDA requested "
                "but unavailable. "
                "Falling back to CPU."
            )

            device = "cpu"

    print(
        f"[INFO] Device: {device}"
    )

    # --------------------------------------------------------
    # Load segmentation model once
    # --------------------------------------------------------

    segmenter = Segmenter(
        device=device
    )

    # --------------------------------------------------------
    # RAW files
    # --------------------------------------------------------

    if args.input.is_file():

        raw_files = [
            args.input
        ]

    else:

        raw_files = (
            collect_raw_files(
                args.input
            )
        )

    if not raw_files:

        print(
            "[ERROR] No RAW files found."
        )

        sys.exit(1)

    print(
        "[INFO] RAW files: "
        f"{len(raw_files)}"
    )

    # --------------------------------------------------------
    # Process
    # --------------------------------------------------------

    for raw_path in raw_files:

        output_path = (
            args.output
            / f"{raw_path.stem}.jpg"
        )

        try:

            process_raw(
                raw_path,
                output_path,
                segmenter,
            )

        except Exception as e:

            print(
                f"[ERROR] Failed: "
                f"{raw_path}"
            )

            print(
                f"        "
                f"{type(e).__name__}: "
                f"{e}"
            )

    print()
    print(
        "=" * 72
    )

    print(
        "[INFO] All processing finished."
    )


if __name__ == "__main__":
    main()