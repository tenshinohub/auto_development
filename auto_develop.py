#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
auto_develop_v15.py

Automatic RAW development v15

Pipeline:

RAW
 ↓
LibRaw / rawpy
 ↓
Camera RGB
 ↓
Camera RGB -> XYZ -> linear sRGB
 ↓
Image analysis
 ↓
Semantic segmentation
 ↓
Saliency
 ↓
Subject ranking
 ↓
Scene classification
 ↓
Global parameter optimization
 ↓
Region-aware development
   ├─ sky
   ├─ person
   ├─ animal
   ├─ vegetation
   ├─ building
   ├─ vehicle
   ├─ water
   └─ background
 ↓
Luminance tone mapping
 ↓
Denoise
 ↓
Sharpen
 ↓
JPEG

Usage:

    python3 auto_develop_v15.py ./RAW -o ./output --device cuda

Requirements:

    pip install rawpy numpy opencv-python pillow torch torchvision scipy

Optional:

    sudo apt install exiftool
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import rawpy

try:
    from scipy.ndimage import gaussian_filter
except ImportError:
    gaussian_filter = None

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

CLASS_ID = {name: i for i, name in enumerate(VOC_CLASSES)}


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


REGION_CLASSES = {
    "sky": [
        "sky",
    ],
    "person": [
        "person",
    ],
    "vegetation": [
        "pottedplant",
    ],
    "vehicle": [
        "car",
        "bus",
        "motorbike",
        "bicycle",
        "train",
        "boat",
    ],
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
class SceneProfile:
    exposure: float
    contrast: float
    saturation: float

    highlight_protection: float
    shadow_lift: float

    subject_strength: float
    background_suppression: float

    denoise: float
    sharpen: float


@dataclass
class RegionProfile:
    exposure: float = 0.0
    contrast: float = 1.0
    saturation: float = 1.0

    highlight_protection: float = 0.0
    shadow_lift: float = 0.0

    clarity: float = 0.0
    denoise: float = 0.0


# ============================================================
# Utility
# ============================================================

def clamp01(x):
    return np.clip(x, 0.0, 1.0)


def safe_float(v):
    try:
        return float(v)
    except Exception:
        return None


def normalize_percentile(x, p_low=0.5, p_high=99.5):
    lo = np.percentile(x, p_low)
    hi = np.percentile(x, p_high)

    if hi <= lo + 1e-8:
        return np.clip(x, 0.0, 1.0)

    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def resize_for_analysis(img, max_size=768):
    h, w = img.shape[:2]

    scale = min(1.0, max_size / max(h, w))

    if scale == 1.0:
        return img

    return cv2.resize(
        img,
        (int(w * scale), int(h * scale)),
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

        data = json.loads(result.stdout)

        if not data:
            return None

        return data[0]

    except Exception:
        return None


def read_pillow_exif(path: Path):
    if Image is None:
        return {}

    try:
        img = Image.open(path)
        exif = img.getexif()

        if not exif:
            return {}

        result = {}

        for tag_id, value in exif.items():
            name = ExifTags.TAGS.get(tag_id, tag_id)
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

        meta.lens_make = str(data.get("LensMake") or "")
        meta.lens_model = str(data.get("LensModel") or "")

        meta.iso = safe_float(data.get("ISO"))
        meta.exposure_time = safe_float(data.get("ExposureTime"))
        meta.f_number = safe_float(data.get("FNumber"))
        meta.focal_length = safe_float(data.get("FocalLength"))

        meta.white_balance = str(data.get("WhiteBalance") or "")
        meta.color_temperature = safe_float(
            data.get("ColorTemperature")
        )
        meta.color_space = str(data.get("ColorSpace") or "")

        return meta

    # Pillow fallback
    data = read_pillow_exif(path)

    if data:
        meta.source = "pillow"

        meta.make = str(data.get("Make") or "")
        meta.model = str(data.get("Model") or "")
        meta.lens_make = str(data.get("LensMake") or "")
        meta.lens_model = str(data.get("LensModel") or "")

        meta.iso = safe_float(data.get("ISOSpeedRatings"))
        meta.exposure_time = safe_float(data.get("ExposureTime"))
        meta.f_number = safe_float(data.get("FNumber"))
        meta.focal_length = safe_float(data.get("FocalLength"))

    return meta


# ============================================================
# Camera family
# ============================================================

def detect_camera_family(make, model):
    text = f"{make} {model}".lower()

    if "canon" in text:
        return "canon"

    if "nikon" in text:
        return "nikon"

    if "sony" in text:
        return "sony"

    if "fujifilm" in text or "fuji" in text:
        return "fujifilm"

    if "panasonic" in text or "lumix" in text:
        return "panasonic"

    if "olympus" in text or "om system" in text:
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

def make_camera_profile(raw, meta):
    try:
        black = np.asarray(raw.black_level_per_channel, dtype=np.float32)
    except Exception:
        black = None

    try:
        white = float(raw.white_level)
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
        raw_pattern = np.asarray(
            raw.raw_pattern,
        )
    except Exception:
        raw_pattern = None

    try:
        desc = str(raw.color_desc)
    except Exception:
        desc = ""

    try:
        num_colors = int(raw.num_colors)
    except Exception:
        num_colors = 3

    try:
        width = int(raw.sizes.raw_width)
        height = int(raw.sizes.raw_height)
    except Exception:
        width = 0
        height = 0

    try:
        libraw_version = str(rawpy.libraw_version)
    except Exception:
        libraw_version = ""

    family = detect_camera_family(meta.make, meta.model)

    return CameraProfile(
        make=meta.make,
        model=meta.model,
        family=family,

        iso=meta.iso,

        black_level=black,
        white_level=white,
        white_level_per_channel=white_ch,

        camera_wb=camera_wb,

        color_matrix=color_matrix,
        rgb_xyz_matrix=rgb_xyz,

        color_desc=desc,
        num_colors=num_colors,

        raw_width=width,
        raw_height=height,
        raw_pattern=raw_pattern,

        lens_make=meta.lens_make,
        lens_model=meta.lens_model,

        exposure_time=meta.exposure_time,
        f_number=meta.f_number,
        focal_length=meta.focal_length,

        white_balance=meta.white_balance,
        color_temperature=meta.color_temperature,
        color_space=meta.color_space,

        metadata_source=meta.source,
        libraw_version=libraw_version,
    )


# ============================================================
# RAW -> Camera RGB
# ============================================================

def develop_raw_camera_rgb(path: Path):
    raw = rawpy.imread(str(path))

    rgb = raw.postprocess(
        use_camera_wb=True,
        use_auto_wb=False,

        output_color=rawpy.ColorSpace.raw,
        output_bps=16,

        gamma=(1.0, 1.0),

        no_auto_bright=True,

        highlight_mode=rawpy.HighlightMode.Blend,

        half_size=False,
        four_color_rgb=False,

        demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD,
    )

    return raw, rgb


# ============================================================
# Camera RGB -> sRGB
# ============================================================

def camera_rgb_to_srgb(camera_rgb, profile):
    img = camera_rgb.astype(np.float32)

    if img.ndim != 3:
        raise ValueError("Invalid camera RGB image")

    if img.shape[2] < 3:
        raise ValueError("Camera RGB has less than 3 channels")

    img = img[:, :, :3]

    # Normalize based on actual RAW white level when possible.
    white = profile.white_level

    if white is not None and white > 0:
        rgb = img / float(white)
    else:
        rgb = img / 65535.0

    rgb = np.clip(rgb, 0.0, 1.0)

    matrix = profile.rgb_xyz_matrix

    if matrix is None:
        raise RuntimeError(
            "rgb_xyz_matrix is unavailable"
        )

    matrix = np.asarray(matrix, dtype=np.float32)

    if matrix.shape[0] < 3 or matrix.shape[1] < 3:
        raise RuntimeError(
            f"Invalid rgb_xyz_matrix shape: {matrix.shape}"
        )

    matrix = matrix[:3, :3]

    flat = rgb.reshape(-1, 3)

    xyz = flat @ matrix.T

    srgb_linear = xyz @ XYZ_TO_SRGB.T

    srgb_linear = np.maximum(srgb_linear, 0.0)

    # Linear sRGB -> gamma encoded sRGB
    srgb = np.where(
        srgb_linear <= 0.0031308,
        12.92 * srgb_linear,
        1.055 * np.power(
            np.maximum(srgb_linear, 0.0),
            1.0 / 2.4,
        ) - 0.055,
    )

    srgb = srgb.reshape(rgb.shape)

    return np.clip(srgb, 0.0, 1.0)


# ============================================================
# Image analysis
# ============================================================

def analyze_image(img):
    gray = cv2.cvtColor(
        (img * 255).astype(np.uint8),
        cv2.COLOR_RGB2GRAY,
    ).astype(np.float32) / 255.0

    pixels = img.reshape(-1, 3)

    mean = float(np.mean(gray))
    median = float(np.median(gray))

    p01 = float(np.percentile(gray, 1))
    p05 = float(np.percentile(gray, 5))
    p95 = float(np.percentile(gray, 95))
    p99 = float(np.percentile(gray, 99))

    shadow_ratio = float(np.mean(gray < 0.08))
    highlight_ratio = float(np.mean(gray > 0.92))

    dynamic_range = p95 - p05

    saturation_ratio = float(
        np.mean(np.max(pixels, axis=1) > 0.98)
    )

    edges = cv2.Canny(
        (gray * 255).astype(np.uint8),
        50,
        120,
    )

    edge_density = float(np.mean(edges > 0))

    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]

    warm_ratio = float(
        np.mean((r > b * 1.12) & (r > g * 1.03))
    )

    contrast = float(np.std(gray))

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
# Shooting condition
# ============================================================

def analyze_shooting_condition(profile):
    result = ShootingCondition()

    if profile.iso:
        iso = profile.iso

        result.iso_factor = clamp01(
            (math.log2(max(iso, 100) / 100.0)) / 6.0
        )

        result.estimated_noise = result.iso_factor

        result.low_light = clamp01(
            (math.log2(max(iso, 100) / 200.0)) / 5.0
        )

    if profile.exposure_time:
        shutter = profile.exposure_time

        if shutter > 0:
            result.motion_risk = clamp01(
                math.log2(1.0 / shutter) / 8.0
            )

    if profile.f_number:
        aperture = profile.f_number

        result.shallow_dof = clamp01(
            (2.8 / max(aperture, 1.0))
        )

    if profile.focal_length:
        f = profile.focal_length

        result.wide_angle = clamp01(
            (35.0 - f) / 25.0
        )

        result.telephoto = clamp01(
            (f - 70.0) / 200.0
        )

    return result


# ============================================================
# Saliency
# ============================================================

def compute_saliency(img):
    h, w = img.shape[:2]

    gray = cv2.cvtColor(
        (img * 255).astype(np.uint8),
        cv2.COLOR_RGB2GRAY,
    ).astype(np.float32) / 255.0

    # Local luminance contrast
    blur = cv2.GaussianBlur(gray, (0, 0), 15)

    local_contrast = np.abs(gray - blur)

    # Edge
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    edge = cv2.magnitude(gx, gy)
    edge = normalize_percentile(edge)

    # Colorfulness
    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]

    colorfulness = np.max(img, axis=2) - np.min(img, axis=2)
    colorfulness = normalize_percentile(colorfulness)

    # Brightness distinctiveness
    brightness = np.abs(gray - np.mean(gray))
    brightness = normalize_percentile(brightness)

    # Center prior
    yy, xx = np.mgrid[0:h, 0:w]

    nx = (xx - w / 2) / (w / 2)
    ny = (yy - h / 2) / (h / 2)

    center = np.exp(
        -(nx * nx + ny * ny) / 0.7
    )

    saliency = (
        0.30 * normalize_percentile(local_contrast)
        + 0.25 * edge
        + 0.15 * colorfulness
        + 0.20 * brightness
        + 0.10 * center
    )

    return normalize_percentile(saliency)


# ============================================================
# Semantic segmentation
# ============================================================

class Segmenter:

    def __init__(self, device="cpu"):
        self.device = device
        self.model = None

        if torch is None or torchvision is None:
            print(
                "[WARN] PyTorch/torchvision unavailable. "
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
            self.model.to(device)

            print(
                f"[INFO] Semantic segmentation: "
                f"DeepLabV3 MobileNet V3 Large ({device})"
            )

        except Exception as e:
            print(
                f"[WARN] Failed to load segmentation model: {e}"
            )

            self.model = None

    def predict(self, img):
        if self.model is None:
            return None, None

        original_h, original_w = img.shape[:2]

        small = resize_for_analysis(
            img,
            max_size=768,
        )

        tensor = TF.to_tensor(small)

        tensor = tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(tensor)["out"]

        prediction = output.argmax(1)[0]

        prediction = prediction.cpu().numpy()

        prediction = cv2.resize(
            prediction.astype(np.uint8),
            (original_w, original_h),
            interpolation=cv2.INTER_NEAREST,
        )

        confidence = torch.softmax(
            output,
            dim=1,
        ).max(1)[0][0]

        confidence = confidence.cpu().numpy()

        confidence = cv2.resize(
            confidence.astype(np.float32),
            (original_w, original_h),
            interpolation=cv2.INTER_LINEAR,
        )

        return prediction, confidence


# ============================================================
# Subject candidates
# ============================================================

def rank_subjects(img, segmentation, confidence, saliency):
    if segmentation is None:
        return []

    h, w = img.shape[:2]

    subjects = []

    for class_name, class_weight in SUBJECT_CLASSES.items():

        class_id = CLASS_ID.get(class_name)

        if class_id is None:
            continue

        mask = segmentation == class_id

        area = float(np.mean(mask))

        if area < 0.002:
            continue

        conf = float(np.mean(confidence[mask]))

        if np.sum(mask) == 0:
            continue

        ys, xs = np.where(mask)

        cx = np.mean(xs) / w
        cy = np.mean(ys) / h

        center_distance = math.sqrt(
            (cx - 0.5) ** 2
            + (cy - 0.5) ** 2
        )

        center_score = clamp01(
            1.0 - center_distance / 0.707
        )

        sal = float(np.mean(saliency[mask]))

        region = img[mask]

        colorfulness = float(
            np.mean(
                np.max(region, axis=1)
                - np.min(region, axis=1)
            )
        )

        local_contrast = float(
            np.std(
                cv2.cvtColor(
                    (img * 255).astype(np.uint8),
                    cv2.COLOR_RGB2GRAY,
                )[mask]
            ) / 255.0
        )

        score = (
            class_weight
            * (
                0.35 * conf
                + 0.20 * center_score
                + 0.25 * sal
                + 0.10 * clamp01(local_contrast * 3)
                + 0.10 * clamp01(colorfulness * 3)
            )
        )

        subjects.append(
            SubjectCandidate(
                class_name=class_name,
                confidence=conf,
                area=area,
                center_score=center_score,
                saliency=sal,
                local_contrast=local_contrast,
                colorfulness=colorfulness,
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

def classify_scene(stats, profile, segmentation):
    h, w = segmentation.shape if segmentation is not None else (1, 1)

    if segmentation is not None:
        person_id = CLASS_ID["person"]

        person_ratio = float(
            np.mean(segmentation == person_id)
        )
    else:
        person_ratio = 0.0

    if profile.exposure_time:
        slow_shutter = profile.exposure_time > 1 / 20
    else:
        slow_shutter = False

    if stats.mean < 0.22 and (
        stats.shadow_ratio > 0.25
        or slow_shutter
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
# Scene profiles
# ============================================================

SCENE_PROFILES = {
    "portrait": SceneProfile(
        exposure=0.05,
        contrast=1.02,
        saturation=0.97,
        highlight_protection=0.40,
        shadow_lift=0.08,
        subject_strength=0.08,
        background_suppression=0.035,
        denoise=0.55,
        sharpen=0.75,
    ),

    "night": SceneProfile(
        exposure=0.00,
        contrast=1.05,
        saturation=1.03,
        highlight_protection=0.55,
        shadow_lift=0.02,
        subject_strength=0.05,
        background_suppression=0.015,
        denoise=0.85,
        sharpen=0.45,
    ),

    "sunset": SceneProfile(
        exposure=-0.05,
        contrast=1.06,
        saturation=1.08,
        highlight_protection=0.55,
        shadow_lift=0.04,
        subject_strength=0.05,
        background_suppression=0.015,
        denoise=0.30,
        sharpen=0.80,
    ),

    "landscape": SceneProfile(
        exposure=0.03,
        contrast=1.08,
        saturation=1.04,
        highlight_protection=0.40,
        shadow_lift=0.08,
        subject_strength=0.06,
        background_suppression=0.015,
        denoise=0.30,
        sharpen=0.85,
    ),

    "city": SceneProfile(
        exposure=0.02,
        contrast=1.07,
        saturation=1.02,
        highlight_protection=0.45,
        shadow_lift=0.05,
        subject_strength=0.06,
        background_suppression=0.020,
        denoise=0.40,
        sharpen=0.80,
    ),

    "indoor": SceneProfile(
        exposure=0.04,
        contrast=1.03,
        saturation=0.99,
        highlight_protection=0.40,
        shadow_lift=0.08,
        subject_strength=0.05,
        background_suppression=0.015,
        denoise=0.50,
        sharpen=0.65,
    ),

    "general": SceneProfile(
        exposure=0.00,
        contrast=1.04,
        saturation=1.00,
        highlight_protection=0.35,
        shadow_lift=0.06,
        subject_strength=0.04,
        background_suppression=0.015,
        denoise=0.30,
        sharpen=0.75,
    ),
}


# ============================================================
# Automatic global optimization
# ============================================================

def score_global_candidate(img):
    stats = analyze_image(img)

    target_mean = 0.46

    exposure_score = abs(
        stats.mean - target_mean
    )

    saturation_penalty = max(
        0.0,
        stats.saturation_ratio - 0.035,
    )

    shadow_penalty = max(
        0.0,
        stats.shadow_ratio - 0.35,
    )

    highlight_penalty = max(
        0.0,
        stats.highlight_ratio - 0.025,
    )

    contrast_score = abs(
        stats.contrast - 0.22
    )

    score = (
        1.00 * exposure_score
        + 1.30 * saturation_penalty
        + 0.50 * shadow_penalty
        + 1.20 * highlight_penalty
        + 0.25 * contrast_score
    )

    return score


def apply_global_candidate(
    img,
    exposure,
    contrast,
    saturation,
):
    out = img.copy()

    out *= 2.0 ** exposure

    out = np.clip(out, 0.0, 1.0)

    gray = cv2.cvtColor(
        (out * 255).astype(np.uint8),
        cv2.COLOR_RGB2GRAY,
    ).astype(np.float32) / 255.0

    out = (
        (out - 0.5) * contrast
        + 0.5
    )

    hsv = cv2.cvtColor(
        (np.clip(out, 0, 1) * 255).astype(np.uint8),
        cv2.COLOR_RGB2HSV,
    ).astype(np.float32)

    hsv[:, :, 1] *= saturation

    hsv[:, :, 1] = np.clip(
        hsv[:, :, 1],
        0,
        255,
    )

    out = cv2.cvtColor(
        hsv.astype(np.uint8),
        cv2.COLOR_HSV2RGB,
    ).astype(np.float32) / 255.0

    return np.clip(out, 0.0, 1.0)


def optimize_global(img, scene_profile):
    best_score = float("inf")
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

                ev2 = ev + scene_profile.exposure
                c2 = contrast * scene_profile.contrast
                s2 = saturation * scene_profile.saturation

                candidate = apply_global_candidate(
                    img,
                    ev2,
                    c2,
                    s2,
                )

                score = score_global_candidate(
                    candidate
                )

                if score < best_score:
                    best_score = score

                    best = (
                        ev2,
                        c2,
                        s2,
                        candidate,
                    )

    return best


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
        masks["background"] = np.ones(
            (h, w),
            dtype=np.float32,
        )

        return masks

    # Person
    person_mask = (
        segmentation == CLASS_ID["person"]
    ).astype(np.float32)

    masks["person"] = person_mask

    # Vehicles
    vehicle_ids = [
        CLASS_ID[x]
        for x in (
            "car",
            "bus",
            "motorbike",
            "bicycle",
            "train",
            "boat",
        )
    ]

    vehicle_mask = np.isin(
        segmentation,
        vehicle_ids,
    ).astype(np.float32)

    masks["vehicle"] = vehicle_mask

    # Vegetation
    vegetation_mask = (
        segmentation
        == CLASS_ID["pottedplant"]
    ).astype(np.float32)

    masks["vegetation"] = vegetation_mask

    # Sky estimation
    #
    # VOC DeepLab does not have a sky class.
    # Estimate upper image blue/cyan regions.
    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]

    yy = np.arange(h)[:, None] / max(h - 1, 1)

    upper_prior = np.clip(
        1.0 - yy * 1.8,
        0.0,
        1.0,
    )

    blue_score = (
        b - 0.5 * r - 0.2 * g
    )

    sky_mask = (
        (blue_score > 0.03)
        & (upper_prior > 0.15)
    ).astype(np.float32)

    sky_mask *= upper_prior

    masks["sky"] = sky_mask

    # Water estimation
    #
    # Blue/cyan regions in lower/middle part.
    water_color = (
        b - 0.55 * r
        - 0.15 * g
    )

    lower_prior = np.clip(
        (yy - 0.25) / 0.75,
        0.0,
        1.0,
    )

    water_mask = (
        (water_color > 0.025)
        & (lower_prior > 0.15)
    ).astype(np.float32)

    masks["water"] = water_mask * lower_prior

    # Building estimation:
    #
    # Semantic classes do not contain "building".
    # Use non-natural high-edge regions.
    gray = cv2.cvtColor(
        (img * 255).astype(np.uint8),
        cv2.COLOR_RGB2GRAY,
    ).astype(np.float32) / 255.0

    edge = cv2.Canny(
        (gray * 255).astype(np.uint8),
        70,
        140,
    ).astype(np.float32) / 255.0

    building_mask = cv2.GaussianBlur(
        edge,
        (0, 0),
        5,
    )

    building_mask = np.clip(
        building_mask * 2.0,
        0,
        1,
    )

    building_mask *= (
        1.0
        - masks["sky"]
    )

    masks["building"] = building_mask

    # Subject
    subject_mask = np.zeros(
        (h, w),
        dtype=np.float32,
    )

    for s in subjects[:3]:
        if s.mask is not None:
            strength = clamp01(
                s.score * 1.5
            )

            subject_mask = np.maximum(
                subject_mask,
                s.mask.astype(np.float32)
                * strength,
            )

    masks["subject"] = subject_mask

    # Background
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

    background = 1.0 - occupied

    masks["background"] = np.clip(
        background,
        0,
        1,
    )

    # Smooth masks
    for key, mask in masks.items():

        mask = cv2.GaussianBlur(
            mask.astype(np.float32),
            (0, 0),
            3,
        )

        masks[key] = np.clip(
            mask,
            0,
            1,
        )

    return masks


# ============================================================
# Region profiles
# ============================================================

def make_region_profiles(
    scene,
    stats,
    shooting,
):
    profiles = {}

    # Sky
    profiles["sky"] = RegionProfile(
        exposure=-0.02,
        contrast=1.01,
        saturation=0.99,
        highlight_protection=0.15,
        shadow_lift=0.00,
        clarity=0.04,
        denoise=0.05,
    )

    # Person
    profiles["person"] = RegionProfile(
        exposure=0.02,
        contrast=0.98,
        saturation=0.97,
        highlight_protection=0.08,
        shadow_lift=0.04,
        clarity=0.01,
        denoise=0.08,
    )

    # Vegetation
    profiles["vegetation"] = RegionProfile(
        exposure=0.01,
        contrast=1.02,
        saturation=1.04,
        highlight_protection=0.04,
        shadow_lift=0.01,
        clarity=0.04,
        denoise=0.02,
    )

    # Building
    profiles["building"] = RegionProfile(
        exposure=0.00,
        contrast=1.05,
        saturation=0.99,
        highlight_protection=0.05,
        shadow_lift=0.02,
        clarity=0.07,
        denoise=0.02,
    )

    # Water
    profiles["water"] = RegionProfile(
        exposure=0.00,
        contrast=1.02,
        saturation=1.02,
        highlight_protection=0.10,
        shadow_lift=0.02,
        clarity=0.03,
        denoise=0.04,
    )

    # Vehicle
    profiles["vehicle"] = RegionProfile(
        exposure=0.01,
        contrast=1.04,
        saturation=1.01,
        highlight_protection=0.07,
        shadow_lift=0.02,
        clarity=0.05,
        denoise=0.03,
    )

    # Background
    profiles["background"] = RegionProfile(
        exposure=-0.01,
        contrast=0.99,
        saturation=0.98,
        highlight_protection=0.02,
        shadow_lift=0.00,
        clarity=-0.02,
        denoise=0.05,
    )

    # Scene-dependent adjustments

    if scene == "portrait":
        profiles["person"].saturation *= 0.98
        profiles["person"].contrast *= 0.99

    elif scene == "night":
        profiles["sky"].highlight_protection += 0.08
        profiles["background"].denoise += 0.10

    elif scene == "sunset":
        profiles["sky"].highlight_protection += 0.10
        profiles["sky"].saturation *= 0.98

    elif scene == "landscape":
        profiles["vegetation"].clarity += 0.02
        profiles["water"].highlight_protection += 0.05

    # High ISO
    if shooting.estimated_noise > 0.55:
        for profile in profiles.values():
            profile.denoise += (
                0.10
                * shooting.estimated_noise
            )

            profile.clarity *= 0.75

    return profiles


# ============================================================
# Region adjustment
# ============================================================

def apply_region_adjustment(
    img,
    mask,
    profile,
    raw_headroom,
):
    if np.max(mask) < 1e-5:
        return img

    out = img.copy()

    # Exposure
    exposure = profile.exposure

    out *= (
        2.0 ** (
            exposure * mask[:, :, None]
        )
    )

    # Contrast
    local_gray = cv2.cvtColor(
        np.clip(out, 0, 1).astype(np.float32),
        cv2.COLOR_RGB2GRAY,
    )

    centered = (
        out
        - local_gray[:, :, None]
    )

    out = (
        local_gray[:, :, None]
        + centered
        * (
            1.0
            + (
                profile.contrast - 1.0
            )
            * mask[:, :, None]
        )
    )

    # Saturation
    hsv = cv2.cvtColor(
        np.clip(out, 0, 1).astype(np.float32),
        cv2.COLOR_RGB2HSV,
    )

    hsv[:, :, 1] *= (
        1.0
        + (
            profile.saturation - 1.0
        )
        * mask
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

    # Highlight protection
    luminance = cv2.cvtColor(
        np.clip(out, 0, 1).astype(np.float32),
        cv2.COLOR_RGB2GRAY,
    )

    highlight = np.clip(
        (luminance - 0.65) / 0.35,
        0,
        1,
    )

    protection = (
        profile.highlight_protection
        * (0.5 + 0.5 * raw_headroom)
        * highlight
        * mask
    )

    out *= (
        1.0
        - protection[:, :, None]
    )

    # Shadow lift
    shadow = np.clip(
        (0.35 - luminance) / 0.35,
        0,
        1,
    )

    lift = (
        profile.shadow_lift
        * shadow
        * mask
    )

    out += lift[:, :, None]

    # Clarity
    if abs(profile.clarity) > 1e-5:

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
            profile.clarity
            * detail[:, :, None]
            * mask[:, :, None]
        )

    return np.clip(out, 0.0, 1.0)


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
        out.astype(np.float32),
        cv2.COLOR_RGB2GRAY,
    )

    # Shadow lift
    shadow = np.clip(
        (0.35 - y) / 0.35,
        0,
        1,
    )

    y2 = y + shadow * shadow_lift

    # Highlight compression
    highlight = np.clip(
        (y2 - 0.65) / 0.35,
        0,
        1,
    )

    compression = (
        highlight
        * highlight_protection
        * 0.20
    )

    y2 = y2 * (
        1.0 - compression
    )

    # Smooth S curve
    y2 = np.clip(y2, 0, 1)

    y2 = (
        y2
        * y2
        * (3.0 - 2.0 * y2)
    )

    ratio = (
        y2
        / np.maximum(y, 1e-4)
    )

    out *= ratio[:, :, None]

    return np.clip(out, 0, 1)


# ============================================================
# Denoise
# ============================================================

def denoise_image(img, strength):
    strength = float(np.clip(strength, 0, 1))

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
        img.astype(np.float32),
        d=0,
        sigmaColor=sigma_color,
        sigmaSpace=sigma_space,
    )

    return np.clip(out, 0, 1)


# ============================================================
# Sharpen
# ============================================================

def sharpen_image(img, strength):
    strength = float(np.clip(strength, 0, 1))

    if strength <= 0.001:
        return img

    blur = cv2.GaussianBlur(
        img,
        (0, 0),
        1.1,
    )

    detail = img - blur

    out = (
        img
        + detail * (
            0.35
            + 0.75 * strength
        )
    )

    return np.clip(out, 0, 1)


# ============================================================
# Final local subject emphasis
# ============================================================

def apply_subject_emphasis(
    img,
    subjects,
    scene_profile,
):
    if not subjects:
        return img

    out = img.copy()

    for subject in subjects[:3]:

        if subject.mask is None:
            continue

        mask = subject.mask.astype(
            np.float32
        )

        mask = cv2.GaussianBlur(
            mask,
            (0, 0),
            8,
        )

        strength = (
            scene_profile.subject_strength
            * clamp01(
                subject.score * 1.3
            )
        )

        if strength <= 0:
            continue

        gray = cv2.cvtColor(
            out.astype(np.float32),
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
            * strength
            * mask[:, :, None]
        )

    return np.clip(out, 0, 1)


# ============================================================
# Background suppression
# ============================================================

def suppress_background(
    img,
    masks,
    scene_profile,
):
    background = masks.get(
        "background"
    )

    if background is None:
        return img

    strength = (
        scene_profile.background_suppression
    )

    if strength <= 0:
        return img

    out = img.copy()

    # Slightly reduce saturation
    hsv = cv2.cvtColor(
        out.astype(np.float32),
        cv2.COLOR_RGB2HSV,
    )

    hsv[:, :, 1] *= (
        1.0
        - strength * background
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

    # Slightly reduce local contrast
    gray = cv2.cvtColor(
        out.astype(np.float32),
        cv2.COLOR_RGB2GRAY,
    )

    blur = cv2.GaussianBlur(
        gray,
        (0, 0),
        9,
    )

    detail = gray - blur

    out -= (
        detail[:, :, None]
        * strength
        * background[:, :, None]
    )

    return np.clip(out, 0, 1)


# ============================================================
# JPEG
# ============================================================

def save_jpeg(img, path, quality=95):
    arr = np.clip(
        img * 255.0,
        0,
        255,
    ).astype(np.uint8)

    # Internal processing is RGB.
    # OpenCV writes BGR.
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

def print_camera_profile(profile):
    print(
        f"[INFO] RAW camera: "
        f"{profile.make or 'UNKNOWN'} "
        f"{profile.model or 'UNKNOWN'}"
    )

    print(
        f"[INFO] Camera family: "
        f"{profile.family}"
    )

    if profile.iso is not None:
        print(
            f"[INFO] ISO: "
            f"{profile.iso:g}"
        )

    if profile.exposure_time is not None:
        print(
            f"[INFO] Exposure: "
            f"{profile.exposure_time:g} sec"
        )

    if profile.f_number is not None:
        print(
            f"[INFO] Aperture: "
            f"f/{profile.f_number:g}"
        )

    if profile.focal_length is not None:
        print(
            f"[INFO] Focal length: "
            f"{profile.focal_length:g} mm"
        )

    if profile.lens_model:
        print(
            f"[INFO] Lens: "
            f"{profile.lens_model}"
        )

    if profile.color_temperature is not None:
        print(
            f"[INFO] Color temperature: "
            f"{profile.color_temperature:g} K"
        )

    print(
        f"[INFO] Metadata source: "
        f"{profile.metadata_source}"
    )

    print(
        f"[INFO] LibRaw: "
        f"{profile.libraw_version}"
    )

    print(
        f"[INFO] RAW size: "
        f"{profile.raw_width} x "
        f"{profile.raw_height}"
    )

    if profile.color_desc:
        print(
            f"[INFO] Color description: "
            f"{profile.color_desc}"
        )

    print(
        f"[INFO] Number of colors: "
        f"{profile.num_colors}"
    )

    if profile.white_level is not None:
        print(
            f"[INFO] RAW white level: "
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

    if profile.color_matrix is not None:
        print(
            "[INFO] Color matrix available"
        )


def print_stats(stats):
    print(
        "[INFO] "
        f"mean={stats.mean:.3f}, "
        f"median={stats.median:.3f}, "
        f"p01={stats.p01:.3f}, "
        f"p05={stats.p05:.3f}, "
        f"p95={stats.p95:.3f}, "
        f"p99={stats.p99:.3f}"
    )

    print(
        "[INFO] "
        f"shadow={stats.shadow_ratio:.3f}, "
        f"highlight={stats.highlight_ratio:.3f}, "
        f"saturation={stats.saturation_ratio:.3f}, "
        f"contrast={stats.contrast:.3f}, "
        f"edge={stats.edge_density:.3f}"
    )


def print_subjects(subjects):
    if not subjects:
        print("[INFO] Subjects: none")
        return

    print("[INFO] Subject candidates:")

    for s in subjects[:5]:
        print(
            f"       {s.class_name:12s} "
            f"score={s.score:.3f} "
            f"conf={s.confidence:.3f} "
            f"area={s.area:.3f} "
            f"saliency={s.saliency:.3f}"
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
    print("=" * 72)
    print(f"[INFO] Processing: {path}")

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata = extract_metadata(path)

    # --------------------------------------------------------
    # RAW
    # --------------------------------------------------------

    raw, camera_rgb = develop_raw_camera_rgb(
        path
    )

    try:
        profile = make_camera_profile(
            raw,
            metadata,
        )

        print_camera_profile(profile)

        # ----------------------------------------------------
        # Camera RGB -> sRGB
        # ----------------------------------------------------

        img = camera_rgb_to_srgb(
            camera_rgb,
            profile,
        )

    finally:
        raw.close()

    # --------------------------------------------------------
    # Resize for expensive analysis
    # --------------------------------------------------------

    analysis_img = resize_for_analysis(
        img,
        max_size=1024,
    )

    # --------------------------------------------------------
    # Image analysis
    # --------------------------------------------------------

    stats = analyze_image(
        analysis_img
    )

    print_stats(stats)

    shooting = analyze_shooting_condition(
        profile
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
    # Semantic segmentation
    # --------------------------------------------------------

    segmentation, confidence = (
        segmenter.predict(
            analysis_img
        )
    )

    # --------------------------------------------------------
    # Subject ranking
    # --------------------------------------------------------

    subjects = rank_subjects(
        analysis_img,
        segmentation,
        confidence,
        saliency,
    )

    print_subjects(subjects)

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

    scene_profile = SCENE_PROFILES[
        scene
    ]

    # --------------------------------------------------------
    # Global optimization
    # --------------------------------------------------------

    print(
        "[INFO] Running global parameter search..."
    )

    global_ev, global_contrast, global_sat, out = (
        optimize_global(
            analysis_img,
            scene_profile,
        )
    )

    print(
        "[INFO] Global parameters: "
        f"EV={global_ev:+.2f}, "
        f"contrast={global_contrast:.3f}, "
        f"saturation={global_sat:.3f}"
    )

    # --------------------------------------------------------
    # Region masks
    # --------------------------------------------------------

    masks = build_region_masks(
        analysis_img,
        segmentation,
        subjects,
    )

    region_profiles = make_region_profiles(
        scene,
        stats,
        shooting,
    )

    # --------------------------------------------------------
    # RAW headroom
    # --------------------------------------------------------

    if profile.white_level:
        white_level = profile.white_level

        # Camera RGB is normalized to white level.
        # Use image highlight occupancy as practical
        # headroom estimate.
        headroom = clamp01(
            1.0 - stats.highlight_ratio * 8.0
        )
    else:
        headroom = 0.5

    # --------------------------------------------------------
    # Region-aware development
    # --------------------------------------------------------

    region_order = [
        "sky",
        "water",
        "vegetation",
        "building",
        "vehicle",
        "person",
        "background",
    ]

    for region_name in region_order:

        mask = masks.get(
            region_name
        )

        region_profile = region_profiles.get(
            region_name
        )

        if mask is None or region_profile is None:
            continue

        coverage = float(
            np.mean(mask > 0.05)
        )

        if coverage < 0.002:
            continue

        print(
            f"[INFO] Region {region_name}: "
            f"coverage={coverage:.3f}, "
            f"EV={region_profile.exposure:+.2f}, "
            f"contrast={region_profile.contrast:.3f}, "
            f"saturation={region_profile.saturation:.3f}"
        )

        out = apply_region_adjustment(
            out,
            mask,
            region_profile,
            headroom,
        )

    # --------------------------------------------------------
    # Subject emphasis
    # --------------------------------------------------------

    out = apply_subject_emphasis(
        out,
        subjects,
        scene_profile,
    )

    # --------------------------------------------------------
    # Background suppression
    # --------------------------------------------------------

    out = suppress_background(
        out,
        masks,
        scene_profile,
    )

    # --------------------------------------------------------
    # Tone mapping
    # --------------------------------------------------------

    out = luminance_tone_map(
        out,
        scene_profile.highlight_protection,
        scene_profile.shadow_lift,
    )

    # --------------------------------------------------------
    # Denoise
    # --------------------------------------------------------

    out = denoise_image(
        out,
        scene_profile.denoise
        + shooting.estimated_noise * 0.20,
    )

    # --------------------------------------------------------
    # Sharpen
    # --------------------------------------------------------

    sharpen_strength = (
        scene_profile.sharpen
        * (
            1.0
            - 0.35
            * shooting.estimated_noise
        )
    )

    out = sharpen_image(
        out,
        sharpen_strength,
    )

    # --------------------------------------------------------
    # Final output
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
# Collect RAW files
# ============================================================

def collect_raw_files(directory):
    files = []

    for path in directory.rglob("*"):
        if (
            path.is_file()
            and path.suffix.lower()
            in RAW_EXTENSIONS
        ):
            files.append(path)

    return sorted(files)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Automatic RAW development v15"
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
        default=Path("./output"),
        help="Output directory",
    )

    parser.add_argument(
        "--device",
        default="cpu",
        choices=[
            "cpu",
            "cuda",
        ],
        help="Segmentation device",
    )

    args = parser.parse_args()

    if not args.input.exists():
        print(
            f"[ERROR] Not found: "
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
                "[WARN] CUDA requested but "
                "CUDA is unavailable. "
                "Using CPU."
            )

            device = "cpu"

    print(
        f"[INFO] Device: {device}"
    )

    # --------------------------------------------------------
    # Load model ONCE
    # --------------------------------------------------------

    segmenter = Segmenter(
        device=device
    )

    # --------------------------------------------------------
    # Input files
    # --------------------------------------------------------

    if args.input.is_file():

        raw_files = [
            args.input
        ]

    else:

        raw_files = collect_raw_files(
            args.input
        )

    if not raw_files:
        print(
            "[ERROR] No RAW files found."
        )
        sys.exit(1)

    print(
        f"[INFO] RAW files: "
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
                f"        {type(e).__name__}: "
                f"{e}"
            )

    print()
    print("=" * 72)
    print("[INFO] All processing finished.")


if __name__ == "__main__":
    main()