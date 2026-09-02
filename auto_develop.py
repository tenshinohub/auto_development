#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
auto_develop_v19.py

Automatic RAW Development v19 Debug Edition

v18 -> v19 changes
-------------------
1. LibRaw sRGB color conversion is retained.
2. gamma=(1,1) output is treated as LINEAR RGB.
   Do NOT apply srgb_to_linear() to it.
3. Analysis is performed on display-referred sRGB generated from
   linear RGB.
4. --debug only:
   - detailed statistics
   - intermediate images
   - JSON report
   - segmentation / saliency / masks
   - development stage images

Usage
-----
Normal:
    python3 auto_develop_v19.py photos -o output

Debug:
    python3 auto_develop_v19.py photos -o output --debug

CUDA:
    python3 auto_develop_v19.py photos -o output --device cuda --debug
"""

import argparse
import json
import math
import os
import shutil
import subprocess
import warnings

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import cv2
import numpy as np
import rawpy
from PIL import Image

import torch
from torchvision.models.segmentation import (
    deeplabv3_mobilenet_v3_large,
    DeepLabV3_MobileNet_V3_Large_Weights,
)


VERSION = "v19"


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


ANIMAL_CLASSES = {
    "bird",
    "cat",
    "cow",
    "dog",
    "horse",
    "sheep",
}

VEHICLE_CLASSES = {
    "aeroplane",
    "bicycle",
    "boat",
    "bus",
    "car",
    "motorbike",
    "train",
}

PLANT_CLASSES = {
    "pottedplant",
}


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

    source: str = ""


@dataclass
class CameraProfile:
    make: str = ""
    model: str = ""
    family: str = ""

    iso: Optional[float] = None
    black_level: Optional[float] = None
    white_level: Optional[float] = None

    camera_wb: Optional[List[float]] = None
    color_matrix: Optional[List[List[float]]] = None
    rgb_xyz_matrix: Optional[List[List[float]]] = None

    raw_width: int = 0
    raw_height: int = 0

    lens: str = ""
    metadata_source: str = ""
    libraw_version: str = ""


@dataclass
class ImageStats:
    mean: float = 0.0
    median: float = 0.0

    p01: float = 0.0
    p05: float = 0.0
    p25: float = 0.0
    p75: float = 0.0
    p95: float = 0.0
    p99: float = 0.0

    min_value: float = 0.0
    max_value: float = 0.0

    shadow_ratio: float = 0.0
    highlight_ratio: float = 0.0

    dynamic_range: float = 0.0
    saturation_ratio: float = 0.0

    edge_density: float = 0.0
    warm_ratio: float = 0.0

    contrast: float = 0.0
    mean_luminance: float = 0.0

    r_mean: float = 0.0
    g_mean: float = 0.0
    b_mean: float = 0.0

    rg_ratio: float = 0.0
    gb_ratio: float = 0.0


@dataclass
class ShootingCondition:
    iso_factor: float = 1.0
    low_light: bool = False
    motion_risk: float = 0.0
    shallow_dof: bool = False
    wide_angle: bool = False
    telephoto: bool = False
    estimated_noise: float = 0.0


@dataclass
class SubjectCandidate:
    class_name: str = ""
    confidence: float = 0.0
    area: float = 0.0

    center_x: float = 0.0
    center_y: float = 0.0

    saliency: float = 0.0
    local_contrast: float = 0.0
    colorfulness: float = 0.0

    score: float = 0.0


@dataclass
class SceneResult:
    scene: str = "general"
    confidence: float = 0.0


@dataclass
class RegionStats:
    name: str = ""
    area: float = 0.0
    mean_luminance: float = 0.0
    mean_saturation: float = 0.0


@dataclass
class DevelopParams:
    exposure_ev: float = 0.0
    contrast: float = 1.04
    saturation: float = 1.0

    highlight_protection: float = 0.35
    shadow_lift: float = 0.06

    subject_exposure: float = 0.04
    subject_contrast: float = 1.02
    background_suppression: float = 0.015

    denoise: float = 0.30
    sharpen: float = 0.75

    region_skin_saturation: float = 0.96
    region_green_saturation: float = 1.01
    region_water_highlight: float = 0.07
    region_upper_highlight: float = 0.12

    tone_strength: float = 0.52


# ============================================================
# Utility
# ============================================================

def clamp01(x):
    return np.clip(x, 0.0, 1.0)


def safe_float(value, default=None):
    try:
        if value is None:
            return default

        if isinstance(value, str):
            value = value.strip()

        return float(value)
    except Exception:
        return default


def safe_int(value, default=None):
    try:
        return int(value)
    except Exception:
        return default


def percentile(arr, p):
    if arr.size == 0:
        return 0.0
    return float(np.percentile(arr, p))


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


# ============================================================
# Color conversion
# ============================================================

def linear_to_srgb(x):
    x = np.clip(x, 0.0, 1.0)

    return np.where(
        x <= 0.0031308,
        12.92 * x,
        1.055 * np.power(np.maximum(x, 0.0), 1.0 / 2.4) - 0.055,
    )


def srgb_to_linear(x):
    x = np.clip(x, 0.0, 1.0)

    return np.where(
        x <= 0.04045,
        x / 12.92,
        np.power((x + 0.055) / 1.055, 2.4),
    )


# ============================================================
# Statistics
# ============================================================

def calculate_stats(rgb: np.ndarray) -> ImageStats:
    rgb = np.asarray(rgb, dtype=np.float32)
    rgb = np.nan_to_num(rgb, nan=0.0, posinf=1.0, neginf=0.0)

    rgb = np.clip(rgb, 0.0, 1.0)

    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("RGB image required")

    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]

    lum = (
        0.2126 * r +
        0.7152 * g +
        0.0722 * b
    )

    flat = lum.reshape(-1)

    mean = float(np.mean(flat))
    median = float(np.median(flat))

    p01 = percentile(flat, 1)
    p05 = percentile(flat, 5)
    p25 = percentile(flat, 25)
    p75 = percentile(flat, 75)
    p95 = percentile(flat, 95)
    p99 = percentile(flat, 99)

    min_value = float(np.min(flat))
    max_value = float(np.max(flat))

    shadow_ratio = float(np.mean(lum <= 0.01))
    highlight_ratio = float(np.mean(lum >= 0.99))

    dynamic_range = math.log10(
        max(p95, 1e-6) / max(p05, 1e-6)
    )

    hsv = cv2.cvtColor(
        (rgb * 255.0).astype(np.uint8),
        cv2.COLOR_RGB2HSV,
    )

    saturation = hsv[:, :, 1].astype(np.float32) / 255.0

    saturation_ratio = float(
        np.mean(saturation > 0.90)
    )

    warm = (
        (r > b * 1.15) &
        (r > g * 0.95) &
        (g > b * 1.05)
    )

    warm_ratio = float(np.mean(warm))

    gx = cv2.Sobel(
        lum,
        cv2.CV_32F,
        1,
        0,
        ksize=3,
    )

    gy = cv2.Sobel(
        lum,
        cv2.CV_32F,
        0,
        1,
        ksize=3,
    )

    edge = cv2.magnitude(gx, gy)

    edge_density = float(
        np.mean(edge > 0.08)
    )

    contrast = float(np.std(lum))

    r_mean = float(np.mean(r))
    g_mean = float(np.mean(g))
    b_mean = float(np.mean(b))

    rg_ratio = r_mean / max(g_mean, 1e-6)
    gb_ratio = g_mean / max(b_mean, 1e-6)

    return ImageStats(
        mean=mean,
        median=median,
        p01=p01,
        p05=p05,
        p25=p25,
        p75=p75,
        p95=p95,
        p99=p99,
        min_value=min_value,
        max_value=max_value,
        shadow_ratio=shadow_ratio,
        highlight_ratio=highlight_ratio,
        dynamic_range=dynamic_range,
        saturation_ratio=saturation_ratio,
        edge_density=edge_density,
        warm_ratio=warm_ratio,
        contrast=contrast,
        mean_luminance=mean,
        r_mean=r_mean,
        g_mean=g_mean,
        b_mean=b_mean,
        rg_ratio=rg_ratio,
        gb_ratio=gb_ratio,
    )


def print_stats(name: str, rgb: np.ndarray):
    stats = calculate_stats(rgb)

    print(f"\n[DEBUG] ===== {name} =====")
    print(
        f"  min/max      : "
        f"{stats.min_value:.6f} / {stats.max_value:.6f}"
    )
    print(
        f"  mean/median  : "
        f"{stats.mean:.6f} / {stats.median:.6f}"
    )

    print(
        f"  p01/p05      : "
        f"{stats.p01:.6f} / {stats.p05:.6f}"
    )

    print(
        f"  p25/p75      : "
        f"{stats.p25:.6f} / {stats.p75:.6f}"
    )

    print(
        f"  p95/p99      : "
        f"{stats.p95:.6f} / {stats.p99:.6f}"
    )

    print(
        f"  shadow       : "
        f"{stats.shadow_ratio * 100:.3f}%"
    )

    print(
        f"  highlight    : "
        f"{stats.highlight_ratio * 100:.3f}%"
    )

    print(
        f"  dynamic range: "
        f"{stats.dynamic_range:.3f}"
    )

    print(
        f"  saturation   : "
        f"{stats.saturation_ratio * 100:.3f}%"
    )

    print(
        f"  edge density : "
        f"{stats.edge_density:.4f}"
    )

    print(
        f"  contrast     : "
        f"{stats.contrast:.6f}"
    )

    print(
        f"  RGB mean     : "
        f"{stats.r_mean:.6f}, "
        f"{stats.g_mean:.6f}, "
        f"{stats.b_mean:.6f}"
    )

    print(
        f"  R/G ratio    : {stats.rg_ratio:.4f}"
    )

    print(
        f"  G/B ratio    : {stats.gb_ratio:.4f}"
    )

    return stats


# ============================================================
# Image saving
# ============================================================

def save_rgb(path: Path, rgb: np.ndarray):
    rgb = np.clip(rgb, 0.0, 1.0)

    img8 = np.round(
        rgb * 255.0
    ).astype(np.uint8)

    Image.fromarray(
        img8,
        mode="RGB",
    ).save(path)


def save_gray(path: Path, image: np.ndarray):
    image = np.clip(image, 0.0, 1.0)

    img8 = np.round(
        image * 255.0
    ).astype(np.uint8)

    Image.fromarray(
        img8,
        mode="L",
    ).save(path)


# ============================================================
# EXIF
# ============================================================

class MetadataReader:

    def __init__(self):
        self.exiftool = shutil.which("exiftool")

    def read(self, path: Path) -> ExifMetadata:
        metadata = ExifMetadata()

        if self.exiftool:
            try:
                metadata = self._read_exiftool(path)
                metadata.source = "exiftool"
                return metadata
            except Exception as e:
                warnings.warn(
                    f"ExifTool failed for {path}: {e}"
                )

        try:
            metadata = self._read_pillow(path)
            metadata.source = "pillow"
        except Exception:
            pass

        return metadata

    def _read_exiftool(self, path: Path) -> ExifMetadata:
        cmd = [
            self.exiftool,
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
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )

        data = json.loads(result.stdout)[0]

        make = (
            data.get("Make") or
            ""
        )

        model = (
            data.get("CameraModelName") or
            data.get("UniqueCameraModel") or
            data.get("Model") or
            ""
        )

        return ExifMetadata(
            make=str(make),
            model=str(model),
            lens_make=str(data.get("LensMake") or ""),
            lens_model=str(data.get("LensModel") or ""),
            iso=safe_float(data.get("ISO")),
            exposure_time=safe_float(
                data.get("ExposureTime")
            ),
            f_number=safe_float(
                data.get("FNumber")
            ),
            focal_length=safe_float(
                data.get("FocalLength")
            ),
            white_balance=str(
                data.get("WhiteBalance") or ""
            ),
            color_temperature=safe_float(
                data.get("ColorTemperature")
            ),
            color_space=str(
                data.get("ColorSpace") or ""
            ),
            source="exiftool",
        )

    def _read_pillow(self, path: Path) -> ExifMetadata:
        with Image.open(path) as img:
            exif = img.getexif()

        return ExifMetadata(
            make=str(exif.get(271, "") or ""),
            model=str(exif.get(272, "") or ""),
            exposure_time=safe_float(
                exif.get(33434)
            ),
            f_number=safe_float(
                exif.get(33437)
            ),
            focal_length=safe_float(
                exif.get(37386)
            ),
            iso=safe_float(
                exif.get(34855)
            ),
            source="pillow",
        )


# ============================================================
# Camera profile
# ============================================================

def detect_camera_family(make: str, model: str) -> str:
    text = f"{make} {model}".lower()

    families = {
        "canon": ["canon"],
        "nikon": ["nikon"],
        "sony": ["sony"],
        "fujifilm": ["fujifilm", "fuji"],
        "panasonic": ["panasonic", "lumix"],
        "olympus": ["olympus", "om digital"],
        "leica": ["leica"],
        "pentax": ["pentax"],
        "ricoh": ["ricoh"],
        "sigma": ["sigma"],
        "hasselblad": ["hasselblad"],
    }

    for family, names in families.items():
        if any(name in text for name in names):
            return family

    return "unknown"


# ============================================================
# RAW decoder
# ============================================================

class RawDecoder:

    def __init__(self):
        try:
            self.libraw_version = rawpy.libraw_version
        except Exception:
            self.libraw_version = ""

    def load(
        self,
        path: Path,
    ) -> Tuple[np.ndarray, CameraProfile, Dict]:

        with rawpy.imread(str(path)) as raw:

            profile = CameraProfile()

            profile.raw_width = int(
                raw.sizes.width
            )

            profile.raw_height = int(
                raw.sizes.height
            )

            profile.libraw_version = str(
                self.libraw_version
            )

            try:
                wb = raw.camera_whitebalance

                if wb is not None:
                    profile.camera_wb = [
                        float(x)
                        for x in np.asarray(wb).reshape(-1)
                    ]

            except Exception:
                pass

            try:
                profile.black_level = float(
                    np.mean(raw.black_level_per_channel)
                )
            except Exception:
                pass

            try:
                profile.white_level = float(
                    raw.white_level
                )
            except Exception:
                pass

            try:
                profile.color_matrix = (
                    np.asarray(
                        raw.color_matrix,
                        dtype=np.float32,
                    ).tolist()
                )
            except Exception:
                pass

            try:
                profile.rgb_xyz_matrix = (
                    np.asarray(
                        raw.rgb_xyz_matrix,
                        dtype=np.float32,
                    ).tolist()
                )
            except Exception:
                pass

            rgb16 = raw.postprocess(
                use_camera_wb=True,
                use_auto_wb=False,
                output_color=rawpy.ColorSpace.sRGB,
                output_bps=16,

                # IMPORTANT:
                # gamma=(1,1) means identity transfer here.
                # v19 treats this as linear RGB.
                gamma=(1, 1),

                no_auto_bright=True,

                highlight_mode=rawpy.HighlightMode.Blend,

                half_size=False,
                four_color_rgb=False,

                demosaic_algorithm=
                rawpy.DemosaicAlgorithm.AHD,
            )

            raw_info = {
                "shape": list(rgb16.shape),
                "dtype": str(rgb16.dtype),
                "raw_width": int(raw.sizes.width),
                "raw_height": int(raw.sizes.height),
            }

        linear_rgb = (
            np.asarray(
                rgb16,
                dtype=np.float32,
            ) / 65535.0
        )

        linear_rgb = np.clip(
            linear_rgb,
            0.0,
            1.0,
        )

        return linear_rgb, profile, raw_info


# ============================================================
# Shooting condition
# ============================================================

def analyze_shooting(
    metadata: ExifMetadata,
) -> ShootingCondition:

    iso = metadata.iso or 100.0

    iso_factor = math.sqrt(
        max(iso, 100.0) / 100.0
    )

    iso_factor = float(
        np.clip(
            iso_factor,
            1.0,
            5.0,
        )
    )

    low_light = (
        iso >= 1600
    )

    exposure = metadata.exposure_time

    if exposure is None:
        motion_risk = 0.0
    else:
        motion_risk = float(
            np.clip(
                (1.0 / max(exposure, 1e-6) - 1.0)
                / 30.0,
                0.0,
                1.0,
            )
        )

    f_number = metadata.f_number

    shallow_dof = (
        f_number is not None and
        f_number <= 2.8
    )

    focal = metadata.focal_length

    wide_angle = (
        focal is not None and
        focal <= 28
    )

    telephoto = (
        focal is not None and
        focal >= 85
    )

    estimated_noise = float(
        np.clip(
            (iso_factor - 1.0) / 4.0,
            0.0,
            1.0,
        )
    )

    return ShootingCondition(
        iso_factor=iso_factor,
        low_light=low_light,
        motion_risk=motion_risk,
        shallow_dof=shallow_dof,
        wide_angle=wide_angle,
        telephoto=telephoto,
        estimated_noise=estimated_noise,
    )


# ============================================================
# Semantic segmentation
# ============================================================

class SemanticSegmenter:

    def __init__(
        self,
        device: str = "auto",
        max_dim: int = 768,
    ):

        if device == "auto":
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        if device == "cuda" and not torch.cuda.is_available():
            print(
                "[WARN] CUDA requested but unavailable. "
                "Falling back to CPU."
            )
            device = "cpu"

        self.device = torch.device(device)
        self.max_dim = max_dim

        print(
            f"[INFO] Loading DeepLabV3 on "
            f"{self.device}"
        )

        try:
            weights = (
                DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
            )

            self.model = (
                deeplabv3_mobilenet_v3_large(
                    weights=weights
                )
            )

            self.transforms = (
                weights.transforms()
            )

        except Exception:
            self.model = (
                deeplabv3_mobilenet_v3_large(
                    pretrained=True
                )
            )

            self.transforms = None

        self.model.eval()
        self.model.to(self.device)

    def predict(
        self,
        rgb: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:

        h, w = rgb.shape[:2]

        scale = min(
            1.0,
            self.max_dim / max(h, w),
        )

        nh = max(
            1,
            int(h * scale),
        )

        nw = max(
            1,
            int(w * scale),
        )

        small = cv2.resize(
            rgb,
            (nw, nh),
            interpolation=cv2.INTER_AREA,
        )

        pil = Image.fromarray(
            np.round(
                np.clip(
                    small,
                    0.0,
                    1.0,
                ) * 255.0
            ).astype(np.uint8)
        )

        if self.transforms is not None:
            tensor = self.transforms(
                pil
            ).unsqueeze(0)

        else:
            arr = np.asarray(
                pil,
                dtype=np.float32,
            ) / 255.0

            tensor = torch.from_numpy(
                arr
            ).permute(
                2, 0, 1
            ).unsqueeze(0)

        tensor = tensor.to(
            self.device
        )

        with torch.inference_mode():
            output = self.model(
                tensor
            )["out"][0]

        probabilities = torch.softmax(
            output,
            dim=0,
        )

        confidence_small, labels_small = (
            probabilities.max(dim=0)
        )

        labels_small = (
            labels_small.cpu()
            .numpy()
            .astype(np.uint8)
        )

        confidence_small = (
            confidence_small.cpu()
            .numpy()
            .astype(np.float32)
        )

        labels = cv2.resize(
            labels_small,
            (w, h),
            interpolation=cv2.INTER_NEAREST,
        )

        confidence = cv2.resize(
            confidence_small,
            (w, h),
            interpolation=cv2.INTER_LINEAR,
        )

        return labels, confidence


# ============================================================
# Saliency
# ============================================================

def compute_saliency(
    rgb: np.ndarray,
) -> np.ndarray:

    lum = (
        0.2126 * rgb[:, :, 0] +
        0.7152 * rgb[:, :, 1] +
        0.0722 * rgb[:, :, 2]
    )

    local_mean = cv2.GaussianBlur(
        lum,
        (0, 0),
        9,
    )

    local_contrast = np.abs(
        lum - local_mean
    )

    local_contrast /= max(
        float(np.percentile(
            local_contrast,
            99
        )),
        1e-6,
    )

    gx = cv2.Sobel(
        lum,
        cv2.CV_32F,
        1,
        0,
        ksize=3,
    )

    gy = cv2.Sobel(
        lum,
        cv2.CV_32F,
        0,
        1,
        ksize=3,
    )

    edge = cv2.magnitude(
        gx,
        gy,
    )

    edge /= max(
        float(np.percentile(
            edge,
            99
        )),
        1e-6,
    )

    hsv = cv2.cvtColor(
        (rgb * 255).astype(np.uint8),
        cv2.COLOR_RGB2HSV,
    )

    saturation = (
        hsv[:, :, 1].astype(np.float32)
        / 255.0
    )

    brightness_dist = np.abs(
        lum -
        cv2.GaussianBlur(
            lum,
            (0, 0),
            15,
        )
    )

    brightness_dist /= max(
        float(np.percentile(
            brightness_dist,
            99
        )),
        1e-6,
    )

    h, w = lum.shape

    yy, xx = np.mgrid[
        0:h,
        0:w,
    ]

    cx = w / 2.0
    cy = h / 2.0

    dx = (
        (xx - cx) /
        max(cx, 1)
    )

    dy = (
        (yy - cy) /
        max(cy, 1)
    )

    center = np.exp(
        -(dx * dx + dy * dy)
        / 0.8
    )

    saliency = (
        0.30 * local_contrast +
        0.25 * edge +
        0.15 * saturation +
        0.20 * brightness_dist +
        0.10 * center
    )

    return np.clip(
        saliency,
        0.0,
        1.0,
    )


# ============================================================
# Subject ranking
# ============================================================

def rank_subjects(
    labels: np.ndarray,
    confidence: np.ndarray,
    saliency: np.ndarray,
    rgb: np.ndarray,
) -> List[SubjectCandidate]:

    h, w = labels.shape

    hsv = cv2.cvtColor(
        (rgb * 255).astype(np.uint8),
        cv2.COLOR_RGB2HSV,
    )

    saturation = (
        hsv[:, :, 1].astype(np.float32)
        / 255.0
    )

    candidates = []

    for class_id, class_name in enumerate(
        VOC_CLASSES
    ):

        if class_name == "background":
            continue

        mask = labels == class_id

        area = float(
            np.mean(mask)
        )

        if area < 0.003:
            continue

        ys, xs = np.where(mask)

        if len(xs) == 0:
            continue

        cx = float(
            np.mean(xs) / w
        )

        cy = float(
            np.mean(ys) / h
        )

        conf = float(
            np.mean(
                confidence[mask]
            )
        )

        sal = float(
            np.mean(
                saliency[mask]
            )
        )

        lum = (
            0.2126 * rgb[:, :, 0] +
            0.7152 * rgb[:, :, 1] +
            0.0722 * rgb[:, :, 2]
        )

        local_mean = cv2.GaussianBlur(
            lum,
            (0, 0),
            9,
        )

        local_contrast = float(
            np.mean(
                np.abs(
                    lum[mask] -
                    local_mean[mask]
                )
            )
        )

        colorfulness = float(
            np.mean(
                saturation[mask]
            )
        )

        center_distance = math.sqrt(
            (cx - 0.5) ** 2 +
            (cy - 0.5) ** 2
        )

        center_score = max(
            0.0,
            1.0 - center_distance / 0.707
        )

        prior = 1.0

        if class_name == "person":
            prior = 1.15
        elif class_name in ANIMAL_CLASSES:
            prior = 1.05
        elif class_name in VEHICLE_CLASSES:
            prior = 1.0
        elif class_name in PLANT_CLASSES:
            prior = 0.90
        elif class_name == "bottle":
            prior = 0.85

        score = prior * (
            0.30 * conf +
            0.15 * math.sqrt(area) +
            0.15 * center_score +
            0.20 * sal +
            0.10 * min(
                local_contrast * 5.0,
                1.0
            ) +
            0.10 * colorfulness
        )

        candidates.append(
            SubjectCandidate(
                class_name=class_name,
                confidence=conf,
                area=area,
                center_x=cx,
                center_y=cy,
                saliency=sal,
                local_contrast=local_contrast,
                colorfulness=colorfulness,
                score=float(score),
            )
        )

    candidates.sort(
        key=lambda x: x.score,
        reverse=True,
    )

    return candidates[:10]


# ============================================================
# Scene classification
# ============================================================

def classify_scene(
    stats: ImageStats,
    shooting: ShootingCondition,
    subjects: List[SubjectCandidate],
) -> SceneResult:

    person_area = sum(
        x.area
        for x in subjects
        if x.class_name == "person"
    )

    vehicle_area = sum(
        x.area
        for x in subjects
        if x.class_name in VEHICLE_CLASSES
    )

    if (
        person_area > 0.015 and
        stats.median > 0.08
    ):
        return SceneResult(
            "portrait",
            0.75,
        )

    if (
        shooting.low_light and
        stats.median < 0.10
    ):
        return SceneResult(
            "night",
            0.80,
        )

    if (
        stats.warm_ratio > 0.18 and
        stats.p95 > 0.55
    ):
        return SceneResult(
            "sunset",
            0.70,
        )

    if (
        shooting.wide_angle and
        stats.edge_density < 0.16 and
        stats.dynamic_range > 5
    ):
        return SceneResult(
            "landscape",
            0.65,
        )

    if (
        vehicle_area > 0.01 and
        stats.edge_density > 0.10
    ):
        return SceneResult(
            "city",
            0.65,
        )

    if (
        stats.median < 0.18 and
        stats.warm_ratio > 0.10
    ):
        return SceneResult(
            "indoor",
            0.60,
        )

    return SceneResult(
        "general",
        0.50,
    )


# ============================================================
# Scene profiles
# ============================================================

SCENE_PROFILES = {

    "portrait": DevelopParams(
        exposure_ev=0.05,
        contrast=1.02,
        saturation=0.97,
        highlight_protection=0.42,
        shadow_lift=0.08,
        subject_exposure=0.08,
        subject_contrast=1.02,
        background_suppression=0.035,
        denoise=0.55,
        sharpen=0.75,
        region_skin_saturation=0.94,
        region_green_saturation=1.00,
        region_water_highlight=0.06,
        region_upper_highlight=0.12,
        tone_strength=0.55,
    ),

    "night": DevelopParams(
        exposure_ev=0.0,
        contrast=1.05,
        saturation=1.03,
        highlight_protection=0.55,
        shadow_lift=0.02,
        subject_exposure=0.05,
        subject_contrast=1.03,
        background_suppression=0.015,
        denoise=0.85,
        sharpen=0.45,
        region_skin_saturation=0.94,
        region_green_saturation=1.00,
        region_water_highlight=0.10,
        region_upper_highlight=0.18,
        tone_strength=0.45,
    ),

    "sunset": DevelopParams(
        exposure_ev=-0.05,
        contrast=1.06,
        saturation=1.08,
        highlight_protection=0.55,
        shadow_lift=0.04,
        subject_exposure=0.05,
        subject_contrast=1.03,
        background_suppression=0.015,
        denoise=0.30,
        sharpen=0.80,
        region_skin_saturation=0.96,
        region_green_saturation=1.02,
        region_water_highlight=0.10,
        region_upper_highlight=0.20,
        tone_strength=0.60,
    ),

    "landscape": DevelopParams(
        exposure_ev=0.03,
        contrast=1.08,
        saturation=1.04,
        highlight_protection=0.40,
        shadow_lift=0.08,
        subject_exposure=0.06,
        subject_contrast=1.03,
        background_suppression=0.015,
        denoise=0.30,
        sharpen=0.85,
        region_skin_saturation=0.96,
        region_green_saturation=1.03,
        region_water_highlight=0.10,
        region_upper_highlight=0.18,
        tone_strength=0.60,
    ),

    "city": DevelopParams(
        exposure_ev=0.02,
        contrast=1.07,
        saturation=1.02,
        highlight_protection=0.45,
        shadow_lift=0.05,
        subject_exposure=0.06,
        subject_contrast=1.03,
        background_suppression=0.02,
        denoise=0.40,
        sharpen=0.80,
        region_skin_saturation=0.95,
        region_green_saturation=1.01,
        region_water_highlight=0.08,
        region_upper_highlight=0.16,
        tone_strength=0.58,
    ),

    "indoor": DevelopParams(
        exposure_ev=0.04,
        contrast=1.03,
        saturation=0.99,
        highlight_protection=0.40,
        shadow_lift=0.08,
        subject_exposure=0.05,
        subject_contrast=1.02,
        background_suppression=0.015,
        denoise=0.50,
        sharpen=0.65,
        region_skin_saturation=0.94,
        region_green_saturation=1.00,
        region_water_highlight=0.05,
        region_upper_highlight=0.10,
        tone_strength=0.50,
    ),

    "general": DevelopParams(
        exposure_ev=0.0,
        contrast=1.04,
        saturation=1.00,
        highlight_protection=0.35,
        shadow_lift=0.06,
        subject_exposure=0.04,
        subject_contrast=1.02,
        background_suppression=0.015,
        denoise=0.30,
        sharpen=0.75,
        region_skin_saturation=0.96,
        region_green_saturation=1.01,
        region_water_highlight=0.07,
        region_upper_highlight=0.12,
        tone_strength=0.52,
    ),
}


# ============================================================
# Exposure estimation
# ============================================================

TARGET_MEDIANS = {
    "portrait": 0.18,
    "night": 0.08,
    "sunset": 0.14,
    "landscape": 0.20,
    "city": 0.18,
    "indoor": 0.17,
    "general": 0.18,
}


def estimate_exposure(
    stats: ImageStats,
    scene: str,
) -> float:

    target = TARGET_MEDIANS.get(
        scene,
        0.18,
    )

    median = max(
        stats.median,
        1e-4,
    )

    p95 = max(
        stats.p95,
        1e-4,
    )

    ev_mid = math.log2(
        target / median
    )

    ev_high = math.log2(
        0.72 / p95
    )

    ev = (
        0.60 * ev_mid +
        0.40 * ev_high
    )

    return float(
        np.clip(
            ev,
            -0.75,
            0.75,
        )
    )


# ============================================================
# Exposure application
# ============================================================

def apply_exposure(
    linear_rgb: np.ndarray,
    ev: float,
) -> np.ndarray:

    gain = 2.0 ** ev

    return np.clip(
        linear_rgb * gain,
        0.0,
        1.0,
    )


# ============================================================
# Global contrast
# ============================================================

def apply_contrast(
    linear_rgb: np.ndarray,
    contrast: float,
) -> np.ndarray:

    lum = (
        0.2126 * linear_rgb[:, :, 0] +
        0.7152 * linear_rgb[:, :, 1] +
        0.0722 * linear_rgb[:, :, 2]
    )

    mean_lum = float(
        np.median(lum)
    )

    new_lum = (
        (lum - mean_lum) *
        contrast +
        mean_lum
    )

    ratio = (
        new_lum /
        np.maximum(lum, 1e-5)
    )

    result = linear_rgb * ratio[:, :, None]

    return np.clip(
        result,
        0.0,
        1.0,
    )


# ============================================================
# Global saturation
# ============================================================

def apply_saturation(
    linear_rgb: np.ndarray,
    saturation: float,
) -> np.ndarray:

    lum = (
        0.2126 * linear_rgb[:, :, 0] +
        0.7152 * linear_rgb[:, :, 1] +
        0.0722 * linear_rgb[:, :, 2]
    )

    result = (
        lum[:, :, None] +
        (
            linear_rgb -
            lum[:, :, None]
        ) * saturation
    )

    return np.clip(
        result,
        0.0,
        1.0,
    )


# ============================================================
# Tone curve
# ============================================================

def apply_tone(
    linear_rgb: np.ndarray,
    params: DevelopParams,
) -> np.ndarray:

    lum = (
        0.2126 * linear_rgb[:, :, 0] +
        0.7152 * linear_rgb[:, :, 1] +
        0.0722 * linear_rgb[:, :, 2]
    )

    strength = params.tone_strength

    # Shadow lift
    shadow_mask = np.clip(
        (0.25 - lum) / 0.25,
        0.0,
        1.0,
    )

    lifted = (
        lum +
        params.shadow_lift *
        shadow_mask *
        (1.0 - lum)
    )

    # Highlight rolloff
    highlight_mask = np.clip(
        (lum - 0.65) / 0.35,
        0.0,
        1.0,
    )

    rolled = (
        lifted -
        params.highlight_protection *
        highlight_mask *
        np.maximum(
            lifted - 0.65,
            0.0
        ) *
        0.45
    )

    # Mild S curve
    centered = rolled - 0.5

    curve = (
        centered *
        (
            1.0 +
            strength *
            0.20 *
            (
                1.0 -
                4.0 *
                centered *
                centered
            )
        )
    )

    new_lum = np.clip(
        0.5 + curve,
        0.0,
        1.0,
    )

    ratio = (
        new_lum /
        np.maximum(
            lum,
            1e-5
        )
    )

    result = (
        linear_rgb *
        ratio[:, :, None]
    )

    return np.clip(
        result,
        0.0,
        1.0,
    )


# ============================================================
# Region masks
# ============================================================

def create_region_masks(
    rgb: np.ndarray,
    labels: np.ndarray,
    subjects: List[SubjectCandidate],
) -> Dict[str, np.ndarray]:

    h, w = labels.shape

    person = (
        labels ==
        VOC_CLASSES.index("person")
    )

    animal = np.zeros(
        (h, w),
        dtype=bool,
    )

    for name in ANIMAL_CLASSES:
        animal |= (
            labels ==
            VOC_CLASSES.index(name)
        )

    vehicle = np.zeros(
        (h, w),
        dtype=bool,
    )

    for name in VEHICLE_CLASSES:
        vehicle |= (
            labels ==
            VOC_CLASSES.index(name)
        )

    plant = np.zeros(
        (h, w),
        dtype=bool,
    )

    for name in PLANT_CLASSES:
        plant |= (
            labels ==
            VOC_CLASSES.index(name)
        )

    subject = (
        person |
        animal |
        vehicle |
        plant
    )

    hsv = cv2.cvtColor(
        (rgb * 255).astype(np.uint8),
        cv2.COLOR_RGB2HSV,
    )

    h_channel = hsv[:, :, 0]
    s_channel = hsv[:, :, 1]
    v_channel = hsv[:, :, 2]

    skin = (
        person &
        (
            (
                h_channel >= 0
            ) &
            (
                h_channel <= 25
            )
        ) &
        (s_channel >= 30) &
        (s_channel <= 190) &
        (v_channel >= 50)
    )

    green = (
        (h_channel >= 30) &
        (h_channel <= 95) &
        (s_channel >= 45) &
        (v_channel >= 30)
    )

    blue = (
        (h_channel >= 80) &
        (h_channel <= 135) &
        (s_channel >= 40) &
        (v_channel >= 40)
    )

    lum = (
        0.2126 * rgb[:, :, 0] +
        0.7152 * rgb[:, :, 1] +
        0.0722 * rgb[:, :, 2]
    )

    p75 = np.percentile(
        lum,
        75,
    )

    yy = np.arange(h)[:, None]

    upper_bright = (
        (yy < h * 0.45) &
        (lum > p75)
    )

    lower_half = (
        yy >= h * 0.50
    )

    local_std = cv2.GaussianBlur(
        (lum * lum).astype(np.float32),
        (0, 0),
        5,
    ) - (
        cv2.GaussianBlur(
            lum,
            (0, 0),
            5,
        ) ** 2
    )

    water = (
        blue &
        lower_half &
        (
            local_std < 0.02
        )
    )

    return {
        "person": person,
        "animal": animal,
        "vehicle": vehicle,
        "plant": plant,
        "subject": subject,
        "skin": skin,
        "green": green,
        "blue": blue,
        "upper_bright": upper_bright,
        "water": water,
    }


# ============================================================
# Region statistics
# ============================================================

def calculate_region_stats(
    rgb: np.ndarray,
    masks: Dict[str, np.ndarray],
) -> List[RegionStats]:

    hsv = cv2.cvtColor(
        (rgb * 255).astype(np.uint8),
        cv2.COLOR_RGB2HSV,
    )

    saturation = (
        hsv[:, :, 1].astype(np.float32)
        / 255.0
    )

    lum = (
        0.2126 * rgb[:, :, 0] +
        0.7152 * rgb[:, :, 1] +
        0.0722 * rgb[:, :, 2]
    )

    result = []

    total = rgb.shape[0] * rgb.shape[1]

    for name, mask in masks.items():

        area = float(
            np.sum(mask) /
            max(total, 1)
        )

        if not np.any(mask):
            mean_lum = 0.0
            mean_sat = 0.0
        else:
            mean_lum = float(
                np.mean(
                    lum[mask]
                )
            )

            mean_sat = float(
                np.mean(
                    saturation[mask]
                )
            )

        result.append(
            RegionStats(
                name=name,
                area=area,
                mean_luminance=mean_lum,
                mean_saturation=mean_sat,
            )
        )

    return result


# ============================================================
# Region development
# ============================================================

def apply_region_development(
    linear_rgb: np.ndarray,
    masks: Dict[str, np.ndarray],
    params: DevelopParams,
) -> np.ndarray:

    result = linear_rgb.copy()

    if np.any(masks["subject"]):

        subject = masks["subject"]

        gain = (
            2.0 **
            params.subject_exposure
        )

        result[subject] *= gain

    if np.any(masks["subject"]):

        subject = masks["subject"]

        lum = (
            0.2126 * result[:, :, 0] +
            0.7152 * result[:, :, 1] +
            0.0722 * result[:, :, 2]
        )

        mean_lum = float(
            np.mean(
                lum[subject]
            )
        )

        new_lum = (
            (
                lum -
                mean_lum
            ) *
            params.subject_contrast +
            mean_lum
        )

        ratio = (
            new_lum /
            np.maximum(
                lum,
                1e-5,
            )
        )

        temp = (
            result *
            ratio[:, :, None]
        )

        result[subject] = (
            temp[subject]
        )

    if np.any(masks["skin"]):

        skin = masks["skin"]

        lum = (
            0.2126 * result[:, :, 0] +
            0.7152 * result[:, :, 1] +
            0.0722 * result[:, :, 2]
        )

        temp = (
            lum[:, :, None] +
            (
                result -
                lum[:, :, None]
            ) *
            params.region_skin_saturation
        )

        result[skin] = (
            temp[skin]
        )

    if np.any(masks["green"]):

        green = masks["green"]

        lum = (
            0.2126 * result[:, :, 0] +
            0.7152 * result[:, :, 1] +
            0.0722 * result[:, :, 2]
        )

        temp = (
            lum[:, :, None] +
            (
                result -
                lum[:, :, None]
            ) *
            params.region_green_saturation
        )

        result[green] = (
            temp[green]
        )

    if np.any(masks["water"]):

        water = masks["water"]

        lum = (
            0.2126 * result[:, :, 0] +
            0.7152 * result[:, :, 1] +
            0.0722 * result[:, :, 2]
        )

        highlight = np.clip(
            (lum - 0.55) / 0.45,
            0.0,
            1.0,
        )

        result[water] *= (
            1.0 -
            params.region_water_highlight *
            highlight[water, None]
        )

    if np.any(masks["upper_bright"]):

        upper = masks["upper_bright"]

        lum = (
            0.2126 * result[:, :, 0] +
            0.7152 * result[:, :, 1] +
            0.0722 * result[:, :, 2]
        )

        highlight = np.clip(
            (lum - 0.55) / 0.45,
            0.0,
            1.0,
        )

        result[upper] *= (
            1.0 -
            params.region_upper_highlight *
            highlight[upper, None]
        )

    # Background suppression
    background = ~masks["subject"]

    if np.any(background):

        lum = (
            0.2126 * result[:, :, 0] +
            0.7152 * result[:, :, 1] +
            0.0722 * result[:, :, 2]
        )

        temp = (
            lum[:, :, None] +
            (
                result -
                lum[:, :, None]
            ) *
            (
                1.0 -
                params.background_suppression
            )
        )

        result[background] = (
            temp[background]
        )

    return np.clip(
        result,
        0.0,
        1.0,
    )


# ============================================================
# Denoise
# ============================================================

def apply_denoise(
    linear_rgb: np.ndarray,
    strength: float,
) -> np.ndarray:

    if strength <= 0.01:
        return linear_rgb

    srgb = linear_to_srgb(
        linear_rgb
    )

    img8 = np.clip(
        srgb * 255.0,
        0,
        255,
    ).astype(np.uint8)

    sigma_space = (
        2.0 +
        strength * 4.0
    )

    sigma_color = (
        15.0 +
        strength * 35.0
    )

    filtered = cv2.bilateralFilter(
        img8,
        d=5,
        sigmaColor=sigma_color,
        sigmaSpace=sigma_space,
    )

    filtered = (
        filtered.astype(np.float32)
        / 255.0
    )

    return np.clip(
        srgb_to_linear(filtered),
        0.0,
        1.0,
    )


# ============================================================
# Sharpen
# ============================================================

def apply_sharpen(
    linear_rgb: np.ndarray,
    strength: float,
) -> np.ndarray:

    if strength <= 0.01:
        return linear_rgb

    srgb = linear_to_srgb(
        linear_rgb
    )

    lum = (
        0.2126 * srgb[:, :, 0] +
        0.7152 * srgb[:, :, 1] +
        0.0722 * srgb[:, :, 2]
    )

    blur = cv2.GaussianBlur(
        lum,
        (0, 0),
        1.2,
    )

    detail = (
        lum - blur
    )

    sharpened_lum = np.clip(
        lum +
        detail *
        strength *
        1.2,
        0.0,
        1.0,
    )

    ratio = (
        sharpened_lum /
        np.maximum(
            lum,
            1e-5,
        )
    )

    result = (
        srgb *
        ratio[:, :, None]
    )

    result = np.clip(
        result,
        0.0,
        1.0,
    )

    return np.clip(
        srgb_to_linear(result),
        0.0,
        1.0,
    )


# ============================================================
# Development
# ============================================================

def develop(
    linear_rgb: np.ndarray,
    params: DevelopParams,
    masks: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:

    stages = {}

    # 1. Exposure
    x = apply_exposure(
        linear_rgb,
        params.exposure_ev,
    )

    stages["after_exposure"] = x.copy()

    # 2. Contrast
    x = apply_contrast(
        x,
        params.contrast,
    )

    stages["after_contrast"] = x.copy()

    # 3. Saturation
    x = apply_saturation(
        x,
        params.saturation,
    )

    stages["after_saturation"] = x.copy()

    # 4. Tone
    x = apply_tone(
        x,
        params,
    )

    stages["after_tone"] = x.copy()

    # 5. Region processing
    x = apply_region_development(
        x,
        masks,
        params,
    )

    stages["after_region"] = x.copy()

    # 6. Denoise
    x = apply_denoise(
        x,
        params.denoise,
    )

    stages["after_denoise"] = x.copy()

    # 7. Sharpen
    x = apply_sharpen(
        x,
        params.sharpen,
    )

    stages["after_sharpen"] = x.copy()

    stages["final_linear"] = x.copy()

    return stages


# ============================================================
# Automatic search
# ============================================================

def resize_for_search(
    rgb: np.ndarray,
    max_dim: int = 512,
) -> np.ndarray:

    h, w = rgb.shape[:2]

    scale = min(
        1.0,
        max_dim / max(h, w),
    )

    if scale >= 1.0:
        return rgb

    return cv2.resize(
        rgb,
        (
            int(w * scale),
            int(h * scale),
        ),
        interpolation=cv2.INTER_AREA,
    )


def search_score(
    rgb: np.ndarray,
    scene: str,
) -> float:

    stats = calculate_stats(
        rgb
    )

    target = TARGET_MEDIANS.get(
        scene,
        0.18,
    )

    median_error = abs(
        math.log(
            max(stats.median, 1e-5) /
            max(target, 1e-5)
        )
    )

    contrast_target = {
        "portrait": 0.18,
        "night": 0.16,
        "sunset": 0.22,
        "landscape": 0.23,
        "city": 0.22,
        "indoor": 0.18,
        "general": 0.20,
    }.get(
        scene,
        0.20,
    )

    contrast_error = abs(
        stats.contrast -
        contrast_target
    )

    oversaturation = max(
        0.0,
        stats.saturation_ratio -
        0.03,
    )

    score = (
        4.0 *
        stats.highlight_ratio +
        1.0 *
        stats.shadow_ratio +
        0.65 *
        median_error +
        0.50 *
        contrast_error +
        0.80 *
        oversaturation
    )

    return float(score)


def automatic_search(
    linear_rgb: np.ndarray,
    base_params: DevelopParams,
    masks: Dict[str, np.ndarray],
    scene: str,
) -> Tuple[DevelopParams, float, List[Dict]]:

    small = resize_for_search(
        linear_rgb,
        512,
    )

    small_masks = {}

    for name, mask in masks.items():
        small_masks[name] = cv2.resize(
            mask.astype(np.uint8),
            (
                small.shape[1],
                small.shape[0],
            ),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)

    ev_candidates = [
        -0.25,
        -0.12,
        0.0,
        0.12,
        0.25,
    ]

    contrast_candidates = [
        0.98,
        1.00,
        1.03,
        1.06,
    ]

    saturation_candidates = [
        0.97,
        1.00,
        1.03,
    ]

    results = []

    best_score = float("inf")
    best_params = None

    for ev_offset in ev_candidates:

        for contrast in contrast_candidates:

            for saturation in saturation_candidates:

                params = DevelopParams(
                    **asdict(base_params)
                )

                params.exposure_ev += (
                    ev_offset
                )

                params.contrast = contrast
                params.saturation = saturation

                stages = develop(
                    small,
                    params,
                    small_masks,
                )

                output = linear_to_srgb(
                    stages["final_linear"]
                )

                score = search_score(
                    output,
                    scene,
                )

                results.append({
                    "exposure_ev": params.exposure_ev,
                    "contrast": params.contrast,
                    "saturation": params.saturation,
                    "score": score,
                })

                if score < best_score:
                    best_score = score
                    best_params = params

    results.sort(
        key=lambda x: x["score"]
    )

    return (
        best_params,
        float(best_score),
        results[:10],
    )


# ============================================================
# Debug report
# ============================================================

def save_debug_report(
    debug_dir: Path,
    input_path: Path,
    output_path: Path,
    metadata: ExifMetadata,
    profile: CameraProfile,
    raw_info: Dict,
    shooting: ShootingCondition,
    scene: SceneResult,
    subjects: List[SubjectCandidate],
    region_stats: List[RegionStats],
    stage_stats: Dict[str, ImageStats],
    params: DevelopParams,
    search_score_value: float,
    search_results: List[Dict],
):

    report = {
        "version": VERSION,

        "input": str(input_path),
        "output": str(output_path),

        "metadata": asdict(metadata),
        "camera_profile": asdict(profile),
        "raw_info": raw_info,

        "shooting_condition":
            asdict(shooting),

        "scene":
            asdict(scene),

        "subjects": [
            asdict(x)
            for x in subjects
        ],

        "regions": [
            asdict(x)
            for x in region_stats
        ],

        "stage_statistics": {
            name: asdict(stats)
            for name, stats in stage_stats.items()
        },

        "selected_parameters":
            asdict(params),

        "search": {
            "best_score":
                search_score_value,
            "top_candidates":
                search_results,
        },
    }

    path = (
        debug_dir /
        "report.json"
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            report,
            f,
            ensure_ascii=False,
            indent=2,
        )


# ============================================================
# Debug intermediate images
# ============================================================

def save_debug_images(
    debug_dir: Path,
    linear_rgb: np.ndarray,
    stages: Dict[str, np.ndarray],
    labels: Optional[np.ndarray],
    confidence: Optional[np.ndarray],
    saliency: Optional[np.ndarray],
    masks: Optional[Dict[str, np.ndarray]],
):

    # --------------------------------------------------------
    # 01. Original linear RAW output
    # --------------------------------------------------------

    save_rgb(
        debug_dir /
        "01_raw_linear.png",
        linear_rgb,
    )

    # --------------------------------------------------------
    # 02. Linear -> sRGB analysis image
    # --------------------------------------------------------

    analysis_srgb = linear_to_srgb(
        linear_rgb
    )

    save_rgb(
        debug_dir /
        "02_analysis_srgb.png",
        analysis_srgb,
    )

    # --------------------------------------------------------
    # Development stages
    # --------------------------------------------------------

    stage_order = [
        ("after_exposure", "03_after_exposure.png"),
        ("after_contrast", "04_after_contrast.png"),
        ("after_saturation", "05_after_saturation.png"),
        ("after_tone", "06_after_tone.png"),
        ("after_region", "07_after_region.png"),
        ("after_denoise", "08_after_denoise.png"),
        ("after_sharpen", "09_after_sharpen.png"),
        ("final_linear", "10_final_linear.png"),
    ]

    for key, filename in stage_order:

        if key in stages:

            save_rgb(
                debug_dir / filename,
                linear_to_srgb(
                    stages[key]
                ),
            )

    # --------------------------------------------------------
    # Segmentation
    # --------------------------------------------------------

    if labels is not None:

        label_img = (
            labels.astype(
                np.float32
            ) /
            max(
                len(VOC_CLASSES) - 1,
                1,
            )
        )

        save_gray(
            debug_dir /
            "11_segmentation.png",
            label_img,
        )

    if confidence is not None:

        save_gray(
            debug_dir /
            "12_segmentation_confidence.png",
            confidence,
        )

    # --------------------------------------------------------
    # Saliency
    # --------------------------------------------------------

    if saliency is not None:

        save_gray(
            debug_dir /
            "13_saliency.png",
            saliency,
        )

    # --------------------------------------------------------
    # Masks
    # --------------------------------------------------------

    if masks is not None:

        mask_names = [
            "subject",
            "person",
            "animal",
            "vehicle",
            "plant",
            "skin",
            "green",
            "blue",
            "water",
            "upper_bright",
        ]

        for index, name in enumerate(
            mask_names,
            start=14,
        ):

            if name in masks:

                save_gray(
                    debug_dir /
                    f"{index:02d}_mask_{name}.png",
                    masks[name].astype(
                        np.float32
                    ),
                )


# ============================================================
# Developer
# ============================================================

class AutoDeveloper:

    def __init__(
        self,
        device: str = "auto",
        debug: bool = False,
    ):

        self.debug = debug

        self.metadata_reader = (
            MetadataReader()
        )

        self.decoder = RawDecoder()

        self.segmenter = (
            SemanticSegmenter(
                device=device
            )
        )

    def process(
        self,
        input_path: Path,
        output_path: Path,
        debug_root: Optional[Path] = None,
    ):

        print(
            f"\n[INFO] Processing: "
            f"{input_path}"
        )

        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------

        metadata = (
            self.metadata_reader.read(
                input_path
            )
        )

        print(
            f"[INFO] Camera: "
            f"{metadata.make} "
            f"{metadata.model}"
        )

        print(
            f"[INFO] ISO: "
            f"{metadata.iso}"
        )

        print(
            f"[INFO] Exposure: "
            f"{metadata.exposure_time}"
        )

        # ----------------------------------------------------
        # RAW
        # ----------------------------------------------------

        linear_rgb, profile, raw_info = (
            self.decoder.load(
                input_path
            )
        )

        profile.make = (
            metadata.make
        )

        profile.model = (
            metadata.model
        )

        profile.family = (
            detect_camera_family(
                metadata.make,
                metadata.model,
            )
        )

        profile.iso = (
            metadata.iso
        )

        profile.lens = (
            metadata.lens_model
        )

        profile.metadata_source = (
            metadata.source
        )

        # ----------------------------------------------------
        # Debug: RAW linear
        # ----------------------------------------------------

        stage_stats = {}

        if self.debug:

            stage_stats[
                "01_raw_linear"
            ] = print_stats(
                "01 RAW / LibRaw linear RGB",
                linear_rgb,
            )

        # ----------------------------------------------------
        # Analysis image
        #
        # IMPORTANT:
        # gamma=(1,1) output is already treated
        # as linear, so convert it to sRGB only
        # for visual analysis / segmentation.
        # ----------------------------------------------------

        analysis_srgb = linear_to_srgb(
            linear_rgb
        )

        if self.debug:

            stage_stats[
                "02_analysis_srgb"
            ] = print_stats(
                "02 Analysis sRGB",
                analysis_srgb,
            )

        # ----------------------------------------------------
        # Shooting
        # ----------------------------------------------------

        shooting = analyze_shooting(
            metadata
        )

        print(
            f"[INFO] ISO factor: "
            f"{shooting.iso_factor:.3f}"
        )

        print(
            f"[INFO] Estimated noise: "
            f"{shooting.estimated_noise:.3f}"
        )

        # ----------------------------------------------------
        # Semantic segmentation
        # ----------------------------------------------------

        labels, confidence = (
            self.segmenter.predict(
                analysis_srgb
            )
        )

        # ----------------------------------------------------
        # Saliency
        # ----------------------------------------------------

        saliency = compute_saliency(
            analysis_srgb
        )

        # ----------------------------------------------------
        # Subjects
        # ----------------------------------------------------

        subjects = rank_subjects(
            labels,
            confidence,
            saliency,
            analysis_srgb,
        )

        print(
            "[INFO] Subjects:"
        )

        for subject in subjects[:5]:

            print(
                f"  {subject.class_name}: "
                f"score={subject.score:.3f}, "
                f"area={subject.area:.3f}, "
                f"confidence={subject.confidence:.3f}"
            )

        # ----------------------------------------------------
        # Scene
        # ----------------------------------------------------

        analysis_stats = calculate_stats(
            analysis_srgb
        )

        scene = classify_scene(
            analysis_stats,
            shooting,
            subjects,
        )

        print(
            f"[INFO] Scene: "
            f"{scene.scene} "
            f"(confidence={scene.confidence:.3f})"
        )

        # ----------------------------------------------------
        # Base params
        # ----------------------------------------------------

        params = DevelopParams(
            **asdict(
                SCENE_PROFILES[
                    scene.scene
                ]
            )
        )

        exposure_ev = estimate_exposure(
            analysis_stats,
            scene.scene,
        )

        params.exposure_ev += (
            exposure_ev
        )

        print(
            f"[INFO] Estimated EV: "
            f"{exposure_ev:+.3f}"
        )

        # ----------------------------------------------------
        # Region masks
        # ----------------------------------------------------

        masks = create_region_masks(
            analysis_srgb,
            labels,
            subjects,
        )

        regions = calculate_region_stats(
            analysis_srgb,
            masks,
        )

        # ----------------------------------------------------
        # Automatic parameter search
        # ----------------------------------------------------

        print(
            "[INFO] Running parameter search..."
        )

        params, best_score, search_results = (
            automatic_search(
                linear_rgb,
                params,
                masks,
                scene.scene,
            )
        )

        print(
            f"[INFO] Search score: "
            f"{best_score:.6f}"
        )

        print(
            f"[INFO] Selected EV: "
            f"{params.exposure_ev:+.3f}"
        )

        print(
            f"[INFO] Selected contrast: "
            f"{params.contrast:.3f}"
        )

        print(
            f"[INFO] Selected saturation: "
            f"{params.saturation:.3f}"
        )

        # ----------------------------------------------------
        # Final development
        # ----------------------------------------------------

        stages = develop(
            linear_rgb,
            params,
            masks,
        )

        # ----------------------------------------------------
        # Debug stage statistics
        # ----------------------------------------------------

        if self.debug:

            for key, image in stages.items():

                srgb = linear_to_srgb(
                    image
                )

                stage_stats[
                    key
                ] = print_stats(
                    key,
                    srgb,
                )

        # ----------------------------------------------------
        # Final
        # ----------------------------------------------------

        final_linear = (
            stages["final_linear"]
        )

        final_srgb = linear_to_srgb(
            final_linear
        )

        if self.debug:

            stage_stats[
                "final_srgb"
            ] = print_stats(
                "FINAL JPEG sRGB",
                final_srgb,
            )

        # ----------------------------------------------------
        # Save JPEG
        # ----------------------------------------------------

        ensure_dir(
            output_path.parent
        )

        save_rgb(
            output_path,
            final_srgb,
        )

        print(
            f"[INFO] Saved: "
            f"{output_path}"
        )

        # ----------------------------------------------------
        # Debug
        # ----------------------------------------------------

        if self.debug:

            if debug_root is None:

                debug_root = (
                    output_path.parent /
                    "debug"
                )

            debug_dir = (
                debug_root /
                input_path.stem
            )

            ensure_dir(
                debug_dir
            )

            save_debug_images(
                debug_dir,
                linear_rgb,
                stages,
                labels,
                confidence,
                saliency,
                masks,
            )

            save_debug_report(
                debug_dir,
                input_path,
                output_path,
                metadata,
                profile,
                raw_info,
                shooting,
                scene,
                subjects,
                regions,
                stage_stats,
                params,
                best_score,
                search_results,
            )

            print(
                f"[DEBUG] Debug output: "
                f"{debug_dir}"
            )


# ============================================================
# RAW collection
# ============================================================

def collect_raw_files(
    root: Path,
) -> List[Path]:

    files = []

    for path in root.rglob("*"):

        if (
            path.is_file() and
            path.suffix.lower()
            in RAW_EXTENSIONS
        ):
            files.append(path)

    return sorted(
        files
    )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Automatic RAW developer v19"
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

    parser.add_argument(
        "--device",
        choices=[
            "auto",
            "cpu",
            "cuda",
        ],
        default="auto",
        help="Segmentation device",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help=(
            "Enable detailed statistics, "
            "intermediate images and JSON"
        ),
    )

    args = parser.parse_args()

    input_path = args.input

    if input_path.is_file():

        raw_files = [
            input_path
        ]

    elif input_path.is_dir():

        raw_files = collect_raw_files(
            input_path
        )

    else:

        raise FileNotFoundError(
            input_path
        )

    if not raw_files:

        print(
            "[ERROR] No RAW files found."
        )

        return

    print(
        f"[INFO] Found "
        f"{len(raw_files)} RAW files."
    )

    if args.debug:

        print(
            "[INFO] DEBUG MODE ENABLED"
        )

    developer = AutoDeveloper(
        device=args.device,
        debug=args.debug,
    )

    for raw_file in raw_files:

        try:

            if input_path.is_file():

                relative = (
                    raw_file.stem +
                    ".jpg"
                )

            else:

                relative = (
                    raw_file.relative_to(
                        input_path
                    ).with_suffix(
                        ".jpg"
                    )
                )

            output_file = (
                args.output /
                relative
            )

            debug_root = (
                args.output /
                "debug"
            )

            developer.process(
                raw_file,
                output_file,
                debug_root=debug_root,
            )

        except Exception as e:

            print(
                f"[ERROR] Failed: "
                f"{raw_file}"
            )

            print(
                f"        "
                f"{type(e).__name__}: {e}"
            )


if __name__ == "__main__":
    main()