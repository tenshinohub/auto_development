#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Automatic RAW Developer v23

Pipeline
--------
RAW
 ↓
LibRaw / rawpy
 ↓
linear RGB (LibRaw sRGB primaries, gamma=(1,1))
 ↓
analysis sRGB
 ↓
semantic segmentation
 ↓
saliency
 ↓
subject ranking
 ↓
scene classification
 ↓
automatic exposure search
 ↓
contrast / saturation
 ↓
gentle luminance tone
 ↓
local subject/background development
 ↓
luminance denoise
 ↓
luminance sharpen
 ↓
JPEG

Tested design target
--------------------
Python 3.12
Ubuntu 24
rawpy
numpy
opencv-python
Pillow
torch
torchvision

Example
-------
python3 auto_develop_v23.py photos -o developed --device cuda --debug
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import warnings

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import rawpy
from PIL import Image

import torch
import torchvision
from torchvision.models.segmentation import (
    deeplabv3_mobilenet_v3_large,
)


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

PERSON_CLASSES = {"person"}


# ============================================================
# Dataclasses
# ============================================================

@dataclass
class ExifMetadata:
    camera_make: str = ""
    camera_model: str = ""
    iso: float = 100.0
    exposure_time: float = 0.0
    aperture: float = 0.0
    focal_length: float = 0.0
    width: int = 0
    height: int = 0


@dataclass
class CameraProfile:
    make: str = ""
    model: str = ""
    family: str = "generic"


@dataclass
class ImageStats:
    mean: float
    median: float
    p01: float
    p05: float
    p25: float
    p75: float
    p95: float
    p99: float

    shadow_ratio: float
    highlight_ratio: float
    dynamic_range: float

    saturation_ratio: float
    edge_density: float
    contrast: float

    mean_luminance: float

    r_mean: float
    g_mean: float
    b_mean: float

    rg_ratio: float
    gb_ratio: float

    # v23
    warm_ratio: float


@dataclass
class ShootingCondition:
    iso_factor: float
    low_light: bool
    motion_risk: float
    shallow_dof: bool
    wide_angle: bool
    telephoto: bool
    estimated_noise: float


@dataclass
class SubjectCandidate:
    label: str
    class_id: int
    score: float
    area: float
    confidence: float
    center_score: float
    saliency_score: float
    local_contrast: float


@dataclass
class SceneResult:
    scene: str
    confidence: float


@dataclass
class RegionStats:
    subject_median: Optional[float]
    background_median: Optional[float]
    subject_area: float
    background_area: float


@dataclass
class DevelopParams:
    exposure_ev: float
    contrast: float
    saturation: float

    highlight_protection: float
    shadow_lift: float

    subject_exposure: float
    subject_contrast: float

    background_suppression: float

    denoise: float
    sharpen: float

    skin_saturation: float
    green_saturation: float
    water_saturation: float
    upper_brightness: float

    tone_strength: float


# ============================================================
# Utility
# ============================================================

def clamp(x: float, lo: float, hi: float) -> float:
    return float(np.clip(x, lo, hi))


def safe_div(a: float, b: float, eps: float = 1e-8) -> float:
    return float(a / max(abs(b), eps))


def fmt_optional(value: Optional[float]) -> str:
    if value is None:
        return "None"
    return f"{value:.3f}"


def ensure_float32(image: np.ndarray) -> np.ndarray:
    return np.asarray(image, dtype=np.float32)


def normalize_image(image: np.ndarray) -> np.ndarray:
    return np.clip(image.astype(np.float32), 0.0, 1.0)


# ============================================================
# Color conversion
# ============================================================

def linear_to_srgb(rgb: np.ndarray) -> np.ndarray:
    rgb = np.clip(rgb, 0.0, 1.0)

    a = 0.055

    return np.where(
        rgb <= 0.0031308,
        rgb * 12.92,
        (1.0 + a) * np.power(np.maximum(rgb, 0.0), 1.0 / 2.4) - a,
    )


def srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    rgb = np.clip(rgb, 0.0, 1.0)

    return np.where(
        rgb <= 0.04045,
        rgb / 12.92,
        np.power((rgb + 0.055) / 1.055, 2.4),
    )


def luminance(rgb: np.ndarray) -> np.ndarray:
    return (
        rgb[..., 0] * 0.2126
        + rgb[..., 1] * 0.7152
        + rgb[..., 2] * 0.0722
    )


# ============================================================
# RAW
# ============================================================

def raw_to_linear_rgb(raw: rawpy.RawPy) -> np.ndarray:
    """
    Important:
    LibRaw is asked for sRGB primaries with gamma=(1,1).

    gamma=(1,1) means that the returned numerical values are treated
    as linear for this pipeline.

    Do NOT apply srgb_to_linear() here.
    """

    try:
        rgb16 = raw.postprocess(
            use_camera_wb=True,
            use_auto_wb=False,
            output_color=rawpy.ColorSpace.sRGB,
            output_bps=16,
            gamma=(1, 1),
            no_auto_bright=True,
            highlight_mode=rawpy.HighlightMode.Blend,
            half_size=False,
            four_color_rgb=False,
            demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD,
        )

    except Exception as exc:
        warnings.warn(
            f"Camera RGB development failed, falling back to LibRaw sRGB: {exc}"
        )

        rgb16 = raw.postprocess(
            use_camera_wb=True,
            use_auto_wb=False,
            output_color=rawpy.ColorSpace.sRGB,
            output_bps=16,
            gamma=(1, 1),
            no_auto_bright=True,
            highlight_mode=rawpy.HighlightMode.Clip,
            half_size=False,
        )

    rgb = rgb16.astype(np.float32) / 65535.0

    return np.clip(rgb, 0.0, 1.0)


# ============================================================
# Metadata
# ============================================================

def run_exiftool(path: Path) -> dict:
    try:
        result = subprocess.run(
            [
                "exiftool",
                "-j",
                "-ISO",
                "-ExposureTime",
                "-FNumber",
                "-FocalLength",
                "-Make",
                "-Model",
                "-ImageWidth",
                "-ImageHeight",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            return {}

        data = json.loads(result.stdout)

        if not data:
            return {}

        return data[0]

    except Exception:
        return {}


def parse_number(value, default=0.0) -> float:
    if value is None:
        return default

    if isinstance(value, (int, float)):
        return float(value)

    try:
        text = str(value).strip()

        if "/" in text:
            a, b = text.split("/", 1)
            return float(a) / float(b)

        return float(text)

    except Exception:
        return default


def read_metadata(path: Path) -> ExifMetadata:
    data = run_exiftool(path)

    if data:
        return ExifMetadata(
            camera_make=str(data.get("Make", "")),
            camera_model=str(data.get("Model", "")),
            iso=parse_number(data.get("ISO"), 100.0),
            exposure_time=parse_number(
                data.get("ExposureTime"),
                0.0,
            ),
            aperture=parse_number(
                data.get("FNumber"),
                0.0,
            ),
            focal_length=parse_number(
                data.get("FocalLength"),
                0.0,
            ),
            width=int(
                parse_number(
                    data.get("ImageWidth"),
                    0,
                )
            ),
            height=int(
                parse_number(
                    data.get("ImageHeight"),
                    0,
                )
            ),
        )

    try:
        with Image.open(path) as img:
            exif = img.getexif()

            make = str(exif.get(271, ""))
            model = str(exif.get(272, ""))

            iso = parse_number(
                exif.get(34855),
                100.0,
            )

            exposure = parse_number(
                exif.get(33434),
                0.0,
            )

            aperture = parse_number(
                exif.get(33437),
                0.0,
            )

            focal = parse_number(
                exif.get(37386),
                0.0,
            )

            return ExifMetadata(
                camera_make=make,
                camera_model=model,
                iso=iso,
                exposure_time=exposure,
                aperture=aperture,
                focal_length=focal,
                width=img.width,
                height=img.height,
            )

    except Exception:
        return ExifMetadata()


def detect_camera_family(meta: ExifMetadata) -> CameraProfile:
    make = meta.camera_make.lower()
    model = meta.camera_model.lower()

    text = f"{make} {model}"

    if "canon" in text:
        family = "canon"
    elif "nikon" in text:
        family = "nikon"
    elif "sony" in text:
        family = "sony"
    elif "fujifilm" in text or "fuji" in text:
        family = "fujifilm"
    elif "panasonic" in text:
        family = "panasonic"
    elif "olympus" in text or "om system" in text:
        family = "olympus"
    elif "pentax" in text:
        family = "pentax"
    else:
        family = "generic"

    return CameraProfile(
        make=meta.camera_make,
        model=meta.camera_model,
        family=family,
    )


# ============================================================
# Image statistics
# ============================================================

def calculate_warm_ratio(rgb: np.ndarray) -> float:
    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]

    warm = (
        (r > b * 1.08)
        & (r > g * 1.02)
        & (g >= b * 0.95)
    )

    return float(np.mean(warm))


def calculate_stats(rgb: np.ndarray) -> ImageStats:
    rgb = normalize_image(rgb)

    y = luminance(rgb)

    flat = y.reshape(-1)

    mean = float(np.mean(flat))
    median = float(np.median(flat))

    p01, p05, p25, p75, p95, p99 = np.percentile(
        flat,
        [1, 5, 25, 75, 95, 99],
    )

    shadow_ratio = float(np.mean(y < 0.05))
    highlight_ratio = float(np.mean(y > 0.95))

    dynamic_range = math.log10(
        max(float(p95), 1e-6)
        / max(float(p05), 1e-6)
    )

    max_rgb = np.max(rgb, axis=2)
    min_rgb = np.min(rgb, axis=2)

    saturation_ratio = float(
        np.mean((max_rgb - min_rgb) > 0.08)
    )

    gray = np.clip(
        y * 255.0,
        0,
        255,
    ).astype(np.uint8)

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

    magnitude = cv2.magnitude(
        sobel_x,
        sobel_y,
    )

    edge_density = float(
        np.mean(magnitude > 20.0)
    )

    contrast = float(np.std(y))

    r_mean = float(np.mean(rgb[..., 0]))
    g_mean = float(np.mean(rgb[..., 1]))
    b_mean = float(np.mean(rgb[..., 2]))

    rg_ratio = safe_div(r_mean, g_mean)
    gb_ratio = safe_div(g_mean, b_mean)

    warm_ratio = calculate_warm_ratio(rgb)

    return ImageStats(
        mean=mean,
        median=median,
        p01=float(p01),
        p05=float(p05),
        p25=float(p25),
        p75=float(p75),
        p95=float(p95),
        p99=float(p99),
        shadow_ratio=shadow_ratio,
        highlight_ratio=highlight_ratio,
        dynamic_range=dynamic_range,
        saturation_ratio=saturation_ratio,
        edge_density=edge_density,
        contrast=contrast,
        mean_luminance=mean,
        r_mean=r_mean,
        g_mean=g_mean,
        b_mean=b_mean,
        rg_ratio=rg_ratio,
        gb_ratio=gb_ratio,
        warm_ratio=warm_ratio,
    )


def print_stats(
    name: str,
    rgb: np.ndarray,
    stats: Optional[ImageStats] = None,
    display_transform: bool = False,
):
    if stats is None:
        stats = calculate_stats(rgb)

    print(f"\n{name}:")

    print(
        f"  min/max           : "
        f"{np.min(rgb):.6f} / {np.max(rgb):.6f}"
    )

    print(
        f"  mean/median       : "
        f"{stats.mean:.6f} / {stats.median:.6f}"
    )

    print(
        f"  p01/p05           : "
        f"{stats.p01:.6f} / {stats.p05:.6f}"
    )

    print(
        f"  p95/p99           : "
        f"{stats.p95:.6f} / {stats.p99:.6f}"
    )

    print(
        f"  shadow            : "
        f"{stats.shadow_ratio * 100:.3f}%"
    )

    print(
        f"  highlight         : "
        f"{stats.highlight_ratio * 100:.3f}%"
    )

    print(
        f"  dynamic           : "
        f"{stats.dynamic_range:.3f}"
    )

    print(
        f"  saturation        : "
        f"{stats.saturation_ratio * 100:.3f}%"
    )

    print(
        f"  edge              : "
        f"{stats.edge_density:.4f}"
    )

    print(
        f"  contrast          : "
        f"{stats.contrast:.6f}"
    )

    print(
        f"  RGB               : "
        f"{stats.r_mean:.6f}, "
        f"{stats.g_mean:.6f}, "
        f"{stats.b_mean:.6f}"
    )

    print(
        f"  R/G               : "
        f"{stats.rg_ratio:.4f}"
    )

    print(
        f"  G/B               : "
        f"{stats.gb_ratio:.4f}"
    )

    print(
        f"  warm ratio        : "
        f"{stats.warm_ratio * 100:.3f}%"
    )


# ============================================================
# Shooting conditions
# ============================================================

def analyze_shooting(meta: ExifMetadata) -> ShootingCondition:
    iso = max(meta.iso, 100.0)

    iso_factor = math.sqrt(iso / 100.0)
    iso_factor = clamp(
        iso_factor,
        1.0,
        5.0,
    )

    exposure = meta.exposure_time

    low_light = (
        iso >= 800
        or (
            exposure > 0
            and exposure >= 1 / 60
            and iso >= 400
        )
    )

    if exposure <= 0:
        motion_risk = 0.0
    else:
        motion_risk = clamp(
            (1 / 60 - exposure) * 80,
            0.0,
            1.0,
        )

    shallow_dof = (
        meta.aperture > 0
        and meta.aperture <= 2.8
    )

    wide_angle = (
        meta.focal_length > 0
        and meta.focal_length <= 28
    )

    telephoto = (
        meta.focal_length >= 85
    )

    estimated_noise = clamp(
        (iso_factor - 1.0) / 4.0,
        0.0,
        1.0,
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

class Segmenter:
    def __init__(
        self,
        device: str = "auto",
        max_size: int = 768,
    ):
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        elif device == "cuda":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                print(
                    "CUDA is not available. Falling back to CPU."
                )
                self.device = torch.device("cpu")
        else:
            self.device = torch.device("cpu")

        self.max_size = max_size

        print(
            f"Segmentation device: {self.device}"
        )

        try:
            weights = (
                torchvision.models.segmentation
                .DeepLabV3_MobileNet_V3_Large_Weights
                .DEFAULT
            )

            self.model = deeplabv3_mobilenet_v3_large(
                weights=weights
            )

            self.preprocess = weights.transforms()

        except Exception:
            print(
                "Could not load pretrained DeepLabV3 weights."
            )

            self.model = deeplabv3_mobilenet_v3_large(
                weights=None
            )

            self.preprocess = None

        self.model.eval()
        self.model.to(self.device)

    def _resize(
        self,
        image: np.ndarray,
    ) -> tuple[np.ndarray, float, float]:
        h, w = image.shape[:2]

        scale = min(
            1.0,
            self.max_size / max(h, w),
        )

        if scale == 1.0:
            return image, 1.0, 1.0

        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))

        resized = cv2.resize(
            image,
            (new_w, new_h),
            interpolation=cv2.INTER_AREA,
        )

        return resized, scale, scale

    @torch.inference_mode()
    def predict(
        self,
        image: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        image = normalize_image(image)

        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                f"Expected HxWx3 RGB image, got shape={image.shape}"
            )

        resized, _, _ = self._resize(image)

        # torchvision's weights.transforms() in the installed version
        # expects PIL Image or Tensor, not numpy.ndarray.
        # Convert the resized RGB float image [0,1] to PIL RGB here.
        resized_u8 = np.clip(
            resized * 255.0,
            0,
            255,
        ).astype(np.uint8)

        pil_image = Image.fromarray(
            resized_u8,
            mode="RGB",
        )

        if self.preprocess is not None:
            tensor = self.preprocess(
                pil_image
            ).unsqueeze(0)
        else:
            tensor = torch.from_numpy(
                resized.transpose(2, 0, 1)
            ).float().unsqueeze(0)

        tensor = tensor.to(self.device)

        output = self.model(tensor)["out"]

        probabilities = torch.softmax(
            output,
            dim=1,
        )

        confidence, classes = torch.max(
            probabilities,
            dim=1,
        )

        classes = classes[0].cpu().numpy()
        confidence = confidence[0].cpu().numpy()

        original_h, original_w = image.shape[:2]

        classes = cv2.resize(
            classes.astype(np.uint8),
            (original_w, original_h),
            interpolation=cv2.INTER_NEAREST,
        )

        confidence = cv2.resize(
            confidence.astype(np.float32),
            (original_w, original_h),
            interpolation=cv2.INTER_LINEAR,
        )

        return classes, confidence



# ============================================================
# Saliency
# ============================================================

def calculate_saliency(
    image: np.ndarray,
) -> np.ndarray:
    image = normalize_image(image)

    y = luminance(image)

    local = cv2.GaussianBlur(
        y,
        (0, 0),
        9,
    )

    local_contrast = np.abs(
        y - local
    )

    gx = cv2.Sobel(
        y,
        cv2.CV_32F,
        1,
        0,
        ksize=3,
    )

    gy = cv2.Sobel(
        y,
        cv2.CV_32F,
        0,
        1,
        ksize=3,
    )

    edges = cv2.magnitude(
        gx,
        gy,
    )

    edges = edges / (
        np.percentile(edges, 95) + 1e-6
    )

    edges = np.clip(
        edges,
        0,
        1,
    )

    max_rgb = np.max(image, axis=2)
    min_rgb = np.min(image, axis=2)

    saturation = np.clip(
        (max_rgb - min_rgb) * 3.0,
        0,
        1,
    )

    center_y, center_x = np.indices(
        y.shape
    )

    center_x = center_x / max(
        y.shape[1] - 1,
        1,
    )

    center_y = center_y / max(
        y.shape[0] - 1,
        1,
    )

    distance = np.sqrt(
        (center_x - 0.5) ** 2
        + (center_y - 0.5) ** 2
    )

    center = 1.0 - np.clip(
        distance / 0.707,
        0,
        1,
    )

    brightness = np.abs(
        y - np.median(y)
    )

    brightness /= (
        np.percentile(brightness, 95)
        + 1e-6
    )

    brightness = np.clip(
        brightness,
        0,
        1,
    )

    local_contrast /= (
        np.percentile(local_contrast, 95)
        + 1e-6
    )

    local_contrast = np.clip(
        local_contrast,
        0,
        1,
    )

    saliency = (
        local_contrast * 0.30
        + edges * 0.25
        + saturation * 0.15
        + brightness * 0.20
        + center * 0.10
    )

    return np.clip(
        saliency,
        0,
        1,
    )


# ============================================================
# Subject ranking
# ============================================================

def rank_subjects(
    class_map: np.ndarray,
    confidence: np.ndarray,
    saliency: np.ndarray,
    image: np.ndarray,
) -> list[SubjectCandidate]:

    h, w = class_map.shape

    y = luminance(image)

    candidates = []

    for class_id, label in enumerate(VOC_CLASSES):
        if class_id == 0:
            continue

        mask = class_map == class_id

        area = float(np.mean(mask))

        if area < 0.003:
            continue

        conf = float(
            np.mean(
                confidence[mask]
            )
        )

        ys, xs = np.where(mask)

        if len(xs) == 0:
            continue

        cx = float(np.mean(xs) / max(w - 1, 1))
        cy = float(np.mean(ys) / max(h - 1, 1))

        center_score = 1.0 - math.sqrt(
            (cx - 0.5) ** 2
            + (cy - 0.5) ** 2
        ) / 0.707

        center_score = clamp(
            center_score,
            0.0,
            1.0,
        )

        saliency_score = float(
            np.mean(saliency[mask])
        )

        subject_luma = float(
            np.mean(y[mask])
        )

        local_region = cv2.dilate(
            mask.astype(np.uint8),
            np.ones((15, 15), np.uint8),
        ).astype(bool)

        surrounding = (
            local_region
            & ~mask
        )

        if np.any(surrounding):
            surrounding_luma = float(
                np.mean(y[surrounding])
            )

            local_contrast = abs(
                subject_luma
                - surrounding_luma
            )
        else:
            local_contrast = 0.0

        local_contrast = clamp(
            local_contrast * 3.0,
            0.0,
            1.0,
        )

        colorfulness = float(
            np.mean(
                np.max(image[mask], axis=1)
                - np.min(image[mask], axis=1)
            )
        )

        colorfulness = clamp(
            colorfulness * 3.0,
            0.0,
            1.0,
        )

        if label in PERSON_CLASSES:
            prior = 1.15
        elif label in ANIMAL_CLASSES:
            prior = 1.05
        elif label in VEHICLE_CLASSES:
            prior = 1.00
        elif label == "pottedplant":
            prior = 0.90
        elif label == "bottle":
            prior = 0.85
        else:
            prior = 0.80

        score = (
            conf * 0.30
            + math.sqrt(area) * 0.20
            + center_score * 0.15
            + saliency_score * 0.15
            + local_contrast * 0.10
            + colorfulness * 0.10
        ) * prior

        candidates.append(
            SubjectCandidate(
                label=label,
                class_id=class_id,
                score=float(score),
                area=area,
                confidence=conf,
                center_score=center_score,
                saliency_score=saliency_score,
                local_contrast=local_contrast,
            )
        )

    candidates.sort(
        key=lambda x: x.score,
        reverse=True,
    )

    return candidates[:10]


# ============================================================
# Region masks
# ============================================================

def make_region_masks(
    class_map: np.ndarray,
    image: np.ndarray,
    subjects: list[SubjectCandidate],
) -> dict[str, np.ndarray]:

    h, w = class_map.shape

    masks: dict[str, np.ndarray] = {}

    person = class_map == VOC_CLASSES.index(
        "person"
    )

    animal = np.zeros_like(
        person,
        dtype=bool,
    )

    for name in ANIMAL_CLASSES:
        animal |= (
            class_map
            == VOC_CLASSES.index(name)
        )

    vehicle = np.zeros_like(
        person,
        dtype=bool,
    )

    for name in VEHICLE_CLASSES:
        vehicle |= (
            class_map
            == VOC_CLASSES.index(name)
        )

    plant = (
        class_map
        == VOC_CLASSES.index(
            "pottedplant"
        )
    )

    subject = (
        person
        | animal
        | vehicle
        | plant
    )

    # If semantic segmentation found no useful subject,
    # use the strongest candidate.
    if not np.any(subject) and subjects:
        subject = (
            class_map
            == subjects[0].class_id
        )

    rgb8 = np.clip(
        image * 255,
        0,
        255,
    ).astype(np.uint8)

    hsv = cv2.cvtColor(
        rgb8,
        cv2.COLOR_RGB2HSV,
    )

    h_channel = hsv[..., 0]
    s_channel = hsv[..., 1]
    v_channel = hsv[..., 2]

    skin = (
        person
        & (
            (h_channel < 25)
            | (h_channel > 170)
        )
        & (s_channel > 35)
        & (v_channel > 50)
    )

    green = (
        (h_channel >= 30)
        & (h_channel <= 95)
        & (s_channel >= 45)
        & (v_channel >= 30)
    )

    blue = (
        (h_channel >= 80)
        & (h_channel <= 135)
        & (s_channel >= 40)
        & (v_channel >= 40)
    )

    y = luminance(image)

    yy = np.indices(
        (h, w)
    )[0] / max(h - 1, 1)

    upper_bright = (
        (yy < 0.45)
        & (y > np.percentile(y, 75))
    )

    gray = np.clip(
        y * 255,
        0,
        255,
    ).astype(np.uint8)

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

    texture = cv2.magnitude(
        gx,
        gy,
    )

    water = (
        blue
        & (yy > 0.35)
        & (texture < np.percentile(texture, 60))
    )

    background = ~subject

    masks["person"] = person
    masks["animal"] = animal
    masks["vehicle"] = vehicle
    masks["plant"] = plant
    masks["skin"] = skin
    masks["green"] = green
    masks["blue"] = blue
    masks["water"] = water
    masks["upper_bright"] = upper_bright
    masks["subject"] = subject
    masks["background"] = background

    return masks


# ============================================================
# Scene classification
# ============================================================

def classify_scene(
    stats: ImageStats,
    shooting: ShootingCondition,
    subjects: list[SubjectCandidate],
    masks: dict[str, np.ndarray],
) -> SceneResult:

    person_area = float(
        np.mean(masks["person"])
    )

    vehicle_area = float(
        np.mean(masks["vehicle"])
    )

    # Portrait
    if (
        person_area > 0.015
        and stats.median > 0.08
    ):
        confidence = clamp(
            0.55
            + person_area * 1.5
            + (0.15 if shooting.shallow_dof else 0),
            0,
            1,
        )

        return SceneResult(
            scene="portrait",
            confidence=confidence,
        )

    # Night
    if (
        shooting.low_light
        and stats.median < 0.10
    ):
        confidence = clamp(
            0.60
            + shooting.estimated_noise * 0.25,
            0,
            1,
        )

        return SceneResult(
            scene="night",
            confidence=confidence,
        )

    # Sunset
    if (
        stats.warm_ratio > 0.18
        and stats.p95 > 0.55
    ):
        confidence = clamp(
            0.55
            + stats.warm_ratio * 1.5,
            0,
            1,
        )

        return SceneResult(
            scene="sunset",
            confidence=confidence,
        )

    # Landscape
    if (
        shooting.wide_angle
        and stats.edge_density < 0.16
        and stats.dynamic_range > 5.0
    ):
        return SceneResult(
            scene="landscape",
            confidence=0.70,
        )

    # City
    if (
        vehicle_area > 0.01
        and stats.edge_density > 0.10
    ):
        return SceneResult(
            scene="city",
            confidence=0.68,
        )

    # Indoor
    if (
        stats.median < 0.18
        and stats.warm_ratio > 0.10
    ):
        return SceneResult(
            scene="indoor",
            confidence=0.62,
        )

    return SceneResult(
        scene="general",
        confidence=0.50,
    )


# ============================================================
# Scene profiles
# ============================================================

SCENE_PROFILES = {
    "portrait": dict(
        exposure=0.00,
        contrast=1.02,
        saturation=0.99,
        highlight=0.35,
        shadow=0.025,
        subject=0.08,
        subject_contrast=1.02,
        background=0.015,
        denoise=0.28,
        sharpen=0.75,
        skin=0.97,
        green=1.00,
        water=0.06,
        upper=0.10,
        tone=0.30,
    ),

    "night": dict(
        exposure=0.00,
        contrast=1.04,
        saturation=1.02,
        highlight=0.45,
        shadow=0.015,
        subject=0.05,
        subject_contrast=1.02,
        background=0.010,
        denoise=0.30,
        sharpen=0.45,
        skin=0.97,
        green=1.00,
        water=0.08,
        upper=0.14,
        tone=0.30,
    ),

    "sunset": dict(
        exposure=-0.02,
        contrast=1.05,
        saturation=1.04,
        highlight=0.45,
        shadow=0.025,
        subject=0.05,
        subject_contrast=1.02,
        background=0.010,
        denoise=0.20,
        sharpen=0.75,
        skin=0.98,
        green=1.01,
        water=0.08,
        upper=0.16,
        tone=0.35,
    ),

    "landscape": dict(
        exposure=0.02,
        contrast=1.06,
        saturation=1.03,
        highlight=0.35,
        shadow=0.035,
        subject=0.05,
        subject_contrast=1.02,
        background=0.010,
        denoise=0.20,
        sharpen=0.80,
        skin=0.98,
        green=1.02,
        water=0.08,
        upper=0.14,
        tone=0.35,
    ),

    "city": dict(
        exposure=0.02,
        contrast=1.05,
        saturation=1.01,
        highlight=0.40,
        shadow=0.025,
        subject=0.05,
        subject_contrast=1.02,
        background=0.012,
        denoise=0.24,
        sharpen=0.75,
        skin=0.98,
        green=1.01,
        water=0.07,
        upper=0.13,
        tone=0.32,
    ),

    "indoor": dict(
        exposure=0.03,
        contrast=1.03,
        saturation=0.99,
        highlight=0.35,
        shadow=0.030,
        subject=0.05,
        subject_contrast=1.02,
        background=0.012,
        denoise=0.28,
        sharpen=0.60,
        skin=0.97,
        green=1.00,
        water=0.05,
        upper=0.10,
        tone=0.30,
    ),

    "general": dict(
        exposure=0.00,
        contrast=1.04,
        saturation=1.00,
        highlight=0.30,
        shadow=0.025,
        subject=0.04,
        subject_contrast=1.02,
        background=0.010,
        denoise=0.22,
        sharpen=0.75,
        skin=0.98,
        green=1.01,
        water=0.06,
        upper=0.12,
        tone=0.30,
    ),
}


# ============================================================
# Exposure model
# ============================================================

def calculate_exposure_target(
    scene: str,
) -> float:

    targets = {
        "portrait": 0.220,
        "night": 0.100,
        "sunset": 0.160,
        "landscape": 0.220,
        "city": 0.210,
        "indoor": 0.200,
        "general": 0.210,
    }

    target = targets.get(
        scene,
        0.210,
    )

    return clamp(
        target,
        0.085,
        0.235,
    )


def estimate_exposure_ev(
    stats: ImageStats,
    target: float,
) -> float:
    """Estimate exposure while making better use of highlight headroom.

    v23 changes:
    - The previous model was too conservative for normally exposed images
      whose p99 was far below the highlight limits.
    - Headroom bonus is now progressive up to p99 < 0.60.
    - The result is still constrained by the actual highlight levels.
    """

    median_error = (
        target - stats.median
    )

    median_ev = math.log2(
        max(target, 1e-5)
        / max(stats.median, 1e-5)
    )

    highlight_soft = 0.680
    highlight_hard = 0.820

    if stats.p95 > highlight_soft:
        highlight_ev = math.log2(
            highlight_soft
            / max(stats.p95, 1e-5)
        )
    else:
        highlight_ev = 0.0

    if stats.p99 > highlight_hard:
        hard_penalty = -0.35
    else:
        hard_penalty = 0.0

    ev = (
        median_ev * 0.70
        + highlight_ev * 0.30
        + hard_penalty
    )

    # Use available highlight headroom more aggressively.
    if stats.p99 < 0.40:
        ev += 0.25
    elif stats.p99 < 0.45:
        ev += 0.20
    elif stats.p99 < 0.50:
        ev += 0.15
    elif stats.p99 < 0.60:
        ev += 0.08

    if median_error > 0.03:
        ev += 0.05

    return clamp(
        ev,
        -0.75,
        1.00,
    )


# ============================================================
# Exposure / contrast / saturation
# ============================================================

def apply_exposure(
    image: np.ndarray,
    ev: float,
) -> np.ndarray:

    gain = 2.0 ** ev

    return np.clip(
        image * gain,
        0,
        1,
    )


def apply_contrast(
    image: np.ndarray,
    contrast: float,
) -> np.ndarray:

    y = luminance(image)

    new_y = (
        (y - 0.18)
        * contrast
        + 0.18
    )

    ratio = new_y / (
        y + 1e-6
    )

    out = image * ratio[..., None]

    return np.clip(
        out,
        0,
        1,
    )


def apply_saturation(
    image: np.ndarray,
    saturation: float,
) -> np.ndarray:

    y = luminance(image)

    out = (
        y[..., None]
        + (image - y[..., None])
        * saturation
    )

    return np.clip(
        out,
        0,
        1,
    )


# ============================================================
# Tone
# ============================================================

def apply_tone(
    image: np.ndarray,
    strength: float,
    shadow_lift: float,
    highlight_protection: float,
) -> np.ndarray:
    """Apply a gentle, mostly brightness-neutral tone adjustment.

    The old S-curve lowered pixels around the normal midtone range because
    tanh((y - 0.45) * 3) is negative for most ordinary photographs.
    v23 anchors the curve around 0.18 and uses separate shadow/highlight
    controls, so tone no longer makes an otherwise correctly exposed image
    globally darker.
    """

    image = normalize_image(
        image
    )

    y = luminance(
        image
    )

    # Very gentle midtone shaping, anchored at 18% luminance.
    # The anchor subtraction keeps 0.18 approximately unchanged.
    anchor = math.tanh(
        (0.18 - 0.45) * 3.0
    )

    curve = (
        np.tanh(
            (y - 0.45) * 3.0
        )
        - anchor
    )

    s = (
        y
        + strength
        * 0.055
        * curve
    )

    # Shadows: lift only the dark range.
    shadow_mask = np.clip(
        (0.22 - y) / 0.22,
        0,
        1,
    )

    shadow_mask *= np.clip(
        y / 0.22,
        0,
        1,
    )

    s += (
        shadow_lift
        * shadow_mask
    )

    # Highlights: compress only where needed.
    highlight_mask = np.clip(
        (y - 0.68) / 0.32,
        0,
        1,
    )

    s -= (
        highlight_protection
        * 0.055
        * highlight_mask
    )

    s = np.clip(
        s,
        0,
        1,
    )

    ratio = s / (
        y + 1e-6
    )

    out = (
        image
        * ratio[..., None]
    )

    return np.clip(
        out,
        0,
        1,
    )


# ============================================================
# Region processing
# ============================================================

def apply_region_processing(
    image: np.ndarray,
    masks: dict[str, np.ndarray],
    params: DevelopParams,
) -> np.ndarray:

    out = image.copy()

    subject = masks["subject"]
    background = masks["background"]

    # --------------------------------------------------------
    # Subject
    # --------------------------------------------------------

    if np.any(subject):
        subject_img = out[subject]

        subject_y = luminance(
            subject_img
        )

        subject_y2 = (
            (subject_y - 0.18)
            * params.subject_contrast
            + 0.18
        )

        subject_ratio = (
            subject_y2
            / (subject_y + 1e-6)
        )

        subject_img = (
            subject_img
            * subject_ratio[:, None]
        )

        subject_img *= (
            2.0
            ** params.subject_exposure
        )

        out[subject] = np.clip(
            subject_img,
            0,
            1,
        )

    # --------------------------------------------------------
    # Background suppression
    # --------------------------------------------------------

    if np.any(background):
        bg = out[background]

        bg_y = luminance(bg)

        suppression = (
            1.0
            - params.background_suppression
        )

        bg_y2 = (
            bg_y
            * suppression
        )

        ratio = (
            bg_y2
            / (bg_y + 1e-6)
        )

        bg = bg * ratio[:, None]

        out[background] = np.clip(
            bg,
            0,
            1,
        )

    # --------------------------------------------------------
    # Skin saturation
    # --------------------------------------------------------

    skin = masks["skin"]

    if np.any(skin):
        pix = out[skin]
        y = luminance(pix)

        pix = (
            y[:, None]
            + (pix - y[:, None])
            * params.skin_saturation
        )

        out[skin] = np.clip(
            pix,
            0,
            1,
        )

    # --------------------------------------------------------
    # Green saturation
    # --------------------------------------------------------

    green = masks["green"]

    if np.any(green):
        pix = out[green]
        y = luminance(pix)

        pix = (
            y[:, None]
            + (pix - y[:, None])
            * params.green_saturation
        )

        out[green] = np.clip(
            pix,
            0,
            1,
        )

    # --------------------------------------------------------
    # Water
    # --------------------------------------------------------

    water = masks["water"]

    if np.any(water):
        pix = out[water]
        y = luminance(pix)

        pix = (
            y[:, None]
            + (pix - y[:, None])
            * params.water_saturation
        )

        out[water] = np.clip(
            pix,
            0,
            1,
        )

    # --------------------------------------------------------
    # Upper bright area
    # --------------------------------------------------------

    upper = masks["upper_bright"]

    if np.any(upper):
        pix = out[upper]

        y = luminance(pix)

        lift = (
            1.0
            + params.upper_brightness
            * np.clip(
                (0.80 - y) / 0.80,
                0,
                1,
            )
        )

        pix *= lift[:, None]

        out[upper] = np.clip(
            pix,
            0,
            1,
        )

    return np.clip(
        out,
        0,
        1,
    )


# ============================================================
# Denoise
# ============================================================

def calculate_denoise_strength(
    base: float,
    shooting: ShootingCondition,
) -> float:

    iso_component = shooting.estimated_noise

    strength = (
        base
        * (
            0.55
            + 0.20 * iso_component
        )
    )

    return clamp(
        strength,
        0.05,
        0.28,
    )


def apply_denoise(
    image: np.ndarray,
    strength: float,
) -> np.ndarray:

    image = normalize_image(image)

    y = luminance(image)

    y8 = np.clip(
        y * 255.0,
        0,
        255,
    ).astype(np.uint8)

    sigma_color = (
        5.0
        + 14.0 * strength
    )

    sigma_space = (
        1.2
        + 1.8 * strength
    )

    filtered_y8 = cv2.bilateralFilter(
        y8,
        d=5,
        sigmaColor=sigma_color,
        sigmaSpace=sigma_space,
    )

    filtered_y = (
        filtered_y8.astype(np.float32)
        / 255.0
    )

    # Edge-aware blend.
    gx = cv2.Sobel(
        y8,
        cv2.CV_32F,
        1,
        0,
        ksize=3,
    )

    gy = cv2.Sobel(
        y8,
        cv2.CV_32F,
        0,
        1,
        ksize=3,
    )

    edge = cv2.magnitude(
        gx,
        gy,
    )

    edge /= (
        np.percentile(edge, 95)
        + 1e-6
    )

    edge = np.clip(
        edge,
        0,
        1,
    )

    blend = (
        strength
        * (
            1.0
            - 0.70 * edge
        )
    )

    blend = np.clip(
        blend,
        0,
        0.35,
    )

    new_y = (
        y * (1.0 - blend)
        + filtered_y * blend
    )

    ratio = new_y / (
        y + 1e-6
    )

    out = image * ratio[..., None]

    return np.clip(
        out,
        0,
        1,
    )


# ============================================================
# Sharpen
# ============================================================

def apply_sharpen(
    image: np.ndarray,
    strength: float,
) -> np.ndarray:

    if strength <= 0:
        return image

    y = luminance(image)

    blur = cv2.GaussianBlur(
        y,
        (0, 0),
        1.0,
    )

    amount = (
        0.45
        * strength
    )

    sharp_y = (
        y
        + amount
        * (y - blur)
    )

    sharp_y = np.clip(
        sharp_y,
        0,
        1,
    )

    ratio = (
        sharp_y
        / (y + 1e-6)
    )

    out = (
        image
        * ratio[..., None]
    )

    return np.clip(
        out,
        0,
        1,
    )


# ============================================================
# Region statistics
# ============================================================

def calculate_region_stats(
    image: np.ndarray,
    masks: dict[str, np.ndarray],
) -> RegionStats:

    y = luminance(image)

    subject = masks["subject"]
    background = masks["background"]

    subject_median = None
    background_median = None

    if np.any(subject):
        subject_median = float(
            np.median(
                y[subject]
            )
        )

    if np.any(background):
        background_median = float(
            np.median(
                y[background]
            )
        )

    return RegionStats(
        subject_median=subject_median,
        background_median=background_median,
        subject_area=float(
            np.mean(subject)
        ),
        background_area=float(
            np.mean(background)
        ),
    )


# ============================================================
# Automatic parameter search
# ============================================================

def score_candidate(
    image: np.ndarray,
    scene: str,
    target: float,
    subject_mask: Optional[np.ndarray],
) -> float:
    stats = calculate_stats(
        image
    )

    score = 0.0

    # Global exposure.
    score -= abs(
        stats.median
        - target
    ) * 5.0

    # Prefer a useful overall brightness when the image has
    # substantial highlight headroom.
    if stats.p99 < 0.50:
        score += min(
            0.08,
            (0.50 - stats.p99) * 0.20,
        )

    # Highlight protection.
    if stats.p95 > 0.68:
        score -= (
            stats.p95 - 0.68
        ) * 5.0

    if stats.p99 > 0.82:
        score -= (
            stats.p99 - 0.82
        ) * 7.0

    # Avoid crushed shadows.
    if stats.shadow_ratio > 0.12:
        score -= (
            stats.shadow_ratio
            - 0.12
        ) * 2.0

    # Reasonable contrast.
    score -= abs(
        stats.contrast
        - 0.10
    ) * 0.5

    # Saturation.
    if stats.saturation_ratio > 0.75:
        score -= (
            stats.saturation_ratio
            - 0.75
        )

    # Subject.
    if (
        subject_mask is not None
        and np.any(subject_mask)
    ):
        y = luminance(image)

        subject_median = float(
            np.median(
                y[subject_mask]
            )
        )

        subject_target = (
            0.285
            if scene == "portrait"
            else min(
                target + 0.04,
                0.27,
            )
        )

        score -= abs(
            subject_median
            - subject_target
        ) * 2.0

    return float(score)


def automatic_parameter_search(
    image: np.ndarray,
    scene: str,
    profile: dict,
    estimated_ev: float,
    masks: dict[str, np.ndarray],
) -> DevelopParams:

    target = calculate_exposure_target(
        scene
    )

    offsets = [
        -0.25,
        -0.15,
        -0.08,
        0.00,
        0.08,
        0.15,
        0.25,
    ]

    contrasts = [
        0.98,
        1.00,
        1.03,
        1.06,
    ]

    saturations = [
        0.97,
        1.00,
        1.03,
    ]

    subject_mask = masks.get(
        "subject"
    )

    best_score = -float("inf")
    best = None

    for offset in offsets:
        ev = clamp(
            estimated_ev + offset,
            -1.0,
            1.0,
        )

        exposure = apply_exposure(
            image,
            ev,
        )

        for contrast in contrasts:
            contrast_img = apply_contrast(
                exposure,
                contrast,
            )

            for saturation in saturations:
                candidate = apply_saturation(
                    contrast_img,
                    saturation,
                )

                # Score the candidate after the tone stage as well.
                # This prevents the search from selecting an EV that looks
                # correct before tone but becomes dark afterward.
                candidate_tone = apply_tone(
                    candidate,
                    profile["tone"],
                    profile["shadow"],
                    profile["highlight"],
                )

                score = score_candidate(
                    candidate_tone,
                    scene,
                    target,
                    subject_mask,
                )

                if score > best_score:
                    best_score = score

                    best = (
                        ev,
                        contrast,
                        saturation,
                    )

    assert best is not None

    ev, contrast, saturation = best

    print(
        f"Search selected EV {ev:+.3f}, "
        f"contrast {contrast:.3f}, "
        f"saturation {saturation:.3f}"
    )

    return DevelopParams(
        exposure_ev=ev,
        contrast=contrast,
        saturation=saturation,

        highlight_protection=profile[
            "highlight"
        ],

        shadow_lift=profile[
            "shadow"
        ],

        subject_exposure=profile[
            "subject"
        ],

        subject_contrast=profile[
            "subject_contrast"
        ],

        background_suppression=profile[
            "background"
        ],

        denoise=profile[
            "denoise"
        ],

        sharpen=profile[
            "sharpen"
        ],

        skin_saturation=profile[
            "skin"
        ],

        green_saturation=profile[
            "green"
        ],

        water_saturation=profile[
            "water"
        ],

        upper_brightness=profile[
            "upper"
        ],

        tone_strength=profile[
            "tone"
        ],
    )


# ============================================================
# Debug output
# ============================================================

def save_stage(
    debug_dir: Path,
    name: str,
    display_image: np.ndarray,
    stats_image: Optional[np.ndarray] = None,
):
    debug_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    display_image = normalize_image(
        display_image
    )

    png = np.clip(
        display_image * 255.0,
        0,
        255,
    ).astype(np.uint8)

    Image.fromarray(
        png,
        mode="RGB",
    ).save(
        debug_dir / f"{name}.png"
    )

    if stats_image is None:
        stats_image = display_image

    stats = calculate_stats(
        stats_image
    )

    with open(
        debug_dir / f"{name}.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            asdict(stats),
            f,
            indent=2,
            ensure_ascii=False,
        )


# ============================================================
# Auto developer
# ============================================================

class AutoDeveloper:

    def __init__(
        self,
        device: str = "auto",
        debug: bool = False,
    ):
        self.debug = debug

        self.segmenter = Segmenter(
            device=device
        )

    def process_file(
        self,
        raw_path: Path,
        output_path: Path,
    ):

        print()
        print("=" * 72)
        print(
            f"Processing: {raw_path}"
        )
        print("=" * 72)

        meta = read_metadata(
            raw_path
        )

        camera = detect_camera_family(
            meta
        )

        shooting = analyze_shooting(
            meta
        )

        print(
            f"Camera: "
            f"{meta.camera_make} "
            f"{meta.camera_model}"
        )

        print(
            f"ISO: {meta.iso:g}"
        )

        print(
            f"Exposure: "
            f"{meta.exposure_time:g}"
        )

        print(
            f"Aperture: "
            f"{meta.aperture:g}"
        )

        print(
            f"Focal length: "
            f"{meta.focal_length:g}"
        )

        print(
            f"Camera family: "
            f"{camera.family}"
        )

        # ----------------------------------------------------
        # RAW
        # ----------------------------------------------------

        with rawpy.imread(
            str(raw_path)
        ) as raw:

            linear_rgb = raw_to_linear_rgb(
                raw
            )

        linear_rgb = normalize_image(
            linear_rgb
        )

        # ----------------------------------------------------
        # Debug: true linear stats
        # ----------------------------------------------------

        analysis_srgb = linear_to_srgb(
            linear_rgb
        )

        if self.debug:
            print_stats(
                "01_raw_linear",
                linear_rgb,
            )

            save_stage(
                self.debug_dir,
                "01_raw_linear",
                analysis_srgb,
                stats_image=linear_rgb,
            )

            print_stats(
                "02_analysis_srgb",
                analysis_srgb,
            )

            save_stage(
                self.debug_dir,
                "02_analysis_srgb",
                analysis_srgb,
                stats_image=analysis_srgb,
            )

        else:
            print_stats(
                "RAW / LibRaw linear RGB",
                linear_rgb,
            )

        # ----------------------------------------------------
        # Semantic segmentation
        # ----------------------------------------------------

        class_map, confidence = (
            self.segmenter.predict(
                analysis_srgb
            )
        )

        saliency = calculate_saliency(
            analysis_srgb
        )

        subjects = rank_subjects(
            class_map,
            confidence,
            saliency,
            analysis_srgb,
        )

        masks = make_region_masks(
            class_map,
            analysis_srgb,
            subjects,
        )

        # ----------------------------------------------------
        # Subjects
        # ----------------------------------------------------

        print()

        if subjects:
            for subject in subjects[:5]:
                print(
                    f"Subject "
                    f"{subject.label}: "
                    f"score {subject.score:.3f} "
                    f"area {subject.area:.3f} "
                    f"conf {subject.confidence:.3f}"
                )
        else:
            print(
                "Subject: none"
            )

        # ----------------------------------------------------
        # Scene
        # ----------------------------------------------------

        raw_stats = calculate_stats(
            analysis_srgb
        )

        scene_result = classify_scene(
            raw_stats,
            shooting,
            subjects,
            masks,
        )

        print(
            f"Scene: "
            f"{scene_result.scene} "
            f"confidence "
            f"{scene_result.confidence:.3f}"
        )

        scene = scene_result.scene

        profile = SCENE_PROFILES[
            scene
        ]

        # ----------------------------------------------------
        # Exposure
        # ----------------------------------------------------

        target = calculate_exposure_target(
            scene
        )

        estimated_ev = estimate_exposure_ev(
            raw_stats,
            target,
        )

        print(
            f"Global target      : "
            f"{target:.3f}"
        )

        if scene == "portrait":
            subject_target = 0.285
        elif scene == "night":
            subject_target = 0.16
        elif scene == "indoor":
            subject_target = 0.24
        else:
            subject_target = min(
                target + 0.04,
                0.27,
            )

        print(
            f"Subject target     : "
            f"{subject_target:.3f}"
        )

        print(
            f"Highlight soft     : "
            f"{0.680:.3f}"
        )

        print(
            f"Highlight hard     : "
            f"{0.820:.3f}"
        )

        print(
            f"Estimated EV       : "
            f"{estimated_ev:+.3f}"
        )

        # ----------------------------------------------------
        # Search
        # ----------------------------------------------------

        params = automatic_parameter_search(
            analysis_srgb,
            scene,
            profile,
            estimated_ev,
            masks,
        )

        denoise_strength = (
            calculate_denoise_strength(
                params.denoise,
                shooting,
            )
        )

        print(
            f"Denoise            : "
            f"{denoise_strength:.3f}"
        )

        print(
            f"Tone               : "
            f"{params.tone_strength:.3f}"
        )

        print(
            f"Shadow lift        : "
            f"{params.shadow_lift:.3f}"
        )

        print(
            f"Highlight protect  : "
            f"{params.highlight_protection:.3f}"
        )

        # ----------------------------------------------------
        # Stage 03: exposure
        # ----------------------------------------------------

        current = apply_exposure(
            analysis_srgb,
            params.exposure_ev,
        )

        if self.debug:
            save_stage(
                self.debug_dir,
                "03_after_exposure",
                current,
            )

            print_stats(
                "03_after_exposure",
                current,
            )

        # ----------------------------------------------------
        # Stage 04: contrast
        # ----------------------------------------------------

        current = apply_contrast(
            current,
            params.contrast,
        )

        if self.debug:
            save_stage(
                self.debug_dir,
                "04_after_contrast",
                current,
            )

            print_stats(
                "04_after_contrast",
                current,
            )

        # ----------------------------------------------------
        # Stage 05: saturation
        # ----------------------------------------------------

        current = apply_saturation(
            current,
            params.saturation,
        )

        if self.debug:
            save_stage(
                self.debug_dir,
                "05_after_saturation",
                current,
            )

            print_stats(
                "05_after_saturation",
                current,
            )

        # ----------------------------------------------------
        # Stage 06: tone
        # ----------------------------------------------------

        current = apply_tone(
            current,
            params.tone_strength,
            params.shadow_lift,
            params.highlight_protection,
        )

        if self.debug:
            save_stage(
                self.debug_dir,
                "06_after_tone",
                current,
            )

            print_stats(
                "06_after_tone",
                current,
            )

        # ----------------------------------------------------
        # Region stats before local processing
        # ----------------------------------------------------

        region_before = (
            calculate_region_stats(
                current,
                masks,
            )
        )

        print()
        print(
            "Region before local:"
        )

        print(
            f"  subject median    : "
            f"{fmt_optional(region_before.subject_median)}"
        )

        print(
            f"  background median : "
            f"{fmt_optional(region_before.background_median)}"
        )

        # ----------------------------------------------------
        # Stage 07: local region
        # ----------------------------------------------------

        current = apply_region_processing(
            current,
            masks,
            params,
        )

        region_after = (
            calculate_region_stats(
                current,
                masks,
            )
        )

        print(
            "Region after local:"
        )

        print(
            f"  subject median    : "
            f"{fmt_optional(region_after.subject_median)}"
        )

        print(
            f"  background median : "
            f"{fmt_optional(region_after.background_median)}"
        )

        if self.debug:
            save_stage(
                self.debug_dir,
                "07_after_region",
                current,
            )

            print_stats(
                "07_after_region",
                current,
            )

        # ----------------------------------------------------
        # Stage 08: denoise
        # ----------------------------------------------------

        current = apply_denoise(
            current,
            denoise_strength,
        )

        if self.debug:
            save_stage(
                self.debug_dir,
                "08_after_denoise",
                current,
            )

            print_stats(
                "08_after_denoise",
                current,
            )

        # ----------------------------------------------------
        # Stage 09: sharpen
        # ----------------------------------------------------

        current = apply_sharpen(
            current,
            params.sharpen,
        )

        # ----------------------------------------------------
        # Final brightness feedback
        # ----------------------------------------------------
        # Tone/local processing can change the global median.  Apply only
        # a small corrective EV so the final image does not remain dark.
        final_stats_before_feedback = calculate_stats(
            current
        )

        final_target = calculate_exposure_target(
            scene
        )

        final_error = (
            final_target
            - final_stats_before_feedback.median
        )

        if final_error > 0.015:
            correction_ev = clamp(
                math.log2(
                    max(final_target, 1e-5)
                    / max(
                        final_stats_before_feedback.median,
                        1e-5,
                    )
                ) * 0.55,
                0.0,
                0.30,
            )

            # Never use the feedback to push an already bright/highlighted
            # image upward.
            if (
                final_stats_before_feedback.p99 < 0.65
                and correction_ev > 0.0
            ):
                current = apply_exposure(
                    current,
                    correction_ev,
                )

                print(
                    f"Final brightness correction: "
                    f"{correction_ev:+.3f} EV"
                )
            else:
                print(
                    "Final brightness correction: "
                    "skipped (highlight headroom)"
                )
        else:
            print(
                "Final brightness correction: "
                "not needed"
            )

        if self.debug:
            save_stage(
                self.debug_dir,
                "09_after_sharpen",
                current,
            )

            print_stats(
                "09_after_sharpen",
                current,
            )

        # ----------------------------------------------------
        # Final
        # ----------------------------------------------------

        final_srgb = np.clip(
            current,
            0,
            1,
        )

        final_linear = srgb_to_linear(
            final_srgb
        )

        if self.debug:

            # Important:
            # display image is sRGB,
            # stats image is true linear RGB.
            save_stage(
                self.debug_dir,
                "10_final_linear",
                final_srgb,
                stats_image=final_linear,
            )

            print_stats(
                "10_final_linear",
                final_linear,
            )

            save_stage(
                self.debug_dir,
                "11_final_srgb",
                final_srgb,
            )

            print_stats(
                "11_final_srgb",
                final_srgb,
            )

        final_u8 = np.clip(
            final_srgb * 255.0,
            0,
            255,
        ).astype(np.uint8)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        Image.fromarray(
            final_u8,
            mode="RGB",
        ).save(
            output_path,
            quality=95,
            subsampling=0,
        )

        final_region = (
            calculate_region_stats(
                final_srgb,
                masks,
            )
        )

        print()
        print(
            "Final region:"
        )

        print(
            f"  subject median    : "
            f"{fmt_optional(final_region.subject_median)}"
        )

        print(
            f"  background median : "
            f"{fmt_optional(final_region.background_median)}"
        )

        print(
            f"Saved: {output_path}"
        )

    # --------------------------------------------------------
    # Debug directory
    # --------------------------------------------------------

    @property
    def debug_dir(self) -> Path:
        return self._debug_dir

    @debug_dir.setter
    def debug_dir(self, value: Path):
        self._debug_dir = value


# ============================================================
# RAW collection
# ============================================================

def collect_raw_files(
    input_path: Path,
) -> list[Path]:

    if input_path.is_file():

        if (
            input_path.suffix.lower()
            in RAW_EXTENSIONS
        ):
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

    files.sort()

    return files


# ============================================================
# CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Automatic RAW developer v23"
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
        default=Path("developed"),
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
        help="Save intermediate images/statistics",
    )

    args = parser.parse_args()

    raw_files = collect_raw_files(
        args.input
    )

    if not raw_files:
        print(
            "No RAW files found."
        )
        return 1

    developer = AutoDeveloper(
        device=args.device,
        debug=args.debug,
    )

    for raw_path in raw_files:

        if args.input.is_file():
            relative = raw_path.name
        else:
            try:
                relative = raw_path.relative_to(
                    args.input
                )
            except ValueError:
                relative = raw_path.name

        output_name = Path(
            relative
        ).with_suffix(".jpg")

        output_path = (
            args.output
            / output_name
        )

        if args.debug:
            developer.debug_dir = (
                args.output
                / Path(relative).with_suffix("")
                / "debug"
            )

        try:
            developer.process_file(
                raw_path,
                output_path,
            )

        except Exception as exc:
            print()
            print(
                f"[ERROR] {raw_path}"
            )
            print(
                f"{type(exc).__name__}: "
                f"{exc}"
            )
            import traceback
            traceback.print_exc()

    print()
    print("=" * 72)
    print("Finished.")
    print("=" * 72)

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )