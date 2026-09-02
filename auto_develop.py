#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Automatic RAW Developer v14

Automatic development based on:

    RAW
      |
      +-- Camera metadata
      |     +-- Make / Model
      |     +-- Lens
      |     +-- ISO
      |     +-- Shutter
      |     +-- Aperture
      |     +-- Focal length
      |     +-- WB / Color temperature
      |
      +-- RAW characteristics
      |     +-- Black level
      |     +-- White level
      |     +-- Camera WB
      |     +-- RGB->XYZ matrix
      |
      +-- Image analysis
      |
      +-- Semantic segmentation
      |
      +-- Saliency
      |
      +-- Scene classification
      |
      +-- Shooting-condition analysis
      |
      +-- Automatic parameter search
      |
      +-- Local subject development
      |
      +-- Tone / CLAHE
      |
      +-- Denoise
      |
      +-- Sharpen
      |
      +-- JPEG

Requirements
------------

    pip install rawpy pillow numpy opencv-python torch torchvision

Optional but strongly recommended:

    sudo apt install exiftool

Usage
-----

    python3 auto_develop_v14.py ./RAW \
        -o ./output \
        --device cuda

"""


from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
import sys

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ExifTags

import rawpy


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
    "person",
    "bird",
    "bicycle",
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

    make: str
    model: str
    iso: int

    black_level: np.ndarray
    white_level: float
    white_level_per_channel: np.ndarray

    camera_wb: np.ndarray

    color_matrix: np.ndarray
    rgb_xyz_matrix: np.ndarray

    color_desc: str
    num_colors: int

    raw_width: int
    raw_height: int
    raw_pattern: object

    lens_make: str
    lens_model: str

    exposure_time: Optional[float]
    f_number: Optional[float]
    focal_length: Optional[float]

    white_balance: str
    color_temperature: Optional[float]
    color_space: str

    metadata_source: str
    libraw_version: str


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
    saturation_ratio: float

    edge_density: float
    warm_ratio: float

    contrast: float


@dataclass
class ShootingCondition:

    iso_factor: float
    shutter_factor: float
    aperture_factor: float
    focal_factor: float

    low_light: float
    motion_risk: float
    shallow_dof: float

    wide_angle: float
    telephoto: float

    estimated_noise: float


@dataclass
class SubjectCandidate:

    class_name: str

    confidence: float
    area: float
    center_score: float
    saliency_score: float
    local_contrast: float
    colorfulness: float

    score: float


@dataclass
class SceneProfile:

    name: str

    exposure: float
    contrast: float
    saturation: float

    highlight: float
    shadow: float

    subject_strength: float
    background_suppression: float

    denoise: float
    sharpen: float


# ============================================================
# Utility
# ============================================================

def clamp01(x):

    return np.clip(
        x,
        0.0,
        1.0,
    )


def normalize_map(x):

    x = x.astype(np.float32)

    mn = float(np.min(x))
    mx = float(np.max(x))

    if mx - mn < 1e-8:
        return np.zeros_like(x)

    return (
        (x - mn)
        /
        (mx - mn)
    )


def luminance(img):

    return (
        img[..., 0] * 0.2126
        +
        img[..., 1] * 0.7152
        +
        img[..., 2] * 0.0722
    )


def safe_float(value):

    if value is None:
        return None

    if isinstance(
        value,
        (int, float),
    ):
        return float(value)

    text = str(value).strip()

    if not text:
        return None

    try:
        return float(text)

    except ValueError:
        pass

    match = re.match(
        r"^\s*"
        r"(\d+(?:\.\d+)?)"
        r"\s*/\s*"
        r"(\d+(?:\.\d+)?)"
        r"\s*$",
        text,
    )

    if match:

        numerator = float(
            match.group(1)
        )

        denominator = float(
            match.group(2)
        )

        if denominator != 0:
            return (
                numerator
                /
                denominator
            )

    return None


def clean_text(value):

    if value is None:
        return ""

    if isinstance(
        value,
        (list, tuple),
    ):

        return " ".join(
            str(v)
            for v in value
        )

    return str(value).strip()


def first_nonempty(*values):

    for value in values:

        if value is None:
            continue

        if isinstance(
            value,
            str,
        ):

            if value.strip():
                return value.strip()

        else:
            return value

    return ""


def get_array(
    value,
    dtype=np.float32,
):

    try:

        return np.asarray(
            value,
            dtype=dtype,
        )

    except Exception:

        return np.array(
            [],
            dtype=dtype,
        )


# ============================================================
# ExifTool
# ============================================================

def exiftool_available():

    return (
        shutil.which(
            "exiftool"
        )
        is not None
    )


def read_exiftool(path):

    if not exiftool_available():

        return ExifMetadata(
            source="not_available"
        )

    tags = [
        "Make",
        "Model",
        "CameraModelName",
        "UniqueCameraModel",

        "LensMake",
        "LensModel",

        "ISO",
        "ExposureTime",
        "FNumber",
        "FocalLength",

        "WhiteBalance",
        "ColorTemperature",
        "ColorSpace",
    ]

    command = [
        "exiftool",
        "-j",
        "-n",
    ]

    for tag in tags:

        command.append(
            f"-{tag}"
        )

    command.append(
        str(path)
    )

    try:

        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=20,
            check=False,
        )

    except Exception:

        return ExifMetadata(
            source="exiftool_error"
        )

    if result.returncode != 0:

        return ExifMetadata(
            source="exiftool_error"
        )

    try:

        data = json.loads(
            result.stdout
        )

        if not data:

            return ExifMetadata(
                source="exiftool_empty"
            )

        d = data[0]

    except Exception:

        return ExifMetadata(
            source="exiftool_parse_error"
        )

    make = clean_text(
        d.get("Make")
    )

    model = first_nonempty(
        clean_text(
            d.get("Model")
        ),
        clean_text(
            d.get("CameraModelName")
        ),
        clean_text(
            d.get("UniqueCameraModel")
        ),
    )

    lens_make = clean_text(
        d.get("LensMake")
    )

    lens_model = clean_text(
        d.get("LensModel")
    )

    return ExifMetadata(

        make=make,

        model=model,

        lens_make=lens_make,

        lens_model=lens_model,

        iso=safe_float(
            d.get("ISO")
        ),

        exposure_time=safe_float(
            d.get("ExposureTime")
        ),

        f_number=safe_float(
            d.get("FNumber")
        ),

        focal_length=safe_float(
            d.get("FocalLength")
        ),

        white_balance=clean_text(
            d.get("WhiteBalance")
        ),

        color_temperature=safe_float(
            d.get("ColorTemperature")
        ),

        color_space=clean_text(
            d.get("ColorSpace")
        ),

        source="exiftool",
    )


# ============================================================
# Pillow fallback
# ============================================================

def read_pillow_exif(path):

    try:

        with Image.open(path) as img:

            exif = img.getexif()

            if not exif:

                return ExifMetadata(
                    source="pillow_empty"
                )

            decoded = {}

            for key, value in exif.items():

                name = ExifTags.TAGS.get(
                    key,
                    str(key),
                )

                decoded[name] = value

            return ExifMetadata(

                make=clean_text(
                    decoded.get("Make")
                ),

                model=clean_text(
                    decoded.get("Model")
                ),

                lens_model=clean_text(
                    decoded.get("LensModel")
                ),

                iso=safe_float(
                    decoded.get(
                        "ISOSpeedRatings"
                    )
                ),

                exposure_time=safe_float(
                    decoded.get(
                        "ExposureTime"
                    )
                ),

                f_number=safe_float(
                    decoded.get(
                        "FNumber"
                    )
                ),

                focal_length=safe_float(
                    decoded.get(
                        "FocalLength"
                    )
                ),

                source="pillow",
            )

    except Exception:

        return ExifMetadata(
            source="pillow_error"
        )


# ============================================================
# Metadata merge
# ============================================================

def merge_metadata(
    exiftool,
    pillow,
):

    return ExifMetadata(

        make=first_nonempty(
            exiftool.make,
            pillow.make,
        ),

        model=first_nonempty(
            exiftool.model,
            pillow.model,
        ),

        lens_make=first_nonempty(
            exiftool.lens_make,
            pillow.lens_make,
        ),

        lens_model=first_nonempty(
            exiftool.lens_model,
            pillow.lens_model,
        ),

        iso=(
            exiftool.iso
            if exiftool.iso is not None
            else pillow.iso
        ),

        exposure_time=(
            exiftool.exposure_time
            if exiftool.exposure_time is not None
            else pillow.exposure_time
        ),

        f_number=(
            exiftool.f_number
            if exiftool.f_number is not None
            else pillow.f_number
        ),

        focal_length=(
            exiftool.focal_length
            if exiftool.focal_length is not None
            else pillow.focal_length
        ),

        white_balance=first_nonempty(
            exiftool.white_balance,
            pillow.white_balance,
        ),

        color_temperature=(
            exiftool.color_temperature
            if exiftool.color_temperature is not None
            else pillow.color_temperature
        ),

        color_space=first_nonempty(
            exiftool.color_space,
            pillow.color_space,
        ),

        source=(
            "exiftool+pillow"
            if exiftool.source == "exiftool"
            else pillow.source
        ),
    )


# ============================================================
# Camera family
# ============================================================

def normalize_camera_string(text):

    text = clean_text(
        text
    ).lower()

    replacements = [
        ("om system", "olympus"),
        ("olympus imaging", "olympus"),
        ("panasonic corporation", "panasonic"),
        ("fujifilm", "fujifilm"),
        ("fuji photo film", "fujifilm"),
        ("sony corporation", "sony"),
        ("canon inc.", "canon"),
        ("nikon corporation", "nikon"),
        ("ricoh imaging", "ricoh"),
    ]

    for old, new in replacements:

        text = text.replace(
            old,
            new,
        )

    return text.strip()


def camera_family(
    make,
    model,
):

    text = normalize_camera_string(
        f"{make} {model}"
    )

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

    if "olympus" in text:
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
# RAW camera profile
# ============================================================

def build_camera_profile(
    raw,
    metadata,
):

    iso = (
        int(round(metadata.iso))
        if metadata.iso is not None
        else 100
    )

    try:

        black_level = get_array(
            raw.black_level_per_channel
        )

    except Exception:

        black_level = np.array(
            [0, 0, 0],
            dtype=np.float32,
        )

    try:

        white_level_per_channel = get_array(
            raw.camera_white_level_per_channel
        )

    except Exception:

        white_level_per_channel = np.array(
            [],
            dtype=np.float32,
        )

    try:

        white_level = float(
            raw.white_level
        )

    except Exception:

        if white_level_per_channel.size:

            white_level = float(
                np.max(
                    white_level_per_channel
                )
            )

        else:

            white_level = 65535.0

    try:

        camera_wb = get_array(
            raw.camera_whitebalance
        )

    except Exception:

        camera_wb = np.array(
            [],
            dtype=np.float32,
        )

    try:

        color_matrix = get_array(
            raw.color_matrix
        )

    except Exception:

        color_matrix = np.empty(
            (0, 0),
            dtype=np.float32,
        )

    try:

        rgb_xyz_matrix = get_array(
            raw.rgb_xyz_matrix
        )

    except Exception:

        rgb_xyz_matrix = np.empty(
            (0, 0),
            dtype=np.float32,
        )

    try:

        color_desc = clean_text(
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

        raw_pattern = raw.raw_pattern

    except Exception:

        raw_pattern = None

    try:

        raw_width = int(
            raw.sizes.raw_width
        )

        raw_height = int(
            raw.sizes.raw_height
        )

    except Exception:

        try:

            h, w = (
                raw.raw_image_visible.shape
            )

            raw_width = int(w)
            raw_height = int(h)

        except Exception:

            raw_width = 0
            raw_height = 0

    try:

        libraw_version = str(
            rawpy.libraw_version
        )

    except Exception:

        libraw_version = "unknown"

    return CameraProfile(

        make=metadata.make or "UNKNOWN",

        model=metadata.model or "UNKNOWN",

        iso=iso,

        black_level=black_level,

        white_level=white_level,

        white_level_per_channel=
            white_level_per_channel,

        camera_wb=camera_wb,

        color_matrix=color_matrix,

        rgb_xyz_matrix=rgb_xyz_matrix,

        color_desc=color_desc,

        num_colors=num_colors,

        raw_width=raw_width,

        raw_height=raw_height,

        raw_pattern=raw_pattern,

        lens_make=metadata.lens_make,

        lens_model=metadata.lens_model,

        exposure_time=metadata.exposure_time,

        f_number=metadata.f_number,

        focal_length=metadata.focal_length,

        white_balance=metadata.white_balance,

        color_temperature=
            metadata.color_temperature,

        color_space=metadata.color_space,

        metadata_source=metadata.source,

        libraw_version=libraw_version,
    )


# ============================================================
# Camera logging
# ============================================================

def print_camera_profile(profile):

    family = camera_family(
        profile.make,
        profile.model,
    )

    print()

    print(
        f"[INFO] RAW camera: "
        f"{profile.make} "
        f"{profile.model}"
    )

    print(
        f"[INFO] Camera family: "
        f"{family}"
    )

    print(
        f"[INFO] Metadata source: "
        f"{profile.metadata_source}"
    )

    if (
        profile.lens_make
        or profile.lens_model
    ):

        print(
            f"[INFO] Lens: "
            f"{profile.lens_make} "
            f"{profile.lens_model}".strip()
        )

    print(
        f"[INFO] ISO: "
        f"{profile.iso}"
    )

    if profile.exposure_time is not None:

        print(
            "[INFO] Exposure time: "
            f"{profile.exposure_time:.6f}s"
        )

    if profile.f_number is not None:

        print(
            "[INFO] F-number: "
            f"f/{profile.f_number:.1f}"
        )

    if profile.focal_length is not None:

        print(
            "[INFO] Focal length: "
            f"{profile.focal_length:.1f}mm"
        )

    if profile.white_balance:

        print(
            "[INFO] White balance: "
            f"{profile.white_balance}"
        )

    if (
        profile.color_temperature
        is not None
    ):

        print(
            "[INFO] Color temperature: "
            f"{profile.color_temperature:.0f}K"
        )

    print(
        f"[INFO] RAW size: "
        f"{profile.raw_width} "
        f"x "
        f"{profile.raw_height}"
    )

    print(
        f"[INFO] Number of colors: "
        f"{profile.num_colors}"
    )

    print(
        f"[INFO] Color description: "
        f"{profile.color_desc}"
    )

    print(
        f"[INFO] LibRaw: "
        f"{profile.libraw_version}"
    )

    if profile.black_level.size:

        print(
            f"[INFO] Black level: "
            f"{profile.black_level}"
        )

    print(
        f"[INFO] White level: "
        f"{profile.white_level:.1f}"
    )

    if profile.camera_wb.size:

        print(
            f"[INFO] Camera WB: "
            f"{profile.camera_wb}"
        )

    if profile.color_matrix.size:

        print(
            "[INFO] Color Matrix:"
        )

        print(
            profile.color_matrix
        )

    if profile.rgb_xyz_matrix.size:

        print(
            "[INFO] RGB -> XYZ Matrix:"
        )

        print(
            profile.rgb_xyz_matrix
        )


# ============================================================
# RAW -> camera RGB
# ============================================================

def load_linear_camera_rgb(path):

    raw = rawpy.imread(
        str(path)
    )

    rgb = raw.postprocess(

        use_camera_wb=True,

        use_auto_wb=False,

        output_color=rawpy.ColorSpace.raw,

        output_bps=16,

        gamma=(1, 1),

        no_auto_bright=True,

        highlight_mode=
            rawpy.HighlightMode.Blend,

        half_size=False,

        four_color_rgb=False,

        demosaic_algorithm=
            rawpy.DemosaicAlgorithm.AHD,
    )

    rgb = rgb.astype(
        np.float32
    )

    max_value = float(
        np.max(rgb)
    )

    if max_value > 1.0:

        rgb /= 65535.0

    rgb = clamp01(rgb)

    return raw, rgb


# ============================================================
# Camera RGB -> sRGB
# ============================================================

def camera_rgb_to_srgb(
    rgb,
    profile,
):

    matrix = profile.rgb_xyz_matrix

    if matrix.size == 0:

        print(
            "[WARN] RGB->XYZ matrix "
            "unavailable."
        )

        return clamp01(rgb)

    matrix = np.asarray(
        matrix,
        dtype=np.float32,
    )

    if matrix.ndim != 2:
        return clamp01(rgb)

    if (
        matrix.shape[0] != 3
        or
        matrix.shape[1] < 3
    ):

        return clamp01(rgb)

    matrix = matrix[:, :3]

    flat = rgb.reshape(
        -1,
        3,
    )

    xyz = (
        flat
        @
        matrix.T
    )

    xyz_to_srgb = np.array(
        [
            [
                3.2406,
                -1.5372,
                -0.4986,
            ],
            [
                -0.9689,
                1.8758,
                0.0415,
            ],
            [
                0.0557,
                -0.2040,
                1.0570,
            ],
        ],
        dtype=np.float32,
    )

    srgb = (
        xyz
        @
        xyz_to_srgb.T
    )

    srgb = srgb.reshape(
        rgb.shape
    )

    srgb = np.maximum(
        srgb,
        0.0,
    )

    p99 = float(
        np.percentile(
            srgb,
            99.5,
        )
    )

    if p99 > 1e-6:

        srgb /= p99

    return clamp01(
        srgb
    )


# ============================================================
# Image analysis
# ============================================================

def analyze_image(img):

    lum = luminance(img)

    mean = float(
        np.mean(lum)
    )

    median = float(
        np.median(lum)
    )

    p01, p05, p95, p99 = np.percentile(
        lum,
        [
            1,
            5,
            95,
            99,
        ],
    )

    shadow_ratio = float(
        np.mean(
            lum < 0.05
        )
    )

    highlight_ratio = float(
        np.mean(
            lum > 0.95
        )
    )

    dynamic_range = float(
        np.log10(
            max(
                float(p99),
                1e-5,
            )
            /
            max(
                float(p01),
                1e-5,
            )
        )
    )

    saturation_ratio = float(
        np.mean(
            (
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
            < 0.02
        )
    )

    gray = (
        clamp01(lum)
        * 255
    ).astype(
        np.uint8
    )

    edges = cv2.Canny(
        gray,
        50,
        150,
    )

    edge_density = float(
        np.mean(
            edges > 0
        )
    )

    r = img[..., 0]
    g = img[..., 1]
    b = img[..., 2]

    warm_ratio = float(
        np.mean(
            (
                r > b * 1.10
            )
            &
            (
                r > g * 1.03
            )
        )
    )

    contrast = float(
        np.std(lum)
    )

    return ImageAnalysis(

        mean=mean,

        median=median,

        p01=float(p01),
        p05=float(p05),
        p95=float(p95),
        p99=float(p99),

        shadow_ratio=shadow_ratio,
        highlight_ratio=highlight_ratio,

        dynamic_range=dynamic_range,

        saturation_ratio=saturation_ratio,

        edge_density=edge_density,

        warm_ratio=warm_ratio,

        contrast=contrast,
    )


# ============================================================
# Shooting condition analysis
# ============================================================

def analyze_shooting_condition(
    profile,
    analysis,
):

    iso = max(
        profile.iso,
        100,
    )

    iso_factor = clamp01(
        math.log2(
            iso / 100.0
        )
        /
        math.log2(12800 / 100)
    )

    # --------------------------------------------------------
    # Shutter
    # --------------------------------------------------------

    shutter = (
        profile.exposure_time
        if profile.exposure_time
        is not None
        else 1 / 125
    )

    # Longer exposure => higher motion risk.
    if shutter >= 1.0:
        motion_risk = 1.0

    elif shutter >= 0.5:
        motion_risk = 0.90

    elif shutter >= 0.25:
        motion_risk = 0.75

    elif shutter >= 0.125:
        motion_risk = 0.55

    elif shutter >= 0.06:
        motion_risk = 0.30

    else:
        motion_risk = 0.05

    # Low-light indicator
    low_light = clamp01(
        (
            0.30
            -
            analysis.mean
        )
        /
        0.30
    )

    low_light = max(
        low_light,
        iso_factor * 0.7,
    )

    # --------------------------------------------------------
    # Aperture
    # --------------------------------------------------------

    aperture = (
        profile.f_number
        if profile.f_number
        is not None
        else 5.6
    )

    shallow_dof = clamp01(
        (
            5.6
            -
            aperture
        )
        /
        4.0
    )

    # --------------------------------------------------------
    # Focal length
    # --------------------------------------------------------

    focal = (
        profile.focal_length
        if profile.focal_length
        is not None
        else 35.0
    )

    wide_angle = clamp01(
        (
            35.0
            -
            focal
        )
        /
        25.0
    )

    telephoto = clamp01(
        (
            focal
            -
            70.0
        )
        /
        130.0
    )

    # --------------------------------------------------------
    # Noise estimate
    # --------------------------------------------------------

    estimated_noise = clamp01(
        iso_factor * 0.75
        +
        low_light * 0.25
    )

    return ShootingCondition(

        iso_factor=float(
            iso_factor
        ),

        shutter_factor=float(
            motion_risk
        ),

        aperture_factor=float(
            shallow_dof
        ),

        focal_factor=float(
            max(
                wide_angle,
                telephoto,
            )
        ),

        low_light=float(
            low_light
        ),

        motion_risk=float(
            motion_risk
        ),

        shallow_dof=float(
            shallow_dof
        ),

        wide_angle=float(
            wide_angle
        ),

        telephoto=float(
            telephoto
        ),

        estimated_noise=float(
            estimated_noise
        ),
    )


def print_shooting_condition(
    condition,
):

    print(
        "[INFO] Shooting condition:"
    )

    print(
        f"  ISO factor      : "
        f"{condition.iso_factor:.2f}"
    )

    print(
        f"  Low light       : "
        f"{condition.low_light:.2f}"
    )

    print(
        f"  Motion risk     : "
        f"{condition.motion_risk:.2f}"
    )

    print(
        f"  Shallow DOF     : "
        f"{condition.shallow_dof:.2f}"
    )

    print(
        f"  Wide angle      : "
        f"{condition.wide_angle:.2f}"
    )

    print(
        f"  Telephoto       : "
        f"{condition.telephoto:.2f}"
    )

    print(
        f"  Estimated noise : "
        f"{condition.estimated_noise:.2f}"
    )


# ============================================================
# Semantic segmentation
# ============================================================

class SemanticSegmenter:

    def __init__(
        self,
        device="cpu",
        max_size=768,
    ):

        import torch
        import torchvision

        self.torch = torch

        if (
            device == "cuda"
            and
            not torch.cuda.is_available()
        ):

            device = "cpu"

        self.device = torch.device(
            device
        )

        print(
            "[INFO] Semantic device: "
            f"{self.device}"
        )

        weights = (
            torchvision.models
            .segmentation
            .DeepLabV3_MobileNet_V3_Large_Weights
            .DEFAULT
        )

        self.model = (
            torchvision.models
            .segmentation
            .deeplabv3_mobilenet_v3_large(
                weights=weights
            )
        )

        self.model.eval()

        self.model.to(
            self.device
        )

        self.preprocess = (
            weights.transforms()
        )

        self.max_size = max_size

    def predict(self, img):

        h, w = img.shape[:2]

        scale = min(
            1.0,
            self.max_size
            /
            max(h, w),
        )

        if scale < 1.0:

            nw = int(
                w * scale
            )

            nh = int(
                h * scale
            )

            small = cv2.resize(
                img,
                (
                    nw,
                    nh,
                ),
                interpolation=
                    cv2.INTER_AREA,
            )

        else:

            small = img

        rgb8 = (
            clamp01(small)
            * 255
        ).astype(
            np.uint8
        )

        pil = Image.fromarray(
            rgb8
        )

        tensor = (
            self.preprocess(
                pil
            )
            .unsqueeze(0)
            .to(self.device)
        )

        with self.torch.no_grad():

            output = self.model(
                tensor
            )["out"]

            probabilities = (
                self.torch.softmax(
                    output,
                    dim=1,
                )[0]
                .cpu()
                .numpy()
            )

        labels = np.argmax(
            probabilities,
            axis=0,
        )

        confidence = np.max(
            probabilities,
            axis=0,
        )

        labels = cv2.resize(
            labels.astype(
                np.uint8
            ),
            (
                w,
                h,
            ),
            interpolation=
                cv2.INTER_NEAREST,
        )

        confidence = cv2.resize(
            confidence.astype(
                np.float32
            ),
            (
                w,
                h,
            ),
            interpolation=
                cv2.INTER_LINEAR,
        )

        return (
            labels,
            confidence,
        )


# ============================================================
# Saliency
# ============================================================

def calculate_saliency_map(
    img,
):

    h, w = img.shape[:2]

    lum = luminance(
        img
    )

    local = cv2.GaussianBlur(
        lum,
        (0, 0),
        9,
    )

    local_contrast = np.abs(
        lum
        -
        local
    )

    gray = (
        clamp01(lum)
        * 255
    ).astype(
        np.uint8
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

    edge = np.sqrt(
        gx * gx
        +
        gy * gy
    )

    edge = normalize_map(
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

    mean_blur = cv2.GaussianBlur(
        lum,
        (0, 0),
        15,
    )

    brightness_distinct = np.abs(
        lum
        -
        mean_blur
    )

    yy, xx = np.mgrid[
        0:h,
        0:w,
    ]

    cx = (
        w - 1
    ) / 2

    cy = (
        h - 1
    ) / 2

    dx = (
        xx - cx
    ) / max(
        cx,
        1,
    )

    dy = (
        yy - cy
    ) / max(
        cy,
        1,
    )

    distance = np.sqrt(
        dx * dx
        +
        dy * dy
    )

    center = np.exp(
        -1.8
        *
        distance
        *
        distance
    )

    saliency = (

        0.30
        *
        normalize_map(
            local_contrast
        )

        +

        0.25
        *
        edge

        +

        0.15
        *
        normalize_map(
            colorfulness
        )

        +

        0.20
        *
        normalize_map(
            brightness_distinct
        )

        +

        0.10
        *
        center
    )

    saliency = cv2.GaussianBlur(
        saliency.astype(
            np.float32
        ),
        (0, 0),
        3,
    )

    return normalize_map(
        saliency
    )


# ============================================================
# Subject ranking
# ============================================================

def class_prior(
    class_name,
):

    if class_name == "person":
        return 1.15

    if class_name in ANIMAL_CLASSES:
        return 1.05

    if class_name in VEHICLE_CLASSES:
        return 1.00

    if class_name == "pottedplant":
        return 0.90

    if class_name == "bottle":
        return 0.85

    return 1.0


def calculate_subjects(
    labels,
    confidence,
    saliency,
    img,
):

    h, w = labels.shape

    yy, xx = np.mgrid[
        0:h,
        0:w,
    ]

    center = np.exp(
        -1.8
        *
        (
            (
                (xx - (w - 1) / 2)
                /
                max((w - 1) / 2, 1)
            ) ** 2
            +
            (
                (yy - (h - 1) / 2)
                /
                max((h - 1) / 2, 1)
            ) ** 2
        )
    )

    lum = luminance(
        img
    )

    local_blur = cv2.GaussianBlur(
        lum,
        (0, 0),
        15,
    )

    results = []

    for class_id, class_name in enumerate(
        VOC_CLASSES
    ):

        if class_name not in SUBJECT_CLASSES:
            continue

        mask = (
            labels == class_id
        )

        if not np.any(mask):
            continue

        confidence_mean = float(
            np.mean(
                confidence[mask]
            )
        )

        if confidence_mean < 0.35:
            continue

        area = float(
            np.mean(mask)
        )

        ys, xs = np.where(
            mask
        )

        if len(xs) == 0:
            continue

        cy = int(
            np.mean(ys)
        )

        cx = int(
            np.mean(xs)
        )

        center_score = float(
            center[
                cy,
                cx,
            ]
        )

        saliency_score = float(
            np.mean(
                saliency[mask]
            )
        )

        local_contrast = float(
            np.mean(
                np.abs(
                    lum[mask]
                    -
                    local_blur[mask]
                )
            )
        )

        pixels = img[mask]

        colorfulness = float(
            np.mean(
                np.max(
                    pixels,
                    axis=1,
                )
                -
                np.min(
                    pixels,
                    axis=1,
                )
            )
        )

        area_score = min(
            math.sqrt(
                area * 20
            ),
            1.0,
        )

        local_contrast_score = min(
            local_contrast / 0.15,
            1.0,
        )

        score = (

            confidence_mean

            *

            (
                0.25
                * area_score

                +

                0.20
                * center_score

                +

                0.25
                * saliency_score

                +

                0.15
                * local_contrast_score

                +

                0.15
                * colorfulness
            )

            *

            class_prior(
                class_name
            )
        )

        results.append(
            SubjectCandidate(

                class_name=class_name,

                confidence=
                    confidence_mean,

                area=area,

                center_score=
                    center_score,

                saliency_score=
                    saliency_score,

                local_contrast=
                    local_contrast,

                colorfulness=
                    colorfulness,

                score=float(score),
            )
        )

    results.sort(
        key=lambda x: x.score,
        reverse=True,
    )

    return results


# ============================================================
# Scene classification
# ============================================================

def classify_scene(
    analysis,
    subjects,
    condition,
):

    if (
        analysis.mean < 0.12
        or
        condition.low_light > 0.75
    ):

        return "night"

    if (
        analysis.warm_ratio > 0.28
        and
        analysis.highlight_ratio > 0.03
    ):

        return "sunset"

    has_person = any(
        s.class_name == "person"
        for s in subjects[:5]
    )

    if (
        has_person
        and
        analysis.edge_density < 0.15
    ):

        return "portrait"

    if (
        condition.telephoto > 0.6
        and
        has_person
    ):

        return "portrait"

    if (
        analysis.edge_density > 0.18
        and
        analysis.mean > 0.30
    ):

        return "city"

    if (
        analysis.dynamic_range > 1.0
        and
        analysis.edge_density < 0.16
    ):

        return "landscape"

    if (
        analysis.mean > 0.55
        and
        condition.low_light < 0.3
    ):

        return "indoor"

    return "general"


# ============================================================
# Scene profiles
# ============================================================

SCENE_PROFILES = {

    "portrait": SceneProfile(
        name="portrait",
        exposure=0.05,
        contrast=1.02,
        saturation=0.97,
        highlight=0.35,
        shadow=0.08,
        subject_strength=0.08,
        background_suppression=0.035,
        denoise=0.55,
        sharpen=0.75,
    ),

    "night": SceneProfile(
        name="night",
        exposure=0.00,
        contrast=1.05,
        saturation=1.03,
        highlight=0.55,
        shadow=0.02,
        subject_strength=0.05,
        background_suppression=0.015,
        denoise=0.85,
        sharpen=0.45,
    ),

    "sunset": SceneProfile(
        name="sunset",
        exposure=-0.05,
        contrast=1.06,
        saturation=1.08,
        highlight=0.55,
        shadow=0.04,
        subject_strength=0.05,
        background_suppression=0.015,
        denoise=0.30,
        sharpen=0.80,
    ),

    "landscape": SceneProfile(
        name="landscape",
        exposure=0.03,
        contrast=1.08,
        saturation=1.04,
        highlight=0.40,
        shadow=0.08,
        subject_strength=0.06,
        background_suppression=0.015,
        denoise=0.30,
        sharpen=0.85,
    ),

    "city": SceneProfile(
        name="city",
        exposure=0.02,
        contrast=1.07,
        saturation=1.02,
        highlight=0.45,
        shadow=0.05,
        subject_strength=0.06,
        background_suppression=0.02,
        denoise=0.40,
        sharpen=0.80,
    ),

    "indoor": SceneProfile(
        name="indoor",
        exposure=0.04,
        contrast=1.03,
        saturation=0.99,
        highlight=0.40,
        shadow=0.08,
        subject_strength=0.05,
        background_suppression=0.015,
        denoise=0.50,
        sharpen=0.65,
    ),

    "general": SceneProfile(
        name="general",
        exposure=0.00,
        contrast=1.04,
        saturation=1.00,
        highlight=0.35,
        shadow=0.06,
        subject_strength=0.04,
        background_suppression=0.015,
        denoise=0.30,
        sharpen=0.75,
    ),
}


# ============================================================
# Camera adjustment
# ============================================================

def camera_adjustment(
    profile,
):

    family = camera_family(
        profile.make,
        profile.model,
    )

    text = normalize_camera_string(
        f"{profile.make} {profile.model}"
    )

    exposure = 0.0
    contrast = 1.0
    saturation = 1.0

    if family == "canon":

        contrast = 1.005

    elif family == "nikon":

        saturation = 0.995

    elif family == "sony":

        contrast = 1.005

    elif family == "fujifilm":

        saturation = 1.005

    elif family == "panasonic":

        contrast = 1.003

    elif family == "olympus":

        saturation = 1.005

    # Conservative model-level corrections.
    if "a7r" in text:

        contrast += 0.003

    if (
        "z8" in text
        or
        "z9" in text
    ):

        contrast += 0.003

    if (
        "r5" in text
        or
        "r6" in text
    ):

        saturation += 0.003

    return (
        exposure,
        contrast,
        saturation,
    )


# ============================================================
# Color temperature adjustment
# ============================================================

def color_temperature_adjustment(
    img,
    profile,
):

    cct = profile.color_temperature

    if cct is None:
        return img

    # Very conservative correction.
    #
    # Camera WB is already applied by LibRaw.
    # Therefore this is NOT a WB replacement.
    # It is only a small rendering correction.

    if cct < 3000:

        # Avoid excessive blue cast.
        red_gain = 1.005
        blue_gain = 0.995

    elif cct < 4000:

        red_gain = 1.003
        blue_gain = 0.998

    elif cct > 7000:

        red_gain = 0.997
        blue_gain = 1.003

    elif cct > 6000:

        red_gain = 0.999
        blue_gain = 1.002

    else:

        red_gain = 1.0
        blue_gain = 1.0

    result = img.copy()

    result[..., 0] *= red_gain
    result[..., 2] *= blue_gain

    return clamp01(
        result
    )


# ============================================================
# Tone
# ============================================================

def apply_exposure(
    img,
    ev,
):

    gain = 2.0 ** ev

    return clamp01(
        img * gain
    )


def apply_contrast(
    img,
    contrast,
):

    return clamp01(
        (
            img
            -
            0.5
        )
        *
        contrast
        +
        0.5
    )


def apply_saturation(
    img,
    saturation,
):

    lum = luminance(
        img
    )[..., None]

    return clamp01(
        lum
        +
        (
            img
            -
            lum
        )
        *
        saturation
    )


# ============================================================
# RAW level based highlight/shadow
# ============================================================

def calculate_raw_headroom(
    profile,
):

    if (
        profile.black_level.size == 0
        or
        profile.white_level <= 0
    ):

        return 0.5

    black = float(
        np.mean(
            profile.black_level
        )
    )

    white = float(
        profile.white_level
    )

    if white <= black:

        return 0.5

    headroom = (
        white
        -
        black
    ) / white

    return float(
        clamp01(headroom)
    )


def apply_tone_protection(
    img,
    profile,
    highlight_strength,
    shadow_strength,
):

    lum = luminance(
        img
    )

    headroom = calculate_raw_headroom(
        profile
    )

    # More RAW headroom -> stronger highlight preservation.
    highlight_strength *= (
        0.8
        +
        0.4 * headroom
    )

    highlight = np.clip(
        (
            lum
            -
            0.65
        )
        /
        0.35,
        0.0,
        1.0,
    )

    highlight_compress = (
        highlight ** 1.5
    ) * highlight_strength

    img = (
        img
        -
        highlight_compress[
            ...,
            None,
        ]
        *
        0.10
    )

    shadow = np.clip(
        (
            0.30
            -
            lum
        )
        /
        0.30,
        0.0,
        1.0,
    )

    shadow_lift = (
        shadow ** 1.4
    ) * shadow_strength

    img = (
        img
        +
        shadow_lift[
            ...,
            None,
        ]
        *
        0.08
    )

    return clamp01(
        img
    )


# ============================================================
# CLAHE
# ============================================================

def apply_clahe(
    img,
    strength=0.5,
):

    if strength <= 0:
        return img

    lab = cv2.cvtColor(
        (
            clamp01(img)
            * 255
        ).astype(
            np.uint8
        ),
        cv2.COLOR_RGB2LAB,
    )

    l, a, b = cv2.split(
        lab
    )

    clahe = cv2.createCLAHE(
        clipLimit=
            1.5
            +
            strength * 1.5,

        tileGridSize=(
            8,
            8,
        ),
    )

    l2 = clahe.apply(
        l
    )

    lab2 = cv2.merge(
        [
            l2,
            a,
            b,
        ]
    )

    result = cv2.cvtColor(
        lab2,
        cv2.COLOR_LAB2RGB,
    )

    return (
        result.astype(
            np.float32
        )
        /
        255.0
    )


# ============================================================
# Local subject
# ============================================================

def apply_local_subject(
    img,
    labels,
    subjects,
    profile,
    condition,
):

    if not subjects:
        return img

    best = subjects[0]

    class_id = VOC_CLASSES.index(
        best.class_name
    )

    mask = (
        labels == class_id
    ).astype(
        np.float32
    )

    if np.mean(mask) < 0.002:
        return img

    mask = cv2.GaussianBlur(
        mask,
        (0, 0),
        9,
    )

    mask = clamp01(
        mask
    )

    strength = (
        profile.subject_strength
    )

    # Stronger subject emphasis for
    # shallow depth-of-field shots.
    strength *= (
        1.0
        +
        condition.shallow_dof
        *
        0.5
    )

    foreground = (
        1.0
        +
        strength
        * mask
    )

    result = clamp01(
        img
        *
        foreground[
            ...,
            None,
        ]
    )

    if (
        profile.background_suppression
        > 0
    ):

        suppression = (
            1.0
            -
            profile.background_suppression
            *
            (
                1.0
                -
                mask
            )
        )

        result = clamp01(
            result
            *
            suppression[
                ...,
                None,
            ]
        )

    return result


# ============================================================
# Denoise
# ============================================================

def apply_denoise(
    img,
    strength,
):

    if strength <= 0:
        return img

    img8 = (
        clamp01(img)
        * 255
    ).astype(
        np.uint8
    )

    sigma_color = (
        15.0
        +
        35.0 * strength
    )

    sigma_space = (
        3.0
        +
        4.0 * strength
    )

    result = cv2.bilateralFilter(
        img8,
        d=7,
        sigmaColor=sigma_color,
        sigmaSpace=sigma_space,
    )

    return (
        result.astype(
            np.float32
        )
        /
        255.0
    )


# ============================================================
# Sharpen
# ============================================================

def apply_sharpen(
    img,
    strength,
):

    if strength <= 0:
        return img

    blur = cv2.GaussianBlur(
        img,
        (0, 0),
        1.2,
    )

    amount = (
        0.25
        +
        0.85 * strength
    )

    result = (
        img
        +
        amount
        *
        (
            img
            -
            blur
        )
    )

    return clamp01(
        result
    )


# ============================================================
# Automatic parameter search
# ============================================================

def evaluate_candidate(
    img,
    analysis,
    condition,
):

    lum = luminance(
        img
    )

    mean = float(
        np.mean(lum)
    )

    shadow = float(
        np.mean(
            lum < 0.04
        )
    )

    highlight = float(
        np.mean(
            lum > 0.98
        )
    )

    contrast = float(
        np.std(lum)
    )

    exposure_score = 1.0 - min(
        abs(
            mean
            -
            0.46
        )
        /
        0.46,
        1.0,
    )

    clipping_penalty = (
        shadow * 1.5
        +
        highlight * 2.0
    )

    contrast_score = min(
        contrast / 0.25,
        1.0,
    )

    score = (

        exposure_score
        * 0.45

        +

        contrast_score
        * 0.30

        -

        clipping_penalty
        * 0.20
    )

    # --------------------------------------------------------
    # Shooting condition constraints
    # --------------------------------------------------------

    # High ISO:
    # discourage aggressive contrast.
    if condition.iso_factor > 0.6:

        score -= (
            max(
                contrast - 0.25,
                0,
            )
            *
            condition.iso_factor
            *
            0.5
        )

    # Low-light images:
    # protect shadows.
    if condition.low_light > 0.6:

        dark_ratio = float(
            np.mean(
                lum < 0.08
            )
        )

        score -= (
            dark_ratio
            *
            condition.low_light
            *
            0.15
        )

    # Don't change exposure too much.
    score -= (
        abs(
            mean
            -
            analysis.mean
        )
        *
        0.15
    )

    return float(
        score
    )


def automatic_parameter_search(
    img,
    analysis,
    profile,
    condition,
):

    best_score = -float(
        "inf"
    )

    best = (
        0.0,
        1.0,
        1.0,
    )

    ev_values = [
        -0.30,
        -0.15,
        0.00,
        0.15,
        0.30,
    ]

    contrast_values = [
        0.96,
        1.00,
        1.03,
        1.06,
    ]

    saturation_values = [
        0.96,
        1.00,
        1.03,
        1.05,
    ]

    for ev in ev_values:

        for contrast in contrast_values:

            for saturation in saturation_values:

                test = apply_exposure(
                    img,
                    ev
                    +
                    profile.exposure,
                )

                test = apply_contrast(
                    test,
                    contrast
                    *
                    profile.contrast,
                )

                test = apply_saturation(
                    test,
                    saturation
                    *
                    profile.saturation,
                )

                score = evaluate_candidate(
                    test,
                    analysis,
                    condition,
                )

                if score > best_score:

                    best_score = score

                    best = (
                        ev
                        +
                        profile.exposure,

                        contrast
                        *
                        profile.contrast,

                        saturation
                        *
                        profile.saturation,
                    )

    print(
        "[INFO] Auto parameters: "
        f"EV={best[0]:+.2f}, "
        f"contrast={best[1]:.3f}, "
        f"saturation={best[2]:.3f}"
    )

    print(
        f"[INFO] Parameter score: "
        f"{best_score:.4f}"
    )

    return best


# ============================================================
# ISO / shooting condition adjustment
# ============================================================

def calculate_final_processing(
    profile,
    scene_profile,
    condition,
):

    denoise = max(
        scene_profile.denoise,
        condition.estimated_noise,
    )

    sharpen = (
        scene_profile.sharpen
    )

    # High ISO
    sharpen *= (
        1.0
        -
        0.35
        *
        condition.iso_factor
    )

    # Low light
    sharpen *= (
        1.0
        -
        0.15
        *
        condition.low_light
    )

    # Long exposure
    sharpen *= (
        1.0
        -
        0.10
        *
        condition.motion_risk
    )

    denoise = clamp01(
        denoise
    )

    sharpen = clamp01(
        sharpen
    )

    return (
        denoise,
        sharpen,
    )


# ============================================================
# Main development
# ============================================================

def develop_image(
    rgb,
    profile,
    segmenter,
):

    print(
        "[INFO] Analyzing image..."
    )

    analysis = analyze_image(
        rgb
    )

    print(
        "[INFO] "
        f"mean={analysis.mean:.3f} "
        f"median={analysis.median:.3f} "
        f"p01={analysis.p01:.3f} "
        f"p99={analysis.p99:.3f}"
    )

    print(
        "[INFO] "
        f"shadow={analysis.shadow_ratio:.3f} "
        f"highlight={analysis.highlight_ratio:.3f} "
        f"edge={analysis.edge_density:.3f} "
        f"contrast={analysis.contrast:.3f}"
    )

    condition = analyze_shooting_condition(
        profile,
        analysis,
    )

    print_shooting_condition(
        condition
    )

    print(
        "[INFO] Semantic segmentation..."
    )

    labels, confidence = (
        segmenter.predict(
            rgb
        )
    )

    print(
        "[INFO] Saliency..."
    )

    saliency = calculate_saliency_map(
        rgb
    )

    subjects = calculate_subjects(
        labels,
        confidence,
        saliency,
        rgb,
    )

    if subjects:

        print(
            "[INFO] Subjects:"
        )

        for s in subjects[:5]:

            print(
                "  "
                f"{s.class_name}: "
                f"conf={s.confidence:.2f}, "
                f"area={s.area:.3f}, "
                f"saliency={s.saliency_score:.2f}, "
                f"score={s.score:.3f}"
            )

    else:

        print(
            "[INFO] Subject: none"
        )

    scene_name = classify_scene(
        analysis,
        subjects,
        condition,
    )

    scene_profile = (
        SCENE_PROFILES[
            scene_name
        ]
    )

    print(
        f"[INFO] Scene: "
        f"{scene_name}"
    )

    denoise_strength, sharpen_strength = (
        calculate_final_processing(
            profile,
            scene_profile,
            condition,
        )
    )

    print(
        f"[INFO] Final denoise: "
        f"{denoise_strength:.2f}"
    )

    print(
        f"[INFO] Final sharpen: "
        f"{sharpen_strength:.2f}"
    )

    cam_ev, cam_contrast, cam_saturation = (
        camera_adjustment(
            profile
        )
    )

    working = rgb.copy()

    # --------------------------------------------------------
    # Camera profile
    # --------------------------------------------------------

    working = apply_exposure(
        working,
        cam_ev,
    )

    working = apply_contrast(
        working,
        cam_contrast,
    )

    working = apply_saturation(
        working,
        cam_saturation,
    )

    # --------------------------------------------------------
    # CCT
    # --------------------------------------------------------

    working = color_temperature_adjustment(
        working,
        profile,
    )

    # --------------------------------------------------------
    # Automatic search
    # --------------------------------------------------------

    ev, contrast, saturation = (
        automatic_parameter_search(
            working,
            analysis,
            scene_profile,
            condition,
        )
    )

    working = apply_exposure(
        working,
        ev,
    )

    working = apply_contrast(
        working,
        contrast,
    )

    working = apply_saturation(
        working,
        saturation,
    )

    # --------------------------------------------------------
    # Tone protection
    # --------------------------------------------------------

    working = apply_tone_protection(
        working,
        profile,
        scene_profile.highlight,
        scene_profile.shadow,
    )

    # --------------------------------------------------------
    # Local subject
    # --------------------------------------------------------

    working = apply_local_subject(
        working,
        labels,
        subjects,
        scene_profile,
        condition,
    )

    # --------------------------------------------------------
    # CLAHE
    # --------------------------------------------------------

    clahe_strength = 0.35

    if scene_name == "night":

        clahe_strength = 0.20

    elif scene_name == "portrait":

        clahe_strength = 0.15

    elif condition.iso_factor > 0.7:

        clahe_strength = 0.15

    working = apply_clahe(
        working,
        clahe_strength,
    )

    # --------------------------------------------------------
    # Denoise
    # --------------------------------------------------------

    working = apply_denoise(
        working,
        denoise_strength,
    )

    # --------------------------------------------------------
    # Sharpen
    # --------------------------------------------------------

    working = apply_sharpen(
        working,
        sharpen_strength,
    )

    return clamp01(
        working
    )


# ============================================================
# Save JPEG
# ============================================================

def save_jpeg(
    img,
    output_path,
    quality=95,
):

    img8 = (
        clamp01(img)
        * 255
    ).astype(
        np.uint8
    )

    pil = Image.fromarray(
        img8,
        mode="RGB",
    )

    pil.save(
        output_path,
        "JPEG",
        quality=quality,
        subsampling=0,
        optimize=True,
    )


# ============================================================
# Process RAW
# ============================================================

def process_raw(
    path,
    output_path,
    device,
):

    print()
    print(
        "=" * 70
    )

    print(
        f"[INFO] Processing: "
        f"{path}"
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    exiftool_meta = read_exiftool(
        path
    )

    pillow_meta = read_pillow_exif(
        path
    )

    metadata = merge_metadata(
        exiftool_meta,
        pillow_meta,
    )

    if (
        metadata.make
        or
        metadata.model
    ):

        print(
            "[INFO] EXIF camera: "
            f"{metadata.make} "
            f"{metadata.model}"
        )

    else:

        print(
            "[WARN] EXIF camera "
            "information unavailable."
        )

    # --------------------------------------------------------
    # RAW
    # --------------------------------------------------------

    try:

        raw, camera_rgb = (
            load_linear_camera_rgb(
                path
            )
        )

        profile = build_camera_profile(
            raw,
            metadata,
        )

        print_camera_profile(
            profile
        )

        # ----------------------------------------------------
        # Camera color conversion
        # ----------------------------------------------------

        print(
            "[INFO] Camera RGB -> sRGB..."
        )

        rgb = camera_rgb_to_srgb(
            camera_rgb,
            profile,
        )

        raw.close()

    except Exception as e:

        print(
            "[ERROR] RAW processing failed: "
            f"{e}"
        )

        return False

    # --------------------------------------------------------
    # Semantic model
    # --------------------------------------------------------

    try:

        segmenter = SemanticSegmenter(
            device=device,
            max_size=768,
        )

    except Exception as e:

        print(
            "[ERROR] Semantic model "
            f"initialization failed: {e}"
        )

        segmenter = None

    # --------------------------------------------------------
    # Develop
    # --------------------------------------------------------

    if segmenter is not None:

        result = develop_image(
            rgb,
            profile,
            segmenter,
        )

    else:

        print(
            "[WARN] Running basic "
            "automatic development."
        )

        analysis = analyze_image(
            rgb
        )

        condition = (
            analyze_shooting_condition(
                profile,
                analysis,
            )
        )

        scene_profile = (
            SCENE_PROFILES[
                "general"
            ]
        )

        ev, contrast, saturation = (
            automatic_parameter_search(
                rgb,
                analysis,
                scene_profile,
                condition,
            )
        )

        result = apply_exposure(
            rgb,
            ev,
        )

        result = apply_contrast(
            result,
            contrast,
        )

        result = apply_saturation(
            result,
            saturation,
        )

        result = apply_tone_protection(
            result,
            profile,
            scene_profile.highlight,
            scene_profile.shadow,
        )

        denoise, sharpen = (
            calculate_final_processing(
                profile,
                scene_profile,
                condition,
            )
        )

        result = apply_denoise(
            result,
            denoise,
        )

        result = apply_sharpen(
            result,
            sharpen,
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_jpeg(
        result,
        output_path,
        quality=95,
    )

    print(
        f"[INFO] Saved: "
        f"{output_path}"
    )

    return True


# ============================================================
# RAW collection
# ============================================================

def collect_raw_files(
    input_path,
):

    if input_path.is_file():

        if (
            input_path.suffix.lower()
            in RAW_EXTENSIONS
        ):

            return [
                input_path
            ]

        return []

    files = []

    for path in input_path.rglob("*"):

        if (
            path.is_file()
            and
            path.suffix.lower()
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
            "Automatic RAW Developer v14"
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
        default="cpu",
        choices=[
            "cpu",
            "cuda",
        ],
        help=(
            "Semantic segmentation device"
        ),
    )

    args = parser.parse_args()

    if not args.input.exists():

        print(
            "[ERROR] Input does not exist: "
            f"{args.input}"
        )

        sys.exit(1)

    if (
        args.device == "cuda"
    ):

        try:

            import torch

            if not torch.cuda.is_available():

                print(
                    "[WARN] CUDA unavailable. "
                    "Using CPU."
                )

                args.device = "cpu"

        except Exception:

            print(
                "[WARN] PyTorch unavailable "
                "for CUDA check."
            )

            args.device = "cpu"

    files = collect_raw_files(
        args.input
    )

    if not files:

        print(
            "[ERROR] No RAW files found."
        )

        sys.exit(1)

    print(
        f"[INFO] RAW files: "
        f"{len(files)}"
    )

    success = 0

    for raw_path in files:

        if args.input.is_file():

            relative = (
                raw_path.stem
                +
                ".jpg"
            )

        else:

            try:

                relative = (
                    raw_path
                    .relative_to(
                        args.input
                    )
                    .with_suffix(
                        ".jpg"
                    )
                )

            except ValueError:

                relative = (
                    raw_path.stem
                    +
                    ".jpg"
                )

        output_path = (
            args.output
            /
            relative
        )

        if process_raw(
            raw_path,
            output_path,
            args.device,
        ):

            success += 1

    print()

    print(
        "=" * 70
    )

    print(
        f"[INFO] Finished: "
        f"{success}/{len(files)}"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":

    main()

