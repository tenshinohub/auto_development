#!/usr/bin/env python3

import argparse
import json
import math
import os
import shutil
import subprocess
import warnings

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import rawpy
from PIL import Image, ExifTags

import torch
import torchvision
from torchvision import transforms


warnings.filterwarnings("ignore")


VERSION = "v20"

RAW_EXTENSIONS = {
    ".cr2", ".cr3",
    ".nef", ".nrw",
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

SUBJECT_CLASSES = {
    "person",
    "bird",
    "cat",
    "cow",
    "dog",
    "horse",
    "sheep",
    "aeroplane",
    "bicycle",
    "boat",
    "bus",
    "car",
    "motorbike",
    "train",
    "pottedplant",
}


XYZ_TO_SRGB = np.array(
    [
        [ 3.2404542, -1.5371385, -0.4985314],
        [-0.9692660,  1.8760108,  0.0415560],
        [ 0.0556434, -0.2040259,  1.0572252],
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

    iso: float = 0.0
    exposure_time: float = 0.0
    f_number: float = 0.0
    focal_length: float = 0.0

    white_balance: str = ""
    color_temperature: float = 0.0
    color_space: str = ""

    source: str = ""


@dataclass
class CameraProfile:
    make: str = ""
    model: str = ""
    family: str = ""

    iso: float = 0.0
    black_level: float = 0.0
    white_level: float = 0.0

    camera_white_balance: List[float] = field(default_factory=list)
    color_matrix: List[List[float]] = field(default_factory=list)
    rgb_xyz_matrix: List[List[float]] = field(default_factory=list)

    raw_width: int = 0
    raw_height: int = 0

    lens: str = ""
    metadata_source: str = ""
    libraw_version: str = ""


@dataclass
class ImageStats:
    min: float
    max: float

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

    warm_ratio: float
    contrast: float

    mean_luminance: float

    r_mean: float
    g_mean: float
    b_mean: float

    rg_ratio: float
    gb_ratio: float


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
    class_name: str
    confidence: float
    area: float

    center_x: float
    center_y: float

    saliency: float
    local_contrast: float
    colorfulness: float

    score: float


@dataclass
class SceneResult:
    scene: str
    confidence: float


@dataclass
class RegionStats:
    name: str
    area: float
    mean_luminance: float
    mean_saturation: float


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

    region_skin_saturation: float
    region_green_saturation: float
    region_water_highlight: float
    region_upper_highlight: float

    tone_strength: float


# ============================================================
# Utility
# ============================================================

def clamp01(x):
    return np.clip(x, 0.0, 1.0)


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


def rgb_luminance(rgb):
    return (
        rgb[..., 0] * 0.2126
        + rgb[..., 1] * 0.7152
        + rgb[..., 2] * 0.0722
    )


def safe_ratio(a, b, eps=1e-6):
    return float(a) / max(float(b), eps)


def resize_keep_aspect(image, max_dim):
    h, w = image.shape[:2]

    if max(h, w) <= max_dim:
        return image

    scale = max_dim / max(h, w)

    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))

    return cv2.resize(
        image,
        (nw, nh),
        interpolation=cv2.INTER_AREA,
    )


def save_rgb(path, image):
    image = np.clip(image, 0.0, 1.0)

    arr = (image * 255.0 + 0.5).astype(np.uint8)

    Image.fromarray(arr, "RGB").save(
        str(path),
        quality=95,
    )


def save_gray(path, image):
    image = np.clip(image, 0.0, 1.0)

    arr = (image * 255.0 + 0.5).astype(np.uint8)

    Image.fromarray(arr, "L").save(str(path))


# ============================================================
# Statistics
# ============================================================

def calculate_stats(image):
    image = np.asarray(image, dtype=np.float32)

    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)

    h, w = image.shape[:2]

    # Downsample for statistics
    if max(h, w) > 1024:
        small = resize_keep_aspect(image, 1024)
    else:
        small = image

    small = np.clip(small, 0.0, 1.0)

    lum = rgb_luminance(small)

    flat = lum.reshape(-1)

    p01 = float(np.percentile(flat, 1))
    p05 = float(np.percentile(flat, 5))
    p25 = float(np.percentile(flat, 25))
    p50 = float(np.percentile(flat, 50))
    p75 = float(np.percentile(flat, 75))
    p95 = float(np.percentile(flat, 95))
    p99 = float(np.percentile(flat, 99))

    shadow_ratio = float(np.mean(lum < 0.02))
    highlight_ratio = float(np.mean(lum > 0.98))

    dynamic_range = math.log10(
        max(p95, 1e-6) / max(p05, 1e-6)
    )

    max_rgb = np.max(small, axis=2)
    min_rgb = np.min(small, axis=2)

    saturation_ratio = float(
        np.mean((max_rgb - min_rgb) > 0.08)
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

    grad = cv2.magnitude(gx, gy)

    edge_density = float(
        np.mean(grad > 0.08)
    )

    hsv = cv2.cvtColor(
        (small * 255).astype(np.uint8),
        cv2.COLOR_RGB2HSV,
    )

    warm_ratio = float(
        np.mean(
            (
                ((hsv[..., 0] < 25) | (hsv[..., 0] > 170))
                & (hsv[..., 1] > 70)
                & (hsv[..., 2] > 50)
            )
        )
    )

    contrast = float(np.std(lum))

    r_mean = float(np.mean(small[..., 0]))
    g_mean = float(np.mean(small[..., 1]))
    b_mean = float(np.mean(small[..., 2]))

    return ImageStats(
        min=float(np.min(lum)),
        max=float(np.max(lum)),

        mean=float(np.mean(lum)),
        median=p50,

        p01=p01,
        p05=p05,
        p25=p25,
        p75=p75,
        p95=p95,
        p99=p99,

        shadow_ratio=shadow_ratio,
        highlight_ratio=highlight_ratio,

        dynamic_range=dynamic_range,
        saturation_ratio=saturation_ratio,
        edge_density=edge_density,

        warm_ratio=warm_ratio,
        contrast=contrast,

        mean_luminance=float(np.mean(lum)),

        r_mean=r_mean,
        g_mean=g_mean,
        b_mean=b_mean,

        rg_ratio=safe_ratio(r_mean, g_mean),
        gb_ratio=safe_ratio(g_mean, b_mean),
    )


def print_stats(name, color_space, stats):
    print(f"\n[DEBUG] ===== {name} ({color_space}) =====")

    print(
        f"  min/max      : "
        f"{stats.min:.6f} / {stats.max:.6f}"
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
        f"  R/G ratio    : "
        f"{stats.rg_ratio:.4f}"
    )

    print(
        f"  G/B ratio    : "
        f"{stats.gb_ratio:.4f}"
    )


# ============================================================
# Metadata
# ============================================================

def run_exiftool(path):
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
            check=True,
        )

        data = json.loads(result.stdout)

        if not data:
            return {}

        return data[0]

    except Exception:
        return {}


def get_pillow_exif(path):
    result = {}

    try:
        with Image.open(path) as img:
            exif = img.getexif()

            for tag_id, value in exif.items():
                name = ExifTags.TAGS.get(tag_id, tag_id)
                result[name] = value

    except Exception:
        pass

    return result


def first_value(d, *keys, default=""):
    for key in keys:
        if key in d:
            value = d[key]

            if value is not None and value != "":
                return value

    return default


def to_float(value, default=0.0):
    try:
        if isinstance(value, str):
            value = value.strip()

            if "/" in value:
                a, b = value.split("/", 1)
                return float(a) / float(b)

        return float(value)

    except Exception:
        return default


def get_metadata(path):
    exiftool_data = run_exiftool(path)
    pillow_data = get_pillow_exif(path)

    source_parts = []

    if exiftool_data:
        source_parts.append("ExifTool")

    if pillow_data:
        source_parts.append("Pillow")

    make = first_value(
        exiftool_data,
        "Make",
        default=first_value(pillow_data, "Make"),
    )

    model = first_value(
        exiftool_data,
        "CameraModelName",
        "Model",
        "UniqueCameraModel",
        default=first_value(pillow_data, "Model"),
    )

    lens_make = first_value(
        exiftool_data,
        "LensMake",
        default=first_value(pillow_data, "LensMake"),
    )

    lens_model = first_value(
        exiftool_data,
        "LensModel",
        default=first_value(pillow_data, "LensModel"),
    )

    iso = to_float(
        first_value(
            exiftool_data,
            "ISO",
            default=first_value(pillow_data, "ISOSpeedRatings"),
        )
    )

    exposure_time = to_float(
        first_value(
            exiftool_data,
            "ExposureTime",
            default=first_value(pillow_data, "ExposureTime"),
        )
    )

    f_number = to_float(
        first_value(
            exiftool_data,
            "FNumber",
            default=first_value(pillow_data, "FNumber"),
        )
    )

    focal_length = to_float(
        first_value(
            exiftool_data,
            "FocalLength",
            default=first_value(pillow_data, "FocalLength"),
        )
    )

    white_balance = str(
        first_value(
            exiftool_data,
            "WhiteBalance",
            default=first_value(pillow_data, "WhiteBalance"),
        )
    )

    color_temperature = to_float(
        first_value(
            exiftool_data,
            "ColorTemperature",
            default=first_value(
                pillow_data,
                "ColorTemperature",
            ),
        )
    )

    color_space = str(
        first_value(
            exiftool_data,
            "ColorSpace",
            default=first_value(
                pillow_data,
                "ColorSpace",
            ),
        )
    )

    return ExifMetadata(
        make=str(make),
        model=str(model),
        lens_make=str(lens_make),
        lens_model=str(lens_model),

        iso=iso,
        exposure_time=exposure_time,
        f_number=f_number,
        focal_length=focal_length,

        white_balance=white_balance,
        color_temperature=color_temperature,
        color_space=color_space,

        source="+".join(source_parts),
    )


def detect_camera_family(make, model):
    text = f"{make} {model}".lower()

    families = [
        ("canon", "Canon"),
        ("nikon", "Nikon"),
        ("sony", "Sony"),
        ("fujifilm", "Fujifilm"),
        ("panasonic", "Panasonic"),
        ("olympus", "Olympus"),
        ("om system", "Olympus"),
        ("leica", "Leica"),
        ("pentax", "Pentax"),
        ("ricoh", "Ricoh"),
        ("sigma", "Sigma"),
        ("hasselblad", "Hasselblad"),
    ]

    for key, name in families:
        if key in text:
            return name

    return "Unknown"


def get_camera_profile(path, metadata):
    profile = CameraProfile()

    profile.make = metadata.make
    profile.model = metadata.model
    profile.family = detect_camera_family(
        metadata.make,
        metadata.model,
    )

    profile.iso = metadata.iso

    profile.lens = metadata.lens_model
    profile.metadata_source = metadata.source

    try:
        profile.libraw_version = str(rawpy.libraw_version)

        with rawpy.imread(str(path)) as raw:

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
                profile.camera_white_balance = [
                    float(x)
                    for x in raw.camera_whitebalance
                ]
            except Exception:
                pass

            try:
                profile.color_matrix = (
                    np.asarray(
                        raw.color_matrix
                    ).tolist()
                )
            except Exception:
                pass

            try:
                profile.rgb_xyz_matrix = (
                    np.asarray(
                        raw.rgb_xyz_matrix
                    ).tolist()
                )
            except Exception:
                pass

            try:
                profile.raw_width = int(raw.sizes.width)
                profile.raw_height = int(raw.sizes.height)
            except Exception:
                pass

    except Exception:
        pass

    return profile


# ============================================================
# RAW development
# ============================================================

def raw_to_linear_rgb(path):
    """
    IMPORTANT

    LibRaw is asked for sRGB primaries with gamma=(1,1).

    Therefore the transfer function has intentionally NOT
    been applied at the LibRaw stage.

    The returned values are treated as linear RGB values
    with sRGB primaries.

    We do NOT apply srgb_to_linear() here.
    """

    with rawpy.imread(str(path)) as raw:

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

            method = "LibRaw sRGB primaries / gamma=(1,1)"

        except Exception as e:

            print(
                "[WARN] Camera processing failed:"
                f" {type(e).__name__}: {e}"
            )

            print(
                "[WARN] Falling back to LibRaw sRGB"
            )

            rgb16 = raw.postprocess(
                use_camera_wb=True,
                use_auto_wb=False,

                output_color=rawpy.ColorSpace.sRGB,
                output_bps=16,

                gamma=(1, 1),

                no_auto_bright=False,

                highlight_mode=rawpy.HighlightMode.Blend,

                half_size=False,
                four_color_rgb=False,

                demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD,
            )

            method = (
                "LibRaw sRGB fallback / gamma=(1,1)"
            )

    linear_rgb = (
        np.asarray(rgb16, dtype=np.float32)
        / 65535.0
    )

    linear_rgb = np.clip(
        linear_rgb,
        0.0,
        1.0,
    )

    return linear_rgb, method


# ============================================================
# Shooting condition
# ============================================================

def analyze_shooting(metadata):
    iso = max(metadata.iso, 100.0)

    iso_factor = math.sqrt(iso / 100.0)
    iso_factor = float(
        np.clip(
            iso_factor,
            1.0,
            5.0,
        )
    )

    low_light = (
        iso >= 1600
        or metadata.exposure_time >= 0.05
    )

    if metadata.exposure_time <= 0:
        motion_risk = 0.0

    elif metadata.exposure_time >= 0.1:
        motion_risk = 1.0

    elif metadata.exposure_time >= 0.05:
        motion_risk = 0.7

    elif metadata.exposure_time >= 0.02:
        motion_risk = 0.4

    else:
        motion_risk = 0.1

    shallow_dof = (
        metadata.f_number > 0
        and metadata.f_number <= 2.8
    )

    wide_angle = (
        metadata.focal_length > 0
        and metadata.focal_length <= 28
    )

    telephoto = (
        metadata.focal_length >= 85
    )

    estimated_noise = float(
        np.clip(
            0.10
            + 0.13 * (iso_factor - 1.0),
            0.0,
            0.85,
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

    def __init__(self, device="auto"):

        if device == "cuda":
            if not torch.cuda.is_available():
                print(
                    "[WARN] CUDA unavailable. "
                    "Falling back to CPU."
                )
                device = "cpu"

        elif device == "auto":
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        self.device = torch.device(device)

        print(
            "[INFO] Loading DeepLabV3 on"
            f" {self.device}"
        )

        try:
            weights = (
                torchvision.models.segmentation
                .DeepLabV3_MobileNet_V3_Large_Weights
                .DEFAULT
            )

            self.model = (
                torchvision.models.segmentation
                .deeplabv3_mobilenet_v3_large(
                    weights=weights
                )
            )

            self.preprocess = weights.transforms()

        except Exception:
            self.model = (
                torchvision.models.segmentation
                .deeplabv3_mobilenet_v3_large(
                    pretrained=True
                )
            )

            self.preprocess = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[
                        0.485,
                        0.456,
                        0.406,
                    ],
                    std=[
                        0.229,
                        0.224,
                        0.225,
                    ],
                ),
            ])

        self.model.eval()
        self.model.to(self.device)

    @torch.inference_mode()
    def predict(self, image):

        original_h, original_w = image.shape[:2]

        small = resize_keep_aspect(
            image,
            768,
        )

        pil = Image.fromarray(
            (
                np.clip(small, 0, 1)
                * 255
            ).astype(np.uint8)
        )

        tensor = self.preprocess(pil)
        tensor = tensor.unsqueeze(0)
        tensor = tensor.to(self.device)

        output = self.model(tensor)["out"][0]

        probabilities = torch.softmax(
            output,
            dim=0,
        )

        confidence, labels = torch.max(
            probabilities,
            dim=0,
        )

        labels = (
            labels.detach()
            .cpu()
            .numpy()
            .astype(np.int16)
        )

        confidence = (
            confidence.detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        labels = cv2.resize(
            labels,
            (original_w, original_h),
            interpolation=cv2.INTER_NEAREST,
        )

        confidence = cv2.resize(
            confidence,
            (original_w, original_h),
            interpolation=cv2.INTER_LINEAR,
        )

        return labels, confidence


# ============================================================
# Saliency
# ============================================================

def compute_saliency(image):

    image = np.clip(
        image,
        0.0,
        1.0,
    )

    lum = rgb_luminance(image)

    blur = cv2.GaussianBlur(
        lum,
        (0, 0),
        5,
    )

    local_contrast = np.abs(
        lum - blur
    )

    local_contrast = cv2.normalize(
        local_contrast,
        None,
        0,
        1,
        cv2.NORM_MINMAX,
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

    edge = cv2.magnitude(gx, gy)

    edge = cv2.normalize(
        edge,
        None,
        0,
        1,
        cv2.NORM_MINMAX,
    )

    hsv = cv2.cvtColor(
        (image * 255).astype(np.uint8),
        cv2.COLOR_RGB2HSV,
    )

    saturation = hsv[..., 1].astype(
        np.float32
    ) / 255.0

    mean_lum = float(np.mean(lum))

    brightness_distinct = np.abs(
        lum - mean_lum
    )

    brightness_distinct = cv2.normalize(
        brightness_distinct,
        None,
        0,
        1,
        cv2.NORM_MINMAX,
    )

    h, w = lum.shape

    yy, xx = np.mgrid[
        0:h,
        0:w,
    ]

    cx = w / 2.0
    cy = h / 2.0

    dx = (xx - cx) / max(cx, 1)
    dy = (yy - cy) / max(cy, 1)

    center_weight = np.exp(
        -(dx * dx + dy * dy) * 1.5
    )

    saliency = (
        0.30 * local_contrast
        + 0.25 * edge
        + 0.15 * saturation
        + 0.20 * brightness_distinct
        + 0.10 * center_weight
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
    labels,
    confidence,
    saliency,
    image,
):

    h, w = labels.shape

    hsv = cv2.cvtColor(
        (np.clip(image, 0, 1) * 255).astype(
            np.uint8
        ),
        cv2.COLOR_RGB2HSV,
    )

    saturation = hsv[..., 1].astype(
        np.float32
    ) / 255.0

    lum = rgb_luminance(image)

    blur = cv2.GaussianBlur(
        lum,
        (0, 0),
        5,
    )

    local_contrast = np.abs(
        lum - blur
    )

    candidates = []

    for idx, class_name in enumerate(
        VOC_CLASSES
    ):

        if class_name == "background":
            continue

        mask = labels == idx

        area = float(np.mean(mask))

        if area < 0.003:
            continue

        ys, xs = np.where(mask)

        if len(xs) == 0:
            continue

        conf = float(
            np.mean(confidence[mask])
        )

        center_x = float(
            np.mean(xs) / max(w - 1, 1)
        )

        center_y = float(
            np.mean(ys) / max(h - 1, 1)
        )

        center_distance = math.sqrt(
            (center_x - 0.5) ** 2
            + (center_y - 0.5) ** 2
        )

        center_score = float(
            1.0 - min(
                center_distance / 0.7072,
                1.0,
            )
        )

        sal = float(
            np.mean(saliency[mask])
        )

        lc = float(
            np.mean(local_contrast[mask])
        )

        colorfulness = float(
            np.mean(saturation[mask])
        )

        prior = 1.0

        if class_name == "person":
            prior = 1.15

        elif class_name in ANIMAL_CLASSES:
            prior = 1.05

        elif class_name in VEHICLE_CLASSES:
            prior = 1.0

        elif class_name == "pottedplant":
            prior = 0.90

        elif class_name == "bottle":
            prior = 0.85

        score = (
            0.30 * conf
            + 0.15 * math.sqrt(area)
            + 0.15 * center_score
            + 0.20 * sal
            + 0.10 * lc
            + 0.10 * colorfulness
        )

        score *= prior

        candidates.append(
            SubjectCandidate(
                class_name=class_name,
                confidence=conf,
                area=area,

                center_x=center_x,
                center_y=center_y,

                saliency=sal,
                local_contrast=lc,
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
    stats,
    shooting,
    subjects,
):

    person_area = 0.0
    vehicle_area = 0.0

    for s in subjects:

        if s.class_name == "person":
            person_area += s.area

        if s.class_name in VEHICLE_CLASSES:
            vehicle_area += s.area

    if (
        person_area > 0.015
        and stats.median > 0.08
    ):
        confidence = min(
            0.55
            + person_area * 1.5,
            0.95,
        )

        return SceneResult(
            scene="portrait",
            confidence=float(confidence),
        )

    if (
        shooting.low_light
        and stats.median < 0.10
    ):
        return SceneResult(
            scene="night",
            confidence=0.80,
        )

    if (
        stats.warm_ratio > 0.18
        and stats.p95 > 0.55
    ):
        return SceneResult(
            scene="sunset",
            confidence=0.75,
        )

    if (
        shooting.wide_angle
        and stats.edge_density < 0.16
        and stats.dynamic_range > 5
    ):
        return SceneResult(
            scene="landscape",
            confidence=0.72,
        )

    if (
        vehicle_area > 0.01
        and stats.edge_density > 0.10
    ):
        return SceneResult(
            scene="city",
            confidence=0.70,
        )

    if (
        stats.median < 0.18
        and stats.warm_ratio > 0.10
    ):
        return SceneResult(
            scene="indoor",
            confidence=0.65,
        )

    return SceneResult(
        scene="general",
        confidence=0.50,
    )


# ============================================================
# Region masks
# ============================================================

def build_region_masks(
    image,
    labels,
):

    h, w = labels.shape

    masks = {}

    person = labels == VOC_CLASSES.index(
        "person"
    )

    animal = np.zeros_like(
        person,
        dtype=bool,
    )

    vehicle = np.zeros_like(
        person,
        dtype=bool,
    )

    plant = labels == VOC_CLASSES.index(
        "pottedplant"
    )

    for name in ANIMAL_CLASSES:
        animal |= (
            labels == VOC_CLASSES.index(name)
        )

    for name in VEHICLE_CLASSES:
        vehicle |= (
            labels == VOC_CLASSES.index(name)
        )

    subject = (
        person
        | animal
        | vehicle
        | plant
    )

    hsv = cv2.cvtColor(
        (np.clip(image, 0, 1) * 255).astype(
            np.uint8
        ),
        cv2.COLOR_RGB2HSV,
    )

    H = hsv[..., 0]
    S = hsv[..., 1]
    V = hsv[..., 2]

    # --------------------------------------------------------
    # Skin
    # --------------------------------------------------------

    skin = (
        person
        & (H <= 25)
        & (H >= 0)
        & (S >= 35)
        & (S <= 190)
        & (V >= 60)
    )

    # --------------------------------------------------------
    # Green
    # --------------------------------------------------------

    green = (
        (H >= 30)
        & (H <= 95)
        & (S >= 45)
        & (V >= 30)
    )

    # --------------------------------------------------------
    # Blue
    # --------------------------------------------------------

    blue = (
        (H >= 80)
        & (H <= 135)
        & (S >= 40)
        & (V >= 40)
    )

    lum = rgb_luminance(image)

    # --------------------------------------------------------
    # Upper bright
    #
    # NOTE:
    # This is intentionally NOT called "sky".
    # DeepLab VOC has no sky class.
    # --------------------------------------------------------

    yy = np.arange(h)[:, None]

    upper = (
        yy < int(h * 0.45)
    )

    upper_bright = (
        upper
        & (lum > np.percentile(lum, 75))
    )

    # --------------------------------------------------------
    # Water heuristic
    # --------------------------------------------------------

    texture = cv2.Laplacian(
        lum,
        cv2.CV_32F,
    )

    texture = np.abs(texture)

    texture_threshold = np.percentile(
        texture,
        55,
    )

    lower_half = (
        yy >= int(h * 0.45)
    )

    water = (
        blue
        & lower_half
        & (texture < texture_threshold)
    )

    masks["person"] = person
    masks["animal"] = animal
    masks["vehicle"] = vehicle
    masks["plant"] = plant
    masks["subject"] = subject

    masks["skin"] = skin
    masks["green"] = green
    masks["blue"] = blue

    masks["upper_bright"] = upper_bright
    masks["water"] = water

    return masks


# ============================================================
# Region statistics
# ============================================================

def region_stats(
    image,
    masks,
):

    hsv = cv2.cvtColor(
        (np.clip(image, 0, 1) * 255).astype(
            np.uint8
        ),
        cv2.COLOR_RGB2HSV,
    )

    sat = (
        hsv[..., 1].astype(np.float32)
        / 255.0
    )

    lum = rgb_luminance(image)

    result = []

    for name, mask in masks.items():

        area = float(np.mean(mask))

        if area <= 0:
            continue

        result.append(
            RegionStats(
                name=name,
                area=area,
                mean_luminance=float(
                    np.mean(lum[mask])
                ),
                mean_saturation=float(
                    np.mean(sat[mask])
                ),
            )
        )

    return result


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

        denoise=0.38,
        sharpen=0.75,

        region_skin_saturation=0.94,
        region_green_saturation=1.00,
        region_water_highlight=0.06,
        region_upper_highlight=0.12,

        tone_strength=0.55,
    ),

    "night": DevelopParams(
        exposure_ev=0.00,
        contrast=1.05,
        saturation=1.03,

        highlight_protection=0.55,
        shadow_lift=0.02,

        subject_exposure=0.05,
        subject_contrast=1.03,
        background_suppression=0.015,

        denoise=0.60,
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

        denoise=0.45,
        sharpen=0.65,

        region_skin_saturation=0.94,
        region_green_saturation=1.00,
        region_water_highlight=0.05,
        region_upper_highlight=0.10,

        tone_strength=0.50,
    ),

    "general": DevelopParams(
        exposure_ev=0.00,
        contrast=1.04,
        saturation=1.00,

        highlight_protection=0.35,
        shadow_lift=0.06,

        subject_exposure=0.04,
        subject_contrast=1.02,
        background_suppression=0.015,

        denoise=0.32,
        sharpen=0.75,

        region_skin_saturation=0.96,
        region_green_saturation=1.01,
        region_water_highlight=0.07,
        region_upper_highlight=0.12,

        tone_strength=0.52,
    ),
}


# ============================================================
# v20 Exposure target
# ============================================================

def base_exposure_target(scene):

    targets = {
        "portrait": 0.22,
        "night": 0.10,
        "sunset": 0.16,
        "landscape": 0.22,
        "city": 0.21,
        "indoor": 0.20,
        "general": 0.21,
    }

    return targets.get(
        scene,
        0.21,
    )


def calculate_exposure_target(
    scene,
    stats,
    subject_mask=None,
):

    target = base_exposure_target(scene)

    # --------------------------------------------------------
    # v20:
    # Use highlight headroom.
    #
    # If the image has substantial room above the p99,
    # we can safely make the image somewhat brighter.
    # --------------------------------------------------------

    if (
        stats.p99 < 0.45
        and stats.highlight_ratio < 0.0005
    ):
        target += 0.025

    elif (
        stats.p99 < 0.55
        and stats.highlight_ratio < 0.001
    ):
        target += 0.015

    # --------------------------------------------------------
    # Subject brightness
    # --------------------------------------------------------

    if (
        scene == "portrait"
        and subject_mask is not None
        and np.any(subject_mask)
    ):

        lum = rgb_luminance(
            subject_mask["image"]
        )

        subject_median = float(
            np.median(
                lum[subject_mask["mask"]]
            )
        )

        if subject_median < 0.16:
            target += 0.025

        elif subject_median < 0.20:
            target += 0.012

    return float(
        np.clip(
            target,
            0.08,
            0.28,
        )
    )


def estimate_exposure_ev(
    stats,
    scene,
    target,
):

    median = max(
        stats.median,
        1e-5,
    )

    p95 = max(
        stats.p95,
        1e-5,
    )

    ev_mid = math.log2(
        target / median
    )

    # Keep highlights from becoming excessive.
    ev_high = math.log2(
        0.78 / p95
    )

    ev = (
        0.70 * ev_mid
        + 0.30 * ev_high
    )

    # v20 headroom bonus.
    if (
        stats.p99 < 0.45
        and stats.highlight_ratio < 0.0005
    ):
        ev += 0.12

    elif (
        stats.p99 < 0.55
        and stats.highlight_ratio < 0.001
    ):
        ev += 0.06

    if scene == "night":
        ev = min(ev, 0.60)

    return float(
        np.clip(
            ev,
            -0.75,
            1.00,
        )
    )


# ============================================================
# Exposure
# ============================================================

def apply_exposure(
    image,
    ev,
):

    linear = srgb_to_linear(image)

    gain = 2.0 ** float(ev)

    linear *= gain

    linear = np.clip(
        linear,
        0.0,
        1.0,
    )

    return linear_to_srgb(
        linear
    )


# ============================================================
# Contrast
# ============================================================

def apply_contrast(
    image,
    factor,
):

    lum = rgb_luminance(image)

    mean_lum = float(
        np.median(lum)
    )

    new_lum = (
        mean_lum
        + (lum - mean_lum) * factor
    )

    ratio = (
        new_lum
        / np.maximum(lum, 1e-6)
    )

    result = image * ratio[..., None]

    return np.clip(
        result,
        0,
        1,
    )


# ============================================================
# Saturation
# ============================================================

def apply_saturation(
    image,
    factor,
):

    lum = rgb_luminance(image)

    result = (
        lum[..., None]
        + (
            image
            - lum[..., None]
        ) * factor
    )

    return np.clip(
        result,
        0,
        1,
    )


# ============================================================
# Tone curve
# ============================================================

def apply_tone(
    image,
    strength,
    shadow_lift,
    highlight_protection,
):

    lum = rgb_luminance(image)

    original = lum.copy()

    # --------------------------------------------------------
    # Gentle shadow lift
    # --------------------------------------------------------

    shadow_weight = np.clip(
        (0.25 - lum) / 0.25,
        0,
        1,
    )

    lifted = (
        lum
        + shadow_weight
        * shadow_lift
        * (0.25 - lum)
    )

    # --------------------------------------------------------
    # Gentle highlight rolloff
    # --------------------------------------------------------

    highlight_weight = np.clip(
        (lum - 0.70) / 0.30,
        0,
        1,
    )

    rolled = (
        lifted
        - highlight_weight
        * highlight_protection
        * np.maximum(
            lifted - 0.70,
            0,
        )
        * 0.35
    )

    # --------------------------------------------------------
    # Mild S-curve
    #
    # v20 deliberately keeps this weak.
    # --------------------------------------------------------

    curve = (
        rolled
        + strength
        * 0.12
        * rolled
        * (1.0 - rolled)
        * (
            2.0 * rolled - 1.0
        )
    )

    new_lum = np.clip(
        curve,
        0,
        1,
    )

    ratio = (
        new_lum
        / np.maximum(
            original,
            1e-5,
        )
    )

    result = image * ratio[..., None]

    return np.clip(
        result,
        0,
        1,
    )


# ============================================================
# Region processing
# ============================================================

def apply_region_processing(
    image,
    masks,
    params,
):

    result = image.copy()

    lum = rgb_luminance(result)

    # --------------------------------------------------------
    # Subject exposure / contrast
    # --------------------------------------------------------

    subject = masks.get(
        "subject",
        np.zeros(
            lum.shape,
            dtype=bool,
        ),
    )

    if np.any(subject):

        gain = 2.0 ** params.subject_exposure

        result[subject] *= gain

        result[subject] = np.clip(
            result[subject],
            0,
            1,
        )

        subject_lum = rgb_luminance(
            result
        )

        median_subject = float(
            np.median(
                subject_lum[subject]
            )
        )

        new_lum = (
            median_subject
            + (
                subject_lum
                - median_subject
            )
            * params.subject_contrast
        )

        ratio = (
            new_lum
            / np.maximum(
                subject_lum,
                1e-5,
            )
        )

        result[subject] *= ratio[
            subject,
            None,
        ]

    # --------------------------------------------------------
    # Skin
    # --------------------------------------------------------

    skin = masks.get("skin")

    if skin is not None and np.any(skin):

        lum = rgb_luminance(result)

        result[skin] = (
            lum[skin, None]
            + (
                result[skin]
                - lum[skin, None]
            )
            * params.region_skin_saturation
        )

    # --------------------------------------------------------
    # Green
    # --------------------------------------------------------

    green = masks.get("green")

    if green is not None and np.any(green):

        lum = rgb_luminance(result)

        result[green] = (
            lum[green, None]
            + (
                result[green]
                - lum[green, None]
            )
            * params.region_green_saturation
        )

    # --------------------------------------------------------
    # Water highlight protection
    # --------------------------------------------------------

    water = masks.get("water")

    if water is not None and np.any(water):

        lum = rgb_luminance(result)

        weight = np.clip(
            (lum - 0.65) / 0.35,
            0,
            1,
        )

        weight *= (
            water
            * params.region_water_highlight
        )

        new_lum = (
            lum
            - weight
            * np.maximum(
                lum - 0.65,
                0,
            )
        )

        ratio = (
            new_lum
            / np.maximum(
                lum,
                1e-5,
            )
        )

        result *= ratio[..., None]

    # --------------------------------------------------------
    # Upper bright highlight protection
    # --------------------------------------------------------

    upper = masks.get(
        "upper_bright"
    )

    if upper is not None and np.any(upper):

        lum = rgb_luminance(result)

        weight = np.clip(
            (lum - 0.70) / 0.30,
            0,
            1,
        )

        weight *= (
            upper
            * params.region_upper_highlight
        )

        new_lum = (
            lum
            - weight
            * np.maximum(
                lum - 0.70,
                0,
            )
        )

        ratio = (
            new_lum
            / np.maximum(
                lum,
                1e-5,
            )
        )

        result *= ratio[..., None]

    # --------------------------------------------------------
    # Background suppression
    # --------------------------------------------------------

    h, w = lum.shape

    background = ~subject

    if np.any(background):

        bg_lum = rgb_luminance(result)

        bg_sat = (
            result
            - bg_lum[..., None]
        )

        result[background] = (
            bg_lum[background, None]
            + bg_sat[background]
            * (
                1.0
                - params.background_suppression
            )
        )

    return np.clip(
        result,
        0,
        1,
    )


# ============================================================
# v20 Denoise
# ============================================================

def calculate_denoise_strength(
    base_strength,
    iso_factor,
):

    """
    v20:

    Instead of simply applying the scene profile value,
    scale denoise according to ISO.

    ISO 1250 -> around 0.35-0.40
    ISO 3200 -> around 0.50
    ISO 6400 -> around 0.65
    """

    iso_component = np.clip(
        (iso_factor - 2.0) / 3.0,
        0,
        1,
    )

    strength = (
        0.70 * base_strength
        + 0.30 * (
            0.20
            + 0.60 * iso_component
        )
    )

    return float(
        np.clip(
            strength,
            0.15,
            0.70,
        )
    )


def denoise_luminance(
    image,
    strength,
):

    if strength <= 0.01:
        return image.copy()

    rgb8 = (
        np.clip(
            image,
            0,
            1,
        )
        * 255
    ).astype(np.uint8)

    ycrcb = cv2.cvtColor(
        rgb8,
        cv2.COLOR_RGB2YCrCb,
    )

    y = ycrcb[..., 0]

    # --------------------------------------------------------
    # Luminance only.
    #
    # This avoids the v19 problem where RGB bilateral
    # filtering reduced apparent saturation dramatically.
    # --------------------------------------------------------

    sigma_color = (
        8.0
        + 22.0 * strength
    )

    sigma_space = (
        2.0
        + 3.0 * strength
    )

    filtered_y = cv2.bilateralFilter(
        y,
        d=5,
        sigmaColor=sigma_color,
        sigmaSpace=sigma_space,
    )

    mix = np.clip(
        strength,
        0,
        1,
    )

    y_new = (
        y.astype(np.float32)
        * (1.0 - mix)
        + filtered_y.astype(np.float32)
        * mix
    )

    ycrcb[..., 0] = np.clip(
        y_new,
        0,
        255,
    ).astype(np.uint8)

    result8 = cv2.cvtColor(
        ycrcb,
        cv2.COLOR_YCrCb2RGB,
    )

    return (
        result8.astype(np.float32)
        / 255.0
    )


# ============================================================
# Sharpen
# ============================================================

def sharpen_luminance(
    image,
    strength,
):

    if strength <= 0:
        return image.copy()

    lum = rgb_luminance(image)

    blur = cv2.GaussianBlur(
        lum,
        (0, 0),
        sigmaX=1.1,
    )

    detail = (
        lum - blur
    )

    amount = (
        0.65
        * strength
    )

    new_lum = np.clip(
        lum
        + detail * amount,
        0,
        1,
    )

    ratio = (
        new_lum
        / np.maximum(
            lum,
            1e-5,
        )
    )

    result = (
        image
        * ratio[..., None]
    )

    return np.clip(
        result,
        0,
        1,
    )


# ============================================================
# Search
# ============================================================

def evaluate_candidate(
    image,
    scene,
    target,
    subject_mask=None,
):

    stats = calculate_stats(
        image
    )

    # --------------------------------------------------------
    # Main exposure error
    # --------------------------------------------------------

    median_error = abs(
        stats.median
        - target
    )

    # --------------------------------------------------------
    # Highlight penalty
    # --------------------------------------------------------

    highlight_penalty = (
        stats.highlight_ratio
        * 5.0
    )

    if stats.p99 > 0.85:
        highlight_penalty += (
            stats.p99 - 0.85
        ) * 2.0

    # --------------------------------------------------------
    # Shadow penalty
    # --------------------------------------------------------

    shadow_weight = (
        0.25
        if scene == "night"
        else 1.0
    )

    shadow_penalty = (
        stats.shadow_ratio
        * shadow_weight
    )

    # --------------------------------------------------------
    # Contrast
    # --------------------------------------------------------

    contrast_target = {
        "portrait": 0.11,
        "night": 0.09,
        "sunset": 0.13,
        "landscape": 0.14,
        "city": 0.13,
        "indoor": 0.11,
        "general": 0.11,
    }.get(scene, 0.11)

    contrast_penalty = abs(
        stats.contrast
        - contrast_target
    ) * 0.50

    # --------------------------------------------------------
    # Saturation
    # --------------------------------------------------------

    saturation_penalty = 0.0

    if stats.saturation_ratio > 0.30:
        saturation_penalty = (
            stats.saturation_ratio
            - 0.30
        ) * 0.8

    # --------------------------------------------------------
    # Subject brightness
    # --------------------------------------------------------

    subject_penalty = 0.0

    if (
        scene == "portrait"
        and subject_mask is not None
        and np.any(subject_mask)
    ):

        lum = rgb_luminance(image)

        subject_median = float(
            np.median(
                lum[subject_mask]
            )
        )

        subject_target = 0.23

        subject_penalty = (
            abs(
                subject_median
                - subject_target
            )
            * 0.8
        )

    score = (
        median_error * 0.65
        + highlight_penalty * 4.0
        + shadow_penalty * 1.0
        + contrast_penalty
        + saturation_penalty
        + subject_penalty
    )

    return float(score), stats


def search_parameters(
    image,
    scene,
    estimated_ev,
    target,
    subject_mask=None,
):

    print(
        "[INFO] Running parameter search..."
    )

    small = resize_keep_aspect(
        image,
        512,
    )

    # --------------------------------------------------------
    # v20:
    # Search around the estimated EV.
    # --------------------------------------------------------

    ev_candidates = [
        estimated_ev - 0.30,
        estimated_ev - 0.18,
        estimated_ev - 0.06,
        estimated_ev + 0.06,
        estimated_ev + 0.18,
        estimated_ev + 0.30,
    ]

    ev_candidates = [
        float(
            np.clip(
                x,
                -0.75,
                1.00,
            )
        )
        for x in ev_candidates
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

    best = None

    for ev in ev_candidates:

        exposed = apply_exposure(
            small,
            ev,
        )

        for contrast in contrast_candidates:

            contrasted = apply_contrast(
                exposed,
                contrast,
            )

            for saturation in saturation_candidates:

                candidate = apply_saturation(
                    contrasted,
                    saturation,
                )

                mask_small = None

                if subject_mask is not None:

                    mask_small = cv2.resize(
                        subject_mask.astype(
                            np.uint8
                        ),
                        (
                            candidate.shape[1],
                            candidate.shape[0],
                        ),
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(bool)

                score, stats = evaluate_candidate(
                    candidate,
                    scene,
                    target,
                    mask_small,
                )

                if (
                    best is None
                    or score < best["score"]
                ):
                    best = {
                        "score": score,
                        "ev": ev,
                        "contrast": contrast,
                        "saturation": saturation,
                        "stats": stats,
                    }

    return best


# ============================================================
# Developer
# ============================================================

class AutoDeveloper:

    def __init__(
        self,
        output_dir,
        device="auto",
        debug=False,
    ):

        self.output_dir = Path(
            output_dir
        )

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.debug = debug

        self.debug_root = (
            self.output_dir / "debug"
        )

        if self.debug:
            self.debug_root.mkdir(
                parents=True,
                exist_ok=True,
            )

        self.segmenter = SemanticSegmenter(
            device=device
        )

    # --------------------------------------------------------
    # Debug
    # --------------------------------------------------------

    def save_stage(
        self,
        debug_dir,
        name,
        image,
        color_space,
        stats_dict,
    ):

        if not self.debug:
            return

        path = (
            debug_dir
            / f"{name}.jpg"
        )

        save_rgb(
            path,
            image,
        )

        stats = calculate_stats(
            image
        )

        stats_dict[name] = {
            "color_space": color_space,
            **asdict(stats),
        }

        print_stats(
            name,
            color_space,
            stats,
        )

    # --------------------------------------------------------
    # Process
    # --------------------------------------------------------

    def process(self, path):

        print(
            f"\n[INFO] Processing: {path}"
        )

        metadata = get_metadata(
            path
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

        camera_profile = get_camera_profile(
            path,
            metadata,
        )

        linear_rgb, conversion_method = (
            raw_to_linear_rgb(path)
        )

        debug_dir = (
            self.debug_root
            / Path(path).stem
        )

        if self.debug:
            if debug_dir.exists():
                shutil.rmtree(
                    debug_dir
                )

            debug_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

        stage_stats = {}

        # ----------------------------------------------------
        # 01 RAW linear RGB
        # ----------------------------------------------------

        if self.debug:

            self.save_stage(
                debug_dir,
                "01_raw_linear",
                linear_to_srgb(
                    linear_rgb
                ),
                "linear_rgb",
                stage_stats,
            )

        # ----------------------------------------------------
        # 02 Analysis sRGB
        # ----------------------------------------------------

        analysis_srgb = linear_to_srgb(
            linear_rgb
        )

        if self.debug:

            self.save_stage(
                debug_dir,
                "02_analysis_srgb",
                analysis_srgb,
                "sRGB",
                stage_stats,
            )

        # ----------------------------------------------------
        # Metadata / shooting
        # ----------------------------------------------------

        stats = calculate_stats(
            analysis_srgb
        )

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
        # Segmentation
        # ----------------------------------------------------

        labels, segmentation_conf = (
            self.segmenter.predict(
                analysis_srgb
            )
        )

        saliency = compute_saliency(
            analysis_srgb
        )

        subjects = rank_subjects(
            labels,
            segmentation_conf,
            saliency,
            analysis_srgb,
        )

        print("[INFO] Subjects:")

        for s in subjects:
            print(
                f"  {s.class_name}: "
                f"score={s.score:.3f}, "
                f"area={s.area:.3f}, "
                f"confidence={s.confidence:.3f}"
            )

        scene_result = classify_scene(
            stats,
            shooting,
            subjects,
        )

        print(
            f"[INFO] Scene: "
            f"{scene_result.scene} "
            f"(confidence="
            f"{scene_result.confidence:.3f})"
        )

        masks = build_region_masks(
            analysis_srgb,
            labels,
        )

        regions = region_stats(
            analysis_srgb,
            masks,
        )

        # ----------------------------------------------------
        # Debug segmentation
        # ----------------------------------------------------

        if self.debug:

            seg_rgb = np.zeros_like(
                analysis_srgb
            )

            # Simple visualization.
            for idx in range(
                1,
                len(VOC_CLASSES)
            ):

                mask = labels == idx

                if not np.any(mask):
                    continue

                value = (
                    idx
                    / max(
                        len(VOC_CLASSES) - 1,
                        1,
                    )
                )

                seg_rgb[mask] = [
                    value,
                    1.0 - value,
                    0.5,
                ]

            save_rgb(
                debug_dir
                / "03_segmentation.jpg",
                seg_rgb,
            )

            save_gray(
                debug_dir
                / "04_segmentation_confidence.jpg",
                segmentation_conf,
            )

            save_gray(
                debug_dir
                / "05_saliency.jpg",
                saliency,
            )

            for name, mask in masks.items():

                save_gray(
                    debug_dir
                    / f"mask_{name}.jpg",
                    mask.astype(
                        np.float32
                    ),
                )

        # ----------------------------------------------------
        # v20 exposure target
        # ----------------------------------------------------

        subject_mask_info = None

        if (
            scene_result.scene == "portrait"
            and np.any(masks["person"])
        ):
            subject_mask_info = {
                "image": analysis_srgb,
                "mask": masks["person"],
            }

        target = calculate_exposure_target(
            scene_result.scene,
            stats,
            subject_mask_info,
        )

        estimated_ev = estimate_exposure_ev(
            stats,
            scene_result.scene,
            target,
        )

        print(
            f"[INFO] Exposure target: "
            f"{target:.3f}"
        )

        print(
            f"[INFO] Estimated EV: "
            f"{estimated_ev:+.3f}"
        )

        # ----------------------------------------------------
        # Search
        # ----------------------------------------------------

        best = search_parameters(
            analysis_srgb,
            scene_result.scene,
            estimated_ev,
            target,
            masks["person"]
            if scene_result.scene == "portrait"
            else masks["subject"],
        )

        print(
            f"[INFO] Search score: "
            f"{best['score']:.6f}"
        )

        print(
            f"[INFO] Selected EV: "
            f"{best['ev']:+.3f}"
        )

        print(
            f"[INFO] Selected contrast: "
            f"{best['contrast']:.3f}"
        )

        print(
            f"[INFO] Selected saturation: "
            f"{best['saturation']:.3f}"
        )

        profile = SCENE_PROFILES[
            scene_result.scene
        ]

        # ----------------------------------------------------
        # Build final parameters
        # ----------------------------------------------------

        denoise_strength = (
            calculate_denoise_strength(
                profile.denoise,
                shooting.iso_factor,
            )
        )

        params = DevelopParams(
            exposure_ev=best["ev"],
            contrast=best["contrast"],
            saturation=best["saturation"],

            highlight_protection=(
                profile.highlight_protection
            ),

            shadow_lift=(
                profile.shadow_lift
            ),

            subject_exposure=(
                profile.subject_exposure
            ),

            subject_contrast=(
                profile.subject_contrast
            ),

            background_suppression=(
                profile.background_suppression
            ),

            denoise=denoise_strength,

            sharpen=profile.sharpen,

            region_skin_saturation=(
                profile.region_skin_saturation
            ),

            region_green_saturation=(
                profile.region_green_saturation
            ),

            region_water_highlight=(
                profile.region_water_highlight
            ),

            region_upper_highlight=(
                profile.region_upper_highlight
            ),

            tone_strength=profile.tone_strength,
        )

        print(
            f"[INFO] Denoise strength: "
            f"{params.denoise:.3f}"
        )

        print(
            f"[INFO] Tone strength: "
            f"{params.tone_strength:.3f}"
        )

        print(
            f"[INFO] Shadow lift: "
            f"{params.shadow_lift:.3f}"
        )

        print(
            f"[INFO] Highlight protection: "
            f"{params.highlight_protection:.3f}"
        )

        # ----------------------------------------------------
        # Development
        # ----------------------------------------------------

        current = analysis_srgb

        # 03 exposure
        current = apply_exposure(
            current,
            params.exposure_ev,
        )

        self.save_stage(
            debug_dir,
            "after_exposure",
            current,
            "sRGB",
            stage_stats,
        )

        # 04 contrast
        current = apply_contrast(
            current,
            params.contrast,
        )

        self.save_stage(
            debug_dir,
            "after_contrast",
            current,
            "sRGB",
            stage_stats,
        )

        # 05 saturation
        current = apply_saturation(
            current,
            params.saturation,
        )

        self.save_stage(
            debug_dir,
            "after_saturation",
            current,
            "sRGB",
            stage_stats,
        )

        # 06 tone
        current = apply_tone(
            current,
            params.tone_strength,
            params.shadow_lift,
            params.highlight_protection,
        )

        self.save_stage(
            debug_dir,
            "after_tone",
            current,
            "sRGB",
            stage_stats,
        )

        # 07 region
        current = apply_region_processing(
            current,
            masks,
            params,
        )

        self.save_stage(
            debug_dir,
            "after_region",
            current,
            "sRGB",
            stage_stats,
        )

        # 08 denoise
        current = denoise_luminance(
            current,
            params.denoise,
        )

        self.save_stage(
            debug_dir,
            "after_denoise",
            current,
            "sRGB",
            stage_stats,
        )

        # 09 sharpen
        current = sharpen_luminance(
            current,
            params.sharpen,
        )

        self.save_stage(
            debug_dir,
            "after_sharpen",
            current,
            "sRGB",
            stage_stats,
        )

        # ----------------------------------------------------
        # Final linear RGB
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

            final_linear_display = (
                linear_to_srgb(
                    final_linear
                )
            )

            self.save_stage(
                debug_dir,
                "final_linear",
                final_linear_display,
                "linear_rgb",
                stage_stats,
            )

        # ----------------------------------------------------
        # Final JPEG
        # ----------------------------------------------------

        output_path = (
            self.output_dir
            / f"{Path(path).stem}.jpg"
        )

        save_rgb(
            output_path,
            final_srgb,
        )

        if self.debug:

            self.save_stage(
                debug_dir,
                "final_srgb",
                final_srgb,
                "sRGB",
                stage_stats,
            )

            report = {
                "version": VERSION,

                "input": str(path),
                "output": str(output_path),

                "conversion_method": (
                    conversion_method
                ),

                "camera": asdict(
                    camera_profile
                ),

                "metadata": asdict(
                    metadata
                ),

                "stats": {
                    "analysis": asdict(stats),
                    "stages": stage_stats,
                },

                "shooting": asdict(
                    shooting
                ),

                "scene": asdict(
                    scene_result
                ),

                "subjects": [
                    asdict(s)
                    for s in subjects
                ],

                "regions": [
                    asdict(r)
                    for r in regions
                ],

                "exposure": {
                    "target": target,
                    "estimated_ev": estimated_ev,
                },

                "search": {
                    "score": best["score"],
                    "ev": best["ev"],
                    "contrast": best["contrast"],
                    "saturation": best["saturation"],
                },

                "selected_parameters": asdict(
                    params
                ),
            }

            with open(
                debug_dir / "report.json",
                "w",
                encoding="utf-8",
            ) as f:

                json.dump(
                    report,
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

            print(
                f"[DEBUG] Debug output: "
                f"{debug_dir}"
            )

        print(
            f"[INFO] Saved: "
            f"{output_path}"
        )

        return output_path


# ============================================================
# RAW collection
# ============================================================

def collect_raw_files(
    input_path,
):

    path = Path(
        input_path
    )

    if path.is_file():

        if (
            path.suffix.lower()
            in RAW_EXTENSIONS
        ):
            return [path]

        return []

    files = []

    for p in path.rglob("*"):

        if (
            p.is_file()
            and p.suffix.lower()
            in RAW_EXTENSIONS
        ):
            files.append(p)

    return sorted(files)


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Automatic RAW photo developer v20"
        )
    )

    parser.add_argument(
        "input",
        help="RAW file or directory",
    )

    parser.add_argument(
        "-o",
        "--output",
        default="developed",
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
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help=(
            "Save intermediate images, "
            "statistics and JSON report"
        ),
    )

    args = parser.parse_args()

    raw_files = collect_raw_files(
        args.input
    )

    print(
        f"[INFO] Found "
        f"{len(raw_files)} RAW files."
    )

    if not raw_files:
        print(
            "[ERROR] No RAW files found."
        )
        return 1

    if args.debug:
        print(
            "[INFO] DEBUG MODE ENABLED"
        )

    developer = AutoDeveloper(
        output_dir=args.output,
        device=args.device,
        debug=args.debug,
    )

    success = 0
    failed = 0

    for path in raw_files:

        try:

            developer.process(
                path
            )

            success += 1

        except Exception as e:

            failed += 1

            print(
                f"\n[ERROR] Failed: {path}"
            )

            print(
                f"        "
                f"{type(e).__name__}: {e}"
            )

    print(
        "\n[INFO] Finished."
    )

    print(
        f"[INFO] Success: {success}"
    )

    print(
        f"[INFO] Failed : {failed}"
    )

    return (
        0
        if failed == 0
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(
        main()
    )