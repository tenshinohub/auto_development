#!/usr/bin/env python3

import argparse
import json
import math
import os
import shutil
import subprocess
import warnings

from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np
import rawpy
from PIL import Image, ExifTags

import torch
from torchvision import models


VERSION = "v21"


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


XYZ_TO_SRGB = np.array(
    [
        [3.2406, -1.5372, -0.4986],
        [-0.9689, 1.8758, 0.0415],
        [0.0557, -0.2040, 1.0570],
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

    camera_whitebalance: list = None
    color_matrix: list = None
    rgb_xyz_matrix: list = None

    raw_width: int = 0
    raw_height: int = 0

    lens_make: str = ""
    lens_model: str = ""

    metadata_source: str = ""
    libraw_version: str = ""


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
    warm_ratio: float

    contrast: float
    mean_luminance: float

    rgb_mean_r: float
    rgb_mean_g: float
    rgb_mean_b: float

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


@dataclass
class ExposureModel:
    global_target: float
    subject_target: float

    highlight_soft_limit: float
    highlight_hard_limit: float

    shadow_target: float

    subject_weight: float
    global_weight: float
    highlight_weight: float
    shadow_weight: float
    background_weight: float


# ============================================================
# Utility
# ============================================================

def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def safe_float(value, default=0.0):
    try:
        if value is None:
            return default

        if isinstance(value, str):
            value = value.strip()

        return float(value)
    except Exception:
        return default


def resize_max(image, max_dim):
    h, w = image.shape[:2]

    if max(h, w) <= max_dim:
        return image

    scale = max_dim / max(h, w)

    return cv2.resize(
        image,
        (int(w * scale), int(h * scale)),
        interpolation=cv2.INTER_AREA,
    )


def resize_mask(mask, shape):
    h, w = shape[:2]

    return cv2.resize(
        mask.astype(np.float32),
        (w, h),
        interpolation=cv2.INTER_NEAREST,
    )


def normalize01(x):
    x = np.asarray(x, dtype=np.float32)

    mn = np.min(x)
    mx = np.max(x)

    if mx - mn < 1e-8:
        return np.zeros_like(x)

    return np.clip((x - mn) / (mx - mn), 0.0, 1.0)


def rgb_luminance(rgb):
    return (
        0.2126 * rgb[..., 0]
        + 0.7152 * rgb[..., 1]
        + 0.0722 * rgb[..., 2]
    )


def srgb_to_linear(srgb):
    srgb = np.clip(srgb, 0.0, 1.0)

    return np.where(
        srgb <= 0.04045,
        srgb / 12.92,
        ((srgb + 0.055) / 1.055) ** 2.4,
    )


def linear_to_srgb(linear):
    linear = np.clip(linear, 0.0, 1.0)

    return np.where(
        linear <= 0.0031308,
        linear * 12.92,
        1.055 * np.power(linear, 1.0 / 2.4) - 0.055,
    )


def apply_ev(linear_rgb, ev):
    gain = 2.0 ** ev

    return np.clip(linear_rgb * gain, 0.0, 1.0)


def rgb_to_hsv_float(rgb):
    rgb8 = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)

    return cv2.cvtColor(rgb8, cv2.COLOR_RGB2HSV).astype(np.float32)


def mask_mean(image, mask):
    if mask is None:
        return float(np.mean(image))

    mask = mask > 0.5

    if not np.any(mask):
        return float(np.mean(image))

    return float(np.mean(image[mask]))


def mask_median(image, mask):
    if mask is None:
        return float(np.median(image))

    mask = mask > 0.5

    if not np.any(mask):
        return float(np.median(image))

    return float(np.median(image[mask]))


def mask_area(mask):
    if mask is None:
        return 0.0

    return float(np.mean(mask > 0.5))


# ============================================================
# Statistics
# ============================================================

def calculate_stats(rgb):
    rgb = np.clip(rgb.astype(np.float32), 0.0, 1.0)

    lum = rgb_luminance(rgb)

    values = lum.reshape(-1)

    mean = float(np.mean(values))
    median = float(np.median(values))

    p01, p05, p25, p75, p95, p99 = np.percentile(
        values,
        [1, 5, 25, 75, 95, 99],
    )

    shadow_ratio = float(np.mean(values < 0.02))
    highlight_ratio = float(np.mean(values > 0.98))

    dynamic_range = float(
        math.log10(max(p95, 1e-6) / max(p05, 1e-6))
    )

    hsv = rgb_to_hsv_float(rgb)

    saturation_ratio = float(
        np.mean(hsv[..., 1] >= 200)
    )

    gray = np.clip(lum * 255.0, 0, 255).astype(np.uint8)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    magnitude = cv2.magnitude(gx, gy)

    edge_density = float(
        np.mean(magnitude > 40.0)
    )

    warm_ratio = float(
        np.mean(
            (rgb[..., 0] > rgb[..., 2] * 1.15)
            & (rgb[..., 0] > rgb[..., 1] * 1.02)
        )
    )

    contrast = float(np.std(lum))

    rgb_mean = np.mean(rgb.reshape(-1, 3), axis=0)

    r_mean = float(rgb_mean[0])
    g_mean = float(rgb_mean[1])
    b_mean = float(rgb_mean[2])

    rg_ratio = r_mean / max(g_mean, 1e-6)
    gb_ratio = g_mean / max(b_mean, 1e-6)

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
        warm_ratio=warm_ratio,

        contrast=contrast,
        mean_luminance=mean,

        rgb_mean_r=r_mean,
        rgb_mean_g=g_mean,
        rgb_mean_b=b_mean,

        rg_ratio=rg_ratio,
        gb_ratio=gb_ratio,
    )


def print_stats(name, stats, color_space):
    print()
    print(f"[DEBUG] ===== {name} ({color_space}) =====")

    print(
        f"  min/max      : "
        f"{stats.p01:.6f} / {stats.p99:.6f}"
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
        f"{stats.rgb_mean_r:.6f}, "
        f"{stats.rgb_mean_g:.6f}, "
        f"{stats.rgb_mean_b:.6f}"
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

def exiftool_metadata(path):
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
            capture_output=True,
            text=True,
            timeout=10,
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


def pillow_metadata(path):
    result = {}

    try:
        with Image.open(path) as img:
            exif = img.getexif()

            for key, value in exif.items():
                tag = ExifTags.TAGS.get(key, key)
                result[tag] = value

    except Exception:
        pass

    return result


def get_metadata(path):
    exiftool = exiftool_metadata(path)
    pillow = pillow_metadata(path)

    exiftool = exiftool or {}
    pillow = pillow or {}

    def get_value(*keys):
        for key in keys:
            if key in exiftool:
                return exiftool[key]

            if key in pillow:
                return pillow[key]

        return ""

    return ExifMetadata(
        make=str(
            get_value("Make")
        ),

        model=str(
            get_value(
                "CameraModelName",
                "UniqueCameraModel",
                "Model",
            )
        ),

        lens_make=str(
            get_value("LensMake")
        ),

        lens_model=str(
            get_value("LensModel")
        ),

        iso=safe_float(
            get_value("ISO")
        ),

        exposure_time=safe_float(
            get_value("ExposureTime")
        ),

        f_number=safe_float(
            get_value("FNumber")
        ),

        focal_length=safe_float(
            get_value("FocalLength")
        ),

        white_balance=str(
            get_value("WhiteBalance")
        ),

        color_temperature=safe_float(
            get_value("ColorTemperature")
        ),

        color_space=str(
            get_value("ColorSpace")
        ),

        source="ExifTool"
        if exiftool
        else "Pillow",
    )


def detect_camera_family(make, model):
    text = f"{make} {model}".lower()

    if "canon" in text:
        return "Canon"

    if "nikon" in text:
        return "Nikon"

    if "sony" in text:
        return "Sony"

    if "fujifilm" in text or "fuji" in text:
        return "Fujifilm"

    if "panasonic" in text or "lumix" in text:
        return "Panasonic"

    if "olympus" in text or "om system" in text:
        return "Olympus"

    if "leica" in text:
        return "Leica"

    if "pentax" in text:
        return "Pentax"

    if "ricoh" in text:
        return "Ricoh"

    if "sigma" in text:
        return "Sigma"

    if "hasselblad" in text:
        return "Hasselblad"

    return "Unknown"


def get_camera_profile(raw, metadata):
    try:
        black_level = float(
            np.mean(np.asarray(raw.black_level_per_channel))
        )
    except Exception:
        black_level = 0.0

    try:
        white_level = float(raw.white_level)
    except Exception:
        white_level = 0.0

    try:
        camera_wb = np.asarray(
            raw.camera_whitebalance,
            dtype=np.float32,
        ).tolist()
    except Exception:
        camera_wb = []

    try:
        color_matrix = np.asarray(
            raw.color_matrix,
            dtype=np.float32,
        ).tolist()
    except Exception:
        color_matrix = []

    try:
        rgb_xyz_matrix = np.asarray(
            raw.rgb_xyz_matrix,
            dtype=np.float32,
        ).tolist()
    except Exception:
        rgb_xyz_matrix = []

    try:
        raw_width = int(raw.sizes.width)
        raw_height = int(raw.sizes.height)
    except Exception:
        raw_width = 0
        raw_height = 0

    try:
        libraw_version = str(rawpy.libraw_version)
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

        black_level=black_level,
        white_level=white_level,

        camera_whitebalance=camera_wb,
        color_matrix=color_matrix,
        rgb_xyz_matrix=rgb_xyz_matrix,

        raw_width=raw_width,
        raw_height=raw_height,

        lens_make=metadata.lens_make,
        lens_model=metadata.lens_model,

        metadata_source=metadata.source,
        libraw_version=libraw_version,
    )


# ============================================================
# RAW conversion
# ============================================================

def raw_to_linear_rgb(raw):
    """
    v21 RAW pipeline

    LibRaw:
        sRGB primaries
        gamma=(1,1)

    The resulting 16-bit values are treated as linear RGB
    because gamma is explicitly disabled.

    Important:
        Do NOT apply srgb_to_linear() here.
    """

    try:
        image16 = raw.postprocess(
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

        linear_rgb = (
            image16.astype(np.float32) / 65535.0
        )

        return np.clip(linear_rgb, 0.0, 1.0), "LibRaw sRGB / gamma=1"

    except Exception as exc:
        warnings.warn(
            f"RAW linear conversion failed: {exc}"
        )

        image16 = raw.postprocess(
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

        linear_rgb = (
            image16.astype(np.float32) / 65535.0
        )

        return np.clip(linear_rgb, 0.0, 1.0), (
            "LibRaw sRGB / gamma=1 / fallback"
        )


# ============================================================
# Shooting condition
# ============================================================

def analyze_shooting(metadata):
    iso = max(metadata.iso, 100.0)

    iso_factor = clamp(
        math.sqrt(iso / 100.0),
        1.0,
        5.0,
    )

    low_light = (
        iso >= 1600
        or (
            metadata.exposure_time > 0
            and metadata.exposure_time < 1 / 30
        )
    )

    motion_risk = 0.0

    if metadata.exposure_time > 0:
        if metadata.exposure_time >= 1 / 30:
            motion_risk = 0.1
        elif metadata.exposure_time >= 1 / 60:
            motion_risk = 0.25
        elif metadata.exposure_time >= 1 / 125:
            motion_risk = 0.45
        elif metadata.exposure_time >= 1 / 250:
            motion_risk = 0.7
        else:
            motion_risk = 0.9

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

class SemanticSegmenter:

    def __init__(self, device="auto"):
        if device == "auto":
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        if device == "cuda" and not torch.cuda.is_available():
            print(
                "[WARN] CUDA requested but unavailable. "
                "Using CPU."
            )

            device = "cpu"

        self.device = torch.device(device)

        print(
            f"[INFO] Loading DeepLabV3 on "
            f"{self.device}"
        )

        try:
            weights = (
                models.segmentation.DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
            )

            self.model = (
                models.segmentation.deeplabv3_mobilenet_v3_large(
                    weights=weights
                )
            )

            self.transform = weights.transforms()

        except Exception:
            self.model = (
                models.segmentation.deeplabv3_mobilenet_v3_large(
                    pretrained=True
                )
            )

            self.transform = None

        self.model.to(self.device)
        self.model.eval()

    def predict(self, rgb):
        original_h, original_w = rgb.shape[:2]

        small = resize_max(rgb, 768)

        image = np.clip(
            small * 255.0,
            0,
            255,
        ).astype(np.uint8)

        pil = Image.fromarray(image)

        if self.transform is not None:
            tensor = self.transform(pil)
        else:
            arr = (
                np.asarray(pil)
                .astype(np.float32)
                / 255.0
            )

            arr = np.transpose(
                arr,
                (2, 0, 1),
            )

            tensor = torch.from_numpy(arr)

        tensor = tensor.unsqueeze(0).to(self.device)

        with torch.inference_mode():
            result = self.model(tensor)["out"][0]

        probabilities = torch.softmax(
            result,
            dim=0,
        )

        labels_small = torch.argmax(
            probabilities,
            dim=0,
        ).cpu().numpy()

        confidence_small = torch.max(
            probabilities,
            dim=0,
        ).values.cpu().numpy()

        labels = cv2.resize(
            labels_small.astype(np.uint8),
            (original_w, original_h),
            interpolation=cv2.INTER_NEAREST,
        )

        confidence = cv2.resize(
            confidence_small.astype(np.float32),
            (original_w, original_h),
            interpolation=cv2.INTER_LINEAR,
        )

        return labels, confidence


# ============================================================
# Saliency
# ============================================================

def calculate_saliency(rgb):
    lum = rgb_luminance(rgb)

    local_mean = cv2.GaussianBlur(
        lum,
        (0, 0),
        sigmaX=15,
    )

    local_contrast = np.abs(
        lum - local_mean
    )

    local_contrast = normalize01(
        local_contrast
    )

    gray = np.clip(
        lum * 255,
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

    edge = cv2.magnitude(gx, gy)

    edge = normalize01(edge)

    hsv = rgb_to_hsv_float(rgb)

    saturation = hsv[..., 1] / 255.0

    brightness = lum

    brightness_mean = cv2.GaussianBlur(
        brightness,
        (0, 0),
        sigmaX=15,
    )

    brightness_distinct = np.abs(
        brightness - brightness_mean
    )

    brightness_distinct = normalize01(
        brightness_distinct
    )

    h, w = lum.shape

    yy, xx = np.mgrid[0:h, 0:w]

    cx = w * 0.5
    cy = h * 0.5

    dx = (xx - cx) / max(w * 0.5, 1)
    dy = (yy - cy) / max(h * 0.5, 1)

    center = np.exp(
        -(dx * dx + dy * dy) / 0.8
    )

    saliency = (
        0.30 * local_contrast
        + 0.25 * edge
        + 0.15 * saturation
        + 0.20 * brightness_distinct
        + 0.10 * center
    )

    return np.clip(
        saliency,
        0.0,
        1.0,
    ).astype(np.float32)


# ============================================================
# Subject ranking
# ============================================================

def rank_subjects(
    labels,
    confidence,
    saliency,
    rgb,
):
    h, w = labels.shape

    subjects = []

    class_priors = {
        "person": 1.15,

        "cat": 1.05,
        "dog": 1.05,
        "bird": 1.05,
        "horse": 1.05,
        "cow": 1.05,
        "sheep": 1.05,

        "car": 1.00,
        "bus": 1.00,
        "train": 1.00,
        "boat": 1.00,
        "bicycle": 1.00,
        "motorbike": 1.00,

        "pottedplant": 0.90,
        "bottle": 0.85,
    }

    hsv = rgb_to_hsv_float(rgb)

    for class_id, class_name in enumerate(VOC_CLASSES):

        if class_name == "background":
            continue

        mask = labels == class_id

        area = float(np.mean(mask))

        if area < 0.003:
            continue

        conf = float(
            np.mean(confidence[mask])
        )

        ys, xs = np.where(mask)

        if len(xs) == 0:
            continue

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

        center_score = clamp(
            1.0 - center_distance * 1.4,
            0.0,
            1.0,
        )

        sal = float(
            np.mean(saliency[mask])
        )

        lum = rgb_luminance(rgb)

        local_mean = cv2.GaussianBlur(
            lum,
            (0, 0),
            sigmaX=15,
        )

        local_contrast = float(
            np.mean(
                np.abs(
                    lum[mask]
                    - local_mean[mask]
                )
            )
        )

        local_contrast = clamp(
            local_contrast * 4.0,
            0.0,
            1.0,
        )

        colorfulness = float(
            np.mean(
                hsv[..., 1][mask]
            ) / 255.0
        )

        prior = class_priors.get(
            class_name,
            1.0,
        )

        score = prior * (
            0.30 * conf
            + 0.15 * math.sqrt(area)
            + 0.15 * center_score
            + 0.20 * sal
            + 0.10 * local_contrast
            + 0.10 * colorfulness
        )

        subjects.append(
            SubjectCandidate(
                class_name=class_name,
                confidence=conf,
                area=area,
                center_x=center_x,
                center_y=center_y,
                saliency=sal,
                local_contrast=local_contrast,
                colorfulness=colorfulness,
                score=float(score),
            )
        )

    subjects.sort(
        key=lambda x: x.score,
        reverse=True,
    )

    return subjects[:10]


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

    for subject in subjects:
        if subject.class_name == "person":
            person_area += subject.area

        if subject.class_name in {
            "car",
            "bus",
            "train",
            "motorbike",
            "bicycle",
        }:
            vehicle_area += subject.area

    if (
        person_area > 0.015
        and stats.median > 0.08
    ):
        confidence = clamp(
            0.65
            + min(person_area, 0.25) * 0.8,
            0.0,
            0.95,
        )

        return SceneResult(
            "portrait",
            confidence,
        )

    if (
        shooting.low_light
        and stats.median < 0.10
    ):
        confidence = clamp(
            0.65
            + (0.10 - stats.median) * 2.0,
            0.0,
            0.95,
        )

        return SceneResult(
            "night",
            confidence,
        )

    if (
        stats.warm_ratio > 0.18
        and stats.p95 > 0.55
    ):
        return SceneResult(
            "sunset",
            0.75,
        )

    if (
        shooting.wide_angle
        and stats.edge_density < 0.16
        and stats.dynamic_range > 0.9
    ):
        return SceneResult(
            "landscape",
            0.70,
        )

    if (
        vehicle_area > 0.01
        and stats.edge_density > 0.10
    ):
        return SceneResult(
            "city",
            0.70,
        )

    if (
        stats.median < 0.18
        and stats.warm_ratio > 0.10
    ):
        return SceneResult(
            "indoor",
            0.65,
        )

    return SceneResult(
        "general",
        0.50,
    )


# ============================================================
# Region masks
# ============================================================

def build_region_masks(
    rgb,
    labels,
    subjects,
):
    h, w = labels.shape

    masks = {}

    def voc_mask(names):
        ids = [
            VOC_CLASSES.index(name)
            for name in names
            if name in VOC_CLASSES
        ]

        if not ids:
            return np.zeros(
                (h, w),
                dtype=np.float32,
            )

        mask = np.isin(
            labels,
            ids,
        )

        return mask.astype(np.float32)

    person = voc_mask(
        ["person"]
    )

    animal = voc_mask(
        [
            "cat",
            "dog",
            "bird",
            "horse",
            "cow",
            "sheep",
        ]
    )

    vehicle = voc_mask(
        [
            "car",
            "bus",
            "train",
            "boat",
            "bicycle",
            "motorbike",
        ]
    )

    plant = voc_mask(
        ["pottedplant"]
    )

    masks["person"] = person
    masks["animal"] = animal
    masks["vehicle"] = vehicle
    masks["plant"] = plant

    subject = (
        person
        + animal
        + vehicle
        + plant
    )

    masks["subject"] = np.clip(
        subject,
        0.0,
        1.0,
    )

    hsv = rgb_to_hsv_float(rgb)

    H = hsv[..., 0] * 2.0
    S = hsv[..., 1]
    V = hsv[..., 2]

    # Skin heuristic.
    skin = (
        (H >= 0)
        & (H <= 50)
        & (S >= 35)
        & (S <= 190)
        & (V >= 45)
        & (V <= 250)
    )

    masks["skin"] = (
        skin.astype(np.float32)
        * person
    )

    # Green.
    green = (
        (H >= 30)
        & (H <= 95)
        & (S >= 45)
        & (V >= 30)
    )

    masks["green"] = (
        green.astype(np.float32)
    )

    # Blue.
    blue = (
        (H >= 80)
        & (H <= 135)
        & (S >= 40)
        & (V >= 40)
    )

    masks["blue"] = (
        blue.astype(np.float32)
    )

    lum = rgb_luminance(rgb)

    # Deliberately called upper_bright.
    # This is NOT a claim that the region is sky.
    upper = np.zeros(
        (h, w),
        dtype=bool,
    )

    upper[: int(h * 0.45), :] = True

    p75 = np.percentile(
        lum,
        75,
    )

    upper_bright = (
        upper
        & (lum > p75)
    )

    masks["upper_bright"] = (
        upper_bright.astype(np.float32)
    )

    # Water heuristic:
    # blue + lower half + relatively low texture.
    gray = np.clip(
        lum * 255,
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

    lower = np.zeros(
        (h, w),
        dtype=bool,
    )

    lower[int(h * 0.45):, :] = True

    water = (
        blue
        & lower
        & (texture < 35)
    )

    masks["water"] = (
        water.astype(np.float32)
    )

    # Background.
    masks["background"] = (
        1.0
        - np.clip(
            masks["subject"],
            0.0,
            1.0,
        )
    )

    return masks


def calculate_region_stats(
    rgb,
    masks,
):
    hsv = rgb_to_hsv_float(rgb)

    lum = rgb_luminance(rgb)

    results = []

    for name, mask in masks.items():

        area = mask_area(mask)

        if area < 0.001:
            continue

        mean_luminance = mask_mean(
            lum,
            mask,
        )

        mean_saturation = (
            mask_mean(
                hsv[..., 1] / 255.0,
                mask,
            )
        )

        results.append(
            RegionStats(
                name=name,
                area=area,
                mean_luminance=mean_luminance,
                mean_saturation=mean_saturation,
            )
        )

    return results


# ============================================================
# Adaptive exposure model
# ============================================================

def create_exposure_model(
    scene,
    stats,
    masks,
    rgb,
):
    """
    v21 exposure model.

    The important change from v20:

        Do not increase the global target merely because
        the scene is portrait.

    Instead:

        global target
        + subject target
        + highlight headroom
        + shadow information
        + background information

    are evaluated independently.
    """

    scene_targets = {
        "portrait": 0.215,
        "night": 0.100,
        "sunset": 0.165,
        "landscape": 0.220,
        "city": 0.205,
        "indoor": 0.195,
        "general": 0.205,
    }

    global_target = scene_targets.get(
        scene.scene,
        0.205,
    )

    # Slightly adapt global target according to actual image.
    #
    # Do not make this adjustment large.
    # The purpose is only to avoid systematically dark/bright
    # output while leaving the parameter search to do the work.
    if stats.p99 < 0.45:
        global_target += 0.010

    elif stats.p99 < 0.55:
        global_target += 0.005

    if stats.shadow_ratio > 0.12:
        global_target += 0.005

    if scene.scene == "night":
        global_target = min(
            global_target,
            0.115,
        )

    global_target = clamp(
        global_target,
        0.095,
        0.235,
    )

    # Subject target.
    #
    # Person is more important than generic semantic subjects
    # for portrait exposure.
    if mask_area(masks.get("person")) > 0.01:

        if scene.scene == "portrait":
            subject_target = 0.285
        else:
            subject_target = 0.265

    elif mask_area(masks.get("subject")) > 0.01:
        subject_target = 0.255

    else:
        subject_target = global_target

    if scene.scene == "night":
        subject_target *= 0.88

    subject_target = clamp(
        subject_target,
        0.12,
        0.34,
    )

    if scene.scene in {
        "portrait",
        "sunset",
        "landscape",
        "city",
    }:
        highlight_soft_limit = 0.68
        highlight_hard_limit = 0.82

    elif scene.scene == "night":
        highlight_soft_limit = 0.62
        highlight_hard_limit = 0.80

    else:
        highlight_soft_limit = 0.66
        highlight_hard_limit = 0.82

    shadow_target = 0.018

    if scene.scene == "night":
        shadow_target = 0.012

    # Weights.
    global_weight = 0.34
    subject_weight = 0.34
    highlight_weight = 0.20
    shadow_weight = 0.07
    background_weight = 0.05

    if mask_area(masks.get("person")) < 0.01:
        subject_weight = 0.20
        global_weight = 0.46
        background_weight = 0.08

    if scene.scene == "night":
        highlight_weight = 0.28
        shadow_weight = 0.10
        subject_weight *= 0.90
        global_weight += 0.04

    return ExposureModel(
        global_target=global_target,
        subject_target=subject_target,

        highlight_soft_limit=highlight_soft_limit,
        highlight_hard_limit=highlight_hard_limit,

        shadow_target=shadow_target,

        subject_weight=subject_weight,
        global_weight=global_weight,
        highlight_weight=highlight_weight,
        shadow_weight=shadow_weight,
        background_weight=background_weight,
    )


def calculate_subject_luminance(
    rgb,
    masks,
):
    lum = rgb_luminance(rgb)

    person = masks.get("person")

    if mask_area(person) >= 0.01:
        return mask_median(
            lum,
            person,
        )

    subject = masks.get("subject")

    if mask_area(subject) >= 0.01:
        return mask_median(
            lum,
            subject,
        )

    return None


def calculate_background_luminance(
    rgb,
    masks,
):
    lum = rgb_luminance(rgb)

    background = masks.get(
        "background"
    )

    if mask_area(background) < 0.10:
        return None

    return mask_median(
        lum,
        background,
    )


def estimate_exposure_ev(
    rgb,
    scene,
    stats,
    masks,
    model,
):
    """
    Estimate EV only as a center point for the search.

    v21 deliberately keeps this moderate.
    The actual decision is made by search_parameters().
    """

    median = max(
        stats.median,
        1e-5,
    )

    ev_global = math.log2(
        model.global_target / median
    )

    subject_lum = calculate_subject_luminance(
        rgb,
        masks,
    )

    if (
        subject_lum is not None
        and subject_lum > 1e-5
    ):
        ev_subject = math.log2(
            model.subject_target
            / subject_lum
        )

        ev = (
            0.55 * ev_global
            + 0.45 * ev_subject
        )

    else:
        ev = ev_global

    # Highlight headroom.
    #
    # This is intentionally smaller than v20.
    if stats.p99 < 0.45:
        ev += 0.10

    elif stats.p99 < 0.55:
        ev += 0.05

    elif stats.p99 > 0.78:
        ev -= 0.10

    if stats.highlight_ratio > 0.001:
        ev -= 0.12

    # Very noisy high ISO images should not be aggressively
    # pushed merely because they are dark.
    if (
        scene.scene != "night"
        and stats.median < 0.12
        and stats.p99 > 0.60
    ):
        ev -= 0.05

    return clamp(
        ev,
        -0.75,
        1.0,
    )


# ============================================================
# Parameter search
# ============================================================

def apply_candidate_exposure(
    rgb,
    ev,
):
    linear = srgb_to_linear(
        np.clip(rgb, 0.0, 1.0)
    )

    linear = apply_ev(
        linear,
        ev,
    )

    return linear_to_srgb(
        linear
    ).astype(np.float32)


def evaluate_candidate(
    rgb,
    scene,
    model,
    masks,
    base_stats,
):
    stats = calculate_stats(rgb)

    lum = rgb_luminance(rgb)

    score = 0.0

    # --------------------------------------------------------
    # Global exposure
    # --------------------------------------------------------

    global_error = abs(
        stats.median
        - model.global_target
    )

    score += (
        model.global_weight
        * global_error
    )

    # --------------------------------------------------------
    # Subject exposure
    # --------------------------------------------------------

    subject_lum = calculate_subject_luminance(
        rgb,
        masks,
    )

    if subject_lum is not None:

        subject_error = abs(
            subject_lum
            - model.subject_target
        )

        # Being too dark is somewhat more important than
        # being slightly bright.
        if subject_lum < model.subject_target:
            subject_error *= 1.15

        score += (
            model.subject_weight
            * subject_error
        )

    # --------------------------------------------------------
    # Highlight handling
    # --------------------------------------------------------

    p99 = stats.p99

    if p99 <= model.highlight_soft_limit:
        highlight_error = 0.0

        # Small reward for using available headroom.
        if p99 < 0.45:
            score -= 0.012

        elif p99 < 0.55:
            score -= 0.006

    else:
        highlight_error = (
            p99
            - model.highlight_soft_limit
        )

    if p99 > model.highlight_hard_limit:
        highlight_error += (
            p99
            - model.highlight_hard_limit
        ) * 2.5

    if stats.highlight_ratio > 0:
        highlight_error += (
            stats.highlight_ratio
            * 3.0
        )

    score += (
        model.highlight_weight
        * highlight_error
    )

    # --------------------------------------------------------
    # Shadow handling
    # --------------------------------------------------------

    shadow_error = 0.0

    if scene.scene != "night":

        if stats.shadow_ratio > 0.20:
            shadow_error = (
                stats.shadow_ratio
                - 0.20
            )

        elif stats.shadow_ratio > 0.12:
            shadow_error = (
                stats.shadow_ratio
                - 0.12
            ) * 0.4

    else:

        if stats.shadow_ratio > 0.35:
            shadow_error = (
                stats.shadow_ratio
                - 0.35
            ) * 0.25

    score += (
        model.shadow_weight
        * shadow_error
    )

    # --------------------------------------------------------
    # Background handling
    # --------------------------------------------------------

    background_lum = calculate_background_luminance(
        rgb,
        masks,
    )

    if background_lum is not None:

        # Background can be bright, but an excessively bright
        # background can compete with the subject.
        if scene.scene == "portrait":

            if background_lum > 0.55:
                background_error = (
                    background_lum
                    - 0.55
                )

            else:
                background_error = 0.0

            score += (
                model.background_weight
                * background_error
            )

    # --------------------------------------------------------
    # Contrast sanity
    # --------------------------------------------------------

    if stats.contrast < 0.035:
        score += (
            0.025
            * (0.035 - stats.contrast)
        )

    # Avoid excessive saturation caused by clipping.
    if stats.saturation_ratio > 0.65:
        score += (
            stats.saturation_ratio
            - 0.65
        ) * 0.08

    # --------------------------------------------------------
    # High ISO protection
    # --------------------------------------------------------

    if base_stats.saturation_ratio > 0.0:
        saturation_change = (
            stats.saturation_ratio
            - base_stats.saturation_ratio
        )

        if saturation_change > 0.30:
            score += saturation_change * 0.03

    return float(score)


def search_parameters(
    rgb,
    scene,
    model,
    masks,
    estimated_ev,
):
    small = resize_max(
        rgb,
        512,
    )

    small_masks = {}

    for name, mask in masks.items():
        small_masks[name] = resize_mask(
            mask,
            small.shape,
        )

    base_stats = calculate_stats(
        small
    )

    # More detailed around the estimated point.
    ev_offsets = [
        -0.30,
        -0.22,
        -0.15,
        -0.08,
        0.00,
        0.08,
        0.15,
        0.22,
        0.30,
    ]

    ev_candidates = sorted(
        set(
            clamp(
                estimated_ev + offset,
                -1.0,
                1.0,
            )
            for offset in ev_offsets
        )
    )

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

    best_score = float("inf")
    best = None

    for ev in ev_candidates:

        candidate = apply_candidate_exposure(
            small,
            ev,
        )

        for contrast in contrast_candidates:

            contrasted = (
                (
                    candidate
                    - 0.18
                )
                * contrast
                + 0.18
            )

            contrasted = np.clip(
                contrasted,
                0.0,
                1.0,
            )

            for saturation in saturation_candidates:

                lum = rgb_luminance(
                    contrasted
                )

                result = (
                    lum[..., None]
                    + (
                        contrasted
                        - lum[..., None]
                    )
                    * saturation
                )

                result = np.clip(
                    result,
                    0.0,
                    1.0,
                )

                score = evaluate_candidate(
                    result,
                    scene,
                    model,
                    small_masks,
                    base_stats,
                )

                if score < best_score:

                    best_score = score

                    best = (
                        ev,
                        contrast,
                        saturation,
                    )

    return best, best_score


# ============================================================
# Scene profiles
# ============================================================

def scene_profile(scene):
    profiles = {

        "portrait": DevelopParams(
            exposure_ev=0.0,
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
            exposure_ev=0.0,
            contrast=1.05,
            saturation=1.03,

            highlight_protection=0.55,
            shadow_lift=0.02,

            subject_exposure=0.05,
            subject_contrast=1.03,
            background_suppression=0.015,

            denoise=0.42,
            sharpen=0.45,

            region_skin_saturation=0.94,
            region_green_saturation=1.00,
            region_water_highlight=0.10,
            region_upper_highlight=0.18,

            tone_strength=0.45,
        ),

        "sunset": DevelopParams(
            exposure_ev=0.0,
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
            exposure_ev=0.0,
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
            exposure_ev=0.0,
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
            exposure_ev=0.0,
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

            denoise=0.32,
            sharpen=0.75,

            region_skin_saturation=0.96,
            region_green_saturation=1.01,
            region_water_highlight=0.07,
            region_upper_highlight=0.12,

            tone_strength=0.52,
        ),
    }

    return profiles.get(
        scene,
        profiles["general"],
    )


# ============================================================
# Development operations
# ============================================================

def apply_exposure(
    rgb,
    ev,
):
    linear = srgb_to_linear(
        rgb
    )

    linear = apply_ev(
        linear,
        ev,
    )

    return linear_to_srgb(
        linear
    ).astype(np.float32)


def apply_contrast(
    rgb,
    contrast,
):
    lum = rgb_luminance(rgb)

    result = (
        (
            rgb
            - lum[..., None]
        )
        * contrast
        + lum[..., None]
    )

    # Mild global luminance contrast.
    result_lum = rgb_luminance(
        result
    )

    result_lum = (
        (
            result_lum
            - 0.18
        )
        * contrast
        + 0.18
    )

    ratio = (
        result_lum
        / np.maximum(
            rgb_luminance(result),
            1e-5,
        )
    )

    result *= ratio[..., None]

    return np.clip(
        result,
        0.0,
        1.0,
    ).astype(np.float32)


def apply_saturation(
    rgb,
    saturation,
):
    lum = rgb_luminance(rgb)

    result = (
        lum[..., None]
        + (
            rgb
            - lum[..., None]
        )
        * saturation
    )

    return np.clip(
        result,
        0.0,
        1.0,
    ).astype(np.float32)


def apply_tone_curve(
    rgb,
    strength,
    shadow_lift,
    highlight_protection,
):
    lum = rgb_luminance(rgb)

    x = np.clip(
        lum,
        0.0,
        1.0,
    )

    # --------------------------------------------------------
    # Shadow lift
    # --------------------------------------------------------

    shadow_weight = np.clip(
        (0.25 - x) / 0.25,
        0.0,
        1.0,
    )

    x = (
        x
        + shadow_lift
        * shadow_weight
        * (1.0 - x)
    )

    # --------------------------------------------------------
    # Highlight rolloff
    # --------------------------------------------------------

    highlight_weight = np.clip(
        (x - 0.55) / 0.45,
        0.0,
        1.0,
    )

    rolloff = (
        highlight_protection
        * highlight_weight
        * highlight_weight
        * 0.18
    )

    x = x - rolloff

    # --------------------------------------------------------
    # Mild S-curve
    # --------------------------------------------------------

    centered = x - 0.5

    curve = (
        centered
        + strength
        * centered
        * (
            1.0
            - 4.0 * centered * centered
        )
        * 0.18
    )

    new_lum = np.clip(
        curve + 0.5,
        0.0,
        1.0,
    )

    old_lum = np.maximum(
        rgb_luminance(rgb),
        1e-5,
    )

    ratio = (
        new_lum
        / old_lum
    )

    result = (
        rgb
        * ratio[..., None]
    )

    return np.clip(
        result,
        0.0,
        1.0,
    ).astype(np.float32)


def apply_region_processing(
    rgb,
    masks,
    params,
):
    result = rgb.copy()

    lum = rgb_luminance(
        result
    )

    # --------------------------------------------------------
    # Subject
    # --------------------------------------------------------

    subject = masks.get("subject")

    if subject is not None:
        weight = cv2.GaussianBlur(
            subject.astype(np.float32),
            (0, 0),
            sigmaX=5,
        )

        linear = srgb_to_linear(
            result
        )

        gain = (
            2.0
            ** (
                params.subject_exposure
                * weight
            )
        )

        linear *= gain[..., None]

        result = linear_to_srgb(
            linear
        )

        # Subject contrast.
        subject_lum = rgb_luminance(
            result
        )

        subject_center = 0.20

        adjusted_lum = (
            subject_center
            + (
                subject_lum
                - subject_center
            )
            * (
                1.0
                + (
                    params.subject_contrast
                    - 1.0
                )
                * weight
            )
        )

        ratio = (
            adjusted_lum
            / np.maximum(
                subject_lum,
                1e-5,
            )
        )

        result *= ratio[..., None]

    # --------------------------------------------------------
    # Skin
    # --------------------------------------------------------

    skin = masks.get("skin")

    if skin is not None:
        weight = cv2.GaussianBlur(
            skin.astype(np.float32),
            (0, 0),
            sigmaX=4,
        )

        lum = rgb_luminance(
            result
        )

        skin_saturation = (
            params.region_skin_saturation
        )

        result = (
            lum[..., None]
            + (
                result
                - lum[..., None]
            )
            * (
                1.0
                - weight
                + weight
                * skin_saturation
            )[..., None]
        )

    # --------------------------------------------------------
    # Green
    # --------------------------------------------------------

    green = masks.get("green")

    if green is not None:
        weight = cv2.GaussianBlur(
            green.astype(np.float32),
            (0, 0),
            sigmaX=5,
        )

        lum = rgb_luminance(
            result
        )

        saturation = (
            1.0
            + weight
            * (
                params.region_green_saturation
                - 1.0
            )
        )

        result = (
            lum[..., None]
            + (
                result
                - lum[..., None]
            )
            * saturation[..., None]
        )

    # --------------------------------------------------------
    # Water
    # --------------------------------------------------------

    water = masks.get("water")

    if water is not None:
        weight = cv2.GaussianBlur(
            water.astype(np.float32),
            (0, 0),
            sigmaX=6,
        )

        lum = rgb_luminance(
            result
        )

        # Mild saturation enhancement.
        saturation = (
            1.0
            + weight * 0.02
        )

        result = (
            lum[..., None]
            + (
                result
                - lum[..., None]
            )
            * saturation[..., None]
        )

        # Highlight protection.
        water_highlight = np.clip(
            (
                lum
                - 0.55
            ) / 0.45,
            0.0,
            1.0,
        )

        reduction = (
            weight
            * water_highlight
            * params.region_water_highlight
            * 0.12
        )

        result *= (
            1.0
            - reduction[..., None]
        )

    # --------------------------------------------------------
    # Upper bright region
    # --------------------------------------------------------

    upper = masks.get(
        "upper_bright"
    )

    if upper is not None:

        weight = cv2.GaussianBlur(
            upper.astype(np.float32),
            (0, 0),
            sigmaX=8,
        )

        lum = rgb_luminance(
            result
        )

        highlight = np.clip(
            (
                lum
                - 0.55
            ) / 0.45,
            0.0,
            1.0,
        )

        reduction = (
            weight
            * highlight
            * params.region_upper_highlight
            * 0.15
        )

        result *= (
            1.0
            - reduction[..., None]
        )

    # --------------------------------------------------------
    # Background suppression
    # --------------------------------------------------------

    background = masks.get(
        "background"
    )

    if background is not None:

        weight = cv2.GaussianBlur(
            background.astype(np.float32),
            (0, 0),
            sigmaX=10,
        )

        lum = rgb_luminance(
            result
        )

        # Slightly reduce saturation.
        result = (
            lum[..., None]
            + (
                result
                - lum[..., None]
            )
            * (
                1.0
                - params.background_suppression
                * weight
            )[..., None]
        )

    return np.clip(
        result,
        0.0,
        1.0,
    ).astype(np.float32)


# ============================================================
# Denoise
# ============================================================

def calculate_denoise_strength(
    base_strength,
    iso_factor,
):
    """
    v21:
    Preserve v20's moderate denoise behavior.

    Important:
        Denoise is applied only to luminance.
        Chroma is intentionally preserved.
    """

    iso_component = clamp(
        (iso_factor - 1.0) / 4.0,
        0.0,
        1.0,
    )

    strength = (
        base_strength
        * (
            0.70
            + 0.45 * iso_component
        )
    )

    return clamp(
        strength,
        0.05,
        0.65,
    )


def denoise_luminance(
    rgb,
    strength,
):
    if strength <= 0.01:
        return rgb

    rgb8 = np.clip(
        rgb * 255.0,
        0,
        255,
    ).astype(np.uint8)

    ycrcb = cv2.cvtColor(
        rgb8,
        cv2.COLOR_RGB2YCrCb,
    )

    sigma_color = (
        8.0
        + 28.0 * strength
    )

    sigma_space = (
        2.0
        + 3.0 * strength
    )

    y = ycrcb[..., 0]

    filtered_y = cv2.bilateralFilter(
        y,
        d=7,
        sigmaColor=sigma_color,
        sigmaSpace=sigma_space,
    )

    ycrcb[..., 0] = filtered_y

    result = cv2.cvtColor(
        ycrcb,
        cv2.COLOR_YCrCb2RGB,
    )

    return (
        result.astype(np.float32)
        / 255.0
    )


# ============================================================
# Sharpen
# ============================================================

def sharpen_luminance(
    rgb,
    strength,
):
    if strength <= 0.01:
        return rgb

    lum = rgb_luminance(
        rgb
    )

    blur = cv2.GaussianBlur(
        lum,
        (0, 0),
        sigmaX=1.0,
    )

    detail = (
        lum
        - blur
    )

    amount = (
        0.25
        * strength
    )

    new_lum = np.clip(
        lum
        + detail * amount,
        0.0,
        1.0,
    )

    ratio = (
        new_lum
        / np.maximum(
            lum,
            1e-5,
        )
    )

    result = (
        rgb
        * ratio[..., None]
    )

    return np.clip(
        result,
        0.0,
        1.0,
    ).astype(np.float32)


# ============================================================
# Debug output
# ============================================================

def save_rgb_jpeg(
    path,
    rgb,
    quality=95,
):
    rgb8 = np.clip(
        rgb * 255.0,
        0,
        255,
    ).astype(np.uint8)

    Image.fromarray(
        rgb8,
        mode="RGB",
    ).save(
        path,
        quality=quality,
        subsampling=0,
    )


def save_stage(
    debug_dir,
    name,
    display_image,
    color_space,
    stats_image=None,
):
    """
    v21 fix:

        display_image:
            image used to create the debug JPEG

        stats_image:
            image used to calculate statistics

    This is important for linear RGB.

    Example:

        stats_image = true linear RGB

        display_image =
            linear_to_srgb(linear RGB)

    Therefore the JPEG can be viewed normally while
    statistics remain genuinely linear.
    """

    if stats_image is None:
        stats_image = display_image

    stats = calculate_stats(
        stats_image
    )

    print_stats(
        name,
        stats,
        color_space,
    )

    path = (
        debug_dir
        / f"{name}.jpg"
    )

    save_rgb_jpeg(
        path,
        display_image,
    )

    return stats


def save_mask(
    debug_dir,
    name,
    mask,
):
    image = np.clip(
        mask * 255.0,
        0,
        255,
    ).astype(np.uint8)

    path = (
        debug_dir
        / f"{name}.png"
    )

    Image.fromarray(
        image,
        mode="L",
    ).save(path)


def save_segmentation(
    debug_dir,
    labels,
):
    h, w = labels.shape

    output = np.zeros(
        (h, w, 3),
        dtype=np.uint8,
    )

    for idx in range(
        len(VOC_CLASSES)
    ):
        mask = labels == idx

        if idx == 0:
            value = 0
        else:
            value = int(
                40
                + (
                    idx * 37
                ) % 200
            )

        output[mask] = (
            value,
            value,
            value,
        )

    Image.fromarray(
        output,
        mode="RGB",
    ).save(
        debug_dir
        / "segmentation.png"
    )


def save_saliency(
    debug_dir,
    saliency,
):
    image = np.clip(
        saliency * 255,
        0,
        255,
    ).astype(np.uint8)

    Image.fromarray(
        image,
        mode="L",
    ).save(
        debug_dir
        / "saliency.png"
    )


# ============================================================
# Developer
# ============================================================

class AutoDeveloper:

    def __init__(
        self,
        device="auto",
        debug=False,
    ):
        self.debug = debug

        self.segmenter = SemanticSegmenter(
            device
        )

    def process_file(
        self,
        input_path,
        output_path,
        debug_root=None,
    ):
        print()
        print(
            f"[INFO] Processing: "
            f"{input_path}"
        )

        metadata = get_metadata(
            input_path
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

        if debug_root is not None:
            debug_dir = (
                debug_root
                / Path(input_path).stem
            )

            debug_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

        else:
            debug_dir = None

        # ----------------------------------------------------
        # RAW
        # ----------------------------------------------------

        with rawpy.imread(
            str(input_path)
        ) as raw:

            camera_profile = (
                get_camera_profile(
                    raw,
                    metadata,
                )
            )

            linear_rgb, conversion_method = (
                raw_to_linear_rgb(raw)
            )

        # ----------------------------------------------------
        # Debug: true linear statistics
        # ----------------------------------------------------

        if self.debug:

            save_stage(
                debug_dir,
                "01_raw_linear",
                linear_to_srgb(
                    linear_rgb
                ),
                "linear_rgb",
                stats_image=linear_rgb,
            )

        # ----------------------------------------------------
        # Analysis sRGB
        # ----------------------------------------------------

        analysis_srgb = linear_to_srgb(
            linear_rgb
        ).astype(np.float32)

        if self.debug:

            save_stage(
                debug_dir,
                "02_analysis_srgb",
                analysis_srgb,
                "sRGB",
                stats_image=analysis_srgb,
            )

        # ----------------------------------------------------
        # Statistics
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
        # Semantic segmentation
        # ----------------------------------------------------

        labels, confidence = (
            self.segmenter.predict(
                analysis_srgb
            )
        )

        saliency = calculate_saliency(
            analysis_srgb
        )

        subjects = rank_subjects(
            labels,
            confidence,
            saliency,
            analysis_srgb,
        )

        print("[INFO] Subjects:")

        if subjects:
            for subject in subjects:
                print(
                    f"  {subject.class_name}: "
                    f"score={subject.score:.3f}, "
                    f"area={subject.area:.3f}, "
                    f"confidence={subject.confidence:.3f}"
                )
        else:
            print("  none")

        # ----------------------------------------------------
        # Scene
        # ----------------------------------------------------

        scene = classify_scene(
            stats,
            shooting,
            subjects,
        )

        print(
            f"[INFO] Scene: "
            f"{scene.scene} "
            f"(confidence="
            f"{scene.confidence:.3f})"
        )

        # ----------------------------------------------------
        # Region masks
        # ----------------------------------------------------

        masks = build_region_masks(
            analysis_srgb,
            labels,
            subjects,
        )

        region_stats = calculate_region_stats(
            analysis_srgb,
            masks,
        )

        # ----------------------------------------------------
        # Exposure model
        # ----------------------------------------------------

        exposure_model = (
            create_exposure_model(
                scene,
                stats,
                masks,
                analysis_srgb,
            )
        )

        print(
            f"[INFO] Exposure model:"
        )

        print(
            f"  global target : "
            f"{exposure_model.global_target:.3f}"
        )

        print(
            f"  subject target: "
            f"{exposure_model.subject_target:.3f}"
        )

        print(
            f"  highlight soft: "
            f"{exposure_model.highlight_soft_limit:.3f}"
        )

        print(
            f"  highlight hard: "
            f"{exposure_model.highlight_hard_limit:.3f}"
        )

        # ----------------------------------------------------
        # Estimated EV
        # ----------------------------------------------------

        estimated_ev = (
            estimate_exposure_ev(
                analysis_srgb,
                scene,
                stats,
                masks,
                exposure_model,
            )
        )

        print(
            f"[INFO] Estimated EV: "
            f"{estimated_ev:+.3f}"
        )

        # ----------------------------------------------------
        # Search
        # ----------------------------------------------------

        print(
            "[INFO] Running parameter search..."
        )

        best, search_score = (
            search_parameters(
                analysis_srgb,
                scene,
                exposure_model,
                masks,
                estimated_ev,
            )
        )

        selected_ev = best[0]
        selected_contrast = best[1]
        selected_saturation = best[2]

        print(
            f"[INFO] Search score: "
            f"{search_score:.6f}"
        )

        print(
            f"[INFO] Selected EV: "
            f"{selected_ev:+.3f}"
        )

        print(
            f"[INFO] Selected contrast: "
            f"{selected_contrast:.3f}"
        )

        print(
            f"[INFO] Selected saturation: "
            f"{selected_saturation:.3f}"
        )

        # ----------------------------------------------------
        # Scene profile
        # ----------------------------------------------------

        params = scene_profile(
            scene.scene
        )

        params.exposure_ev = (
            selected_ev
        )

        params.contrast = (
            selected_contrast
        )

        params.saturation = (
            selected_saturation
        )

        # ----------------------------------------------------
        # Denoise
        # ----------------------------------------------------

        denoise_strength = (
            calculate_denoise_strength(
                params.denoise,
                shooting.iso_factor,
            )
        )

        print(
            f"[INFO] Denoise strength: "
            f"{denoise_strength:.3f}"
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

        developed = apply_exposure(
            analysis_srgb,
            params.exposure_ev,
        )

        if self.debug:

            save_stage(
                debug_dir,
                "03_after_exposure",
                developed,
                "sRGB",
            )

        developed = apply_contrast(
            developed,
            params.contrast,
        )

        if self.debug:

            save_stage(
                debug_dir,
                "04_after_contrast",
                developed,
                "sRGB",
            )

        developed = apply_saturation(
            developed,
            params.saturation,
        )

        if self.debug:

            save_stage(
                debug_dir,
                "05_after_saturation",
                developed,
                "sRGB",
            )

        developed = apply_tone_curve(
            developed,
            params.tone_strength,
            params.shadow_lift,
            params.highlight_protection,
        )

        if self.debug:

            save_stage(
                debug_dir,
                "06_after_tone",
                developed,
                "sRGB",
            )

        developed = apply_region_processing(
            developed,
            masks,
            params,
        )

        if self.debug:

            save_stage(
                debug_dir,
                "07_after_region",
                developed,
                "sRGB",
            )

        developed = denoise_luminance(
            developed,
            denoise_strength,
        )

        if self.debug:

            save_stage(
                debug_dir,
                "08_after_denoise",
                developed,
                "sRGB",
            )

        developed = sharpen_luminance(
            developed,
            params.sharpen,
        )

        if self.debug:

            save_stage(
                debug_dir,
                "09_after_sharpen",
                developed,
                "sRGB",
            )

        # ----------------------------------------------------
        # Final linear
        # ----------------------------------------------------

        final_srgb = np.clip(
            developed,
            0.0,
            1.0,
        ).astype(np.float32)

        final_linear = srgb_to_linear(
            final_srgb
        )

        if self.debug:

            save_stage(
                debug_dir,
                "10_final_linear",
                linear_to_srgb(
                    final_linear
                ),
                "linear_rgb",
                stats_image=final_linear,
            )

            save_stage(
                debug_dir,
                "11_final_srgb",
                final_srgb,
                "sRGB",
                stats_image=final_srgb,
            )

            save_segmentation(
                debug_dir,
                labels,
            )

            save_saliency(
                debug_dir,
                saliency,
            )

            for name, mask in masks.items():
                save_mask(
                    debug_dir,
                    f"mask_{name}",
                    mask,
                )

        # ----------------------------------------------------
        # Save JPEG
        # ----------------------------------------------------

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        save_rgb_jpeg(
            output_path,
            final_srgb,
            quality=95,
        )

        # ----------------------------------------------------
        # Final stats
        # ----------------------------------------------------

        final_stats = calculate_stats(
            final_srgb
        )

        final_linear_stats = calculate_stats(
            final_linear
        )

        # ----------------------------------------------------
        # JSON report
        # ----------------------------------------------------

        if self.debug:

            report = {
                "version": VERSION,

                "input": str(input_path),
                "output": str(output_path),

                "conversion_method":
                    conversion_method,

                "camera":
                    asdict(camera_profile),

                "metadata":
                    asdict(metadata),

                "input_stats_srgb":
                    asdict(stats),

                "input_stats_linear":
                    asdict(
                        calculate_stats(
                            linear_rgb
                        )
                    ),

                "shooting":
                    asdict(shooting),

                "scene":
                    asdict(scene),

                "subjects": [
                    asdict(subject)
                    for subject in subjects
                ],

                "regions": [
                    asdict(region)
                    for region in region_stats
                ],

                "exposure_model":
                    asdict(exposure_model),

                "estimated_ev":
                    estimated_ev,

                "search_score":
                    search_score,

                "selected_parameters":
                    asdict(params),

                "denoise_strength":
                    denoise_strength,

                "final_stats_srgb":
                    asdict(final_stats),

                "final_stats_linear":
                    asdict(final_linear_stats),
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


# ============================================================
# File collection
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
            "Automatic RAW developer "
            f"{VERSION}"
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
        help="Inference device",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output",
    )

    args = parser.parse_args()

    input_path = Path(
        args.input
    )

    output_dir = Path(
        args.output
    )

    raw_files = collect_raw_files(
        input_path
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
        device=args.device,
        debug=args.debug,
    )

    debug_root = None

    if args.debug:
        debug_root = (
            output_dir
            / "debug"
        )

        debug_root.mkdir(
            parents=True,
            exist_ok=True,
        )

    success = 0
    failed = 0

    for input_file in raw_files:

        try:

            if input_path.is_file():
                relative_name = (
                    input_file.stem
                    + ".jpg"
                )

            else:
                relative_name = (
                    input_file.relative_to(
                        input_path
                    ).with_suffix(".jpg")
                )

            output_file = (
                output_dir
                / relative_name
            )

            developer.process_file(
                input_file,
                output_file,
                debug_root,
            )

            success += 1

        except Exception as exc:

            failed += 1

            print(
                f"[ERROR] Failed: "
                f"{input_file}"
            )

            print(
                f"        "
                f"{type(exc).__name__}: "
                f"{exc}"
            )

    print()
    print(
        "[INFO] Finished."
    )

    print(
        f"[INFO] Success: "
        f"{success}"
    )

    print(
        f"[INFO] Failed : "
        f"{failed}"
    )

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(
        main()
    )