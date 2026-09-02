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
from torchvision.models.segmentation import (
    deeplabv3_mobilenet_v3_large,
    DeepLabV3_MobileNet_V3_Large_Weights,
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

JPEG_QUALITY = 95

XYZ_TO_SRGB = np.array(
    [
        [3.2404542, -1.5371385, -0.4985314],
        [-0.9692660, 1.8760108, 0.0415560],
        [0.0556434, -0.2040259, 1.0572252],
    ],
    dtype=np.float32,
)


# Pascal VOC classes used by DeepLabV3
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
    "dog",
    "horse",
    "cow",
    "sheep",
    "bicycle",
    "motorbike",
    "car",
    "bus",
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
    camera_family: str = ""

    iso: float = 0.0

    black_level: float = 0.0
    white_level: float = 65535.0

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


@dataclass
class ShootingCondition:
    iso_factor: float = 1.0

    low_light: bool = False
    motion_risk: float = 0.0
    shallow_dof: float = 0.0

    wide_angle: bool = False
    telephoto: bool = False

    estimated_noise: float = 0.0


@dataclass
class SubjectCandidate:
    class_name: str
    confidence: float
    area: float
    center: float
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

def safe_float(value, default=0.0):
    try:
        if value is None:
            return default

        if isinstance(value, (list, tuple)):
            if not value:
                return default
            value = value[0]

        return float(value)
    except Exception:
        return default


def safe_string(value):
    if value is None:
        return ""

    if isinstance(value, (list, tuple)):
        if not value:
            return ""
        value = value[0]

    return str(value).strip()


def smoothstep(edge0, edge1, x):
    t = np.clip((x - edge0) / max(edge1 - edge0, 1e-8), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def luminance(rgb):
    return (
        rgb[..., 0] * 0.2126
        + rgb[..., 1] * 0.7152
        + rgb[..., 2] * 0.0722
    )


def colorfulness(rgb):
    mx = np.max(rgb, axis=2)
    mn = np.min(rgb, axis=2)
    return np.mean(mx - mn)


def linear_to_srgb(x):
    x = np.maximum(x, 0.0)

    return np.where(
        x <= 0.0031308,
        12.92 * x,
        1.055 * np.power(x, 1.0 / 2.4) - 0.055,
    )


def srgb_to_linear(x):
    x = np.clip(x, 0.0, 1.0)

    return np.where(
        x <= 0.04045,
        x / 12.92,
        np.power((x + 0.055) / 1.055, 2.4),
    )


def apply_exposure(rgb, ev):
    return rgb * (2.0 ** ev)


# ============================================================
# Metadata
# ============================================================

def run_exiftool(path):
    if shutil.which("exiftool") is None:
        return {}

    cmd = [
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
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )

        data = json.loads(result.stdout)

        if data:
            return data[0]

    except Exception:
        pass

    return {}


def read_pillow_exif(path):
    result = {}

    try:
        with Image.open(path) as img:
            exif = img.getexif()

            for key, value in exif.items():
                name = ExifTags.TAGS.get(key, str(key))
                result[name] = value

    except Exception:
        pass

    return result


def get_metadata(path):
    exiftool_data = run_exiftool(path)

    pillow_data = read_pillow_exif(path)

    if exiftool_data:
        source = "ExifTool"
        data = dict(pillow_data)
        data.update(exiftool_data)
    else:
        source = "Pillow"
        data = pillow_data

    make = (
        data.get("Make")
        or data.get("CameraMake")
        or ""
    )

    model = (
        data.get("CameraModelName")
        or data.get("UniqueCameraModel")
        or data.get("Model")
        or ""
    )

    lens_make = data.get("LensMake", "")
    lens_model = data.get("LensModel", "")

    return ExifMetadata(
        make=safe_string(make),
        model=safe_string(model),

        lens_make=safe_string(lens_make),
        lens_model=safe_string(lens_model),

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

        white_balance=safe_string(
            data.get("WhiteBalance")
        ),

        color_temperature=safe_float(
            data.get("ColorTemperature")
        ),

        color_space=safe_string(
            data.get("ColorSpace")
        ),

        source=source,
    )


def detect_camera_family(make, model):
    text = f"{make} {model}".lower()

    families = [
        ("Canon", ["canon"]),
        ("Nikon", ["nikon"]),
        ("Sony", ["sony"]),
        ("Fujifilm", ["fujifilm", "fuji"]),
        ("Panasonic", ["panasonic", "lumix"]),
        ("Olympus", ["olympus", "om system"]),
        ("Leica", ["leica"]),
        ("Pentax", ["pentax"]),
        ("Ricoh", ["ricoh"]),
        ("Sigma", ["sigma"]),
        ("Hasselblad", ["hasselblad"]),
    ]

    for family, keywords in families:
        for keyword in keywords:
            if keyword in text:
                return family

    return "Unknown"


# ============================================================
# RAW
# ============================================================

def get_libraw_version():
    try:
        return str(rawpy.libraw_version)
    except Exception:
        return ""


def build_camera_profile(raw, metadata):
    try:
        black = float(np.mean(raw.black_level_per_channel))
    except Exception:
        black = 0.0

    try:
        white = float(np.mean(raw.camera_white_level_per_channel))
    except Exception:
        white = 65535.0

    try:
        wb = np.asarray(
            raw.camera_whitebalance,
            dtype=np.float32,
        ).tolist()
    except Exception:
        wb = []

    try:
        color_matrix = np.asarray(
            raw.color_matrix,
            dtype=np.float32,
        ).tolist()
    except Exception:
        color_matrix = []

    try:
        rgb_xyz = np.asarray(
            raw.rgb_xyz_matrix,
            dtype=np.float32,
        ).tolist()
    except Exception:
        rgb_xyz = []

    try:
        height, width = raw.raw_image_visible.shape
    except Exception:
        width = 0
        height = 0

    return CameraProfile(
        make=metadata.make,
        model=metadata.model,

        camera_family=detect_camera_family(
            metadata.make,
            metadata.model,
        ),

        iso=metadata.iso,

        black_level=black,
        white_level=white,

        camera_whitebalance=wb,
        color_matrix=color_matrix,
        rgb_xyz_matrix=rgb_xyz,

        raw_width=width,
        raw_height=height,

        lens_make=metadata.lens_make,
        lens_model=metadata.lens_model,

        metadata_source=metadata.source,

        libraw_version=get_libraw_version(),
    )


def raw_to_camera_rgb(raw):
    """
    Try to obtain camera RGB.
    """

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

    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(
            f"Unexpected camera RGB shape: {rgb.shape}"
        )

    return rgb


def raw_to_srgb_fallback(raw):
    """
    LibRaw sRGB fallback.
    """

    rgb = raw.postprocess(
        use_camera_wb=True,
        use_auto_wb=False,

        output_color=rawpy.ColorSpace.sRGB,
        output_bps=16,

        gamma=(1.0, 1.0),

        no_auto_bright=True,

        highlight_mode=rawpy.HighlightMode.Blend,

        half_size=False,

        four_color_rgb=False,

        demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD,
    )

    rgb = rgb.astype(np.float32) / 65535.0

    rgb = np.clip(rgb, 0.0, 1.0)

    return srgb_to_linear(rgb)


def camera_rgb_to_linear_srgb(camera_rgb, profile):
    """
    Convert camera RGB -> XYZ -> linear sRGB.

    No arbitrary p99 normalization is used here.
    """

    rgb = camera_rgb.astype(np.float32) / 65535.0

    matrix = np.asarray(
        profile.rgb_xyz_matrix,
        dtype=np.float32,
    )

    if matrix.ndim != 2:
        raise ValueError("Invalid RGB XYZ matrix")

    if matrix.shape[0] < 3 or matrix.shape[1] < 3:
        raise ValueError(
            f"Invalid RGB XYZ matrix shape: {matrix.shape}"
        )

    matrix = matrix[:3, :3]

    if not np.all(np.isfinite(matrix)):
        raise ValueError("RGB XYZ matrix contains NaN/Inf")

    xyz = np.tensordot(
        rgb,
        matrix.T,
        axes=1,
    )

    srgb = np.tensordot(
        xyz,
        XYZ_TO_SRGB.T,
        axes=1,
    )

    srgb = np.maximum(srgb, 0.0)

    return srgb.astype(np.float32)


# ============================================================
# Image Analysis
# ============================================================

def analyze_image(rgb):
    y = luminance(rgb)

    mean = float(np.mean(y))
    median = float(np.median(y))

    p01 = float(np.percentile(y, 1))
    p05 = float(np.percentile(y, 5))
    p95 = float(np.percentile(y, 95))
    p99 = float(np.percentile(y, 99))

    shadow_ratio = float(np.mean(y < 0.01))
    highlight_ratio = float(np.mean(y > 0.98))

    dynamic_range = float(
        np.log2(
            max(p95, 1e-6)
            /
            max(p05, 1e-6)
        )
    )

    sat = (
        np.max(rgb, axis=2)
        -
        np.min(rgb, axis=2)
    )

    saturation_ratio = float(
        np.mean(sat > 0.65)
    )

    gray = cv2.cvtColor(
        np.clip(rgb * 255.0, 0, 255).astype(np.uint8),
        cv2.COLOR_RGB2GRAY,
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

    mag = cv2.magnitude(gx, gy)

    edge_density = float(
        np.mean(mag > 40.0)
    )

    warm_mask = (
        (rgb[..., 0] > rgb[..., 2] * 1.12)
        &
        (rgb[..., 0] > rgb[..., 1] * 1.02)
    )

    warm_ratio = float(
        np.mean(warm_mask)
    )

    contrast = float(
        np.std(y)
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

        mean_luminance=mean,
    )


# ============================================================
# Shooting Condition
# ============================================================

def analyze_shooting(metadata, stats):
    iso = metadata.iso
    exposure = metadata.exposure_time
    aperture = metadata.f_number
    focal = metadata.focal_length

    iso_factor = np.clip(
        math.sqrt(max(iso, 100.0) / 100.0),
        1.0,
        5.0,
    )

    low_light = (
        stats.median < 0.12
        or
        iso >= 1600
    )

    if exposure > 0:
        motion_risk = float(
            np.clip(
                1.0
                /
                max(exposure * 30.0, 0.05),
                0.0,
                1.0,
            )
        )
    else:
        motion_risk = 0.0

    if aperture > 0:
        shallow_dof = float(
            np.clip(
                (4.0 - aperture) / 3.0,
                0.0,
                1.0,
            )
        )
    else:
        shallow_dof = 0.0

    wide_angle = (
        focal > 0
        and focal <= 28
    )

    telephoto = (
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
        iso_factor=float(iso_factor),

        low_light=bool(low_light),
        motion_risk=motion_risk,
        shallow_dof=shallow_dof,

        wide_angle=bool(wide_angle),
        telephoto=bool(telephoto),

        estimated_noise=estimated_noise,
    )


# ============================================================
# Scene
# ============================================================

def classify_scene(stats, shooting, subjects):
    person_area = sum(
        s.area
        for s in subjects
        if s.class_name == "person"
    )

    animal_area = sum(
        s.area
        for s in subjects
        if s.class_name in ANIMAL_CLASSES
    )

    vehicle_area = sum(
        s.area
        for s in subjects
        if s.class_name in VEHICLE_CLASSES
    )

    if (
        person_area > 0.015
        and
        stats.median > 0.08
    ):
        return SceneResult(
            scene="portrait",
            confidence=0.80,
        )

    if (
        shooting.low_light
        and
        stats.median < 0.10
    ):
        return SceneResult(
            scene="night",
            confidence=0.80,
        )

    if (
        stats.warm_ratio > 0.18
        and
        stats.p95 > 0.55
    ):
        return SceneResult(
            scene="sunset",
            confidence=0.68,
        )

    if (
        shooting.wide_angle
        and
        stats.edge_density < 0.16
        and
        stats.dynamic_range > 5.0
    ):
        return SceneResult(
            scene="landscape",
            confidence=0.65,
        )

    if (
        vehicle_area > 0.01
        and
        stats.edge_density > 0.10
    ):
        return SceneResult(
            scene="city",
            confidence=0.65,
        )

    if (
        stats.median < 0.18
        and
        stats.warm_ratio > 0.10
    ):
        return SceneResult(
            scene="indoor",
            confidence=0.55,
        )

    if animal_area > 0.01:
        return SceneResult(
            scene="general",
            confidence=0.50,
        )

    return SceneResult(
        scene="general",
        confidence=0.45,
    )


# ============================================================
# Segmentation
# ============================================================

class SemanticSegmenter:

    def __init__(self, device):
        self.device = device

        print(
            f"[INFO] Loading semantic segmentation model on "
            f"{device}"
        )

        try:
            weights = (
                DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
            )

            self.model = (
                deeplabv3_mobilenet_v3_large(
                    weights=weights
                )
                .to(device)
                .eval()
            )

            self.transforms = weights.transforms()

        except Exception as exc:
            warnings.warn(
                f"Modern torchvision API failed: {exc}"
            )

            self.model = (
                deeplabv3_mobilenet_v3_large(
                    pretrained=True
                )
                .to(device)
                .eval()
            )

            self.transforms = None

    def predict(self, rgb):
        h, w = rgb.shape[:2]

        max_size = 768

        scale = min(
            1.0,
            max_size / max(h, w),
        )

        nh = max(1, int(h * scale))
        nw = max(1, int(w * scale))

        small = cv2.resize(
            rgb,
            (nw, nh),
            interpolation=cv2.INTER_AREA,
        )

        image = Image.fromarray(
            np.clip(
                small * 255.0,
                0,
                255,
            ).astype(np.uint8)
        )

        if self.transforms is not None:
            tensor = self.transforms(image)
        else:
            arr = np.asarray(image).astype(
                np.float32
            ) / 255.0

            tensor = torch.from_numpy(
                arr.transpose(2, 0, 1)
            )

            mean = torch.tensor(
                [0.485, 0.456, 0.406]
            )[:, None, None]

            std = torch.tensor(
                [0.229, 0.224, 0.225]
            )[:, None, None]

            tensor = (
                tensor - mean
            ) / std

        tensor = tensor.unsqueeze(0).to(
            self.device
        )

        with torch.inference_mode():
            output = self.model(tensor)["out"]

            probs = torch.softmax(
                output,
                dim=1,
            )

            confidence, labels = torch.max(
                probs,
                dim=1,
            )

        labels = labels[0].cpu().numpy()
        confidence = confidence[0].cpu().numpy()

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

        return labels, confidence


# ============================================================
# Saliency
# ============================================================

def compute_saliency(rgb):
    h, w = rgb.shape[:2]

    y = luminance(rgb)

    blurred = cv2.GaussianBlur(
        y,
        (0, 0),
        7,
    )

    local_contrast = np.abs(
        y - blurred
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

    sat = (
        np.max(rgb, axis=2)
        -
        np.min(rgb, axis=2)
    )

    center_y, center_x = np.mgrid[
        0:h,
        0:w,
    ]

    center_x = (
        center_x / max(w - 1, 1)
    )

    center_y = (
        center_y / max(h - 1, 1)
    )

    center = 1.0 - np.sqrt(
        (center_x - 0.5) ** 2
        +
        (center_y - 0.5) ** 2
    ) / 0.7072

    center = np.clip(
        center,
        0,
        1,
    )

    local_contrast = np.clip(
        local_contrast * 5.0,
        0,
        1,
    )

    brightness_distinct = np.abs(
        y - np.mean(y)
    )

    brightness_distinct /= (
        np.percentile(
            brightness_distinct,
            95,
        )
        + 1e-6
    )

    brightness_distinct = np.clip(
        brightness_distinct,
        0,
        1,
    )

    saliency = (
        0.30 * local_contrast
        +
        0.25 * edge
        +
        0.15 * sat
        +
        0.20 * brightness_distinct
        +
        0.10 * center
    )

    return np.clip(
        saliency,
        0,
        1,
    ).astype(np.float32)


# ============================================================
# Subject ranking
# ============================================================

def rank_subjects(
    rgb,
    labels,
    confidence,
    saliency,
):
    h, w = labels.shape

    total = h * w

    result = []

    for class_id, class_name in enumerate(
        VOC_CLASSES
    ):
        if class_name == "background":
            continue

        mask = labels == class_id

        area_pixels = int(
            np.sum(mask)
        )

        area = area_pixels / total

        if area < 0.003:
            continue

        ys, xs = np.where(mask)

        if len(xs) == 0:
            continue

        cx = np.mean(xs) / max(w - 1, 1)
        cy = np.mean(ys) / max(h - 1, 1)

        center_distance = math.sqrt(
            (cx - 0.5) ** 2
            +
            (cy - 0.5) ** 2
        )

        center_score = 1.0 - np.clip(
            center_distance / 0.7072,
            0,
            1,
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

        y = luminance(rgb)

        local_mean = cv2.GaussianBlur(
            y,
            (0, 0),
            7,
        )

        local_contrast = float(
            np.mean(
                np.abs(
                    y[mask]
                    -
                    local_mean[mask]
                )
            )
        )

        local_contrast = float(
            np.clip(
                local_contrast * 5.0,
                0,
                1,
            )
        )

        cf = float(
            np.mean(
                (
                    np.max(
                        rgb[mask],
                        axis=1,
                    )
                    -
                    np.min(
                        rgb[mask],
                        axis=1,
                    )
                )
            )
        )

        prior = 1.0

        if class_name == "person":
            prior = 1.15

        elif class_name in ANIMAL_CLASSES:
            prior = 1.05

        elif class_name in VEHICLE_CLASSES:
            prior = 1.00

        elif class_name == "pottedplant":
            prior = 0.90

        elif class_name == "bottle":
            prior = 0.85

        score = (
            0.25 * conf
            +
            0.15 * np.sqrt(
                min(area / 0.25, 1.0)
            )
            +
            0.15 * center_score
            +
            0.25 * sal
            +
            0.10 * local_contrast
            +
            0.10 * cf
        ) * prior

        result.append(
            SubjectCandidate(
                class_name=class_name,
                confidence=conf,
                area=area,
                center=float(center_score),
                saliency=sal,
                local_contrast=local_contrast,
                colorfulness=cf,
                score=float(score),
            )
        )

    result.sort(
        key=lambda x: x.score,
        reverse=True,
    )

    return result[:10]


# ============================================================
# Region Masks
# ============================================================

def build_region_masks(
    rgb,
    labels,
    subjects,
):
    h, w = labels.shape

    masks = {}

    def class_mask(names):
        ids = [
            VOC_CLASSES.index(name)
            for name in names
            if name in VOC_CLASSES
        ]

        if not ids:
            return np.zeros(
                (h, w),
                dtype=bool,
            )

        result = np.zeros(
            (h, w),
            dtype=bool,
        )

        for idx in ids:
            result |= labels == idx

        return result

    masks["person"] = class_mask(
        {"person"}
    )

    masks["animal"] = class_mask(
        ANIMAL_CLASSES
    )

    masks["vehicle"] = class_mask(
        VEHICLE_CLASSES
    )

    masks["plant"] = class_mask(
        {"pottedplant"}
    )

    # --------------------------------------------------------
    # Skin heuristic inside person regions
    # --------------------------------------------------------

    rgb8 = np.clip(
        rgb * 255.0,
        0,
        255,
    ).astype(np.uint8)

    hsv = cv2.cvtColor(
        rgb8,
        cv2.COLOR_RGB2HSV,
    )

    H = hsv[..., 0]
    S = hsv[..., 1]
    V = hsv[..., 2]

    skin_color = (
        (
            (H <= 25)
            |
            (H >= 165)
        )
        &
        (S >= 20)
        &
        (S <= 180)
        &
        (V >= 45)
    )

    masks["skin"] = (
        masks["person"]
        &
        skin_color
    )

    # --------------------------------------------------------
    # Green vegetation heuristic
    # --------------------------------------------------------

    green = (
        (H >= 30)
        &
        (H <= 95)
        &
        (S >= 45)
        &
        (V >= 30)
    )

    masks["green"] = green

    # --------------------------------------------------------
    # Blue / cyan regions
    # --------------------------------------------------------

    blue = (
        (H >= 80)
        &
        (H <= 135)
        &
        (S >= 40)
        &
        (V >= 40)
    )

    masks["blue"] = blue

    # --------------------------------------------------------
    # Upper bright region
    #
    # VOC has no sky class, so this is deliberately named
    # upper_bright instead of sky.
    # --------------------------------------------------------

    y = luminance(rgb)

    yy = np.arange(h)[:, None] / max(h - 1, 1)

    upper = yy < 0.45

    bright = y > np.percentile(
        y,
        75,
    )

    masks["upper_bright"] = (
        upper
        &
        bright
    )

    # --------------------------------------------------------
    # Water heuristic
    #
    # Conservative: blue/cyan + lower half + relatively
    # low local edge density.
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        rgb8,
        cv2.COLOR_RGB2GRAY,
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

    low_texture = edge < np.percentile(
        edge,
        65,
    )

    lower = yy > 0.35

    masks["water"] = (
        blue
        &
        lower
        &
        low_texture
    )

    # --------------------------------------------------------
    # Subject mask
    # --------------------------------------------------------

    subject_ids = [
        VOC_CLASSES.index(s.class_name)
        for s in subjects
        if s.class_name in VOC_CLASSES
    ]

    subject_mask = np.zeros(
        (h, w),
        dtype=bool,
    )

    for idx in subject_ids:
        subject_mask |= labels == idx

    masks["subject"] = subject_mask

    return masks


# ============================================================
# Region Statistics
# ============================================================

def calculate_region_stats(
    rgb,
    masks,
):
    result = []

    hsv = cv2.cvtColor(
        np.clip(
            rgb * 255.0,
            0,
            255,
        ).astype(np.uint8),
        cv2.COLOR_RGB2HSV,
    )

    saturation = hsv[..., 1] / 255.0

    y = luminance(rgb)

    for name, mask in masks.items():
        area = float(
            np.mean(mask)
        )

        if area < 0.001:
            continue

        mean_y = float(
            np.mean(y[mask])
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
                mean_luminance=mean_y,
                mean_saturation=mean_sat,
            )
        )

    return result


# ============================================================
# Scene Profiles
# ============================================================

SCENE_PROFILES = {
    "portrait": {
        "exposure": 0.05,
        "contrast": 1.02,
        "saturation": 0.97,
        "highlight": 0.42,
        "shadow": 0.08,
        "subject": 0.08,
        "subject_contrast": 1.02,
        "background": 0.035,
        "denoise": 0.55,
        "sharpen": 0.75,
        "skin_sat": 0.94,
        "green_sat": 1.00,
        "water_highlight": 0.06,
        "upper_highlight": 0.12,
        "tone": 0.55,
    },

    "night": {
        "exposure": 0.00,
        "contrast": 1.05,
        "saturation": 1.03,
        "highlight": 0.55,
        "shadow": 0.02,
        "subject": 0.05,
        "subject_contrast": 1.03,
        "background": 0.015,
        "denoise": 0.85,
        "sharpen": 0.45,
        "skin_sat": 0.94,
        "green_sat": 1.00,
        "water_highlight": 0.10,
        "upper_highlight": 0.18,
        "tone": 0.45,
    },

    "sunset": {
        "exposure": -0.05,
        "contrast": 1.06,
        "saturation": 1.08,
        "highlight": 0.55,
        "shadow": 0.04,
        "subject": 0.05,
        "subject_contrast": 1.03,
        "background": 0.015,
        "denoise": 0.30,
        "sharpen": 0.80,
        "skin_sat": 0.96,
        "green_sat": 1.02,
        "water_highlight": 0.10,
        "upper_highlight": 0.20,
        "tone": 0.60,
    },

    "landscape": {
        "exposure": 0.03,
        "contrast": 1.08,
        "saturation": 1.04,
        "highlight": 0.40,
        "shadow": 0.08,
        "subject": 0.06,
        "subject_contrast": 1.03,
        "background": 0.015,
        "denoise": 0.30,
        "sharpen": 0.85,
        "skin_sat": 0.96,
        "green_sat": 1.03,
        "water_highlight": 0.10,
        "upper_highlight": 0.18,
        "tone": 0.60,
    },

    "city": {
        "exposure": 0.02,
        "contrast": 1.07,
        "saturation": 1.02,
        "highlight": 0.45,
        "shadow": 0.05,
        "subject": 0.06,
        "subject_contrast": 1.03,
        "background": 0.02,
        "denoise": 0.40,
        "sharpen": 0.80,
        "skin_sat": 0.95,
        "green_sat": 1.01,
        "water_highlight": 0.08,
        "upper_highlight": 0.16,
        "tone": 0.58,
    },

    "indoor": {
        "exposure": 0.04,
        "contrast": 1.03,
        "saturation": 0.99,
        "highlight": 0.40,
        "shadow": 0.08,
        "subject": 0.05,
        "subject_contrast": 1.02,
        "background": 0.015,
        "denoise": 0.50,
        "sharpen": 0.65,
        "skin_sat": 0.94,
        "green_sat": 1.00,
        "water_highlight": 0.05,
        "upper_highlight": 0.10,
        "tone": 0.50,
    },

    "general": {
        "exposure": 0.00,
        "contrast": 1.04,
        "saturation": 1.00,
        "highlight": 0.35,
        "shadow": 0.06,
        "subject": 0.04,
        "subject_contrast": 1.02,
        "background": 0.015,
        "denoise": 0.30,
        "sharpen": 0.75,
        "skin_sat": 0.96,
        "green_sat": 1.01,
        "water_highlight": 0.07,
        "upper_highlight": 0.12,
        "tone": 0.52,
    },
}


# ============================================================
# Auto Exposure
# ============================================================

def estimate_auto_exposure(stats, scene):
    """
    Estimate a conservative global EV correction.

    This is deliberately bounded to avoid extreme changes.
    """

    targets = {
        "portrait": 0.18,
        "night": 0.08,
        "sunset": 0.14,
        "landscape": 0.20,
        "city": 0.18,
        "indoor": 0.17,
        "general": 0.18,
    }

    target = targets.get(
        scene.scene,
        0.18,
    )

    median = max(
        stats.median,
        1e-5,
    )

    ev_mid = math.log2(
        target / median
    )

    p95 = max(
        stats.p95,
        1e-5,
    )

    # Avoid blowing highlights.
    ev_high = math.log2(
        0.72 / p95
    )

    ev = (
        0.60 * ev_mid
        +
        0.40 * ev_high
    )

    ev = float(
        np.clip(
            ev,
            -0.75,
            0.75,
        )
    )

    return ev


# ============================================================
# Tone Processing
# ============================================================

def tone_map_luminance(
    rgb,
    contrast,
    highlight_strength,
    shadow_strength,
    tone_strength,
):
    y = luminance(rgb)

    # Shadow lift
    shadow_zone = 1.0 - smoothstep(
        0.0,
        0.35,
        y,
    )

    y2 = y + (
        shadow_strength
        * 0.055
        * shadow_zone
    )

    # Highlight roll-off
    highlight_zone = smoothstep(
        0.60,
        1.00,
        y2,
    )

    y2 = y2 * (
        1.0
        -
        highlight_strength
        * 0.15
        * highlight_zone
    )

    # Controlled contrast
    pivot = 0.18

    y_contrast = (
        pivot
        +
        (y2 - pivot)
        * contrast
    )

    # Additional mild S-curve
    if tone_strength > 0:
        normalized = np.clip(
            y_contrast,
            0,
            1,
        )

        curve = (
            normalized
            +
            tone_strength
            * 0.08
            * (
                normalized
                -
                normalized ** 2
            )
        )

        y_contrast = (
            0.75 * y_contrast
            +
            0.25 * curve
        )

    y_contrast = np.clip(
        y_contrast,
        0,
        None,
    )

    ratio = (
        y_contrast
        /
        np.maximum(y, 1e-5)
    )

    return rgb * ratio[..., None]


# ============================================================
# Region-aware processing
# ============================================================

def apply_region_adjustments(
    rgb,
    masks,
    params,
):
    result = rgb.copy()

    y = luminance(result)

    # --------------------------------------------------------
    # Main subject
    # --------------------------------------------------------

    subject = masks.get(
        "subject",
        np.zeros(
            y.shape,
            dtype=bool,
        ),
    )

    if np.any(subject):
        subject_gain = (
            2.0 ** params.subject_exposure
        )

        result[subject] *= subject_gain

        ys = y[subject]

        if ys.size > 0:
            pivot = float(
                np.median(ys)
            )

            local_y = luminance(
                result[subject]
            )

            local_y2 = (
                pivot
                +
                (
                    local_y
                    -
                    pivot
                )
                *
                params.subject_contrast
            )

            ratio = (
                local_y2
                /
                np.maximum(
                    local_y,
                    1e-5,
                )
            )

            result[subject] *= (
                ratio[:, None]
            )

    # --------------------------------------------------------
    # Skin
    # --------------------------------------------------------

    skin = masks.get(
        "skin",
        np.zeros(
            y.shape,
            dtype=bool,
        ),
    )

    if np.any(skin):
        yy = luminance(result)

        skin_sat = (
            params.region_skin_saturation
        )

        result[skin] = (
            yy[skin, None]
            +
            (
                result[skin]
                -
                yy[skin, None]
            )
            * skin_sat
        )

    # --------------------------------------------------------
    # Green vegetation
    # --------------------------------------------------------

    green = masks.get(
        "green",
        np.zeros(
            y.shape,
            dtype=bool,
        ),
    )

    if np.any(green):
        yy = luminance(result)

        result[green] = (
            yy[green, None]
            +
            (
                result[green]
                -
                yy[green, None]
            )
            *
            params.region_green_saturation
        )

    # --------------------------------------------------------
    # Water
    # --------------------------------------------------------

    water = masks.get(
        "water",
        np.zeros(
            y.shape,
            dtype=bool,
        ),
    )

    if np.any(water):
        yy = luminance(result)

        result[water] = (
            yy[water, None]
            +
            (
                result[water]
                -
                yy[water, None]
            )
            * 1.02
        )

        wy = luminance(
            result[water]
        )

        protection = smoothstep(
            0.60,
            1.0,
            wy,
        )

        result[water] *= (
            1.0
            -
            params.region_water_highlight
            * 0.12
            * protection[:, None]
        )

    # --------------------------------------------------------
    # Bright upper region
    #
    # Not called "sky" because VOC has no sky class.
    # --------------------------------------------------------

    upper = masks.get(
        "upper_bright",
        np.zeros(
            y.shape,
            dtype=bool,
        ),
    )

    if np.any(upper):
        yy = luminance(
            result[upper]
        )

        protection = smoothstep(
            0.55,
            1.0,
            yy,
        )

        result[upper] *= (
            1.0
            -
            params.region_upper_highlight
            * 0.16
            * protection[:, None]
        )

    # --------------------------------------------------------
    # Background suppression
    # --------------------------------------------------------

    if np.any(subject):
        background = ~subject

        bg_gain = (
            1.0
            -
            params.background_suppression
            * 0.05
        )

        yy = luminance(
            result[background]
        )

        result[background] = (
            yy[:, None]
            +
            (
                result[background]
                -
                yy[:, None]
            )
            * bg_gain
        )

    return np.clip(
        result,
        0,
        None,
    )


# ============================================================
# Render
# ============================================================

def render_candidate(
    linear_rgb,
    params,
    masks,
):
    rgb = linear_rgb.copy()

    rgb = apply_exposure(
        rgb,
        params.exposure_ev,
    )

    rgb = tone_map_luminance(
        rgb,

        contrast=params.contrast,

        highlight_strength=(
            params.highlight_protection
        ),

        shadow_strength=(
            params.shadow_lift
        ),

        tone_strength=params.tone_strength,
    )

    rgb = apply_region_adjustments(
        rgb,
        masks,
        params,
    )

    rgb = np.maximum(
        rgb,
        0,
    )

    return rgb


# ============================================================
# Candidate Scoring
# ============================================================

def score_candidate(
    linear_rgb,
    scene,
):
    y = luminance(
        np.clip(
            linear_rgb,
            0,
            1,
        )
    )

    p01 = np.percentile(
        y,
        1,
    )

    p50 = np.percentile(
        y,
        50,
    )

    p95 = np.percentile(
        y,
        95,
    )

    highlight_clip = float(
        np.mean(y > 0.985)
    )

    shadow_clip = float(
        np.mean(y < 0.008)
    )

    # Scene-specific median targets.
    targets = {
        "portrait": 0.18,
        "night": 0.08,
        "sunset": 0.14,
        "landscape": 0.20,
        "city": 0.18,
        "indoor": 0.17,
        "general": 0.18,
    }

    target = targets.get(
        scene.scene,
        0.18,
    )

    mid_error = abs(
        math.log2(
            max(p50, 1e-5)
            /
            target
        )
    )

    contrast = float(
        np.std(y)
    )

    target_contrast = {
        "portrait": 0.22,
        "night": 0.18,
        "sunset": 0.25,
        "landscape": 0.24,
        "city": 0.22,
        "indoor": 0.20,
        "general": 0.21,
    }.get(
        scene.scene,
        0.21,
    )

    contrast_error = abs(
        contrast
        -
        target_contrast
    )

    sat = (
        np.max(
            np.clip(
                linear_rgb,
                0,
                1,
            ),
            axis=2,
        )
        -
        np.min(
            np.clip(
                linear_rgb,
                0,
                1,
            ),
            axis=2,
        )
    )

    oversaturation = float(
        np.mean(sat > 0.72)
    )

    # Prefer preserved highlight detail.
    score = (
        4.0 * highlight_clip
        +
        1.0 * shadow_clip
        +
        0.65 * mid_error
        +
        0.50 * contrast_error
        +
        0.80 * oversaturation
    )

    # Night scenes should tolerate darker shadows.
    if scene.scene == "night":
        score -= 0.15 * (
            1.0 - p95
        )

    # Landscapes benefit from slightly higher contrast.
    if scene.scene == "landscape":
        score -= 0.15 * contrast

    return float(score)


# ============================================================
# Automatic parameter search
# ============================================================

def search_parameters(
    linear_rgb,
    profile,
    stats,
    scene,
    shooting,
    masks,
):
    base = SCENE_PROFILES.get(
        scene.scene,
        SCENE_PROFILES["general"],
    )

    auto_ev = estimate_auto_exposure(
        stats,
        scene,
    )

    ev_values = [
        -0.25,
        -0.12,
        0.0,
        0.12,
        0.25,
    ]

    contrast_values = [
        0.98,
        1.00,
        1.03,
        1.06,
    ]

    saturation_values = [
        0.97,
        1.00,
        1.03,
    ]

    best_score = float("inf")
    best_params = None

    # Search on reduced image for speed.
    h, w = linear_rgb.shape[:2]

    scale = min(
        1.0,
        512 / max(h, w),
    )

    if scale < 1.0:
        small = cv2.resize(
            linear_rgb,
            (
                int(w * scale),
                int(h * scale),
            ),
            interpolation=cv2.INTER_AREA,
        )

        small_masks = {}

        for name, mask in masks.items():
            small_masks[name] = cv2.resize(
                mask.astype(np.uint8),
                (
                    int(w * scale),
                    int(h * scale),
                ),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)

    else:
        small = linear_rgb
        small_masks = masks

    for ev in ev_values:
        for contrast in contrast_values:
            for saturation in saturation_values:

                params = DevelopParams(
                    exposure_ev=(
                        base["exposure"]
                        +
                        auto_ev
                        +
                        ev
                    ),

                    contrast=(
                        base["contrast"]
                        *
                        contrast
                    ),

                    saturation=(
                        base["saturation"]
                        *
                        saturation
                    ),

                    highlight_protection=(
                        base["highlight"]
                    ),

                    shadow_lift=(
                        base["shadow"]
                    ),

                    subject_exposure=(
                        base["subject"]
                    ),

                    subject_contrast=(
                        base["subject_contrast"]
                    ),

                    background_suppression=(
                        base["background"]
                    ),

                    denoise=(
                        base["denoise"]
                        *
                        (
                            1.0
                            +
                            shooting.estimated_noise
                        )
                    ),

                    sharpen=(
                        base["sharpen"]
                        *
                        (
                            1.0
                            -
                            0.35
                            * shooting.estimated_noise
                        )
                    ),

                    region_skin_saturation=(
                        base["skin_sat"]
                    ),

                    region_green_saturation=(
                        base["green_sat"]
                    ),

                    region_water_highlight=(
                        base["water_highlight"]
                    ),

                    region_upper_highlight=(
                        base["upper_highlight"]
                    ),

                    tone_strength=(
                        base["tone"]
                    ),
                )

                candidate = render_candidate(
                    small,
                    params,
                    small_masks,
                )

                # Saturation adjustment is done around luminance.
                y = luminance(candidate)

                candidate = (
                    y[..., None]
                    +
                    (
                        candidate
                        -
                        y[..., None]
                    )
                    * params.saturation
                )

                candidate = np.clip(
                    candidate,
                    0,
                    None,
                )

                score = score_candidate(
                    candidate,
                    scene,
                )

                if score < best_score:
                    best_score = score
                    best_params = params

    return best_params, best_score


# ============================================================
# Denoise
# ============================================================

def denoise_image(
    rgb,
    strength,
):
    if strength <= 0.01:
        return rgb

    rgb8 = np.clip(
        linear_to_srgb(
            rgb
        ) * 255.0,
        0,
        255,
    ).astype(np.uint8)

    # Keep it relatively conservative.
    sigma_color = 10.0 + 20.0 * strength
    sigma_space = 3.0 + 3.0 * strength

    filtered = cv2.bilateralFilter(
        rgb8,
        d=7,
        sigmaColor=sigma_color,
        sigmaSpace=sigma_space,
    )

    result = (
        filtered.astype(np.float32)
        /
        255.0
    )

    return srgb_to_linear(
        result
    )


# ============================================================
# Sharpen
# ============================================================

def sharpen_luminance(
    rgb,
    amount,
):
    if amount <= 0.01:
        return rgb

    y = luminance(rgb)

    sigma = 0.8

    blur = cv2.GaussianBlur(
        y,
        (0, 0),
        sigma,
    )

    detail = y - blur

    y2 = y + (
        amount
        * 0.8
        * detail
    )

    y2 = np.clip(
        y2,
        0,
        None,
    )

    ratio = (
        y2
        /
        np.maximum(y, 1e-5)
    )

    result = (
        rgb
        *
        ratio[..., None]
    )

    return np.maximum(
        result,
        0,
    )


# ============================================================
# Save
# ============================================================

def save_jpeg(
    linear_rgb,
    path,
    quality=95,
):
    srgb = linear_to_srgb(
        np.maximum(
            linear_rgb,
            0,
        )
    )

    srgb = np.clip(
        srgb,
        0,
        1,
    )

    image = Image.fromarray(
        (
            srgb * 255.0
        ).round().astype(
            np.uint8
        ),
        mode="RGB",
    )

    image.save(
        path,
        "JPEG",
        quality=quality,
        optimize=True,
    )


# ============================================================
# Debug
# ============================================================

def save_debug_images(
    output_dir,
    stem,
    rgb,
    labels,
    saliency,
    masks,
):
    debug_dir = (
        output_dir
        /
        "debug"
    )

    debug_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Segmentation
    # --------------------------------------------------------

    seg = np.zeros(
        (*labels.shape, 3),
        dtype=np.uint8,
    )

    for idx in range(
        len(VOC_CLASSES)
    ):
        mask = labels == idx

        value = (
            (idx * 37) % 255
        )

        seg[mask] = (
            value,
            (value * 3) % 255,
            (value * 7) % 255,
        )

    Image.fromarray(
        seg
    ).save(
        debug_dir
        /
        f"{stem}_seg.png"
    )

    # --------------------------------------------------------
    # Saliency
    # --------------------------------------------------------

    sal = (
        np.clip(
            saliency,
            0,
            1,
        )
        * 255
    ).astype(
        np.uint8
    )

    sal_color = cv2.applyColorMap(
        sal,
        cv2.COLORMAP_JET,
    )

    cv2.imwrite(
        str(
            debug_dir
            /
            f"{stem}_saliency.png"
        ),
        sal_color,
    )

    # --------------------------------------------------------
    # Region masks
    # --------------------------------------------------------

    for name, mask in masks.items():
        mask8 = (
            mask.astype(np.uint8)
            * 255
        )

        cv2.imwrite(
            str(
                debug_dir
                /
                f"{stem}_{name}.png"
            ),
            mask8,
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

        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device(
                    "cuda"
                )
            else:
                self.device = torch.device(
                    "cpu"
                )
        else:
            if (
                device == "cuda"
                and
                not torch.cuda.is_available()
            ):
                print(
                    "[WARN] CUDA requested but "
                    "not available. Falling back "
                    "to CPU."
                )

                self.device = torch.device(
                    "cpu"
                )
            else:
                self.device = torch.device(
                    device
                )

        print(
            f"[INFO] Device: {self.device}"
        )

        self.segmenter = SemanticSegmenter(
            self.device
        )

    def process(
        self,
        input_path,
        output_path,
        json_path,
        debug_dir,
    ):
        print(
            f"[INFO] Processing: {input_path}"
        )

        metadata = get_metadata(
            input_path
        )

        with rawpy.imread(
            str(input_path)
        ) as raw:

            profile = build_camera_profile(
                raw,
                metadata,
            )

            # ------------------------------------------------
            # RAW -> camera RGB
            # ------------------------------------------------

            try:
                camera_rgb = raw_to_camera_rgb(
                    raw
                )

                linear_rgb = (
                    camera_rgb_to_linear_srgb(
                        camera_rgb,
                        profile,
                    )
                )

                conversion_method = (
                    "camera_rgb_matrix"
                )

            except Exception as exc:
                print(
                    "[WARN] Camera RGB conversion "
                    f"failed: {exc}"
                )

                print(
                    "[INFO] Falling back to "
                    "LibRaw sRGB conversion."
                )

                linear_rgb = (
                    raw_to_srgb_fallback(
                        raw
                    )
                )

                conversion_method = (
                    "libraw_srgb_fallback"
                )

        # ----------------------------------------------------
        # Basic analysis
        # ----------------------------------------------------

        stats = analyze_image(
            np.clip(
                linear_rgb,
                0,
                1,
            )
        )

        shooting = analyze_shooting(
            metadata,
            stats,
        )

        # ----------------------------------------------------
        # Semantic segmentation
        # ----------------------------------------------------

        labels, confidence = (
            self.segmenter.predict(
                np.clip(
                    linear_rgb,
                    0,
                    1,
                )
            )
        )

        # ----------------------------------------------------
        # Saliency
        # ----------------------------------------------------

        saliency = compute_saliency(
            np.clip(
                linear_rgb,
                0,
                1,
            )
        )

        # ----------------------------------------------------
        # Subjects
        # ----------------------------------------------------

        subjects = rank_subjects(
            np.clip(
                linear_rgb,
                0,
                1,
            ),
            labels,
            confidence,
            saliency,
        )

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
            f"({scene.confidence:.2f})"
        )

        # ----------------------------------------------------
        # Regions
        # ----------------------------------------------------

        masks = build_region_masks(
            np.clip(
                linear_rgb,
                0,
                1,
            ),
            labels,
            subjects,
        )

        region_stats = calculate_region_stats(
            np.clip(
                linear_rgb,
                0,
                1,
            ),
            masks,
        )

        # ----------------------------------------------------
        # Parameter search
        # ----------------------------------------------------

        params, search_score = (
            search_parameters(
                linear_rgb,
                profile,
                stats,
                scene,
                shooting,
                masks,
            )
        )

        print(
            "[INFO] Selected parameters:"
        )

        print(
            f"       EV       = "
            f"{params.exposure_ev:+.3f}"
        )

        print(
            f"       Contrast = "
            f"{params.contrast:.3f}"
        )

        print(
            f"       Saturation = "
            f"{params.saturation:.3f}"
        )

        print(
            f"       Denoise = "
            f"{params.denoise:.3f}"
        )

        print(
            f"       Sharpen = "
            f"{params.sharpen:.3f}"
        )

        # ----------------------------------------------------
        # Final render
        # ----------------------------------------------------

        result = render_candidate(
            linear_rgb,
            params,
            masks,
        )

        # Global saturation
        y = luminance(result)

        result = (
            y[..., None]
            +
            (
                result
                -
                y[..., None]
            )
            * params.saturation
        )

        result = np.maximum(
            result,
            0,
        )

        # ----------------------------------------------------
        # Denoise
        # ----------------------------------------------------

        result = denoise_image(
            result,
            params.denoise,
        )

        # ----------------------------------------------------
        # Sharpen
        # ----------------------------------------------------

        result = sharpen_luminance(
            result,
            params.sharpen,
        )

        # ----------------------------------------------------
        # Final clip
        # ----------------------------------------------------

        result = np.clip(
            result,
            0,
            1,
        )

        # ----------------------------------------------------
        # Save JPEG
        # ----------------------------------------------------

        save_jpeg(
            result,
            output_path,
            quality=JPEG_QUALITY,
        )

        # ----------------------------------------------------
        # Debug
        # ----------------------------------------------------

        if self.debug:
            save_debug_images(
                debug_dir,
                Path(input_path).stem,
                np.clip(
                    linear_rgb,
                    0,
                    1,
                ),
                labels,
                saliency,
                masks,
            )

        # ----------------------------------------------------
        # JSON
        # ----------------------------------------------------

        report = {
            "version": "v17",

            "input": str(
                input_path
            ),

            "output": str(
                output_path
            ),

            "conversion_method": (
                conversion_method
            ),

            "camera": asdict(
                profile
            ),

            "metadata": asdict(
                metadata
            ),

            "stats": asdict(
                stats
            ),

            "shooting": asdict(
                shooting
            ),

            "scene": asdict(
                scene
            ),

            "subjects": [
                asdict(x)
                for x in subjects
            ],

            "regions": [
                asdict(x)
                for x in region_stats
            ],

            "selected_parameters": asdict(
                params
            ),

            "search_score": (
                search_score
            ),
        }

        with open(
            json_path,
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
            f"[INFO] Saved: {output_path}"
        )

        print(
            f"[INFO] Report: {json_path}"
        )


# ============================================================
# File Collection
# ============================================================

def collect_raw_files(input_path):
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
            and
            path.suffix.lower()
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
            "Automatic RAW photo development "
            "v17"
        )
    )

    parser.add_argument(
        "input",
        help="RAW file or directory",
    )

    parser.add_argument(
        "-o",
        "--output",
        default="./output",
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
        help="Save segmentation/saliency debug images",
    )

    args = parser.parse_args()

    input_path = Path(
        args.input
    )

    output_dir = Path(
        args.output
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    debug_dir = (
        output_dir
        /
        "debug"
    )

    raw_files = collect_raw_files(
        input_path
    )

    if not raw_files:
        print(
            "[ERROR] No RAW files found."
        )

        return 1

    print(
        f"[INFO] RAW files: "
        f"{len(raw_files)}"
    )

    developer = AutoDeveloper(
        device=args.device,
        debug=args.debug,
    )

    success = 0
    failed = 0

    for raw_path in raw_files:

        try:
            if input_path.is_dir():
                relative = raw_path.relative_to(
                    input_path
                )

                output_path = (
                    output_dir
                    /
                    relative.parent
                    /
                    f"{raw_path.stem}_developed.jpg"
                )

                json_path = (
                    output_dir
                    /
                    relative.parent
                    /
                    f"{raw_path.stem}_analysis.json"
                )

            else:
                output_path = (
                    output_dir
                    /
                    f"{raw_path.stem}_developed.jpg"
                )

                json_path = (
                    output_dir
                    /
                    f"{raw_path.stem}_analysis.json"
                )

            output_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            json_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            developer.process(
                raw_path,
                output_path,
                json_path,
                debug_dir,
            )

            success += 1

        except Exception as exc:
            failed += 1

            print(
                f"[ERROR] Failed: "
                f"{raw_path}"
            )

            print(
                f"        "
                f"{type(exc).__name__}: "
                f"{exc}"
            )

    print()
    print(
        "========================================"
    )

    print(
        f"[INFO] Success: {success}"
    )

    print(
        f"[INFO] Failed : {failed}"
    )

    print(
        "========================================"
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