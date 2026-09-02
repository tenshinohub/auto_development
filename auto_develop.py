#!/usr/bin/env python3

import argparse
import json
import math
import os
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


VERSION = "v18"

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


# ============================================================
# Data classes
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

    black_level: list = None
    white_level: list = None

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
    mean: float = 0.0
    median: float = 0.0
    p01: float = 0.0
    p05: float = 0.0
    p95: float = 0.0
    p99: float = 0.0

    shadow_ratio: float = 0.0
    highlight_ratio: float = 0.0
    dynamic_range: float = 0.0
    saturation_ratio: float = 0.0

    edge_density: float = 0.0
    warm_ratio: float = 0.0
    contrast: float = 0.0
    mean_luminance: float = 0.0


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
    class_name: str
    confidence: float
    area: float
    center: tuple
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

def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def safe_float(value, default=0.0):
    try:
        if value is None:
            return default

        if isinstance(value, str):
            value = value.strip()

            if not value:
                return default

            if "/" in value:
                a, b = value.split("/", 1)
                return float(a) / float(b)

        return float(value)

    except Exception:
        return default


def safe_str(value, default=""):
    if value is None:
        return default
    return str(value)


def percentile(arr, p):
    if arr.size == 0:
        return 0.0
    return float(np.percentile(arr, p))


# ============================================================
# ExifTool
# ============================================================

def run_exiftool(path):
    try:
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

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10,
        )

        if result.returncode != 0:
            return {}

        data = json.loads(result.stdout)

        if not data:
            return {}

        return data[0]

    except Exception:
        return {}


# ============================================================
# Pillow EXIF
# ============================================================

def read_pillow_exif(path):
    result = {}

    try:
        with Image.open(path) as img:
            exif = img.getexif()

            if not exif:
                return result

            tag_map = {}

            for key, value in exif.items():
                name = ExifTags.TAGS.get(key, key)
                tag_map[name] = value

            result["Make"] = tag_map.get("Make", "")
            result["Model"] = tag_map.get("Model", "")

            result["LensModel"] = tag_map.get(
                "LensModel",
                tag_map.get("LensSpecification", ""),
            )

            result["ISO"] = tag_map.get(
                "ISOSpeedRatings",
                tag_map.get("PhotographicSensitivity", 0),
            )

            result["ExposureTime"] = tag_map.get(
                "ExposureTime",
                0,
            )

            result["FNumber"] = tag_map.get(
                "FNumber",
                0,
            )

            result["FocalLength"] = tag_map.get(
                "FocalLength",
                0,
            )

            result["WhiteBalance"] = tag_map.get(
                "WhiteBalance",
                "",
            )

            result["ColorSpace"] = tag_map.get(
                "ColorSpace",
                "",
            )

    except Exception:
        pass

    return result


# ============================================================
# Metadata
# ============================================================

def get_metadata(path):
    data = run_exiftool(path)
    source = "exiftool"

    if not data:
        data = read_pillow_exif(path)
        source = "pillow"

    make = safe_str(data.get("Make", ""))
    model = safe_str(
        data.get(
            "CameraModelName",
            data.get(
                "UniqueCameraModel",
                data.get("Model", ""),
            ),
        )
    )

    return ExifMetadata(
        make=make,
        model=model,
        lens_make=safe_str(data.get("LensMake", "")),
        lens_model=safe_str(data.get("LensModel", "")),

        iso=safe_float(data.get("ISO", 0)),
        exposure_time=safe_float(data.get("ExposureTime", 0)),
        f_number=safe_float(data.get("FNumber", 0)),
        focal_length=safe_float(data.get("FocalLength", 0)),

        white_balance=safe_str(data.get("WhiteBalance", "")),
        color_temperature=safe_float(
            data.get("ColorTemperature", 0)
        ),
        color_space=safe_str(
            data.get("ColorSpace", "")
        ),

        source=source,
    )


# ============================================================
# Camera family
# ============================================================

def detect_camera_family(make, model):
    text = f"{make} {model}".lower()

    families = [
        ("canon", ["canon"]),
        ("nikon", ["nikon"]),
        ("sony", ["sony"]),
        ("fujifilm", ["fujifilm", "fuji"]),
        ("panasonic", ["panasonic", "lumix"]),
        ("olympus", ["olympus", "om system"]),
        ("leica", ["leica"]),
        ("pentax", ["pentax"]),
        ("ricoh", ["ricoh"]),
        ("sigma", ["sigma"]),
        ("hasselblad", ["hasselblad"]),
    ]

    for family, keywords in families:
        if any(k in text for k in keywords):
            return family

    return "unknown"


# ============================================================
# Camera profile
# ============================================================

def make_camera_profile(raw, metadata):
    try:
        black_level = np.asarray(
            raw.black_level_per_channel
        ).astype(float).tolist()
    except Exception:
        black_level = []

    try:
        white_level = np.asarray(
            raw.camera_white_level_per_channel
        ).astype(float).tolist()
    except Exception:
        white_level = []

    try:
        camera_wb = np.asarray(
            raw.camera_whitebalance
        ).astype(float).tolist()
    except Exception:
        camera_wb = []

    try:
        color_matrix = np.asarray(
            raw.color_matrix
        ).astype(float).tolist()
    except Exception:
        color_matrix = []

    try:
        rgb_xyz_matrix = np.asarray(
            raw.rgb_xyz_matrix
        ).astype(float).tolist()
    except Exception:
        rgb_xyz_matrix = []

    try:
        raw_width, raw_height = raw.raw_image.shape[1], raw.raw_image.shape[0]
    except Exception:
        raw_width = 0
        raw_height = 0

    try:
        libraw_version = safe_str(
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
# RAW development
# ============================================================

def raw_to_srgb(raw):
    """
    v18:
    独自の camera RGB -> XYZ -> sRGB 変換を廃止。

    LibRawにsRGB変換を任せる。
    これによりv17で発生していた強い緑被りを回避する。
    """

    rgb = raw.postprocess(
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

    return rgb


def raw_to_srgb_fallback(raw):
    """
    念のため通常のLibRaw sRGB処理へフォールバック。
    """

    rgb = raw.postprocess(
        use_camera_wb=True,
        use_auto_wb=False,

        output_color=rawpy.ColorSpace.sRGB,

        output_bps=16,

        no_auto_bright=False,

        highlight_mode=rawpy.HighlightMode.Blend,

        half_size=False,
        four_color_rgb=False,

        demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD,
    )

    return rgb


# ============================================================
# RGB conversion
# ============================================================

def srgb16_to_float(rgb):
    """
    16bit sRGB -> float [0,1]
    """

    return np.asarray(rgb, dtype=np.float32) / 65535.0


def srgb_to_linear(rgb):
    rgb = np.clip(rgb, 0.0, 1.0)

    return np.where(
        rgb <= 0.04045,
        rgb / 12.92,
        ((rgb + 0.055) / 1.055) ** 2.4,
    )


def linear_to_srgb(rgb):
    rgb = np.clip(rgb, 0.0, 1.0)

    return np.where(
        rgb <= 0.0031308,
        rgb * 12.92,
        1.055 * np.power(rgb, 1.0 / 2.4) - 0.055,
    )


# ============================================================
# Image analysis
# ============================================================

def luminance(rgb):
    return (
        0.2126 * rgb[..., 0]
        + 0.7152 * rgb[..., 1]
        + 0.0722 * rgb[..., 2]
    )


def analyze_image(rgb):
    lum = luminance(rgb)

    flat = lum.reshape(-1)

    mean = float(np.mean(flat))
    median = float(np.median(flat))

    p01 = percentile(flat, 1)
    p05 = percentile(flat, 5)
    p95 = percentile(flat, 95)
    p99 = percentile(flat, 99)

    shadow_ratio = float(np.mean(flat < 0.02))
    highlight_ratio = float(np.mean(flat > 0.98))

    dynamic_range = math.log2(
        max(p95, 1e-4) /
        max(p05, 1e-4)
    )

    hsv = cv2.cvtColor(
        np.clip(rgb * 255, 0, 255).astype(np.uint8),
        cv2.COLOR_RGB2HSV,
    )

    saturation = hsv[..., 1] / 255.0

    saturation_ratio = float(
        np.mean(saturation > 0.85)
    )

    gray = np.clip(
        lum * 255,
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

    edge = cv2.magnitude(
        sobel_x,
        sobel_y,
    )

    edge_density = float(
        np.mean(edge > 35)
    )

    warm_ratio = float(
        np.mean(
            (rgb[..., 0] > rgb[..., 2] * 1.15)
            &
            (rgb[..., 0] > rgb[..., 1] * 1.03)
        )
    )

    contrast = float(np.std(lum))

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
# Shooting conditions
# ============================================================

def analyze_shooting(metadata):
    iso = metadata.iso
    exposure = metadata.exposure_time
    f_number = metadata.f_number
    focal = metadata.focal_length

    if iso <= 0:
        iso = 100

    iso_factor = clamp(
        math.sqrt(iso / 100.0),
        1.0,
        5.0,
    )

    if exposure > 0:
        motion_risk = clamp(
            math.log2(
                max(exposure, 1e-6) / (1 / 125)
            ) / 4 + 0.5,
            0.0,
            1.0,
        )
    else:
        motion_risk = 0.0

    shallow_dof = (
        0 < f_number <= 2.8
    )

    wide_angle = (
        0 < focal <= 28
    )

    telephoto = (
        focal >= 85
    )

    low_light = (
        iso >= 1600
    )

    estimated_noise = clamp(
        (iso_factor - 1) / 4,
        0,
        1,
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
# DeepLab segmentation
# ============================================================

class SemanticSegmenter:

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

    def __init__(self, device="auto"):

        if device == "auto":
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        if device == "cuda" and not torch.cuda.is_available():
            warnings.warn(
                "CUDA requested but unavailable. Falling back to CPU."
            )
            device = "cpu"

        self.device = torch.device(device)

        print(
            f"[INFO] Segmentation device: "
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

            self.transform = weights.transforms()

        except Exception:

            self.model = (
                deeplabv3_mobilenet_v3_large(
                    pretrained=True
                )
            )

            self.transform = None

        self.model.to(self.device)
        self.model.eval()

    def predict(self, rgb):

        h, w = rgb.shape[:2]

        scale = min(
            1.0,
            768.0 / max(h, w)
        )

        if scale < 1:

            small = cv2.resize(
                rgb,
                (
                    int(w * scale),
                    int(h * scale),
                ),
                interpolation=cv2.INTER_AREA,
            )

        else:
            small = rgb

        tensor = (
            torch.from_numpy(
                small.transpose(2, 0, 1)
            )
            .float()
        )

        if self.transform is not None:

            pil = Image.fromarray(
                np.clip(
                    small * 255,
                    0,
                    255,
                ).astype(np.uint8)
            )

            tensor = self.transform(pil)

        else:

            tensor = (
                tensor / 255.0
            )

        tensor = tensor.unsqueeze(0).to(
            self.device
        )

        with torch.inference_mode():

            output = self.model(
                tensor
            )["out"][0]

            probs = torch.softmax(
                output,
                dim=0,
            )

            confidence, labels = torch.max(
                probs,
                dim=0,
            )

        labels = (
            labels
            .cpu()
            .numpy()
            .astype(np.uint8)
        )

        confidence = (
            confidence
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        labels = cv2.resize(
            labels,
            (w, h),
            interpolation=cv2.INTER_NEAREST,
        )

        confidence = cv2.resize(
            confidence,
            (w, h),
            interpolation=cv2.INTER_LINEAR,
        )

        return labels, confidence


# ============================================================
# Saliency
# ============================================================

def compute_saliency(rgb):

    lum = luminance(rgb)

    gray = np.clip(
        lum * 255,
        0,
        255,
    ).astype(np.uint8)

    blur = cv2.GaussianBlur(
        gray,
        (0, 0),
        5,
    )

    local_contrast = cv2.absdiff(
        gray,
        blur,
    ).astype(np.float32) / 255.0

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

    edge /= (
        np.percentile(edge, 99)
        + 1e-6
    )

    edge = np.clip(
        edge,
        0,
        1,
    )

    hsv = cv2.cvtColor(
        np.clip(
            rgb * 255,
            0,
            255,
        ).astype(np.uint8),
        cv2.COLOR_RGB2HSV,
    )

    saturation = (
        hsv[..., 1].astype(np.float32)
        / 255.0
    )

    mean_lum = np.mean(lum)

    brightness_dist = np.abs(
        lum - mean_lum
    )

    brightness_dist /= (
        np.percentile(
            brightness_dist,
            99,
        )
        + 1e-6
    )

    brightness_dist = np.clip(
        brightness_dist,
        0,
        1,
    )

    h, w = lum.shape

    yy, xx = np.mgrid[
        0:h,
        0:w,
    ]

    cx = (w - 1) / 2
    cy = (h - 1) / 2

    dist = np.sqrt(
        ((xx - cx) / max(w, 1)) ** 2
        +
        ((yy - cy) / max(h, 1)) ** 2
    )

    center = np.clip(
        1.0 - dist * 1.4,
        0,
        1,
    )

    saliency = (
        local_contrast * 0.30
        +
        edge * 0.25
        +
        saturation * 0.15
        +
        brightness_dist * 0.20
        +
        center * 0.10
    )

    saliency -= saliency.min()

    saliency /= (
        saliency.max()
        + 1e-6
    )

    return saliency.astype(
        np.float32
    )


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

    subjects = []

    excluded = {
        "background",
    }

    priors = {
        "person": 1.15,

        "cat": 1.05,
        "dog": 1.05,
        "horse": 1.05,
        "cow": 1.05,
        "sheep": 1.05,
        "bird": 1.05,

        "car": 1.0,
        "bus": 1.0,
        "train": 1.0,
        "boat": 1.0,
        "bicycle": 1.0,
        "motorbike": 1.0,

        "pottedplant": 0.9,
        "bottle": 0.85,
    }

    for class_id, class_name in enumerate(
        SemanticSegmenter.VOC_CLASSES
    ):

        if class_name in excluded:
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

        center = 1.0 - min(
            1.0,
            math.sqrt(
                (cx - 0.5) ** 2
                +
                (cy - 0.5) ** 2
            ) * 1.5
        )

        mean_conf = float(
            np.mean(
                confidence[mask]
            )
        )

        mean_saliency = float(
            np.mean(
                saliency[mask]
            )
        )

        local_lum = luminance(rgb)

        region_lum = (
            local_lum[mask]
        )

        local_contrast = float(
            np.std(region_lum)
        )

        hsv = cv2.cvtColor(
            np.clip(
                rgb * 255,
                0,
                255,
            ).astype(np.uint8),
            cv2.COLOR_RGB2HSV,
        )

        colorfulness = float(
            np.mean(
                hsv[..., 1][mask]
            ) / 255.0
        )

        score = (
            mean_conf * 0.35
            +
            math.sqrt(area) * 0.20
            +
            center * 0.15
            +
            mean_saliency * 0.15
            +
            min(local_contrast * 5, 1)
            * 0.10
            +
            colorfulness * 0.05
        )

        score *= priors.get(
            class_name,
            1.0,
        )

        subjects.append(
            SubjectCandidate(
                class_name=class_name,
                confidence=mean_conf,
                area=area,
                center=(cx, cy),
                saliency=mean_saliency,
                local_contrast=local_contrast,
                colorfulness=colorfulness,
                score=score,
            )
        )

    subjects.sort(
        key=lambda x: x.score,
        reverse=True,
    )

    return subjects[:10]


# ============================================================
# Region masks
# ============================================================

def make_region_masks(
    rgb,
    labels,
):

    h, w = labels.shape

    masks = {}

    person = (
        labels ==
        SemanticSegmenter.VOC_CLASSES.index(
            "person"
        )
    )

    animal_ids = [
        SemanticSegmenter.VOC_CLASSES.index(x)
        for x in [
            "cat",
            "dog",
            "horse",
            "cow",
            "sheep",
            "bird",
        ]
    ]

    vehicle_ids = [
        SemanticSegmenter.VOC_CLASSES.index(x)
        for x in [
            "car",
            "bus",
            "train",
            "boat",
            "bicycle",
            "motorbike",
        ]
    ]

    plant = (
        labels ==
        SemanticSegmenter.VOC_CLASSES.index(
            "pottedplant"
        )
    )

    animal = np.isin(
        labels,
        animal_ids,
    )

    vehicle = np.isin(
        labels,
        vehicle_ids,
    )

    subject = (
        person
        |
        animal
        |
        vehicle
        |
        plant
    )

    hsv = cv2.cvtColor(
        np.clip(
            rgb * 255,
            0,
            255,
        ).astype(np.uint8),
        cv2.COLOR_RGB2HSV,
    )

    H = hsv[..., 0]
    S = hsv[..., 1]
    V = hsv[..., 2]

    skin = (
        person
        &
        (H >= 0)
        &
        (H <= 25)
        &
        (S >= 30)
        &
        (S <= 190)
        &
        (V >= 60)
    )

    green = (
        (H >= 30)
        &
        (H <= 95)
        &
        (S >= 45)
        &
        (V >= 30)
    )

    blue = (
        (H >= 80)
        &
        (H <= 135)
        &
        (S >= 40)
        &
        (V >= 40)
    )

    lum = luminance(rgb)

    p75 = np.percentile(
        lum,
        75,
    )

    upper_bright = (
        (
            np.indices(
                (h, w)
            )[0]
            <
            h * 0.45
        )
        &
        (lum > p75)
    )

    lower_half = (
        np.indices(
            (h, w)
        )[0]
        >
        h * 0.50
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

    texture = cv2.magnitude(
        gx,
        gy,
    )

    water = (
        blue
        &
        lower_half
        &
        (
            texture
            <
            np.percentile(
                texture,
                55,
            )
        )
    )

    masks["person"] = person
    masks["animal"] = animal
    masks["vehicle"] = vehicle
    masks["plant"] = plant
    masks["subject"] = subject

    masks["skin"] = skin
    masks["green"] = green
    masks["water"] = water
    masks["upper_bright"] = upper_bright

    return masks


# ============================================================
# Region statistics
# ============================================================

def calculate_region_stats(
    rgb,
    masks,
):

    hsv = cv2.cvtColor(
        np.clip(
            rgb * 255,
            0,
            255,
        ).astype(np.uint8),
        cv2.COLOR_RGB2HSV,
    )

    lum = luminance(rgb)

    result = []

    for name, mask in masks.items():

        area = float(
            np.mean(mask)
        )

        if area <= 0:
            continue

        mean_lum = float(
            np.mean(
                lum[mask]
            )
        )

        mean_sat = float(
            np.mean(
                hsv[..., 1][mask]
            ) / 255.0
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

        if s.class_name in {
            "car",
            "bus",
            "train",
            "boat",
            "bicycle",
            "motorbike",
        }:
            vehicle_area += s.area

    if (
        person_area > 0.015
        and stats.median > 0.08
    ):
        return SceneResult(
            "portrait",
            0.85,
        )

    if (
        shooting.low_light
        and stats.median < 0.10
    ):
        return SceneResult(
            "night",
            0.85,
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
        and stats.dynamic_range > 5
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
        0.05,
        1.02,
        0.97,

        0.42,
        0.08,

        0.08,
        1.02,
        0.035,

        0.55,
        0.75,

        0.94,
        1.00,
        0.06,
        0.12,

        0.55,
    ),

    "night": DevelopParams(
        0.00,
        1.05,
        1.03,

        0.55,
        0.02,

        0.05,
        1.03,
        0.015,

        0.85,
        0.45,

        0.94,
        1.00,
        0.10,
        0.18,

        0.45,
    ),

    "sunset": DevelopParams(
        -0.05,
        1.06,
        1.08,

        0.55,
        0.04,

        0.05,
        1.03,
        0.015,

        0.30,
        0.80,

        0.96,
        1.02,
        0.10,
        0.20,

        0.60,
    ),

    "landscape": DevelopParams(
        0.03,
        1.08,
        1.04,

        0.40,
        0.08,

        0.06,
        1.03,
        0.015,

        0.30,
        0.85,

        0.96,
        1.03,
        0.10,
        0.18,

        0.60,
    ),

    "city": DevelopParams(
        0.02,
        1.07,
        1.02,

        0.45,
        0.05,

        0.06,
        1.03,
        0.02,

        0.40,
        0.80,

        0.95,
        1.01,
        0.08,
        0.16,

        0.58,
    ),

    "indoor": DevelopParams(
        0.04,
        1.03,
        0.99,

        0.40,
        0.08,

        0.05,
        1.02,
        0.015,

        0.50,
        0.65,

        0.94,
        1.00,
        0.05,
        0.10,

        0.50,
    ),

    "general": DevelopParams(
        0.00,
        1.04,
        1.00,

        0.35,
        0.06,

        0.04,
        1.02,
        0.015,

        0.30,
        0.75,

        0.96,
        1.01,
        0.07,
        0.12,

        0.52,
    ),
}


# ============================================================
# Auto exposure
# ============================================================

TARGET_MEDIAN = {
    "portrait": 0.18,
    "night": 0.08,
    "sunset": 0.14,
    "landscape": 0.20,
    "city": 0.18,
    "indoor": 0.17,
    "general": 0.18,
}


def estimate_exposure(
    stats,
    scene,
):

    target = TARGET_MEDIAN.get(
        scene,
        0.18,
    )

    median = max(
        stats.median,
        1e-4,
    )

    ev_mid = math.log2(
        target / median
    )

    ev_high = math.log2(
        0.72 /
        max(stats.p95, 1e-4)
    )

    ev = (
        ev_mid * 0.60
        +
        ev_high * 0.40
    )

    return clamp(
        ev,
        -0.75,
        0.75,
    )


# ============================================================
# Tone mapping
# ============================================================

def tone_map_luminance(
    linear_rgb,
    exposure_ev,
    contrast,
    highlight_protection,
    shadow_lift,
    tone_strength,
):

    rgb = np.clip(
        linear_rgb,
        0,
        None,
    )

    lum = luminance(rgb)

    exposure = 2.0 ** exposure_ev

    rgb = rgb * exposure

    lum = luminance(rgb)

    # --------------------------------------------------------
    # Shadow lift
    # --------------------------------------------------------

    shadow_mask = np.clip(
        1.0 -
        lum / 0.25,
        0,
        1,
    )

    lift = (
        shadow_mask
        *
        shadow_lift
        *
        0.12
    )

    lum = lum + lift

    # --------------------------------------------------------
    # Highlight rolloff
    # --------------------------------------------------------

    high_mask = np.clip(
        (lum - 0.55) / 0.45,
        0,
        1,
    )

    roll = (
        high_mask
        *
        highlight_protection
        *
        0.20
    )

    lum = lum - roll

    # --------------------------------------------------------
    # Contrast
    # --------------------------------------------------------

    pivot = 0.18

    lum = (
        (lum - pivot)
        * contrast
        +
        pivot
    )

    lum = np.clip(
        lum,
        0,
        1,
    )

    # --------------------------------------------------------
    # Mild S curve
    # --------------------------------------------------------

    if tone_strength > 0:

        x = lum

        s_curve = (
            x
            +
            tone_strength
            *
            (
                x * (1 - x)
                *
                (x - 0.5)
                *
                0.45
            )
        )

        lum = np.clip(
            s_curve,
            0,
            1,
        )

    old_lum = luminance(
        np.maximum(rgb, 1e-8)
    )

    scale = (
        lum /
        np.maximum(
            old_lum,
            1e-6,
        )
    )

    result = rgb * scale[..., None]

    return np.clip(
        result,
        0,
        1,
    )


# ============================================================
# Region adjustments
# ============================================================

def apply_region_adjustments(
    linear_rgb,
    masks,
    params,
):

    rgb = linear_rgb.copy()

    lum = luminance(rgb)

    # --------------------------------------------------------
    # Subject
    # --------------------------------------------------------

    subject = masks.get(
        "subject"
    )

    if subject is not None and np.any(subject):

        factor = (
            2.0 **
            params.subject_exposure
        )

        rgb[subject] *= factor

        subject_lum = luminance(
            rgb[subject]
        )

        mean_subject = float(
            np.mean(subject_lum)
        )

        adjusted = (
            (subject_lum - mean_subject)
            *
            params.subject_contrast
            +
            mean_subject
        )

        ratio = (
            adjusted /
            np.maximum(
                subject_lum,
                1e-6,
            )
        )

        rgb[subject] *= (
            ratio[:, None]
        )

    # --------------------------------------------------------
    # Skin saturation
    # --------------------------------------------------------

    skin = masks.get(
        "skin"
    )

    if skin is not None and np.any(skin):

        local = rgb[skin]

        l = (
            0.2126 * local[:, 0]
            +
            0.7152 * local[:, 1]
            +
            0.0722 * local[:, 2]
        )

        local = (
            l[:, None]
            +
            (
                local
                -
                l[:, None]
            )
            *
            params.region_skin_saturation
        )

        rgb[skin] = np.clip(
            local,
            0,
            1,
        )

    # --------------------------------------------------------
    # Green
    # --------------------------------------------------------

    green = masks.get(
        "green"
    )

    if green is not None and np.any(green):

        local = rgb[green]

        l = (
            0.2126 * local[:, 0]
            +
            0.7152 * local[:, 1]
            +
            0.0722 * local[:, 2]
        )

        local = (
            l[:, None]
            +
            (
                local
                -
                l[:, None]
            )
            *
            params.region_green_saturation
        )

        rgb[green] = np.clip(
            local,
            0,
            1,
        )

    # --------------------------------------------------------
    # Water
    # --------------------------------------------------------

    water = masks.get(
        "water"
    )

    if water is not None and np.any(water):

        local = rgb[water]

        l = (
            0.2126 * local[:, 0]
            +
            0.7152 * local[:, 1]
            +
            0.0722 * local[:, 2]
        )

        local = (
            l[:, None]
            +
            (
                local
                -
                l[:, None]
            )
            * 1.02
        )

        rgb[water] = np.clip(
            local,
            0,
            1,
        )

        wl = luminance(
            rgb[water]
        )

        protection = (
            np.clip(
                (wl - 0.65) / 0.35,
                0,
                1,
            )
            *
            params.region_water_highlight
        )

        rgb[water] *= (
            1.0 - protection[:, None]
        )

    # --------------------------------------------------------
    # Upper bright region
    # --------------------------------------------------------

    upper = masks.get(
        "upper_bright"
    )

    if upper is not None and np.any(upper):

        ul = luminance(
            rgb[upper]
        )

        protection = (
            np.clip(
                (ul - 0.60) / 0.40,
                0,
                1,
            )
            *
            params.region_upper_highlight
        )

        rgb[upper] *= (
            1.0 -
            protection[:, None] *
            0.20
        )

    # --------------------------------------------------------
    # Background suppression
    # --------------------------------------------------------

    if subject is not None and np.any(subject):

        background = ~subject

        if np.any(background):

            local = rgb[background]

            l = (
                0.2126 * local[:, 0]
                +
                0.7152 * local[:, 1]
                +
                0.0722 * local[:, 2]
            )

            local = (
                l[:, None]
                +
                (
                    local
                    -
                    l[:, None]
                )
                *
                (
                    1.0
                    -
                    params.background_suppression
                )
            )

            rgb[background] = np.clip(
                local,
                0,
                1,
            )

    return np.clip(
        rgb,
        0,
        1,
    )


# ============================================================
# Global saturation
# ============================================================

def apply_saturation(
    linear_rgb,
    saturation,
):

    rgb = np.clip(
        linear_rgb,
        0,
        1,
    )

    lum = luminance(rgb)

    result = (
        lum[..., None]
        +
        (
            rgb
            -
            lum[..., None]
        )
        *
        saturation
    )

    return np.clip(
        result,
        0,
        1,
    )


# ============================================================
# Denoise
# ============================================================

def apply_denoise(
    linear_rgb,
    strength,
):

    if strength <= 0:
        return linear_rgb

    srgb = linear_to_srgb(
        linear_rgb
    )

    img8 = np.clip(
        srgb * 255,
        0,
        255,
    ).astype(np.uint8)

    sigma_color = (
        10
        +
        strength * 30
    )

    sigma_space = (
        2
        +
        strength * 2
    )

    filtered = cv2.bilateralFilter(
        img8,
        d=7,
        sigmaColor=sigma_color,
        sigmaSpace=sigma_space,
    )

    filtered = (
        filtered.astype(
            np.float32
        )
        / 255.0
    )

    return srgb_to_linear(
        filtered
    )


# ============================================================
# Sharpen
# ============================================================

def apply_sharpen(
    linear_rgb,
    strength,
):

    if strength <= 0:
        return linear_rgb

    srgb = linear_to_srgb(
        linear_rgb
    )

    lum = luminance(
        srgb
    )

    blur = cv2.GaussianBlur(
        lum,
        (0, 0),
        1.0,
    )

    amount = (
        0.35
        *
        strength
    )

    sharp_lum = (
        lum
        +
        amount
        *
        (
            lum
            -
            blur
        )
    )

    ratio = (
        sharp_lum /
        np.maximum(
            lum,
            1e-5,
        )
    )

    result = (
        srgb
        *
        ratio[..., None]
    )

    result = np.clip(
        result,
        0,
        1,
    )

    return srgb_to_linear(
        result
    )


# ============================================================
# Render
# ============================================================

def render_image(
    linear_rgb,
    masks,
    params,
):

    result = tone_map_luminance(
        linear_rgb,

        params.exposure_ev,
        params.contrast,

        params.highlight_protection,
        params.shadow_lift,

        params.tone_strength,
    )

    result = apply_region_adjustments(
        result,
        masks,
        params,
    )

    result = apply_saturation(
        result,
        params.saturation,
    )

    result = apply_denoise(
        result,
        params.denoise,
    )

    result = apply_sharpen(
        result,
        params.sharpen,
    )

    return np.clip(
        result,
        0,
        1,
    )


# ============================================================
# Automatic parameter search
# ============================================================

def evaluate_render(
    linear_rgb,
    scene,
    params,
):

    small_h = min(
        512,
        linear_rgb.shape[0]
    )

    scale = (
        small_h /
        linear_rgb.shape[0]
    )

    small = cv2.resize(
        linear_rgb,
        (
            int(
                linear_rgb.shape[1]
                *
                scale
            ),
            small_h,
        ),
        interpolation=cv2.INTER_AREA,
    )

    masks = {
        "subject": np.zeros(
            small.shape[:2],
            dtype=bool
        ),
        "skin": np.zeros(
            small.shape[:2],
            dtype=bool
        ),
        "green": np.zeros(
            small.shape[:2],
            dtype=bool
        ),
        "water": np.zeros(
            small.shape[:2],
            dtype=bool
        ),
        "upper_bright": np.zeros(
            small.shape[:2],
            dtype=bool
        ),
    }

    rendered = render_image(
        small,
        masks,
        params,
    )

    stats = analyze_image(
        rendered
    )

    target = TARGET_MEDIAN.get(
        scene,
        0.18,
    )

    mid_error = abs(
        stats.median - target
    )

    highlight_penalty = (
        stats.highlight_ratio
        * 4.0
    )

    shadow_penalty = (
        stats.shadow_ratio
        * (
            0.25
            if scene == "night"
            else 1.0
        )
    )

    contrast_target = {
        "portrait": 0.16,
        "night": 0.12,
        "sunset": 0.18,
        "landscape": 0.20,
        "city": 0.18,
        "indoor": 0.15,
        "general": 0.17,
    }.get(
        scene,
        0.17,
    )

    contrast_error = abs(
        stats.contrast
        -
        contrast_target
    )

    oversaturation = max(
        0.0,
        stats.saturation_ratio
        -
        0.08,
    )

    score = (
        mid_error * 0.65
        +
        highlight_penalty
        +
        shadow_penalty
        +
        contrast_error * 0.50
        +
        oversaturation * 0.80
    )

    return float(score)


def search_parameters(
    linear_rgb,
    base_params,
    scene,
):

    exposure_candidates = [
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

    best_params = None
    best_score = float("inf")

    for ev_offset in exposure_candidates:

        for contrast in contrast_candidates:

            for saturation in saturation_candidates:

                params = DevelopParams(
                    exposure_ev=clamp(
                        base_params.exposure_ev
                        +
                        ev_offset,
                        -1.0,
                        1.0,
                    ),

                    contrast=contrast,

                    saturation=(
                        base_params.saturation
                        *
                        saturation
                    ),

                    highlight_protection=(
                        base_params.highlight_protection
                    ),

                    shadow_lift=(
                        base_params.shadow_lift
                    ),

                    subject_exposure=(
                        base_params.subject_exposure
                    ),

                    subject_contrast=(
                        base_params.subject_contrast
                    ),

                    background_suppression=(
                        base_params.background_suppression
                    ),

                    denoise=(
                        base_params.denoise
                    ),

                    sharpen=(
                        base_params.sharpen
                    ),

                    region_skin_saturation=(
                        base_params.region_skin_saturation
                    ),

                    region_green_saturation=(
                        base_params.region_green_saturation
                    ),

                    region_water_highlight=(
                        base_params.region_water_highlight
                    ),

                    region_upper_highlight=(
                        base_params.region_upper_highlight
                    ),

                    tone_strength=(
                        base_params.tone_strength
                    ),
                )

                score = evaluate_render(
                    linear_rgb,
                    scene,
                    params,
                )

                if score < best_score:

                    best_score = score
                    best_params = params

    return best_params, best_score


# ============================================================
# Debug output
# ============================================================

def save_debug(
    debug_dir,
    stem,
    labels,
    saliency,
    masks,
):

    debug_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    label_img = (
        labels.astype(
            np.uint8
        )
    )

    cv2.imwrite(
        str(
            debug_dir /
            f"{stem}_segmentation.png"
        ),
        label_img,
    )

    saliency_img = np.clip(
        saliency * 255,
        0,
        255,
    ).astype(np.uint8)

    cv2.imwrite(
        str(
            debug_dir /
            f"{stem}_saliency.png"
        ),
        saliency_img,
    )

    for name, mask in masks.items():

        img = (
            mask.astype(
                np.uint8
            )
            *
            255
        )

        cv2.imwrite(
            str(
                debug_dir /
                f"{stem}_{name}.png"
            ),
            img,
        )


# ============================================================
# Main developer
# ============================================================

class AutoDeveloper:

    def __init__(
        self,
        device="auto",
        debug=False,
    ):

        self.debug = debug

        if device == "auto":

            device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        if (
            device == "cuda"
            and
            not torch.cuda.is_available()
        ):

            print(
                "[WARN] CUDA unavailable. "
                "Using CPU."
            )

            device = "cpu"

        self.device = device

        print(
            f"[INFO] v{VERSION}"
        )

        print(
            f"[INFO] Device: "
            f"{self.device}"
        )

        self.segmenter = (
            SemanticSegmenter(
                self.device
            )
        )

    def process_file(
        self,
        input_path,
        output_path,
        debug_dir=None,
    ):

        print(
            f"[INFO] Processing: "
            f"{input_path}"
        )

        metadata = get_metadata(
            input_path
        )

        with rawpy.imread(
            str(input_path)
        ) as raw:

            camera = make_camera_profile(
                raw,
                metadata,
            )

            # ------------------------------------------------
            # v18: LibRaw sRGB
            # ------------------------------------------------

            try:

                rgb16 = raw_to_srgb(
                    raw
                )

                conversion_method = (
                    "LibRaw_sRGB_cameraWB"
                )

            except Exception as e:

                print(
                    "[WARN] Primary RAW "
                    f"conversion failed: {e}"
                )

                rgb16 = raw_to_srgb_fallback(
                    raw
                )

                conversion_method = (
                    "LibRaw_sRGB_fallback"
                )

        # ----------------------------------------------------
        # RGB
        # ----------------------------------------------------

        srgb = srgb16_to_float(
            rgb16
        )

        srgb = np.clip(
            srgb,
            0,
            1,
        )

        # ----------------------------------------------------
        # For analysis and development:
        # sRGB -> linear
        # ----------------------------------------------------

        linear_rgb = srgb_to_linear(
            srgb
        )

        stats = analyze_image(
            srgb
        )

        shooting = analyze_shooting(
            metadata
        )

        # ----------------------------------------------------
        # Semantic segmentation
        # ----------------------------------------------------

        labels, confidence = (
            self.segmenter.predict(
                srgb
            )
        )

        # ----------------------------------------------------
        # Saliency
        # ----------------------------------------------------

        saliency = compute_saliency(
            srgb
        )

        # ----------------------------------------------------
        # Subject ranking
        # ----------------------------------------------------

        subjects = rank_subjects(
            srgb,
            labels,
            confidence,
            saliency,
        )

        # ----------------------------------------------------
        # Regions
        # ----------------------------------------------------

        masks = make_region_masks(
            srgb,
            labels,
        )

        region_stats = (
            calculate_region_stats(
                srgb,
                masks,
            )
        )

        # ----------------------------------------------------
        # Scene
        # ----------------------------------------------------

        scene_result = classify_scene(
            stats,
            shooting,
            subjects,
        )

        print(
            f"[INFO] Scene: "
            f"{scene_result.scene} "
            f"({scene_result.confidence:.2f})"
        )

        # ----------------------------------------------------
        # Base profile
        # ----------------------------------------------------

        base_params = (
            SCENE_PROFILES[
                scene_result.scene
            ]
        )

        # ----------------------------------------------------
        # Automatic exposure
        # ----------------------------------------------------

        auto_ev = estimate_exposure(
            stats,
            scene_result.scene,
        )

        base_params = DevelopParams(
            exposure_ev=(
                base_params.exposure_ev
                +
                auto_ev
            ),

            contrast=(
                base_params.contrast
            ),

            saturation=(
                base_params.saturation
            ),

            highlight_protection=(
                base_params.highlight_protection
            ),

            shadow_lift=(
                base_params.shadow_lift
            ),

            subject_exposure=(
                base_params.subject_exposure
            ),

            subject_contrast=(
                base_params.subject_contrast
            ),

            background_suppression=(
                base_params.background_suppression
            ),

            denoise=(
                base_params.denoise
                *
                (
                    0.7
                    +
                    shooting.estimated_noise
                    *
                    0.6
                )
            ),

            sharpen=(
                base_params.sharpen
                *
                (
                    1.0
                    -
                    shooting.estimated_noise
                    *
                    0.25
                )
            ),

            region_skin_saturation=(
                base_params.region_skin_saturation
            ),

            region_green_saturation=(
                base_params.region_green_saturation
            ),

            region_water_highlight=(
                base_params.region_water_highlight
            ),

            region_upper_highlight=(
                base_params.region_upper_highlight
            ),

            tone_strength=(
                base_params.tone_strength
            ),
        )

        # ----------------------------------------------------
        # Search
        # ----------------------------------------------------

        best_params, search_score = (
            search_parameters(
                linear_rgb,
                base_params,
                scene_result.scene,
            )
        )

        # ----------------------------------------------------
        # Final render
        # ----------------------------------------------------

        rendered_linear = render_image(
            linear_rgb,
            masks,
            best_params,
        )

        rendered_srgb = linear_to_srgb(
            rendered_linear
        )

        rendered_srgb = np.clip(
            rendered_srgb,
            0,
            1,
        )

        output_rgb = (
            rendered_srgb * 255
        ).astype(
            np.uint8
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        Image.fromarray(
            output_rgb,
            mode="RGB",
        ).save(
            output_path,
            quality=95,
            subsampling=0,
        )

        # ----------------------------------------------------
        # Debug
        # ----------------------------------------------------

        if self.debug and debug_dir:

            save_debug(
                debug_dir,
                input_path.stem,
                labels,
                saliency,
                masks,
            )

        # ----------------------------------------------------
        # Analysis report
        # ----------------------------------------------------

        report = {

            "version": VERSION,

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
                camera
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
                scene_result
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
                best_params
            ),

            "search_score": search_score,
        }

        json_path = (
            output_path.with_suffix(
                ".json"
            )
        )

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
            f"[INFO] Saved: "
            f"{output_path}"
        )

        print(
            f"[INFO] JSON: "
            f"{json_path}"
        )

        return report


# ============================================================
# RAW collection
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
            and
            path.suffix.lower()
            in RAW_EXTENSIONS
        ):
            files.append(path)

    return sorted(files)


# ============================================================
# CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Automatic RAW Developer v18"
        )
    )

    parser.add_argument(
        "input",
        help="RAW file or directory",
    )

    parser.add_argument(
        "-o",
        "--output",
        default="output",
        help="Output directory",
    )

    parser.add_argument(
        "--device",
        default="auto",
        choices=[
            "auto",
            "cpu",
            "cuda",
        ],
        help="Inference device",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Save segmentation/saliency/region masks",
    )

    args = parser.parse_args()

    input_path = Path(
        args.input
    )

    output_dir = Path(
        args.output
    )

    files = collect_raw_files(
        input_path
    )

    if not files:

        print(
            "[ERROR] No RAW files found."
        )

        return 1

    print(
        f"[INFO] RAW files: "
        f"{len(files)}"
    )

    developer = AutoDeveloper(
        device=args.device,
        debug=args.debug,
    )

    success = 0
    failed = 0

    for raw_path in files:

        try:

            if input_path.is_file():

                relative = (
                    raw_path.name
                )

            else:

                try:
                    relative = (
                        raw_path.relative_to(
                            input_path
                        )
                    )
                except ValueError:
                    relative = raw_path.name

            relative = Path(
                relative
            )

            output_path = (
                output_dir
                /
                relative.with_suffix(
                    ".jpg"
                )
            )

            debug_dir = (
                output_dir
                /
                "debug"
            )

            developer.process_file(
                raw_path,
                output_path,
                debug_dir,
            )

            success += 1

        except Exception as e:

            failed += 1

            print(
                f"[ERROR] Failed: "
                f"{raw_path}"
            )

            print(
                f"        {type(e).__name__}: "
                f"{e}"
            )

    print()
    print(
        "[INFO] Finished"
    )

    print(
        f"[INFO] Success: "
        f"{success}"
    )

    print(
        f"[INFO] Failed: "
        f"{failed}"
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
