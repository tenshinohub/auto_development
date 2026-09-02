#!/usr/bin/env python3

import argparse
import logging
import math
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

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

SUBJECT_CLASSES = {
    "aeroplane",
    "bicycle",
    "bird",
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

PERSON_CLASS = "person"
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
class CameraProfile:
    make: str
    model: str
    iso: int

    black_level: np.ndarray
    white_level: float
    white_level_per_channel: Optional[np.ndarray]

    camera_wb: np.ndarray

    color_matrix: np.ndarray
    rgb_xyz_matrix: np.ndarray

    color_desc: str
    num_colors: int

    raw_width: int
    raw_height: int

    raw_pattern: Optional[np.ndarray]

    lens_make: str = ""
    lens_model: str = ""

    metadata_source: str = "rawpy"


@dataclass
class ImageAnalysis:
    mean_luma: float
    median_luma: float

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


@dataclass
class SubjectCandidate:
    class_name: str
    confidence: float
    mask: np.ndarray

    area_ratio: float
    center_score: float
    saliency_score: float
    local_contrast: float
    colorfulness: float

    importance: float


@dataclass
class SceneProfile:
    name: str

    exposure: float
    contrast: float
    saturation: float

    highlight: float
    shadow: float

    subject: float
    background_suppression: float

    denoise: float
    sharpen: float


# ============================================================
# Logging
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)

logger = logging.getLogger("auto_develop")


# ============================================================
# Utility
# ============================================================

def normalize_map(x: np.ndarray) -> np.ndarray:
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    mn = float(x.min())
    mx = float(x.max())

    if mx - mn < 1e-8:
        return np.zeros_like(x, dtype=np.float32)

    return ((x - mn) / (mx - mn)).astype(np.float32)


def luminance(img: np.ndarray) -> np.ndarray:
    return (
        img[..., 0] * 0.2126
        + img[..., 1] * 0.7152
        + img[..., 2] * 0.0722
    )


def create_center_weight(h: int, w: int) -> np.ndarray:
    y, x = np.mgrid[0:h, 0:w]

    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0

    dx = (x - cx) / max(w, 1)
    dy = (y - cy) / max(h, 1)

    distance = np.sqrt(dx * dx + dy * dy)

    weight = np.exp(-distance * 4.0)

    return normalize_map(weight)


def resize_for_analysis(
    img: np.ndarray,
    max_size: int = 768,
):
    h, w = img.shape[:2]

    scale = min(1.0, max_size / max(h, w))

    if scale == 1.0:
        return img, 1.0

    nw = max(1, int(w * scale))
    nh = max(1, int(h * scale))

    resized = cv2.resize(
        img,
        (nw, nh),
        interpolation=cv2.INTER_AREA,
    )

    return resized, scale


# ============================================================
# EXIF Camera Information
# ============================================================

def read_exif_camera_info(filename: Path):
    """
    Read Make / Model from EXIF using Pillow.

    Returns:
        make, model
    """

    make = ""
    model = ""

    try:
        with Image.open(filename) as im:
            exif = im.getexif()

            if not exif:
                return make, model

            tag_map = {
                ExifTags.TAGS.get(k, k): v
                for k, v in exif.items()
            }

            make = str(tag_map.get("Make", "")).strip()
            model = str(tag_map.get("Model", "")).strip()

    except Exception:
        pass

    return make, model


def read_exiftool_camera_info(filename: Path):
    """
    Optional exiftool fallback.

    exiftool is not required, but if installed it can recover
    camera metadata that Pillow/rawpy cannot expose.
    """

    if shutil.which("exiftool") is None:
        return "", ""

    make = ""
    model = ""

    try:
        result = subprocess.run(
            [
                "exiftool",
                "-s3",
                "-Make",
                "-Model",
                str(filename),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
            check=False,
        )

        lines = [
            line.strip()
            for line in result.stdout.splitlines()
            if line.strip()
        ]

        if len(lines) >= 1:
            make = lines[0]

        if len(lines) >= 2:
            model = lines[1]

    except Exception:
        pass

    return make, model


# ============================================================
# Camera Model Identification
# ============================================================

def get_rawpy_camera_info(raw):
    """
    Try to retrieve camera make/model from rawpy.

    Some rawpy/LibRaw combinations expose these as attributes,
    while others do not.
    """

    make = ""
    model = ""

    try:
        value = getattr(raw, "camera_make", "")
        if value is not None:
            make = str(value).strip()
    except Exception:
        pass

    try:
        value = getattr(raw, "camera_model", "")
        if value is not None:
            model = str(value).strip()
    except Exception:
        pass

    return make, model


def normalize_make_model(make: str, model: str):
    make = make.strip()
    model = model.strip()

    if not make and model:
        upper = model.upper()

        manufacturers = [
            "CANON",
            "NIKON",
            "SONY",
            "FUJIFILM",
            "FUJI",
            "PANASONIC",
            "OLYMPUS",
            "OM SYSTEM",
            "LEICA",
            "PENTAX",
            "RICOH",
            "SIGMA",
        ]

        for m in manufacturers:
            if upper.startswith(m):
                make = m
                break

    return make, model


def get_camera_identity(raw, filename: Path):
    """
    Multi-stage camera identification.

    Priority:
        1. rawpy / LibRaw
        2. Pillow EXIF
        3. exiftool
    """

    make, model = get_rawpy_camera_info(raw)

    source = "rawpy"

    if not make or not model:
        exif_make, exif_model = read_exif_camera_info(filename)

        if not make:
            make = exif_make

        if not model:
            model = exif_model

        if exif_make or exif_model:
            source = "Pillow EXIF"

    if not make or not model:
        tool_make, tool_model = read_exiftool_camera_info(filename)

        if not make:
            make = tool_make

        if not model:
            model = tool_model

        if tool_make or tool_model:
            source = "exiftool"

    make, model = normalize_make_model(make, model)

    return make, model, source


# ============================================================
# Camera Profile
# ============================================================

def safe_float_array(value, default):
    try:
        arr = np.asarray(value, dtype=np.float32)

        if arr.size == 0:
            return np.asarray(default, dtype=np.float32)

        return arr.copy()

    except Exception:
        return np.asarray(default, dtype=np.float32)


def load_raw(filename: Path):
    """
    Load RAW and create CameraProfile.

    Important:
    output_color=raw is intentional.

    We obtain camera-space RGB first, then use the camera's
    RGB->XYZ matrix ourselves.
    """

    raw = rawpy.imread(str(filename))

    make, model, metadata_source = get_camera_identity(
        raw,
        filename,
    )

    try:
        iso = int(round(float(raw.other.iso_speed)))
    except Exception:
        iso = 100

    if iso <= 0:
        iso = 100

    black_level = safe_float_array(
        raw.black_level_per_channel,
        [0, 0, 0, 0],
    )

    white_level = float(raw.white_level)

    try:
        white_level_per_channel = raw.camera_white_level_per_channel

        if white_level_per_channel is not None:
            white_level_per_channel = safe_float_array(
                white_level_per_channel,
                [white_level] * 4,
            )
    except Exception:
        white_level_per_channel = None

    camera_wb = safe_float_array(
        raw.camera_whitebalance,
        [1, 1, 1, 1],
    )

    color_matrix = safe_float_array(
        raw.color_matrix,
        np.zeros((3, 4), dtype=np.float32),
    )

    rgb_xyz_matrix = safe_float_array(
        raw.rgb_xyz_matrix,
        np.zeros((4, 3), dtype=np.float32),
    )

    try:
        color_desc = raw.color_desc.decode(
            "ascii",
            errors="ignore",
        )
    except Exception:
        color_desc = ""

    try:
        num_colors = int(raw.num_colors)
    except Exception:
        num_colors = 3

    try:
        raw_width = int(raw.sizes.raw_width)
        raw_height = int(raw.sizes.raw_height)
    except Exception:
        raw_width = 0
        raw_height = 0

    try:
        raw_pattern = raw.raw_pattern.copy()
    except Exception:
        raw_pattern = None

    try:
        lens = raw.lens
        lens_make = str(lens.make).strip()
        lens_model = str(lens.model).strip()
    except Exception:
        lens_make = ""
        lens_model = ""

    profile = CameraProfile(
        make=make,
        model=model,
        iso=iso,

        black_level=black_level,
        white_level=white_level,
        white_level_per_channel=white_level_per_channel,

        camera_wb=camera_wb,

        color_matrix=color_matrix,
        rgb_xyz_matrix=rgb_xyz_matrix,

        color_desc=color_desc,
        num_colors=num_colors,

        raw_width=raw_width,
        raw_height=raw_height,

        raw_pattern=raw_pattern,

        lens_make=lens_make,
        lens_model=lens_model,

        metadata_source=metadata_source,
    )

    return raw, profile


def print_camera_profile(profile: CameraProfile):
    camera_name = (
        f"{profile.make} {profile.model}"
    ).strip()

    if not camera_name:
        camera_name = "UNKNOWN"

    logger.info(
        "RAW camera: %s",
        camera_name,
    )

    logger.info(
        "Camera metadata source: %s",
        profile.metadata_source,
    )

    logger.info(
        "ISO: %d",
        profile.iso,
    )

    logger.info(
        "RAW size: %dx%d",
        profile.raw_width,
        profile.raw_height,
    )

    logger.info(
        "Black level: %s",
        np.array2string(
            profile.black_level,
            precision=1,
        ),
    )

    logger.info(
        "White level: %.1f",
        profile.white_level,
    )

    if profile.white_level_per_channel is not None:
        logger.info(
            "White level/channel: %s",
            np.array2string(
                profile.white_level_per_channel,
                precision=1,
            ),
        )

    logger.info(
        "Camera WB: %s",
        np.array2string(
            profile.camera_wb,
            precision=4,
        ),
    )

    logger.info(
        "Color description: %s",
        profile.color_desc,
    )

    logger.info(
        "Num colors: %d",
        profile.num_colors,
    )

    logger.info(
        "Camera RGB -> XYZ matrix:\n%s",
        np.array2string(
            profile.rgb_xyz_matrix,
            precision=6,
            suppress_small=True,
        ),
    )

    logger.info(
        "Color matrix:\n%s",
        np.array2string(
            profile.color_matrix,
            precision=6,
            suppress_small=True,
        ),
    )

    if profile.lens_model:
        logger.info(
            "Lens: %s %s",
            profile.lens_make,
            profile.lens_model,
        )

    if profile.raw_pattern is not None:
        logger.info(
            "RAW pattern:\n%s",
            profile.raw_pattern,
        )


# ============================================================
# Camera Color Matrix
# ============================================================

# sRGB D65 XYZ -> linear sRGB
XYZ_TO_SRGB = np.array(
    [
        [ 3.2404542, -1.5371385, -0.4985314],
        [-0.9692660,  1.8760108,  0.0415560],
        [ 0.0556434, -0.2040259,  1.0572252],
    ],
    dtype=np.float32,
)


def build_effective_camera_matrix(
    rgb_xyz_matrix: np.ndarray,
    color_desc: str,
) -> np.ndarray:
    """
    Convert LibRaw's 4x3 camera RGB->XYZ matrix into
    an effective 3x3 matrix for demosaiced RGB.

    For RGBG:
        R = channel R
        G = average of the two green channels
        B = channel B
    """

    matrix = np.asarray(
        rgb_xyz_matrix,
        dtype=np.float32,
    )

    if matrix.shape != (4, 3):
        return np.eye(3, dtype=np.float32)

    desc = color_desc.upper()

    if len(desc) < 3:
        desc = "RGB"

    groups = {
        "R": [],
        "G": [],
        "B": [],
    }

    for i, c in enumerate(desc[:4]):
        if c in groups:
            groups[c].append(i)

    effective = np.zeros(
        (3, 3),
        dtype=np.float32,
    )

    for out_idx, color in enumerate(("R", "G", "B")):
        indices = groups[color]

        if not indices:
            # Fallback for unusual cameras.
            if out_idx < matrix.shape[0]:
                effective[out_idx] = matrix[out_idx]
            continue

        effective[out_idx] = np.mean(
            matrix[indices],
            axis=0,
        )

    if not np.any(np.abs(effective) > 1e-8):
        return np.eye(3, dtype=np.float32)

    return effective


def apply_camera_color_matrix(
    camera_rgb: np.ndarray,
    profile: CameraProfile,
) -> np.ndarray:
    """
    Camera RGB -> XYZ -> sRGB.

    This is the important v12 change:
    the camera matrix is actually part of the image pipeline.
    """

    camera_matrix = build_effective_camera_matrix(
        profile.rgb_xyz_matrix,
        profile.color_desc,
    )

    h, w, _ = camera_rgb.shape

    flat = camera_rgb.reshape(-1, 3)

    # Camera RGB -> XYZ
    xyz = flat @ camera_matrix

    # XYZ -> linear sRGB
    srgb_linear = xyz @ XYZ_TO_SRGB.T

    srgb_linear = srgb_linear.reshape(
        h,
        w,
        3,
    )

    return srgb_linear.astype(np.float32)


# ============================================================
# RAW Development
# ============================================================

def load_linear_camera_rgb(
    raw,
    profile: CameraProfile,
):
    """
    Decode RAW to camera RGB.

    We deliberately do NOT ask LibRaw to convert to sRGB.

    Instead:
        RAW
        -> black/white normalization
        -> camera WB
        -> demosaic
        -> camera RGB
        -> our matrix conversion
    """

    rgb16 = raw.postprocess(
        use_camera_wb=True,
        use_auto_wb=False,

        output_color=rawpy.ColorSpace.raw,
        output_bps=16,

        gamma=(1, 1),

        no_auto_bright=True,

        highlight_mode=rawpy.HighlightMode.Blend,

        half_size=False,
        four_color_rgb=False,

        demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD,
    )

    rgb = rgb16.astype(np.float32) / 65535.0

    return rgb


def develop_camera_rgb(
    raw,
    profile: CameraProfile,
):
    camera_rgb = load_linear_camera_rgb(
        raw,
        profile,
    )

    # Camera RGB -> XYZ -> sRGB
    rgb = apply_camera_color_matrix(
        camera_rgb,
        profile,
    )

    # Negative values can occur after matrix conversion.
    rgb = np.maximum(
        rgb,
        0.0,
    )

    return rgb


# ============================================================
# Image Analysis
# ============================================================

def analyze_image(
    img: np.ndarray,
) -> ImageAnalysis:

    luma = luminance(img)

    mean_luma = float(np.mean(luma))
    median_luma = float(np.median(luma))

    p01, p05, p95, p99 = np.percentile(
        luma,
        [1, 5, 95, 99],
    )

    shadow_ratio = float(
        np.mean(luma < 0.05)
    )

    highlight_ratio = float(
        np.mean(luma > 0.95)
    )

    dynamic_range = float(
        p95 / max(p05, 1e-5)
    )

    saturation_ratio = float(
        np.mean(
            np.max(img, axis=2) > 0.98
        )
    )

    gray = (
        np.clip(
            luma * 255.0,
            0,
            255,
        )
        .astype(np.uint8)
    )

    edges = cv2.Canny(
        gray,
        50,
        150,
    )

    edge_density = float(
        np.mean(edges > 0)
    )

    r = img[..., 0]
    g = img[..., 1]
    b = img[..., 2]

    warm_ratio = float(
        np.mean(
            (r > b * 1.12)
            & (r > g * 1.02)
        )
    )

    return ImageAnalysis(
        mean_luma=mean_luma,
        median_luma=median_luma,
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
    )


# ============================================================
# Semantic Segmentation
# ============================================================

class SemanticSegmenter:

    def __init__(
        self,
        device: str = "auto",
        max_size: int = 768,
    ):
        self.max_size = max_size

        if device == "auto":
            self.device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        else:
            self.device = device

        logger.info(
            "Segmentation device: %s",
            self.device,
        )

        weights = (
            DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
        )

        self.model = deeplabv3_mobilenet_v3_large(
            weights=weights,
        )

        self.model.eval()
        self.model.to(self.device)

        self.transforms = weights.transforms()

    @torch.inference_mode()
    def predict(
        self,
        img: np.ndarray,
    ):
        small, _ = resize_for_analysis(
            img,
            self.max_size,
        )

        rgb8 = np.clip(
            small * 255.0,
            0,
            255,
        ).astype(np.uint8)

        tensor = self.transforms(
            torch.from_numpy(rgb8)
            .permute(2, 0, 1)
        )

        tensor = tensor.unsqueeze(0)
        tensor = tensor.to(self.device)

        output = self.model(
            tensor
        )["out"][0]

        probability = torch.softmax(
            output,
            dim=0,
        )

        labels = torch.argmax(
            probability,
            dim=0,
        ).cpu().numpy()

        confidence = (
            probability.max(dim=0)
            .values
            .cpu()
            .numpy()
        )

        return labels, confidence


# ============================================================
# Sky
# ============================================================

def create_sky_mask(
    img: np.ndarray,
) -> np.ndarray:

    h, w = img.shape[:2]

    y = np.arange(h).reshape(-1, 1)

    upper_weight = 1.0 - (
        y / max(h - 1, 1)
    )

    upper_weight = np.repeat(
        upper_weight,
        w,
        axis=1,
    )

    r = img[..., 0]
    g = img[..., 1]
    b = img[..., 2]

    blue = (
        (b > r * 1.05)
        & (b > g * 1.01)
    )

    bright = luminance(img) > 0.35

    mask = (
        blue
        & bright
        & (upper_weight > 0.35)
    )

    mask = mask.astype(np.float32)

    mask = cv2.GaussianBlur(
        mask,
        (0, 0),
        5,
    )

    return normalize_map(mask)


# ============================================================
# Saliency
# ============================================================

def calculate_saliency_map(
    img: np.ndarray,
) -> np.ndarray:

    h, w = img.shape[:2]

    luma = luminance(img)

    local_mean = cv2.GaussianBlur(
        luma,
        (0, 0),
        15,
    )

    local_contrast = np.abs(
        luma - local_mean
    )

    gx = cv2.Sobel(
        luma,
        cv2.CV_32F,
        1,
        0,
        ksize=3,
    )

    gy = cv2.Sobel(
        luma,
        cv2.CV_32F,
        0,
        1,
        ksize=3,
    )

    edge = np.sqrt(
        gx * gx + gy * gy
    )

    max_rgb = np.max(
        img,
        axis=2,
    )

    min_rgb = np.min(
        img,
        axis=2,
    )

    colorfulness = (
        max_rgb - min_rgb
    )

    brightness_distinct = np.abs(
        luma - np.mean(luma)
    )

    center = create_center_weight(
        h,
        w,
    )

    saliency = (
        0.30 * normalize_map(
            local_contrast
        )
        + 0.25 * normalize_map(
            edge
        )
        + 0.15 * normalize_map(
            colorfulness
        )
        + 0.20 * normalize_map(
            brightness_distinct
        )
        + 0.10 * center
    )

    saliency = cv2.GaussianBlur(
        saliency.astype(np.float32),
        (0, 0),
        4,
    )

    return normalize_map(
        saliency
    )


# ============================================================
# Subject Analysis
# ============================================================

def calculate_subject_importance(
    img: np.ndarray,
    mask: np.ndarray,
    class_name: str,
    confidence: float,
    saliency: np.ndarray,
):
    area_ratio = float(
        np.mean(mask > 0.5)
    )

    ys, xs = np.where(
        mask > 0.5
    )

    if len(xs) == 0:
        center_score = 0.0
        local_contrast = 0.0
        colorfulness = 0.0
    else:
        cx = np.mean(xs)
        cy = np.mean(ys)

        h, w = mask.shape

        dx = (
            cx - w / 2
        ) / max(w / 2, 1)

        dy = (
            cy - h / 2
        ) / max(h / 2, 1)

        distance = math.sqrt(
            dx * dx + dy * dy
        )

        center_score = max(
            0.0,
            1.0 - distance,
        )

        luma = luminance(img)

        subject_luma = luma[mask > 0.5]

        surrounding = cv2.GaussianBlur(
            luma,
            (0, 0),
            15,
        )

        surrounding_values = surrounding[
            mask > 0.5
        ]

        local_contrast = float(
            abs(
                np.mean(subject_luma)
                - np.mean(surrounding_values)
            )
        )

        rgb_subject = img[
            mask > 0.5
        ]

        colorfulness = float(
            np.mean(
                np.max(
                    rgb_subject,
                    axis=1,
                )
                -
                np.min(
                    rgb_subject,
                    axis=1,
                )
            )
        )

    if np.any(mask > 0.5):
        saliency_score = float(
            np.mean(
                saliency[mask > 0.5]
            )
        )
    else:
        saliency_score = 0.0

    class_prior = 1.0

    if class_name == PERSON_CLASS:
        class_prior = 1.15

    elif class_name in ANIMAL_CLASSES:
        class_prior = 1.05

    elif class_name in VEHICLE_CLASSES:
        class_prior = 1.00

    elif class_name == "pottedplant":
        class_prior = 0.90

    elif class_name == "bottle":
        class_prior = 0.85

    area_score = min(
        area_ratio / 0.25,
        1.0,
    )

    contrast_score = min(
        local_contrast / 0.20,
        1.0,
    )

    color_score = min(
        colorfulness / 0.30,
        1.0,
    )

    importance = (
        0.20 * area_score
        + 0.20 * confidence
        + 0.15 * center_score
        + 0.20 * saliency_score
        + 0.15 * contrast_score
        + 0.10 * color_score
    )

    importance *= class_prior

    return (
        area_ratio,
        center_score,
        saliency_score,
        contrast_score,
        colorfulness,
        min(importance, 1.5),
    )


def rank_subjects(
    img: np.ndarray,
    labels: np.ndarray,
    confidence: np.ndarray,
    saliency: np.ndarray,
):
    h, w = labels.shape

    candidates = []

    for class_id, class_name in enumerate(
        VOC_CLASSES
    ):

        if class_name not in SUBJECT_CLASSES:
            continue

        mask = (
            (labels == class_id)
            & (confidence > 0.45)
        ).astype(np.float32)

        area = float(
            np.mean(mask > 0.5)
        )

        if area < 0.003:
            continue

        (
            area_ratio,
            center_score,
            saliency_score,
            local_contrast,
            colorfulness,
            importance,
        ) = calculate_subject_importance(
            img,
            mask,
            class_name,
            float(
                np.mean(
                    confidence[
                        labels == class_id
                    ]
                )
            ),
            saliency,
        )

        candidates.append(
            SubjectCandidate(
                class_name=class_name,
                confidence=float(
                    np.mean(
                        confidence[
                            labels == class_id
                        ]
                    )
                ),
                mask=mask,

                area_ratio=area_ratio,
                center_score=center_score,
                saliency_score=saliency_score,
                local_contrast=local_contrast,
                colorfulness=colorfulness,

                importance=importance,
            )
        )

    candidates.sort(
        key=lambda x: x.importance,
        reverse=True,
    )

    return candidates


def create_subject_attention(
    subjects,
    shape,
):
    attention = np.zeros(
        shape,
        dtype=np.float32,
    )

    for subject in subjects:
        weight = min(
            1.0,
            subject.importance,
        )

        attention = np.maximum(
            attention,
            subject.mask * weight,
        )

    return normalize_map(
        attention
    )


def combine_subject_and_saliency(
    subject_attention,
    saliency,
    importance,
):
    subject_weight = (
        0.45
        + 0.25 * min(
            importance,
            1.0,
        )
    )

    saliency_weight = (
        1.0 - subject_weight
    )

    combined = (
        subject_attention
        * subject_weight
        +
        saliency
        * saliency_weight
    )

    return normalize_map(
        combined
    )


# ============================================================
# Scene Classification
# ============================================================

def classify_scene(
    analysis: ImageAnalysis,
    subjects,
    sky_mask: np.ndarray,
):
    person_ratio = 0.0
    vehicle_ratio = 0.0

    for subject in subjects:
        if subject.class_name == "person":
            person_ratio += (
                subject.area_ratio
            )

        if subject.class_name in VEHICLE_CLASSES:
            vehicle_ratio += (
                subject.area_ratio
            )

    sky_ratio = float(
        np.mean(sky_mask > 0.5)
    )

    if (
        person_ratio > 0.015
        and person_ratio < 0.45
    ):
        return "portrait"

    if (
        analysis.mean_luma < 0.20
        and analysis.highlight_ratio < 0.08
    ):
        return "night"

    if (
        sky_ratio > 0.08
        and analysis.warm_ratio > 0.10
    ):
        return "sunset"

    if (
        sky_ratio > 0.15
        and analysis.edge_density < 0.15
    ):
        return "landscape"

    if (
        vehicle_ratio > 0.02
        or analysis.edge_density > 0.18
    ):
        return "city"

    if (
        analysis.mean_luma > 0.15
        and analysis.edge_density > 0.08
        and sky_ratio < 0.03
    ):
        return "indoor"

    return "general"


# ============================================================
# Scene Profiles
# ============================================================

SCENE_PROFILES = {

    "portrait": SceneProfile(
        "portrait",
        0.05,
        1.02,
        0.97,
        0.35,
        0.08,
        0.08,
        0.035,
        0.55,
        0.75,
    ),

    "night": SceneProfile(
        "night",
        0.00,
        1.05,
        1.03,
        0.55,
        0.02,
        0.05,
        0.015,
        0.85,
        0.45,
    ),

    "sunset": SceneProfile(
        "sunset",
        -0.05,
        1.06,
        1.08,
        0.55,
        0.04,
        0.05,
        0.015,
        0.30,
        0.80,
    ),

    "landscape": SceneProfile(
        "landscape",
        0.03,
        1.08,
        1.04,
        0.40,
        0.08,
        0.06,
        0.015,
        0.30,
        0.85,
    ),

    "city": SceneProfile(
        "city",
        0.02,
        1.07,
        1.02,
        0.45,
        0.05,
        0.06,
        0.02,
        0.40,
        0.80,
    ),

    "indoor": SceneProfile(
        "indoor",
        0.04,
        1.03,
        0.99,
        0.40,
        0.08,
        0.05,
        0.015,
        0.50,
        0.65,
    ),

    "general": SceneProfile(
        "general",
        0.00,
        1.04,
        1.00,
        0.35,
        0.06,
        0.04,
        0.015,
        0.30,
        0.75,
    ),
}


# ============================================================
# Camera Adaptive Parameters
# ============================================================

def camera_family(
    profile: CameraProfile,
):
    text = (
        f"{profile.make} "
        f"{profile.model}"
    ).upper()

    if "CANON" in text:
        return "canon"

    if "NIKON" in text:
        return "nikon"

    if (
        "SONY" in text
        or "ILCE" in text
        or "A7" in text
    ):
        return "sony"

    if (
        "FUJI" in text
        or "FUJIFILM" in text
    ):
        return "fujifilm"

    if "PANASONIC" in text:
        return "panasonic"

    if (
        "OLYMPUS" in text
        or "OM SYSTEM" in text
    ):
        return "olympus"

    if "LEICA" in text:
        return "leica"

    if (
        "PENTAX" in text
        or "RICOH" in text
    ):
        return "pentax"

    return "unknown"


def calculate_camera_adjustment(
    profile: CameraProfile,
):
    """
    Camera-specific automatic tuning.

    This does not try to emulate a particular manufacturer's JPEG.
    It only adapts the development according to measurable
    camera characteristics.
    """

    family = camera_family(
        profile
    )

    iso = profile.iso

    # --------------------------------------------------------
    # ISO-dependent noise
    # --------------------------------------------------------

    if iso <= 200:
        denoise = 0.10
        sharpen = 0.95

    elif iso <= 800:
        denoise = 0.25
        sharpen = 0.85

    elif iso <= 3200:
        denoise = 0.45
        sharpen = 0.70

    elif iso <= 12800:
        denoise = 0.70
        sharpen = 0.50

    else:
        denoise = 0.90
        sharpen = 0.35

    # --------------------------------------------------------
    # Dynamic range estimate
    # --------------------------------------------------------

    black = float(
        np.mean(
            profile.black_level
        )
    )

    white = max(
        profile.white_level,
        1.0,
    )

    usable_range = max(
        white - black,
        1.0,
    )

    dynamic_ratio = (
        usable_range / white
    )

    # Less usable range -> more conservative highlights
    highlight_strength = (
        0.35
        + (1.0 - dynamic_ratio) * 0.80
    )

    highlight_strength = np.clip(
        highlight_strength,
        0.25,
        0.85,
    )

    # --------------------------------------------------------
    # Manufacturer family
    # --------------------------------------------------------

    saturation = 1.0
    contrast = 1.0

    if family == "canon":
        saturation *= 0.995

    elif family == "nikon":
        saturation *= 1.005

    elif family == "sony":
        highlight_strength *= 1.05
        contrast *= 0.995

    elif family == "fujifilm":
        saturation *= 1.015

    elif family == "panasonic":
        saturation *= 1.005

    elif family == "olympus":
        saturation *= 1.010

    elif family == "leica":
        contrast *= 1.01

    return {
        "family": family,
        "denoise": float(denoise),
        "sharpen": float(sharpen),
        "highlight": float(
            np.clip(
                highlight_strength,
                0.20,
                0.90,
            )
        ),
        "saturation": float(saturation),
        "contrast": float(contrast),
    }


# ============================================================
# Exposure
# ============================================================

def apply_exposure(
    img,
    ev,
):
    factor = 2.0 ** ev

    return np.maximum(
        img * factor,
        0.0,
    )


def auto_exposure(
    img,
    target=0.42,
):
    luma = luminance(img)

    median = float(
        np.median(luma)
    )

    if median <= 1e-5:
        return 0.0

    ev = math.log2(
        target / median
    )

    return float(
        np.clip(
            ev,
            -1.0,
            1.0,
        )
    )


# ============================================================
# Highlight / Shadow
# ============================================================

def highlight_recovery(
    img,
    strength,
):
    luma = luminance(img)

    mask = np.clip(
        (luma - 0.65) / 0.35,
        0.0,
        1.0,
    )

    factor = (
        1.0
        - mask * strength * 0.25
    )

    out = img * factor[..., None]

    return np.maximum(
        out,
        0.0,
    )


def shadow_lift(
    img,
    strength,
):
    luma = luminance(img)

    mask = np.clip(
        (0.35 - luma) / 0.35,
        0.0,
        1.0,
    )

    lift = (
        mask
        * strength
        * 0.10
    )

    out = img + lift[..., None]

    return np.maximum(
        out,
        0.0,
    )


# ============================================================
# Local Development
# ============================================================

def apply_local_exposure(
    img,
    attention,
    amount,
):
    factor = (
        1.0
        + attention * amount
    )

    return img * factor[..., None]


def apply_background_suppression(
    img,
    attention,
    amount,
):
    bg = 1.0 - attention

    factor = (
        1.0
        - bg * amount
    )

    return img * factor[..., None]


def apply_local_saturation(
    img,
    attention,
    amount,
):
    luma = luminance(img)

    chroma = (
        img
        - luma[..., None]
    )

    factor = (
        1.0
        + attention * amount
    )

    return (
        luma[..., None]
        + chroma * factor[..., None]
    )


def protect_person_saturation(
    img,
    subjects,
):
    person_masks = [
        s.mask
        for s in subjects
        if s.class_name == "person"
    ]

    if not person_masks:
        return img

    mask = np.maximum.reduce(
        person_masks
    )

    luma = luminance(img)

    chroma = (
        img
        - luma[..., None]
    )

    factor = (
        1.0
        - 0.08 * mask
    )

    return (
        luma[..., None]
        + chroma * factor[..., None]
    )


# ============================================================
# Tone Curve
# ============================================================

def filmic_tone_curve(
    img,
    contrast=1.0,
):
    """
    Simple filmic S-curve.

    Input is linear RGB.
    """

    x = np.maximum(
        img,
        0.0,
    )

    # Normalize gently before curve.
    x = x / (
        1.0 + x
    )

    x = np.clip(
        x,
        0.0,
        1.0,
    )

    # Contrast around middle gray.
    x = (
        x - 0.5
    ) * contrast + 0.5

    return np.clip(
        x,
        0.0,
        1.0,
    )


# ============================================================
# CLAHE
# ============================================================

def apply_clahe(
    img,
    clip_limit=1.5,
):
    img8 = np.clip(
        img * 255.0,
        0,
        255,
    ).astype(np.uint8)

    lab = cv2.cvtColor(
        img8,
        cv2.COLOR_RGB2LAB,
    )

    l, a, b = cv2.split(
        lab
    )

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(8, 8),
    )

    l = clahe.apply(l)

    lab = cv2.merge(
        [l, a, b]
    )

    result = cv2.cvtColor(
        lab,
        cv2.COLOR_LAB2RGB,
    )

    return (
        result.astype(np.float32)
        / 255.0
    )


# ============================================================
# Saturation
# ============================================================

def apply_saturation(
    img,
    factor,
):
    luma = luminance(img)

    out = (
        luma[..., None]
        + (
            img
            - luma[..., None]
        )
        * factor
    )

    return np.clip(
        out,
        0.0,
        None,
    )


# ============================================================
# Linear -> sRGB
# ============================================================

def linear_to_srgb(
    img,
):
    x = np.clip(
        img,
        0.0,
        1.0,
    )

    a = 0.055

    out = np.where(
        x <= 0.0031308,
        12.92 * x,
        (1 + a)
        * np.power(
            x,
            1 / 2.4,
        )
        - a,
    )

    return np.clip(
        out,
        0.0,
        1.0,
    )


# ============================================================
# Denoise
# ============================================================

def apply_denoise(
    img,
    strength,
):
    if strength <= 0.01:
        return img

    sigma_color = (
        0.02
        + strength * 0.06
    )

    sigma_space = (
        1.5
        + strength * 3.0
    )

    out = cv2.bilateralFilter(
        img.astype(np.float32),
        d=0,
        sigmaColor=sigma_color,
        sigmaSpace=sigma_space,
    )

    return out


# ============================================================
# Sharpen
# ============================================================

def apply_sharpen(
    img,
    strength,
):
    if strength <= 0.01:
        return img

    blur = cv2.GaussianBlur(
        img,
        (0, 0),
        1.2,
    )

    amount = (
        0.4
        * strength
    )

    out = (
        img
        + amount
        * (img - blur)
    )

    return np.clip(
        out,
        0.0,
        1.0,
    )


# ============================================================
# Auto Parameter Search
# ============================================================

def image_quality_score(
    img,
):
    luma = luminance(img)

    mean = float(
        np.mean(luma)
    )

    shadow = float(
        np.mean(luma < 0.02)
    )

    highlight = float(
        np.mean(luma > 0.98)
    )

    gray = (
        np.clip(
            luma * 255,
            0,
            255,
        )
        .astype(np.uint8)
    )

    edges = cv2.Canny(
        gray,
        50,
        150,
    )

    edge_density = float(
        np.mean(edges > 0)
    )

    # Penalize crushed blacks.
    shadow_penalty = (
        shadow * 1.2
    )

    # Penalize clipped highlights.
    highlight_penalty = (
        highlight * 1.5
    )

    # Reward reasonable midtones.
    midtone_score = math.exp(
        -(
            (mean - 0.42)
            ** 2
        )
        / 0.06
    )

    # Avoid extreme oversharpening.
    edge_score = min(
        edge_density / 0.15,
        1.0,
    )

    score = (
        2.0 * midtone_score
        + 0.3 * edge_score
        - shadow_penalty
        - highlight_penalty
    )

    return score


def auto_parameter_search(
    img,
):
    best_score = -float(
        "inf"
    )

    best_params = {
        "exposure": 0.0,
        "contrast": 1.0,
        "saturation": 1.0,
    }

    exposure_values = [
        -0.30,
        0.0,
        0.30,
    ]

    contrast_values = [
        0.96,
        1.00,
        1.06,
    ]

    saturation_values = [
        0.96,
        1.00,
        1.05,
    ]

    for ev in exposure_values:
        test = apply_exposure(
            img,
            ev,
        )

        for contrast in contrast_values:

            test2 = filmic_tone_curve(
                test,
                contrast,
            )

            for saturation in saturation_values:

                test3 = apply_saturation(
                    test2,
                    saturation,
                )

                score = image_quality_score(
                    test3
                )

                if score > best_score:
                    best_score = score

                    best_params = {
                        "exposure": ev,
                        "contrast": contrast,
                        "saturation": saturation,
                    }

    return (
        best_params,
        best_score,
    )


# ============================================================
# Camera Matrix Validation
# ============================================================

def matrix_quality(
    matrix: np.ndarray,
):
    """
    Check whether a usable camera matrix exists.
    """

    if matrix.shape != (4, 3):
        return False

    if not np.any(
        np.abs(matrix) > 1e-6
    ):
        return False

    finite = np.all(
        np.isfinite(matrix)
    )

    return bool(finite)


# ============================================================
# Main Development
# ============================================================

def auto_develop(
    filename: Path,
    output: Path,
    segmenter: SemanticSegmenter,
):
    logger.info(
        "Processing: %s",
        filename,
    )

    # --------------------------------------------------------
    # RAW + camera profile
    # --------------------------------------------------------

    try:
        raw, camera = load_raw(
            filename
        )
    except Exception as e:
        logger.error(
            "RAW loading failed: %s",
            e,
        )
        return False

    try:
        print_camera_profile(
            camera
        )

        # ----------------------------------------------------
        # Camera adaptive settings
        # ----------------------------------------------------

        camera_adjustment = (
            calculate_camera_adjustment(
                camera
            )
        )

        logger.info(
            "Camera family: %s",
            camera_adjustment[
                "family"
            ],
        )

        logger.info(
            "Camera denoise: %.3f",
            camera_adjustment[
                "denoise"
            ],
        )

        logger.info(
            "Camera sharpen: %.3f",
            camera_adjustment[
                "sharpen"
            ],
        )

        logger.info(
            "Camera highlight: %.3f",
            camera_adjustment[
                "highlight"
            ],
        )

        # ----------------------------------------------------
        # Check color matrix
        # ----------------------------------------------------

        matrix_ok = matrix_quality(
            camera.rgb_xyz_matrix
        )

        if matrix_ok:
            logger.info(
                "Camera color matrix: ENABLED"
            )
        else:
            logger.warning(
                "Camera color matrix unavailable."
            )

        # ----------------------------------------------------
        # RAW -> linear sRGB
        # ----------------------------------------------------

        img = develop_camera_rgb(
            raw,
            camera,
        )

    finally:
        raw.close()

    # --------------------------------------------------------
    # Basic analysis
    # --------------------------------------------------------

    analysis = analyze_image(
        img
    )

    logger.info(
        "Luma mean: %.4f",
        analysis.mean_luma,
    )

    logger.info(
        "Luma median: %.4f",
        analysis.median_luma,
    )

    logger.info(
        "Highlights: %.3f",
        analysis.highlight_ratio,
    )

    logger.info(
        "Shadows: %.3f",
        analysis.shadow_ratio,
    )

    # --------------------------------------------------------
    # Semantic segmentation
    # --------------------------------------------------------

    labels, confidence = (
        segmenter.predict(img)
    )

    # Segmentation is performed at reduced resolution.
    # Resize back to original size.

    h, w = img.shape[:2]

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

    # --------------------------------------------------------
    # Saliency
    # --------------------------------------------------------

    saliency = (
        calculate_saliency_map(
            img
        )
    )

    # --------------------------------------------------------
    # Subjects
    # --------------------------------------------------------

    subjects = rank_subjects(
        img,
        labels,
        confidence,
        saliency,
    )

    if subjects:
        for i, subject in enumerate(
            subjects[:5]
        ):
            logger.info(
                "Subject %d: %s "
                "confidence=%.2f "
                "importance=%.2f "
                "area=%.3f",
                i + 1,
                subject.class_name,
                subject.confidence,
                subject.importance,
                subject.area_ratio,
            )

    else:
        logger.info(
            "Subject: none"
        )

    # --------------------------------------------------------
    # Attention
    # --------------------------------------------------------

    subject_attention = (
        create_subject_attention(
            subjects,
            labels.shape,
        )
    )

    if subjects:
        main_importance = (
            subjects[0].importance
        )
    else:
        main_importance = 0.0

    attention = (
        combine_subject_and_saliency(
            subject_attention,
            saliency,
            main_importance,
        )
    )

    # --------------------------------------------------------
    # Scene
    # --------------------------------------------------------

    sky_mask = create_sky_mask(
        img
    )

    scene_name = classify_scene(
        analysis,
        subjects,
        sky_mask,
    )

    scene = SCENE_PROFILES[
        scene_name
    ]

    logger.info(
        "Scene: %s",
        scene_name,
    )

    # --------------------------------------------------------
    # Exposure
    # --------------------------------------------------------

    auto_ev = auto_exposure(
        img
    )

    exposure = (
        scene.exposure
        + auto_ev
    )

    logger.info(
        "Auto exposure: %.2f EV",
        exposure,
    )

    img = apply_exposure(
        img,
        exposure,
    )

    # --------------------------------------------------------
    # Camera highlight recovery
    # --------------------------------------------------------

    highlight_strength = max(
        scene.highlight,
        camera_adjustment[
            "highlight"
        ],
    )

    img = highlight_recovery(
        img,
        highlight_strength,
    )

    # --------------------------------------------------------
    # Shadows
    # --------------------------------------------------------

    img = shadow_lift(
        img,
        scene.shadow,
    )

    # --------------------------------------------------------
    # Sky
    # --------------------------------------------------------

    if scene_name in {
        "landscape",
        "sunset",
    }:

        sky_strength = (
            0.05
            if scene_name == "landscape"
            else 0.08
        )

        img = apply_local_exposure(
            img,
            sky_mask,
            -sky_strength,
        )

    # --------------------------------------------------------
    # Main subject
    # --------------------------------------------------------

    img = apply_local_exposure(
        img,
        attention,
        scene.subject,
    )

    # --------------------------------------------------------
    # Background
    # --------------------------------------------------------

    img = apply_background_suppression(
        img,
        attention,
        scene.background_suppression,
    )

    # --------------------------------------------------------
    # Person saturation protection
    # --------------------------------------------------------

    img = protect_person_saturation(
        img,
        subjects,
    )

    # --------------------------------------------------------
    # Scene tone
    # --------------------------------------------------------

    camera_contrast = (
        camera_adjustment[
            "contrast"
        ]
    )

    # --------------------------------------------------------
    # Automatic parameter search
    # --------------------------------------------------------

    search_params, search_score = (
        auto_parameter_search(
            img
        )
    )

    logger.info(
        "Auto search: "
        "EV=%.2f "
        "contrast=%.3f "
        "saturation=%.3f "
        "score=%.3f",
        search_params[
            "exposure"
        ],
        search_params[
            "contrast"
        ],
        search_params[
            "saturation"
        ],
        search_score,
    )

    img = apply_exposure(
        img,
        search_params[
            "exposure"
        ],
    )

    # --------------------------------------------------------
    # Tone curve
    # --------------------------------------------------------

    contrast = (
        scene.contrast
        * camera_contrast
        * search_params[
            "contrast"
        ]
    )

    img = filmic_tone_curve(
        img,
        contrast,
    )

    # --------------------------------------------------------
    # CLAHE
    # --------------------------------------------------------

    if scene_name in {
        "landscape",
        "city",
        "indoor",
        "general",
    }:
        img = apply_clahe(
            img,
            clip_limit=1.2,
        )

    # --------------------------------------------------------
    # Saturation
    # --------------------------------------------------------

    saturation = (
        scene.saturation
        * camera_adjustment[
            "saturation"
        ]
        * search_params[
            "saturation"
        ]
    )

    img = apply_saturation(
        img,
        saturation,
    )

    # --------------------------------------------------------
    # Clip before sRGB conversion
    # --------------------------------------------------------

    img = np.clip(
        img,
        0.0,
        1.0,
    )

    # --------------------------------------------------------
    # Linear -> sRGB
    # --------------------------------------------------------

    img = linear_to_srgb(
        img
    )

    # --------------------------------------------------------
    # Denoise
    # --------------------------------------------------------

    denoise = max(
        scene.denoise,
        camera_adjustment[
            "denoise"
        ],
    )

    img = apply_denoise(
        img,
        denoise,
    )

    # --------------------------------------------------------
    # Sharpen
    # --------------------------------------------------------

    sharpen = (
        scene.sharpen
        * camera_adjustment[
            "sharpen"
        ]
    )

    img = apply_sharpen(
        img,
        sharpen,
    )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    img8 = np.clip(
        img * 255.0,
        0,
        255,
    ).astype(np.uint8)

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    Image.fromarray(
        img8,
        mode="RGB",
    ).save(
        output,
        quality=95,
        subsampling=0,
    )

    logger.info(
        "Saved: %s",
        output,
    )

    return True


# ============================================================
# RAW File Search
# ============================================================

def collect_raw_files(
    input_path: Path,
):
    if input_path.is_file():

        if (
            input_path.suffix.lower()
            in RAW_EXTENSIONS
        ):
            return [input_path]

        return []

    files = []

    for path in input_path.rglob("*"):

        if not path.is_file():
            continue

        if (
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
            "Automatic RAW photo developer v12"
        )
    )

    parser.add_argument(
        "input",
        type=Path,
        help=(
            "RAW file or directory"
        ),
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("./output"),
        help=(
            "Output directory"
        ),
    )

    parser.add_argument(
        "--device",
        choices=[
            "auto",
            "cpu",
            "cuda",
        ],
        default="auto",
        help=(
            "Segmentation device"
        ),
    )

    parser.add_argument(
        "--max-size",
        type=int,
        default=768,
        help=(
            "Maximum segmentation size"
        ),
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    if (
        args.device == "cuda"
        and not torch.cuda.is_available()
    ):
        logger.error(
            "CUDA was requested but "
            "PyTorch CUDA is unavailable."
        )
        sys.exit(1)

    # --------------------------------------------------------
    # Files
    # --------------------------------------------------------

    files = collect_raw_files(
        args.input
    )

    if not files:
        logger.error(
            "No RAW files found."
        )
        sys.exit(1)

    logger.info(
        "RAW files: %d",
        len(files),
    )

    # --------------------------------------------------------
    # Segmentation
    # --------------------------------------------------------

    segmenter = SemanticSegmenter(
        device=args.device,
        max_size=args.max_size,
    )

    # --------------------------------------------------------
    # Process
    # --------------------------------------------------------

    success = 0

    for index, filename in enumerate(
        files,
        start=1,
    ):

        logger.info(
            "========================================"
        )

        logger.info(
            "[%d/%d]",
            index,
            len(files),
        )

        # Preserve directory structure
        if args.input.is_dir():

            relative = filename.relative_to(
                args.input
            )

            output = (
                args.output
                / relative.parent
                / (
                    relative.stem
                    + "_developed.jpg"
                )
            )

        else:

            output = (
                args.output
                / (
                    filename.stem
                    + "_developed.jpg"
                )
            )

        try:

            if auto_develop(
                filename,
                output,
                segmenter,
            ):
                success += 1

        except KeyboardInterrupt:
            logger.info(
                "Interrupted."
            )
            break

        except Exception as e:
            logger.exception(
                "Failed: %s",
                e,
            )

    logger.info(
        "========================================"
    )

    logger.info(
        "Completed: %d/%d",
        success,
        len(files),
    )


if __name__ == "__main__":
    main()