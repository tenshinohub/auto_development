from __future__ import annotations

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


class ExifMetadata:
    camera_make: str = ""
    camera_model: str = ""
    iso: float = 100.0
    exposure_time: float = 0.0
    aperture: float = 0.0
    focal_length: float = 0.0
    width: int = 0
    height: int = 0

class CameraProfile:
    make: str = ""
    model: str = ""
    family: str = "generic"

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

class ShootingCondition:
    iso_factor: float
    low_light: bool
    motion_risk: float
    shallow_dof: bool
    wide_angle: bool
    telephoto: bool
    estimated_noise: float

class SubjectCandidate:
    label: str
    class_id: int
    score: float
    area: float
    confidence: float
    center_score: float
    saliency_score: float
    local_contrast: float

class SceneResult:
    scene: str
    confidence: float

class RegionStats:
    subject_median: Optional[float]
    background_median: Optional[float]
    subject_area: float
    background_area: float

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
