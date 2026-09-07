from __future__ import annotations

import json
import subprocess
from pathlib import Path
from PIL import Image

from .constants import ExifMetadata, CameraProfile

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
