from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from .stats import calculate_stats
from .utils import normalize_image

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
