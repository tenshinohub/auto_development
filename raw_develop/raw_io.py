from __future__ import annotations

import warnings
import numpy as np
import rawpy

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
