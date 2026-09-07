from __future__ import annotations

from pathlib import Path
import numpy as np
import rawpy
from PIL import Image

from .metadata import read_metadata, detect_camera_family
from .shooting import analyze_shooting
from .raw_io import raw_to_linear_rgb
from .color import linear_to_srgb, srgb_to_linear
from .stats import calculate_stats, print_stats
from .segmentation import Segmenter
from .saliency import calculate_saliency
from .subjects import rank_subjects, make_region_masks
from .scene import SCENE_PROFILES, classify_scene
from .tone import (
    calculate_exposure_target,
    estimate_exposure_ev,
    apply_exposure,
    apply_contrast,
    apply_saturation,
    apply_tone,
)
from .search import automatic_parameter_search
from .regions import (
    calculate_region_stats,
    apply_region_processing,
    print_region_color_drift,
)
from .filters import calculate_denoise_strength, apply_denoise, apply_sharpen
from .debug import save_stage

class AutoDeveloper:

    def __init__(
        self,
        device: str = "auto",
        debug: bool = False,
    ):
        self.debug = debug

        self.segmenter = Segmenter(
            device=device
        )

    def process_file(
        self,
        raw_path: Path,
        output_path: Path,
    ):

        print()
        print("=" * 72)
        print(
            f"Processing: {raw_path}"
        )
        print("=" * 72)

        meta = read_metadata(
            raw_path
        )

        camera = detect_camera_family(
            meta
        )

        shooting = analyze_shooting(
            meta
        )

        print(
            f"Camera: "
            f"{meta.camera_make} "
            f"{meta.camera_model}"
        )

        print(
            f"ISO: {meta.iso:g}"
        )

        print(
            f"Exposure: "
            f"{meta.exposure_time:g}"
        )

        print(
            f"Aperture: "
            f"{meta.aperture:g}"
        )

        print(
            f"Focal length: "
            f"{meta.focal_length:g}"
        )

        print(
            f"Camera family: "
            f"{camera.family}"
        )

        # ----------------------------------------------------
        # RAW
        # ----------------------------------------------------

        with rawpy.imread(
            str(raw_path)
        ) as raw:

            linear_rgb = raw_to_linear_rgb(
                raw
            )

        linear_rgb = normalize_image(
            linear_rgb
        )

        # ----------------------------------------------------
        # Debug: true linear stats
        # ----------------------------------------------------

        analysis_srgb = linear_to_srgb(
            linear_rgb
        )

        if self.debug:
            print_stats(
                "01_raw_linear",
                linear_rgb,
            )

            save_stage(
                self.debug_dir,
                "01_raw_linear",
                analysis_srgb,
                stats_image=linear_rgb,
            )

            print_stats(
                "02_analysis_srgb",
                analysis_srgb,
            )

            save_stage(
                self.debug_dir,
                "02_analysis_srgb",
                analysis_srgb,
                stats_image=analysis_srgb,
            )

        else:
            print_stats(
                "RAW / LibRaw linear RGB",
                linear_rgb,
            )

        # ----------------------------------------------------
        # Semantic segmentation
        # ----------------------------------------------------

        class_map, confidence = (
            self.segmenter.predict(
                analysis_srgb
            )
        )

        saliency = calculate_saliency(
            analysis_srgb
        )

        subjects = rank_subjects(
            class_map,
            confidence,
            saliency,
            analysis_srgb,
        )

        masks = make_region_masks(
            class_map,
            analysis_srgb,
            subjects,
        )

        # ----------------------------------------------------
        # Subjects
        # ----------------------------------------------------

        semantic_subject_area = float(
            np.mean(masks["semantic_subject"])
        )
        candidate_subject_area = float(
            np.mean(masks["candidate_subject"])
        )

        print()
        print(
            f"Semantic subject area : "
            f"{semantic_subject_area:.3f}"
        )
        print(
            f"Fallback subject area : "
            f"{candidate_subject_area:.3f}"
        )

        print(
            f"Face candidate area   : "
            f"{float(np.mean(masks['face'])):.4f} "
            f"confidence {float(masks.get('face_confidence', 0.0)):.3f}"
        )

        if subjects:
            for subject in subjects[:5]:
                print(
                    f"Subject "
                    f"{subject.label}: "
                    f"score {subject.score:.3f} "
                    f"area {subject.area:.3f} "
                    f"conf {subject.confidence:.3f}"
                )
        else:
            print(
                "Subject: none"
            )

        # ----------------------------------------------------
        # Scene
        # ----------------------------------------------------

        raw_stats = calculate_stats(
            analysis_srgb
        )

        scene_result = classify_scene(
            raw_stats,
            shooting,
            subjects,
            masks,
        )

        print(
            f"Scene: "
            f"{scene_result.scene} "
            f"confidence "
            f"{scene_result.confidence:.3f}"
        )

        scene = scene_result.scene

        profile = SCENE_PROFILES[
            scene
        ]

        # ----------------------------------------------------
        # Exposure
        # ----------------------------------------------------

        target = calculate_exposure_target(
            scene
        )

        estimated_ev = estimate_exposure_ev(
            raw_stats,
            target,
        )

        print(
            f"Global target      : "
            f"{target:.3f}"
        )

        if scene == "portrait":
            subject_target = 0.260
        elif scene == "night":
            subject_target = 0.20
        elif scene == "indoor":
            subject_target = 0.25
        else:
            subject_target = min(
                target + 0.02,
                0.26,
            )

        print(
            f"Subject target     : "
            f"{subject_target:.3f}"
        )

        print(
            f"Highlight soft     : "
            f"{0.740:.3f}"
        )

        print(
            f"Highlight hard     : "
            f"{0.880:.3f}"
        )

        print(
            f"Estimated EV       : "
            f"{estimated_ev:+.3f}"
        )

        # ----------------------------------------------------
        # Search
        # ----------------------------------------------------

        params = automatic_parameter_search(
            analysis_srgb,
            scene,
            profile,
            estimated_ev,
            masks,
        )

        denoise_strength = (
            calculate_denoise_strength(
                params.denoise,
                shooting,
            )
        )

        print(
            f"Denoise            : "
            f"{denoise_strength:.3f}"
        )

        print(
            f"Tone               : "
            f"{params.tone_strength:.3f}"
        )

        print(
            f"Shadow lift        : "
            f"{params.shadow_lift:.3f}"
        )

        print(
            f"Highlight protect  : "
            f"{params.highlight_protection:.3f}"
        )

        # ----------------------------------------------------
        # Stage 03: exposure
        # ----------------------------------------------------

        current = apply_exposure(
            analysis_srgb,
            params.exposure_ev,
        )

        if self.debug:
            save_stage(
                self.debug_dir,
                "03_after_exposure",
                current,
            )

            print_stats(
                "03_after_exposure",
                current,
            )

        # ----------------------------------------------------
        # Stage 04: contrast
        # ----------------------------------------------------

        current = apply_contrast(
            current,
            params.contrast,
        )

        if self.debug:
            save_stage(
                self.debug_dir,
                "04_after_contrast",
                current,
            )

            print_stats(
                "04_after_contrast",
                current,
            )

        # ----------------------------------------------------
        # Stage 05: saturation
        # ----------------------------------------------------

        current = apply_saturation(
            current,
            params.saturation,
        )

        if self.debug:
            save_stage(
                self.debug_dir,
                "05_after_saturation",
                current,
            )

            print_stats(
                "05_after_saturation",
                current,
            )

        # ----------------------------------------------------
        # Stage 06: tone
        # ----------------------------------------------------

        current = apply_tone(
            current,
            params.tone_strength,
            params.shadow_lift,
            params.highlight_protection,
        )

        if self.debug:
            save_stage(
                self.debug_dir,
                "06_after_tone",
                current,
            )

            print_stats(
                "06_after_tone",
                current,
            )

        # ----------------------------------------------------
        # Region stats before local processing
        # ----------------------------------------------------

        region_before = (
            calculate_region_stats(
                current,
                masks,
            )
        )

        print()
        print(
            "Region before local:"
        )

        print(
            f"  subject median    : "
            f"{fmt_optional(region_before.subject_median)}"
        )

        print(
            f"  background median : "
            f"{fmt_optional(region_before.background_median)}"
        )

        # ----------------------------------------------------
        # Stage 07: local region
        # ----------------------------------------------------

        region_before_image = current.copy()

        current = apply_region_processing(
            current,
            masks,
            params,
        )

        print_region_color_drift(
            region_before_image,
            current,
        )

        region_after = (
            calculate_region_stats(
                current,
                masks,
            )
        )

        print(
            "Region after local:"
        )

        print(
            f"  subject median    : "
            f"{fmt_optional(region_after.subject_median)}"
        )

        print(
            f"  background median : "
            f"{fmt_optional(region_after.background_median)}"
        )

        if self.debug:
            save_stage(
                self.debug_dir,
                "07_after_region",
                current,
            )

            print_stats(
                "07_after_region",
                current,
            )

        # ----------------------------------------------------
        # Stage 08: denoise
        # ----------------------------------------------------

        current = apply_denoise(
            current,
            denoise_strength,
        )

        if self.debug:
            save_stage(
                self.debug_dir,
                "08_after_denoise",
                current,
            )

            print_stats(
                "08_after_denoise",
                current,
            )

        # ----------------------------------------------------
        # Stage 09: sharpen
        # ----------------------------------------------------

        current = apply_sharpen(
            current,
            params.sharpen,
        )

        # ----------------------------------------------------
        # Final brightness feedback
        # ----------------------------------------------------
        # Tone/local processing can change the global median.  Apply only
        # a small corrective EV so the final image does not remain dark.
        final_stats_before_feedback = calculate_stats(
            current
        )

        final_target = calculate_exposure_target(
            scene
        )

        final_error = (
            final_target
            - final_stats_before_feedback.median
        )

        if final_error > 0.010:
            correction_ev = clamp(
                math.log2(
                    max(final_target, 1e-5)
                    / max(
                        final_stats_before_feedback.median,
                        1e-5,
                    )
                ) * 0.75,
                0.0,
                0.45,
            )

            # Never use the feedback to push an already bright/highlighted
            # image upward.
            if (
                final_stats_before_feedback.p99 < 0.78
                and correction_ev > 0.0
            ):
                current = apply_exposure(
                    current,
                    correction_ev,
                )

                print(
                    f"Final brightness correction: "
                    f"{correction_ev:+.3f} EV"
                )
            else:
                print(
                    "Final brightness correction: "
                    "skipped (highlight headroom)"
                )
        else:
            print(
                "Final brightness correction: "
                "not needed"
            )

        if self.debug:
            save_stage(
                self.debug_dir,
                "09_after_sharpen",
                current,
            )

            print_stats(
                "09_after_sharpen",
                current,
            )

        # ----------------------------------------------------
        # Final
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

            # Important:
            # display image is sRGB,
            # stats image is true linear RGB.
            save_stage(
                self.debug_dir,
                "10_final_linear",
                final_srgb,
                stats_image=final_linear,
            )

            print_stats(
                "10_final_linear",
                final_linear,
            )

            save_stage(
                self.debug_dir,
                "11_final_srgb",
                final_srgb,
            )

            print_stats(
                "11_final_srgb",
                final_srgb,
            )

        final_u8 = np.clip(
            final_srgb * 255.0,
            0,
            255,
        ).astype(np.uint8)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        Image.fromarray(
            final_u8,
            mode="RGB",
        ).save(
            output_path,
            quality=95,
            subsampling=0,
        )

        final_region = (
            calculate_region_stats(
                final_srgb,
                masks,
            )
        )

        print()
        print(
            "Final region:"
        )

        print(
            f"  subject median    : "
            f"{fmt_optional(final_region.subject_median)}"
        )

        print(
            f"  background median : "
            f"{fmt_optional(final_region.background_median)}"
        )

        print(
            f"Saved: {output_path}"
        )

    # --------------------------------------------------------
    # Debug directory
    # --------------------------------------------------------

    @property
    def debug_dir(self) -> Path:
        return self._debug_dir

    @debug_dir.setter
    def debug_dir(self, value: Path):
        self._debug_dir = value
