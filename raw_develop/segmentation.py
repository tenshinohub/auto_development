from __future__ import annotations

import cv2
import numpy as np
from PIL import Image
import torch
import torchvision
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large

class Segmenter:
    def __init__(
        self,
        device: str = "auto",
        max_size: int = 768,
    ):
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        elif device == "cuda":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                print(
                    "CUDA is not available. Falling back to CPU."
                )
                self.device = torch.device("cpu")
        else:
            self.device = torch.device("cpu")

        self.max_size = max_size

        print(
            f"Segmentation device: {self.device}"
        )

        try:
            weights = (
                torchvision.models.segmentation
                .DeepLabV3_MobileNet_V3_Large_Weights
                .DEFAULT
            )

            self.model = deeplabv3_mobilenet_v3_large(
                weights=weights
            )

            self.preprocess = weights.transforms()

        except Exception:
            print(
                "Could not load pretrained DeepLabV3 weights."
            )

            self.model = deeplabv3_mobilenet_v3_large(
                weights=None
            )

            self.preprocess = None

        self.model.eval()
        self.model.to(self.device)

    def _resize(
        self,
        image: np.ndarray,
    ) -> tuple[np.ndarray, float, float]:
        h, w = image.shape[:2]

        scale = min(
            1.0,
            self.max_size / max(h, w),
        )

        if scale == 1.0:
            return image, 1.0, 1.0

        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))

        resized = cv2.resize(
            image,
            (new_w, new_h),
            interpolation=cv2.INTER_AREA,
        )

        return resized, scale, scale

    @torch.inference_mode()
    def predict(
        self,
        image: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        image = normalize_image(image)

        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                f"Expected HxWx3 RGB image, got shape={image.shape}"
            )

        resized, _, _ = self._resize(image)

        # torchvision's weights.transforms() in the installed version
        # expects PIL Image or Tensor, not numpy.ndarray.
        # Convert the resized RGB float image [0,1] to PIL RGB here.
        resized_u8 = np.clip(
            resized * 255.0,
            0,
            255,
        ).astype(np.uint8)

        pil_image = Image.fromarray(
            resized_u8,
            mode="RGB",
        )

        if self.preprocess is not None:
            tensor = self.preprocess(
                pil_image
            ).unsqueeze(0)
        else:
            tensor = torch.from_numpy(
                resized.transpose(2, 0, 1)
            ).float().unsqueeze(0)

        tensor = tensor.to(self.device)

        output = self.model(tensor)["out"]

        probabilities = torch.softmax(
            output,
            dim=1,
        )

        confidence, classes = torch.max(
            probabilities,
            dim=1,
        )

        classes = classes[0].cpu().numpy()
        confidence = confidence[0].cpu().numpy()

        original_h, original_w = image.shape[:2]

        classes = cv2.resize(
            classes.astype(np.uint8),
            (original_w, original_h),
            interpolation=cv2.INTER_NEAREST,
        )

        confidence = cv2.resize(
            confidence.astype(np.float32),
            (original_w, original_h),
            interpolation=cv2.INTER_LINEAR,
        )

        return classes, confidence
