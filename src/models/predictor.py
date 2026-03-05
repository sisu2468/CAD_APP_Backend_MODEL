"""
Run a trained outline segmentation model on an image and return a binary mask.

Backward compatible: auto-detects whether the checkpoint was saved from
ResNet34UNet (new, pretrained backbone) or the old OutlineUNet by inspecting
the state-dict keys.  No manual flag needed.
"""

from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Union

from src.models.outline_unet import OutlineUNet, MODEL_INPUT_SIZE as UNET_SIZE, load_outline_model
from src.models.pretrained_unet import (
    ResNet34UNet,
    PRETRAINED_MODEL_INPUT_SIZE,
    load_pretrained_model,
    is_pretrained_checkpoint,
)
from src.outline_extractor import OutlineExtractor, OutlineResult, GeometricOutline

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class OutlineModelPredictor:
    """Load a checkpoint and predict outline masks / GeometricOutlines from images.

    Supports both checkpoint formats:
    - ResNet34UNet  (new, pretrained backbone, 512×512 input)
    - OutlineUNet   (original lightweight model, 256×256 input)

    The correct model type is detected automatically from the checkpoint.
    """

    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        device: Optional[str] = None,
    ) -> None:
        if not TORCH_AVAILABLE:
            raise RuntimeError(
                "PyTorch is required for model inference. Install: pip install torch torchvision"
            )
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        # Auto-detect model type from checkpoint keys
        self._is_pretrained = is_pretrained_checkpoint(str(self.checkpoint_path))
        if self._is_pretrained:
            self.model = load_pretrained_model(str(self.checkpoint_path), self.device)
            self.input_size = PRETRAINED_MODEL_INPUT_SIZE
        else:
            self.model = load_outline_model(str(self.checkpoint_path), self.device)
            self.input_size = UNET_SIZE

        self._extractor = OutlineExtractor()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _preprocess(self, image: np.ndarray) -> "torch.Tensor":
        """BGR (H, W, 3) -> normalised tensor (1, 3, input_size, input_size)."""
        img = cv2.resize(image, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (img - _IMAGENET_MEAN) / _IMAGENET_STD
        x = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        return x

    def _run_model(self, x: "torch.Tensor") -> np.ndarray:
        """Run forward pass -> uint8 probability map [0, 255]."""
        with torch.no_grad():
            logits = self.model(x)
            prob = torch.sigmoid(logits).squeeze().cpu().numpy()
        return (prob * 255).astype(np.uint8)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict_mask(self, image_path: str) -> np.ndarray:
        """Load image from path, run model -> binary mask (H, W) at original resolution."""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Cannot read image: {image_path}")
        return self.predict_mask_from_array(img)

    def predict_mask_from_array(self, image: np.ndarray) -> np.ndarray:
        """Predict mask from BGR numpy image (H, W, 3). Returns binary mask same size."""
        orig_h, orig_w = image.shape[:2]
        prob = self._run_model(self._preprocess(image))
        # Threshold at 0.5 (=127 in uint8 scale)
        binary = (prob > 127).astype(np.uint8) * 255
        return cv2.resize(binary, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    def predict_prob_map(self, image_path: str) -> np.ndarray:
        """Return full probability map [0, 255] at original resolution (not thresholded)."""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Cannot read image: {image_path}")
        orig_h, orig_w = img.shape[:2]
        prob = self._run_model(self._preprocess(img))
        return cv2.resize(prob, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

    def extract(self, image_path: str) -> OutlineResult:
        """Predict mask then extract outline + holes using rule-based post-processing."""
        mask = self.predict_mask(image_path)
        return self._extractor.extract_from_mask(mask)

    def extract_geometric(
        self,
        image_path: str,
        min_line_length: float = 20.0,
    ) -> GeometricOutline:
        """Predict mask then extract GeometricOutline with LSD line segments.

        Returns a GeometricOutline containing:
        - outline:       raw simplified contour (for polyline fallback)
        - holes:         algebraically-fitted circles
        - line_segments: LSD-detected straight boundary segments

        These can be exported to DXF as LINE + CIRCLE entities for much
        higher geometric accuracy than a plain LWPOLYLINE.
        """
        mask = self.predict_mask(image_path)
        return self._extractor.extract_geometric(mask, min_line_length=min_line_length)


# ------------------------------------------------------------------
# Default model path resolution
# ------------------------------------------------------------------

def get_default_model_path() -> Optional[Path]:
    """Return path to the best available checkpoint, or None."""
    base = Path(__file__).resolve().parent.parent.parent
    candidates = [
        base / "models" / "outline_model.pt",
        base / "models" / "outline_model.pth",
        base / "training_data" / "outline_model.pt",
        base / "training_data" / "best_outline.pt",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None
