"""
Train the outline segmentation model on training_data samples.

Improvements over the original:
1. ResNet-34 backbone pretrained on ImageNet — works with as few as 5 samples.
2. Combined Dice + BCE loss — much better for imbalanced foreground/background.
3. Data augmentation — flip, rotation, scale-crop, brightness — generates ~40×
   more effective training pairs from the same 5 samples.
4. 512×512 input resolution — preserves fine edge detail.
5. Two-phase training: frozen encoder first (fast decoder warmup), then full
   fine-tuning with differential learning rates.
6. Backward compatible: falls back to old OutlineUNet if torchvision is absent.

Usage:
    # New pretrained model (recommended):
    python -m src.training.train_model --model pretrained --epochs 200

    # Resume from checkpoint:
    python -m src.training.train_model --model pretrained --resume training_data/outline_model.pt --epochs 100

    # Old lightweight model (CPU-only):
    python -m src.training.train_model --model unet --epochs 100
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from src.training.train_from_samples import get_training_samples, get_target_contours, _pdf_to_image

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from src.models.outline_unet import OutlineUNet, MODEL_INPUT_SIZE as UNET_INPUT_SIZE
from src.models.pretrained_unet import ResNet34UNet, PRETRAINED_MODEL_INPUT_SIZE


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def dice_bce_loss(pred: "torch.Tensor", target: "torch.Tensor", smooth: float = 1.0) -> "torch.Tensor":
    """Combined Dice + BCE loss.

    BCE handles per-pixel class balance; Dice maximises overlap directly.
    Together they train much faster than BCE alone on imbalanced masks.
    """
    bce = F.binary_cross_entropy_with_logits(pred, target)
    p = torch.sigmoid(pred)
    intersection = (p * target).sum()
    dice = 1.0 - (2.0 * intersection + smooth) / (p.sum() + target.sum() + smooth)
    return bce + dice


# ---------------------------------------------------------------------------
# Augmentation (pure numpy/cv2 — no extra dependencies)
# ---------------------------------------------------------------------------

def _augment(image: np.ndarray, mask: np.ndarray, size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Apply random augmentations to (image, mask) pair.

    All spatial transforms are applied identically to both arrays.
    Image-only transforms (brightness/contrast) are applied only to image.
    """
    h, w = image.shape[:2]

    # --- Horizontal flip ---
    if random.random() > 0.5:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)

    # --- Vertical flip ---
    if random.random() > 0.5:
        image = cv2.flip(image, 0)
        mask = cv2.flip(mask, 0)

    # --- Random rotation (-35° to +35°) ---
    angle = random.uniform(-35.0, 35.0)
    cx, cy = w / 2.0, h / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    image = cv2.warpAffine(image, M, (w, h),
                           flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REFLECT_101)
    mask = cv2.warpAffine(mask, M, (w, h),
                          flags=cv2.INTER_NEAREST,
                          borderMode=cv2.BORDER_REFLECT_101)

    # --- Random scale-crop (zoom in 80%–100% of the image) ---
    scale = random.uniform(0.75, 1.0)
    sh, sw = int(h * scale), int(w * scale)
    y0 = random.randint(0, max(0, h - sh))
    x0 = random.randint(0, max(0, w - sw))
    image = image[y0: y0 + sh, x0: x0 + sw]
    mask = mask[y0: y0 + sh, x0: x0 + sw]

    # --- Resize to target size ---
    image = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)

    # --- Brightness / contrast jitter (image only) ---
    alpha = random.uniform(0.65, 1.35)   # contrast multiplier
    beta = random.randint(-40, 40)        # brightness offset
    image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    # --- Random Gaussian noise (image only) ---
    if random.random() > 0.5:
        noise = np.random.normal(0, random.uniform(3, 12), image.shape).astype(np.float32)
        image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return image, mask


# ---------------------------------------------------------------------------
# Target mask
# ---------------------------------------------------------------------------

def target_to_mask(target_path: str, size: int) -> Optional[np.ndarray]:
    """Load a CAD drawing (PNG/JPG/PDF) → filled binary mask (size × size).

    A CAD drawing has dark lines on a white background.  The mask we generate is:
      1 = solid part material (filled interior, holes punched out)
      0 = background and through-holes

    Steps:
    1. Binarize: dark ink → white (foreground), white background → black.
    2. Morphological close: fills small gaps in the line work so contours close.
    3. RETR_CCOMP contour hierarchy: gives outer boundary and inner hole boundaries.
    4. Fill the largest outer contour solid, then erase each direct child (hole).
    """
    path = Path(target_path)
    img: Optional[np.ndarray] = None
    if path.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
        img = cv2.imread(str(path))
    elif path.suffix.lower() == ".pdf":
        img = _pdf_to_image(str(path))
    if img is None:
        return None

    h_img, w_img = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Binarize: pixels darker than 200 (the CAD lines) → foreground
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    # Close small gaps so line segments form closed boundaries
    close_k = max(5, min(h_img, w_img) // 60)
    if close_k % 2 == 0:
        close_k += 1
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel)

    # Two-level contour hierarchy: level-0 = outer, level-1 = inner holes
    all_contours, hierarchy = cv2.findContours(closed, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if not all_contours or hierarchy is None:
        return None
    hierarchy = hierarchy[0]  # shape (N, 4): [next, prev, first_child, parent]

    # Outer contours have parent == -1
    outer_indices = [i for i in range(len(all_contours)) if hierarchy[i][3] == -1]
    if not outer_indices:
        return None

    # Largest outer contour = main part boundary
    largest_idx = max(outer_indices, key=lambda i: cv2.contourArea(all_contours[i]))
    if cv2.contourArea(all_contours[largest_idx]) < 200:
        return None

    # Fill the main part solid
    filled = np.zeros((h_img, w_img), dtype=np.uint8)
    cv2.drawContours(filled, [all_contours[largest_idx]], -1, 255, cv2.FILLED)

    # Punch out holes: direct children of the largest contour
    child_idx = hierarchy[largest_idx][2]   # first child index (-1 if none)
    while child_idx >= 0:
        cv2.drawContours(filled, [all_contours[child_idx]], -1, 0, cv2.FILLED)
        child_idx = hierarchy[child_idx][0]  # next sibling

    if filled.sum() < 100 * 255:
        return None

    mask = cv2.resize(filled, (size, size), interpolation=cv2.INTER_NEAREST)
    return (mask > 0).astype(np.uint8)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class OutlineDataset(Dataset):
    """(image path, target mask) pairs with optional augmentation."""

    def __init__(
        self,
        samples: List[Tuple[List[str], str]],
        size: int,
        augment: bool = True,
        aug_factor: int = 40,
    ) -> None:
        """
        Args:
            samples: From get_training_samples().  Each entry is
                     (list_of_input_photo_paths, cad_target_path).
            size: Square target resolution (e.g. 512).
            augment: Apply random augmentation.
            aug_factor: Augmented copies per base (photo, mask) pair.
                        With 9 base pairs × 40 → 360 items/epoch.
        """
        self.size = size
        self.augment = augment
        self.aug_factor = aug_factor if augment else 1

        self._base: List[Tuple[str, np.ndarray]] = []
        for input_paths, target_path in samples:
            mask = target_to_mask(target_path, size)
            if mask is None:
                continue
            # Register ONE entry per input photo — not just the first.
            # Samples 1–4 have 2 photos each → up to 9 base pairs total.
            for inp in input_paths:
                if Path(inp).exists():
                    self._base.append((inp, mask))

    def __len__(self) -> int:
        return len(self._base) * self.aug_factor

    def __getitem__(self, idx: int):
        import torch  # local import to keep top-level optional
        base_idx = idx % len(self._base)
        inp_path, mask = self._base[base_idx]

        img = cv2.imread(inp_path)
        if img is None:
            img = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        else:
            img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_LINEAR)

        # Clone mask for augmentation
        mask_aug = mask.copy()

        if self.augment:
            img, mask_aug = _augment(img, mask_aug, self.size)
        else:
            # Just resize to ensure exact size
            img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
            mask_aug = cv2.resize(mask_aug, (self.size, self.size), interpolation=cv2.INTER_NEAREST)

        # Normalise image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (img - _IMAGENET_MEAN) / _IMAGENET_STD

        x = torch.from_numpy(img).permute(2, 0, 1).float()
        y = torch.from_numpy(mask_aug).unsqueeze(0).float()
        return x, y


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def _train_epoch(model, loader, criterion, optimizer, device) -> float:
    model.train()
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total += loss.item()
        n += 1
    return total / n if n else 0.0


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------

def run_training(
    data_dir: str,
    output_dir: Optional[str] = None,
    model_type: str = "pretrained",   # "pretrained" or "unet"
    epochs: int = 200,
    batch_size: int = 2,
    lr: float = 1e-3,
    resume: Optional[str] = None,
    save_every: int = 20,
    freeze_epochs: int = 30,          # epochs to train with frozen encoder
    aug_factor: int = 40,
) -> str:
    """
    Train or continue training the outline segmentation model.

    Two-phase training (pretrained model only):
    1. Phase 1 — frozen encoder, `freeze_epochs` epochs with lr=1e-3 (decoder warmup).
    2. Phase 2 — full model, remaining epochs with differential LRs:
         encoder: lr × 0.1,  decoder: lr.

    Returns path to best checkpoint.
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch required. Install: pip install torch torchvision")

    data_dir = Path(data_dir)
    output_dir = Path(output_dir or data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ----- Dataset -----
    samples = get_training_samples(str(data_dir))
    if not samples:
        raise ValueError("No training samples found in " + str(data_dir))

    use_pretrained = (model_type == "pretrained")
    input_size = PRETRAINED_MODEL_INPUT_SIZE if use_pretrained else UNET_INPUT_SIZE

    dataset = OutlineDataset(
        samples,
        size=input_size,
        augment=True,
        aug_factor=aug_factor,
    )
    if len(dataset) == 0:
        raise ValueError("No valid (input, target) pairs — check training_data folder.")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )
    n_base = len(dataset._base)
    print(f"Dataset: {n_base} base (photo, mask) pairs × {aug_factor} aug = {len(dataset)} items/epoch")
    if n_base < 5:
        print(f"  WARNING: only {n_base} base pairs found. Check training_data/ contents and file names.")

    # ----- Model -----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if use_pretrained:
        print("Loading ResNet-34 backbone (pretrained on ImageNet)...")
        model = ResNet34UNet(pretrained=(resume is None)).to(device)
    else:
        model = OutlineUNet().to(device)

    criterion = dice_bce_loss
    start_epoch = 0
    best_loss = float("inf")

    # ----- Resume from checkpoint -----
    if resume and Path(resume).exists():
        state = torch.load(resume, map_location=device)
        model.load_state_dict(state.get("model_state_dict", state))
        start_epoch = state.get("epoch", 0) + 1
        best_loss = state.get("best_loss", best_loss)
        print(f"Resumed from epoch {start_epoch}, best_loss={best_loss:.4f}")

    best_path = output_dir / "outline_model.pt"

    def _save(epoch: int, loss: float, is_best: bool, tag: Optional[str] = None) -> None:
        payload = {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "best_loss": best_loss,
            "model_type": model_type,
            "input_size": input_size,
        }
        if is_best:
            torch.save(payload, best_path)
        if tag:
            torch.save(payload, output_dir / f"outline_{tag}.pt")

    # ----- Phase 1: frozen encoder warmup (pretrained model only) -----
    if use_pretrained and resume is None and freeze_epochs > 0:
        print(f"\n--- Phase 1: Decoder warmup ({freeze_epochs} epochs, encoder frozen) ---")
        model.freeze_encoder()
        optimizer = torch.optim.AdamW(model.decoder_parameters(), lr=lr, weight_decay=1e-4)
        for epoch in range(start_epoch, min(start_epoch + freeze_epochs, epochs)):
            loss = _train_epoch(model, loader, criterion, optimizer, device)
            print(f"  [Frozen] Epoch {epoch + 1}/{epochs}  loss={loss:.4f}")
            if loss < best_loss:
                best_loss = loss
                _save(epoch, loss, is_best=True)
            if (epoch + 1) % save_every == 0:
                _save(epoch, loss, is_best=False, tag=f"epoch_{epoch + 1}")
        start_epoch = min(start_epoch + freeze_epochs, epochs)
        model.unfreeze_encoder()

    # ----- Phase 2: full fine-tuning -----
    if start_epoch < epochs:
        print(f"\n--- Phase 2: Full fine-tuning, epochs {start_epoch + 1}–{epochs} ---")
        if use_pretrained:
            optimizer = torch.optim.AdamW([
                {"params": model.encoder_parameters(), "lr": lr * 0.05},
                {"params": model.decoder_parameters(), "lr": lr},
            ], weight_decay=1e-4)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # Reload optimizer state if resuming (best-effort)
        if resume and Path(resume).exists():
            state = torch.load(resume, map_location=device)
            if "optimizer_state_dict" in state:
                try:
                    optimizer.load_state_dict(state["optimizer_state_dict"])
                except Exception:
                    pass  # shape mismatch when switching phases

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs - start_epoch, eta_min=lr * 0.01
        )

        for epoch in range(start_epoch, epochs):
            loss = _train_epoch(model, loader, criterion, optimizer, device)
            scheduler.step()
            print(f"  Epoch {epoch + 1}/{epochs}  loss={loss:.4f}")
            if loss < best_loss:
                best_loss = loss
                _save(epoch, loss, is_best=True)
            if (epoch + 1) % save_every == 0:
                _save(epoch, loss, is_best=False, tag=f"epoch_{epoch + 1}")

    print(f"\nBest model (loss={best_loss:.4f}) saved to {best_path}")
    return str(best_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train outline segmentation model (pretrained ResNet-34 or lightweight UNet)"
    )
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument(
        "--model",
        type=str,
        default="pretrained",
        choices=["pretrained", "unet"],
        help="'pretrained' = ResNet-34 backbone (recommended); 'unet' = old lightweight model",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--freeze-epochs", type=int, default=30,
                        help="Epochs to train with frozen encoder before full fine-tuning")
    parser.add_argument("--aug-factor", type=int, default=40,
                        help="Augmented copies per real sample per epoch")
    args = parser.parse_args()

    data_dir = args.data_dir or str(
        Path(__file__).resolve().parent.parent.parent / "training_data"
    )
    run_training(
        data_dir=data_dir,
        output_dir=args.output_dir,
        model_type=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        resume=args.resume,
        save_every=args.save_every,
        freeze_epochs=args.freeze_epochs,
        aug_factor=args.aug_factor,
    )


if __name__ == "__main__":
    main()
