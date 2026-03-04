"""
Train outline extractor parameters from training_data folder.

Correct training data structure (confirmed by user):
- Sample 1: inputs = 1 (1).jpg + 1 (2).jpg  │  CAD target = 1 (1).png
- Sample 2: inputs = 2 (1).jpg + 2 (2).jpg  │  CAD target = 2 (3).jpg
- Sample 3: inputs = 3 (1).jpg + 3 (2).jpg  │  CAD target = 3 (1).png
- Sample 4: inputs = 4 (2).jpg + 4 (3).jpg  │  CAD target = 4 (1).jpg
- Sample 5: input  = 5.jpg                  │  CAD target = 5.pdf

The .png/.pdf/.jpg target files are the clean CAD drawings;
the .jpg input files are the actual smartphone photos of the machine part.
"""

import cv2
import numpy as np
import json
import os
from pathlib import Path
from typing import List, Tuple, Dict, Optional

from src.outline_extractor import OutlineExtractor, OutlineResult, create_extractor_from_params


# Correct mapping: (input photo paths, CAD target path) per sample
def get_training_samples(data_dir: str) -> List[Tuple[List[str], str]]:
    """Return list of (input_photo_paths, cad_target_path) for all 5 samples.

    Input paths  = smartphone photos of the machine part.
    Target path  = clean CAD drawing (PNG / JPG / PDF).

    Mapping confirmed by user:
      1 → inputs: 1 (1).jpg, 1 (2).jpg   target: 1 (1).png
      2 → inputs: 2 (1).jpg, 2 (2).jpg   target: 2 (3).jpg
      3 → inputs: 3 (1).jpg, 3 (2).jpg   target: 3 (1).png
      4 → inputs: 4 (2).jpg, 4 (3).jpg   target: 4 (1).jpg
      5 → input:  5.jpg                   target: 5.pdf
    """
    data_dir = Path(data_dir)
    samples = [
        # Sample 1: 1(1).png is the CAD drawing; 1(1).jpg and 1(2).jpg are photos
        (
            [str(data_dir / "1 (1).jpg"), str(data_dir / "1 (2).jpg")],
            str(data_dir / "1 (1).png"),
        ),
        # Sample 2: 2(3).jpg is the CAD drawing; 2(1).jpg and 2(2).jpg are photos
        (
            [str(data_dir / "2 (1).jpg"), str(data_dir / "2 (2).jpg")],
            str(data_dir / "2 (3).jpg"),
        ),
        # Sample 3: 3(1).png is the CAD drawing; 3(1).jpg and 3(2).jpg are photos
        (
            [str(data_dir / "3 (1).jpg"), str(data_dir / "3 (2).jpg")],
            str(data_dir / "3 (1).png"),
        ),
        # Sample 4: 4(1).jpg is the CAD drawing; 4(2).jpg and 4(3).jpg are photos
        (
            [str(data_dir / "4 (2).jpg"), str(data_dir / "4 (3).jpg")],
            str(data_dir / "4 (1).jpg"),
        ),
        # Sample 5: 5.pdf is the CAD drawing; 5.jpg is the photo
        (
            [str(data_dir / "5.jpg")],
            str(data_dir / "5.pdf"),
        ),
    ]
    out = []
    for inputs, target in samples:
        if not Path(target).exists():
            continue
        existing_inputs = [p for p in inputs if Path(p).exists()]
        if not existing_inputs:
            continue
        out.append((existing_inputs, target))
    return out


def discover_training_pairs(data_dir: str) -> List[Tuple[str, str]]:
    """Return ALL (input_photo, cad_target) pairs — one per input photo.

    With the corrected mapping this yields up to 9 pairs (was 5).
    All photos for a sample share the same target CAD drawing.
    """
    samples = get_training_samples(data_dir)
    pairs: List[Tuple[str, str]] = []
    for input_paths, target_path in samples:
        for inp in input_paths:
            pairs.append((inp, target_path))
    return pairs


def _pdf_to_image(pdf_path: str) -> Optional[np.ndarray]:
    """Render first page of PDF to numpy image (BGR). Returns None if PDF cannot be read."""
    try:
        import fitz
        doc = fitz.open(pdf_path)
        page = doc[0]
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        doc.close()
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        elif img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img
    except Exception:
        pass
    try:
        from pdf2image import convert_from_path
        pages = convert_from_path(pdf_path, first_page=1, last_page=1, dpi=150)
        if pages:
            return cv2.cvtColor(np.array(pages[0]), cv2.COLOR_RGB2BGR)
    except Exception:
        pass
    return None


def get_target_contours(target_path: str) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Load target image (jpg/png) or PDF (render first page). Binarize and return contours."""
    path = Path(target_path)
    img = None
    if path.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
        img = cv2.imread(str(path))
    elif path.suffix.lower() == ".pdf":
        img = _pdf_to_image(str(path))
    if img is None:
        return np.zeros((1, 1), dtype=np.uint8), []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return binary, contours


def contour_to_mask(contour: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """Draw contour filled on mask (1 inside, 0 outside)."""
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 1, -1)
    return mask


def normalize_contour_to_unit(contour: np.ndarray) -> np.ndarray:
    """Scale and shift contour so bbox fits in [0,1] x [0,1]."""
    pts = contour.reshape(-1, 2).astype(np.float64)
    if len(pts) < 3:
        return pts
    mn = pts.min(axis=0)
    mx = pts.max(axis=0)
    span = (mx - mn)
    span[span < 1e-6] = 1.0
    pts = (pts - mn) / span
    return pts


def contour_area_ratio(a: np.ndarray, b: np.ndarray, size: int = 256) -> float:
    """
    Approximate IoU of two contours by rasterizing to size x size.
    Returns overlap ratio (intersection / union).
    """
    a = normalize_contour_to_unit(a)
    b = normalize_contour_to_unit(b)
    if len(a) < 3 or len(b) < 3:
        return 0.0
    mask_a = np.zeros((size, size), dtype=np.uint8)
    mask_b = np.zeros((size, size), dtype=np.uint8)
    aa = (a * (size - 1)).astype(np.int32)
    bb = (b * (size - 1)).astype(np.int32)
    cv2.fillPoly(mask_a, [aa], 1)
    cv2.fillPoly(mask_b, [bb], 1)
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 0.0
    return float(inter) / float(union)


def score_extraction(
    result: OutlineResult,
    target_contours: List[np.ndarray],
    size: int = 256,
) -> float:
    """
    Score how well result.outline matches target (best overlap with largest target contour).
    """
    if target_contours is None or len(target_contours) == 0:
        return 0.0
    if result.outline is None or len(result.outline) < 3:
        return 0.0
    target_largest = max(target_contours, key=cv2.contourArea)
    return contour_area_ratio(result.outline.astype(np.float64), target_largest, size=size)


def grid_search(
    pairs: List[Tuple[str, str]],
    param_grid: Optional[Dict[str, List]] = None,
) -> Tuple[Dict, float]:
    """Simple grid search over extractor params. Returns best params and best score."""
    if param_grid is None:
        param_grid = {
            "canny_low": [35, 50],
            "canny_high": [110, 140],
            "blur_kernel": [5, 7],
            "min_contour_area_ratio": [0.008, 0.015],
            "simplify_epsilon_ratio": [0.006, 0.012],
        }
    keys = list(param_grid.keys())
    best_score = -1.0
    best_params = {}

    def recurse(idx: int, current: Dict) -> None:
        nonlocal best_score, best_params
        if idx == len(keys):
            full = {**defaults, **current}
            valid = ("canny_low", "canny_high", "blur_kernel", "min_contour_area_ratio",
                     "simplify_epsilon_ratio", "max_hole_area_ratio", "max_hole_radius_ratio",
                     "use_hough_circles", "use_adaptive_thresh", "adaptive_block", "adaptive_c")
            ext = OutlineExtractor(**{k: full[k] for k in valid if k in full})
            total = 0.0
            n = 0
            for input_path, target_path in pairs:
                _, target_contours = get_target_contours(target_path)
                try:
                    res = ext.extract(input_path)
                    s = score_extraction(res, target_contours)
                    total += s
                    n += 1
                except Exception:
                    pass
            if n > 0:
                score = total / n
                if score > best_score:
                    best_score = score
                    best_params = dict(current)
            return
        key = keys[idx]
        for v in param_grid[key]:
            current[key] = v
            recurse(idx + 1, current)

    defaults = {
        "canny_low": 40,
        "canny_high": 120,
        "blur_kernel": 5,
        "min_contour_area_ratio": 0.01,
        "simplify_epsilon_ratio": 0.008,
        "max_hole_area_ratio": 0.35,
        "max_hole_radius_ratio": 0.35,
        "use_hough_circles": False,
        "use_adaptive_thresh": True,
        "adaptive_block": 11,
        "adaptive_c": 2,
    }
    current = dict(defaults)
    recurse(0, current)
    return best_params, best_score


def run_training(data_dir: str, output_params_path: Optional[str] = None) -> Dict:
    """
    Discover pairs, run grid search, save best params to training_data/params.json.
    """
    data_dir = Path(data_dir)
    if output_params_path is None:
        output_params_path = data_dir / "params.json"
    pairs = discover_training_pairs(str(data_dir))
    if not pairs:
        print("No training samples found in", data_dir,
              "(expected: 1 (1).jpg + 1 (2).jpg, 2 (1)/(2).jpg + 2 (3).jpg, etc.)")
        return {}
    print(f"Found {len(pairs)} training pairs")
    best_params, best_score = grid_search(pairs)
    print(f"Best IoU score: {best_score:.4f}")
    print("Best params:", best_params)
    with open(output_params_path, "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=2)
    print("Saved to", output_params_path)
    return best_params


if __name__ == "__main__":
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).resolve().parent.parent.parent / "training_data")
    run_training(data_dir)
