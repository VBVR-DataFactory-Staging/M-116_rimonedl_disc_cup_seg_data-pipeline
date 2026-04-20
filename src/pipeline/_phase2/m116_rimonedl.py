"""M-116: RIM-ONE DL optic disc and cup segmentation.

Dataset: RIM-ONE DL (485 fundus images: 313 normal + 172 glaucoma).
Each image has reference disc + cup segmentation in PNG form.

Raw layout (after download via bit.ly/rim-one-dl-{images,reference-segmentations}):
    _extracted/M-116_RIM-ONE-DL/data/images/<split>/<class>/*.png
    _extracted/M-116_RIM-ONE-DL/data/segmentations/<split>/<class>/*.png  (PNG with disc/cup gray levels)

Case D (single fundus image): loop the frame n=fps*4 times → 4-second video.
Overlay: disc (green) + cup (red, inside disc).
"""
from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np
from common import (
    DATA_ROOT, write_task, COLORS, fit_square, overlay_mask, loop_frames,
)

PID = "M-116"
TASK_NAME = "rimonedl_disc_cup_seg"
FPS = 8

# PNG mask gray values in RIM-ONE DL reference segs:
#   disc region   ≈ 128 (mid gray)
#   cup region    ≈ 255 (white)
#   background    ≈ 0
# (per github.com/miag-ull/rim-one-dl/dcseg_to_binary_mask.py)
DISC_GRAY_MIN = 64
CUP_GRAY_MIN = 192

PROMPT = (
    "This is a color fundus photograph from the RIM-ONE DL dataset used for glaucoma assessment. "
    "Segment the optic disc (green contour) and optic cup (red contour, nested inside the disc). "
    "The cup-to-disc ratio is clinically important for glaucoma diagnosis."
)


def find_pairs(root: Path):
    """Find (image_path, mask_path) pairs across train/test and normal/glaucoma splits."""
    pairs = []
    img_root = root / "images"
    seg_root = root / "segmentations"
    if not img_root.exists():
        # Some layouts put things under rim-one-dl-master/ or similar
        for candidate in root.rglob("*.png"):
            if "segmentation" in str(candidate).lower() or "mask" in str(candidate).lower():
                continue
            # Attempt matching mask under the same relative path under segmentations/
            rel = candidate.relative_to(root)
            mask = seg_root / rel if seg_root.exists() else None
            if mask and mask.exists():
                pairs.append((candidate, mask))
        return pairs

    for img_path in sorted(img_root.rglob("*.png")):
        # Mirror the path under segmentations/
        rel = img_path.relative_to(img_root)
        mask_path = seg_root / rel
        if not mask_path.exists():
            # Try with .jpg → .png swaps or variant naming
            alt = seg_root / rel.with_suffix(".png")
            if alt.exists():
                mask_path = alt
            else:
                continue
        pairs.append((img_path, mask_path))
    return pairs


def process_case(img_path: Path, mask_path: Path, task_idx: int):
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    mask_gray = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask_gray is None:
        return None

    # Resize to equal dimensions if needed
    if mask_gray.shape[:2] != img.shape[:2]:
        mask_gray = cv2.resize(mask_gray, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Extract disc + cup binary masks from gray levels
    disc_mask = (mask_gray >= DISC_GRAY_MIN).astype(np.uint8)
    cup_mask = (mask_gray >= CUP_GRAY_MIN).astype(np.uint8)

    img_r = fit_square(img, 512)
    disc_r = cv2.resize(disc_mask, (512, 512), interpolation=cv2.INTER_NEAREST)
    cup_r = cv2.resize(cup_mask, (512, 512), interpolation=cv2.INTER_NEAREST)

    # First frame: plain fundus
    # Final frame: overlay disc (green) + cup (red)
    annotated = overlay_mask(img_r, disc_r, color=COLORS["green"], alpha=0.35)
    annotated = overlay_mask(annotated, cup_r, color=COLORS["red"], alpha=0.5)

    # Video = loop annotated n=fps*4 (4 sec at fps=8 → 32 frames)
    n_frames = FPS * 4
    first_frames = loop_frames(img_r, n=n_frames)
    last_frames = loop_frames(annotated, n=n_frames)
    gt_frames = last_frames  # ground_truth = same as last since static

    # Compute CDR for metadata
    disc_area = int(disc_r.sum())
    cup_area = int(cup_r.sum())
    cdr = float(cup_area) / float(disc_area) if disc_area > 0 else 0.0
    split = "glaucoma" if "glaucoma" in str(img_path).lower() else ("normal" if "normal" in str(img_path).lower() else "unknown")

    meta = {
        "task": "RIM-ONE DL optic disc and cup segmentation",
        "dataset": "RIM-ONE DL",
        "case_id": img_path.stem,
        "modality": "color fundus photography",
        "classes": ["optic_disc", "optic_cup"],
        "colors": {"optic_disc": "green", "optic_cup": "red"},
        "fps": FPS,
        "frames_per_video": n_frames,
        "case_type": "D_single_image_loop",
        "source_split": split,
        "disc_area_px": disc_area,
        "cup_area_px": cup_area,
        "cup_to_disc_ratio": round(cdr, 4),
    }
    return write_task(PID, TASK_NAME, task_idx,
                      img_r, annotated,
                      first_frames, last_frames, gt_frames,
                      PROMPT, meta, FPS)


def main():
    root = DATA_ROOT / "_extracted" / "M-116_RIM-ONE-DL" / "data"
    pairs = find_pairs(root)
    print(f"  {len(pairs)} RIM-ONE DL (image, mask) pairs")
    for i, (img, mask) in enumerate(pairs):
        d = process_case(img, mask, i)
        if d:
            print(f"  wrote {d}")


if __name__ == "__main__":
    main()
