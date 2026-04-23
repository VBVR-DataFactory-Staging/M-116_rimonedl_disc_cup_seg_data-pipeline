"""M-116: RIM-ONE DL optic disc and cup segmentation.

Dataset: RIM-ONE DL (485 fundus images: 313 normal + 172 glaucoma).
Each image has reference disc + cup segmentation as SEPARATE PNGs
(one `*-Disc-T.png`, one `*-Cup-T.png` per image) at
    _extracted/M-116_RIM-ONE-DL/data/segmentations/RIM-ONE_DL_reference_segmentations/{glaucoma,normal}/<caseid>-<k>-{Disc,Cup}-T.png

Image root:
    _extracted/M-116_RIM-ONE-DL/data/images/RIM-ONE_DL_images/partitioned_by_*/*/<class>/<caseid>.png

Case D (single fundus image): loop the frame n=fps*4 times → 4-second video.
Overlay: disc (green) + cup (red, nested inside disc).
"""
from __future__ import annotations
import re
from collections import defaultdict
from pathlib import Path
import cv2
import numpy as np
from common import (
    DATA_ROOT, write_task, COLORS, fit_square, overlay_mask,
)


def loop_frames(frame, n: int):
    """Repeat a single frame n times → video frames list."""
    return [frame.copy() for _ in range(n)]

PID = "M-116"
TASK_NAME = "rimonedl_disc_cup_seg"
FPS = 8

PROMPT = (
    "This is a color fundus photograph from the RIM-ONE DL dataset used for glaucoma assessment. "
    "Segment the optic disc (green contour) and optic cup (red contour, nested inside the disc). "
    "The cup-to-disc ratio is clinically important for glaucoma diagnosis."
)


# seg file: e.g.  r1_Im001-1-Disc-T.png  or  r1_Im069-2-Cup-T.png
_SEG_RE = re.compile(r"^(?P<case>.+?)-\d+-(?P<kind>Disc|Cup)-T\.png$", re.IGNORECASE)


def find_pairs(root: Path):
    """Walk .png files; pair each image with its Disc + Cup segmentation PNGs.

    Returns list of (img_path, disc_mask_path, cup_mask_path).
    """
    all_pngs = list(root.rglob("*.png"))
    imgs: dict[str, Path] = {}      # case_id -> image path
    discs: dict[str, Path] = {}     # case_id -> disc mask path
    cups: dict[str, Path] = {}      # case_id -> cup mask path

    for p in all_pngs:
        if p.name.lower().startswith("license"):
            continue
        pstr = str(p).lower().replace("\\", "/")
        is_seg = ("/segmentation" in pstr) or ("reference_segmentation" in pstr)
        if is_seg:
            m = _SEG_RE.match(p.name)
            if not m:
                continue
            case = m.group("case")
            if m.group("kind").lower() == "disc":
                # Prefer first-seen; if multiple versions, keep lexically smallest
                if case not in discs or str(p) < str(discs[case]):
                    discs[case] = p
            else:
                if case not in cups or str(p) < str(cups[case]):
                    cups[case] = p
        elif ("/images/" in pstr) or ("rim-one_dl_images" in pstr):
            case = p.stem  # e.g. r1_Im001
            if case not in imgs:
                imgs[case] = p

    pairs = []
    for case in sorted(imgs.keys()):
        d = discs.get(case)
        c = cups.get(case)
        if d and c:
            pairs.append((imgs[case], d, c))
    return pairs


def _binarize(mask_path: Path) -> np.ndarray | None:
    m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    # Reference masks have FG=dark (per dcseg_to_binary_mask.py: thresh <128 => FG=1).
    # Be robust: whichever polarity has fewer pixels is the object.
    fg_dark = (m < 128).astype(np.uint8)
    fg_light = (m >= 128).astype(np.uint8)
    # Pick the one that represents the plausible object (smaller area, typically <40%)
    if 0 < fg_dark.sum() <= fg_light.sum():
        return fg_dark
    return fg_light


def process_case(img_path: Path, disc_path: Path, cup_path: Path, task_idx: int):
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    disc_mask = _binarize(disc_path)
    cup_mask = _binarize(cup_path)
    if disc_mask is None or cup_mask is None:
        return None

    # Resize masks to image dims if needed
    h, w = img.shape[:2]
    if disc_mask.shape != (h, w):
        disc_mask = cv2.resize(disc_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    if cup_mask.shape != (h, w):
        cup_mask = cv2.resize(cup_mask, (w, h), interpolation=cv2.INTER_NEAREST)

    img_r = fit_square(img, 512)
    disc_r = cv2.resize(disc_mask, (512, 512), interpolation=cv2.INTER_NEAREST)
    cup_r = cv2.resize(cup_mask, (512, 512), interpolation=cv2.INTER_NEAREST)

    # Overlay: disc (green), cup (red) on top
    annotated = overlay_mask(img_r, disc_r, color=COLORS["green"], alpha=0.35)
    annotated = overlay_mask(annotated, cup_r, color=COLORS["red"], alpha=0.5)

    n_frames = FPS * 4
    first_frames = loop_frames(img_r, n=n_frames)
    last_frames = loop_frames(annotated, n=n_frames)
    gt_frames = last_frames  # ground_truth = same as last since static

    disc_area = int(disc_r.sum())
    cup_area = int(cup_r.sum())
    cdr = float(cup_area) / float(disc_area) if disc_area > 0 else 0.0
    pstr = str(img_path).lower()
    split = "glaucoma" if "glaucoma" in pstr else ("normal" if "normal" in pstr else "unknown")

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
    print(f"  {len(pairs)} RIM-ONE DL (image, disc, cup) triples")
    for i, (img, disc, cup) in enumerate(pairs):
        d = process_case(img, disc, cup, i)
        if d:
            print(f"  wrote {d}")


if __name__ == "__main__":
    main()
