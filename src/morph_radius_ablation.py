from pathlib import Path
from typing import List, Tuple
import cv2
import numpy as np
import pandas as pd


# ------------------ Config ------------------

# radius pairs to test: (r_core, r_outer)
RADIUS_PAIRS: List[Tuple[int, int]] = [
    (1, 3),
    (2, 4),
    (3, 6),
]


# ----------------------------- Helpers -----------------------------


def make_disk(radius: int) -> np.ndarray:
    # create a disk-shaped structuring element with given radius

    k = 2 * radius + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def clean_mask(mask_u8: np.ndarray) -> np.ndarray:

    m = (mask_u8 > 0).astype(np.uint8) * 255

    # morphological close and open with 3x3
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k3)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k3)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if num_labels <= 1:
        return m

    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_idx = 1 + int(np.argmax(areas))

    clean = np.zeros_like(m)
    clean[labels == largest_idx] = 255
    return clean


def measure_perimeter(mask_u8: np.ndarray) -> float:
    # approx perimeter length of the largest contour in the mask
    m = (mask_u8 > 0).astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return 0.0
    c = max(contours, key=cv2.contourArea)
    return float(cv2.arcLength(c, True))


# ----------------------------- Main experiment -----------------------------


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"
    mask_dir = data_dir / "masks"
    out_csv = data_dir / "morph_radius_ablation.csv"

    if not mask_dir.exists():
        print(f"[ERROR] Mask directory not found: {mask_dir}")
        return

    mask_files = sorted(mask_dir.glob("*.png"))
    if not mask_files:
        print(f"[ERROR] No PNG masks found in {mask_dir}")
        return

    print(f"[INFO] Found {len(mask_files)} masks in {mask_dir}")

    rows = []

    for mask_path in mask_files:
        print(f"\n[INFO] Processing {mask_path.name}")
        mask_u8 = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask_u8 is None:
            print("  [WARN] Could not read mask, skipping.")
            continue

        mask_clean = clean_mask(mask_u8)
        area_mask = int(np.count_nonzero(mask_clean))

        if area_mask == 0:
            print("  [WARN] Cleaned mask empty, skipping radii.")
            continue

        perim_mask = measure_perimeter(mask_clean)

        for r_core, r_outer in RADIUS_PAIRS:
            # core = erosion with r_core
            if r_core > 0:
                se_core = make_disk(r_core)
                core = cv2.erode(mask_clean, se_core)
            else:
                core = mask_clean.copy()

            # outer = dilation with r_outer
            if r_outer > 0:
                se_outer = make_disk(r_outer)
                outer = cv2.dilate(mask_clean, se_outer)
            else:
                outer = mask_clean.copy()

            # band = outer \ core
            band = outer.copy()
            band[core > 0] = 0

            area_core = int(np.count_nonzero(core))
            area_outer = int(np.count_nonzero(outer))
            area_band = int(np.count_nonzero(band))

            core_frac = area_core / area_mask
            outer_frac = area_outer / area_mask
            band_frac = area_band / area_mask

            perim_core = measure_perimeter(core)
            perim_outer = measure_perimeter(outer)

            rows.append(
                {
                    "filename": mask_path.name,
                    "r_core": r_core,
                    "r_outer": r_outer,
                    "area_mask": area_mask,
                    "area_core": area_core,
                    "area_outer": area_outer,
                    "area_band": area_band,
                    "core_frac": core_frac,
                    "outer_frac": outer_frac,
                    "band_frac": band_frac,
                    "perim_mask": perim_mask,
                    "perim_core": perim_core,
                    "perim_outer": perim_outer,
                }
            )

    if not rows:
        print("[ERROR] No rows collected; nothing to save.")
        return

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"\n[INFO] Saved radius ablation stats → {out_csv}")


if __name__ == "__main__":
    main()
