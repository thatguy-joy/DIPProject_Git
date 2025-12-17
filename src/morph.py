from pathlib import Path
import csv
import numpy as np
import cv2
import imageio.v2 as imageio

MASKS_DIR = Path("data/masks")
OUT_ROOT = Path("data/morph_masks")

CORE_DIR = OUT_ROOT / "core"
OUTER_DIR = OUT_ROOT / "outer"
BAND_DIR = OUT_ROOT / "band"
CSV_PATH = OUT_ROOT / "morph_stats.csv"

for d in [OUT_ROOT, CORE_DIR, OUTER_DIR, BAND_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ----------------------------- Helpers -----------------------------


def make_disk(radius: int) -> np.ndarray:
    # (2r+1, 2r+1) elliptical structuring element (approx)

    k = 2 * radius + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def clean_and_largest(mask_u8: np.ndarray) -> np.ndarray:
    """
    Basic cleanup:
      - close + open with 3x3
      - keep largest connected component
    Input: uint8 mask (0/255).
    Returns: uint8 mask (0/255) of cleaned main lesion.
    """

    kernel3 = np.ones((3, 3), np.uint8)
    m = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel3, iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel3, iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if num_labels <= 1:
        return m

    areas = stats[1:, cv2.CC_STAT_AREA]
    best_label = 1 + np.argmax(areas)
    cleaned = np.zeros_like(m)
    cleaned[labels == best_label] = 255
    return cleaned


# ----------------------------------- Process a single mask -----------------------------------


def process_mask(mask_path: Path):
    """
    For one auto mask:
      - clean it
      - compute equivalent-disc radius
      - choose r_core, r_outer from R_eq
      - build core / outer / band masks
      - save them + return stats row
    """
    mask = imageio.imread(mask_path)
    if mask.ndim == 3:
        mask = mask[..., 0]
    mask_u8 = (mask > 0).astype(np.uint8) * 255

    # cleanup + keep largest lesion
    mask_clean = clean_and_largest(mask_u8)
    M = mask_clean > 0
    area_mask = int(M.sum())
    if area_mask == 0:
        print(f"[WARN] {mask_path.name}: empty after cleanup, skipping.")
        return None

    # equivalent-radius based radii
    # equivalent radius of a disc: area = pi * R_eq^2  ->  R_eq = sqrt(area / pi)
    R_eq = float(np.sqrt(area_mask / np.pi))

    # r_core  = max(1, round(0.2 * R_eq))
    # r_outer = max(r_core + 1, round(0.35 * R_eq))
    # r_outer = min(r_outer, 7)

    r_core = max(1, int(round(0.2 * R_eq)))
    r_outer = max(r_core + 1, int(round(0.35 * R_eq)))
    r_outer = min(r_outer, 7)

    # safety net: ensure r_outer > r_core even after clipping.
    if r_outer <= r_core:
        # clamp core down if needed so outer can still be > core

        r_core = max(1, min(r_core, r_outer - 1))

    print(
        f"[INFO] {mask_path.name}: area={area_mask}, "
        f"R_eq≈{R_eq:.2f}, r_core={r_core}, r_outer={r_outer}"
    )

    # ------------------------------- Build core / outer / band -------------------------------

    M_u8 = M.astype(np.uint8) * 255

    if r_core > 0:
        se_core = make_disk(r_core)
        core_u8 = cv2.erode(M_u8, se_core, iterations=1)
    else:
        core_u8 = M_u8.copy()
    M_core = core_u8 > 0

    if r_outer > 0:
        se_outer = make_disk(r_outer)
        outer_u8 = cv2.dilate(M_u8, se_outer, iterations=1)
    else:
        outer_u8 = M_u8.copy()
    M_outer = outer_u8 > 0

    M_band = np.logical_and(M_outer, np.logical_not(M_core))

    area_core = int(M_core.sum())
    area_outer = int(M_outer.sum())
    area_band = int(M_band.sum())

    core_frac = area_core / area_mask
    outer_frac = area_outer / area_mask
    band_frac = area_band / area_mask

    stem = mask_path.stem

    core_out_path = CORE_DIR / f"{stem}_core.png"
    outer_out_path = OUTER_DIR / f"{stem}_outer.png"
    band_out_path = BAND_DIR / f"{stem}_band.png"

    imageio.imwrite(core_out_path, (M_core.astype(np.uint8) * 255))
    imageio.imwrite(outer_out_path, (M_outer.astype(np.uint8) * 255))
    imageio.imwrite(band_out_path, (M_band.astype(np.uint8) * 255))

    return {
        "filename": mask_path.name,
        "area_mask": area_mask,
        "area_core": area_core,
        "area_outer": area_outer,
        "core_frac": core_frac,
        "outer_frac": outer_frac,
        "band_frac": band_frac,
        "R_eq": R_eq,
        "r_core": r_core,
        "r_outer": r_outer,
    }


# -------------- main ---------------


def main():
    mask_files = sorted(MASKS_DIR.glob("*.png"))
    print(f"[INFO] Found {len(mask_files)} masks in {MASKS_DIR}")

    rows = []
    for mpath in mask_files:
        print("\n====================================")
        print(f"[PROCESS] {mpath.name}")
        row = process_mask(mpath)
        if row is not None:
            rows.append(row)

    if not rows:
        print("[WARN] No valid masks processed. Nothing to write.")
        return

    fieldnames = [
        "filename",
        "area_mask",
        "area_core",
        "area_outer",
        "core_frac",
        "outer_frac",
        "band_frac",
        "R_eq",
        "r_core",
        "r_outer",
    ]

    with CSV_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print("\n====================================")
    print(f"[DONE] Wrote stats to {CSV_PATH}")
    print(f"       Core masks  → {CORE_DIR}")
    print(f"       Outer masks → {OUTER_DIR}")
    print(f"       Band masks  → {BAND_DIR}")


if __name__ == "__main__":
    main()
