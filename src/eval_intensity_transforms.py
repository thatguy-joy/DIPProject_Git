from pathlib import Path
import csv
import math
import cv2
import numpy as np
import imageio.v2 as imageio

from detect_calipers import get_wedge_mask

DATA_DIR = Path(r"X:\DIP-Project\data")
PLAUS_CSV = DATA_DIR / "lesion_measurements_plausibility.csv"
MASKS_DIR = DATA_DIR / "masks"
OUT_CSV = DATA_DIR / "intensity_transform_metrics.csv"


# ------------------------------ Intensity transforms ------------------------------
def to_float01(gray_u8: np.ndarray) -> np.ndarray:
    # Convert uint8 [0,255] to float32 [0,1]
    return gray_u8.astype(np.float32) / 255.0


def to_uint8(gray_f: np.ndarray) -> np.ndarray:
    # Convert float [0,1] to uint8 [0,255] with clipping
    g = np.clip(gray_f, 0.0, 1.0)
    return (g * 255.0 + 0.5).astype(np.uint8)


def apply_clahe(
    gray_f: np.ndarray, clip_limit: float = 2.0, tile_grid_size=(8, 8)
) -> np.ndarray:
    """
    CLAHE on float [0,1]. Returns float [0,1].
    """
    gray_u8 = to_uint8(gray_f)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    out_u8 = clahe.apply(gray_u8)
    return to_float01(out_u8)


def transform_baseline(gray_f: np.ndarray) -> np.ndarray:
    return gray_f.copy()


def transform_gamma(gray_f: np.ndarray, gamma: float = 0.6) -> np.ndarray:
    """
    gamma transform: s = r^gamma, r in [0,1].
    gamma < 1 brightens dark regions.
    """
    g = np.clip(gray_f, 0.0, 1.0)
    return np.power(g, gamma, dtype=np.float32)


def transform_log(gray_f: np.ndarray) -> np.ndarray:
    """
    log transform: s = c * log(1 + r), r in [0,1].
    We normalize back to [0,1].
    """
    g = np.clip(gray_f, 0.0, 1.0)
    s = np.log1p(g)  # log(1 + r)
    s_min, s_max = s.min(), s.max()
    if s_max > s_min:
        s = (s - s_min) / (s_max - s_min)
    else:
        s = np.zeros_like(s, dtype=np.float32)
    return s.astype(np.float32)


def get_transforms():

    return {
        "baseline": lambda g: transform_baseline(g),
        "clahe": lambda g: apply_clahe(g),
        "gamma0.6_clahe": lambda g: apply_clahe(transform_gamma(g, gamma=0.6)),
        "log_clahe": lambda g: apply_clahe(transform_log(g)),
    }


# ------------------------ Background definition ------------------------


def compute_roi_and_ring(
    gray_u8: np.ndarray,
    mask_u8: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Given:
        gray_u8: (H,W) uint8 grayscale
        mask_u8: (H,W) uint8, 0/255 lesion

    Returns:
        roi_mask: bool array for lesion
        ring_mask: bool array for background ring around lesion inside wedge
    """
    H, W = gray_u8.shape

    # lesion ROI
    roi_mask = mask_u8 > 0

    if not roi_mask.any():
        # empty lesion → return empty masks
        return roi_mask, np.zeros_like(roi_mask, dtype=bool)

    # wedge mask using same helper as detect_calipers
    wedge_u8 = get_wedge_mask(gray_u8)
    wedge_bool = wedge_u8 > 0

    # we want a ring around the lesion
    roi_u8 = roi_mask.astype(np.uint8) * 255

    # structuring element
    radius = 15  # we can tune this accordingly
    k = 2 * radius + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    dilated = cv2.dilate(roi_u8, kernel, iterations=1) > 0

    ring_mask = dilated & (~roi_mask) & wedge_bool

    # if ring is too small, fall back to wedge minus ROI
    if ring_mask.sum() < 100:
        ring_mask = wedge_bool & (~roi_mask)

    return roi_mask, ring_mask


# ------------------------ Metrics on ROI / background ------------------------


def metrics_for_masks(
    gray_f: np.ndarray, roi_mask: np.ndarray, bg_mask: np.ndarray
) -> dict:
    """
    gray_f: (H,W) float32 [0,1]
    roi_mask, bg_mask: bool

    Returns:
        dict with mu/ sigma for ROI and BG, contrast, CNR.
    """
    eps = 1e-8

    roi_vals = gray_f[roi_mask]
    bg_vals = gray_f[bg_mask]

    if roi_vals.size == 0 or bg_vals.size == 0:
        return {
            "mu_roi": math.nan,
            "sigma_roi": math.nan,
            "mu_bg": math.nan,
            "sigma_bg": math.nan,
            "contrast": math.nan,
            "cnr": math.nan,
            "n_roi": int(roi_vals.size),
            "n_bg": int(bg_vals.size),
        }

    mu_roi = float(roi_vals.mean())
    sigma_roi = float(roi_vals.std())

    mu_bg = float(bg_vals.mean())
    sigma_bg = float(bg_vals.std())

    diff = abs(mu_roi - mu_bg)
    denom = math.sqrt(sigma_roi**2 + sigma_bg**2 + eps)
    cnr = diff / denom

    return {
        "mu_roi": mu_roi,
        "sigma_roi": sigma_roi,
        "mu_bg": mu_bg,
        "sigma_bg": sigma_bg,
        "contrast": diff,
        "cnr": cnr,
        "n_roi": int(roi_vals.size),
        "n_bg": int(bg_vals.size),
    }


# ------------------- Main loop over plausible lesions -------------------


def main():
    print(f"Reading plausibility CSV: {PLAUS_CSV}")
    if not PLAUS_CSV.exists():
        raise FileNotFoundError(f"File not found: {PLAUS_CSV}")

    transforms = get_transforms()

    with PLAUS_CSV.open("r", newline="") as f_in, OUT_CSV.open(
        "w", newline=""
    ) as f_out:
        reader = csv.DictReader(f_in)

        fieldnames = [
            "image_path",
            "filename",
            "area_plausibility",
            "transform",
            "mu_roi",
            "sigma_roi",
            "mu_bg",
            "sigma_bg",
            "contrast",
            "cnr",
            "n_roi",
            "n_bg",
        ]
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        n_total = 0
        n_plausible = 0
        n_written = 0

        for row in reader:
            n_total += 1
            plaus = row.get("area_plausibility", "")

            if plaus != "plausible":
                continue

            n_plausible += 1

            img_path = Path(row["image_path"])
            if not img_path.exists():
                print(f"[WARN] Image not found, skip: {img_path}")
                continue

            stem = img_path.stem
            mask_path = MASKS_DIR / f"{stem}_auto_mask.png"
            if not mask_path.exists():
                print(f"[WARN] Mask not found, skip: {mask_path}")
                continue

            # load image + mask
            img = imageio.imread(img_path)
            if img.ndim == 3:
                gray_u8 = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            else:
                gray_u8 = img.astype(np.uint8)

            mask_u8 = imageio.imread(mask_path)
            if mask_u8.ndim == 3:
                mask_u8 = cv2.cvtColor(mask_u8, cv2.COLOR_RGB2GRAY)

            roi_mask, ring_mask = compute_roi_and_ring(gray_u8, mask_u8)
            if not roi_mask.any():
                print(f"[WARN] Empty ROI for {img_path}, skip.")
                continue

            gray_f = to_float01(gray_u8)

            # apply each transform and compute metrics
            for name, fn in transforms.items():
                g_t = fn(gray_f)
                m = metrics_for_masks(g_t, roi_mask, ring_mask)

                out_row = {
                    "image_path": str(img_path),
                    "filename": img_path.name,
                    "area_plausibility": plaus,
                    "transform": name,
                    "mu_roi": m["mu_roi"],
                    "sigma_roi": m["sigma_roi"],
                    "mu_bg": m["mu_bg"],
                    "sigma_bg": m["sigma_bg"],
                    "contrast": m["contrast"],
                    "cnr": m["cnr"],
                    "n_roi": m["n_roi"],
                    "n_bg": m["n_bg"],
                }
                writer.writerow(out_row)
                n_written += 1

    print(f"Total rows in plausibility CSV : {n_total}")
    print(f"Plausible lesions used         : {n_plausible}")
    print(f"Metric rows written            : {n_written}")
    print(f"Output CSV                     : {OUT_CSV}")


if __name__ == "__main__":
    main()
