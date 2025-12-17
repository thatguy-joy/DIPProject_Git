from pathlib import Path
import csv
import cv2
import numpy as np
import imageio.v2 as imageio

# ------------------------- Config -------------------------

DATA_DIR = Path("data")
FRAMES_DIR = DATA_DIR / "frames_full"
MASKS_DIR = DATA_DIR / "masks"
WEDGE_DIR = DATA_DIR / "detect_calipers_output"
OUT_CSV = DATA_DIR / "denoising_metrics.csv"

# intensity transforms and denoisers to test
TRANSFORMS = ["baseline", "clahe", "gamma0.6_clahe", "log_clahe"]
DENOISERS = ["none", "median3", "log_gaussian", "bilateral"]


# ------------------------------------- Helper: intensity transforms -------------------------------------


def _clahe_from_float01(img01: np.ndarray) -> np.ndarray:
    # Apply CLAHE to a [0,1] float image, return [0,1] float

    img01 = np.clip(img01, 0.0, 1.0).astype(np.float32)
    img_u8 = (img01 * 255.0).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    out_u8 = clahe.apply(img_u8)
    return out_u8.astype(np.float32) / 255.0


def apply_intensity_transform(img01: np.ndarray, name: str) -> np.ndarray:

    img01 = np.clip(img01, 0.0, 1.0).astype(np.float32)

    if name == "baseline":
        return img01

    if name == "clahe":
        return _clahe_from_float01(img01)

    if name == "gamma0.6_clahe":
        gamma = 0.6
        gamma_img = np.power(img01, gamma)
        return _clahe_from_float01(gamma_img)

    if name == "log_clahe":
        # log(1+r) scaled to ~[0,1]
        eps = 1e-6
        log_img = np.log1p(img01) / np.log(2.0 + eps)
        return _clahe_from_float01(log_img)

    raise ValueError(f"Unknown transform: {name}")


# -------------------------------------- Helper: denoisers --------------------------------------


def apply_denoiser(img01: np.ndarray, name: str) -> np.ndarray:
    """
    img01: float32 in [0,1], single-channel.
    name: 'none', 'median3', 'log_gaussian', or 'bilateral'
    """
    img01 = np.clip(img01, 0.0, 1.0).astype(np.float32)

    if name == "none":
        return img01

    if name == "median3":
        img_u8 = (img01 * 255.0).astype(np.uint8)
        out_u8 = cv2.medianBlur(img_u8, 5)
        return out_u8.astype(np.float32) / 255.0

    if name == "log_gaussian":
        eps = 1e-6
        logI = np.log(img01 + eps)
        logI_blur = cv2.GaussianBlur(logI, (7, 7), 0)
        out = np.exp(logI_blur) - eps
        return np.clip(out, 0.0, 1.0).astype(np.float32)

    if name == "bilateral":
        img_u8 = (img01 * 255.0).astype(np.uint8)
        out_u8 = cv2.bilateralFilter(img_u8, d=9, sigmaColor=75, sigmaSpace=75)
        return out_u8.astype(np.float32) / 255.0

    raise ValueError(f"Unknown denoiser: {name}")


# -------------------------------------- Helper: ROI / background stats --------------------------------------


def compute_roi_bg_stats(
    img01: np.ndarray,
    lesion_mask: np.ndarray,
    wedge_mask: np.ndarray | None,
    min_pixels: int = 20,
):

    roi_mask = lesion_mask.astype(bool)
    if roi_mask.sum() < min_pixels:
        return None

    H, W = roi_mask.shape

    if wedge_mask is None:
        wedge = np.ones_like(roi_mask, dtype=bool)
    else:
        wedge = wedge_mask.astype(bool)

    lesion_u8 = roi_mask.astype(np.uint8) * 255

    # outer / inner kernels ~ radii 10 and 5 pixels
    k_outer = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    k_inner = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))

    dil_outer = cv2.dilate(lesion_u8, k_outer)
    dil_inner = cv2.dilate(lesion_u8, k_inner)

    ring_mask = (dil_outer > 0) & ~(dil_inner > 0)

    bg_mask = ring_mask & wedge & (~roi_mask)

    if bg_mask.sum() < min_pixels:
        bg_mask = wedge & (~roi_mask)

    if bg_mask.sum() < min_pixels:
        return None

    roi_vals = img01[roi_mask]
    bg_vals = img01[bg_mask]

    mu_roi = float(roi_vals.mean())
    mu_bg = float(bg_vals.mean())
    sigma_roi = float(roi_vals.std(ddof=0))
    sigma_bg = float(bg_vals.std(ddof=0))

    contrast = abs(mu_roi - mu_bg)
    denom = np.sqrt(sigma_roi**2 + sigma_bg**2 + 1e-8)
    cnr = contrast / denom

    return (
        mu_roi,
        mu_bg,
        sigma_roi,
        sigma_bg,
        contrast,
        cnr,
        int(roi_mask.sum()),
        int(bg_mask.sum()),
    )


# --------------------------- Main loop: images × denoisers × transforms ---------------------------


def main():
    if not MASKS_DIR.exists():
        raise FileNotFoundError(f"Mask directory not found: {MASKS_DIR}")
    if not FRAMES_DIR.exists():
        raise FileNotFoundError(f"Frames directory not found: {FRAMES_DIR}")

    mask_files = sorted(MASKS_DIR.glob("*_auto_mask.png"))
    if not mask_files:
        print(f"No *_auto_mask.png files found in {MASKS_DIR}")
        return

    header = [
        "image_stem",
        "transform",
        "denoiser",
        "mu_roi",
        "mu_bg",
        "sigma_roi",
        "sigma_bg",
        "contrast",
        "cnr",
        "n_roi",
        "n_bg",
    ]

    write_header = not OUT_CSV.exists()

    with OUT_CSV.open("w", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)

        for mask_path in mask_files:
            stem = mask_path.stem
            if stem.endswith("_auto_mask"):
                frame_stem = stem[:-10]
            else:
                frame_stem = stem

            frame_path = FRAMES_DIR / f"{frame_stem}.png"
            wedge_path = WEDGE_DIR / f"{frame_stem}_wedge_mask.png"

            if not frame_path.exists():
                print(
                    f"[WARN] Missing frame for mask {mask_path.name} -> {frame_path} not found, skipping."
                )
                continue

            # load frame
            img = imageio.imread(frame_path)
            if img.ndim == 3:
                img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            else:
                img_gray = img
            img01 = img_gray.astype(np.float32) / 255.0

            # load lesion mask
            mask_u8 = imageio.imread(mask_path)
            lesion_mask = mask_u8 > 0

            # load wedge mask
            if wedge_path.exists():
                wedge_u8 = imageio.imread(wedge_path)
                wedge_mask = wedge_u8 > 0
            else:
                wedge_mask = None

            for d_name in DENOISERS:
                img_denoised = apply_denoiser(img01, d_name)

                for t_name in TRANSFORMS:
                    img_final = apply_intensity_transform(img_denoised, t_name)

                    stats = compute_roi_bg_stats(img_final, lesion_mask, wedge_mask)
                    if stats is None:
                        print(
                            f"[WARN] Not enough ROI/BG pixels for {frame_stem}, {t_name}, {d_name} - skipping row."
                        )
                        continue

                    mu_roi, mu_bg, sigma_roi, sigma_bg, contrast, cnr, n_roi, n_bg = (
                        stats
                    )

                    row = [
                        frame_stem,
                        t_name,
                        d_name,
                        mu_roi,
                        mu_bg,
                        sigma_roi,
                        sigma_bg,
                        contrast,
                        cnr,
                        n_roi,
                        n_bg,
                    ]
                    writer.writerow(row)

    print(f"Saved denoising metrics to: {OUT_CSV}")


if __name__ == "__main__":
    main()
