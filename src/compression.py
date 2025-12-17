from __future__ import annotations
from pathlib import Path
import csv
import cv2
import numpy as np
import imageio.v2 as imageio


# ----------------------------- Helpers: basic image ops -----------------------------


def to_gray_f32(img_rgb: np.ndarray) -> np.ndarray:
    
    # Convert RGB uint8 image to float32 grayscale in [0,1]
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gray /= 255.0
    return gray


def apply_log_clahe(gray: np.ndarray) -> np.ndarray:
    
    """
    Log + CLAHE on [0,1] gray → uint8 [0,255],
    fixed pre-processing used as "reference" contrast enhancement
    """
    eps = 1e-6
    log_img = np.log1p(gray)
    log_img = (log_img - log_img.min()) / (log_img.max() - log_img.min() + eps)

    log_u8 = (255.0 * log_img).astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_u8 = clahe.apply(log_u8)
    return clahe_u8


def apply_denoiser(gray_u8: np.ndarray) -> np.ndarray:

    return cv2.bilateralFilter(gray_u8, d=5, sigmaColor=20, sigmaSpace=5)


def jpeg_encode_to_bytes(img_bgr: np.ndarray, quality: int) -> bytes:

    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    ok, buf = cv2.imencode(".jpg", img_bgr, encode_params)
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return buf.tobytes()


# ----------------------------- Helpers: PSNR / entropy -----------------------------


def mse_psnr(img_ref: np.ndarray, img_cmp: np.ndarray) -> tuple[float, float]:
    diff = img_ref.astype(np.float32) - img_cmp.astype(np.float32)
    mse = float(np.mean(diff**2))
    if mse == 0:
        return 0.0, float("inf")
    psnr = 10.0 * np.log10((255.0**2) / mse)
    return mse, psnr


def mask_metrics(
    gray_ref: np.ndarray, gray_cmp: np.ndarray, mask_bool: np.ndarray
) -> tuple[float, float]:
    
    # MSE, PSNR restricted to pixels where mask_bool is True
    ref_roi = gray_ref[mask_bool]
    cmp_roi = gray_cmp[mask_bool]
    diff = ref_roi.astype(np.float32) - cmp_roi.astype(np.float32)
    mse = float(np.mean(diff**2))
    if mse == 0:
        return 0.0, float("inf")
    psnr = 10.0 * np.log10((255.0**2) / mse)
    return mse, psnr


def entropy_gray(gray_u8: np.ndarray) -> float:

    hist = cv2.calcHist([gray_u8], [0], None, [256], [0, 256]).ravel()
    p = hist / (hist.sum() + 1e-12)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


# ----------------------------- Helpers: SSIM (Wang–Bovik style) -----------------------------


def ssim_index(
    gray_ref: np.ndarray, gray_cmp: np.ndarray, mask_bool: np.ndarray | None = None
) -> float:
    
    """
    Single-scale SSIM between two uint8 grayscale images.
    Compute a local SSIM map using a Gaussian window, then:
      - if mask_bool is None: return mean over entire image
      - else: return mean over pixels where mask_bool is True
    """
    x = gray_ref.astype(np.float32)
    y = gray_cmp.astype(np.float32)

    if x.shape != y.shape:
        raise ValueError("ssim_index: input shapes do not match")

    L = 255.0
    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    mu_x = cv2.GaussianBlur(x, (11, 11), 1.5)
    mu_y = cv2.GaussianBlur(y, (11, 11), 1.5)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = cv2.GaussianBlur(x * x, (11, 11), 1.5) - mu_x2
    sigma_y2 = cv2.GaussianBlur(y * y, (11, 11), 1.5) - mu_y2
    sigma_xy = cv2.GaussianBlur(x * y, (11, 11), 1.5) - mu_xy

    num = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
    den = (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)

    ssim_map = np.ones_like(den, dtype=np.float32)
    valid = den > 0
    ssim_map[valid] = num[valid] / den[valid]

    if mask_bool is not None:
        mask_bool = mask_bool.astype(bool)
        if ssim_map.shape != mask_bool.shape:
            raise ValueError("ssim_index: mask shape does not match image shape")
        vals = ssim_map[mask_bool]
    else:
        vals = ssim_map.ravel()

    if vals.size == 0:
        return 0.0

    return float(np.mean(vals))


# ----------------------------- Helpers: bitrate-matched quality search -----------------------------


def find_quality_for_target_bytes(
    img_bgr: np.ndarray,
    target_bytes: int,
    q_min: int = 10,
    q_max: int = 95,
    max_iter: int = 8,
    rel_tol: float = 0.01,
) -> tuple[int, int]:
    
    """
    Binary search over JPEG quality to approximate a target file size.
    Assumes that higher JPEG quality produces (on average) more bytes,
    returns (best_quality, best_num_bytes).
    """
    best_q = q_min
    best_bytes = None

    low = q_min
    high = q_max

    for _ in range(max_iter):
        q_mid = int((low + high) // 2)
        jpeg_bytes = jpeg_encode_to_bytes(img_bgr, q_mid)
        n_bytes = len(jpeg_bytes)

        if best_bytes is None or abs(n_bytes - target_bytes) < abs(
            best_bytes - target_bytes
        ):
            best_q = q_mid
            best_bytes = n_bytes

        if target_bytes > 0:
            rel_err = abs(n_bytes - target_bytes) / target_bytes
            if rel_err <= rel_tol:
                break

        if n_bytes < target_bytes:
            low = q_mid + 1
        else:
            high = q_mid - 1

        if low > high:
            break

    if best_bytes is None:
        jpeg_bytes = jpeg_encode_to_bytes(img_bgr, best_q)
        best_bytes = len(jpeg_bytes)

    return best_q, best_bytes


# ----------------------------- Main experiment -----------------------------


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"
    FRAMES_DIR = data_dir / "frames_full"
    MASKS_DIR = data_dir / "masks"
    QMAP_DIR = data_dir / "quality_maps"

    OUT_DIR = data_dir / "jpeg_out"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = data_dir / "compression_metrics.csv"

    if not QMAP_DIR.exists():
        print(f"[ERROR] Quality map dir not found: {QMAP_DIR}")
        return

    qmap_files = sorted(QMAP_DIR.glob("*_qmap.png"))
    print(f"[INFO] Found {len(qmap_files)} quality maps in {QMAP_DIR}")

    header = [
        "base",
        "H",
        "W",
        "baseline_q",
        "roi_q",
        "bytes_baseline",
        "bytes_roi",
        "rel_bytes_diff",
        "cr_baseline",
        "cr_roi",
        "H_bits_ref",
        "H_bits_roi_pre",
        "psnr_global_baseline",
        "psnr_global_roi",
        "psnr_roi_baseline",
        "psnr_roi_roi",
        "ssim_global_baseline",
        "ssim_global_roi",
        "ssim_roi_baseline",
        "ssim_roi_roi",
    ]

    write_header = not csv_path.exists()
    f_csv = csv_path.open("a", newline="")
    writer = csv.writer(f_csv)
    if write_header:
        writer.writerow(header)

    # per-frame processing
    for qmap_path in qmap_files:
        base = qmap_path.name.replace("_qmap.png", "")
        print("\n====================================")
        print(f"[INFO] Processing base: {base}")

        frame_path = FRAMES_DIR / f"{base}.png"
        mask_path = MASKS_DIR / f"{base}_auto_mask.png"

        print(f"  frame: {frame_path}")
        print(f"  mask : {mask_path}")
        print(f"  qmap : {qmap_path}")

        if not frame_path.exists():
            print("  [WARN] frame not found, skipping.")
            continue
        if not mask_path.exists():
            print("  [WARN] mask not found, skipping.")
            continue

        # load frame(reference)
        frame_rgb = imageio.imread(frame_path)
        if frame_rgb.ndim == 2:
            frame_rgb = cv2.cvtColor(frame_rgb, cv2.COLOR_GRAY2RGB)
        elif frame_rgb.shape[-1] == 4:
            frame_rgb = cv2.cvtColor(frame_rgb, cv2.COLOR_RGBA2RGB)
        frame_rgb = frame_rgb.astype(np.uint8)

        H, W, _ = frame_rgb.shape

        # load mask
        mask_u8 = imageio.imread(mask_path)
        if mask_u8.ndim == 3:
            mask_u8 = mask_u8[..., 0]
        mask_bool = mask_u8 > 0

        # load quality map
        qmap = imageio.imread(qmap_path).astype(np.float32) / 255.0
        if qmap.shape != (H, W):
            print("  [WARN] qmap shape mismatch, skipping.")
            continue

        # reference enhanced gray(fixed transform)
        gray_f = to_gray_f32(frame_rgb)
        ref_u8 = apply_log_clahe(gray_f)
        ref_bgr = cv2.cvtColor(ref_u8, cv2.COLOR_GRAY2BGR)

        H_bits_ref = entropy_gray(ref_u8)

        # baseline: mild denoise + JPEG at fixed quality Q_BASE
        Q_BASE = 40
        ref_denoised = apply_denoiser(ref_u8)
        base_bgr = cv2.cvtColor(ref_denoised, cv2.COLOR_GRAY2BGR)

        jpeg_bytes_baseline = jpeg_encode_to_bytes(base_bgr, Q_BASE)
        bytes_baseline = len(jpeg_bytes_baseline)

        buf_arr = np.frombuffer(jpeg_bytes_baseline, dtype=np.uint8)
        baseline_bgr = cv2.imdecode(buf_arr, cv2.IMREAD_COLOR)
        baseline_gray = cv2.cvtColor(baseline_bgr, cv2.COLOR_BGR2GRAY)

        # ROI-aware pre-processing driven by qmap
        sigma_bg = 2.5
        sigma_core = 0.3

        blur_bg = cv2.GaussianBlur(ref_u8, (0, 0), sigmaX=sigma_bg)
        blur_core = cv2.GaussianBlur(ref_u8, (0, 0), sigmaX=sigma_core)

        Q_LO, Q_HI = 0.3, 1.0
        alpha = (qmap - Q_LO) / (Q_HI - Q_LO + 1e-6)
        alpha = np.clip(alpha, 0.0, 1.0)

        # high alpha : closer to blur_core (high-fidelity region)
        # low alpha  : closer to blur_bg   (more aggressively smoothed)
        roi_aware = alpha * blur_core + (1.0 - alpha) * blur_bg
        roi_aware_u8 = np.clip(roi_aware, 0, 255).astype(np.uint8)
        roi_bgr = cv2.cvtColor(roi_aware_u8, cv2.COLOR_GRAY2BGR)

        # entropy after ROI-aware pre-filtering(before JPEG)
        H_bits_roi_pre = entropy_gray(roi_aware_u8)

        # bitrate-matched JPEG quality search for ROI-aware path
        Q_MIN, Q_MAX = 10, 95
        q_roi_opt, _ = find_quality_for_target_bytes(
            roi_bgr,
            target_bytes=bytes_baseline,
            q_min=Q_MIN,
            q_max=Q_MAX,
            max_iter=8,
            rel_tol=0.01,
        )

        jpeg_bytes_roi = jpeg_encode_to_bytes(roi_bgr, q_roi_opt)
        bytes_roi = len(jpeg_bytes_roi)

        rel_bytes_diff = 0.0
        if bytes_baseline > 0:
            rel_bytes_diff = (bytes_roi - bytes_baseline) / bytes_baseline

        # decode ROI-aware JPEG
        buf_arr2 = np.frombuffer(jpeg_bytes_roi, dtype=np.uint8)
        roi_bgr_dec = cv2.imdecode(buf_arr2, cv2.IMREAD_COLOR)
        roi_gray_dec = cv2.cvtColor(roi_bgr_dec, cv2.COLOR_BGR2GRAY)

        # write out JPEGs to inspect visually
        out_baseline_path = OUT_DIR / f"{base}_baseline_q{Q_BASE}.jpg"
        out_roi_path = OUT_DIR / f"{base}_roi_q{q_roi_opt}.jpg"
        out_baseline_path.write_bytes(jpeg_bytes_baseline)
        out_roi_path.write_bytes(jpeg_bytes_roi)

        _, psnr_global_baseline = mse_psnr(ref_denoised, baseline_gray)

        _, psnr_global_roi = mse_psnr(roi_aware_u8, roi_gray_dec)

        _, psnr_roi_baseline = mask_metrics(ref_denoised, baseline_gray, mask_bool)
        _, psnr_roi_roi = mask_metrics(roi_aware_u8, roi_gray_dec, mask_bool)

        ssim_global_baseline = ssim_index(ref_denoised, baseline_gray, mask_bool=None)
        ssim_global_roi = ssim_index(roi_aware_u8, roi_gray_dec, mask_bool=None)

        ssim_roi_baseline = ssim_index(ref_denoised, baseline_gray, mask_bool=mask_bool)
        ssim_roi_roi = ssim_index(roi_aware_u8, roi_gray_dec, mask_bool=mask_bool)

        writer.writerow(
            [
                base,
                H,
                W,
                Q_BASE,
                q_roi_opt,
                bytes_baseline,
                bytes_roi,
                rel_bytes_diff,
                cr_baseline := (H * W * 8) / (bytes_baseline * 8.0),
                cr_roi := (H * W * 8) / (bytes_roi * 8.0),
                H_bits_ref,
                H_bits_roi_pre,
                psnr_global_baseline,
                psnr_global_roi,
                psnr_roi_baseline,
                psnr_roi_roi,
                ssim_global_baseline,
                ssim_global_roi,
                ssim_roi_baseline,
                ssim_roi_roi,
            ]
        )
        print(
            f"  [OK] base={base} "
            f"Q_roi={q_roi_opt} "
            f"bytes_baseline={bytes_baseline} bytes_roi={bytes_roi} "
            f"rel_diff={rel_bytes_diff:.3%} "
            f"psnr_roi_base={psnr_roi_baseline:.2f} "
            f"psnr_roi_roi={psnr_roi_roi:.2f}"
        )

    f_csv.close()
    print(f"\n[INFO] Done. Metrics in: {csv_path}")


if __name__ == "__main__":
    main()
