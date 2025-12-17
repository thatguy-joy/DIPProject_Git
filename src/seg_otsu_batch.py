from pathlib import Path
import cv2
import numpy as np
import imageio.v2 as imageio

from detect_calipers import detect_caliper_seeds

FRAMES_DIR = Path("data/frames_full")
OUT_DIR = Path("data/otsu_masks")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def otsu_local_lesion(
    gray01: np.ndarray,
    sx: int,
    sy: int,
    pixel_dist: float,
    wedge_mask: np.ndarray | None = None,
    min_radius: int = 20,
    max_radius: int | None = None,
) -> np.ndarray:

    H, W = gray01.shape

    # local radius R(scaled by caliper distance)
    R = int(0.6 * pixel_dist)
    R = max(R, min_radius)
    if max_radius is not None:
        R = min(R, max_radius)

    x0 = max(sx - R, 0)
    x1 = min(sx + R, W)
    y0 = max(sy - R, 0)
    y1 = min(sy + R, H)

    if x1 <= x0 or y1 <= y0:
        raise ValueError("Invalid crop box for Otsu")

    gray_crop = gray01[y0:y1, x0:x1]
    crop_u8 = (np.clip(gray_crop, 0.0, 1.0) * 255.0).astype(np.uint8)

    _, th = cv2.threshold(crop_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    seed_cx = sx - x0
    seed_cy = sy - y0
    if not (0 <= seed_cx < th.shape[1] and 0 <= seed_cy < th.shape[0]):
        raise ValueError("Seed lies outside Otsu crop")

    if th[seed_cy, seed_cx] == 0:
        th = cv2.bitwise_not(th)

    num_labels, labels = cv2.connectedComponents(th)
    seed_label = labels[seed_cy, seed_cx]

    mask_crop = np.zeros_like(th)
    mask_crop[labels == seed_label] = 255

    kernel = np.ones((3, 3), np.uint8)
    mask_crop = cv2.morphologyEx(mask_crop, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask_crop = cv2.morphologyEx(mask_crop, cv2.MORPH_OPEN, kernel, iterations=1)

    full_mask_u8 = np.zeros((H, W), dtype=np.uint8)
    full_mask_u8[y0:y1, x0:x1] = mask_crop

    if wedge_mask is not None:
        wedge_bool = wedge_mask > 0
        full_mask_u8[~wedge_bool] = 0

    return full_mask_u8


def process_one_image(img_path: Path) -> None:

    print(f"\n[INFO] Processing {img_path}")

    img = imageio.imread(img_path)
    if img.ndim == 2:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.ndim == 3 and img.shape[-1] == 4:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    else:
        img_rgb = img.astype(np.uint8)

    H, W, _ = img_rgb.shape

    seeds, wedge_mask, mask_yellow, mask_in_wedge = detect_caliper_seeds(
        img_rgb, max_seeds=4
    )
    print("  Seeds (x,y):", seeds)

    if len(seeds) < 2:
        print("  [WARN] <2 seeds found → skipping Otsu for this image.")
        return

    pts = np.array(seeds, dtype=np.float32)

    dists = np.linalg.norm(pts[None, :, :] - pts[:, None, :], axis=-1)
    np.fill_diagonal(dists, 1e9)
    i, j = np.unravel_index(np.argmin(dists), dists.shape)
    p1 = pts[i]
    p2 = pts[j]

    sx = int(round((p1[0] + p2[0]) / 2.0))
    sy = int(round((p1[1] + p2[1]) / 2.0))
    pixel_dist = float(np.linalg.norm(p1 - p2))

    print("  Using seed (x,y) =", (sx, sy))
    print("  Caliper pixel distance:", pixel_dist, "px")

    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

    try:
        otsu_mask_u8 = otsu_local_lesion(
            gray01=gray,
            sx=sx,
            sy=sy,
            pixel_dist=pixel_dist,
            wedge_mask=wedge_mask,
        )
    except Exception as e:
        print("  [WARN] Otsu failed for this image:", e)
        return

    out_name = img_path.stem + "_otsu_mask.png"
    out_path = OUT_DIR / out_name
    imageio.imwrite(out_path, otsu_mask_u8)
    print("  -> Saved Otsu mask:", out_path)


def main() -> None:
    if not FRAMES_DIR.exists():
        print(f"[ERROR] Frames directory does not exist: {FRAMES_DIR}")
        return

    png_files = sorted(FRAMES_DIR.glob("*.png"))
    print(f"[INFO] Found {len(png_files)} PNG files in {FRAMES_DIR}")

    for img_path in png_files:
        if img_path.name.endswith("ceus.png"):
            continue

        process_one_image(img_path)


if __name__ == "__main__":
    main()
