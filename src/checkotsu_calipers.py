from pathlib import Path
import csv
import cv2
import numpy as np
import imageio.v2 as imageio

from detect_calipers import detect_caliper_seeds


def fraction_mask_between_calipers(
    mask_u8: np.ndarray,
    p1: tuple[int, int],
    p2: tuple[int, int],
):
    """
    mask_u8 : (H, W) uint8, 0/255
    p1, p2  : caliper centers (x,y)
    Returns:
        frac           : fraction of lesion pixels whose projection along p1->p2
                         lies between the two calipers (0 <= t <= 1)
        lesion_pixels  : total lesion pixel count
        between_pixels : number of lesion pixels between calipers
        t_min, t_max   : min/max projection parameter along the segment
    """
    H, W = mask_u8.shape
    x1, y1 = p1
    x2, y2 = p2

    vx = x2 - x1
    vy = y2 - y1
    denom = vx * vx + vy * vy
    if denom == 0:
        raise ValueError("Caliper points are identical")

    # lesion pixels
    ys, xs = np.nonzero(mask_u8 > 0)
    if xs.size == 0:
        print("  [WARN] mask is empty.")
        return 0.0, 0, 0, 0.0, 0.0

    dx = xs - x1
    dy = ys - y1

    # projection parameter t along the line p1->p2
    t = (dx * vx + dy * vy) / float(denom)

    # count pixels - projection lies between the calipers
    between = (t >= 0.0) & (t <= 1.0)
    between_count = int(between.sum())
    lesion_pixels = int(xs.size)
    frac = float(between_count) / float(lesion_pixels)

    t_min = float(t.min())
    t_max = float(t.max())

    print(f"  lesion pixels: {lesion_pixels}")
    print(f"  between calipers: {between_count} ({100 * frac:.1f}%)")
    print(f"  t_min={t_min:.2f}, t_max={t_max:.2f}")

    return frac, lesion_pixels, between_count, t_min, t_max


def main():

    MASK_DIR = Path("data/otsu_masks")
    FRAME_DIR = Path("data/frames_full")
    DATA_DIR = Path("data")
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not MASK_DIR.exists():
        print(f"[ERROR] Mask directory does not exist: {MASK_DIR}")
        return
    if not FRAME_DIR.exists():
        print(f"[ERROR] Frame directory does not exist: {FRAME_DIR}")
        return

    mask_files = sorted(MASK_DIR.glob("*.png"))
    print(f"[INFO] Found {len(mask_files)} Otsu masks in {MASK_DIR}")

    if not mask_files:
        return

    n_total = 0
    n_good = 0
    rows = []

    for mask_path in mask_files:
        print("\n============================================")
        print(f"[INFO] Mask: {mask_path.name}")

        img_name = mask_path.name.replace("_otsu_mask", "")
        img_path = FRAME_DIR / img_name

        if not img_path.exists():
            print(f"  [WARN] No matching frame for {mask_path.name} at {img_path}")
            continue

        print(f"  Using frame: {img_path.name}")

        img = imageio.imread(img_path)
        if img.ndim == 2:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.ndim == 3 and img.shape[-1] == 4:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        else:
            img_rgb = img.astype(np.uint8)

        seeds, wedge_mask, mask_yellow, mask_in_wedge = detect_caliper_seeds(
            img_rgb, max_seeds=4
        )
        print("  Seeds (x,y):", seeds)
        if len(seeds) < 2:
            print("  [WARN] Not enough calipers detected → skipping.")
            continue

        pts = np.array(seeds, dtype=np.float32)
        dists = np.linalg.norm(pts[None, :, :] - pts[:, None, :], axis=-1)
        np.fill_diagonal(dists, 1e9)
        i, j = np.unravel_index(np.argmin(dists), dists.shape)
        p1 = tuple(map(int, pts[i]))
        p2 = tuple(map(int, pts[j]))
        print("  Using caliper pair:", p1, p2)

        mask_u8 = imageio.imread(mask_path)
        if mask_u8.ndim == 3:
            mask_u8 = mask_u8[..., 0]

        frac, lesion_pixels, between_pixels, t_min, t_max = (
            fraction_mask_between_calipers(mask_u8, p1, p2)
        )

        n_total += 1
        is_good = frac >= 0.9
        if is_good:
            print("  → Mask is mostly between calipers (good).")
            n_good += 1
        else:
            print("  → Mask extends significantly beyond calipers (check this case).")

        rows.append(
            [
                mask_path.name,
                img_path.name,
                p1[0],
                p1[1],
                p2[0],
                p2[1],
                lesion_pixels,
                between_pixels,
                frac,
                t_min,
                t_max,
                int(is_good),
            ]
        )

    print("\n============================================")
    print(f"[SUMMARY] Checked {n_total} masks.")
    if n_total > 0:
        print(
            f"          {n_good} / {n_total} "
            f"({100.0 * n_good / n_total:.1f}%) were ≥ 90% between calipers."
        )

    if rows:
        csv_path = DATA_DIR / "checkotsu_caliper.csv"
        header = [
            "mask_name",
            "frame_name",
            "p1_x",
            "p1_y",
            "p2_x",
            "p2_y",
            "lesion_pixels",
            "between_pixels",
            "fraction_between",
            "t_min",
            "t_max",
            "is_good_90pct",
        ]
        write_header = not csv_path.exists()

        with csv_path.open("a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(header)
            writer.writerows(rows)

        print(f"[INFO] Saved QC results to {csv_path}")


if __name__ == "__main__":
    main()
