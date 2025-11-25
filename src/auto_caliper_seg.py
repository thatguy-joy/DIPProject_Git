#still under construction!!!!⚒️

from pathlib import Path
import sys
from collections import deque
import csv
import cv2
import numpy as np
import imageio.v2 as imageio

from detect_calipers import detect_caliper_seeds

# ---------------------------------------------------------
# Region growing (4-connected)
# ---------------------------------------------------------
def region_grow(
    gray: np.ndarray,
    seed_rc: tuple[int, int],
    allowed: np.ndarray | None = None,
    tol: float = 0.10,
) -> np.ndarray:
    """
    gray   : (H, W) float32 in [0,1]
    seed_rc: (row, col) = (y, x)
    allowed: optional bool mask (H, W); False = forbidden pixels
    tol    : intensity tolerance relative to seed gray value
    Returns:
        mask: bool array (H, W) of grown region
    """
    H, W = gray.shape
    sy, sx = seed_rc

    if not (0 <= sy < H and 0 <= sx < W):
        raise ValueError(f"Seed {seed_rc} outside image of shape {gray.shape}")

    if allowed is None:
        allowed = np.ones_like(gray, dtype=bool)
    else:
        allowed = allowed.astype(bool)

    if not allowed[sy, sx]:
        raise ValueError("Seed lies outside allowed region")

    seed_val = float(gray[sy, sx])

    visited = np.zeros_like(gray, dtype=bool)
    mask = np.zeros_like(gray, dtype=bool)

    q: deque[tuple[int, int]] = deque()
    q.append((sy, sx))
    visited[sy, sx] = True

    while q:
        y, x = q.popleft()
        if not allowed[y, x]:
            continue

        if abs(float(gray[y, x]) - seed_val) <= tol:
            mask[y, x] = True
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < H and 0 <= nx < W and not visited[ny, nx]:
                    visited[ny, nx] = True
                    q.append((ny, nx))

    return mask


# ---------------------------------------------------------
# Greedy pairing of caliper seeds (for multi-axis measurements)
# ---------------------------------------------------------
def pair_seeds_greedy(pts: np.ndarray):
    """
    pts: (N, 2) float32 array of (x, y) caliper centres
    Returns:
        pairs: list of (p1, p2) where each p* is a (2,) float32 point.
               N=2 -> one pair; N=4 -> two pairs; etc.
    """
    N = len(pts)
    unused = list(range(N))
    pairs: list[tuple[np.ndarray, np.ndarray]] = []

    while len(unused) >= 2:
        best = None
        best_d = np.inf
        for a_i, i in enumerate(unused):
            for j in unused[a_i + 1 :]:
                d = float(np.linalg.norm(pts[i] - pts[j]))
                if d < best_d:
                    best_d = d
                    best = (i, j)
        i, j = best
        pairs.append((pts[i], pts[j]))
        unused.remove(i)
        unused.remove(j)

    return pairs


# ---------------------------------------------------------
# Main -> Auto segment lesion from calipers + measure it
# ---------------------------------------------------------
def main():
    """
    For frames with 4 calipers (two axes), pass the **longest** L
    from the bottom overlay (e.g. 1.79 cm). For single-axis images,
    pass the usual L value.
    """
    # --------------------------------------------------------------
    # Input frame + physical caliper length (longest L if 2 are present)
    # --------------------------------------------------------------
    if len(sys.argv) >= 2:
        img_path = Path(sys.argv[1])
    else:
        img_path = Path(r"X:\DIP-Project\data\frames_full\LiverUS-05_1-05_f000.png")

    if len(sys.argv) >= 3:
        measured_length_cm = float(sys.argv[2])
    else:
        measured_length_cm = 3.47

    if not img_path.exists():
        raise FileNotFoundError(img_path)

    print("Input frame:", img_path)
    print("Reference caliper length from machine:", measured_length_cm, "cm")

    # --------------------------------------------------------------
    #     Derive data directories (…/data/…)
    #     img_path = .../data/frames_full/<file>.png
    #     -> data_dir = .../data
    # --------------------------------------------------------------
    data_dir = img_path.parent.parent  
    bmode_dir = data_dir / "bmode_frames"
    masks_dir = data_dir / "masks"
    csv_path = data_dir / "lesion_measurements.csv"

    bmode_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    stem = img_path.stem
    out_mask_path = masks_dir / f"{stem}_auto_mask.png"
    out_overlay_path = bmode_dir / f"{stem}_auto_overlay.png"

    # --------------------------------------------------------------
    # Load image
    # --------------------------------------------------------------
    img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise RuntimeError(f"cv2.imread failed on {img_path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H, W, _ = img_rgb.shape

    # --------------------------------------------------------------
    # Detect calipers → seeds + wedge + yellow overlay mask
    # --------------------------------------------------------------
    seeds, wedge_mask, mask_yellow, _ = detect_caliper_seeds(
        img_rgb, max_seeds=6
    )
    print("Caliper seeds (x,y):", seeds)

    if len(seeds) < 2:
        print("Need at least two caliper points → aborting.")
        return

    pts = np.array(seeds, dtype=np.float32)

    # ---- Pair seeds (handles 2, 4, … seeds) ----
    pairs = pair_seeds_greedy(pts)
    pair_dists = [float(np.linalg.norm(p1 - p2)) for (p1, p2) in pairs]
    print("Caliper pairs (pixel distances):", pair_dists)

    #Choose the pair with the largest pixel distance as reference
    #(long axis if image has two axes; only pair if N=2).
    ref_idx = int(np.argmax(pair_dists))
    p1_ref, p2_ref = pairs[ref_idx]
    pixel_dist_ref = pair_dists[ref_idx]

    print(
        f"Using pair {ref_idx} as scale reference, pixel distance = "
        f"{pixel_dist_ref:.3f} px"
    )

    cm_per_pixel = measured_length_cm / pixel_dist_ref
    pixel_per_cm = pixel_dist_ref / measured_length_cm
    print("Scale:", cm_per_pixel, "cm/pixel  (", pixel_per_cm, "px/cm )")

    #Seed = average of all detected calipers --> (works for 2 or 4)
    xs = [s[0] for s in seeds]
    ys = [s[1] for s in seeds]
    sx = int(round(np.mean(xs)))
    sy = int(round(np.mean(ys)))
    print("Using seed (x,y) =", (sx, sy))

    #Radius: based on the largest caliper distance so we cover the whole lesion
    max_pair_dist = max(pair_dists)
    R_pix = 0.8 * (max_pair_dist / 2.0)
    print(f"Local region radius R ≈ {R_pix:.2f} px (from max caliper distance)")

    # --------------------------------------------------------------
    # Grayscale + local region growing
    # --------------------------------------------------------------
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    gray_smooth = cv2.GaussianBlur(gray, (3, 3), 0)

    Y, X = np.ogrid[:H, :W]
    dist2 = (X - sx) ** 2 + (Y - sy) ** 2
    local_mask = dist2 <= R_pix ** 2

    #Allowed region = wedge AND local disc AND not yellow overlay
    if wedge_mask is not None:
        base_allowed = wedge_mask > 0
    else:
        base_allowed = np.ones((H, W), dtype=bool)

    if mask_yellow is not None:
        base_allowed = base_allowed & (mask_yellow == 0)

    allowed = base_allowed & local_mask

    lesion_bool = region_grow(
        gray_smooth,
        seed_rc=(sy, sx),  
        allowed=allowed,
        tol=0.10,
    )

    # --------------------------------------------------------------
    # Morphological cleanup + largest component
    # --------------------------------------------------------------
    lesion_u8 = lesion_bool.astype(np.uint8) * 255

    kernel = np.ones((3, 3), np.uint8)
    lesion_u8 = cv2.morphologyEx(lesion_u8, cv2.MORPH_CLOSE, kernel, iterations=1)
    lesion_u8 = cv2.morphologyEx(lesion_u8, cv2.MORPH_OPEN, kernel, iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        lesion_u8, connectivity=8
    )
    if num_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        best_label = 1 + np.argmax(areas)
        cleaned = np.zeros_like(lesion_u8)
        cleaned[labels == best_label] = 255
        lesion_u8 = cleaned

    # --------------------------------------------------------------
    # Area measurement
    # --------------------------------------------------------------
    lesion_pixels = int((lesion_u8 > 0).sum())
    area_cm2 = lesion_pixels * (cm_per_pixel ** 2)

    print("Lesion pixels:", lesion_pixels)
    print(f"Lesion area ≈ {area_cm2:.3f} cm^2")

    # --------------------------------------------------------------
    # Save binary mask (in data/masks)
    # --------------------------------------------------------------
    imageio.imwrite(out_mask_path, lesion_u8)
    print("Saved auto mask →", out_mask_path)

    # --------------------------------------------------------------
    # Save overlay with contour + calipers + text (in data/bmode_frames)
    # --------------------------------------------------------------
    overlay_bgr = img_bgr.copy()

    contours, _ = cv2.findContours(
        lesion_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(overlay_bgr, contours, -1, (0, 0, 255), 2)  # red

    #Draw all detected calipers as red tilted crosses,
    #and the region-growing seed as a green cross.
    for (x, y) in seeds:
        cv2.drawMarker(
            overlay_bgr,
            (int(x), int(y)),
            (0, 0, 255),
            cv2.MARKER_TILTED_CROSS,
            18,
            2,
        )

    cv2.drawMarker(
        overlay_bgr,
        (sx, sy),
        (0, 255, 0),
        cv2.MARKER_CROSS,
        22,
        2,
    )

    text = f"L_ref = {measured_length_cm:.2f} cm, Area ~ {area_cm2:.2f} cm^2"
    cv2.putText(
        overlay_bgr,
        text,
        (20, H - 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255), 
        2,
        cv2.LINE_AA,
    )

    overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
    imageio.imwrite(out_overlay_path, overlay_rgb)
    print("Saved overlay →", out_overlay_path)

    # --------------------------------------------------------------
    # Append measurements to CSV in data/lesion_measurements.csv
    # --------------------------------------------------------------
    header = [
        "image_path",
        "filename",
        "measured_length_cm",
        "ref_pair_pixel_distance",
        "cm_per_pixel",
        "px_per_cm",
        "seed_x",
        "seed_y",
        "lesion_pixels",
        "lesion_area_cm2",
    ]

    row = [
        str(img_path),
        img_path.name,
        measured_length_cm,
        pixel_dist_ref,
        cm_per_pixel,
        pixel_per_cm,
        sx,
        sy,
        lesion_pixels,
        area_cm2,
    ]

    write_header = not csv_path.exists()

    with csv_path.open("a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(row)

    print("Appended measurements →", csv_path)


if __name__ == "__main__":
    main()
