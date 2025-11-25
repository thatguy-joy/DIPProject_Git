#still under construction!!!!⚒️

from pathlib import Path
import cv2
import numpy as np
import imageio.v2 as imageio


# ---------------------------------------------------------
#   Segment the wedge region from grayscale
# ---------------------------------------------------------
def get_wedge_mask(gray: np.ndarray) -> np.ndarray:
    """
    Estimate the ultrasound wedge region by simple thresholding +
    morphology + 'largest connected component'.
    gray: (H, W) uint8
    Returns:
        wedge_mask: (H, W) uint8 in {0, 255}
    """
    H, W = gray.shape

    #everything that is not black
    _, th = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
    
    kernel = np.ones((7, 7), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN,  kernel, iterations=1)

    # --- crop obvious UI/text zones BEFORE CC ---
    # "LIVER TRANS"/"LIVER SAG" text
    cut_top = int(0.10 * H)
    th[:cut_top, :] = 0

    #bottom: measurement panel
    cut_bottom = int(0.82 * H)
    th[cut_bottom:, :] = 0

    #left strip: color bar
    th[:, :int(0.12 * W)] = 0

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        th, connectivity=8
    )
    if num_labels <= 1:
        return th

    areas = stats[1:, cv2.CC_STAT_AREA]
    best_label = 1 + np.argmax(areas)

    wedge = np.zeros_like(th)
    wedge[labels == best_label] = 255
    return wedge


# ---------------------------------------------------------
#   Main detector
# ---------------------------------------------------------
def detect_caliper_seeds(img_rgb: np.ndarray, max_seeds: int = 2):
    """
    Detect yellow '+' calipers in a B-mode ultrasound screenshot.
    Input:
        img_rgb: (H, W, 3) uint8 RGB image
    Returns:
        seeds: list of (x, y) centroids in IMAGE coords
        wedge_mask, mask_yellow, mask_in_wedge: debug masks
    """
    H, W, _ = img_rgb.shape

    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    hsv  = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

    #wedge mask
    wedge_mask = get_wedge_mask(gray)  # 0 / 255

    #raw yellow mask
    lower = np.array([20, 80, 160], dtype=np.uint8)
    upper = np.array([40, 255, 255], dtype=np.uint8)
    mask_yellow = cv2.inRange(hsv, lower, upper)

    #keep only yellow inside wedge
    mask = cv2.bitwise_and(mask_yellow, wedge_mask)

    #no dilation here – we keep raw components so '+' and digits
    #don’t get fused together.
    mask_in_wedge = mask.copy()

    #connected components on yellow∩wedge
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )

    candidates = []

    #plausible vertical band for calipers
    min_y = int(0.20 * H)
    max_y = int(0.85 * H)

    for i in range(1, num_labels):
        x, y, bw, bh, area = stats[i]

        #size filter: '+' should be "small-ish" and not huge
        if area < 10 or area > 300:
            continue

        #aspect ratio: '+' roughly square
        aspect = bw / float(bh)
        if aspect < 0.7 or aspect > 1.4:
            continue

        #centroid in full image coords
        cx_full, cy_full = centroids[i]

        #discard components that are far up or down
        if not (min_y <= cy_full <= max_y):
            continue

        comp = (labels[y:y+bh, x:x+bw] == i).astype(np.uint8)

        # centroid in component coords -> (for row/col check)
        cx = int(round(cx_full - x))
        cy = int(round(cy_full - y))
        cx = np.clip(cx, 0, bw - 1)
        cy = np.clip(cy, 0, bh - 1)

        #central pixel must be foreground
        if comp[cy, cx] == 0:
            continue

        #soft cross-ness check
        row = comp[cy, :]
        col = comp[:, cx]
        row_count = int(row.sum())
        col_count = int(col.sum())
        if row_count < 2 or col_count < 2:
            continue

        diff = abs(row_count - col_count) / float(max(row_count, col_count))
        if diff > 0.7:
            continue

        left_arm  = comp[cy, :cx].sum()
        right_arm = comp[cy, cx+1:].sum()
        up_arm    = comp[:cy, cx].sum()
        down_arm  = comp[cy+1:, cx].sum()
        if min(left_arm, right_arm, up_arm, down_arm) == 0:
            continue

        candidates.append((area, int(round(cx_full)), int(round(cy_full))))

    candidates.sort(key=lambda t: t[0], reverse=True)
    seeds = [(cx, cy) for (area, cx, cy) in candidates[:max_seeds]]

    return seeds, wedge_mask, mask_yellow, mask_in_wedge


# ---------------------------------------------------------
#   Standalone test (Because sometimes the code misses on certain images
#   also, it works only on files that do not end with ceus.png)
# ---------------------------------------------------------
if __name__ == "__main__":
    IMG_PATH = Path(r"X:\DIP-Project\data\frames_full\LiverUS-05_1-05_f000.png")

    img = imageio.imread(IMG_PATH)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.ndim == 3 and img.shape[-1] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

    seeds, wedge_mask, mask_yellow, mask_in_wedge = detect_caliper_seeds(
        img, max_seeds=4
    )
    print("Detected seeds:", seeds)

    data_dir = IMG_PATH.parent.parent
    out_dir = data_dir / "detect_calipers_output"
    out_dir.mkdir(parents=True, exist_ok=True)

    imageio.imwrite(out_dir / f"{IMG_PATH.stem}_wedge_mask.png", wedge_mask)
    imageio.imwrite(out_dir / f"{IMG_PATH.stem}_yellow_mask.png", mask_yellow)
    imageio.imwrite(out_dir / f"{IMG_PATH.stem}_yellow_in_wedge.png", mask_in_wedge)

    vis = img.copy()
    for (sx, sy) in seeds:
        cv2.drawMarker(
            vis,
            (sx, sy),
            color=(255, 0, 0),
            markerType=cv2.MARKER_CROSS,
            markerSize=25,
            thickness=2,
        )

    out_path = out_dir / f"{IMG_PATH.stem}_calipers_overlay.png"
    imageio.imwrite(out_path, vis)
    print("Saved overlay to:", out_path)

