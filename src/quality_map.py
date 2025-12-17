from pathlib import Path
import cv2
import numpy as np
import imageio.v2 as imageio


# ---------------------------------- Config ----------------------------------

DATA_DIR = Path("data")
FRAME_DIR = DATA_DIR / "frames_full"
MASK_DIR = DATA_DIR / "masks"

QMAP_DIR = DATA_DIR / "quality_maps"
QMAP_DIR.mkdir(parents=True, exist_ok=True)

# Quality levels
Q_HI = 1.0  # keep detail - in lesion core
Q_LO = 0.3  # can compress more - far from lesion


# ------------------------------- Helper: compute core / outer / band using equivalent-radius radii -------------------------------


def compute_core_outer_band(
    mask_u8: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """
    Given a binary mask (0/255), compute:
      core_u8, outer_u8, band_u8, r_core, r_outer

    Uses the equivalent-radius scheme:
        R_eq     = sqrt(area / pi)
        r_core   = max(1, round(0.2 * R_eq))
        r_outer  = max(r_core + 1, round(0.35 * R_eq))
        r_outer  = min(r_outer, 7)
    """
    mask_bin = (mask_u8 > 0).astype(np.uint8)
    area = int(mask_bin.sum())
    if area == 0:
        h, w = mask_bin.shape
        core = np.zeros((h, w), np.uint8)
        outer = np.zeros((h, w), np.uint8)
        band = np.zeros((h, w), np.uint8)
        return core, outer, band, 0, 0

    # equivalent radius of same-area disc
    R_eq = np.sqrt(area / np.pi)

    r_core = max(1, int(round(0.2 * R_eq)))
    r_outer = max(r_core + 1, int(round(0.35 * R_eq)))
    r_outer = min(r_outer, 7)

    # structuring elements(disks)
    k_core = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * r_core + 1, 2 * r_core + 1)
    )
    k_outer = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * r_outer + 1, 2 * r_outer + 1)
    )

    core = cv2.erode(mask_bin, k_core)
    outer = cv2.dilate(mask_bin, k_outer)

    # feather band
    band = outer.copy()
    band[core > 0] = 0

    core = (core * 255).astype(np.uint8)
    outer = (outer * 255).astype(np.uint8)
    band = (band * 255).astype(np.uint8)

    return core, outer, band, r_core, r_outer


# ------------------------------------- Helper: build quality map Q(x,y) given core/outer/band -------------------------------------


def build_quality_map(core_u8: np.ndarray, outer_u8: np.ndarray) -> np.ndarray:
    """
    Given core and outer masks (0/255), build a float32 quality map Q in [0,1]:

        Q = Q_HI in core
        Q = Q_LO outside outer
        Q smoothly ramps between Q_HI and Q_LO in the band = outer \ core
    """
    core = core_u8 > 0
    outer = outer_u8 > 0
    band = outer & (~core)

    h, w = core.shape
    Q = np.zeros((h, w), np.float32)

    Q[:] = Q_LO

    Q[core] = Q_HI

    if band.any():
        eps = 1e-6

        # distance to core: make core pixels background (0),
        # everything else foreground (1), then distanceTransform
        core_bg = (~core).astype(np.uint8)
        dist_to_core = cv2.distanceTransform(core_bg, cv2.DIST_L2, 5)

        outer_not = outer.astype(np.uint8)
        dist_to_outside = cv2.distanceTransform(outer_not, cv2.DIST_L2, 5)

        # normalize: t = 0 near core, t = 1 near outside
        t = dist_to_core / (dist_to_core + dist_to_outside + eps)

        # linear ramp between Q_HI and Q_LO
        Q_band = Q_HI * (1.0 - t) + Q_LO * t

        # only apply ramp inside band
        Q[band] = Q_band[band]

    Q = np.clip(Q, 0.0, 1.0)
    return Q


# --------------------------------------- Main ---------------------------------------


def main() -> None:
    if not MASK_DIR.exists():
        print(f"[ERROR] Mask directory does not exist: {MASK_DIR}")
        return
    if not FRAME_DIR.exists():
        print(
            f"[WARN] Frame directory does not exist: {FRAME_DIR} (overlays will fail)"
        )

    mask_files = sorted(MASK_DIR.glob("*_auto_mask.png"))
    print(f"[INFO] Found {len(mask_files)} auto masks in {MASK_DIR}")

    for mask_path in mask_files:
        print("\n===================================")
        print(f"[INFO] Processing mask: {mask_path.name}")

        frame_name = mask_path.name.replace("_auto_mask", "")
        frame_path = FRAME_DIR / frame_name

        if not frame_path.exists():
            print(f"  [WARN] Frame not found at {frame_path} (will still build Q-map).")

        mask_u8 = imageio.imread(mask_path)
        if mask_u8.ndim == 3:
            mask_u8 = mask_u8[..., 0]
        mask_u8 = (mask_u8 > 0).astype(np.uint8) * 255

        core_u8, outer_u8, band_u8, r_core, r_outer = compute_core_outer_band(mask_u8)
        print(
            f"  area_mask = {(mask_u8 > 0).sum()} px, r_core={r_core}, r_outer={r_outer}"
        )

        # build Q-map
        Q = build_quality_map(core_u8, outer_u8)

        # save float map as .npy
        stem = frame_name.rsplit(".", 1)[0]
        npy_path = QMAP_DIR / f"{stem}_qmap.npy"
        np.save(npy_path, Q)
        print(f"  [SAVE] quality map (npy) → {npy_path}")

        # save grayscale visualization
        Q_u8 = (255.0 * Q).astype(np.uint8)
        q_png_path = QMAP_DIR / f"{stem}_qmap.png"
        imageio.imwrite(q_png_path, Q_u8)
        print(f"  [SAVE] quality map (PNG) → {q_png_path}")

        # optional: pseudocolor overlay on frame
        if frame_path.exists():
            frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if frame is None:
                print(f"  [WARN] Could not read frame {frame_path} for overlay.")
            else:
                Q_color = cv2.applyColorMap(Q_u8, cv2.COLORMAP_JET)
                alpha = 0.5
                overlay = cv2.addWeighted(frame, 1.0, Q_color, alpha, 0)

                overlay_path = QMAP_DIR / f"{stem}_qoverlay.png"
                cv2.imwrite(str(overlay_path), overlay)
                print(f"  [SAVE] overlay → {overlay_path}")

    print("\n[INFO] Done building quality maps.")


if __name__ == "__main__":
    main()
