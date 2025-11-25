#still under construction!!!!⚒️

from pathlib import Path
import pydicom
import numpy as np
import imageio.v2 as imageio

RAW_ROOT = Path("data/raw")
OUT_DIR = Path("data/frames_full")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def _normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    
    """Normalize array to [0, 255] uint8 (per image)."""
    
    a = arr.astype(np.float32)
    a = (a - a.min()) / (a.max() - a.min() + 1e-8)
    a = (255 * a).astype(np.uint8)
    return a


def _maybe_split_dual_panel(arr: np.ndarray):
    """
    detect B-mode | CEUS dual-panel.

    Heuristic:
      - must be very wide: W > 1.6 * H
      - look for a dark vertical band near the center:
        min column-mean intensity in central 20% of columns
        must be < 0.5 * global mean.
    Returns:
      (left, right) if split is detected, otherwise None.
    """
    h, w = arr.shape[:2]

    #check aspect ratio
    if w <= 1.6 * h:
        return None

    #grayscale
    if arr.ndim == 2:
        gray = arr.astype(np.float32)
    else:
        gray = arr.astype(np.float32).mean(axis=-1)

    global_mean = gray.mean()
    if global_mean <= 1e-6:
        return None

    #central 20% of columns
    c0 = int(0.4 * w)
    c1 = int(0.6 * w)
    if c1 <= c0:
        return None

    col_means = gray.mean(axis=0)
    central = col_means[c0:c1]

    if central.size == 0:
        return None

    min_idx = int(np.argmin(central))
    min_val = float(central[min_idx])

    #clear dark band required
    if min_val >= 0.5 * global_mean:
        return None

    #column index in full image where we split
    mid = c0 + min_idx
    left = arr[:, :mid, ...]
    right = arr[:, mid:, ...]

    return left, right


def save_frame(arr: np.ndarray, base_name: str, frame_idx: int) -> None:
    """
    Normalize to 0–255 and save as PNG.
    If the frame looks like a dual-panel (B-mode | CEUS) screenshot
    according to _maybe_split_dual_panel, split and save as:
        <base>_fXXX_bmode.png
        <base>_fXXX_ceus.png
    Otherwise save a single:
        <base>_fXXX.png
    """
    h, w = arr.shape[:2]

    split = _maybe_split_dual_panel(arr)

    if split is not None:
        left, right = split
        left_u8 = _normalize_to_uint8(left)
        right_u8 = _normalize_to_uint8(right)

        out_name_bmode = f"{base_name}_f{frame_idx:03d}_bmode.png"
        out_name_ceus = f"{base_name}_f{frame_idx:03d}_ceus.png"

        out_path_bmode = OUT_DIR / out_name_bmode
        out_path_ceus = OUT_DIR / out_name_ceus

        imageio.imwrite(out_path_bmode, left_u8)
        imageio.imwrite(out_path_ceus, right_u8)

        print(
            f"  -> Dual-panel detected (H={h}, W={w}). "
            f"Saved B-mode: {out_path_bmode}, CEUS: {out_path_ceus}"
        )
    else:
        a = _normalize_to_uint8(arr)
        out_name = f"{base_name}_f{frame_idx:03d}.png"
        out_path = OUT_DIR / out_name
        imageio.imwrite(out_path, a)
        print(f"  -> Saved {out_path}")

def main() -> None:
    if not RAW_ROOT.exists():
        print(f"[ERROR] RAW_ROOT does not exist: {RAW_ROOT}")
        return

    dcm_files = sorted(RAW_ROOT.rglob("*.dcm"))
    print(f"Found {len(dcm_files)} DICOM files under {RAW_ROOT}")

    for dcm_path in dcm_files:
        print(f"\nProcessing {dcm_path} ...")

        try:
            ds = pydicom.dcmread(dcm_path)
            arr = ds.pixel_array
        except Exception as e:
            print(f"  [WARN] Skipping {dcm_path}: {e}")
            continue

        patient_folder = dcm_path.parent.name
        base = f"{patient_folder}_{dcm_path.stem}"

        #Case 1: single-frame 2D(H, W)
        if arr.ndim == 2:
            save_frame(arr, base, 0)

        #Case 2: single-frame with channels(H, W, C)
        elif arr.ndim == 3 and arr.shape[-1] in (1, 3):
            save_frame(arr, base, 0)

        #Case 3: multi-frame grayscale(T, H, W)
        elif arr.ndim == 3:
            T = arr.shape[0]
            print(f"  [INFO] Multi-frame array with {T} frames (T, H, W)")
            for i in range(T):
                save_frame(arr[i], base, i)

        #Case 4: multi-frame color(T, H, W, C)
        elif arr.ndim == 4:
            T = arr.shape[0]
            print(f"  [INFO] Multi-frame array with {T} frames (T, H, W, C)")
            for i in range(T):
                save_frame(arr[i], base, i)

        else:
            print(f"  [WARN] Skipping {dcm_path}, unexpected shape: {arr.shape}")


if __name__ == "__main__":
    main()
