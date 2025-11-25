#still under construction!!!!⚒️

from pathlib import Path
from typing import Optional, Dict, Any
import pydicom

RAW_ROOT = Path("data/raw")
DRY_RUN = False     #Turn it to True to avoid deleting files

def classify_dicom(dcm_path: Path) -> Optional[Dict[str, Any]]:
    """
    Read a DICOM file, determine how many frames it has, and whether it's a cine loop.
    Returns a dict with:
        - path: Path
        - shape: tuple
        - num_frames: int
        - is_cine: bool
    Returns None if the file can't be read or has no pixel data.
    """
    try:
        ds = pydicom.dcmread(dcm_path)
    except Exception as e:
        print(f"[WARN] Could not read {dcm_path}: {e}")
        return None

    try:
        arr = ds.pixel_array
    except Exception as e:
        print(f"[WARN] No pixel data in {dcm_path}: {e}")
        return None

    shape = arr.shape

    #Prefer 'DICOM' tag if present
    num_frames_tag = getattr(ds, "NumberOfFrames", None)
    num_frames: Optional[int] = None

    if num_frames_tag is not None:
        try:
            num_frames = int(str(num_frames_tag))
        except ValueError:
            num_frames = None

    #Fallback -> infer from array shape
    if num_frames is None:
        if arr.ndim == 2:
            #(H, W) → single-frame
            num_frames = 1
        elif arr.ndim == 3:
            #Either -> (H, W, C) or (T, H, W)
            if shape[-1] in (1, 3):
                #(H, W, C)
                num_frames = 1
            else:
                #(T, H, W)
                num_frames = shape[0]
        elif arr.ndim == 4:
            #(T, H, W, C)
            num_frames = shape[0]
        else:
            # Unknowncase –> treat as single-frame
            num_frames = 1

    is_cine = num_frames > 1

    return {
        "path": dcm_path,
        "shape": shape,
        "num_frames": num_frames,
        "is_cine": is_cine,
    }


def main() -> None:
    if not RAW_ROOT.exists():
        print(f"[ERROR] RAW_ROOT does not exist: {RAW_ROOT}")
        return

    all_dicoms = sorted(RAW_ROOT.rglob("*.dcm"))
    print(f"Found {len(all_dicoms)} DICOM files under {RAW_ROOT}")

    cine_files = []
    single_files = []

    for dcm_path in all_dicoms:
        info = classify_dicom(dcm_path)
        if info is None:
            continue

        path = info["path"]
        shape = info["shape"]
        num_frames = info["num_frames"]
        is_cine = info["is_cine"]

        if is_cine:
            cine_files.append(info)
            print(f"[CINE ] {path} | shape={shape}, num_frames={num_frames}")
        else:
            single_files.append(info)
            print(f"[SINGLE] {path} | shape={shape}, num_frames={num_frames}")

    print("\n=== SUMMARY ===")
    print(f"Single-frame DICOMs: {len(single_files)}")
    print(f"Cine-loop DICOMs  : {len(cine_files)}")

    if not cine_files:
        print("No cine loops detected. Nothing to remove.")
        return

    if DRY_RUN: 
        return

    print("\nDeleting cine-loop DICOMs...")
    for info in cine_files:
        path: Path = info["path"]
        shape = info["shape"]
        num_frames = info["num_frames"]

        try:
            path.unlink()
            print(f"[DELETE] {path} | shape={shape}, num_frames={num_frames}")
        except Exception as e:
            print(f"[ERROR] Could not delete {path}: {e}")

    print("\nDone. Cine-loop DICOMs removed.")


if __name__ == "__main__":
    main()
