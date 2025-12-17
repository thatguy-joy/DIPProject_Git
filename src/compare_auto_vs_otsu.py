from pathlib import Path
import csv
import numpy as np
import imageio.v2 as imageio

AUTO_DIR = Path("data/masks")
OTSU_DIR = Path("data/otsu_masks")
OUT_CSV = Path("data/seg_overlap_auto_vs_otsu.csv")


def load_binary_mask(path: Path) -> np.ndarray:

    # load PNG and convert to binary mask
    m = imageio.imread(path)
    if m.ndim == 3:
        m = m[..., 0]
    return m > 0


def compute_overlap_metrics(ref: np.ndarray, pred: np.ndarray):

    if ref.shape != pred.shape:
        raise ValueError(f"Shape mismatch: ref {ref.shape}, pred {pred.shape}")

    ref_f = ref.astype(bool)
    pred_f = pred.astype(bool)

    tp = np.logical_and(ref_f, pred_f).sum()
    fp = np.logical_and(~ref_f, pred_f).sum()
    fn = np.logical_and(ref_f, ~pred_f).sum()

    tp = float(tp)
    fp = float(fp)
    fn = float(fn)

    denom_dice = 2 * tp + fp + fn
    denom_iou = tp + fp + fn
    denom_p = tp + fp
    denom_r = tp + fn

    dice = 2 * tp / denom_dice if denom_dice > 0 else float("nan")
    iou = tp / denom_iou if denom_iou > 0 else float("nan")
    prec = tp / denom_p if denom_p > 0 else float("nan")
    rec = tp / denom_r if denom_r > 0 else float("nan")

    return tp, fp, fn, dice, iou, prec, rec


def main():
    if not AUTO_DIR.exists():
        print(f"[ERROR] Auto-mask directory does not exist: {AUTO_DIR}")
        return
    if not OTSU_DIR.exists():
        print(f"[ERROR] Otsu-mask directory does not exist: {OTSU_DIR}")
        return

    otsu_files = sorted(OTSU_DIR.glob("*_otsu_mask.png"))
    print(f"[INFO] Found {len(otsu_files)} Otsu masks in {OTSU_DIR}")

    if not otsu_files:
        return

    rows = []
    n_pairs = 0

    for otsu_path in otsu_files:
        stem = otsu_path.stem
        frame_stem = stem.replace("_otsu_mask", "")

        auto_name = frame_stem + "_auto_mask.png"
        auto_path = AUTO_DIR / auto_name

        print("\n===================================")
        print(f"[INFO] Otsu mask: {otsu_path.name}")
        print(f"       Looking for auto mask: {auto_name}")

        if not auto_path.exists():
            print("  [WARN] Auto mask not found → skipping this pair.")
            continue

        ref = load_binary_mask(auto_path)
        pred = load_binary_mask(otsu_path)

        if ref.shape != pred.shape:
            print(
                f"  [WARN] Shape mismatch ref={ref.shape}, pred={pred.shape} → skipping."
            )
            continue

        tp, fp, fn, dice, iou, prec, rec = compute_overlap_metrics(ref, pred)

        auto_pixels = int(ref.sum())
        otsu_pixels = int(pred.sum())

        print(f"  auto_pixels = {auto_pixels}, otsu_pixels = {otsu_pixels}")
        print(f"  TP={tp:.0f}, FP={fp:.0f}, FN={fn:.0f}")
        print(
            f"  Dice={dice:.3f}, IoU={iou:.3f}, Precision={prec:.3f}, Recall={rec:.3f}"
        )

        rows.append(
            [
                frame_stem,
                auto_path.name,
                otsu_path.name,
                auto_pixels,
                otsu_pixels,
                int(tp),
                int(fp),
                int(fn),
                dice,
                iou,
                prec,
                rec,
            ]
        )

        n_pairs += 1

    print("\n===================================")
    print(f"[SUMMARY] Evaluated {n_pairs} auto/Otsu pairs.")

    if rows:
        OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        write_header = not OUT_CSV.exists()

        header = [
            "frame_stem",
            "auto_mask_name",
            "otsu_mask_name",
            "auto_pixels",
            "otsu_pixels",
            "tp",
            "fp",
            "fn",
            "dice",
            "iou",
            "precision",
            "recall",
        ]

        with OUT_CSV.open("a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(header)
            writer.writerows(rows)

        print(f"[INFO] Saved metrics to {OUT_CSV}")


if __name__ == "__main__":
    main()
