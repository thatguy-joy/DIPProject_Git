from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"
    csv_path = data_dir / "seg_overlap_auto_vs_otsu.csv"

    if not csv_path.exists():
        print(f"[ERROR] seg_overlap_auto_vs_otsu.csv not found at: {csv_path}")
        return

    print(f"[INFO] Loading overlap metrics from: {csv_path}")
    df = pd.read_csv(csv_path)

    required = {"dice", "iou"}
    missing = required - set(df.columns)
    if missing:
        print(f"[ERROR] Missing expected columns in CSV: {missing}")
        return

    df = df.dropna(subset=["dice", "iou"])
    if df.empty:
        print("[ERROR] No valid rows after dropping NaNs.")
        return

    for m in ["dice", "iou"]:
        med = df[m].median()
        q1 = df[m].quantile(0.25)
        q3 = df[m].quantile(0.75)
        print(f"{m}: median={med:.3f}, IQR=[{q1:.3f}, {q3:.3f}]")

    sns.set(style="whitegrid")

    # dice plot
    fig, ax = plt.subplots(figsize=(5, 6))

    sns.boxplot(y=df["dice"], ax=ax)
    sns.stripplot(y=df["dice"], color="black", alpha=0.5, jitter=0.1, ax=ax)

    ax.set_ylabel("Dice coefficient (auto vs Otsu)")
    ax.set_xticks([])
    ax.set_xlabel("")
    ax.set_title("Overlap between auto caliper masks and Otsu masks")

    fig.tight_layout()
    fig.subplots_adjust(top=0.9)

    out_dice = data_dir / "seg_overlap_dice_box.png"
    fig.savefig(out_dice, dpi=200)
    plt.close(fig)
    print(f"[INFO] Saved Dice boxplot → {out_dice}")

    # IoU plot
    fig, ax = plt.subplots(figsize=(5, 6))

    sns.boxplot(y=df["iou"], ax=ax)
    sns.stripplot(y=df["iou"], color="black", alpha=0.5, jitter=0.1, ax=ax)

    ax.set_ylabel("IoU (Jaccard index)")
    ax.set_xticks([])
    ax.set_xlabel("")
    ax.set_title("Overlap between auto caliper masks and Otsu masks")

    fig.tight_layout()
    fig.subplots_adjust(top=0.9)

    out_iou = data_dir / "seg_overlap_iou_box.png"
    fig.savefig(out_iou, dpi=200)
    plt.close(fig)
    print(f"[INFO] Saved IoU boxplot → {out_iou}")


if __name__ == "__main__":
    main()
