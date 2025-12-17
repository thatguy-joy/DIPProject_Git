from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def make_long_roi_metric(
    df: pd.DataFrame, col_baseline: str, col_roi: str, value_name: str
):
    rows = []
    for _, row in df.iterrows():
        base_id = row["base"]
        rows.append(
            {
                "base": base_id,
                "scheme": "baseline_raw",
                value_name: row[col_baseline],
            }
        )
        rows.append(
            {
                "base": base_id,
                "scheme": "roi_aware_bitrate_matched",
                value_name: row[col_roi],
            }
        )
    df_long = pd.DataFrame(rows)
    df_long["scheme"] = pd.Categorical(
        df_long["scheme"],
        categories=["baseline_raw", "roi_aware_bitrate_matched"],
        ordered=True,
    )
    return df_long


def main():
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"
    csv_path = data_dir / "compression_metrics.csv"

    if not csv_path.exists():
        print(f"[ERROR] compression_metrics.csv not found at: {csv_path}")
        return

    print(f"[INFO] Loading compression metrics from: {csv_path}")
    df = pd.read_csv(csv_path)

    required_cols = {
        "psnr_roi_baseline",
        "psnr_roi_roi",
        "ssim_roi_baseline",
        "ssim_roi_roi",
        "rel_bytes_diff",
        "H_bits_ref",
        "H_bits_roi_pre",
    }
    missing = required_cols - set(df.columns)
    if missing:
        print(f"[ERROR] Missing expected columns: {missing}")
        return

    sns.set(style="whitegrid")

    # ------------------------------ 1) PSNR in ROI ------------------------------

    df_psnr = make_long_roi_metric(df, "psnr_roi_baseline", "psnr_roi_roi", "psnr_roi")

    plt.figure(figsize=(6, 5))
    ax = sns.boxplot(
        data=df_psnr,
        x="scheme",
        y="psnr_roi",
        width=0.5,
        showfliers=False,
    )
    sns.stripplot(
        data=df_psnr,
        x="scheme",
        y="psnr_roi",
        dodge=False,
        alpha=0.6,
        size=5,
        color="black",
    )

    ax.set_xlabel("Compression configuration")
    ax.set_ylabel("PSNR in lesion ROI (dB)")
    ax.set_title("Lesion-quality comparison at matched bitrate (PSNR)")

    ax.set_xticklabels(
        ["Baseline (global preprocessing)", "ROI-aware (bitrate-matched)"],
        rotation=10,
    )

    plt.tight_layout()
    out_psnr = data_dir / "compression_psnr_roi.png"
    plt.savefig(out_psnr, dpi=200)
    plt.close()
    print(f"[INFO] Saved PSNR-ROI plot → {out_psnr}")

    # ----------------------------- 2) SSIM in ROI -----------------------------

    df_ssim = make_long_roi_metric(df, "ssim_roi_baseline", "ssim_roi_roi", "ssim_roi")

    plt.figure(figsize=(6, 5))
    ax = sns.boxplot(
        data=df_ssim,
        x="scheme",
        y="ssim_roi",
        width=0.5,
        showfliers=False,
    )
    sns.stripplot(
        data=df_ssim,
        x="scheme",
        y="ssim_roi",
        dodge=False,
        alpha=0.6,
        size=5,
        color="black",
    )

    ax.set_xlabel("Compression configuration")
    ax.set_ylabel("SSIM in lesion ROI")
    ax.set_title("Lesion-quality comparison at matched bitrate (SSIM)")

    ax.set_xticklabels(
        ["Baseline (global preprocessing)", "ROI-aware (bitrate-matched)"],
        rotation=10,
    )

    plt.tight_layout()
    out_ssim = data_dir / "compression_ssim_roi.png"
    plt.savefig(out_ssim, dpi=200)
    plt.close()
    print(f"[INFO] Saved SSIM-ROI plot → {out_ssim}")

    # ----------------------------- 3) Bitrate-matching sanity check -----------------------------

    plt.figure(figsize=(6, 4))
    ax = sns.histplot(df["rel_bytes_diff"], bins=10, kde=False)
    ax.axvline(0.0, color="red", linestyle="--", linewidth=1)

    ax.set_xlabel("Relative bitrate difference (ROI-aware − baseline)")
    ax.set_ylabel("Number of frames")
    ax.set_title("Bitrate matching quality (0 = perfect match)")

    plt.tight_layout()
    out_hist = data_dir / "compression_rel_bytes_diff_hist.png"
    plt.savefig(out_hist, dpi=200)
    plt.close()
    print(f"[INFO] Saved bitrate-diff histogram → {out_hist}")

    # ----------------------------- 4) PSNR gain vs entropy reduction -----------------------------

    df["psnr_gain_roi"] = df["psnr_roi_roi"] - df["psnr_roi_baseline"]
    df["entropy_drop"] = df["H_bits_ref"] - df["H_bits_roi_pre"]

    plt.figure(figsize=(6, 5))
    ax = sns.scatterplot(
        data=df,
        x="entropy_drop",
        y="psnr_gain_roi",
    )
    ax.axhline(0.0, color="red", linestyle="--", linewidth=1)

    ax.set_xlabel("Entropy reduction (H_ref − H_roi_pre) [bits]")
    ax.set_ylabel("PSNR gain in ROI (ROI-aware − baseline) [dB]")
    ax.set_title("Link between complexity reduction and ROI PSNR gain")

    plt.tight_layout()
    out_scatter = data_dir / "compression_psnr_gain_vs_entropy_drop.png"
    plt.savefig(out_scatter, dpi=200)
    plt.close()
    print(f"[INFO] Saved PSNR-gain vs entropy-drop scatter → {out_scatter}")


if __name__ == "__main__":
    main()
