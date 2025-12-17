from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    csv_path = Path("data/morph_radius_ablation.csv")

    if not csv_path.exists():
        print(f"[ERROR] CSV not found at: {csv_path}")
        return
    print(f"[INFO] Loading: {csv_path}")
    df = pd.read_csv(csv_path)

    df = df[df["area_mask"] > 0].copy()

    # group by radius pair (r_core, r_outer)
    grouped = (
        df.groupby(["r_core", "r_outer"])
        .agg(
            n_images=("filename", "count"),
            core_frac_mean=("core_frac", "mean"),
            core_frac_std=("core_frac", "std"),
            outer_frac_mean=("outer_frac", "mean"),
            outer_frac_std=("outer_frac", "std"),
            area_mask_mean=("area_mask", "mean"),
            perim_mask_mean=("perim_mask", "mean"),
        )
        .reset_index()
    )

    print("\n[INFO] Aggregated by (r_core, r_outer):")
    print(grouped)

    out_summary = csv_path.with_name("morph_radius_ablation_summary.csv")
    grouped.to_csv(out_summary, index=False)
    print(f"[INFO] Saved summary → {out_summary}")

    data_dir = csv_path.parent

    sns.set(style="whitegrid")

    # -------- Plot 1: core/outer fraction vs radius pair --------
    plt.figure(figsize=(8, 5))
    ax = plt.gca()

    # label for x-axis as "r_core,r_outer"
    grouped["radius_pair"] = grouped.apply(
        lambda row: f"({int(row['r_core'])},{int(row['r_outer'])})", axis=1
    )

    x = np.arange(len(grouped))
    width = 0.35

    ax.bar(
        x - width / 2,
        grouped["core_frac_mean"],
        width,
        label="Core / Mask (mean)",
    )
    ax.bar(
        x + width / 2,
        grouped["outer_frac_mean"],
        width,
        label="Outer / Mask (mean)",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(grouped["radius_pair"], rotation=45, ha="right")
    ax.set_ylabel("Fraction of mask")
    ax.set_xlabel("(r_core, r_outer)")
    ax.set_title("Core / Outer Fractions vs Structuring Element Radii")
    ax.legend()

    plt.tight_layout()
    plot1_path = data_dir / "morph_radius_fractions_vs_radii.png"
    plt.savefig(plot1_path, dpi=200)
    plt.close()
    print(f"[INFO] Saved plot → {plot1_path}")

    # -------- Plot 2: perimeter vs radius pair --------
    plt.figure(figsize=(8, 5))
    ax = plt.gca()

    ax.plot(
        grouped["radius_pair"],
        grouped["perim_mask_mean"],
        marker="o",
    )
    ax.set_xlabel("(r_core, r_outer)")
    ax.set_ylabel("Mean lesion perimeter (pixels)")
    ax.set_title("Boundary length vs Structuring Element Radii")
    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()
    plot2_path = data_dir / "morph_radius_perimeter_vs_radii.png"
    plt.savefig(plot2_path, dpi=200)
    plt.close()
    print(f"[INFO] Saved plot → {plot2_path}")

    candidate = grouped[grouped["outer_frac_mean"] < 2.0].copy()
    if not candidate.empty:
        target = 0.7
        candidate["core_diff"] = (candidate["core_frac_mean"] - target).abs()
        best = candidate.sort_values("core_diff").iloc[0]
        print("\n[INFO] Heuristic best (r_core, r_outer) based on fractions:")
        print(best[["r_core", "r_outer", "core_frac_mean", "outer_frac_mean"]])
    else:
        print("\n[WARN] No candidate radii with outer_frac_mean < 2.0.")


if __name__ == "__main__":
    main()
