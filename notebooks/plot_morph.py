from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def main():

    csv_path = Path(r"X:\DIP-Project\data\morph_masks\morph_stats.csv")
    data_dir = csv_path.parent

    if not csv_path.exists():
        print(f"[ERROR] CSV not found at: {csv_path}")
        return
    print(f"[INFO] Loading morphology stats from: {csv_path}")
    df = pd.read_csv(csv_path)

    df = df[df["area_mask"] > 0].copy()

    df_long = pd.melt(
        df,
        id_vars=["filename", "area_mask"],
        value_vars=["core_frac", "outer_frac"],
        var_name="region",
        value_name="frac",
    )

    region_labels = {
        "core_frac": "Core / Mask",
        "outer_frac": "Outer / Mask",
    }
    df_long["region_label"] = df_long["region"].map(region_labels)

    # Figure 1: scatter (area vs fraction), log-x

    sns.set(style="whitegrid")
    plt.figure(figsize=(8, 5))

    ax = sns.scatterplot(
        data=df_long,
        x="area_mask",
        y="frac",
        hue="region_label",
        style="region_label",
        s=40,
        alpha=0.8,
    )

    ax.set_xscale("log")
    ax.set_xlabel("Lesion area (pixels, log scale)")
    ax.set_ylabel("Fraction of mask")
    ax.set_ylim(0, max(2.5, df_long["frac"].max() * 1.1))
    ax.set_title("Core vs Outer Fraction vs Lesion Size")
    ax.legend(title="Region", loc="best")

    scatter_path = data_dir / "morph_scatter.png"
    plt.tight_layout()
    plt.savefig(scatter_path, dpi=200)
    plt.close()
    print(f"[INFO] Saved scatter plot → {scatter_path}")

    # Figure 2: box + swarm

    plt.figure(figsize=(6, 5))

    ax = sns.boxplot(
        data=df_long,
        x="region_label",
        y="frac",
        width=0.5,
        showfliers=False,
    )

    sns.stripplot(
        data=df_long,
        x="region_label",
        y="frac",
        dodge=False,
        alpha=0.5,
        size=4,
        color="black",
    )

    ax.set_xlabel("")
    ax.set_ylabel("Fraction of mask")
    ax.set_title("Distribution of Core / Outer Fractions")

    box_path = data_dir / "morph_box.png"
    plt.tight_layout()
    plt.savefig(box_path, dpi=200)
    plt.close()
    print(f"[INFO] Saved box+points plot → {box_path}")


if __name__ == "__main__":
    main()
