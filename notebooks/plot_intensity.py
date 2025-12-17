from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATA_DIR = Path("data")
CSV_PATH = DATA_DIR / "intensity_transform_metrics.csv"


def main():
    df = pd.read_csv(CSV_PATH)

    if "contrast" not in df.columns:
        raise ValueError("CSV must contain a 'contrast' column.")
    if "cnr" not in df.columns:
        raise ValueError("CSV must contain a 'cnr' column.")
    if "transform" not in df.columns:
        raise ValueError("CSV must contain a 'transform' column.")

    transform_order = ["baseline", "clahe", "gamma0.6_clahe", "log_clahe"]
    transform_order = [t for t in transform_order if t in df["transform"].unique()]

    # dot colors per transform
    point_palette = {
        "baseline": "#1b9e77",
        "clahe": "#d95f02",
        "gamma0.6_clahe": "#7570b3",
        "log_clahe": "#e7298a",
    }

    sns.set(style="whitegrid", context="talk")

    def raincloud(x_col: str, x_label: str, title: str, out_name: str):
        plt.figure(figsize=(10, 6))

        sns.violinplot(
            data=df,
            y="transform",
            x=x_col,
            order=transform_order,
            color="#8dd3c7",
            cut=0,
            inner=None,
        )

        sns.boxplot(
            data=df,
            y="transform",
            x=x_col,
            order=transform_order,
            width=0.15,
            whis=(5, 95),
            showcaps=True,
            boxprops=dict(facecolor="white", edgecolor="black"),
            whiskerprops=dict(color="black"),
            medianprops=dict(color="black"),
            showfliers=False,
        )

        sns.stripplot(
            data=df,
            y="transform",
            x=x_col,
            order=transform_order,
            hue="transform",
            dodge=False,
            size=3,
            palette=point_palette,
            alpha=0.9,
        )

        plt.xlabel(x_label)
        plt.ylabel("Intensity transform")
        plt.title(title)

        plt.tight_layout()
        plt.savefig(DATA_DIR / out_name, dpi=300)
        plt.close()

    raincloud(
        x_col="cnr",
        x_label="CNR",
        title="Lesion vs background CNR by intensity transform",
        out_name="intensity_cnr_summary.png",
    )

    raincloud(
        x_col="contrast",
        x_label="Mean contrast (|μ_roi - μ_bg|)",
        title="Lesion vs background contrast by intensity transform",
        out_name="intensity_contrast_summary.png",
    )

    print("Saved plots to:", DATA_DIR / "intensity_cnr_summary.png")
    print("                ", DATA_DIR / "intensity_contrast_summary.png")


if __name__ == "__main__":
    main()
