from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"
    csv_path = data_dir / "denoising_metrics.csv"

    if not csv_path.exists():
        print(f"[ERROR] denoising_metrics.csv not found at: {csv_path}")
        return

    print(f"[INFO] Loading denoising metrics from: {csv_path}")
    df = pd.read_csv(csv_path)

    required_cols = {"denoiser", "transform", "cnr"}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"[ERROR] Missing expected columns: {missing}")
        return

    denoiser_order = ["none", "median3", "log_gaussian", "bilateral"]
    denoiser_order = [d for d in denoiser_order if d in df["denoiser"].unique()]

    default_transform_order = ["baseline", "clahe", "gamma0.6_clahe", "log_clahe"]
    transform_order = [
        t for t in default_transform_order if t in df["transform"].unique()
    ]
    if not transform_order:
        transform_order = sorted(df["transform"].unique())

    df["denoiser"] = pd.Categorical(
        df["denoiser"], categories=denoiser_order, ordered=True
    )
    df["transform"] = pd.Categorical(
        df["transform"], categories=transform_order, ordered=True
    )

    sns.set(style="whitegrid")

    plt.figure(figsize=(7, 5))

    ax = sns.boxplot(
        data=df,
        x="denoiser",
        y="cnr",
        hue="transform",
        order=denoiser_order,
    )

    sns.stripplot(
        data=df,
        x="denoiser",
        y="cnr",
        hue="transform",
        order=denoiser_order,
        dodge=True,
        alpha=0.4,
        size=3,
        color="black",
        legend=False,
    )

    ax.set_xlabel("Denoiser")
    ax.set_ylabel("CNR (lesion vs background)")
    ax.set_title("Effect of denoiser on CNR for each intensity transform")

    plt.tight_layout()
    out_path = data_dir / "denoising_cnr_summary.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[INFO] Saved CNR vs denoiser plot → {out_path}")


if __name__ == "__main__":
    main()
