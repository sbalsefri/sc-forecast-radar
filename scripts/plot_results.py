import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_CSV = "results/results.csv"
OUT_DIR = Path("results/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

METRICS = ["rmse", "mae", "smape", "mase"]


def plot_metric(df, metric):
    plt.figure(figsize=(6, 4))

    for model in sorted(df["model"].unique()):
        sub = df[df["model"] == model]
        means = (
            sub.groupby("perturbation")[metric]
            .mean()
            .reindex(["clean", "missing_mcar", "outlier_spike"])
        )
        plt.plot(means.index, means.values, marker="o", label=model)

    plt.title(metric.upper())
    plt.xlabel("Perturbation")
    plt.ylabel(metric.upper())
    plt.legend()
    plt.tight_layout()

    out = OUT_DIR / f"{metric}_robustness.png"
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"Saved {out}")


def main():
    df = pd.read_csv(RESULTS_CSV)

    for m in METRICS:
        plot_metric(df, m)


if __name__ == "__main__":
    main()

