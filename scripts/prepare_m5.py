import argparse
from pathlib import Path
import pandas as pd
import numpy as np


def prepare_m5(raw_dir: Path, out_dir: Path, n_items: int, seed: int = 42):
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sales_path = raw_dir / "sales_train_evaluation.csv"
    if not sales_path.exists():
        sales_path = raw_dir / "sales_train_validation.csv"
    if not sales_path.exists():
        raise FileNotFoundError("Missing sales_train_evaluation.csv or sales_train_validation.csv in raw_dir")

    sales = pd.read_csv(sales_path)
    d_cols = [c for c in sales.columns if c.startswith("d_")]

    rng = np.random.default_rng(seed)
    chosen = rng.choice(sales["id"].values, size=min(n_items, len(sales)), replace=False)
    sales = sales[sales["id"].isin(chosen)].reset_index(drop=True)

    long = sales[["id"] + d_cols].melt(id_vars=["id"], var_name="d", value_name="y")
    long["t"] = long["d"].str.replace("d_", "", regex=False).astype(int)
    long = long.drop(columns=["d"]).rename(columns={"id": "series_id"})
    long["y"] = long["y"].astype(float).clip(lower=0.0)

    long.to_parquet(out_dir / "m5_small.parquet", index=False)
    print(f"Saved {out_dir / 'm5_small.parquet'} with {long['series_id'].nunique()} series")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--n_items", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    prepare_m5(Path(args.raw_dir), Path(args.out_dir), args.n_items, args.seed)
