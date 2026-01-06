from pathlib import Path
import pandas as pd


def load_m5_processed(processed_dir: str) -> pd.DataFrame:
    processed_dir = Path(processed_dir)
    p = processed_dir / "m5_small.parquet"
    if not p.exists():
        raise FileNotFoundError(f"Processed M5 parquet not found: {p}")
    return pd.read_parquet(p)
