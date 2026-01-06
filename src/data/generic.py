import pandas as pd


def load_generic_csv(path: str, id_col: str, time_col: str, target_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for c in (id_col, time_col, target_col):
        if c not in df.columns:
            raise ValueError(f"Missing column {c} in {path}")
    df = df.rename(
        columns={id_col: "series_id", time_col: "t", target_col: "y"}
    )
    return df[["series_id", "t", "y"]]
