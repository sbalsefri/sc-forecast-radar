from pathlib import Path
import pandas as pd


def append_results_csv(rows, csv_path: str):
    """
    Append a list of dicts to a CSV file.
    Creates the file if it does not exist.
    """
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows)

    if csv_path.exists():
        df.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        df.to_csv(csv_path, index=False)

