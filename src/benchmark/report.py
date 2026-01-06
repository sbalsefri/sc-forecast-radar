import pandas as pd


def summarize(results_csv: str):
    df = pd.read_csv(results_csv)
    print("CLEAN:")
    print(df[df["perturbation"] == "clean"])
    print("\nWORST by RMSE:")
    print(df[df["perturbation"] != "clean"].sort_values("rmse", ascending=False).head(10))
