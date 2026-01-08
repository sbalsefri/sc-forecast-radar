import numpy as np
import torch

from src.utils.seed import set_seed
from src.utils.metrics import METRIC_FUNCS, mase
from src.utils.logging import append_results_csv
from src.data.generic import load_generic_csv
from src.data.windows import make_windows
from src.models.dlinear import DLinear
from src.models.patchtst_lite import PatchTSTLite
from src.train.trainer import train_torch
from src.perturb.artifacts import (
    missing_mcar,
    outlier_spike,
    block_missing,
    delay_shift,
)


def run_benchmark(
    csv_path: str,
    id_col: str,
    time_col: str,
    target_col: str,
    model_name: str = "dlinear",
    context_length: int = 24,
    horizon: int = 12,
    epochs: int = 5,
    batch_size: int = 256,
    seed: int = 42,
    log_csv: str | None = "results/results.csv",
):
    """
    Artifact-aware benchmarking for univariate forecasting models.
    """

    # --------------------------------------------------
    # 1) Setup
    # --------------------------------------------------
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --------------------------------------------------
    # 2) Load data
    # --------------------------------------------------
    df = load_generic_csv(csv_path, id_col, time_col, target_col)

    # --------------------------------------------------
    # 3) Windowing
    # --------------------------------------------------
    windows = make_windows(
        df,
        context=context_length,
        horizon=horizon,
        val_windows=0,
        test_windows=1,
    )

    X_train = windows["X_train"]
    y_train = windows["y_train"]
    X_test = windows["X_test"]
    y_test = windows["y_test"]

    # For MASE
    insample = np.concatenate(list(windows["insample"].values()))

    # --------------------------------------------------
    # 4) Model selection
    # --------------------------------------------------
    if model_name == "dlinear":
        model = DLinear(context_length, horizon)
    elif model_name == "patchtst":
        model = PatchTSTLite(context_length, horizon)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    # --------------------------------------------------
    # 5) Train
    # --------------------------------------------------
    model = train_torch(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=None,
        y_val=None,
        epochs=epochs,
        batch_size=batch_size,
        lr=1e-3,
        device="auto",
    )

    model.eval()

    # --------------------------------------------------
    # 6) Evaluation helper
    # --------------------------------------------------
    def evaluate_case(name: str, X: np.ndarray):
        with torch.no_grad():
            preds = model(
                torch.from_numpy(X).float().to(device)
            ).cpu().numpy()

        row = {
            "model": model_name,
            "perturbation": name,
            "seed": seed,
        }

        for m, fn in METRIC_FUNCS.items():
            row[m] = fn(y_test.flatten(), preds.flatten())

        row["mase"] = mase(
            y_test.flatten(), preds.flatten(), insample
        )
        return row

    # --------------------------------------------------
    # 7) Run benchmark cases
    # --------------------------------------------------
    rng = np.random.default_rng(seed)
    results = []

    # Clean
    results.append(evaluate_case("clean", X_test))

    # Missing (MCAR)
    Xm = missing_mcar(X_test, rate=0.1, rng=rng)
    results.append(evaluate_case("missing_mcar", Xm))

    # Outliers
    Xo = outlier_spike(X_test, rate=0.05, rng=rng)
    results.append(evaluate_case("outlier_spike", Xo))
    # Block missingness (sensor outage)
    Xb = block_missing(X_test, block_size=4, rng=rng)
    results.append(evaluate_case("block_missing", Xb))

    # Delay / reporting lag
    Xd = delay_shift(X_test, max_delay=3, rng=rng)
    results.append(evaluate_case("delay_shift", Xd))

    # --------------------------------------------------
    # 8) Log results
    # --------------------------------------------------
    if log_csv is not None:
        append_results_csv(results, log_csv)

    return results
