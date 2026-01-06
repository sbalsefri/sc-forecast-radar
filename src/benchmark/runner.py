import numpy as np
import torch

from src.utils.seed import set_seed
from src.utils.metrics import METRIC_FUNCS, mase
from src.data.generic import load_generic_csv
from src.data.windows import make_windows
from src.models.dlinear import DLinear
from src.train.trainer import train_torch
from src.perturb.artifacts import missing_mcar, outlier_spike


def run_benchmark(
    csv_path: str,
    id_col: str,
    time_col: str,
    target_col: str,
    context_length: int = 24,
    horizon: int = 12,
    epochs: int = 5,
    batch_size: int = 256,
    seed: int = 42,
):
    set_seed(seed)

    # Load data
    df = load_generic_csv(csv_path, id_col, time_col, target_col)

    # Windowing
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

    insample = np.concatenate(list(windows["insample"].values()))

    # Train model
    model = DLinear(context_length, horizon)
    model = train_torch(
        model,
        X_train,
        y_train,
        None,
        None,
        epochs=epochs,
        batch_size=batch_size,
        lr=1e-3,
    )

    model.eval()
    results = []
    rng = np.random.default_rng(seed)

    def eval_case(name, X):
        with torch.no_grad():
            preds = model(torch.from_numpy(X).float()).numpy()
        row = {"perturbation": name}
        for m, fn in METRIC_FUNCS.items():
            row[m] = fn(y_test.flatten(), preds.flatten())
        row["mase"] = mase(y_test.flatten(), preds.flatten(), insample)
        return row

    # Clean
    results.append(eval_case("clean", X_test))

    # Missing
    Xm = missing_mcar(X_test, 0.1, rng)
    results.append(eval_case("missing_mcar", Xm))

    # Outliers
    Xo = outlier_spike(X_test, 0.05, rng)
    results.append(eval_case("outlier_spike", Xo))

    return results
